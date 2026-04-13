"""
Error Analysis Script

Analyzes misclassifications to identify patterns and provide insights
for model improvement and ensemble justification.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel


class BinarySafetyClassifier(nn.Module):
    """Binary safety classifier architecture"""

    def __init__(self, embedding_dim, hidden_dim=512):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.safety_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        overall_safety = self.safety_head(features)
        return overall_safety


class EmbedderWrapper:
    """Wrapper for different vision models"""

    def __init__(self, model_name: str, device='cuda'):
        self.device = device
        self.model_name = model_name

        if model_name == 'siglip':
            self.model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
            self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
            self.embedding_dim = 1152
        elif model_name == 'clip':
            self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.embedding_dim = 768
        elif model_name == 'dinov2':
            self.model = AutoModel.from_pretrained("facebook/dinov2-large").to(device)
            self.processor = AutoProcessor.from_pretrained("facebook/dinov2-large")
            self.embedding_dim = 1024
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model.eval()

    @torch.no_grad()
    def embed_image(self, image_path: str):
        """Extract embedding from image"""
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        if self.model_name == 'siglip':
            outputs = self.model.get_image_features(**inputs)
        elif self.model_name == 'clip':
            outputs = self.model.get_image_features(**inputs)
        elif self.model_name == 'dinov2':
            outputs = self.model(**inputs).last_hidden_state.mean(dim=1)

        return outputs


def load_model(model_name: str, checkpoint_path: Path, device):
    """Load trained model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        embedding_dim = checkpoint.get('embedding_dim', None)
    else:
        state_dict = checkpoint
        embedding_dim = None

    if embedding_dim is None and 'feature_extractor.0.weight' in state_dict:
        embedding_dim = state_dict['feature_extractor.0.weight'].shape[1]

    classifier = BinarySafetyClassifier(embedding_dim=embedding_dim, hidden_dim=512).to(device)
    classifier.load_state_dict(state_dict)
    classifier.eval()

    embedder = EmbedderWrapper(model_name, device)

    return embedder, classifier


def analyze_model_errors(model_name: str, checkpoint_path: Path, data_dir: Path,
                         labels_file: Path, device='cuda'):
    """Analyze errors for a single model"""
    print(f"\nAnalyzing {model_name.upper()}...")

    # Load model
    embedder, classifier = load_model(model_name, checkpoint_path, device)

    # Load test data
    with open(labels_file, 'r') as f:
        all_labels = json.load(f)

    test_labels = {k.replace('test/', ''): v for k, v in all_labels.items() if k.startswith('test/')}
    image_paths = [data_dir / 'test' / img_name for img_name in test_labels.keys()]
    true_labels = [test_labels[img_name]['overall_safety'] for img_name in test_labels.keys()]
    classes = [test_labels[img_name]['class'] for img_name in test_labels.keys()]

    # Predict
    predictions = []
    confidences = []

    print("  Making predictions...")
    for img_path in tqdm(image_paths, desc=f"  {model_name}"):
        try:
            embedding = embedder.embed_image(str(img_path)).to(device)
            with torch.no_grad():
                output = classifier(embedding)
                confidence = output.item()
                pred = 1 if confidence > 0.5 else 0

            predictions.append(pred)
            confidences.append(confidence)
        except Exception as e:
            print(f"  Error with {img_path}: {e}")
            predictions.append(-1)
            confidences.append(0.0)

    predictions = np.array(predictions)
    confidences = np.array(confidences)
    true_labels = np.array(true_labels)

    # Identify errors
    correct_mask = predictions == true_labels
    error_mask = ~correct_mask

    false_positive_mask = (predictions == 1) & (true_labels == 0)  # Predicted safe, actually unsafe
    false_negative_mask = (predictions == 0) & (true_labels == 1)  # Predicted unsafe, actually safe

    # Collect error information
    errors = {
        'false_positives': [],
        'false_negatives': []
    }

    for i in range(len(predictions)):
        if false_positive_mask[i]:
            errors['false_positives'].append({
                'image': str(image_paths[i].name),
                'confidence': float(confidences[i]),
                'class': classes[i]
            })
        elif false_negative_mask[i]:
            errors['false_negatives'].append({
                'image': str(image_paths[i].name),
                'confidence': float(confidences[i]),
                'class': classes[i]
            })

    # Statistics
    stats = {
        'total': len(predictions),
        'correct': int(correct_mask.sum()),
        'errors': int(error_mask.sum()),
        'false_positives': int(false_positive_mask.sum()),
        'false_negatives': int(false_negative_mask.sum()),
        'accuracy': float(correct_mask.sum() / len(predictions)),
        'avg_confidence_correct': float(confidences[correct_mask].mean()),
        'avg_confidence_error': float(confidences[error_mask].mean()) if error_mask.sum() > 0 else 0.0
    }

    return {
        'model': model_name,
        'stats': stats,
        'errors': errors,
        'predictions': predictions.tolist(),
        'confidences': confidences.tolist(),
        'image_names': [str(p.name) for p in image_paths]
    }


def compute_error_correlation(all_results: dict):
    """Compute error correlation between models"""
    print("\nComputing error correlation...")

    models = list(all_results.keys())
    n_models = len(models)

    # Get predictions as arrays
    pred_arrays = {model: np.array(all_results[model]['predictions']) for model in models}

    # Compute pairwise correlation of errors
    correlations = {}

    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i < j:
                # Error agreement: both models made same error
                error1 = pred_arrays[model1] != np.array(all_results[model1]['predictions'])
                error2 = pred_arrays[model2] != np.array(all_results[model2]['predictions'])

                # Compute correlation
                correlation = np.corrcoef(error1.astype(int), error2.astype(int))[0, 1]

                correlations[f'{model1}_vs_{model2}'] = float(correlation)

    return correlations


def main():
    parser = argparse.ArgumentParser(description='Error analysis for safety classification')
    parser.add_argument('--models', type=str, default='siglip,clip,dinov2',
                       help='Comma-separated list of models')
    parser.add_argument('--data-dir', type=str, default='data_scenario',
                       help='Data directory')
    parser.add_argument('--results-dir', type=str, default='results/scenario',
                       help='Directory with model checkpoints')
    parser.add_argument('--output', type=str, default='results/analysis',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    labels_file = data_dir / 'labels.json'
    results_dir = project_root / args.results_dir
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print("ERROR ANALYSIS")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Analyze each model
    models = args.models.split(',')
    all_results = {}

    for model_name in models:
        checkpoint_path = results_dir / model_name / 'best_model.pt'

        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found for {model_name}: {checkpoint_path}")
            continue

        result = analyze_model_errors(
            model_name, checkpoint_path, data_dir, labels_file, args.device
        )
        all_results[model_name] = result

    if len(all_results) < 2:
        print("\nError: Need at least 2 models for correlation analysis")
        return

    # Compute error correlation
    correlations = compute_error_correlation(all_results)

    # Compile final report
    report = {
        'models': all_results,
        'error_correlation': correlations
    }

    # Print summary
    print(f"\n{'='*60}")
    print("ERROR ANALYSIS SUMMARY")
    print(f"{'='*60}\n")

    for model_name, result in all_results.items():
        stats = result['stats']
        print(f"{model_name.upper()}:")
        print(f"  Accuracy: {stats['accuracy']*100:.2f}%")
        print(f"  Errors: {stats['errors']} ({stats['errors']/stats['total']*100:.1f}%)")
        print(f"    False Positives: {stats['false_positives']} (predicted safe, actually unsafe)")
        print(f"    False Negatives: {stats['false_negatives']} (predicted unsafe, actually safe)")
        print(f"  Avg Confidence (Correct): {stats['avg_confidence_correct']*100:.1f}%")
        print(f"  Avg Confidence (Error): {stats['avg_confidence_error']*100:.1f}%")
        print()

    print("\nError Correlation (lower = more diverse errors = better ensemble):")
    for pair, corr in correlations.items():
        print(f"  {pair}: {corr:.3f}")

    # Save report
    output_file = output_dir / 'error_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved detailed report to {output_file}")

    # Save error images list
    for model_name, result in all_results.items():
        fp_file = output_dir / f'{model_name}_false_positives.txt'
        fn_file = output_dir / f'{model_name}_false_negatives.txt'

        with open(fp_file, 'w') as f:
            for item in result['errors']['false_positives']:
                f.write(f"{item['image']}\t{item['confidence']:.3f}\t{item['class']}\n")

        with open(fn_file, 'w') as f:
            for item in result['errors']['false_negatives']:
                f.write(f"{item['image']}\t{item['confidence']:.3f}\t{item['class']}\n")

        print(f"  {model_name}: FP={len(result['errors']['false_positives'])}, FN={len(result['errors']['false_negatives'])}")

    print(f"\n{'='*60}")
    print("ERROR ANALYSIS COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
