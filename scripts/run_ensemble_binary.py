"""
Binary Ensemble Experiment Script
Tests ensemble methods using binary-only trained models from danger_al dataset
"""

import sys
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel


# ========== Embedders ==========
class SigLIPEmbedder:
    def __init__(self, model_name="google/siglip-so400m-patch14-384", device="cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim = 1152

    @torch.no_grad()
    def embed_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        return outputs


class CLIPEmbedder:
    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim = 768

    @torch.no_grad()
    def embed_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        return outputs


class DINOv2Embedder:
    def __init__(self, model_name="facebook/dinov2-large", device="cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim = 1024

    @torch.no_grad()
    def embed_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs).last_hidden_state
        return outputs.mean(dim=1)


# Binary-only SafetyClassifier architecture
class BinarySafetyClassifier(nn.Module):
    """Binary-only safety classifier (no dimension heads)"""

    def __init__(self, embedding_dim, hidden_dim=512):
        super().__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Overall safety head
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


def load_test_dataset(data_dir: Path, labels_file: Path):
    """Load test dataset"""
    print(f"Loading test data from {data_dir}")

    # Load labels
    with open(labels_file, 'r') as f:
        all_labels = json.load(f)

    # Filter test labels
    test_labels = {k.replace('test/', ''): v for k, v in all_labels.items() if k.startswith('test/')}

    # Get image paths
    image_paths = [data_dir / 'test' / img_name for img_name in test_labels.keys()]
    labels = [test_labels[img_name]['overall_safety'] for img_name in test_labels.keys()]

    print(f"Loaded {len(image_paths)} test images")
    print(f"  Safe: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"  Unsafe: {len(labels) - sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")

    return image_paths, labels


def load_model(model_name: str, checkpoint_path: Path, device):
    """Load embedder and classifier"""
    print(f"  Loading {model_name}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model state dict if checkpoint contains metadata
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        embedding_dim = checkpoint.get('embedding_dim', None)
    else:
        state_dict = checkpoint
        embedding_dim = None

    # Detect embedding_dim from checkpoint
    if embedding_dim is None and 'feature_extractor.0.weight' in state_dict:
        embedding_dim = state_dict['feature_extractor.0.weight'].shape[1]

    print(f"    Detected embedding_dim: {embedding_dim}")

    # Create classifier and load weights
    classifier = BinarySafetyClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=512
    ).to(device)
    classifier.load_state_dict(state_dict)
    classifier.eval()

    # Create embedder based on model type
    if model_name == 'siglip':
        embedder = SigLIPEmbedder(device=device)
    elif model_name == 'clip':
        embedder = CLIPEmbedder(device=device)
    elif model_name == 'dinov2':
        embedder = DINOv2Embedder(device=device)
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    return embedder, classifier


def predict_image(image_path: Path, embedders: dict, classifiers: dict, device, strategy='weighted_vote'):
    """Predict single image using ensemble"""
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Get predictions from each model
    predictions = []

    for model_name in embedders.keys():
        try:
            # Get embedding
            embedding = embedders[model_name].embed_image(image).to(device)

            # Get prediction
            with torch.no_grad():
                output = classifiers[model_name](embedding)
                pred = output.item()
                predictions.append(pred)
        except Exception as e:
            print(f"\nError with {model_name} on {image_path.name}: {e}")
            continue

    if len(predictions) == 0:
        raise ValueError("All models failed to predict")

    # Ensemble strategy
    if strategy == 'weighted_vote' or strategy == 'average':
        # Simple average for binary
        ensemble_pred = np.mean(predictions)
    elif strategy == 'majority':
        # Majority vote
        binary_preds = [1 if p > 0.5 else 0 for p in predictions]
        ensemble_pred = 1 if sum(binary_preds) > len(binary_preds) / 2 else 0
    else:
        # Default to average
        ensemble_pred = np.mean(predictions)

    return ensemble_pred


def evaluate_ensemble(embedders: dict, classifiers: dict, image_paths: list, labels: list, device, strategy='weighted_vote'):
    """Evaluate ensemble on test set"""
    print(f"\nEvaluating ensemble with strategy: {strategy}")

    predictions = []
    true_labels = []

    start_time = time.time()

    for img_path, true_label in tqdm(zip(image_paths, labels), total=len(image_paths), desc="Processing images"):
        try:
            # Get ensemble prediction
            pred = predict_image(img_path, embedders, classifiers, device, strategy)

            # Binary prediction
            binary_pred = 1 if pred > 0.5 else 0

            predictions.append(binary_pred)
            true_labels.append(true_label)

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue

    processing_time = time.time() - start_time

    # Calculate metrics
    print("\nCalculating metrics...")

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    metrics = {
        'accuracy': float(accuracy_score(true_labels, predictions)),
        'f1': float(f1_score(true_labels, predictions)),
        'precision': float(precision_score(true_labels, predictions, zero_division=0)),
        'recall': float(recall_score(true_labels, predictions, zero_division=0)),
        'auc_roc': float(roc_auc_score(true_labels, predictions))
    }

    results = {
        'model': 'ensemble',
        'dataset': 'danger_al',
        'strategy': strategy,
        'num_models': len(embedders),
        'models_used': list(embedders.keys()),
        'test_size': len(true_labels),
        'processing_time_seconds': processing_time,
        'avg_time_per_image': processing_time / len(true_labels),
        'test_metrics': metrics
    }

    return results


def print_results(results: dict):
    """Print results in a formatted way"""
    print("\n" + "="*60)
    print("BINARY ENSEMBLE EVALUATION RESULTS")
    print("="*60)
    print(f"\nDataset: {results['dataset']}")
    print(f"Strategy: {results['strategy']}")
    print(f"Models Used: {', '.join(results['models_used'])}")
    print(f"Test Size: {results['test_size']} images")
    print(f"Processing Time: {results['processing_time_seconds']:.1f}s ({results['avg_time_per_image']*1000:.1f}ms/image)")

    print("\n" + "-"*60)
    print("PERFORMANCE METRICS")
    print("-"*60)
    metrics = results['test_metrics']
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"F1 Score:  {metrics['f1']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"AUC-ROC:   {metrics['auc_roc']*100:.2f}%")

    print("\n" + "="*60)


def main():
    print("="*60)
    print("BINARY ENSEMBLE EXPERIMENT - DANGER_AL")
    print("="*60)

    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    labels_file = data_dir / 'labels.json'
    model_dir = project_root / 'results' / 'danger_al'
    output_dir = project_root / 'results' / 'danger_al' / 'ensemble'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Models to use
    model_names = []
    for name in ['clip', 'dinov2', 'siglip']:
        checkpoint_path = model_dir / name / 'best_model.pt'
        if checkpoint_path.exists():
            model_names.append(name)
        else:
            print(f"Warning: {name} checkpoint not found at {checkpoint_path}")

    if len(model_names) < 2:
        print(f"\nError: Need at least 2 models for ensemble, found {len(model_names)}")
        return

    print(f"\nUsing models: {', '.join(model_names)}")

    # Load models
    print("\n1. Loading models...")
    embedders = {}
    classifiers = {}

    for model_name in model_names:
        checkpoint_path = model_dir / model_name / 'best_model.pt'
        try:
            embedder, classifier = load_model(model_name, checkpoint_path, device)
            embedders[model_name] = embedder
            classifiers[model_name] = classifier
        except Exception as e:
            print(f"  Error loading {model_name}: {e}")
            continue

    print(f"\nSuccessfully loaded {len(embedders)} models")

    # Load test dataset
    print("\n2. Loading test dataset...")
    image_paths, labels = load_test_dataset(data_dir, labels_file)

    # Evaluate ensemble with different strategies
    strategies = ['weighted_vote', 'majority', 'average']
    all_results = {}

    for strategy in strategies:
        print(f"\n3. Evaluating with strategy: {strategy}")
        try:
            results = evaluate_ensemble(embedders, classifiers, image_paths, labels, device, strategy)
            all_results[strategy] = results
            print_results(results)
        except Exception as e:
            print(f"Error with strategy {strategy}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    print("\n4. Saving results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for strategy, results in all_results.items():
        results_file = output_dir / f'results_{strategy}_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved {strategy} results to {results_file}")

    # Save best result as latest
    if all_results:
        best_strategy = max(all_results.keys(), key=lambda k: all_results[k]['test_metrics']['f1'])
        latest_file = output_dir / 'results.json'
        with open(latest_file, 'w') as f:
            json.dump(all_results[best_strategy], f, indent=2)
        print(f"  Saved best results ({best_strategy}) to {latest_file}")

    print("\n" + "="*60)
    print("BINARY ENSEMBLE EXPERIMENT COMPLETE!")
    print("="*60)

    return all_results


if __name__ == '__main__':
    main()
