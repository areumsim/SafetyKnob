#!/usr/bin/env python3
"""
Predict Caution Images using trained models

This script loads trained models (SigLIP, CLIP, DINOv2) from caution_excluded experiment
and predicts on caution_analysis images to analyze how models trained only on safe/danger
handle ambiguous boundary cases.
"""

import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
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
    def extract_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        return outputs.cpu().numpy().flatten()


class CLIPEmbedder:
    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim = 768

    @torch.no_grad()
    def extract_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        return outputs.cpu().numpy().flatten()


class DINOv2Embedder:
    def __init__(self, model_name="facebook/dinov2-large", device="cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim = 1024

    @torch.no_grad()
    def extract_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs).last_hidden_state
        return outputs.mean(dim=1).cpu().numpy().flatten()


# ========== Binary Classifier ==========
class BinarySafetyClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=512):
        super().__init__()

        # Feature extractor (must match train_binary.py)
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Binary safety head (must match train_binary.py)
        self.safety_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        overall_safety = self.safety_head(features).squeeze()
        return overall_safety


def get_embedder(model_name, device):
    if model_name == "siglip":
        return SigLIPEmbedder(device=device)
    elif model_name == "clip":
        return CLIPEmbedder(device=device)
    elif model_name == "dinov2":
        return DINOv2Embedder(device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_model(model_path, model_name, device):
    """Load trained model checkpoint"""
    embedder = get_embedder(model_name, device)
    classifier = BinarySafetyClassifier(
        embedding_dim=embedder.embedding_dim,
        hidden_dim=512
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    return embedder, classifier


@torch.no_grad()
def predict_batch(embedder, classifier, image_paths, device, batch_size=8):
    """Predict on a batch of images"""
    predictions = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]

        # Extract embeddings
        embeddings = []
        for img_path in batch_paths:
            try:
                emb = embedder.extract_embedding(img_path)
                embeddings.append(torch.tensor(emb, dtype=torch.float32))
            except Exception as e:
                print(f"Warning: Failed to process {img_path}: {e}")
                embeddings.append(torch.zeros(embedder.embedding_dim, dtype=torch.float32))

        # Stack and predict
        embeddings_tensor = torch.stack(embeddings).to(device)
        probs = classifier(embeddings_tensor)

        # Store results
        for j, prob in enumerate(probs):
            prob_val = prob.item()
            predictions.append({
                'image': batch_paths[j].name,
                'prob_safe': prob_val,
                'prob_unsafe': 1 - prob_val,
                'prediction': 'safe' if prob_val >= 0.5 else 'unsafe',
                'confidence': max(prob_val, 1 - prob_val)
            })

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--caution-dir', type=str, default='data_caution_excluded/caution_analysis',
                        help='Directory containing caution images')
    parser.add_argument('--models-dir', type=str, default='results/caution_excluded',
                        help='Directory containing trained models')
    parser.add_argument('--output', type=str, default='results/caution_excluded/caution_predictions.json',
                        help='Output JSON file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for prediction')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    caution_dir = Path(args.caution_dir)
    models_dir = Path(args.models_dir)
    output_path = Path(args.output)

    print("=" * 70)
    print("CAUTION IMAGES PREDICTION")
    print("=" * 70)
    print(f"\nCaution directory: {caution_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Device: {device}")

    # Find caution images
    image_paths = sorted(list(caution_dir.glob("*.jpg")) + list(caution_dir.glob("*.png")))
    print(f"\nFound {len(image_paths)} caution images")

    if len(image_paths) == 0:
        print("Error: No caution images found!")
        return

    # Predict with all models
    all_predictions = {}

    for model_name in ['siglip', 'clip', 'dinov2']:
        model_path = models_dir / model_name / 'best_model.pt'

        if not model_path.exists():
            print(f"\nWarning: {model_name} model not found at {model_path}")
            continue

        print(f"\n{'='*70}")
        print(f"Predicting with {model_name.upper()}")
        print('='*70)

        start_time = time.time()

        # Load model
        print(f"Loading {model_name} model...")
        embedder, classifier = load_model(model_path, model_name, device)

        # Predict
        print(f"Predicting on {len(image_paths)} images...")
        predictions = predict_batch(embedder, classifier, image_paths, device, args.batch_size)

        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.2f}s ({len(image_paths)/elapsed:.2f} images/sec)")

        # Stats
        safe_count = sum(1 for p in predictions if p['prediction'] == 'safe')
        unsafe_count = len(predictions) - safe_count
        avg_conf = np.mean([p['confidence'] for p in predictions])

        print(f"\nPrediction distribution:")
        print(f"  Safe:   {safe_count:4d} ({safe_count/len(predictions)*100:.1f}%)")
        print(f"  Unsafe: {unsafe_count:4d} ({unsafe_count/len(predictions)*100:.1f}%)")
        print(f"  Avg confidence: {avg_conf:.4f}")

        all_predictions[model_name] = predictions

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        'dataset': 'caution_analysis',
        'num_images': len(image_paths),
        'models': list(all_predictions.keys()),
        'predictions': all_predictions,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*70}")
    print("PREDICTION COMPLETE")
    print('='*70)
    print(f"Results saved to: {output_path}")

    # Model agreement analysis
    if len(all_predictions) == 3:
        print(f"\n{'='*70}")
        print("MODEL AGREEMENT ANALYSIS")
        print('='*70)

        agreement_count = {
            'all_agree_safe': 0,
            'all_agree_unsafe': 0,
            'majority_safe': 0,
            'majority_unsafe': 0,
            'complete_disagree': 0
        }

        for i in range(len(image_paths)):
            preds = [all_predictions[m][i]['prediction'] for m in ['siglip', 'clip', 'dinov2']]
            safe_count = preds.count('safe')

            if safe_count == 3:
                agreement_count['all_agree_safe'] += 1
            elif safe_count == 0:
                agreement_count['all_agree_unsafe'] += 1
            elif safe_count == 2:
                agreement_count['majority_safe'] += 1
            elif safe_count == 1:
                agreement_count['majority_unsafe'] += 1

        print(f"\nAll 3 models agree - Safe:   {agreement_count['all_agree_safe']:4d} ({agreement_count['all_agree_safe']/len(image_paths)*100:.1f}%)")
        print(f"All 3 models agree - Unsafe: {agreement_count['all_agree_unsafe']:4d} ({agreement_count['all_agree_unsafe']/len(image_paths)*100:.1f}%)")
        print(f"2 models agree - Safe:       {agreement_count['majority_safe']:4d} ({agreement_count['majority_safe']/len(image_paths)*100:.1f}%)")
        print(f"2 models agree - Unsafe:     {agreement_count['majority_unsafe']:4d} ({agreement_count['majority_unsafe']/len(image_paths)*100:.1f}%)")


if __name__ == '__main__':
    main()
