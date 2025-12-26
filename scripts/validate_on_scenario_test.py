#!/usr/bin/env python3
"""
Validate trained models on scenario-based test set

This script evaluates models trained on random split (danger_al)
on the scenario-based test set to assess generalization to unseen scenarios.
"""

import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
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
    """Binary-only safety classifier (no dimension heads)"""

    def __init__(self, embedding_dim, hidden_dim=512):
        super().__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Binary safety head
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


# ========== Dataset ==========
class ScenarioTestDataset(Dataset):
    """Dataset for scenario-based test set"""

    def __init__(self, image_dir, labels_dict):
        self.image_dir = Path(image_dir)
        self.labels_dict = labels_dict
        self.image_files = list(labels_dict.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = self.image_dir / filename
        label = float(self.labels_dict[filename]['overall_safety'])
        scenario = self.labels_dict[filename].get('scenario', 'unknown')

        return str(img_path), label, scenario


# ========== Evaluation Functions ==========
def extract_embeddings_batch(embedder, image_paths, device):
    """Extract embeddings for a batch of images"""
    embeddings = []
    for img_path in image_paths:
        try:
            emb = embedder.extract_embedding(img_path)
            embeddings.append(torch.tensor(emb, dtype=torch.float32))
        except Exception as e:
            print(f"Warning: Failed to process {img_path}: {e}")
            embeddings.append(torch.zeros(embedder.embedding_dim, dtype=torch.float32))

    return torch.stack(embeddings).to(device)


def evaluate_model(model, dataloader, embedder, device):
    """Evaluate model on dataset"""
    model.eval()
    all_preds, all_targets, all_scores, all_scenarios = [], [], [], []

    with torch.no_grad():
        for image_paths, labels, scenarios in dataloader:
            embeddings = extract_embeddings_batch(embedder, image_paths, device)
            outputs = model(embeddings)

            all_scores.extend(outputs.cpu().numpy())
            all_preds.extend((outputs > 0.5).cpu().numpy().astype(int))
            all_targets.extend(labels.numpy())
            all_scenarios.extend(scenarios)

    metrics = {
        'accuracy': float(accuracy_score(all_targets, all_preds)),
        'f1': float(f1_score(all_targets, all_preds, zero_division=0)),
        'precision': float(precision_score(all_targets, all_preds, zero_division=0)),
        'recall': float(recall_score(all_targets, all_preds, zero_division=0)),
        'auc_roc': float(roc_auc_score(all_targets, all_scores))
    }

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    metrics['confusion_matrix'] = cm.tolist()

    # Per-scenario breakdown
    scenario_metrics = {}
    unique_scenarios = set(all_scenarios)
    for scenario in unique_scenarios:
        scenario_mask = [s == scenario for s in all_scenarios]
        scenario_preds = [p for p, m in zip(all_preds, scenario_mask) if m]
        scenario_targets = [t for t, m in zip(all_targets, scenario_mask) if m]

        if len(scenario_targets) > 0:
            scenario_metrics[scenario] = {
                'count': len(scenario_targets),
                'accuracy': float(accuracy_score(scenario_targets, scenario_preds)),
                'f1': float(f1_score(scenario_targets, scenario_preds, zero_division=0))
            }

    metrics['per_scenario'] = scenario_metrics

    return metrics, all_preds, all_targets, all_scores


def load_model(model_name, checkpoint_path, device):
    """Load trained model"""
    print(f"\nLoading {model_name} model from {checkpoint_path}...")

    # Create embedder
    if model_name == 'siglip':
        embedder = SigLIPEmbedder(device=device)
    elif model_name == 'clip':
        embedder = CLIPEmbedder(device=device)
    elif model_name == 'dinov2':
        embedder = DINOv2Embedder(device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    embedding_dim = embedder.embedding_dim

    # Create classifier
    classifier = BinarySafetyClassifier(embedding_dim=embedding_dim, hidden_dim=512)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.to(device)
    classifier.eval()

    print(f"✓ Loaded {model_name} (embedding_dim={embedding_dim})")
    print(f"  Val F1 (during training): {checkpoint.get('val_f1', 'N/A'):.4f}")

    return embedder, classifier


def main():
    parser = argparse.ArgumentParser(description='Validate on scenario-based test set')
    parser.add_argument('--model', type=str, required=True,
                       choices=['siglip', 'clip', 'dinov2', 'all'],
                       help='Model to evaluate (or "all")')
    parser.add_argument('--data-dir', type=str, default='/workspace/arsim/EmoKnob/data_scenario',
                       help='Scenario-based data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='results/danger_al',
                       help='Directory containing trained model checkpoints')
    parser.add_argument('--output', type=str, default='results/scenario_validation',
                       help='Output directory for validation results')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SCENARIO-BASED TEST SET VALIDATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Data: {args.data_dir}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print(f"Output: {output_path}")

    # Load scenario test labels
    data_path = Path(args.data_dir)
    labels_file = data_path / 'labels.json'

    print(f"\nLoading labels from {labels_file}...")
    with open(labels_file) as f:
        all_labels = json.load(f)

    test_labels = {k.replace('test/', ''): v for k, v in all_labels.items() if k.startswith('test/')}
    print(f"✓ Loaded {len(test_labels)} test images")

    # Create test dataset
    test_dataset = ScenarioTestDataset(data_path / 'test', test_labels)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Determine which models to evaluate
    if args.model == 'all':
        models_to_eval = ['siglip', 'clip', 'dinov2']
    else:
        models_to_eval = [args.model]

    # Evaluate each model
    all_results = {}

    for model_name in models_to_eval:
        print("\n" + "=" * 70)
        print(f"EVALUATING {model_name.upper()}")
        print("=" * 70)

        checkpoint_path = Path(args.checkpoint_dir) / model_name / 'best_model.pt'

        if not checkpoint_path.exists():
            print(f"⚠ Checkpoint not found: {checkpoint_path}")
            print(f"Skipping {model_name}...")
            continue

        # Load model
        embedder, classifier = load_model(model_name, checkpoint_path, device)

        # Evaluate
        print(f"\nEvaluating on scenario test set...")
        start_time = time.time()
        metrics, preds, targets, scores = evaluate_model(classifier, test_loader, embedder, device)
        eval_time = time.time() - start_time

        # Print results
        print(f"\n" + "-" * 70)
        print("OVERALL METRICS")
        print("-" * 70)
        print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"  F1 Score:  {metrics['f1']*100:.2f}%")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall:    {metrics['recall']*100:.2f}%")
        print(f"  AUC-ROC:   {metrics['auc_roc']*100:.2f}%")

        print(f"\n" + "-" * 70)
        print("PER-SCENARIO BREAKDOWN")
        print("-" * 70)
        for scenario, scenario_metrics in sorted(metrics['per_scenario'].items()):
            print(f"  {scenario}:")
            print(f"    Count:    {scenario_metrics['count']}")
            print(f"    Accuracy: {scenario_metrics['accuracy']*100:.2f}%")
            print(f"    F1 Score: {scenario_metrics['f1']*100:.2f}%")

        # Save results
        results = {
            'model': model_name,
            'dataset': 'scenario_test',
            'test_scenarios': ['SO-46', 'SO-47'],
            'test_size': len(test_labels),
            'evaluation_time_seconds': eval_time,
            'test_metrics': {
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'auc_roc': metrics['auc_roc']
            },
            'confusion_matrix': metrics['confusion_matrix'],
            'per_scenario_metrics': metrics['per_scenario']
        }

        results_file = output_path / f'{model_name}_scenario_test.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to {results_file}")

        all_results[model_name] = results

    # Print comparison if multiple models evaluated
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("MODEL COMPARISON ON SCENARIO TEST SET")
        print("=" * 70)
        print(f"\n{'Model':<15} {'Accuracy':<12} {'F1 Score':<12} {'AUC-ROC':<12}")
        print("-" * 70)
        for model_name, results in all_results.items():
            acc = results['test_metrics']['accuracy'] * 100
            f1 = results['test_metrics']['f1'] * 100
            auc = results['test_metrics']['auc_roc'] * 100
            print(f"{model_name:<15} {acc:>10.2f}%  {f1:>10.2f}%  {auc:>10.2f}%")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
