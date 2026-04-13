#!/usr/bin/env python3
"""
Lightweight Probe Training from Pre-extracted Embeddings

Trains probe heads (linear / 1-layer / 2-layer) on cached embeddings.
Epoch time ~10s vs ~30min with on-the-fly extraction.

Usage:
    python3 experiments/train_from_embeddings.py \
        --model siglip \
        --embeddings-dir embeddings/scenario/siglip \
        --probe-depth linear \
        --output results/scenario/siglip_linear \
        --epochs 50 --lr 1e-3
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


# ========== Classifier (same as train_binary.py) ==========
class BinarySafetyClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=512, probe_depth='2layer'):
        super().__init__()
        self.probe_depth = probe_depth

        if probe_depth == 'linear':
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, 1),
                nn.Sigmoid()
            )
        elif probe_depth == '1layer':
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        elif probe_depth == '2layer':
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        elif probe_depth == '3layer':
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown probe_depth: {probe_depth}")

    def forward(self, x):
        return self.classifier(x).squeeze()


def load_split(embeddings_dir, split, category=None):
    """Load pre-extracted embeddings and labels for a split.

    Args:
        embeddings_dir: Path to embeddings directory
        split: 'train', 'val', or 'test'
        category: Optional category code ('A'-'E') to filter by
    """
    emb_data = torch.load(embeddings_dir / f'{split}_embeddings.pt', map_location='cpu')
    label_data = torch.load(embeddings_dir / f'{split}_labels.pt', map_location='cpu')

    embeddings = emb_data['embeddings']
    filenames = emb_data['filenames']
    labels_dict = label_data['labels']

    # Build label tensor
    overall_labels = []
    valid_indices = []

    for i, fname in enumerate(filenames):
        if fname not in labels_dict:
            continue

        # Category filtering
        if category:
            parts = fname.split('_')
            if len(parts) >= 2 and parts[1][0] != category:
                continue

        label = float(labels_dict[fname]['overall_safety'])
        overall_labels.append(label)
        valid_indices.append(i)

    if not valid_indices:
        raise ValueError(f"No samples found for split={split}, category={category}")

    indices = torch.tensor(valid_indices, dtype=torch.long)
    emb_filtered = embeddings[indices]
    labels_tensor = torch.tensor(overall_labels, dtype=torch.float32)

    filtered_filenames = [filenames[i] for i in valid_indices]

    return emb_filtered, labels_tensor, filtered_filenames


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train probe from pre-extracted embeddings')
    parser.add_argument('--model', type=str, required=True,
                       choices=['siglip', 'clip', 'dinov2', 'resnet50'],
                       help='Foundation model (for metadata)')
    parser.add_argument('--embeddings-dir', type=str, required=True,
                       help='Directory with pre-extracted embeddings')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Original data directory (unused, for compatibility)')
    parser.add_argument('--probe-depth', type=str, default='2layer',
                       choices=['linear', '1layer', '2layer', '3layer'],
                       help='Probe head depth')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--category', type=str, default=None,
                       choices=['A', 'B', 'C', 'D', 'E'],
                       help='Filter by category code for independent dimension training')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-embeddings-dir', type=str, default=None,
                       help='Separate embeddings dir for test set (cross-split evaluation)')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save per-image predictions for error analysis')
    parser.add_argument('--train-fraction', type=float, default=1.0,
                       help='Fraction of training data to use (0.0-1.0, for scaling curve)')

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings_dir = Path(args.embeddings_dir)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    cat_names = {'A': 'fall_hazard', 'B': 'collision_risk', 'C': 'equipment_hazard',
                 'D': 'environmental_risk', 'E': 'protective_gear'}

    print("=" * 60)
    print(f"PROBE TRAINING FROM EMBEDDINGS: {args.model.upper()}")
    print("=" * 60)
    print(f"Probe depth: {args.probe_depth}")
    if args.category:
        print(f"Category: {args.category} ({cat_names[args.category]})")
    print(f"Embeddings: {embeddings_dir}")
    print(f"Output: {output_path}")

    # Load data
    # Try to load val split; if not available, skip validation-based early stopping
    has_val = (embeddings_dir / 'val_embeddings.pt').exists()

    train_emb, train_labels, train_files = load_split(embeddings_dir, 'train', args.category)

    # Subsample training data for scaling curve experiments
    if args.train_fraction < 1.0:
        n_total = len(train_emb)
        n_keep = max(1, int(n_total * args.train_fraction))
        # Use seed-based permutation for reproducible subsampling
        perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed))[:n_keep]
        train_emb = train_emb[perm]
        train_labels = train_labels[perm]
        train_files = [train_files[i] for i in perm.tolist()]
        print(f"  Subsampled: {n_total} → {n_keep} ({args.train_fraction:.0%})")

    if has_val:
        val_emb, val_labels, val_files = load_split(embeddings_dir, 'val', args.category)

    # Support cross-split evaluation (e.g., train on scenario, test on temporal)
    if args.test_embeddings_dir:
        test_emb_dir = Path(args.test_embeddings_dir)
        test_emb, test_labels, test_files = load_split(test_emb_dir, 'test', args.category)
        print(f"Cross-split: test from {test_emb_dir}")
    else:
        test_emb, test_labels, test_files = load_split(embeddings_dir, 'test', args.category)

    embedding_dim = train_emb.shape[1]

    print(f"\nEmbedding dim: {embedding_dim}")
    print(f"Train: {len(train_emb)} samples (safe: {int(train_labels.sum())}, unsafe: {len(train_labels) - int(train_labels.sum())})")
    if has_val:
        print(f"Val:   {len(val_emb)} samples")
    else:
        print("Val:   (not available, using train loss for early stopping)")
    print(f"Test:  {len(test_emb)} samples")

    # Create dataloaders
    train_dataset = TensorDataset(train_emb, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    if has_val:
        val_dataset = TensorDataset(val_emb, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = TensorDataset(test_emb, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    model = BinarySafetyClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=512,
        probe_depth=args.probe_depth
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters")

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    best_val_f1 = 0
    patience = 10
    patience_counter = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for emb_batch, label_batch in train_loader:
            emb_batch = emb_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            outputs = model(emb_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation
        if has_val:
            model.eval()
            val_preds, val_targets, val_scores = [], [], []
            with torch.no_grad():
                for emb_batch, label_batch in val_loader:
                    emb_batch = emb_batch.to(device)
                    out = model(emb_batch)
                    val_scores.extend(out.cpu().numpy())
                    val_preds.extend((out > 0.5).cpu().numpy().astype(int))
                    val_targets.extend(label_batch.numpy())

            val_f1 = f1_score(val_targets, val_preds, zero_division=0)
            val_acc = accuracy_score(val_targets, val_preds)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_f1': best_val_f1,
                    'probe_depth': args.probe_depth,
                    'embedding_dim': embedding_dim,
                }, output_path / 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            # No val set: save every improvement on train loss
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}")
            # Save last model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'probe_depth': args.probe_depth,
                'embedding_dim': embedding_dim,
            }, output_path / 'best_model.pt')

    training_time = time.time() - start_time

    # Test evaluation
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)

    checkpoint = torch.load(output_path / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_preds, test_targets, test_scores = [], [], []
    with torch.no_grad():
        for emb_batch, label_batch in test_loader:
            emb_batch = emb_batch.to(device)
            out = model(emb_batch)
            test_scores.extend(out.cpu().numpy())
            test_preds.extend((out > 0.5).cpu().numpy().astype(int))
            test_targets.extend(label_batch.numpy())

    test_metrics = {
        'accuracy': float(accuracy_score(test_targets, test_preds)),
        'f1': float(f1_score(test_targets, test_preds, zero_division=0)),
        'precision': float(precision_score(test_targets, test_preds, zero_division=0)),
        'recall': float(recall_score(test_targets, test_preds, zero_division=0)),
        'auc_roc': float(roc_auc_score(test_targets, test_scores)),
    }

    print(f"Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"F1 Score:  {test_metrics['f1']*100:.2f}%")
    print(f"Precision: {test_metrics['precision']*100:.2f}%")
    print(f"Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"AUC-ROC:   {test_metrics['auc_roc']*100:.2f}%")

    # Save results
    results = {
        'model': args.model,
        'dataset': 'danger_al',
        'task': 'binary_classification',
        'category': args.category,
        'category_name': cat_names.get(args.category) if args.category else None,
        'embedding_dim': embedding_dim,
        'probe_depth': args.probe_depth,
        'total_parameters': total_params,
        'training_time_seconds': training_time,
        'epochs_trained': epoch + 1,
        'best_val_f1': float(best_val_f1) if has_val else None,
        'test_metrics': test_metrics,
        'hyperparameters': {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'hidden_dim': 512,
            'probe_depth': args.probe_depth,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
        }
    }

    # Save per-image predictions if requested
    if args.save_predictions:
        predictions = {}
        for i, fname in enumerate(test_files):
            predictions[fname] = {
                'true_label': int(test_targets[i]),
                'pred_label': int(test_preds[i]),
                'pred_score': float(test_scores[i]),
                'correct': int(test_targets[i]) == int(test_preds[i]),
            }
        results['per_image_predictions'] = predictions

    results_file = output_path / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")
    print(f"Training time: {training_time:.1f}s ({training_time/60:.1f} min)")


if __name__ == '__main__':
    main()
