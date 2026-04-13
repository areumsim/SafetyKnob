#!/usr/bin/env python3
"""
Hierarchical Safety Classification Pipeline

Motivated by Simpson's Paradox finding: category-specific classifiers improve
on temporal shift (+1-4%p) while the binary classifier fails (-30%p).

Architecture:
  Step 1: Category Classifier (A/B/C/D/E) - multi-class
  Step 2: Per-category Safety Classifier (safe/unsafe) - binary per category

This decouples category identification from safety assessment, making the
system robust to label shift (changing category distribution over time).

Usage:
    python3 experiments/train_hierarchical.py \
        --model siglip \
        --embeddings-dir embeddings/scenario_v2/siglip \
        --test-embeddings-dir embeddings/temporal/siglip \
        --output results/hierarchical/siglip \
        --seed 42
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, classification_report)


CATEGORIES = ['A', 'B', 'C', 'D', 'E']
CAT_NAMES = {
    'A': 'fall_hazard', 'B': 'collision_risk', 'C': 'equipment_hazard',
    'D': 'environmental_risk', 'E': 'protective_gear'
}


class CategoryClassifier(nn.Module):
    """Multi-class classifier: embedding -> category (A-E)"""
    def __init__(self, embedding_dim, num_categories=5, hidden_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_categories),
        )

    def forward(self, x):
        return self.classifier(x)


class BinarySafetyClassifier(nn.Module):
    """Binary classifier: embedding -> safe/unsafe"""
    def __init__(self, embedding_dim, hidden_dim=512):
        super().__init__()
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

    def forward(self, x):
        return self.classifier(x).squeeze()


def extract_category(filename):
    """Extract category code (A-E) from filename."""
    match = re.search(r'_([A-F]\d{2})_', filename)
    if match:
        return match.group(1)[0]
    return None


def load_split_with_categories(embeddings_dir, split):
    """Load embeddings with both safety labels and category labels."""
    emb_data = torch.load(embeddings_dir / f'{split}_embeddings.pt', map_location='cpu')
    label_data = torch.load(embeddings_dir / f'{split}_labels.pt', map_location='cpu')

    embeddings = emb_data['embeddings']
    filenames = emb_data['filenames']
    labels_dict = label_data['labels']

    safety_labels = []
    category_labels = []
    valid_indices = []

    for i, fname in enumerate(filenames):
        if fname not in labels_dict:
            continue

        cat = extract_category(fname)
        if cat is None or cat not in CATEGORIES:
            continue

        safety = float(labels_dict[fname]['overall_safety'])
        cat_idx = CATEGORIES.index(cat)

        safety_labels.append(safety)
        category_labels.append(cat_idx)
        valid_indices.append(i)

    indices = torch.tensor(valid_indices, dtype=torch.long)
    emb_filtered = embeddings[indices]
    safety_tensor = torch.tensor(safety_labels, dtype=torch.float32)
    category_tensor = torch.tensor(category_labels, dtype=torch.long)
    filtered_filenames = [filenames[i] for i in valid_indices]

    return emb_filtered, safety_tensor, category_tensor, filtered_filenames


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_category_classifier(train_emb, train_cats, val_emb, val_cats,
                               embedding_dim, device, epochs=50, lr=1e-3):
    """Train the category classifier (Step 1)."""
    model = CategoryClassifier(embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(train_emb, train_cats)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    best_val_acc = 0
    best_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for emb, cats in train_loader:
            emb, cats = emb.to(device), cats.to(device)
            optimizer.zero_grad()
            out = model(emb)
            loss = criterion(out, cats)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_out = model(val_emb.to(device))
            val_preds = val_out.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(val_cats.numpy(), val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if (epoch + 1) % 10 == 0:
            print(f"  [CatCls] Epoch {epoch+1}: val_acc={val_acc*100:.2f}%")

    model.load_state_dict(best_state)
    return model, best_val_acc


def train_per_category_safety(train_emb, train_safety, train_cats,
                               val_emb, val_safety, val_cats,
                               category_idx, embedding_dim, device,
                               epochs=50, lr=1e-3):
    """Train a safety classifier for a single category (Step 2)."""
    # Filter to this category only
    train_mask = train_cats == category_idx
    val_mask = val_cats == category_idx

    if train_mask.sum() < 5 or val_mask.sum() < 2:
        return None, 0.0

    t_emb = train_emb[train_mask]
    t_labels = train_safety[train_mask]
    v_emb = val_emb[val_mask]
    v_labels = val_safety[val_mask]

    model = BinarySafetyClassifier(embedding_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(t_emb, t_labels)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    best_val_f1 = 0
    best_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for emb, labels in train_loader:
            emb, labels = emb.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(emb)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_out = model(v_emb.to(device))
            v_preds = (v_out.cpu() > 0.5).numpy().astype(int)
            v_f1 = f1_score(v_labels.numpy(), v_preds, zero_division=0)

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_f1


def evaluate_hierarchical(cat_model, safety_models, test_emb, test_safety,
                          test_cats, test_files, device):
    """Evaluate the full hierarchical pipeline on test data."""
    cat_model.eval()
    for m in safety_models.values():
        if m is not None:
            m.eval()

    with torch.no_grad():
        # Step 1: Predict category
        cat_out = cat_model(test_emb.to(device))
        pred_cats = cat_out.argmax(dim=1).cpu()

        # Step 2: Per-category safety prediction
        final_preds = torch.zeros(len(test_emb))
        final_scores = torch.zeros(len(test_emb))

        for cat_idx in range(len(CATEGORIES)):
            mask = pred_cats == cat_idx
            if mask.sum() == 0:
                continue

            if cat_idx in safety_models and safety_models[cat_idx] is not None:
                cat_emb = test_emb[mask].to(device)
                scores = safety_models[cat_idx](cat_emb).cpu()
                final_scores[mask] = scores
                final_preds[mask] = (scores > 0.5).float()
            else:
                # Fallback: predict unsafe (conservative for safety)
                final_preds[mask] = 0.0
                final_scores[mask] = 0.3

    # Compute metrics
    targets = test_safety.numpy()
    preds = final_preds.numpy().astype(int)
    scores = final_scores.numpy()

    # Category accuracy
    cat_targets = test_cats.numpy()
    cat_preds = pred_cats.numpy()
    cat_acc = accuracy_score(cat_targets, cat_preds)

    # Safety metrics
    metrics = {
        'category_accuracy': float(cat_acc),
        'safety_accuracy': float(accuracy_score(targets, preds)),
        'safety_f1': float(f1_score(targets, preds, zero_division=0)),
        'safety_precision': float(precision_score(targets, preds, zero_division=0)),
        'safety_recall': float(recall_score(targets, preds, zero_division=0)),
    }

    try:
        metrics['safety_auc_roc'] = float(roc_auc_score(targets, scores))
    except ValueError:
        metrics['safety_auc_roc'] = None

    # Per-category safety metrics
    per_category = {}
    for cat_idx, cat_code in enumerate(CATEGORIES):
        # Using TRUE categories for fair per-category evaluation
        true_mask = cat_targets == cat_idx
        if true_mask.sum() == 0:
            continue

        cat_t = targets[true_mask]
        cat_p = preds[true_mask]
        per_category[cat_code] = {
            'name': CAT_NAMES[cat_code],
            'count': int(true_mask.sum()),
            'f1': float(f1_score(cat_t, cat_p, zero_division=0)),
            'accuracy': float(accuracy_score(cat_t, cat_p)),
        }

    metrics['per_category'] = per_category
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Hierarchical safety classification')
    parser.add_argument('--model', type=str, default='siglip')
    parser.add_argument('--embeddings-dir', type=str, required=True,
                       help='Training embeddings directory (scenario)')
    parser.add_argument('--test-embeddings-dir', type=str, default=None,
                       help='Optional separate test embeddings directory (temporal)')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=str, default=None,
                       help='Comma-separated seeds for multi-seed run')

    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(',')]
    else:
        seeds = [args.seed]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in seeds:
        set_seed(seed)
        print(f"\n{'='*60}")
        print(f"HIERARCHICAL PIPELINE (seed={seed})")
        print(f"{'='*60}")

        # Load training data with category labels
        train_emb, train_safety, train_cats, train_files = \
            load_split_with_categories(embeddings_dir, 'train')
        val_emb, val_safety, val_cats, val_files = \
            load_split_with_categories(embeddings_dir, 'val')
        test_emb, test_safety, test_cats, test_files = \
            load_split_with_categories(embeddings_dir, 'test')

        embedding_dim = train_emb.shape[1]

        # Print data stats
        print(f"\nTrain: {len(train_emb)} (categories: {dict(zip(*np.unique(train_cats.numpy(), return_counts=True)))})")
        print(f"Val:   {len(val_emb)}")
        print(f"Test:  {len(test_emb)}")

        # Step 1: Train category classifier
        print(f"\n--- Step 1: Category Classifier ---")
        cat_model, cat_val_acc = train_category_classifier(
            train_emb, train_cats, val_emb, val_cats,
            embedding_dim, device, epochs=args.epochs, lr=args.lr
        )
        print(f"  Best val category accuracy: {cat_val_acc*100:.2f}%")

        # Step 2: Train per-category safety classifiers
        print(f"\n--- Step 2: Per-Category Safety Classifiers ---")
        safety_models = {}
        for cat_idx, cat_code in enumerate(CATEGORIES):
            model, val_f1 = train_per_category_safety(
                train_emb, train_safety, train_cats,
                val_emb, val_safety, val_cats,
                cat_idx, embedding_dim, device,
                epochs=args.epochs, lr=args.lr
            )
            safety_models[cat_idx] = model
            n_train = (train_cats == cat_idx).sum().item()
            print(f"  {cat_code} ({CAT_NAMES[cat_code]}): val_f1={val_f1*100:.2f}%, n_train={n_train}")

        # Evaluate on scenario test (in-distribution)
        print(f"\n--- Evaluation: Scenario (in-distribution) ---")
        scenario_metrics = evaluate_hierarchical(
            cat_model, safety_models, test_emb, test_safety, test_cats, test_files, device
        )
        print(f"  Category Accuracy: {scenario_metrics['category_accuracy']*100:.2f}%")
        print(f"  Safety F1: {scenario_metrics['safety_f1']*100:.2f}%")
        print(f"  Safety Accuracy: {scenario_metrics['safety_accuracy']*100:.2f}%")

        # Evaluate on temporal test (out-of-distribution) if available
        temporal_metrics = None
        if args.test_embeddings_dir:
            test_emb_dir = Path(args.test_embeddings_dir)
            print(f"\n--- Evaluation: Temporal (out-of-distribution) ---")
            try:
                tgt_test_emb, tgt_test_safety, tgt_test_cats, tgt_test_files = \
                    load_split_with_categories(test_emb_dir, 'test')
                temporal_metrics = evaluate_hierarchical(
                    cat_model, safety_models, tgt_test_emb, tgt_test_safety,
                    tgt_test_cats, tgt_test_files, device
                )
                print(f"  Category Accuracy: {temporal_metrics['category_accuracy']*100:.2f}%")
                print(f"  Safety F1: {temporal_metrics['safety_f1']*100:.2f}%")
                print(f"  Safety Accuracy: {temporal_metrics['safety_accuracy']*100:.2f}%")
            except Exception as e:
                print(f"  Error loading temporal data: {e}")

        result = {
            'seed': seed,
            'model': args.model,
            'method': 'hierarchical',
            'scenario_metrics': scenario_metrics,
            'temporal_metrics': temporal_metrics,
        }
        all_results.append(result)

    # Summary
    if len(seeds) > 1:
        print(f"\n{'='*60}")
        print(f"MULTI-SEED SUMMARY ({len(seeds)} seeds)")
        print(f"{'='*60}")

        scenario_f1s = [r['scenario_metrics']['safety_f1'] * 100 for r in all_results]
        print(f"Scenario F1: {np.mean(scenario_f1s):.2f}% ± {np.std(scenario_f1s):.2f}%")

        if all_results[0]['temporal_metrics']:
            temporal_f1s = [r['temporal_metrics']['safety_f1'] * 100 for r in all_results]
            print(f"Temporal F1: {np.mean(temporal_f1s):.2f}% ± {np.std(temporal_f1s):.2f}%")

    # Save results
    results_file = output_dir / 'hierarchical_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
