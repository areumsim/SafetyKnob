#!/usr/bin/env python3
"""
Label Shift Correction for Temporal Distribution Shift

Tests whether the temporal performance drop is driven by label shift
(category distribution change) by applying importance weighting.

Methods:
  1. Category-aware reweighting: w(x) = P_test(cat(x)) / P_train(cat(x))
  2. Class prior adjustment: shift threshold based on estimated test class prior
  3. Combined: reweighted training + threshold adjustment

Usage:
    python scripts/label_shift_correction.py \
        --model siglip \
        --embeddings-dir embeddings/temporal/siglip \
        --output results/label_shift/siglip \
        --seeds 42,123,456,789,2024
"""

import argparse
import json
import re
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.train_from_embeddings import BinarySafetyClassifier, load_split, set_seed


def extract_category(filename):
    """Extract accident category (A-F) from filename."""
    match = re.search(r'_([A-F])\d{2}_', filename)
    return match.group(1) if match else None


def compute_category_weights(train_files, test_files):
    """Compute importance weights based on category distribution shift."""
    train_cats = Counter(extract_category(f) for f in train_files)
    test_cats = Counter(extract_category(f) for f in test_files)

    # Remove None
    train_cats.pop(None, None)
    test_cats.pop(None, None)

    train_total = sum(train_cats.values())
    test_total = sum(test_cats.values())

    weights = {}
    for cat in set(train_cats.keys()) | set(test_cats.keys()):
        p_train = train_cats.get(cat, 1) / train_total
        p_test = test_cats.get(cat, 1) / test_total
        weights[cat] = p_test / p_train

    return weights, train_cats, test_cats


def train_with_reweighting(embeddings_dir, seed, epochs=50, lr=1e-3, method='reweight'):
    """Train binary classifier with category-aware sample reweighting."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_emb, train_labels, train_files = load_split(embeddings_dir, 'train')
    val_emb, val_labels, val_files = load_split(embeddings_dir, 'val')
    test_emb, test_labels, test_files = load_split(embeddings_dir, 'test')

    embedding_dim = train_emb.shape[1]

    # Compute category weights
    cat_weights, train_dist, test_dist = compute_category_weights(train_files, test_files)

    # Assign per-sample weights
    sample_weights = torch.ones(len(train_files))
    for i, fname in enumerate(train_files):
        cat = extract_category(fname)
        if cat and cat in cat_weights:
            sample_weights[i] = cat_weights[cat]

    model = BinarySafetyClassifier(embedding_dim, hidden_dim=512, probe_depth='2layer').to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if method == 'reweight':
        # Weighted BCE loss
        criterion = nn.BCELoss(reduction='none')
    else:
        criterion = nn.BCELoss()

    # Use weighted random sampler for balanced sampling
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_dataset = TensorDataset(train_emb, train_labels,
                                  torch.arange(len(train_files)))
    train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)

    best_val_f1 = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for emb_batch, label_batch, idx_batch in train_loader:
            emb_batch = emb_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            out = model(emb_batch)

            if method == 'reweight':
                loss_per_sample = criterion(out, label_batch)
                weights_batch = sample_weights[idx_batch].to(device)
                loss = (loss_per_sample * weights_batch).mean()
            else:
                loss = criterion(out, label_batch)

            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_emb.to(device)).cpu()
            val_preds = (val_out > 0.5).numpy().astype(int)
            val_f1 = f1_score(val_labels.numpy() > 0.5, val_preds, zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Evaluate on test
    with torch.no_grad():
        test_out = model(test_emb.to(device)).cpu().numpy()

    test_targets = (test_labels.numpy() > 0.5).astype(int)

    # Standard threshold
    test_preds_05 = (test_out > 0.5).astype(int)
    f1_05 = f1_score(test_targets, test_preds_05, zero_division=0)
    acc_05 = accuracy_score(test_targets, test_preds_05)

    # Optimized threshold (search on val)
    with torch.no_grad():
        val_scores = model(val_emb.to(device)).cpu().numpy()
    val_targets = (val_labels.numpy() > 0.5).astype(int)

    best_thresh = 0.5
    best_thresh_f1 = 0
    for thresh in np.arange(0.3, 0.7, 0.01):
        val_p = (val_scores > thresh).astype(int)
        f1_t = f1_score(val_targets, val_p, zero_division=0)
        if f1_t > best_thresh_f1:
            best_thresh_f1 = f1_t
            best_thresh = thresh

    test_preds_opt = (test_out > best_thresh).astype(int)
    f1_opt = f1_score(test_targets, test_preds_opt, zero_division=0)
    acc_opt = accuracy_score(test_targets, test_preds_opt)

    auc = roc_auc_score(test_targets, test_out) if len(set(test_targets.tolist())) > 1 else 0.0

    return {
        'f1_threshold_05': float(f1_05),
        'accuracy_threshold_05': float(acc_05),
        'f1_optimized_threshold': float(f1_opt),
        'accuracy_optimized_threshold': float(acc_opt),
        'optimal_threshold': float(best_thresh),
        'auc_roc': float(auc),
        'category_weights': {k: float(v) for k, v in cat_weights.items()},
        'train_distribution': dict(train_dist),
        'test_distribution': dict(test_dist),
    }


def train_baseline(embeddings_dir, seed, epochs=50, lr=1e-3):
    """Standard baseline (no reweighting) for comparison."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_emb, train_labels, _ = load_split(embeddings_dir, 'train')
    val_emb, val_labels, _ = load_split(embeddings_dir, 'val')
    test_emb, test_labels, test_files = load_split(embeddings_dir, 'test')

    embedding_dim = train_emb.shape[1]
    model = BinarySafetyClassifier(embedding_dim, hidden_dim=512, probe_depth='2layer').to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_dataset = TensorDataset(train_emb, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    best_val_f1 = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for emb_batch, label_batch in train_loader:
            emb_batch, label_batch = emb_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(emb_batch), label_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(val_emb.to(device)).cpu()
            val_preds = (val_out > 0.5).numpy().astype(int)
            val_f1 = f1_score(val_labels.numpy() > 0.5, val_preds, zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_out = model(test_emb.to(device)).cpu().numpy()
    test_targets = (test_labels.numpy() > 0.5).astype(int)

    # Optimized threshold
    with torch.no_grad():
        val_scores = model(val_emb.to(device)).cpu().numpy()
    val_targets = (val_labels.numpy() > 0.5).astype(int)

    best_thresh = 0.5
    best_thresh_f1 = 0
    for thresh in np.arange(0.3, 0.7, 0.01):
        val_p = (val_scores > thresh).astype(int)
        f1_t = f1_score(val_targets, val_p, zero_division=0)
        if f1_t > best_thresh_f1:
            best_thresh_f1 = f1_t
            best_thresh = thresh

    test_preds_05 = (test_out > 0.5).astype(int)
    test_preds_opt = (test_out > best_thresh).astype(int)
    auc = roc_auc_score(test_targets, test_out) if len(set(test_targets.tolist())) > 1 else 0.0

    return {
        'f1_threshold_05': float(f1_score(test_targets, test_preds_05, zero_division=0)),
        'f1_optimized_threshold': float(f1_score(test_targets, test_preds_opt, zero_division=0)),
        'optimal_threshold': float(best_thresh),
        'auc_roc': float(auc),
    }


def main():
    parser = argparse.ArgumentParser(description='Label shift correction')
    parser.add_argument('--model', type=str, default='siglip')
    parser.add_argument('--embeddings-dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--seeds', type=str, default='42,123,456,789,2024')
    parser.add_argument('--epochs', type=int, default=50)

    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = Path(args.embeddings_dir)

    print(f"{'='*60}")
    print(f"LABEL SHIFT CORRECTION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Seeds: {seeds}")

    methods = {
        'baseline': 'No correction (standard training)',
        'reweight': 'Category-aware sample reweighting',
    }

    all_results = {}

    for method_name, method_desc in methods.items():
        print(f"\n--- Method: {method_desc} ---")

        method_results = []
        for seed in seeds:
            print(f"  Seed {seed}...", end=' ', flush=True)

            if method_name == 'baseline':
                r = train_baseline(embeddings_dir, seed, args.epochs)
            else:
                r = train_with_reweighting(embeddings_dir, seed, args.epochs, method=method_name)

            r['seed'] = seed
            method_results.append(r)
            print(f"F1={r['f1_threshold_05']*100:.2f}% (opt={r['f1_optimized_threshold']*100:.2f}%)")

        f1s = [r['f1_threshold_05'] * 100 for r in method_results]
        f1s_opt = [r['f1_optimized_threshold'] * 100 for r in method_results]

        all_results[method_name] = {
            'description': method_desc,
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)),
            'f1_opt_mean': float(np.mean(f1s_opt)),
            'f1_opt_std': float(np.std(f1s_opt)),
            'per_seed': method_results,
        }

    # Summary
    print(f"\n{'='*60}")
    print(f"LABEL SHIFT CORRECTION SUMMARY ({args.model.upper()})")
    print(f"{'='*60}")
    print(f"{'Method':<40} {'F1 (τ=0.5)':<22} {'F1 (optimal τ)':<22}")
    print("-" * 84)
    for method_name, r in all_results.items():
        print(f"{methods[method_name]:<40} "
              f"{r['f1_mean']:>6.2f}±{r['f1_std']:.2f}%       "
              f"{r['f1_opt_mean']:>6.2f}±{r['f1_opt_std']:.2f}%")

    # Save
    with open(output_dir / 'label_shift_results.json', 'w') as f:
        json.dump({
            'model': args.model,
            'seeds': seeds,
            'methods': all_results,
        }, f, indent=2)
    print(f"\nResults saved to {output_dir}/label_shift_results.json")


if __name__ == '__main__':
    main()
