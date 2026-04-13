#!/usr/bin/env python3
"""
Data Scaling Curve Experiment

Tests SigLIP 2-layer probe at different data fractions (10/25/50/75/100%)
with multiple seeds to produce a learning curve.

Usage:
    python scripts/run_scaling_curve.py --output results/scaling_curve
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
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.train_from_embeddings import BinarySafetyClassifier, load_split


FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
SEEDS = [42, 123, 456, 789, 2024]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_with_fraction(embeddings_dir, fraction, seed, epochs=50, batch_size=256, lr=1e-3):
    """Train a 2-layer probe with a fraction of training data."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_emb, train_labels, _ = load_split(embeddings_dir, 'train')
    test_emb, test_labels, _ = load_split(embeddings_dir, 'test')

    # Subsample training data
    n_total = len(train_emb)
    n_use = max(1, int(n_total * fraction))

    indices = np.random.permutation(n_total)[:n_use]
    train_emb_sub = train_emb[indices]
    train_labels_sub = train_labels[indices]

    embedding_dim = train_emb.shape[1]
    model = BinarySafetyClassifier(embedding_dim, hidden_dim=512, probe_depth='2layer').to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = TensorDataset(train_emb_sub, train_labels_sub)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Try val for early stopping
    has_val = (embeddings_dir / 'val_embeddings.pt').exists()
    if has_val:
        val_emb, val_labels, _ = load_split(embeddings_dir, 'val')

    best_f1 = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for emb_batch, label_batch in loader:
            emb_batch, label_batch = emb_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            out = model(emb_batch)
            loss = criterion(out, label_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                val_out = model(val_emb.to(device)).cpu()
                val_preds = (val_out > 0.5).numpy().astype(int)
                val_f1 = f1_score(val_labels.numpy() > 0.5, val_preds, zero_division=0)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

    if best_state:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(test_emb.to(device)).cpu().numpy()
        test_preds = (test_out > 0.5).astype(int)
        test_targets = test_labels.numpy()

    f1 = f1_score(test_targets > 0.5, test_preds, zero_division=0)
    acc = accuracy_score(test_targets > 0.5, test_preds)
    try:
        auc = roc_auc_score(test_targets, test_out)
    except Exception:
        auc = 0.0

    return {
        'f1': float(f1),
        'accuracy': float(acc),
        'auc_roc': float(auc),
        'n_train': n_use,
        'fraction': fraction,
        'seed': seed,
    }


def main():
    parser = argparse.ArgumentParser(description='Data scaling curve')
    parser.add_argument('--output', type=str, default='results/scaling_curve')
    parser.add_argument('--embeddings-dir', type=str, default='embeddings/scenario/siglip')
    parser.add_argument('--seeds', type=str, default=','.join(map(str, SEEDS)))
    parser.add_argument('--fractions', type=str, default=','.join(map(str, FRACTIONS)))

    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    fractions = [float(f) for f in args.fractions.split(',')]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = Path(args.embeddings_dir)

    print(f"{'='*70}")
    print(f"DATA SCALING CURVE EXPERIMENT")
    print(f"{'='*70}")
    print(f"Fractions: {fractions}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {len(fractions) * len(seeds)}")
    print()

    results = {}
    start_time = time.time()

    for frac in fractions:
        f1_list = []
        print(f"\n--- Fraction: {frac*100:.0f}% ---")

        for seed in seeds:
            print(f"  Seed {seed}...", end=' ', flush=True)
            res = train_with_fraction(embeddings_dir, frac, seed)
            f1_list.append(res['f1'] * 100)
            print(f"F1={res['f1']*100:.2f}% (n={res['n_train']})")

        mean_f1 = np.mean(f1_list)
        std_f1 = np.std(f1_list)
        results[f'{frac*100:.0f}%'] = {
            'fraction': frac,
            'n_train': int(embeddings_dir.parent.parent.name == 'scenario') or res['n_train'],
            'f1_mean': float(mean_f1),
            'f1_std': float(std_f1),
            'per_seed': f1_list,
        }

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*70}")
    print(f"SCALING CURVE SUMMARY (SigLIP 2-layer, {len(seeds)} seeds)")
    print(f"{'='*70}")
    print(f"{'Fraction':>10} {'N_train':>10} {'F1 (mean±std)':>20}")
    print("-" * 45)

    for key, r in sorted(results.items(), key=lambda x: x[1]['fraction']):
        print(f"{key:>10} {r['n_train']:>10} {r['f1_mean']:>10.2f}±{r['f1_std']:.2f}%")

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    output_file = output_dir / 'scaling_curve.json'
    with open(output_file, 'w') as f:
        json.dump({
            'model': 'siglip',
            'probe_depth': '2layer',
            'seeds': seeds,
            'elapsed_seconds': elapsed,
            'results': results,
        }, f, indent=2)
    print(f"Results saved to {output_file}")

    # Generate plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fracs = [r['fraction'] * 100 for r in sorted(results.values(), key=lambda x: x['fraction'])]
        means = [r['f1_mean'] for r in sorted(results.values(), key=lambda x: x['fraction'])]
        stds = [r['f1_std'] for r in sorted(results.values(), key=lambda x: x['fraction'])]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(fracs, means, yerr=stds, fmt='o-', capsize=5, linewidth=2, markersize=8)
        ax.set_xlabel('Training Data Fraction (%)', fontsize=12)
        ax.set_ylabel('Test F1 (%)', fontsize=12)
        ax.set_title('Data Scaling Curve (SigLIP 2-layer probe)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(50, 100)

        plot_file = output_dir / 'scaling_curve.png'
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"Plot saved to {plot_file}")
    except Exception as e:
        print(f"Plot generation failed: {e}")


if __name__ == '__main__':
    main()
