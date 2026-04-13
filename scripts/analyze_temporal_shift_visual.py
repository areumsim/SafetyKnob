#!/usr/bin/env python3
"""
Temporal Shift Visual Analysis

Provides concrete evidence for feature-level temporal distribution shift:
1. t-SNE visualization of train vs test embeddings
2. MMD (Maximum Mean Discrepancy) per category
3. Monthly distribution analysis
"""

import json
import re
from pathlib import Path
from collections import Counter

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score

OUTPUT_DIR = Path('results/figures_final')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'figure.dpi': 150, 'font.family': 'DejaVu Sans',
})


def extract_month(filename):
    """Extract month from filename: H-YYMMDD_..."""
    m = re.search(r'H-(\d{2})(\d{2})(\d{2})_', filename)
    if m:
        return int(m.group(2))  # month
    return None


def extract_category(filename):
    m = re.search(r'_([A-F])\d{2}_', filename)
    return m.group(1) if m else None


def compute_mmd(X, Y, gamma=1.0):
    """Compute Maximum Mean Discrepancy between two sets of embeddings."""
    # Use linear kernel for efficiency on high-dim data
    XX = X @ X.T
    YY = Y @ Y.T
    XY = X @ Y.T
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return float(mmd)


def load_temporal_data(model='siglip'):
    """Load temporal train and test embeddings with metadata."""
    emb_dir = Path(f'embeddings/temporal/{model}')

    train_data = torch.load(emb_dir / 'train_embeddings.pt', map_location='cpu')
    train_labels = torch.load(emb_dir / 'train_labels.pt', map_location='cpu')
    test_data = torch.load(emb_dir / 'test_embeddings.pt', map_location='cpu')
    test_labels = torch.load(emb_dir / 'test_labels.pt', map_location='cpu')

    return {
        'train_emb': train_data['embeddings'],
        'train_files': train_data['filenames'],
        'train_labels_dict': train_labels['labels'],
        'test_emb': test_data['embeddings'],
        'test_files': test_data['filenames'],
        'test_labels_dict': test_labels['labels'],
    }


def fig_tsne_temporal(data, model='siglip'):
    """t-SNE showing train vs test embedding clusters."""
    # Subsample for speed
    n_train = min(2000, len(data['train_emb']))
    n_test = min(2000, len(data['test_emb']))

    np.random.seed(42)
    train_idx = np.random.choice(len(data['train_emb']), n_train, replace=False)
    test_idx = np.random.choice(len(data['test_emb']), n_test, replace=False)

    train_emb = data['train_emb'][train_idx].numpy()
    test_emb = data['test_emb'][test_idx].numpy()

    # Get labels
    train_safety = []
    for i in train_idx:
        fname = data['train_files'][i]
        label = data['train_labels_dict'].get(fname, {}).get('overall_safety', 1)
        train_safety.append(int(float(label)))

    test_safety = []
    for i in test_idx:
        fname = data['test_files'][i]
        label = data['test_labels_dict'].get(fname, {}).get('overall_safety', 1)
        test_safety.append(int(float(label)))

    all_emb = np.vstack([train_emb, test_emb])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(all_emb)

    train_coords = coords[:n_train]
    test_coords = coords[n_train:]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Train vs Test (domain)
    ax = axes[0]
    ax.scatter(train_coords[:, 0], train_coords[:, 1], c='#2196F3', alpha=0.3, s=8, label='Train (Jun-Sep)')
    ax.scatter(test_coords[:, 0], test_coords[:, 1], c='#F44336', alpha=0.3, s=8, label='Test (Oct-Nov)')
    ax.set_title('Temporal Domain Separation')
    ax.legend(markerscale=3, fontsize=10)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')

    # Plot 2: Safe vs Danger (colored by class, shaped by domain)
    ax = axes[1]
    for i, (coords_set, safety_list, domain, marker) in enumerate([
        (train_coords, train_safety, 'Train', 'o'),
        (test_coords, test_safety, 'Test', '^'),
    ]):
        safe_mask = np.array(safety_list) == 1
        danger_mask = ~safe_mask
        ax.scatter(coords_set[safe_mask, 0], coords_set[safe_mask, 1],
                  c='#4CAF50', alpha=0.3, s=8, marker=marker, label=f'{domain} Safe')
        ax.scatter(coords_set[danger_mask, 0], coords_set[danger_mask, 1],
                  c='#FF5722', alpha=0.3, s=8, marker=marker, label=f'{domain} Danger')

    ax.set_title('Safety Class × Domain')
    ax.legend(markerscale=3, fontsize=9)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')

    plt.suptitle(f'Temporal Distribution Shift Visualization ({model.upper()})', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_tsne_temporal_shift.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig4_tsne_temporal_shift.png'}")


def fig_mmd_per_category(data, model='siglip'):
    """Compute and visualize MMD per category between train and test."""
    categories = {'A': 'Fall Hazard', 'B': 'Collision Risk', 'C': 'Equipment Hazard',
                  'D': 'Environmental Risk', 'E': 'Protective Gear'}

    results = {}
    for cat_code, cat_name in categories.items():
        # Filter train
        train_idx = [i for i, f in enumerate(data['train_files']) if extract_category(f) == cat_code]
        test_idx = [i for i, f in enumerate(data['test_files']) if extract_category(f) == cat_code]

        if len(train_idx) < 10 or len(test_idx) < 10:
            continue

        train_emb = data['train_emb'][train_idx].numpy()
        test_emb = data['test_emb'][test_idx].numpy()

        # Normalize for fair MMD
        train_norm = train_emb / (np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-8)
        test_norm = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-8)

        mmd = compute_mmd(train_norm, test_norm)
        results[cat_code] = {'name': cat_name, 'mmd': mmd,
                             'n_train': len(train_idx), 'n_test': len(test_idx)}
        print(f"  {cat_code} ({cat_name}): MMD={mmd:.6f}, n_train={len(train_idx)}, n_test={len(test_idx)}")

    # Also compute overall MMD
    train_norm = data['train_emb'].numpy()
    train_norm = train_norm / (np.linalg.norm(train_norm, axis=1, keepdims=True) + 1e-8)
    test_norm = data['test_emb'].numpy()
    test_norm = test_norm / (np.linalg.norm(test_norm, axis=1, keepdims=True) + 1e-8)
    overall_mmd = compute_mmd(train_norm, test_norm)
    print(f"  Overall MMD: {overall_mmd:.6f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    cats = sorted(results.keys())
    names = [results[c]['name'] for c in cats]
    mmds = [results[c]['mmd'] for c in cats]

    colors = ['#E91E63', '#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
    bars = ax.bar(names, mmds, color=colors[:len(cats)], edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.axhline(y=overall_mmd, color='red', linestyle='--', alpha=0.7, label=f'Overall MMD ({overall_mmd:.4f})')

    for bar, mmd in zip(bars, mmds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{mmd:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('MMD (Linear Kernel)')
    ax.set_title(f'Per-Category Embedding Shift (Train vs Test, {model.upper()})')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_mmd_per_category.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig5_mmd_per_category.png'}")

    return results, overall_mmd


def fig_monthly_distribution(data):
    """Show monthly distribution of images and class balance."""
    train_months = Counter()
    test_months = Counter()
    train_safety = Counter()
    test_safety = Counter()

    for f in data['train_files']:
        m = extract_month(f)
        if m:
            train_months[m] += 1
            label = data['train_labels_dict'].get(f, {}).get('overall_safety', 1)
            train_safety[(m, int(float(label)))] += 1

    for f in data['test_files']:
        m = extract_month(f)
        if m:
            test_months[m] += 1
            label = data['test_labels_dict'].get(f, {}).get('overall_safety', 1)
            test_safety[(m, int(float(label)))] += 1

    months = sorted(set(train_months.keys()) | set(test_months.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Monthly count by split
    ax = axes[0]
    train_counts = [train_months.get(m, 0) for m in months]
    test_counts = [test_months.get(m, 0) for m in months]
    month_labels = [f'2022-{m:02d}' for m in months]

    x = np.arange(len(months))
    ax.bar(x - 0.2, train_counts, 0.4, label='Train (Jun-Sep)', color='#2196F3', alpha=0.8)
    ax.bar(x + 0.2, test_counts, 0.4, label='Test (Oct-Nov)', color='#F44336', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, rotation=45)
    ax.set_ylabel('Image Count')
    ax.set_title('Monthly Distribution by Split')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Safe ratio by month
    ax = axes[1]
    safe_ratios = []
    for m in months:
        total = train_months.get(m, 0) + test_months.get(m, 0)
        safe = train_safety.get((m, 1), 0) + test_safety.get((m, 1), 0)
        safe_ratios.append(safe / total * 100 if total > 0 else 0)

    colors_month = ['#2196F3' if m <= 9 else '#F44336' for m in months]
    ax.bar(month_labels, safe_ratios, color=colors_month, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Safe Image Ratio (%)')
    ax.set_title('Class Balance Shift by Month')
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Temporal Distribution Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_monthly_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig6_monthly_distribution.png'}")


if __name__ == '__main__':
    print("=== Temporal Shift Visual Analysis ===\n")

    model = 'siglip'
    print(f"Loading {model} embeddings...")
    data = load_temporal_data(model)
    print(f"Train: {len(data['train_emb'])}, Test: {len(data['test_emb'])}")

    print("\n--- t-SNE Visualization ---")
    fig_tsne_temporal(data, model)

    print("\n--- MMD Per Category ---")
    mmd_results, overall_mmd = fig_mmd_per_category(data, model)

    print("\n--- Monthly Distribution ---")
    fig_monthly_distribution(data)

    # Save numerical results
    results = {
        'model': model,
        'mmd_per_category': {k: v for k, v in mmd_results.items()},
        'overall_mmd': overall_mmd,
    }
    with open(OUTPUT_DIR / 'temporal_shift_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'temporal_shift_analysis.json'}")
