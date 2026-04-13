#!/usr/bin/env python3
"""
t-SNE Visualization from Cached Embeddings

Generates t-SNE plots using pre-extracted embeddings (no GPU needed for visualization).
Compares scenario vs temporal embeddings to visualize distribution shift.

Usage:
    python scripts/visualize_tsne_cached.py --output results/visualization
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_cached_embeddings(embeddings_dir, split='test', max_samples=2000):
    """Load pre-extracted embeddings."""
    emb_data = torch.load(embeddings_dir / f'{split}_embeddings.pt', map_location='cpu')
    label_data = torch.load(embeddings_dir / f'{split}_labels.pt', map_location='cpu')

    embeddings = emb_data['embeddings'].numpy()
    filenames = emb_data['filenames']
    labels_dict = label_data['labels']

    # Build labels and categories
    overall_labels = []
    categories = []
    valid_idx = []

    for i, fn in enumerate(filenames):
        if fn not in labels_dict:
            continue
        overall_labels.append(float(labels_dict[fn]['overall_safety']))
        # Extract category from filename (e.g., H-220607_B16_Y-14_001_0001.jpg -> B)
        parts = fn.split('_')
        cat = parts[1][0] if len(parts) >= 2 else '?'
        categories.append(cat)
        valid_idx.append(i)

    valid_idx = np.array(valid_idx)
    embeddings = embeddings[valid_idx]
    overall_labels = np.array(overall_labels)
    categories = np.array(categories)

    # Subsample if needed
    if max_samples and len(embeddings) > max_samples:
        idx = np.random.RandomState(42).choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]
        overall_labels = overall_labels[idx]
        categories = categories[idx]

    return embeddings, overall_labels, categories


def plot_tsne_safe_unsafe(embeddings_2d, labels, title, output_file):
    """Plot t-SNE colored by safe/unsafe."""
    fig, ax = plt.subplots(figsize=(10, 8))

    safe_mask = labels > 0.5
    unsafe_mask = ~safe_mask

    ax.scatter(embeddings_2d[safe_mask, 0], embeddings_2d[safe_mask, 1],
              c='#2ecc71', s=15, alpha=0.5, label=f'Safe ({safe_mask.sum()})')
    ax.scatter(embeddings_2d[unsafe_mask, 0], embeddings_2d[unsafe_mask, 1],
              c='#e74c3c', s=15, alpha=0.5, label=f'Unsafe ({unsafe_mask.sum()})')

    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, markerscale=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_tsne_categories(embeddings_2d, categories, title, output_file):
    """Plot t-SNE colored by hazard category."""
    fig, ax = plt.subplots(figsize=(10, 8))

    cat_colors = {'A': '#e74c3c', 'B': '#3498db', 'C': '#f39c12',
                  'D': '#2ecc71', 'E': '#9b59b6'}
    cat_names = {'A': 'Fall', 'B': 'Collision', 'C': 'Equipment',
                 'D': 'Environment', 'E': 'PPE'}

    for cat, color in cat_colors.items():
        mask = categories == cat
        if mask.sum() > 0:
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=color, s=15, alpha=0.5,
                      label=f'{cat_names.get(cat, cat)} ({mask.sum()})')

    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, markerscale=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_tsne_domain_comparison(scenario_2d, scenario_labels,
                                temporal_2d, temporal_labels,
                                title, output_file):
    """Plot combined t-SNE showing scenario vs temporal domains."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, emb_2d, labels, split_name in [
        (axes[0], scenario_2d, scenario_labels, 'Scenario (Random Split)'),
        (axes[1], temporal_2d, temporal_labels, 'Temporal (Oct-Nov Test)')
    ]:
        safe_mask = labels > 0.5
        unsafe_mask = ~safe_mask

        ax.scatter(emb_2d[safe_mask, 0], emb_2d[safe_mask, 1],
                  c='#2ecc71', s=12, alpha=0.4, label=f'Safe ({safe_mask.sum()})')
        ax.scatter(emb_2d[unsafe_mask, 0], emb_2d[unsafe_mask, 1],
                  c='#e74c3c', s=12, alpha=0.4, label=f'Unsafe ({unsafe_mask.sum()})')

        ax.set_xlabel('t-SNE 1', fontsize=11)
        ax.set_ylabel('t-SNE 2', fontsize=11)
        ax.set_title(split_name, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, markerscale=2)
        ax.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='t-SNE from cached embeddings')
    parser.add_argument('--embeddings-base', type=str, default='embeddings')
    parser.add_argument('--output', type=str, default='results/visualization')
    parser.add_argument('--model', type=str, default='siglip',
                       choices=['siglip', 'clip', 'dinov2'])
    parser.add_argument('--max-samples', type=int, default=2000)
    parser.add_argument('--perplexity', type=int, default=30)

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"t-SNE VISUALIZATION FROM CACHED EMBEDDINGS")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Max samples per split: {args.max_samples}")
    print()

    start_time = time.time()

    # Load scenario test embeddings
    scenario_dir = Path(args.embeddings_base) / 'scenario' / args.model
    print("Loading scenario test embeddings...")
    s_emb, s_labels, s_cats = load_cached_embeddings(scenario_dir, 'test', args.max_samples)
    print(f"  Loaded {len(s_emb)} samples")

    # Load temporal test embeddings
    temporal_dir = Path(args.embeddings_base) / 'temporal' / args.model
    print("Loading temporal test embeddings...")
    t_emb, t_labels, t_cats = load_cached_embeddings(temporal_dir, 'test', args.max_samples)
    print(f"  Loaded {len(t_emb)} samples")

    # t-SNE for scenario
    print("\nRunning t-SNE for scenario test set...")
    tsne_s = TSNE(n_components=2, perplexity=args.perplexity, random_state=42, max_iter=1000)
    s_2d = tsne_s.fit_transform(s_emb)

    plot_tsne_safe_unsafe(s_2d, s_labels,
                          f'{args.model.upper()} — Scenario Test (Safe vs Unsafe)',
                          output_dir / f'tsne_scenario_safety_{args.model}.png')

    plot_tsne_categories(s_2d, s_cats,
                        f'{args.model.upper()} — Scenario Test (by Category)',
                        output_dir / f'tsne_scenario_category_{args.model}.png')

    # t-SNE for temporal
    print("Running t-SNE for temporal test set...")
    tsne_t = TSNE(n_components=2, perplexity=args.perplexity, random_state=42, max_iter=1000)
    t_2d = tsne_t.fit_transform(t_emb)

    plot_tsne_safe_unsafe(t_2d, t_labels,
                          f'{args.model.upper()} — Temporal Test (Safe vs Unsafe)',
                          output_dir / f'tsne_temporal_safety_{args.model}.png')

    plot_tsne_categories(t_2d, t_cats,
                        f'{args.model.upper()} — Temporal Test (by Category)',
                        output_dir / f'tsne_temporal_category_{args.model}.png')

    # Combined domain comparison
    print("Running combined t-SNE (scenario + temporal)...")
    combined_emb = np.vstack([s_emb[:1000], t_emb[:1000]])
    combined_labels = np.concatenate([s_labels[:1000], t_labels[:1000]])
    domain_labels = np.array(['scenario'] * min(1000, len(s_emb)) +
                             ['temporal'] * min(1000, len(t_emb)))

    tsne_combined = TSNE(n_components=2, perplexity=args.perplexity, random_state=42, max_iter=1000)
    combined_2d = tsne_combined.fit_transform(combined_emb)

    # Domain shift visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    scenario_mask = domain_labels == 'scenario'
    temporal_mask = domain_labels == 'temporal'

    ax.scatter(combined_2d[scenario_mask, 0], combined_2d[scenario_mask, 1],
              c='#3498db', s=12, alpha=0.4, label=f'Scenario ({scenario_mask.sum()})')
    ax.scatter(combined_2d[temporal_mask, 0], combined_2d[temporal_mask, 1],
              c='#e67e22', s=12, alpha=0.4, label=f'Temporal ({temporal_mask.sum()})')

    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title(f'{args.model.upper()} — Domain Shift Visualization\n(Scenario vs Temporal Test)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, markerscale=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    domain_file = output_dir / f'tsne_domain_shift_{args.model}.png'
    plt.savefig(domain_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {domain_file}")

    # Side-by-side comparison
    n_s = min(1000, len(s_emb))
    n_t = min(1000, len(t_emb))
    plot_tsne_domain_comparison(
        combined_2d[:n_s], combined_labels[:n_s],
        combined_2d[n_s:n_s+n_t], combined_labels[n_s:n_s+n_t],
        f'{args.model.upper()} — Scenario vs Temporal Embedding Space',
        output_dir / f'tsne_comparison_{args.model}.png'
    )

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"All visualizations saved to {output_dir}/")


if __name__ == '__main__':
    main()
