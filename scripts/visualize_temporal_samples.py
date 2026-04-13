#!/usr/bin/env python3
"""
Temporal Shift Sample Visualization (W3 Response)

Creates a grid of train vs test sample images per hazard category,
showing visual differences (season, lighting, environment) between
training period (Jun-Sep) and test period (Oct-Nov).

Usage:
    python scripts/visualize_temporal_samples.py \
        --data-dir data_temporal \
        --output results/figures_final
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'figure.dpi': 150, 'font.family': 'DejaVu Sans',
})

CAT_NAMES = {
    'A': 'Fall Hazard',
    'B': 'Collision Risk',
    'C': 'Equipment Hazard',
    'D': 'Environmental Risk',
    'E': 'Protective Gear',
}

CATEGORIES = ['A', 'B', 'C', 'D', 'E']


def extract_category(filename):
    """Extract category letter from filename."""
    m = re.search(r'_([A-F])\d{2}_', filename)
    return m.group(1) if m else None


def extract_month(filename):
    """Extract month from filename: H-YYMMDD_..."""
    m = re.search(r'H-(\d{2})(\d{2})(\d{2})_', filename)
    if m:
        return int(m.group(2))
    return None


def collect_images_by_category(data_dir, split):
    """Group image paths by category for a given split."""
    split_dir = Path(data_dir) / split
    cat_images = {c: [] for c in CATEGORIES}

    for img_path in sorted(split_dir.glob("*.jpg")):
        cat = extract_category(img_path.name)
        if cat and cat in cat_images:
            cat_images[cat].append(img_path)

    return cat_images


def main():
    parser = argparse.ArgumentParser(description="Temporal shift sample visualization")
    parser.add_argument("--data-dir", type=str, default="data_temporal")
    parser.add_argument("--output", type=str, default="results/figures_final")
    parser.add_argument("--n-samples", type=int, default=3,
                       help="Number of sample images per category per split")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible sampling")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Collect images
    train_images = collect_images_by_category(args.data_dir, "train")
    test_images = collect_images_by_category(args.data_dir, "test")

    n_samples = args.n_samples
    n_cats = len(CATEGORIES)
    n_cols = n_samples * 2  # train + test

    fig, axes = plt.subplots(n_cats, n_cols, figsize=(n_cols * 2.5, n_cats * 2.5))

    for row, cat in enumerate(CATEGORIES):
        # Sample train images
        train_pool = train_images[cat]
        test_pool = test_images[cat]

        if len(train_pool) >= n_samples:
            train_idx = rng.choice(len(train_pool), size=n_samples, replace=False)
            train_selected = [train_pool[i] for i in train_idx]
        else:
            train_selected = train_pool[:n_samples]

        if len(test_pool) >= n_samples:
            test_idx = rng.choice(len(test_pool), size=n_samples, replace=False)
            test_selected = [test_pool[i] for i in test_idx]
        else:
            test_selected = test_pool[:n_samples]

        # Plot train samples
        for col in range(n_samples):
            ax = axes[row, col]
            if col < len(train_selected):
                img = Image.open(train_selected[col]).convert("RGB")
                ax.imshow(img)
                month = extract_month(train_selected[col].name)
                if month and row == n_cats - 1:
                    ax.set_xlabel(f"M{month:02d}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0 and col == n_samples // 2:
                ax.set_title("Train (Jun-Sep)", fontweight='bold', fontsize=12)

        # Plot test samples
        for col in range(n_samples):
            ax = axes[row, n_samples + col]
            if col < len(test_selected):
                img = Image.open(test_selected[col]).convert("RGB")
                ax.imshow(img)
                month = extract_month(test_selected[col].name)
                if month and row == n_cats - 1:
                    ax.set_xlabel(f"M{month:02d}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0 and col == n_samples // 2:
                ax.set_title("Test (Oct-Nov)", fontweight='bold', fontsize=12)

        # Row label
        axes[row, 0].set_ylabel(
            f"{cat}: {CAT_NAMES[cat]}\n(n_train={len(train_pool)}, n_test={len(test_pool)})",
            fontsize=9, rotation=90, labelpad=10
        )

    # Add vertical separator
    for row in range(n_cats):
        axes[row, n_samples - 1].spines['right'].set_visible(True)
        axes[row, n_samples - 1].spines['right'].set_linewidth(2)
        axes[row, n_samples - 1].spines['right'].set_color('red')

    fig.suptitle("Temporal Distribution Shift: Train vs Test Sample Images",
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_path = output_dir / "fig_temporal_samples_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    # Print stats
    print("\nCategory image counts:")
    for cat in CATEGORIES:
        print(f"  {cat} ({CAT_NAMES[cat]}): train={len(train_images[cat])}, test={len(test_images[cat])}")


if __name__ == "__main__":
    main()
