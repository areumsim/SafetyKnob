#!/usr/bin/env python3
"""
LoRA Rank Ablation Visualization (W1 Response)

Reads per-epoch metrics from rank ablation experiments and generates:
1. Learning curves (epoch x test F1, one line per rank)
2. Rank ablation bar chart (rank x final test F1)
3. Augmentation 2x2 comparison heatmap (W2)

Usage:
    python scripts/analyze_lora_ablation.py \
        --results-dir results/lora_rank_ablation \
        --output results/figures_final
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'figure.dpi': 150, 'font.family': 'DejaVu Sans',
})

RANK_COLORS = {
    0: '#95a5a6',   # gray (head-only)
    4: '#3498db',   # blue
    8: '#2ecc71',   # green
    16: '#e74c3c',  # red
    32: '#9b59b6',  # purple
}


def load_epoch_metrics(results_dir, ranks, seed=42):
    """Load epoch metrics for each rank."""
    all_metrics = {}
    for r in ranks:
        path = Path(results_dir) / f"r{r}" / f"epoch_metrics_seed{seed}.json"
        if path.exists():
            with open(path) as f:
                all_metrics[r] = json.load(f)
            print(f"  Loaded r={r}: {len(all_metrics[r])} epochs")
        else:
            print(f"  WARNING: Not found: {path}")
    return all_metrics


def load_final_results(results_dir, ranks):
    """Load final lora_results.json for each rank."""
    final = {}
    for r in ranks:
        path = Path(results_dir) / f"r{r}" / "lora_results.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                final[r] = data[0]  # First seed
            print(f"  Loaded r={r} final: F1={final[r]['test_metrics']['f1']*100:.2f}%")
        else:
            print(f"  WARNING: Not found: {path}")
    return final


def plot_learning_curves(all_metrics, output_dir):
    """Plot epoch x test F1 learning curves for each rank."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Test F1 learning curves
    ax = axes[0]
    for r in sorted(all_metrics.keys()):
        metrics = all_metrics[r]
        epochs = [m["epoch"] for m in metrics]
        test_f1s = [m["test_f1"] * 100 for m in metrics]
        label = f"r={r} (head-only)" if r == 0 else f"r={r}"
        linestyle = "--" if r == 0 else "-"
        ax.plot(epochs, test_f1s, marker='o', markersize=4,
               label=label, color=RANK_COLORS.get(r, 'black'),
               linestyle=linestyle, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test F1 (%)')
    ax.set_title('Test F1 Learning Curves by LoRA Rank')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Panel 2: Val F1 learning curves
    ax = axes[1]
    for r in sorted(all_metrics.keys()):
        metrics = all_metrics[r]
        epochs = [m["epoch"] for m in metrics]
        val_f1s = [m["val_f1"] * 100 for m in metrics]
        label = f"r={r} (head-only)" if r == 0 else f"r={r}"
        linestyle = "--" if r == 0 else "-"
        ax.plot(epochs, val_f1s, marker='s', markersize=4,
               label=label, color=RANK_COLORS.get(r, 'black'),
               linestyle=linestyle, linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val F1 (%)')
    ax.set_title('Val F1 Learning Curves by LoRA Rank')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "fig_lora_learning_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_rank_ablation(all_metrics, final_results, output_dir):
    """Plot rank vs final test F1 bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ranks = sorted(all_metrics.keys())
    # Use last epoch test F1 from epoch metrics
    f1s = []
    for r in ranks:
        if r in final_results:
            f1s.append(final_results[r]["test_metrics"]["f1"] * 100)
        elif r in all_metrics:
            f1s.append(all_metrics[r][-1]["test_f1"] * 100)

    colors = [RANK_COLORS.get(r, '#333333') for r in ranks]
    bars = ax.bar(range(len(ranks)), f1s, color=colors, edgecolor='black', linewidth=0.5)

    # Value annotations
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{f1:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    labels = [f"r=0\n(head-only)" if r == 0 else f"r={r}" for r in ranks]
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('LoRA Rank')
    ax.set_ylabel('Test F1 (%)')
    ax.set_title('LoRA Rank Ablation: Temporal Test Performance')

    # Add trainable params annotation if available
    if final_results:
        for i, r in enumerate(ranks):
            if r in final_results:
                params = final_results[r]["trainable_params"]
                ax.text(i, 2, f'{params:,}\nparams', ha='center', va='bottom',
                       fontsize=7, color='white', fontweight='bold')

    ax.set_ylim(0, max(f1s) + 5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "fig_lora_rank_ablation.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def plot_augmentation_2x2(aug_results_dir, rank_ablation_dir, output_dir):
    """Plot 2x2 augmentation comparison heatmap."""
    # Load results: (r=0, no-aug), (r=0, aug), (r=16, no-aug), (r=16, aug)
    configs = {
        (0, False): rank_ablation_dir / "r0" / "lora_results.json",
        (0, True): Path(aug_results_dir) / "r0_augment" / "lora_results.json",
        (16, False): rank_ablation_dir / "r16" / "lora_results.json",
        (16, True): Path(aug_results_dir) / "r16_augment" / "lora_results.json",
    }

    f1_matrix = np.full((2, 2), np.nan)
    labels_matrix = [['' for _ in range(2)] for _ in range(2)]

    for (r, aug), path in configs.items():
        row = 0 if r == 0 else 1
        col = 0 if not aug else 1
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                f1 = data[0]["test_metrics"]["f1"] * 100
                f1_matrix[row, col] = f1
                labels_matrix[row][col] = f'{f1:.1f}%'
                print(f"  r={r}, aug={aug}: F1={f1:.1f}%")
        else:
            labels_matrix[row][col] = 'N/A'
            print(f"  WARNING: Not found: {path}")

    if np.all(np.isnan(f1_matrix)):
        print("  No augmentation results available, skipping heatmap.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    # Use a diverging colormap centered on the mean
    valid = f1_matrix[~np.isnan(f1_matrix)]
    if len(valid) > 0:
        vmin, vmax = valid.min() - 2, valid.max() + 2
    else:
        vmin, vmax = 50, 100

    im = ax.imshow(f1_matrix, cmap='RdYlGn', vmin=vmin, vmax=vmax, aspect='auto')

    # Annotate cells
    for i in range(2):
        for j in range(2):
            color = 'black' if not np.isnan(f1_matrix[i, j]) else 'gray'
            ax.text(j, i, labels_matrix[i][j], ha='center', va='center',
                   fontsize=16, fontweight='bold', color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Augmentation', 'With Augmentation'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['r=0 (Head-only)', 'r=16 (LoRA)'])
    ax.set_title('Augmentation x LoRA Ablation: Test F1 (%)')

    plt.colorbar(im, ax=ax, label='Test F1 (%)')
    plt.tight_layout()

    out_path = output_dir / "fig_augmentation_2x2.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="LoRA rank ablation visualization")
    parser.add_argument("--results-dir", type=str, default="results/lora_rank_ablation")
    parser.add_argument("--aug-results-dir", type=str, default="results/augmentation_ablation")
    parser.add_argument("--output", type=str, default="results/figures_final")
    parser.add_argument("--ranks", type=str, default="0,4,8,16,32",
                       help="Comma-separated rank values")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir)
    ranks = [int(r) for r in args.ranks.split(",")]

    print("=" * 60)
    print("LORA RANK ABLATION ANALYSIS")
    print("=" * 60)

    # Load epoch metrics
    print("\nLoading epoch metrics...")
    all_metrics = load_epoch_metrics(results_dir, ranks, args.seed)

    # Load final results
    print("\nLoading final results...")
    final_results = load_final_results(results_dir, ranks)

    if not all_metrics:
        print("\nERROR: No epoch metrics found. Run rank ablation experiments first.")
        return

    # Generate figures
    print("\nGenerating learning curves...")
    plot_learning_curves(all_metrics, output_dir)

    print("\nGenerating rank ablation bar chart...")
    plot_rank_ablation(all_metrics, final_results, output_dir)

    # Augmentation 2x2 heatmap
    print("\nGenerating augmentation 2x2 heatmap...")
    plot_augmentation_2x2(args.aug_results_dir, results_dir, output_dir)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"RANK ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Rank':<8} {'Method':<15} {'Final Test F1':>15} {'Best Val F1':>15}")
    print("-" * 55)
    for r in ranks:
        method = "head-only" if r == 0 else f"LoRA r={r}"
        if r in all_metrics:
            test_f1 = all_metrics[r][-1]["test_f1"] * 100
            best_val = max(m["val_f1"] for m in all_metrics[r]) * 100
            print(f"  {r:<6} {method:<15} {test_f1:>13.2f}% {best_val:>13.2f}%")

    # Overfitting analysis
    print(f"\n--- Overfitting Analysis ---")
    for r in sorted(all_metrics.keys()):
        metrics = all_metrics[r]
        test_f1s = [m["test_f1"] * 100 for m in metrics]
        val_f1s = [m["val_f1"] * 100 for m in metrics]
        best_epoch_val = np.argmax(val_f1s) + 1
        best_epoch_test = np.argmax(test_f1s) + 1
        gap = max(val_f1s) - test_f1s[np.argmax(val_f1s)]
        trend = "plateau" if (max(test_f1s) - min(test_f1s[-3:])) < 1.0 else "diverging"
        print(f"  r={r}: best_val@epoch{best_epoch_val}, best_test@epoch{best_epoch_test}, "
              f"val-test gap={gap:.1f}%p, test trend={trend}")


if __name__ == "__main__":
    main()
