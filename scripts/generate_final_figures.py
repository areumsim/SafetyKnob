#!/usr/bin/env python3
"""
Generate final paper figures incorporating all corrected results.

Figures:
  1. Temporal methods comparison (Baseline, DANN, Reweight, Hierarchical, LoRA)
  2. Cross-domain zero-shot transfer comparison (3 models)
  3. LoRA vs Frozen: gap closure visualization
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

OUTPUT_DIR = Path('results/figures_final')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fig1_temporal_methods():
    """Bar chart comparing all temporal shift correction methods."""
    methods = ['Frozen\nBaseline', 'DANN\nClean', 'Label\nReweight', 'Hierarchical', 'LoRA\n(r=16)']
    f1_means = [66.11, 65.28, 66.50, 67.56, None]  # LoRA will be filled from results
    f1_stds = [0.72, 0.88, 0.69, 0.28, None]

    # Try to load LoRA 5-seed results
    lora_f1s = []
    for seed_dir in ['siglip', 'siglip_seed789', 'siglip_seed2024']:
        p = Path(f'results/lora_temporal/{seed_dir}/lora_results.json')
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            for r in data:
                lora_f1s.append(r['test_metrics']['f1'] * 100)

    if lora_f1s:
        f1_means[-1] = np.mean(lora_f1s)
        f1_stds[-1] = np.std(lora_f1s)
        print(f"LoRA: {f1_means[-1]:.2f}±{f1_stds[-1]:.2f}% ({len(lora_f1s)} seeds)")
    else:
        f1_means[-1] = 77.18
        f1_stds[-1] = 0.86
        print("Using cached LoRA values (3-seed)")

    colors = ['#2196F3', '#FF5722', '#FF9800', '#4CAF50', '#9C27B0']
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(methods, f1_means, yerr=f1_stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)

    # Add value labels
    for bar, mean, std in zip(bars, f1_means, f1_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Reference lines
    ax.axhline(y=96.11, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(len(methods)-0.5, 96.5, 'Scenario i.i.d. (96.11%)', ha='right',
            color='green', fontsize=10, alpha=0.7)
    ax.axhline(y=66.11, color='gray', linestyle=':', alpha=0.4)

    ax.set_ylabel('Temporal Test F1 (%)')
    ax.set_title('Temporal Distribution Shift: Method Comparison (SigLIP 2-layer)')
    ax.set_ylim(55, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_temporal_methods.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig1_temporal_methods.png'}")


def fig2_cross_domain():
    """Grouped bar chart for cross-domain results."""
    models = ['SigLIP', 'CLIP', 'DINOv2']
    experiments = ['Zero-shot\nTransfer', 'PPE\nfrom Scratch', 'Pre-train +\nFine-tune']

    # Load results
    data = {}
    for model in ['siglip', 'clip', 'dinov2']:
        p = Path(f'results/cross_domain/{model}/cross_domain_results.json')
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            data[model] = d['results']

    if len(data) < 2:
        print("Insufficient cross-domain results for figure")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(experiments))
    width = 0.25
    colors = ['#E91E63', '#2196F3', '#4CAF50']

    for i, (model_key, model_name) in enumerate(zip(['siglip', 'clip', 'dinov2'], models)):
        if model_key not in data:
            continue
        means = []
        stds = []
        for exp_key in ['zero_shot', 'scratch', 'finetune']:
            if exp_key in data[model_key]:
                means.append(data[model_key][exp_key]['ppe_f1_mean'])
                stds.append(data[model_key][exp_key]['ppe_f1_std'])
            else:
                means.append(0)
                stds.append(0)

        bars = ax.bar(x + i*width, means, width, yerr=stds, capsize=4,
                      label=model_name, color=colors[i], alpha=0.85, edgecolor='black', linewidth=0.5)

        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('PPE Test F1 (%)')
    ax.set_title('Cross-Domain Transfer: AI Hub → Construction-PPE')
    ax.set_xticks(x + width)
    ax.set_xticklabels(experiments)
    ax.legend()
    ax.set_ylim(50, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_cross_domain.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig2_cross_domain.png'}")


def fig3_lora_gap():
    """Visualization of LoRA gap closure."""
    # Load LoRA results
    lora_f1s = []
    for seed_dir in ['siglip', 'siglip_seed789', 'siglip_seed2024']:
        p = Path(f'results/lora_temporal/{seed_dir}/lora_results.json')
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            for r in data:
                lora_f1s.append(r['test_metrics']['f1'] * 100)

    lora_mean = np.mean(lora_f1s) if lora_f1s else 77.18

    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Scenario\n(i.i.d.)', 'Temporal\n(Frozen)', 'Temporal\n(LoRA)']
    f1_values = [96.11, 66.11, lora_mean]
    colors = ['#4CAF50', '#F44336', '#9C27B0']

    bars = ax.bar(categories, f1_values, color=colors, edgecolor='black',
                  linewidth=0.8, alpha=0.85, width=0.6)

    for bar, val in zip(bars, f1_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)

    # Draw gap arrows
    gap_total = 96.11 - 66.11
    gap_recovered = lora_mean - 66.11
    pct_recovered = gap_recovered / gap_total * 100

    ax.annotate('', xy=(1, 96.11), xytext=(1, 66.11),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(1.35, 81, f'-{gap_total:.0f}%p\n(total gap)', ha='left',
            color='red', fontsize=10)

    ax.annotate('', xy=(2, lora_mean), xytext=(2, 66.11),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax.text(2.35, (66.11+lora_mean)/2, f'+{gap_recovered:.1f}%p\n({pct_recovered:.0f}% recovered)',
            ha='left', color='purple', fontsize=10)

    ax.set_ylabel('F1 Score (%)')
    ax.set_title('LoRA Fine-tuning: Temporal Shift Gap Recovery')
    ax.set_ylim(55, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_lora_gap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'fig3_lora_gap.png'}")


if __name__ == '__main__':
    print("Generating final paper figures...")
    fig1_temporal_methods()
    fig2_cross_domain()
    fig3_lora_gap()
    print("\nAll figures saved to results/figures_final/")
