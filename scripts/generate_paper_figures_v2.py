#!/usr/bin/env python3
"""
Generate paper-ready figures for the updated analysis (scenario_v2 results).

Figures:
1. Probe depth ablation (3 models)
2. Data leakage impact (old vs new split)
3. Temporal methods comparison (Binary vs Hierarchical vs DANN)
4. DANN cross-model comparison
5. Category distribution shift (Simpson's Paradox)
"""

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

COLORS = {
    'siglip': '#4C72B0',
    'clip': '#DD8452',
    'dinov2': '#55A868',
    'old': '#C44E52',
    'new': '#4C72B0',
    'binary': '#C44E52',
    'hierarchical': '#8172B2',
    'dann': '#55A868',
}

output_dir = Path('results/figures_v2')
output_dir.mkdir(parents=True, exist_ok=True)


def load_multiseed():
    """Load all multiseed_v2 results."""
    results_dir = Path('results/multiseed_v2')
    summary = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir(): continue
        rfile = d / 'results.json'
        if not rfile.exists(): continue
        with open(rfile) as f:
            data = json.load(f)
        key = d.name.rsplit('_seed', 1)[0]
        if key not in summary: summary[key] = []
        summary[key].append(data['test_metrics']['f1'] * 100)
    return summary


def fig1_probe_depth():
    """Figure 1: Probe depth ablation for all 3 models."""
    summary = load_multiseed()

    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['siglip', 'clip', 'dinov2']
    depths = ['linear', '1layer', '2layer']
    depth_labels = ['Linear', '1-Layer MLP', '2-Layer MLP']
    x = np.arange(len(depths))
    width = 0.25

    for i, model in enumerate(models):
        means = []
        stds = []
        for depth in depths:
            key = f'scenario_v2_{model}_{depth}'
            vals = summary.get(key, [0])
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        bars = ax.bar(x + i * width, means, width, yerr=stds,
                     label=model.upper(), color=COLORS[model],
                     capsize=3, edgecolor='black', linewidth=0.5)

        # Add value labels
        for j, (m, s) in enumerate(zip(means, stds)):
            ax.text(x[j] + i * width, m + s + 0.5, f'{m:.1f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Probe Architecture')
    ax.set_ylabel('F1 Score (%)')
    ax.set_title('RQ1: Probe Depth Ablation\n(Sequence-Level Split, 5-seed mean±std)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(depth_labels)
    ax.legend()
    ax.set_ylim(60, 95)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_probe_depth_v2.png', bbox_inches='tight')
    plt.close()
    print(f'Saved: fig1_probe_depth_v2.png')


def fig2_leakage_impact():
    """Figure 2: Data leakage impact (old vs new split)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ['SigLIP', 'CLIP', 'DINOv2']
    old_f1 = [96.11, 91.54, 90.92]
    new_f1 = [87.28, 78.91, 77.94]
    deltas = [n - o for n, o in zip(new_f1, old_f1)]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, old_f1, width, label='Frame-Level Split (leaked)',
                  color=COLORS['old'], edgecolor='black', linewidth=0.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, new_f1, width, label='Sequence-Level Split (clean)',
                  color=COLORS['new'], edgecolor='black', linewidth=0.5, alpha=0.8)

    # Add delta annotations
    for i, (o, n, d) in enumerate(zip(old_f1, new_f1, deltas)):
        ax.annotate(f'{d:+.1f}%p', xy=(i, min(o, n) - 1),
                   fontsize=10, ha='center', color='red', fontweight='bold')

    ax.set_xlabel('Foundation Model (2-Layer Probe)')
    ax.set_ylabel('F1 Score (%)')
    ax.set_title('Impact of Data Leakage on Reported Performance\n(5-seed mean, 2-layer probe)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(70, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_leakage_impact.png', bbox_inches='tight')
    plt.close()
    print(f'Saved: fig2_leakage_impact.png')


def fig3_temporal_methods():
    """Figure 3: Temporal distribution shift - methods comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [
        'Binary\n(frame-level split)',
        'Binary\n(sequence-level split)',
        'Hierarchical\n(no adaptation)',
        'DANN\n(SigLIP)',
    ]
    f1_values = [66.11, 95.19, 94.88, 99.19]
    colors = [COLORS['binary'], COLORS['new'], COLORS['hierarchical'], COLORS['dann']]

    bars = ax.bar(range(len(methods)), f1_values, color=colors,
                 edgecolor='black', linewidth=0.5, width=0.6)

    # Add value labels
    for i, v in enumerate(f1_values):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom',
               fontsize=11, fontweight='bold')

    # Add arrow showing improvement
    ax.annotate('', xy=(3, 99.19), xytext=(0, 66.11),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, ls='--'))

    ax.set_ylabel('Temporal Test F1 Score (%)')
    ax.set_title('RQ2: Temporal Distribution Shift — Method Comparison\n(SigLIP 2-Layer, Temporal Test Set)')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylim(55, 105)
    ax.axhline(y=90, color='gray', linestyle=':', alpha=0.5, label='90% threshold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_temporal_methods.png', bbox_inches='tight')
    plt.close()
    print(f'Saved: fig3_temporal_methods.png')


def fig4_dann_crossmodel():
    """Figure 4: DANN cross-model comparison."""
    fig, ax = plt.subplots(figsize=(9, 6))

    models = ['SigLIP', 'CLIP', 'DINOv2']
    baseline = [66.11, 60.72, 56.65]
    dann = [99.19, 97.57, 98.11]
    improvement = [d - b for d, b in zip(dann, baseline)]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline (no adaptation)',
                  color=COLORS['binary'], edgecolor='black', linewidth=0.5, alpha=0.7)
    bars2 = ax.bar(x + width/2, dann, width, label='DANN (domain adaptation)',
                  color=COLORS['dann'], edgecolor='black', linewidth=0.5, alpha=0.8)

    # Add improvement annotations
    for i, (b, d, imp) in enumerate(zip(baseline, dann, improvement)):
        ax.annotate(f'+{imp:.1f}%p', xy=(i + width/2, d + 0.5),
                   fontsize=10, ha='center', color='green', fontweight='bold')

    ax.set_xlabel('Foundation Model')
    ax.set_ylabel('Temporal Test F1 Score (%)')
    ax.set_title('DANN Domain Adaptation Across Foundation Models\n(5-seed, source val model selection)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='lower right')
    ax.set_ylim(45, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_dann_crossmodel.png', bbox_inches='tight')
    plt.close()
    print(f'Saved: fig4_dann_crossmodel.png')


def fig5_story_flow():
    """Figure 5: Research story flow diagram."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    stages = [
        ('1. Foundation\nModel Probing', 87.28, 'Scenario F1', COLORS['new']),
        ('2. Temporal\nFailure', 66.11, 'Temporal F1\n(frame split)', COLORS['binary']),
        ('3. Sequence\nSplit Fix', 95.19, 'Temporal F1\n(seq split)', COLORS['new']),
        ('4. Hierarchical\nPipeline', 94.88, 'Temporal F1\n(structural)', COLORS['hierarchical']),
        ('5. DANN\nAdaptation', 99.19, 'Temporal F1\n(learned)', COLORS['dann']),
    ]

    for i, (title, value, ylabel, color) in enumerate(stages):
        ax = axes[i]
        ax.bar([0], [value], color=color, edgecolor='black', linewidth=0.5, width=0.5)
        ax.text(0, value + 1, f'{value:.1f}%', ha='center', va='bottom',
               fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(0, 110)
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3)

    # Add arrows between subplots
    fig.suptitle('Research Story Flow: From Problem to Solution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_story_flow.png', bbox_inches='tight')
    plt.close()
    print(f'Saved: fig5_story_flow.png')


if __name__ == '__main__':
    print("Generating paper figures (v2)...")
    fig1_probe_depth()
    fig2_leakage_impact()
    fig3_temporal_methods()
    fig4_dann_crossmodel()
    fig5_story_flow()
    print(f"\nAll figures saved to {output_dir}/")
