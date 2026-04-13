#!/usr/bin/env python3
"""
논문용 Figure 생성 스크립트

기존 실험 결과(per-image predictions)를 활용하여 논문에 필요한
고해상도 시각화를 생성합니다.

생성 Figure:
  1. Confusion Matrix (Scenario + Temporal, 3모델)
  2. ROC Curve (3모델 비교)
  3. Precision-Recall Curve
  4. Error 신뢰도 분포 (FP/FN 히스토그램)
  5. 모델 간 예측 상관 히트맵
  6. Scaling Curve (에러바 포함)
  7. Temporal Per-Category 비교
  8. Ensemble Ablation 비교
  9. Probe Depth Ablation 비교
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# 한글 폰트 설정 시도
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
except:
    pass

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

RESULTS_DIR = Path('results')
OUTPUT_DIR = RESULTS_DIR / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_COLORS = {
    'siglip': '#2196F3',
    'clip': '#FF9800',
    'dinov2': '#4CAF50',
    'ensemble': '#9C27B0',
    'resnet50': '#F44336',
    'efficientnet': '#795548'
}

MODEL_LABELS = {
    'siglip': 'SigLIP',
    'clip': 'CLIP',
    'dinov2': 'DINOv2',
    'ensemble': 'Ensemble',
    'resnet50': 'ResNet-50',
    'efficientnet': 'EfficientNet-B0'
}


def load_predictions(result_path):
    """Load per-image predictions from results.json."""
    with open(result_path) as f:
        data = json.load(f)
    preds = data.get('per_image_predictions', {})
    if not preds:
        return None, None, None

    images = sorted(preds.keys())
    y_true = np.array([preds[img]['true_label'] for img in images])
    y_score = np.array([preds[img]['pred_score'] for img in images])
    y_pred = np.array([preds[img]['pred_label'] for img in images])
    return y_true, y_score, y_pred


# =============================================
# Figure 1: Confusion Matrices
# =============================================
def generate_confusion_matrices():
    """Generate confusion matrices for 3 models × 2 splits."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Confusion Matrices: Scenario vs Temporal Split', fontsize=16, y=1.02)

    splits = {
        'Scenario': 'results/scenario/{}_2layer/results.json',
        'Temporal': 'results/temporal/{}/results.json'
    }
    models = ['siglip', 'clip', 'dinov2']

    for row, (split_name, path_template) in enumerate(splits.items()):
        for col, model in enumerate(models):
            ax = axes[row, col]
            path = path_template.format(model)

            if not os.path.exists(path):
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(f'{MODEL_LABELS[model]} ({split_name})')
                continue

            y_true, y_score, y_pred = load_predictions(path)
            if y_true is None:
                ax.text(0.5, 0.5, 'No Predictions', ha='center', va='center')
                ax.set_title(f'{MODEL_LABELS[model]} ({split_name})')
                continue

            cm = confusion_matrix(y_true, y_pred)
            total = cm.sum()
            cm_pct = cm / total * 100

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Safe', 'Danger'],
                       yticklabels=['Safe', 'Danger'],
                       cbar=False)

            # Add percentages
            for i in range(2):
                for j in range(2):
                    ax.text(j + 0.5, i + 0.7, f'({cm_pct[i, j]:.1f}%)',
                           ha='center', va='center', fontsize=8, color='gray')

            acc = np.trace(cm) / total * 100
            ax.set_title(f'{MODEL_LABELS[model]} ({split_name})\nAcc: {acc:.1f}%')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_confusion_matrices.png')
    plt.close()
    print('  Figure 1: Confusion Matrices saved')


# =============================================
# Figure 2: ROC Curves
# =============================================
def generate_roc_curves():
    """Generate ROC curves for all models on scenario split."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    splits = {
        'Scenario Split': ('results/scenario/{}_2layer/results.json', axes[0]),
        'Temporal Split': ('results/temporal/{}/results.json', axes[1])
    }

    for split_name, (path_template, ax) in splits.items():
        for model in ['siglip', 'clip', 'dinov2']:
            path = path_template.format(model)
            if not os.path.exists(path):
                continue

            y_true, y_score, _ = load_predictions(path)
            if y_true is None:
                continue

            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=MODEL_COLORS[model], lw=2,
                   label=f'{MODEL_LABELS[model]} (AUC={roc_auc:.4f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {split_name}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_roc_curves.png')
    plt.close()
    print('  Figure 2: ROC Curves saved')


# =============================================
# Figure 3: Precision-Recall Curves
# =============================================
def generate_pr_curves():
    """Generate Precision-Recall curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    splits = {
        'Scenario Split': ('results/scenario/{}_2layer/results.json', axes[0]),
        'Temporal Split': ('results/temporal/{}/results.json', axes[1])
    }

    for split_name, (path_template, ax) in splits.items():
        for model in ['siglip', 'clip', 'dinov2']:
            path = path_template.format(model)
            if not os.path.exists(path):
                continue

            y_true, y_score, _ = load_predictions(path)
            if y_true is None:
                continue

            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            ax.plot(recall, precision, color=MODEL_COLORS[model], lw=2,
                   label=f'{MODEL_LABELS[model]} (AP={ap:.4f})')

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {split_name}')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_pr_curves.png')
    plt.close()
    print('  Figure 3: PR Curves saved')


# =============================================
# Figure 4: Error Confidence Distribution
# =============================================
def generate_error_distribution():
    """Generate error confidence distribution histograms."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, model in enumerate(['siglip', 'clip', 'dinov2']):
        ax = axes[idx]
        path = f'results/scenario/{model}_2layer/results.json'
        if not os.path.exists(path):
            continue

        y_true, y_score, y_pred = load_predictions(path)
        if y_true is None:
            continue

        fp_mask = (y_pred == 1) & (y_true == 0)
        fn_mask = (y_pred == 0) & (y_true == 1)
        tp_mask = (y_pred == 1) & (y_true == 1)
        tn_mask = (y_pred == 0) & (y_true == 0)

        bins = np.linspace(0, 1, 30)
        ax.hist(y_score[tp_mask], bins=bins, alpha=0.5, color='green',
               label=f'TP (n={tp_mask.sum()})', density=True)
        ax.hist(y_score[tn_mask], bins=bins, alpha=0.5, color='blue',
               label=f'TN (n={tn_mask.sum()})', density=True)
        ax.hist(y_score[fp_mask], bins=bins, alpha=0.7, color='red',
               label=f'FP (n={fp_mask.sum()})', density=True)
        ax.hist(y_score[fn_mask], bins=bins, alpha=0.7, color='orange',
               label=f'FN (n={fn_mask.sum()})', density=True)

        ax.set_xlabel('Prediction Score (P(Danger))')
        ax.set_ylabel('Density')
        ax.set_title(f'{MODEL_LABELS[model]} - Prediction Distribution')
        ax.legend(fontsize=9)
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_error_distribution.png')
    plt.close()
    print('  Figure 4: Error Distribution saved')


# =============================================
# Figure 5: Model Prediction Correlation Heatmap
# =============================================
def generate_correlation_heatmap():
    """Generate correlation heatmap of model predictions."""
    import pandas as pd

    model_preds = {}
    for model in ['siglip', 'clip', 'dinov2']:
        path = f'results/scenario/{model}_2layer/results.json'
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        preds = data.get('per_image_predictions', {})
        if preds:
            model_preds[MODEL_LABELS[model]] = preds

    if len(model_preds) < 2:
        print('  Figure 5: Skipped (not enough models)')
        return

    common_images = sorted(set.intersection(*[set(p.keys()) for p in model_preds.values()]))
    score_df = pd.DataFrame({
        model: [preds[img]['pred_score'] for img in common_images]
        for model, preds in model_preds.items()
    })

    corr = score_df.corr(method='pearson')

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt='.4f', cmap='RdYlBu_r',
               vmin=0.7, vmax=1.0, square=True, ax=ax,
               linewidths=0.5)
    ax.set_title(f'Model Prediction Score Correlation\n(Pearson, n={len(common_images)} images)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_correlation_heatmap.png')
    plt.close()

    # Also save correlation values
    with open(OUTPUT_DIR / 'correlation_values.json', 'w') as f:
        json.dump({
            'method': 'pearson',
            'n_images': len(common_images),
            'matrix': corr.to_dict()
        }, f, indent=2)

    print(f'  Figure 5: Correlation Heatmap saved (n={len(common_images)})')


# =============================================
# Figure 6: Data Scaling Curve
# =============================================
def generate_scaling_curve():
    """Generate scaling curve with error bars."""
    path = RESULTS_DIR / 'scaling_curve' / 'scaling_curve.json'
    if not path.exists():
        print('  Figure 6: Skipped (no scaling data)')
        return

    with open(path) as f:
        data = json.load(f)

    fractions = []
    means = []
    stds = []
    n_samples_list = []

    results = data.get('results', data)
    if isinstance(results, dict):
        for k, v in results.items():
            if isinstance(v, dict) and 'f1_mean' in v:
                fractions.append(v.get('fraction', 0))
                means.append(v['f1_mean'])
                stds.append(v.get('f1_std', 0))
                n_samples_list.append(v.get('n_train', 0))
    elif isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                fractions.append(item.get('fraction', 0))
                means.append(item.get('f1_mean', 0))
                stds.append(item.get('f1_std', 0))
                n_samples_list.append(item.get('n_train', 0))

    if not fractions:
        print('  Figure 6: Skipped (could not parse scaling data)')
        return

    fractions = np.array(fractions)
    means = np.array(means)
    stds = np.array(stds)

    # Convert to percentages if needed
    if means.max() < 1.0:
        means *= 100
        stds *= 100

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(fractions * 100, means, yerr=stds * 1.96,
               fmt='o-', color='#2196F3', capsize=5, capthick=2,
               linewidth=2, markersize=8, label='SigLIP 2-layer (5-seed)')

    ax.fill_between(fractions * 100, means - stds * 1.96, means + stds * 1.96,
                   alpha=0.15, color='#2196F3')

    for i, (frac, mean) in enumerate(zip(fractions, means)):
        n = n_samples_list[i] if i < len(n_samples_list) else int(frac * 10175)
        ax.annotate(f'{mean:.1f}%\n(n={n})',
                   (frac * 100, mean), textcoords="offset points",
                   xytext=(0, 15), ha='center', fontsize=9)

    ax.set_xlabel('Training Data Fraction (%)')
    ax.set_ylabel('Test F1 Score (%)')
    ax.set_title('Data Scaling Curve (SigLIP 2-layer, Scenario Split)')
    ax.set_xlim([5, 105])
    ax.set_ylim([75, 100])
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_scaling_curve.png')
    plt.close()
    print('  Figure 6: Scaling Curve saved')


# =============================================
# Figure 7: Temporal Per-Category Comparison
# =============================================
def generate_temporal_category_comparison():
    """Compare scenario vs temporal performance per category."""
    path = RESULTS_DIR / 'temporal_per_category' / 'temporal_per_category.json'
    if not path.exists():
        print('  Figure 7: Skipped (no temporal per-category data)')
        return

    with open(path) as f:
        data = json.load(f)

    categories = []
    scenario_f1 = []
    temporal_f1 = []
    cat_labels = {
        'A': 'Fall\nHazard',
        'B': 'Collision\nRisk',
        'C': 'Equipment\nHazard',
        'D': 'Environmental\nRisk',
        'E': 'Protective\nGear'
    }

    cat_data_dict = data.get('categories', data.get('results', data))
    for cat_key in ['A', 'B', 'C', 'D', 'E']:
        cat_data = cat_data_dict.get(cat_key, {})
        if not cat_data:
            continue
        categories.append(cat_labels.get(cat_key, cat_key))
        scenario_f1.append(cat_data.get('scenario_f1_mean', 0))
        temporal_f1.append(cat_data.get('temporal_f1_mean', 0))

    if not categories:
        print('  Figure 7: Skipped (could not parse category data)')
        return

    # Convert if needed
    scenario_f1 = np.array(scenario_f1)
    temporal_f1 = np.array(temporal_f1)
    if scenario_f1.max() <= 1.0:
        scenario_f1 *= 100
        temporal_f1 *= 100

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, scenario_f1, width, label='Scenario (Random Split)',
                   color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + width/2, temporal_f1, width, label='Temporal Split',
                   color='#FF5722', alpha=0.8)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

    # Add delta annotations
    for i, (s, t) in enumerate(zip(scenario_f1, temporal_f1)):
        delta = t - s
        color = 'green' if delta > 0 else 'red'
        ax.annotate(f'{delta:+.1f}%p', (i, max(s, t) + 1.5),
                   ha='center', fontsize=10, fontweight='bold', color=color)

    # Add overall binary result line
    ax.axhline(y=66.11, color='red', linestyle='--', alpha=0.5)
    ax.text(len(categories) - 0.5, 67, 'Overall Binary\nTemporal F1=66.1%',
           fontsize=8, color='red', ha='right')

    ax.set_xlabel('Safety Dimension')
    ax.set_ylabel('F1 Score (%)')
    ax.set_title('Per-Category Performance: Scenario vs Temporal Split\n(SigLIP 2-layer, 5-seed mean)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim([60, 105])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_temporal_category.png')
    plt.close()
    print('  Figure 7: Temporal Per-Category saved')


# =============================================
# Figure 8: Ensemble Ablation
# =============================================
def generate_ensemble_ablation():
    """Generate ensemble ablation comparison."""
    path = RESULTS_DIR / 'ensemble_ablation' / 'ensemble_ablation.json'
    if not path.exists():
        print('  Figure 8: Skipped (no ensemble ablation data)')
        return

    with open(path) as f:
        data = json.load(f)

    configs = []
    f1_means = []
    f1_stds = []

    results = data.get('results', data)
    for k, v in results.items():
        if isinstance(v, dict) and 'f1_mean' in v:
            configs.append(k)
            f1_means.append(v['f1_mean'])
            f1_stds.append(v.get('f1_std', 0))

    if not configs:
        print('  Figure 8: Skipped (no ensemble results)')
        return

    # Sort by f1
    order = np.argsort(f1_means)[::-1]
    configs = [configs[i] for i in order]
    f1_means = [f1_means[i] for i in order]
    f1_stds = [f1_stds[i] for i in order]

    colors = []
    for c in configs:
        n_models = c.count('+') + 1
        if n_models == 1:
            colors.append('#90CAF9')
        elif n_models == 2:
            colors.append('#42A5F5')
        else:
            colors.append('#1565C0')

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(configs))

    bars = ax.barh(y_pos, f1_means, xerr=[s * 1.96 for s in f1_stds],
                   color=colors, capsize=4, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.replace('+', ' + ').upper() for c in configs])
    ax.set_xlabel('F1 Score (%)')
    ax.set_title('Ensemble Ablation (Scenario Split, 5-seed)')
    ax.set_xlim([88, 99])
    ax.grid(True, alpha=0.3, axis='x')

    for i, (mean, std) in enumerate(zip(f1_means, f1_stds)):
        ax.text(mean + 0.1, i, f'{mean:.2f}%', va='center', fontsize=10)

    # Legend for model count
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#90CAF9', label='Single Model'),
        Patch(facecolor='#42A5F5', label='2-Model Ensemble'),
        Patch(facecolor='#1565C0', label='3-Model Ensemble')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig8_ensemble_ablation.png')
    plt.close()
    print('  Figure 8: Ensemble Ablation saved')


# =============================================
# Figure 9: Probe Depth Ablation
# =============================================
def generate_probe_depth_comparison():
    """Compare probe depths across models."""
    summary_path = RESULTS_DIR / 'multiseed' / 'multiseed_summary.json'
    if not summary_path.exists():
        print('  Figure 9: Skipped (no multiseed summary)')
        return

    with open(summary_path) as f:
        data = json.load(f)

    models = ['siglip', 'clip', 'dinov2']
    probes = ['linear', '1layer', '2layer']
    probe_labels = {'linear': 'Linear', '1layer': '1-Layer MLP', '2layer': '2-Layer MLP'}

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(probes))
    width = 0.25

    for i, model in enumerate(models):
        means = []
        stds = []
        for probe in probes:
            key = f'scenario_{model}_{probe}'
            if key in data:
                means.append(data[key]['metrics']['f1']['mean'])
                stds.append(data[key]['metrics']['f1']['std'])
            else:
                means.append(0)
                stds.append(0)

        bars = ax.bar(x + i * width, means, width, yerr=[s * 1.96 for s in stds],
                     label=MODEL_LABELS[model], color=MODEL_COLORS[model],
                     alpha=0.8, capsize=4)

        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Probe Architecture')
    ax.set_ylabel('F1 Score (%)')
    ax.set_title('Probe Depth Ablation (Scenario Split, 5-seed)')
    ax.set_xticks(x + width)
    ax.set_xticklabels([probe_labels[p] for p in probes])
    ax.set_ylim([65, 100])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig9_probe_depth.png')
    plt.close()
    print('  Figure 9: Probe Depth Ablation saved')


# =============================================
# Main
# =============================================
def main():
    print('=' * 60)
    print('Generating Paper Figures')
    print('=' * 60)
    print(f'Output directory: {OUTPUT_DIR}')
    print()

    generate_confusion_matrices()
    generate_roc_curves()
    generate_pr_curves()
    generate_error_distribution()
    generate_correlation_heatmap()
    generate_scaling_curve()
    generate_temporal_category_comparison()
    generate_ensemble_ablation()
    generate_probe_depth_comparison()

    print()
    print('=' * 60)
    files = list(OUTPUT_DIR.glob('*.png'))
    print(f'Generated {len(files)} figures:')
    for f in sorted(files):
        size_kb = f.stat().st_size / 1024
        print(f'  {f.name} ({size_kb:.0f} KB)')
    print('=' * 60)


if __name__ == '__main__':
    main()
