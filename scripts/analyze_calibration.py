#!/usr/bin/env python3
"""
Confidence Calibration Analysis for Safety-Critical Deployment

Computes:
1. ECE (Expected Calibration Error) and MCE (Maximum Calibration Error)
2. Reliability diagrams
3. Temperature scaling (post-hoc calibration)
4. Safety-specific analysis: FN confidence distribution

Usage:
    python3 scripts/analyze_calibration.py \
        --results-dir results/scenario_v2 \
        --output results/calibration
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def compute_calibration_metrics(y_true, y_prob, n_bins=15):
    """Compute ECE, MCE, and per-bin calibration data."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bins_data = []

    for lower, upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > lower) & (y_prob <= upper)
        n_in_bin = in_bin.sum()

        if n_in_bin == 0:
            bins_data.append({
                'lower': float(lower), 'upper': float(upper),
                'count': 0, 'avg_confidence': 0, 'avg_accuracy': 0, 'gap': 0
            })
            continue

        bin_accuracy = y_true[in_bin].mean()
        bin_confidence = y_prob[in_bin].mean()
        gap = abs(bin_accuracy - bin_confidence)

        ece += gap * (n_in_bin / len(y_true))
        mce = max(mce, gap)

        bins_data.append({
            'lower': float(lower), 'upper': float(upper),
            'count': int(n_in_bin),
            'avg_confidence': float(bin_confidence),
            'avg_accuracy': float(bin_accuracy),
            'gap': float(gap)
        })

    return float(ece), float(mce), bins_data


def temperature_scaling(y_true, logits, lr=0.01, max_iter=1000):
    """Learn optimal temperature for calibration via NLL minimization."""
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    criterion = nn.BCELoss()

    y_true_t = torch.tensor(y_true, dtype=torch.float32)
    logits_t = torch.tensor(logits, dtype=torch.float32)

    def closure():
        optimizer.zero_grad()
        scaled = torch.sigmoid(logits_t / temperature)
        loss = criterion(scaled, y_true_t)
        loss.backward()
        return loss

    optimizer.step(closure)

    optimal_temp = temperature.item()
    calibrated_probs = torch.sigmoid(logits_t / optimal_temp).detach().numpy()

    return optimal_temp, calibrated_probs


def analyze_safety_errors(y_true, y_prob, threshold=0.5):
    """Analyze confidence distribution of safety-critical errors."""
    y_pred = (y_prob > threshold).astype(int)

    # FN: true=0 (unsafe), pred=1 (safe) -- MOST DANGEROUS
    fn_mask = (y_true == 0) & (y_pred == 1)
    # FP: true=1 (safe), pred=0 (unsafe) -- false alarm
    fp_mask = (y_true == 1) & (y_pred == 0)
    # TP: true=0 (unsafe), pred=0 (unsafe) -- correct detection
    tp_mask = (y_true == 0) & (y_pred == 0)
    # TN: true=1 (safe), pred=1 (safe)
    tn_mask = (y_true == 1) & (y_pred == 1)

    results = {
        'false_negatives': {
            'count': int(fn_mask.sum()),
            'description': 'Danger missed as safe (CRITICAL)',
            'confidence_mean': float(y_prob[fn_mask].mean()) if fn_mask.any() else None,
            'confidence_std': float(y_prob[fn_mask].std()) if fn_mask.any() else None,
            'confidence_min': float(y_prob[fn_mask].min()) if fn_mask.any() else None,
            'confidence_max': float(y_prob[fn_mask].max()) if fn_mask.any() else None,
        },
        'false_positives': {
            'count': int(fp_mask.sum()),
            'description': 'Safe flagged as danger (false alarm)',
            'confidence_mean': float(y_prob[fp_mask].mean()) if fp_mask.any() else None,
            'confidence_std': float(y_prob[fp_mask].std()) if fp_mask.any() else None,
        },
        'true_positives': {
            'count': int(tp_mask.sum()),
            'confidence_mean': float(y_prob[tp_mask].mean()) if tp_mask.any() else None,
        },
        'true_negatives': {
            'count': int(tn_mask.sum()),
            'confidence_mean': float(y_prob[tn_mask].mean()) if tn_mask.any() else None,
        },
    }

    # High-recall operating points
    recall_targets = [0.95, 0.97, 0.99]
    operating_points = []

    # For unsafe detection: we want high recall on class 0 (unsafe)
    # Lower threshold = more predictions of "unsafe"
    unsafe_mask = y_true == 0
    n_unsafe = unsafe_mask.sum()

    for target_recall in recall_targets:
        for thresh in np.arange(0.01, 1.0, 0.01):
            preds = (y_prob > thresh).astype(int)
            # Recall for unsafe class: correctly predicted as unsafe (pred=0, true=0)
            if n_unsafe > 0:
                recall_unsafe = ((preds == 0) & (y_true == 0)).sum() / n_unsafe
            else:
                recall_unsafe = 0

            if recall_unsafe >= target_recall:
                precision_unsafe = ((preds == 0) & (y_true == 0)).sum() / max((preds == 0).sum(), 1)
                f1 = 2 * precision_unsafe * recall_unsafe / max(precision_unsafe + recall_unsafe, 1e-8)
                operating_points.append({
                    'target_recall': float(target_recall),
                    'threshold': float(thresh),
                    'actual_recall': float(recall_unsafe),
                    'precision': float(precision_unsafe),
                    'f1': float(f1),
                    'false_alarm_rate': float(((preds == 0) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)),
                })
                break

    results['high_recall_operating_points'] = operating_points
    return results


def plot_reliability_diagram(bins_data, ece, mce, title, output_path):
    """Plot reliability diagram."""
    if not HAS_MPL:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Reliability diagram
    confidences = [b['avg_confidence'] for b in bins_data if b['count'] > 0]
    accuracies = [b['avg_accuracy'] for b in bins_data if b['count'] > 0]
    counts = [b['count'] for b in bins_data if b['count'] > 0]

    ax1.bar(confidences, accuracies, width=0.06, alpha=0.7, label='Actual', edgecolor='black')
    ax1.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(f'{title}\nECE={ece:.4f}, MCE={mce:.4f}')
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Confidence histogram
    bin_edges = [b['lower'] for b in bins_data] + [bins_data[-1]['upper']]
    bin_counts = [b['count'] for b in bins_data]
    ax2.bar([(b['lower'] + b['upper'])/2 for b in bins_data], bin_counts,
            width=0.06, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Calibration analysis')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory with model results containing per_image_predictions')
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir)

    # Find all results.json with per_image_predictions
    result_files = list(results_dir.rglob('results.json'))
    if not result_files:
        print(f"No results.json found in {results_dir}")
        sys.exit(1)

    all_calibration = {}

    for rfile in sorted(result_files):
        with open(rfile) as f:
            data = json.load(f)

        if 'per_image_predictions' not in data:
            print(f"  Skipping {rfile} (no per_image_predictions)")
            continue

        model_name = data.get('model', 'unknown')
        probe = data.get('probe_depth', '2layer')
        category = data.get('category', None)
        key = f"{model_name}_{probe}" + (f"_cat{category}" if category else "")

        preds = data['per_image_predictions']
        y_true = np.array([v['true_label'] for v in preds.values()])
        y_prob = np.array([v['pred_score'] for v in preds.values()])

        print(f"\n{'='*60}")
        print(f"Analyzing: {key} ({len(y_true)} samples)")
        print(f"{'='*60}")

        # 1. Calibration metrics
        ece, mce, bins = compute_calibration_metrics(y_true, y_prob)
        print(f"  ECE: {ece:.4f}")
        print(f"  MCE: {mce:.4f}")

        # 2. Temperature scaling
        # Convert probabilities to logits for temp scaling
        y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))

        try:
            opt_temp, calibrated_probs = temperature_scaling(y_true, logits)
            ece_cal, mce_cal, bins_cal = compute_calibration_metrics(y_true, calibrated_probs)
            print(f"  Optimal Temperature: {opt_temp:.3f}")
            print(f"  ECE after calibration: {ece_cal:.4f} (Δ={ece-ece_cal:+.4f})")
            print(f"  MCE after calibration: {mce_cal:.4f}")
        except Exception as e:
            print(f"  Temperature scaling failed: {e}")
            opt_temp, ece_cal, mce_cal, bins_cal = None, None, None, None

        # 3. Safety error analysis
        safety = analyze_safety_errors(y_true, y_prob)
        fn = safety['false_negatives']
        print(f"\n  Safety Error Analysis:")
        print(f"    False Negatives (danger missed): {fn['count']}")
        if fn['confidence_mean'] is not None:
            print(f"      Confidence: {fn['confidence_mean']:.3f} ± {fn['confidence_std']:.3f}")
        print(f"    False Positives (false alarm): {safety['false_positives']['count']}")

        if safety['high_recall_operating_points']:
            print(f"\n  High-Recall Operating Points:")
            for op in safety['high_recall_operating_points']:
                print(f"    Recall≥{op['target_recall']:.0%}: "
                      f"thresh={op['threshold']:.2f}, "
                      f"precision={op['precision']:.3f}, "
                      f"f1={op['f1']:.3f}, "
                      f"false_alarm={op['false_alarm_rate']:.3f}")

        # 4. Plot reliability diagram
        if HAS_MPL:
            plot_reliability_diagram(bins, ece, mce, f"{key} (before calibration)",
                                   output_dir / f"reliability_{key}.png")
            if bins_cal:
                plot_reliability_diagram(bins_cal, ece_cal, mce_cal,
                                       f"{key} (after temp scaling, T={opt_temp:.2f})",
                                       output_dir / f"reliability_{key}_calibrated.png")

        all_calibration[key] = {
            'model': model_name,
            'probe_depth': probe,
            'category': category,
            'n_samples': len(y_true),
            'ece': ece,
            'mce': mce,
            'optimal_temperature': opt_temp,
            'ece_after_calibration': ece_cal,
            'mce_after_calibration': mce_cal,
            'safety_analysis': safety,
            'source_file': str(rfile),
        }

    # Save all results
    output_file = output_dir / 'calibration_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(all_calibration, f, indent=2)
    print(f"\n\nAll calibration results saved to {output_file}")


if __name__ == '__main__':
    main()
