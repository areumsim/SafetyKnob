#!/usr/bin/env python3
"""
Multi-seed Experiment Runner

Runs all probe experiments with multiple seeds and computes mean±std.
Uses cached embeddings for fast execution (~3s per experiment).

Usage:
    python scripts/run_multiseed.py --output results/multiseed
    python scripts/run_multiseed.py --seeds 42,123,456,789,2024 --output results/multiseed
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np


SEEDS = [42, 123, 456, 789, 2024]

# All experiment configurations: (split, model, probe_depth, category)
EXPERIMENTS = []

# RQ1: Probe depth ablation (scenario_v2 split, 3 models × 3 depths = 9)
for model in ['siglip', 'clip', 'dinov2']:
    for depth in ['linear', '1layer', '2layer']:
        EXPERIMENTS.append(('scenario_v2', model, depth, None))

# RQ2: Temporal split (3 models × 2-layer = 3)
for model in ['siglip', 'clip', 'dinov2']:
    EXPERIMENTS.append(('temporal', model, '2layer', None))

# RQ3: Independent dimension classifiers (scenario_v2, siglip, 5 categories = 5)
for cat in ['A', 'B', 'C', 'D', 'E']:
    EXPERIMENTS.append(('scenario_v2', 'siglip', '2layer', cat))


def run_single_experiment(split, model, probe_depth, category, seed, base_output, embeddings_base):
    """Run a single experiment and return results."""
    embeddings_dir = Path(embeddings_base) / split / model

    # Build output path
    cat_suffix = f'_cat{category}' if category else ''
    output_dir = Path(base_output) / f'{split}_{model}_{probe_depth}{cat_suffix}_seed{seed}'

    cmd = [
        sys.executable, 'experiments/train_from_embeddings.py',
        '--model', model,
        '--embeddings-dir', str(embeddings_dir),
        '--probe-depth', probe_depth,
        '--output', str(output_dir),
        '--seed', str(seed),
        '--epochs', '50',
        '--save-predictions',
    ]

    if category:
        cmd.extend(['--category', category])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-200:]}")
        return None

    # Load results
    results_file = output_dir / 'results.json'
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None


def compute_stats(results_list, metric_path='test_metrics.f1'):
    """Compute mean and std for a metric across seeds."""
    values = []
    for r in results_list:
        if r is None:
            continue
        # Navigate nested path
        obj = r
        for key in metric_path.split('.'):
            obj = obj[key]
        values.append(float(obj))

    if not values:
        return None, None
    return np.mean(values), np.std(values)


def main():
    parser = argparse.ArgumentParser(description='Run all experiments with multiple seeds')
    parser.add_argument('--seeds', type=str, default=','.join(map(str, SEEDS)),
                       help='Comma-separated seed list')
    parser.add_argument('--output', type=str, default='results/multiseed',
                       help='Base output directory')
    parser.add_argument('--embeddings-base', type=str, default='embeddings',
                       help='Base embeddings directory')
    parser.add_argument('--experiments', type=str, default='all',
                       choices=['all', 'rq1', 'rq2', 'rq3'],
                       help='Which experiment set to run')

    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    # Filter experiments
    if args.experiments == 'rq1':
        experiments = [e for e in EXPERIMENTS if e[0] == 'scenario' and e[3] is None]
    elif args.experiments == 'rq2':
        experiments = [e for e in EXPERIMENTS if e[0] == 'temporal']
    elif args.experiments == 'rq3':
        experiments = [e for e in EXPERIMENTS if e[3] is not None]
    else:
        experiments = EXPERIMENTS

    total = len(experiments) * len(seeds)
    print(f"{'='*70}")
    print(f"MULTI-SEED EXPERIMENT RUNNER")
    print(f"{'='*70}")
    print(f"Seeds: {seeds}")
    print(f"Experiments: {len(experiments)}")
    print(f"Total runs: {total}")
    print(f"Output: {output_base}")
    print()

    # Run all experiments
    all_results = defaultdict(list)  # key -> list of results per seed
    start_time = time.time()
    completed = 0

    for split, model, depth, category in experiments:
        cat_str = f'_cat{category}' if category else ''
        key = f'{split}_{model}_{depth}{cat_str}'
        cat_names = {'A': 'Fall', 'B': 'Collision', 'C': 'Equipment',
                     'D': 'Environment', 'E': 'PPE'}

        desc = f'{model}/{depth}'
        if category:
            desc += f'/{cat_names[category]}'
        desc += f' ({split})'

        for seed in seeds:
            completed += 1
            print(f"[{completed}/{total}] {desc} seed={seed} ... ", end='', flush=True)

            result = run_single_experiment(
                split, model, depth, category, seed,
                output_base, args.embeddings_base
            )

            if result:
                f1 = result['test_metrics']['f1'] * 100
                print(f"F1={f1:.2f}%")
                all_results[key].append(result)
            else:
                print("FAILED")

    elapsed = time.time() - start_time

    # Compute summary statistics
    print(f"\n{'='*70}")
    print(f"SUMMARY (mean ± std over {len(seeds)} seeds)")
    print(f"{'='*70}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)\n")

    summary = {}
    cat_names = {'A': 'Fall Hazard', 'B': 'Collision Risk', 'C': 'Equipment Hazard',
                 'D': 'Environmental Risk', 'E': 'Protective Gear'}

    # RQ1: Probe depth ablation
    print("--- RQ1: Probe Depth Ablation (Scenario) ---")
    print(f"{'Model':<10} {'Linear F1':>15} {'1-Layer F1':>15} {'2-Layer F1':>15} {'Gap':>10}")
    print("-" * 70)

    for model in ['siglip', 'clip', 'dinov2']:
        row = {}
        for depth in ['linear', '1layer', '2layer']:
            key = f'scenario_v2_{model}_{depth}'
            if key in all_results:
                mean, std = compute_stats(all_results[key])
                if mean is not None:
                    row[depth] = (mean * 100, std * 100)
                    summary[key] = {'f1_mean': mean * 100, 'f1_std': std * 100}

        if 'linear' in row and '2layer' in row:
            gap = row['2layer'][0] - row['linear'][0]
            print(f"{model:<10} {row.get('linear', (0,0))[0]:>8.2f}±{row.get('linear', (0,0))[1]:.2f}%"
                  f" {row.get('1layer', (0,0))[0]:>8.2f}±{row.get('1layer', (0,0))[1]:.2f}%"
                  f" {row.get('2layer', (0,0))[0]:>8.2f}±{row.get('2layer', (0,0))[1]:.2f}%"
                  f" {gap:>8.2f}%p")

    # RQ2: Temporal
    print(f"\n--- RQ2: Temporal Distribution Shift ---")
    print(f"{'Model':<10} {'Scenario F1':>20} {'Temporal F1':>20} {'Delta':>10}")
    print("-" * 65)

    for model in ['siglip', 'clip', 'dinov2']:
        scenario_key = f'scenario_v2_{model}_2layer'
        temporal_key = f'temporal_{model}_2layer'

        s_mean, s_std = compute_stats(all_results.get(scenario_key, []))
        t_mean, t_std = compute_stats(all_results.get(temporal_key, []))

        if s_mean and t_mean:
            delta = (t_mean - s_mean) * 100
            print(f"{model:<10} {s_mean*100:>10.2f}±{s_std*100:.2f}%"
                  f" {t_mean*100:>10.2f}±{t_std*100:.2f}%"
                  f" {delta:>+8.2f}%p")
            summary[temporal_key] = {'f1_mean': t_mean * 100, 'f1_std': t_std * 100}

    # RQ3: Independent dimensions
    print(f"\n--- RQ3: Independent Category Classifiers (Scenario, SigLIP) ---")
    print(f"{'Category':<20} {'F1 (mean±std)':>20}")
    print("-" * 45)

    for cat in ['A', 'B', 'C', 'D', 'E']:
        key = f'scenario_v2_siglip_2layer_cat{cat}'
        if key in all_results:
            mean, std = compute_stats(all_results[key])
            if mean is not None:
                print(f"{cat_names[cat]:<20} {mean*100:>10.2f}±{std*100:.2f}%")
                summary[key] = {'f1_mean': mean * 100, 'f1_std': std * 100}

    # Save detailed summary
    # Also compute per-metric stats
    detailed_summary = {}
    for key, results in all_results.items():
        metrics = {}
        for metric in ['f1', 'accuracy', 'precision', 'recall', 'auc_roc']:
            mean, std = compute_stats(results, f'test_metrics.{metric}')
            if mean is not None:
                metrics[metric] = {'mean': float(mean * 100), 'std': float(std * 100)}

        detailed_summary[key] = {
            'n_seeds': len(results),
            'seeds': seeds[:len(results)],
            'metrics': metrics,
        }

    summary_file = output_base / 'multiseed_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(detailed_summary, f, indent=2)
    print(f"\nDetailed summary saved to {summary_file}")


if __name__ == '__main__':
    main()
