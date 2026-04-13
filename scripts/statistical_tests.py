#!/usr/bin/env python3
"""
통계적 유의성 검정 스크립트

Multi-seed 실험 결과를 기반으로 모델 간 성능 차이의 통계적 유의성을 검정합니다.
- Paired t-test: 동일 시드에서의 모델 쌍 간 F1 비교
- McNemar's test: Per-image prediction 불일치 검정
- 95% 신뢰구간 (bootstrap)
- Cohen's d 효과 크기
"""

import json
import os
import sys
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats


def load_multiseed_data(results_dir: str) -> dict:
    """Load per-seed F1 scores from multiseed results."""
    seed_data = {}
    multiseed_dir = os.path.join(results_dir, "multiseed")

    for config_dir in os.listdir(multiseed_dir):
        if config_dir.endswith(".json"):
            continue
        path = os.path.join(multiseed_dir, config_dir, "results.json")
        if not os.path.exists(path):
            continue

        with open(path) as f:
            d = json.load(f)

        metrics = d.get("test_metrics", {})
        f1 = metrics.get("f1", metrics.get("f1_score"))
        if f1 is None:
            continue

        parts = config_dir.rsplit("_seed", 1)
        if len(parts) == 2:
            config_name = parts[0]
            seed = int(parts[1])
            if config_name not in seed_data:
                seed_data[config_name] = {}
            seed_data[config_name][seed] = f1

    return seed_data


def load_per_image_predictions(results_dir: str) -> dict:
    """Load per-image predictions from scenario results."""
    predictions = {}
    scenario_dir = os.path.join(results_dir, "scenario")

    if not os.path.exists(scenario_dir):
        return predictions

    for model_dir in os.listdir(scenario_dir):
        path = os.path.join(scenario_dir, model_dir, "results.json")
        if not os.path.exists(path):
            continue

        with open(path) as f:
            d = json.load(f)

        preds = d.get("per_image_predictions", {})
        if preds:
            predictions[model_dir] = preds

    return predictions


def paired_ttest(seed_data: dict, alpha: float = 0.05) -> list:
    """Perform paired t-test between all model pairs."""
    results = []
    configs = sorted(seed_data.keys())

    for a, b in combinations(configs, 2):
        seeds_a = seed_data[a]
        seeds_b = seed_data[b]
        common_seeds = sorted(set(seeds_a.keys()) & set(seeds_b.keys()))

        if len(common_seeds) < 3:
            continue

        vals_a = np.array([seeds_a[s] for s in common_seeds])
        vals_b = np.array([seeds_b[s] for s in common_seeds])

        t_stat, p_value = stats.ttest_rel(vals_a, vals_b)

        # Cohen's d for paired samples
        diff = vals_a - vals_b
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

        results.append({
            "model_a": a,
            "model_b": b,
            "n_seeds": len(common_seeds),
            "mean_a": float(np.mean(vals_a)),
            "mean_b": float(np.mean(vals_b)),
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff, ddof=1)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "cohens_d": float(cohens_d),
            "effect_size": (
                "large" if abs(cohens_d) >= 0.8 else
                "medium" if abs(cohens_d) >= 0.5 else
                "small" if abs(cohens_d) >= 0.2 else
                "negligible"
            )
        })

    # Apply multiple comparison corrections
    n_comparisons = len(results)
    if n_comparisons > 0:
        p_values = np.array([r["p_value"] for r in results])

        # Bonferroni correction
        bonferroni_p = np.minimum(p_values * n_comparisons, 1.0)

        # Holm-Bonferroni (step-down) correction
        sorted_indices = np.argsort(p_values)
        holm_p = np.zeros(n_comparisons)
        for rank, idx in enumerate(sorted_indices):
            holm_p[idx] = p_values[idx] * (n_comparisons - rank)
        # Enforce monotonicity
        for i in range(1, len(sorted_indices)):
            idx = sorted_indices[i]
            prev_idx = sorted_indices[i - 1]
            holm_p[idx] = max(holm_p[idx], holm_p[prev_idx])
        holm_p = np.minimum(holm_p, 1.0)

        for i, r in enumerate(results):
            r["bonferroni_p"] = float(bonferroni_p[i])
            r["holm_bonferroni_p"] = float(holm_p[i])
            r["bonferroni_significant"] = bonferroni_p[i] < alpha
            r["holm_significant"] = holm_p[i] < alpha
            r["n_comparisons"] = n_comparisons

    return sorted(results, key=lambda x: x["p_value"])


def bootstrap_ci(seed_data: dict, n_bootstrap: int = 10000, ci: float = 0.95) -> dict:
    """Compute bootstrap confidence intervals for each configuration."""
    results = {}

    for config, seeds in seed_data.items():
        vals = np.array(list(seeds.values()))
        n = len(vals)

        if n < 2:
            continue

        rng = np.random.default_rng(42)
        boot_means = np.array([
            np.mean(rng.choice(vals, size=n, replace=True))
            for _ in range(n_bootstrap)
        ])

        lower = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
        upper = float(np.percentile(boot_means, (1 + ci) / 2 * 100))

        results[config] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "ci_lower": lower,
            "ci_upper": upper,
            "ci_level": ci,
            "n_seeds": n,
            "n_bootstrap": n_bootstrap
        }

    return results


def mcnemar_test(predictions: dict, alpha: float = 0.05) -> list:
    """Perform McNemar's test on per-image predictions between model pairs."""
    results = []
    models = sorted(predictions.keys())

    for a, b in combinations(models, 2):
        preds_a = predictions[a]
        preds_b = predictions[b]
        common_images = sorted(set(preds_a.keys()) & set(preds_b.keys()))

        if len(common_images) < 10:
            continue

        # Count disagreements
        b_correct_a_wrong = 0  # b correct, a wrong
        a_correct_b_wrong = 0  # a correct, b wrong

        for img in common_images:
            a_correct = preds_a[img]["correct"]
            b_correct = preds_b[img]["correct"]
            if a_correct and not b_correct:
                a_correct_b_wrong += 1
            elif b_correct and not a_correct:
                b_correct_a_wrong += 1

        n = b_correct_a_wrong + a_correct_b_wrong
        if n == 0:
            continue

        # McNemar's test with continuity correction
        chi2 = (abs(b_correct_a_wrong - a_correct_b_wrong) - 1) ** 2 / n if n > 0 else 0
        p_value = float(1 - stats.chi2.cdf(chi2, df=1))

        results.append({
            "model_a": a,
            "model_b": b,
            "n_images": len(common_images),
            "a_correct_b_wrong": a_correct_b_wrong,
            "b_correct_a_wrong": b_correct_a_wrong,
            "chi2_statistic": float(chi2),
            "p_value": p_value,
            "significant": p_value < alpha,
            "better_model": a if a_correct_b_wrong > b_correct_a_wrong else b
        })

    return sorted(results, key=lambda x: x["p_value"])


def format_results(ttest_results, ci_results, mcnemar_results) -> str:
    """Format results as readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("통계적 유의성 검정 결과")
    lines.append("=" * 80)

    # Bootstrap CI
    lines.append("\n## 1. Bootstrap 95% 신뢰구간 (F1 Score)")
    lines.append("-" * 60)
    lines.append(f"{'Configuration':<40} {'Mean':>8} {'95% CI':>20}")
    lines.append("-" * 60)
    for config in sorted(ci_results.keys()):
        ci = ci_results[config]
        lines.append(
            f"{config:<40} {ci['mean']:>8.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
        )

    # Paired t-test
    lines.append("\n## 2. Paired t-test (모델 쌍 간 F1 비교)")
    lines.append("-" * 80)

    n_comp = ttest_results[0].get("n_comparisons", len(ttest_results)) if ttest_results else 0
    lines.append(f"\n총 비교 수: {n_comp}쌍")
    lines.append(f"Bonferroni 보정 α: {0.05/n_comp:.6f}" if n_comp > 0 else "")

    bonf_sig = [r for r in ttest_results if r.get("bonferroni_significant", False)]
    holm_sig = [r for r in ttest_results if r.get("holm_significant", False)]
    significant = [r for r in ttest_results if r["significant"]]
    not_significant = [r for r in ttest_results if not r["significant"]]

    lines.append(f"\n보정 전 유의 (p < 0.05): {len(significant)}쌍")
    lines.append(f"Bonferroni 보정 후 유의: {len(bonf_sig)}쌍")
    lines.append(f"Holm-Bonferroni 보정 후 유의: {len(holm_sig)}쌍")

    lines.append(f"\n### 주요 비교 (Bonferroni 보정 적용)")
    for r in significant[:20]:
        bonf_mark = "**" if r.get("bonferroni_significant") else ""
        lines.append(
            f"  {r['model_a']} vs {r['model_b']}: "
            f"diff={r['mean_diff']:.4f}, p={r['p_value']:.6f}, "
            f"d={r['cohens_d']:.2f} ({r['effect_size']})"
        )

    lines.append(f"\n비유의 (p >= 0.05): {len(not_significant)}쌍")
    for r in not_significant[:10]:
        lines.append(
            f"  {r['model_a']} vs {r['model_b']}: "
            f"diff={r['mean_diff']:.4f}, p={r['p_value']:.6f}"
        )

    # McNemar's test
    if mcnemar_results:
        lines.append("\n## 3. McNemar's Test (Per-image 예측 불일치)")
        lines.append("-" * 80)
        for r in mcnemar_results:
            sig_mark = "*" if r["significant"] else ""
            lines.append(
                f"  {r['model_a']} vs {r['model_b']}: "
                f"χ²={r['chi2_statistic']:.2f}, p={r['p_value']:.6f}{sig_mark}, "
                f"better={r['better_model']}"
            )

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="통계적 유의성 검정")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Results directory")
    parser.add_argument("--output", type=str, default="results/statistical_tests",
                       help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Significance level")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading multi-seed data...")
    seed_data = load_multiseed_data(args.results_dir)
    print(f"  Found {len(seed_data)} configurations")

    print("Loading per-image predictions...")
    predictions = load_per_image_predictions(args.results_dir)
    print(f"  Found {len(predictions)} models with predictions")

    print("\nRunning paired t-tests...")
    ttest_results = paired_ttest(seed_data, alpha=args.alpha)
    print(f"  {len(ttest_results)} pairwise comparisons")

    print("Computing bootstrap confidence intervals...")
    ci_results = bootstrap_ci(seed_data)
    print(f"  {len(ci_results)} configurations")

    print("Running McNemar's tests...")
    mcnemar_results = mcnemar_test(predictions, alpha=args.alpha)
    print(f"  {len(mcnemar_results)} pairwise comparisons")

    # Save JSON results
    all_results = {
        "alpha": args.alpha,
        "paired_ttest": ttest_results,
        "bootstrap_ci": ci_results,
        "mcnemar_test": mcnemar_results
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    json_path = os.path.join(args.output, "statistical_tests.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\nJSON results saved: {json_path}")

    # Save formatted text report
    report = format_results(ttest_results, ci_results, mcnemar_results)
    txt_path = os.path.join(args.output, "statistical_tests_report.txt")
    with open(txt_path, "w") as f:
        f.write(report)
    print(f"Text report saved: {txt_path}")

    # Print summary
    print(report)


if __name__ == "__main__":
    main()
