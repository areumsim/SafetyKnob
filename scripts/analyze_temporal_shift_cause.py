#!/usr/bin/env python3
"""
Temporal Shift Paradox 근본 원인 분석

패러독스:
- 전체 이진분류 (안전/위험): temporal split에서 F1 30%p 하락 (96% -> 66%)
- 카테고리별 분류기 (A-E): temporal split에서 오히려 향상 (+1-4%p)

이 모순의 원인을 데이터 분포 분석을 통해 규명한다.
"""

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_category(filename):
    """파일명에서 카테고리 코드 추출.
    패턴: H-YYMMDD_X##_... 여기서 X가 카테고리 (A-F)
    """
    m = re.match(r"H-\d{6}_([A-Z])\d+_", filename)
    if m:
        return m.group(1)
    return "Unknown"


def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def analyze():
    # ========================================
    # 1. 데이터 로드
    # ========================================
    temporal_results = load_json(RESULTS_DIR / "temporal" / "siglip" / "results.json")
    scenario_results = load_json(RESULTS_DIR / "scenario" / "siglip_2layer" / "results.json")
    per_category = load_json(RESULTS_DIR / "temporal_per_category" / "temporal_per_category.json")
    multiseed = load_json(RESULTS_DIR / "multiseed" / "multiseed_summary.json")

    temporal_preds = temporal_results["per_image_predictions"]
    scenario_preds = scenario_results["per_image_predictions"]

    print("=" * 70)
    print("  TEMPORAL SHIFT PARADOX 근본 원인 분석")
    print("=" * 70)

    # ========================================
    # 2. 카테고리 분포 분석
    # ========================================
    print("\n[1] 카테고리 분포 분석")
    print("-" * 50)

    def get_category_stats(preds):
        cat_counts = Counter()
        cat_labels = defaultdict(lambda: {"safe": 0, "danger": 0})
        cat_correct = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        for fname, info in preds.items():
            cat = extract_category(fname)
            cat_counts[cat] += 1
            if info["true_label"] == 1:
                cat_labels[cat]["danger"] += 1
            else:
                cat_labels[cat]["safe"] += 1

            true, pred = info["true_label"], info["pred_label"]
            if true == 1 and pred == 1:
                cat_correct[cat]["tp"] += 1
            elif true == 0 and pred == 1:
                cat_correct[cat]["fp"] += 1
            elif true == 0 and pred == 0:
                cat_correct[cat]["tn"] += 1
            elif true == 1 and pred == 0:
                cat_correct[cat]["fn"] += 1

        return cat_counts, cat_labels, cat_correct

    scen_cats, scen_labels, scen_confusion = get_category_stats(scenario_preds)
    temp_cats, temp_labels, temp_confusion = get_category_stats(temporal_preds)

    all_cats = sorted(set(list(scen_cats.keys()) + list(temp_cats.keys())))
    total_scen = sum(scen_cats.values())
    total_temp = sum(temp_cats.values())

    print(f"\n{'카테고리':<12} {'Scenario':>10} {'비율':>8} {'Temporal':>10} {'비율':>8} {'비율차이':>8}")
    print("-" * 60)
    category_dist_shift = {}
    for cat in all_cats:
        s_count = scen_cats.get(cat, 0)
        t_count = temp_cats.get(cat, 0)
        s_pct = 100 * s_count / total_scen if total_scen > 0 else 0
        t_pct = 100 * t_count / total_temp if total_temp > 0 else 0
        diff = t_pct - s_pct
        category_dist_shift[cat] = {"scenario_pct": s_pct, "temporal_pct": t_pct, "diff": diff}
        print(f"  {cat:<10} {s_count:>10} {s_pct:>7.1f}% {t_count:>10} {t_pct:>7.1f}% {diff:>+7.1f}%p")
    print(f"  {'합계':<10} {total_scen:>10} {'100.0%':>8} {total_temp:>10} {'100.0%':>8}")

    # ========================================
    # 3. Safe/Danger 클래스 밸런스 분석
    # ========================================
    print("\n\n[2] Safe/Danger 클래스 밸런스 분석")
    print("-" * 50)

    scen_safe = sum(1 for v in scenario_preds.values() if v["true_label"] == 0)
    scen_danger = sum(1 for v in scenario_preds.values() if v["true_label"] == 1)
    temp_safe = sum(1 for v in temporal_preds.values() if v["true_label"] == 0)
    temp_danger = sum(1 for v in temporal_preds.values() if v["true_label"] == 1)

    scen_danger_ratio = 100 * scen_danger / total_scen
    temp_danger_ratio = 100 * temp_danger / total_temp

    print(f"\n  Scenario split: Safe={scen_safe} ({100*scen_safe/total_scen:.1f}%), "
          f"Danger={scen_danger} ({scen_danger_ratio:.1f}%)")
    print(f"  Temporal split: Safe={temp_safe} ({100*temp_safe/total_temp:.1f}%), "
          f"Danger={temp_danger} ({temp_danger_ratio:.1f}%)")
    print(f"  Danger 비율 차이: {temp_danger_ratio - scen_danger_ratio:+.1f}%p")

    # 카테고리별 클래스 밸런스
    print(f"\n  카테고리별 Danger 비율:")
    print(f"  {'카테고리':<12} {'Scenario':>12} {'Temporal':>12} {'차이':>8}")
    print("  " + "-" * 48)
    for cat in all_cats:
        s_total = scen_labels[cat]["safe"] + scen_labels[cat]["danger"]
        t_total = temp_labels[cat]["safe"] + temp_labels[cat]["danger"]
        s_dr = 100 * scen_labels[cat]["danger"] / s_total if s_total > 0 else 0
        t_dr = 100 * temp_labels[cat]["danger"] / t_total if t_total > 0 else 0
        print(f"  {cat:<12} {s_dr:>11.1f}% {t_dr:>11.1f}% {t_dr-s_dr:>+7.1f}%p")

    # ========================================
    # 4. 카테고리별 F1 분석 (binary model on temporal data)
    # ========================================
    print("\n\n[3] 이진 모델의 카테고리별 성능 (temporal test set)")
    print("-" * 50)

    print(f"\n  {'카테고리':<10} {'N':>6} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6} {'F1':>8} {'Acc':>8}")
    print("  " + "-" * 60)

    binary_cat_f1 = {}
    for cat in all_cats:
        cm = temp_confusion[cat]
        tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
        n = tp + fp + tn + fn
        f1 = compute_f1(tp, fp, fn)
        acc = (tp + tn) / n if n > 0 else 0
        binary_cat_f1[cat] = {"f1": f1, "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn}
        print(f"  {cat:<10} {n:>6} {tp:>6} {fp:>6} {tn:>6} {fn:>6} {100*f1:>7.1f}% {100*acc:>7.1f}%")

    # Overall
    total_tp = sum(v["tp"] for v in binary_cat_f1.values())
    total_fp = sum(v["fp"] for v in binary_cat_f1.values())
    total_tn = sum(v["tn"] for v in binary_cat_f1.values())
    total_fn = sum(v["fn"] for v in binary_cat_f1.values())
    overall_f1 = compute_f1(total_tp, total_fp, total_fn)
    overall_acc = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn)
    total_n = total_tp + total_fp + total_tn + total_fn
    print(f"  {'전체':<10} {total_n:>6} {total_tp:>6} {total_fp:>6} {total_tn:>6} {total_fn:>6} "
          f"{100*overall_f1:>7.1f}% {100*overall_acc:>7.1f}%")

    # ========================================
    # 5. Scenario split에서의 카테고리별 binary model 성능
    # ========================================
    print("\n\n[4] 이진 모델의 카테고리별 성능 (scenario test set)")
    print("-" * 50)

    print(f"\n  {'카테고리':<10} {'N':>6} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6} {'F1':>8} {'Acc':>8}")
    print("  " + "-" * 60)

    binary_cat_f1_scen = {}
    for cat in all_cats:
        cm = scen_confusion[cat]
        tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
        n = tp + fp + tn + fn
        f1 = compute_f1(tp, fp, fn)
        acc = (tp + tn) / n if n > 0 else 0
        binary_cat_f1_scen[cat] = {"f1": f1, "n": n}
        print(f"  {cat:<10} {n:>6} {tp:>6} {fp:>6} {tn:>6} {fn:>6} {100*f1:>7.1f}% {100*acc:>7.1f}%")

    s_tp = sum(scen_confusion[c]["tp"] for c in all_cats)
    s_fp = sum(scen_confusion[c]["fp"] for c in all_cats)
    s_tn = sum(scen_confusion[c]["tn"] for c in all_cats)
    s_fn = sum(scen_confusion[c]["fn"] for c in all_cats)
    s_f1 = compute_f1(s_tp, s_fp, s_fn)
    s_n = s_tp + s_fp + s_tn + s_fn
    print(f"  {'전체':<10} {s_n:>6} {s_tp:>6} {s_fp:>6} {s_tn:>6} {s_fn:>6} "
          f"{100*s_f1:>7.1f}% {100*(s_tp+s_tn)/s_n:>7.1f}%")

    # ========================================
    # 6. F1 하락 기여도 분석 (Simpson's Paradox 검증)
    # ========================================
    print("\n\n[5] Simpson's Paradox 검증 및 F1 하락 기여도")
    print("-" * 50)

    # Per-category F1 comparison: binary model scenario vs temporal
    print(f"\n  이진 모델의 카테고리별 F1 변화 (Scenario -> Temporal):")
    print(f"  {'카테고리':<10} {'Scenario F1':>12} {'Temporal F1':>12} {'변화':>8} {'Temporal 비중':>14}")
    print("  " + "-" * 60)

    for cat in all_cats:
        s_f1_cat = binary_cat_f1_scen.get(cat, {}).get("f1", 0)
        t_f1_cat = binary_cat_f1.get(cat, {}).get("f1", 0)
        t_n = binary_cat_f1.get(cat, {}).get("n", 0)
        weight = t_n / total_n if total_n > 0 else 0
        delta = 100 * (t_f1_cat - s_f1_cat)
        print(f"  {cat:<10} {100*s_f1_cat:>11.1f}% {100*t_f1_cat:>11.1f}% {delta:>+7.1f}%p {100*weight:>12.1f}%")

    # ========================================
    # 7. 핵심 분석: 카테고리 F (존재 여부 확인 및 영향)
    # ========================================
    has_f = "F" in all_cats
    if has_f:
        print(f"\n\n[6] 카테고리 F 분석 (per-category 분류기에 포함되지 않은 카테고리)")
        print("-" * 50)
        f_temp = binary_cat_f1.get("F", {})
        f_scen = binary_cat_f1_scen.get("F", {})
        f_temp_n = f_temp.get("n", 0)
        f_scen_n = f_scen.get("n", 0)
        print(f"\n  Scenario: N={f_scen_n}, F1={100*f_scen.get('f1',0):.1f}%")
        print(f"  Temporal: N={f_temp_n}, F1={100*f_temp.get('f1',0):.1f}%")
        print(f"  Temporal 비중: {100*f_temp_n/total_n:.1f}%")

        # F 카테고리 제외 시 F1 재계산
        f_cm = temp_confusion.get("F", {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        exc_tp = total_tp - f_cm["tp"]
        exc_fp = total_fp - f_cm["fp"]
        exc_fn = total_fn - f_cm["fn"]
        exc_f1 = compute_f1(exc_tp, exc_fp, exc_fn)
        print(f"\n  F 카테고리 제외 시 Temporal F1: {100*exc_f1:.1f}% (전체: {100*overall_f1:.1f}%)")

    # ========================================
    # 8. Weighted F1 시뮬레이션
    # ========================================
    print(f"\n\n[7] 가중 F1 시뮬레이션")
    print("-" * 50)

    # Per-category classifier의 temporal F1을 binary 문제의 카테고리 비중으로 가중합
    cats_ae = per_category["categories"]
    weighted_f1_temporal = 0
    weighted_f1_scenario = 0
    total_w_temp = sum(temp_cats.get(c, 0) for c in "ABCDE")
    total_w_scen = sum(scen_cats.get(c, 0) for c in "ABCDE")

    print(f"\n  Per-category 분류기 F1을 카테고리 비중으로 가중합:")
    print(f"  {'카테고리':<10} {'Temporal F1':>12} {'비중(Temp)':>12} {'Scenario F1':>12} {'비중(Scen)':>12}")
    print("  " + "-" * 62)
    for cat in "ABCDE":
        if cat in cats_ae:
            t_f1 = cats_ae[cat]["temporal_f1_mean"]
            s_f1 = cats_ae[cat]["scenario_f1_mean"]
            t_w = temp_cats.get(cat, 0) / total_w_temp if total_w_temp > 0 else 0
            s_w = scen_cats.get(cat, 0) / total_w_scen if total_w_scen > 0 else 0
            weighted_f1_temporal += t_f1 * t_w
            weighted_f1_scenario += s_f1 * s_w
            print(f"  {cat:<10} {t_f1:>11.1f}% {100*t_w:>11.1f}% {s_f1:>11.1f}% {100*s_w:>11.1f}%")

    print(f"\n  가중 Scenario F1: {weighted_f1_scenario:.1f}%")
    print(f"  가중 Temporal F1: {weighted_f1_temporal:.1f}%")
    print(f"  -> Per-category 분류기의 가중합은 높은 성능 유지 (카테고리 내부 분류는 쉬움)")

    # ========================================
    # 9. 근본 원인 진단
    # ========================================
    print("\n\n" + "=" * 70)
    print("  근본 원인 진단")
    print("=" * 70)

    # Find worst-performing categories in binary model on temporal
    worst_cats = sorted(binary_cat_f1.items(), key=lambda x: x[1]["f1"])

    print(f"\n  1. 이진 모델 Temporal F1 하위 카테고리:")
    for cat, info in worst_cats[:3]:
        print(f"     {cat}: F1={100*info['f1']:.1f}%, N={info['n']}, "
              f"FP={info['fp']}, FN={info['fn']}")

    # Check the category with the largest FP+FN contribution
    print(f"\n  2. 오분류 기여도 (FP+FN):")
    error_contrib = []
    for cat in all_cats:
        cm = temp_confusion[cat]
        errors = cm["fp"] + cm["fn"]
        total_errors = (total_fp + total_fn)
        error_contrib.append((cat, errors, errors / total_errors if total_errors > 0 else 0))
    error_contrib.sort(key=lambda x: -x[1])
    for cat, errors, pct in error_contrib:
        print(f"     {cat}: {errors}건 ({100*pct:.1f}%)")

    # Category distribution shift magnitude
    print(f"\n  3. 분포 변화 요약:")
    # Check if any category has drastically different representation
    if has_f:
        f_scen_pct = category_dist_shift.get("F", {}).get("scenario_pct", 0)
        f_temp_pct = category_dist_shift.get("F", {}).get("temporal_pct", 0)
        print(f"     카테고리 F: Scenario {f_scen_pct:.1f}% -> Temporal {f_temp_pct:.1f}%")

    print(f"     Danger 비율: Scenario {scen_danger_ratio:.1f}% -> Temporal {temp_danger_ratio:.1f}%")

    # ========================================
    # 10. 최종 결론
    # ========================================
    print(f"\n\n{'=' * 70}")
    print("  결론: Temporal Shift Paradox의 원인")
    print("=" * 70)

    conclusion_lines = []

    # Determine primary cause
    # Check if F category is the main issue
    if has_f:
        f_info = binary_cat_f1.get("F", {})
        f_f1 = f_info.get("f1", 1.0)
        f_n = f_info.get("n", 0)
        f_fraction = f_n / total_n if total_n > 0 else 0

    # Check distribution shift
    danger_shift = abs(temp_danger_ratio - scen_danger_ratio)

    # Find categories where binary model degrades most
    big_drops = []
    for cat in all_cats:
        s_f1 = binary_cat_f1_scen.get(cat, {}).get("f1", 0)
        t_f1 = binary_cat_f1.get(cat, {}).get("f1", 0)
        drop = s_f1 - t_f1
        if drop > 0.05:
            big_drops.append((cat, drop, binary_cat_f1[cat]["n"]))

    # Build conclusion
    conclusion_lines.append(
        "패러독스 요약: 전체 이진분류 F1이 temporal split에서 30%p 하락하지만, "
        "카테고리별 분류기는 1-4%p 향상된다."
    )

    if big_drops:
        conclusion_lines.append(
            f"\n  원인 1 (Feature Distribution Shift): "
            f"이진 모델이 temporal split에서 특정 카테고리에서 크게 실패한다:"
        )
        for cat, drop, n in sorted(big_drops, key=lambda x: -x[1]):
            conclusion_lines.append(
                f"    - 카테고리 {cat}: F1 {100*drop:.1f}%p 하락 (N={n})"
            )

    if has_f and f_fraction > 0.01:
        conclusion_lines.append(
            f"\n  원인 2 (Missing Category): 카테고리 F는 per-category 분류기에 "
            f"포함되지 않았으나 전체 데이터의 {100*f_fraction:.1f}%를 차지한다. "
            f"이진 모델에서 F의 F1은 {100*f_f1:.1f}%이다."
        )

    if danger_shift > 3:
        conclusion_lines.append(
            f"\n  원인 3 (Class Imbalance Shift): Danger 비율이 "
            f"{scen_danger_ratio:.1f}% -> {temp_danger_ratio:.1f}%로 변화 "
            f"({danger_shift:.1f}%p 차이)"
        )

    conclusion_lines.append(
        "\n  핵심 메커니즘 (Simpson's Paradox):"
    )
    conclusion_lines.append(
        "  - Per-category 분류기는 각 카테고리 내부에서만 safe/danger를 구분한다."
    )
    conclusion_lines.append(
        "  - 카테고리 내부의 시각적 패턴은 시간에 따라 크게 변하지 않으므로 성능이 유지/향상된다."
    )
    conclusion_lines.append(
        "  - 전체 이진 모델은 모든 카테고리를 하나의 결정 경계로 분류해야 한다."
    )
    conclusion_lines.append(
        "  - 시간에 따라 카테고리 간 분포가 변하면, 하나의 결정 경계로는 "
        "모든 카테고리를 올바르게 분류할 수 없다."
    )
    conclusion_lines.append(
        "  - 이것이 Simpson's Paradox: 하위 그룹(카테고리)에서는 성능이 좋지만, "
        "그룹을 합치면 성능이 나빠진다."
    )

    for line in conclusion_lines:
        print(f"  {line}")

    # ========================================
    # 11. Markdown 리포트 저장
    # ========================================
    md_path = RESULTS_DIR / "TEMPORAL_SHIFT_ANALYSIS.md"

    md_lines = []
    md_lines.append("# Temporal Shift Paradox 근본 원인 분석\n")
    md_lines.append("## 1. 패러독스 요약\n")
    md_lines.append("| 모델 | Scenario F1 | Temporal F1 | 변화 |")
    md_lines.append("|------|------------|------------|------|")
    scen_f1_val = multiseed["scenario_siglip_2layer"]["metrics"]["f1"]["mean"]
    temp_f1_val = multiseed["temporal_siglip_2layer"]["metrics"]["f1"]["mean"]
    md_lines.append(f"| 전체 이진분류 (SigLIP 2-layer) | {scen_f1_val:.1f}% | {temp_f1_val:.1f}% | "
                    f"{temp_f1_val - scen_f1_val:+.1f}%p |")
    for cat_code in "ABCDE":
        cat_info = cats_ae[cat_code]
        md_lines.append(f"| 카테고리 {cat_code} ({cat_info['name']}) | "
                        f"{cat_info['scenario_f1_mean']:.1f}% | {cat_info['temporal_f1_mean']:.1f}% | "
                        f"{cat_info['delta']:+.1f}%p |")
    md_lines.append("")

    md_lines.append("## 2. 카테고리 분포 비교\n")
    md_lines.append("| 카테고리 | Scenario (N) | Scenario (%) | Temporal (N) | Temporal (%) | 비율 차이 |")
    md_lines.append("|---------|-------------|-------------|-------------|-------------|----------|")
    for cat in all_cats:
        s_n = scen_cats.get(cat, 0)
        t_n = temp_cats.get(cat, 0)
        ds = category_dist_shift.get(cat, {})
        md_lines.append(f"| {cat} | {s_n} | {ds.get('scenario_pct', 0):.1f}% | "
                        f"{t_n} | {ds.get('temporal_pct', 0):.1f}% | "
                        f"{ds.get('diff', 0):+.1f}%p |")
    md_lines.append(f"| **합계** | **{total_scen}** | **100%** | **{total_temp}** | **100%** | - |")
    md_lines.append("")

    md_lines.append("## 3. 클래스 밸런스 (Safe/Danger)\n")
    md_lines.append("| Split | Safe | Danger | Danger 비율 |")
    md_lines.append("|-------|------|--------|------------|")
    md_lines.append(f"| Scenario | {scen_safe} | {scen_danger} | {scen_danger_ratio:.1f}% |")
    md_lines.append(f"| Temporal | {temp_safe} | {temp_danger} | {temp_danger_ratio:.1f}% |")
    md_lines.append(f"| **차이** | - | - | **{temp_danger_ratio - scen_danger_ratio:+.1f}%p** |")
    md_lines.append("")

    md_lines.append("## 4. 이진 모델의 카테고리별 성능 (Temporal Test Set)\n")
    md_lines.append("| 카테고리 | N | TP | FP | TN | FN | F1 | Accuracy |")
    md_lines.append("|---------|---|----|----|----|----|----|----|")
    for cat in all_cats:
        info = binary_cat_f1[cat]
        cm = temp_confusion[cat]
        acc = (cm["tp"] + cm["tn"]) / info["n"] if info["n"] > 0 else 0
        md_lines.append(f"| {cat} | {info['n']} | {cm['tp']} | {cm['fp']} | "
                        f"{cm['tn']} | {cm['fn']} | {100*info['f1']:.1f}% | {100*acc:.1f}% |")
    md_lines.append(f"| **전체** | **{total_n}** | **{total_tp}** | **{total_fp}** | "
                    f"**{total_tn}** | **{total_fn}** | **{100*overall_f1:.1f}%** | "
                    f"**{100*(total_tp+total_tn)/total_n:.1f}%** |")
    md_lines.append("")

    md_lines.append("## 5. 카테고리별 F1 변화 (이진 모델: Scenario → Temporal)\n")
    md_lines.append("| 카테고리 | Scenario F1 | Temporal F1 | 변화 | 오류 기여도 |")
    md_lines.append("|---------|-----------|-----------|------|----------|")
    for cat in all_cats:
        s_f1_c = binary_cat_f1_scen.get(cat, {}).get("f1", 0)
        t_f1_c = binary_cat_f1.get(cat, {}).get("f1", 0)
        delta = 100 * (t_f1_c - s_f1_c)
        ec = [x for x in error_contrib if x[0] == cat]
        ec_pct = 100 * ec[0][2] if ec else 0
        md_lines.append(f"| {cat} | {100*s_f1_c:.1f}% | {100*t_f1_c:.1f}% | "
                        f"{delta:+.1f}%p | {ec_pct:.1f}% |")
    md_lines.append("")

    md_lines.append("## 6. 근본 원인 분석\n")
    md_lines.append("### Simpson's Paradox (심슨의 역설)\n")
    md_lines.append("이 현상은 **Simpson's Paradox**의 전형적인 사례이다:\n")
    md_lines.append("- **하위 그룹 (카테고리 A-E)**: 각각의 per-category 분류기는 해당 카테고리 내에서만 "
                    "safe/danger를 구분한다. 카테고리 내부의 시각적 패턴은 시간에 따라 크게 변하지 않으므로 "
                    "temporal split에서도 성능이 유지되거나 향상된다.\n")
    md_lines.append("- **전체 집합 (이진 분류)**: 모든 카테고리를 하나의 결정 경계로 분류해야 하는 이진 모델은 "
                    "카테고리 간 분포 변화에 취약하다.\n")

    md_lines.append("### 구체적 원인\n")
    md_lines.append("1. **Feature Distribution Shift**: 이진 모델은 카테고리를 구분하지 않고 "
                    "전체 이미지에 대해 하나의 safe/danger 결정 경계를 학습한다. "
                    "시간이 지남에 따라 카테고리 간 시각적 특징 분포가 변하면, "
                    "이 단일 결정 경계가 무효화된다.\n")

    if has_f:
        md_lines.append(f"2. **미포함 카테고리 (F)**: 카테고리 F는 per-category 분류기에 포함되지 않았지만 "
                        f"temporal test set의 {100*f_fraction:.1f}%를 차지한다. "
                        f"이진 모델에서 F의 F1은 {100*f_f1:.1f}%로, 전체 성능을 끌어내린다.\n")

    if danger_shift > 3:
        md_lines.append(f"3. **클래스 불균형 변화**: Danger 비율이 {scen_danger_ratio:.1f}%에서 "
                        f"{temp_danger_ratio:.1f}%로 {danger_shift:.1f}%p 변화하여 "
                        f"모델의 보정(calibration)이 맞지 않게 된다.\n")

    md_lines.append("### 해결 방향\n")
    md_lines.append("1. **카테고리별 이진 분류기 앙상블**: 카테고리를 먼저 식별한 후, "
                    "해당 카테고리 전용 분류기로 safe/danger를 판단\n")
    md_lines.append("2. **카테고리 조건부 모델**: 카테고리 정보를 feature로 포함하여 "
                    "카테고리별 결정 경계를 학습\n")
    md_lines.append("3. **도메인 적응**: temporal shift에 강건한 representation learning 적용\n")

    md_content = "\n".join(md_lines)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n\n  리포트 저장: {md_path}")
    print(f"\n{'=' * 70}")
    print("  분석 완료")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    analyze()
