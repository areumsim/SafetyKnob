#!/usr/bin/env python3
"""
Category Distribution Analysis for Simpson's Paradox Evidence

Quantifies the category distribution shift between scenario and temporal splits,
providing statistical evidence for why binary classifiers fail under temporal shift
while per-category classifiers remain robust.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from scipy.stats import chi2_contingency
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F']
CAT_NAMES = {
    'A': 'Fall Hazard', 'B': 'Collision Risk', 'C': 'Equipment Hazard',
    'D': 'Environmental Risk', 'E': 'Protective Gear', 'F': 'Caution'
}


def count_categories(data_dir, subset):
    """Count category occurrences in a data split."""
    p = Path(data_dir) / subset
    if not p.exists():
        return {}

    cats = defaultdict(int)
    safety = defaultdict(lambda: {'safe': 0, 'danger': 0, 'caution': 0})

    for f in p.glob('*.jpg'):
        m = re.search(r'_([A-F]\d{2})_', f.name)
        if m:
            code = m.group(1)
            cat = code[0]
            cats[cat] += 1

            # Determine safety from Y/N in filename
            if '_Y-' in f.name:
                safety[cat]['safe'] += 1
            elif '_N-' in f.name:
                safety[cat]['danger'] += 1
            else:
                safety[cat]['caution'] += 1

    return dict(cats), dict(safety)


def main():
    output_dir = Path('results/category_distribution')
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        'scenario_v2_train': ('data_scenario_v2', 'train'),
        'scenario_v2_test': ('data_scenario_v2', 'test'),
        'temporal_train': ('data_temporal', 'train'),
        'temporal_test': ('data_temporal', 'test'),
    }

    print("=" * 70)
    print("CATEGORY DISTRIBUTION ANALYSIS")
    print("Evidence for Simpson's Paradox in Temporal Distribution Shift")
    print("=" * 70)

    all_counts = {}
    all_safety = {}
    for name, (data_dir, subset) in splits.items():
        counts, safety = count_categories(data_dir, subset)
        all_counts[name] = counts
        all_safety[name] = safety
        total = sum(counts.values())
        print(f"\n{name} (n={total}):")
        for cat in CATEGORIES:
            n = counts.get(cat, 0)
            pct = n / total * 100 if total > 0 else 0
            s = safety.get(cat, {})
            safe = s.get('safe', 0)
            danger = s.get('danger', 0) + s.get('caution', 0)
            danger_rate = danger / n * 100 if n > 0 else 0
            print(f"  {cat} ({CAT_NAMES[cat]:<18}): {n:5d} ({pct:5.1f}%)  danger_rate={danger_rate:.1f}%")

    # Compare scenario_v2 train vs temporal test (the critical comparison)
    print("\n" + "=" * 70)
    print("KEY COMPARISON: Scenario Train vs Temporal Test")
    print("(This is what causes Simpson's Paradox)")
    print("=" * 70)

    s_train = all_counts['scenario_v2_train']
    t_test = all_counts['temporal_test']
    s_total = sum(s_train.values())
    t_total = sum(t_test.values())

    print(f"\n{'Category':<25} {'Scenario Train':>15} {'Temporal Test':>15} {'Shift':>10}")
    print("-" * 70)

    max_shift = 0
    max_shift_cat = ''
    for cat in CATEGORIES:
        s_pct = s_train.get(cat, 0) / s_total * 100
        t_pct = t_test.get(cat, 0) / t_total * 100
        shift = t_pct - s_pct
        if abs(shift) > abs(max_shift):
            max_shift = shift
            max_shift_cat = cat
        print(f"  {cat} ({CAT_NAMES[cat]:<18}) {s_pct:>10.1f}%  {t_pct:>10.1f}%  {shift:>+8.1f}%p")

    print(f"\nLargest shift: {max_shift_cat} ({CAT_NAMES[max_shift_cat]}): {max_shift:+.1f}%p")

    # Also compare danger rates per category
    print("\n" + "=" * 70)
    print("DANGER RATE COMPARISON (per category)")
    print("=" * 70)

    s_safety = all_safety['scenario_v2_train']
    t_safety = all_safety['temporal_test']

    print(f"\n{'Category':<25} {'Scenario Danger%':>18} {'Temporal Danger%':>18} {'Shift':>10}")
    print("-" * 75)
    for cat in CATEGORIES[:5]:  # Exclude F (caution)
        s = s_safety.get(cat, {})
        t = t_safety.get(cat, {})
        s_n = sum(s.values()) if s else 1
        t_n = sum(t.values()) if t else 1
        s_dr = (s.get('danger', 0) + s.get('caution', 0)) / s_n * 100
        t_dr = (t.get('danger', 0) + t.get('caution', 0)) / t_n * 100
        shift = t_dr - s_dr
        print(f"  {cat} ({CAT_NAMES[cat]:<18}) {s_dr:>14.1f}%  {t_dr:>14.1f}%  {shift:>+8.1f}%p")

    # Chi-square test
    if HAS_SCIPY:
        print("\n" + "=" * 70)
        print("STATISTICAL TEST: Chi-Square for Distribution Difference")
        print("=" * 70)

        # Construct contingency table
        cats_to_test = ['A', 'B', 'C', 'D', 'E']
        observed = np.array([
            [s_train.get(c, 0) for c in cats_to_test],
            [t_test.get(c, 0) for c in cats_to_test],
        ])

        chi2, p_value, dof, expected = chi2_contingency(observed)
        print(f"\nChi-square statistic: {chi2:.2f}")
        print(f"Degrees of freedom: {dof}")
        print(f"p-value: {p_value:.6f}")
        print(f"Significant (p<0.001): {'YES' if p_value < 0.001 else 'NO'}")

        # Effect size: Cramér's V
        n = observed.sum()
        min_dim = min(observed.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        print(f"Cramér's V (effect size): {cramers_v:.4f}")
        if cramers_v < 0.1:
            print("  → Negligible effect")
        elif cramers_v < 0.3:
            print("  → Small effect")
        elif cramers_v < 0.5:
            print("  → Medium effect")
        else:
            print("  → Large effect")

    # Explanation of Simpson's Paradox
    print("\n" + "=" * 70)
    print("SIMPSON'S PARADOX EXPLANATION")
    print("=" * 70)
    print("""
The binary classifier trains on the MARGINAL distribution P(safe|image),
which implicitly depends on P(category) × P(safe|category).

When the category distribution shifts (e.g., B doubles from 14.6% to 26.8%),
the marginal P(safe) changes even if P(safe|category) stays constant.

The per-category classifiers only learn P(safe|category, image), which is
invariant to P(category) changes → robust to temporal shift.

The hierarchical pipeline explicitly decomposes:
  P(safe|image) = Σ_c P(c|image) × P(safe|image, c)
This makes it naturally robust to label shift (category distribution changes).
""")

    # Save results
    results = {
        'counts': {k: dict(v) for k, v in all_counts.items()},
        'safety': {k: {cat: dict(s) for cat, s in v.items()} for k, v in all_safety.items()},
    }
    if HAS_SCIPY:
        results['chi_square'] = {
            'statistic': float(chi2),
            'p_value': float(p_value),
            'dof': int(dof),
            'cramers_v': float(cramers_v),
        }

    with open(output_dir / 'category_distribution.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        cats = ['A', 'B', 'C', 'D', 'E']
        x = np.arange(len(cats))
        width = 0.35

        # Category distribution comparison
        ax = axes[0]
        s_pcts = [s_train.get(c, 0) / s_total * 100 for c in cats]
        t_pcts = [t_test.get(c, 0) / t_total * 100 for c in cats]
        bars1 = ax.bar(x - width/2, s_pcts, width, label='Scenario Train', color='#4C72B0')
        bars2 = ax.bar(x + width/2, t_pcts, width, label='Temporal Test', color='#DD8452')
        ax.set_xlabel('Accident Category')
        ax.set_ylabel('Proportion (%)')
        ax.set_title('Category Distribution Shift\n(Cause of Simpson\'s Paradox)')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n{CAT_NAMES[c].split()[0]}" for c in cats])
        ax.legend()
        ax.set_ylim(0, 35)

        # Shift magnitude
        ax = axes[1]
        shifts = [t_pcts[i] - s_pcts[i] for i in range(len(cats))]
        colors = ['#C44E52' if s > 0 else '#4C72B0' for s in shifts]
        ax.bar(x, shifts, color=colors)
        ax.set_xlabel('Accident Category')
        ax.set_ylabel('Distribution Shift (%p)')
        ax.set_title('Magnitude of Category Distribution Change')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n{CAT_NAMES[c].split()[0]}" for c in cats])
        ax.axhline(y=0, color='black', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / 'category_distribution_shift.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nFigure saved to {output_dir / 'category_distribution_shift.png'}")

    print(f"\nResults saved to {output_dir / 'category_distribution.json'}")


if __name__ == '__main__':
    main()
