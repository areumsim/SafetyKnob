"""
5-Dimensional Label Generation Script

Automatically generates 5D safety labels from accident type codes in filenames.
Maps accident categories (A-E) to safety dimensions:
- A → Fall Hazard
- B → Collision Risk
- C → Equipment Hazard
- D → Environmental Risk
- E → Protective Gear
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# Category to dimension mapping
CATEGORY_MAPPING = {
    'A': 'fall_hazard',
    'B': 'collision_risk',
    'C': 'equipment_hazard',
    'D': 'environmental_risk',
    'E': 'protective_gear'
}

DIMENSION_NAMES = [
    'fall_hazard',
    'collision_risk',
    'equipment_hazard',
    'environmental_risk',
    'protective_gear'
]


def extract_accident_code(filename):
    """
    Extract accident type code from filename.

    Pattern: H-[date]_[AccidentCode]_[Scenario]_[SubID]_[Frame].jpg
    Example: H-220927_A35_N-07_005_0061.jpg → A35
    """
    match = re.search(r'_([A-F]\d{2})_', filename)
    if match:
        return match.group(1)
    return None


def generate_5d_label(overall_safety, accident_code):
    """
    Generate 5-dimensional label from overall safety and accident code.

    Rules:
    1. Initialize all dimensions to 0.9 (not applicable)
    2. For the matching dimension:
       - If danger (overall_safety=0): set to 0.0 (high risk)
       - If safe (overall_safety=1): set to 1.0 (low risk)
    3. For F-category (caution): set all to 0.5 (ambiguous)
    """
    dimensions = {dim: 0.9 for dim in DIMENSION_NAMES}

    if not accident_code:
        return dimensions

    category = accident_code[0]

    # Special case: F category (caution) → all dimensions 0.5
    if category == 'F':
        dimensions = {dim: 0.5 for dim in DIMENSION_NAMES}
        return dimensions

    # Normal case: A-E categories
    if category in CATEGORY_MAPPING:
        target_dimension = CATEGORY_MAPPING[category]
        # danger=0 → high risk (0.0), safe=1 → low risk (1.0)
        dimensions[target_dimension] = 1.0 if overall_safety == 1 else 0.0

    return dimensions


def main():
    print("=" * 60)
    print("5D Label Generation Script")
    print("=" * 60)

    # Paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / 'data_scenario' / 'labels.json'
    output_file = project_root / 'data_scenario' / 'labels_5d.json'
    stats_file = project_root / 'data_scenario' / '5d_label_stats.json'

    print(f"\nInput:  {input_file}")
    print(f"Output: {output_file}")

    # Load original labels
    with open(input_file, 'r') as f:
        original_labels = json.load(f)

    print(f"\nLoaded {len(original_labels)} labels")

    # Generate 5D labels
    labels_5d = {}
    stats = {
        'total': 0,
        'per_dimension': {dim: {'count': 0, 'danger': 0, 'safe': 0, 'caution': 0} for dim in DIMENSION_NAMES},
        'per_category': {},
        'missing_code': 0
    }

    print("\nGenerating 5D labels...")

    for img_path, label_data in original_labels.items():
        overall_safety = label_data['overall_safety']
        class_label = label_data['class']

        # Extract accident code from filename
        filename = img_path.split('/')[-1]
        accident_code = extract_accident_code(filename)

        if not accident_code:
            stats['missing_code'] += 1
            continue

        # Generate 5D dimensions
        dimensions = generate_5d_label(overall_safety, accident_code)

        # Store extended label
        labels_5d[img_path] = {
            'overall_safety': overall_safety,
            'class': class_label,
            'accident_code': accident_code,
            'dimensions': dimensions
        }

        # Update statistics
        stats['total'] += 1
        category = accident_code[0]

        # Per category stats
        if category not in stats['per_category']:
            stats['per_category'][category] = {'count': 0, 'codes': set()}
        stats['per_category'][category]['count'] += 1
        stats['per_category'][category]['codes'].add(accident_code)

        # Per dimension stats
        if category in CATEGORY_MAPPING:
            dim_name = CATEGORY_MAPPING[category]
            stats['per_dimension'][dim_name]['count'] += 1

            if class_label == 'danger':
                stats['per_dimension'][dim_name]['danger'] += 1
            elif class_label == 'safe':
                stats['per_dimension'][dim_name]['safe'] += 1
            elif class_label == 'caution':
                stats['per_dimension'][dim_name]['caution'] += 1

    # Convert sets to lists for JSON serialization
    for cat_stat in stats['per_category'].values():
        cat_stat['codes'] = sorted(list(cat_stat['codes']))

    # Save 5D labels
    with open(output_file, 'w') as f:
        json.dump(labels_5d, f, indent=2)

    print(f"\n✅ Saved {len(labels_5d)} 5D labels to {output_file}")

    # Save statistics
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Saved statistics to {stats_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("5D Label Statistics")
    print("=" * 60)

    print(f"\nTotal labels: {stats['total']}")
    print(f"Missing codes: {stats['missing_code']}")

    print("\n[Per Dimension]")
    for dim in DIMENSION_NAMES:
        dim_stats = stats['per_dimension'][dim]
        count = dim_stats['count']
        danger = dim_stats['danger']
        safe = dim_stats['safe']
        caution = dim_stats['caution']

        if count > 0:
            danger_pct = danger / count * 100
            safe_pct = safe / count * 100
            print(f"  {dim:24s}: {count:5d} images (Danger: {danger:4d} {danger_pct:5.1f}%, Safe: {safe:4d} {safe_pct:5.1f}%)")

    print("\n[Per Category]")
    for category in sorted(stats['per_category'].keys()):
        cat_stats = stats['per_category'][category]
        count = cat_stats['count']
        codes = cat_stats['codes']

        dim_name = CATEGORY_MAPPING.get(category, 'Caution')
        print(f"  Category {category} ({dim_name:24s}): {count:5d} images ({len(codes)} unique codes)")
        print(f"    Codes: {', '.join(codes)}")

    print("\n" + "=" * 60)
    print("✅ 5D Label Generation Complete!")
    print("=" * 60)

    # Validation checks
    print("\n[Validation]")

    # Check 1: Total count matches
    expected_total = sum(stats['per_category'].values(), key=lambda x: x['count'])
    if stats['total'] == len(labels_5d):
        print("  ✅ Total count matches")
    else:
        print(f"  ❌ Total count mismatch: {stats['total']} vs {len(labels_5d)}")

    # Check 2: Dimension counts
    total_dim_count = sum(s['count'] for s in stats['per_dimension'].values())
    # F-category doesn't map to single dimension, so total_dim_count may differ
    print(f"  ℹ️  Dimension images: {total_dim_count} (excludes F-category)")

    # Check 3: Balance per dimension
    print("\n[Balance Check]")
    for dim in DIMENSION_NAMES:
        dim_stats = stats['per_dimension'][dim]
        count = dim_stats['count']
        danger = dim_stats['danger']
        safe = dim_stats['safe']

        if count > 0:
            balance = abs(danger - safe) / count
            if balance < 0.15:
                status = "✅ Balanced"
            elif balance < 0.25:
                status = "⚠️  Slightly imbalanced"
            else:
                status = "❌ Imbalanced"

            print(f"  {dim:24s}: {status} (diff: {balance*100:.1f}%)")

    print()


if __name__ == '__main__':
    main()
