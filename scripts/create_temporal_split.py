"""
Temporal Split Generation Script

Creates date-based train/test split to evaluate distribution shift robustness.
Uses data from 2022/06-09 for training and 2022/10-11 for testing.

This tests model robustness to temporal changes in:
- Lighting conditions
- Weather patterns
- Seasonal clothing/equipment
- Worker behavior patterns
"""

import json
import re
import shutil
from pathlib import Path
from collections import defaultdict

def extract_date(filename):
    """
    Extract date from filename.

    Pattern: H-[YYMMDD]_[...]
    Example: H-220927_A35_N-07_005_0061.jpg → 220927
    """
    match = re.search(r'H-(\d{6})_', filename)
    if match:
        return match.group(1)
    return None


def parse_date(date_str):
    """
    Parse date string to year-month tuple.

    Args:
        date_str: YYMMDD format (e.g., "220927")

    Returns:
        (year, month) tuple (e.g., (2022, 9))
    """
    if not date_str or len(date_str) != 6:
        return None, None

    year = 2000 + int(date_str[:2])
    month = int(date_str[2:4])

    return year, month


def main():
    print("=" * 60)
    print("Temporal Split Generation")
    print("=" * 60)

    # Paths
    project_root = Path(__file__).parent.parent
    source_dir = project_root / 'data_scenario'
    target_dir = project_root / 'data_temporal'
    labels_file = source_dir / 'labels.json'
    labels_5d_file = source_dir / 'labels_5d.json'

    print(f"\nSource: {source_dir}")
    print(f"Target: {target_dir}")

    # Create target directories
    (target_dir / 'train').mkdir(parents=True, exist_ok=True)
    (target_dir / 'test').mkdir(parents=True, exist_ok=True)

    # Load labels
    print(f"\nLoading labels from {labels_file}...")
    with open(labels_file, 'r') as f:
        all_labels = json.load(f)

    with open(labels_5d_file, 'r') as f:
        all_labels_5d = json.load(f)

    print(f"Loaded {len(all_labels)} labels")

    # Analyze dates
    date_counts = defaultdict(lambda: {'total': 0, 'danger': 0, 'safe': 0, 'caution': 0})
    file_dates = {}

    for img_path in all_labels.keys():
        # Get original filename (remove train/val/test prefix)
        filename = img_path.split('/')[-1]
        date_str = extract_date(filename)

        if not date_str:
            print(f"Warning: Could not extract date from {filename}")
            continue

        year, month = parse_date(date_str)
        if not year or not month:
            print(f"Warning: Invalid date {date_str} from {filename}")
            continue

        period_key = f"{year}-{month:02d}"
        file_dates[img_path] = (year, month)

        label_data = all_labels[img_path]
        class_label = label_data['class']

        date_counts[period_key]['total'] += 1
        date_counts[period_key][class_label] += 1

    # Print date distribution
    print("\n" + "=" * 60)
    print("Date Distribution")
    print("=" * 60)
    print(f"{'Period':<12} {'Total':>8} {'Danger':>8} {'Safe':>8} {'Caution':>8}")
    print("-" * 60)

    for period in sorted(date_counts.keys()):
        counts = date_counts[period]
        print(f"{period:<12} {counts['total']:>8} {counts['danger']:>8} "
              f"{counts['safe']:>8} {counts['caution']:>8}")

    # Split by date
    print("\n" + "=" * 60)
    print("Creating Temporal Split")
    print("=" * 60)
    print("\nSplit Strategy:")
    print("  Train: 2022-06 to 2022-09")
    print("  Test:  2022-10 to 2022-11")

    train_labels = {}
    test_labels = {}
    train_labels_5d = {}
    test_labels_5d = {}

    stats = {
        'train': {'total': 0, 'danger': 0, 'safe': 0, 'caution': 0},
        'test': {'total': 0, 'danger': 0, 'safe': 0, 'caution': 0}
    }

    skipped = 0

    for img_path, (year, month) in file_dates.items():
        # Get original filename
        filename = img_path.split('/')[-1]

        # Determine split
        if 6 <= month <= 9:
            split = 'train'
        elif 10 <= month <= 11:
            split = 'test'
        else:
            # Skip other months (if any)
            skipped += 1
            continue

        # Get source file path (from original data_scenario)
        # img_path is like "train/H-220927_A35_N-07_005_0061.jpg"
        source_subdir = img_path.split('/')[0]  # train, val, or test
        source_file = source_dir / source_subdir / filename

        if not source_file.exists():
            print(f"Warning: Source file not found: {source_file}")
            continue

        # Copy to new split
        target_file = target_dir / split / filename
        shutil.copy2(source_file, target_file)

        # Store labels with new path
        new_path = f"{split}/{filename}"

        # Binary labels
        if split == 'train':
            train_labels[new_path] = all_labels[img_path]
        else:
            test_labels[new_path] = all_labels[img_path]

        # 5D labels
        if img_path in all_labels_5d:
            if split == 'train':
                train_labels_5d[new_path] = all_labels_5d[img_path]
            else:
                test_labels_5d[new_path] = all_labels_5d[img_path]

        # Update stats
        class_label = all_labels[img_path]['class']
        stats[split]['total'] += 1
        stats[split][class_label] += 1

    # Print split statistics
    print("\n" + "=" * 60)
    print("Split Statistics")
    print("=" * 60)

    for split in ['train', 'test']:
        s = stats[split]
        print(f"\n{split.upper()}:")
        print(f"  Total:   {s['total']:6d} images")
        print(f"  Danger:  {s['danger']:6d} ({s['danger']/s['total']*100:.1f}%)")
        print(f"  Safe:    {s['safe']:6d} ({s['safe']/s['total']*100:.1f}%)")
        print(f"  Caution: {s['caution']:6d} ({s['caution']/s['total']*100:.1f}%)")

    if skipped > 0:
        print(f"\nSkipped: {skipped} images (outside 06-11 range)")

    # Save binary labels
    train_labels_file = target_dir / 'labels.json'
    test_labels_file = target_dir / 'test_labels.json'

    # Combine train+test for compatibility with existing code
    combined_labels = {**train_labels, **test_labels}

    with open(train_labels_file, 'w') as f:
        json.dump(combined_labels, f, indent=2)

    print(f"\n✅ Binary labels saved to {train_labels_file}")

    # Save 5D labels
    train_labels_5d_file = target_dir / 'labels_5d.json'
    combined_labels_5d = {**train_labels_5d, **test_labels_5d}

    with open(train_labels_5d_file, 'w') as f:
        json.dump(combined_labels_5d, f, indent=2)

    print(f"✅ 5D labels saved to {train_labels_5d_file}")

    # Save statistics
    stats_file = target_dir / 'temporal_split_stats.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'split_strategy': {
                'train': '2022-06 to 2022-09',
                'test': '2022-10 to 2022-11'
            },
            'statistics': stats,
            'date_distribution': dict(date_counts)
        }, f, indent=2)

    print(f"✅ Statistics saved to {stats_file}")

    # Verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    train_files = list((target_dir / 'train').glob('*.jpg'))
    test_files = list((target_dir / 'test').glob('*.jpg'))

    print(f"\nFiles created:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Test:  {len(test_files)} images")

    print(f"\nLabels created:")
    print(f"  Binary: {len(combined_labels)} entries")
    print(f"  5D:     {len(combined_labels_5d)} entries")

    # Check balance
    print(f"\nClass Balance:")
    for split in ['train', 'test']:
        s = stats[split]
        if s['total'] > 0:
            danger_ratio = s['danger'] / s['total']
            safe_ratio = s['safe'] / s['total']

            if abs(danger_ratio - 0.5) < 0.1:
                balance = "✅ Balanced"
            else:
                balance = "⚠️  Imbalanced"

            print(f"  {split.capitalize():5s}: {balance} "
                  f"(Danger: {danger_ratio*100:.1f}%, Safe: {safe_ratio*100:.1f}%)")

    print("\n" + "=" * 60)
    print("✅ Temporal Split Creation Complete!")
    print("=" * 60)

    print(f"\nDataset ready at: {target_dir}")
    print(f"\nNext steps:")
    print(f"  1. Train models on train split (2022-06 to 09)")
    print(f"  2. Test models on test split (2022-10 to 11)")
    print(f"  3. Compare performance to random split to measure distribution shift impact")
    print()


if __name__ == '__main__':
    main()
