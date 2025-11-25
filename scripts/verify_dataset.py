#!/usr/bin/env python3
"""
Dataset Verification Script

Verifies the integrity of the prepared dataset.

Usage:
    python scripts/verify_dataset.py --data-dir data/processed/
"""

import argparse
import json
from pathlib import Path
from collections import Counter

def verify_dataset(data_dir: str):
    """Verify dataset integrity."""
    data_path = Path(data_dir)

    print("=" * 60)
    print("SafetyKnob Dataset Verification")
    print("=" * 60)

    # Check directory structure
    required_dirs = ['train', 'val', 'test']
    for split in required_dirs:
        split_dir = data_path / split
        if not split_dir.exists():
            print(f"❌ Missing directory: {split_dir}")
            return False
        print(f"✓ Found {split} directory")

    # Check labels file
    labels_file = data_path / 'labels.json'
    if not labels_file.exists():
        print(f"❌ Missing labels.json")
        return False

    with open(labels_file) as f:
        labels = json.load(f)

    print(f"✓ Found labels.json with {len(labels)} entries")

    # Count images per split
    split_counts = {}
    for split in required_dirs:
        split_dir = data_path / split
        images = list(split_dir.glob('*.jpg')) + list(split_dir.glob('*.png'))
        split_counts[split] = len(images)
        print(f"✓ {split.upper()}: {len(images)} images")

    total_images = sum(split_counts.values())
    print(f"\nTotal images: {total_images}")

    # Check label dimensions
    if labels:
        sample_label = next(iter(labels.values()))
        required_dims = ['fall_hazard', 'collision_risk', 'equipment_hazard',
                        'environmental_risk', 'protective_gear', 'overall_safety']

        missing_dims = [dim for dim in required_dims if dim not in sample_label]
        if missing_dims:
            print(f"❌ Missing dimensions: {missing_dims}")
            return False

        print(f"✓ All 5+1 dimensions present: {', '.join(required_dims[:5])}")

        # Count label distribution
        overall_dist = Counter(label.get('overall_safety') for label in labels.values())
        print(f"\nLabel distribution:")
        print(f"  Safe (1): {overall_dist.get(1, 0)} images")
        print(f"  Unsafe (0): {overall_dist.get(0, 0)} images")

    print("\n" + "=" * 60)
    print("✅ Dataset verification passed!")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description='Verify SafetyKnob dataset')
    parser.add_argument('--data-dir', type=str, default='data/processed/',
                       help='Processed dataset directory')
    args = parser.parse_args()

    success = verify_dataset(args.data_dir)
    exit(0 if success else 1)


if __name__ == '__main__':
    main()
