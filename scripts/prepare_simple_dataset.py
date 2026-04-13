#!/usr/bin/env python3
"""
Simple Dataset Preparation from safe/danger directories

Uses existing safe/danger directory structure as labels.

Usage:
    python scripts/prepare_simple_dataset.py \
        --safe-dir data/safe/ \
        --danger-dir data/danger/ \
        --output data/processed/ \
        --train-ratio 0.7 \
        --val-ratio 0.15 \
        --test-ratio 0.15
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict
import random

def collect_labeled_images(safe_dir: str, danger_dir: str) -> Dict[str, List[str]]:
    """Collect images from safe and danger directories."""
    safe_path = Path(safe_dir)
    danger_path = Path(danger_dir)

    safe_images = []
    danger_images = []

    # Collect safe images
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        safe_images.extend([str(p) for p in safe_path.glob(ext)])

    # Collect danger images
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        danger_images.extend([str(p) for p in danger_path.glob(ext)])

    print(f"✓ Collected {len(safe_images)} safe images")
    print(f"✓ Collected {len(danger_images)} danger images")

    return {'safe': safe_images, 'danger': danger_images}


def generate_labels_from_directory(
    safe_images: List[str],
    danger_images: List[str],
    heuristic: str = 'random'
) -> Dict[str, Dict]:
    """
    Generate 5-dimensional labels from directory structure.

    Args:
        safe_images: List of safe image paths
        danger_images: List of danger image paths
        heuristic: 'random' or 'uniform' for 5-dim label generation

    Returns:
        Dict mapping filename to labels
    """
    labels = {}
    dimension_names = ['fall_hazard', 'collision_risk', 'equipment_hazard',
                      'environmental_risk', 'protective_gear']

    # Safe images: overall_safety=1, all dimensions=0 (no hazard)
    for img_path in safe_images:
        filename = Path(img_path).name
        labels[filename] = {
            'fall_hazard': 0,
            'collision_risk': 0,
            'equipment_hazard': 0,
            'environmental_risk': 0,
            'protective_gear': 1,  # Has protective gear (safe)
            'overall_safety': 1,
            'source': 'directory_safe'
        }

    # Danger images: overall_safety=0
    for img_path in danger_images:
        filename = Path(img_path).name

        if heuristic == 'random':
            # Randomly select 1-3 hazard dimensions
            num_hazards = random.randint(1, 3)
            hazard_dims = random.sample(dimension_names[:-1], num_hazards)  # Exclude protective_gear

            dim_labels = {dim: 1 if dim in hazard_dims else 0 for dim in dimension_names[:-1]}
            dim_labels['protective_gear'] = random.choice([0, 1])  # Random gear status

        elif heuristic == 'uniform':
            # Uniformly distribute hazards across danger images
            idx = len([k for k in labels.keys() if labels[k]['source'] == 'directory_danger'])
            primary_hazard = dimension_names[idx % len(dimension_names)]

            dim_labels = {dim: 1 if dim == primary_hazard else 0 for dim in dimension_names[:-1]}
            dim_labels['protective_gear'] = 0  # Danger usually means no gear

        labels[filename] = {
            **dim_labels,
            'overall_safety': 0,
            'source': 'directory_danger'
        }

    return labels


def split_dataset(
    labeled_images: Dict[str, List[str]],
    labels: Dict[str, Dict],
    output_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42
):
    """Split dataset into train/val/test with stratification."""
    random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_images = labeled_images['safe']
    danger_images = labeled_images['danger']

    # Shuffle
    random.shuffle(safe_images)
    random.shuffle(danger_images)

    # Calculate split sizes for each class
    def calculate_splits(images, train_r, val_r, test_r):
        n = len(images)
        train_n = int(n * train_r)
        val_n = int(n * val_r)
        test_n = n - train_n - val_n

        return {
            'train': images[:train_n],
            'val': images[train_n:train_n + val_n],
            'test': images[train_n + val_n:]
        }

    safe_splits = calculate_splits(safe_images, train_ratio, val_ratio, test_ratio)
    danger_splits = calculate_splits(danger_images, train_ratio, val_ratio, test_ratio)

    # Combine splits
    splits = {
        'train': safe_splits['train'] + danger_splits['train'],
        'val': safe_splits['val'] + danger_splits['val'],
        'test': safe_splits['test'] + danger_splits['test']
    }

    # Shuffle combined splits
    for split_name in splits:
        random.shuffle(splits[split_name])

    # Copy images and create labels
    split_labels = {}
    for split_name, images in splits.items():
        split_dir = output_path / split_name
        split_dir.mkdir(exist_ok=True)

        split_labels[split_name] = {}

        for img_path in images:
            filename = Path(img_path).name
            dest_path = split_dir / filename

            # Copy image
            shutil.copy2(img_path, dest_path)

            # Add label
            if filename in labels:
                split_labels[split_name][filename] = labels[filename]

        # Save split labels
        split_label_file = output_path / f'{split_name}_labels.json'
        with open(split_label_file, 'w') as f:
            json.dump(split_labels[split_name], f, indent=2)

        # Count safe/danger
        safe_count = sum(1 for label in split_labels[split_name].values() if label['overall_safety'] == 1)
        danger_count = len(split_labels[split_name]) - safe_count

        print(f"✓ {split_name.upper()}: {len(images)} images (safe: {safe_count}, danger: {danger_count})")

    # Save combined labels
    all_labels_file = output_path / 'labels.json'
    all_labels = {}
    for split_label_dict in split_labels.values():
        all_labels.update(split_label_dict)

    with open(all_labels_file, 'w') as f:
        json.dump(all_labels, f, indent=2)

    # Save dataset info
    dataset_info = {
        'total_images': sum(len(imgs) for imgs in splits.values()),
        'splits': {
            'train': {'count': len(splits['train'])},
            'val': {'count': len(splits['val'])},
            'test': {'count': len(splits['test'])}
        },
        'class_distribution': {
            'safe': len(safe_images),
            'danger': len(danger_images)
        },
        'dimensions': ['fall_hazard', 'collision_risk', 'equipment_hazard',
                      'environmental_risk', 'protective_gear'],
        'label_source': 'directory_structure'
    }

    info_file = output_path / 'dataset_info.json'
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    return dataset_info


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset from safe/danger directories')
    parser.add_argument('--safe-dir', type=str, default='data/safe/',
                       help='Directory containing safe images')
    parser.add_argument('--danger-dir', type=str, default='data/danger/',
                       help='Directory containing danger images')
    parser.add_argument('--output', type=str, default='data/processed/',
                       help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio')
    parser.add_argument('--heuristic', type=str, default='random',
                       choices=['random', 'uniform'],
                       help='Heuristic for 5-dim label generation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        print("❌ Error: train_ratio + val_ratio + test_ratio must equal 1.0")
        return

    print("=" * 60)
    print("Simple Dataset Preparation (safe/danger directories)")
    print("=" * 60)
    print(f"Safe directory: {args.safe_dir}")
    print(f"Danger directory: {args.danger_dir}")
    print(f"Output: {args.output}")
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Step 1: Collect images
    print("\n[1/3] Collecting images...")
    labeled_images = collect_labeled_images(args.safe_dir, args.danger_dir)

    total = len(labeled_images['safe']) + len(labeled_images['danger'])
    print(f"\nTotal images: {total}")

    if total == 0:
        print("\n❌ Error: No images found")
        return

    # Step 2: Generate labels
    print("\n[2/3] Generating labels from directory structure...")
    labels = generate_labels_from_directory(
        labeled_images['safe'],
        labeled_images['danger'],
        heuristic=args.heuristic
    )
    print(f"✓ Generated labels for {len(labels)} images")
    print(f"  5-dimensional labels using '{args.heuristic}' heuristic")

    # Step 3: Split dataset
    print("\n[3/3] Splitting dataset...")
    dataset_info = split_dataset(
        labeled_images,
        labels,
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        seed=args.seed
    )

    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print("=" * 60)
    print(f"Total images: {dataset_info['total_images']}")
    print(f"Train: {dataset_info['splits']['train']['count']}")
    print(f"Val: {dataset_info['splits']['val']['count']}")
    print(f"Test: {dataset_info['splits']['test']['count']}")
    print(f"\nSafe images: {dataset_info['class_distribution']['safe']}")
    print(f"Danger images: {dataset_info['class_distribution']['danger']}")
    print(f"\nOutput directory: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
