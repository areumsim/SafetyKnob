#!/usr/bin/env python3
"""
Dataset Preparation Script for SafetyKnob

This script prepares the AI Hub Construction Safety dataset for training:
1. Collects images from scenario directories (SO-XX)
2. Generates 5-dimensional safety labels (semi-supervised with user review)
3. Splits data into train/val/test sets by scenario
4. Creates labels.json file

Usage:
    python scripts/prepare_dataset.py \
        --scenarios-dir data/ \
        --output data/processed/ \
        --train-scenarios SO-35,SO-36,SO-37,SO-38,SO-41,SO-42,SO-43 \
        --val-scenarios SO-44,SO-45,SO-46 \
        --test-scenarios SO-47
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import random

def collect_scenario_images(scenarios_dir: str, scenario_names: List[str]) -> Dict[str, List[str]]:
    """
    Collect image paths from specified scenarios.

    Args:
        scenarios_dir: Base directory containing SO-XX folders
        scenario_names: List of scenario names (e.g., ['SO-35', 'SO-41'])

    Returns:
        Dict mapping scenario name to list of image paths
    """
    scenario_images = defaultdict(list)

    for scenario in scenario_names:
        scenario_path = Path(scenarios_dir) / scenario
        if not scenario_path.exists():
            print(f"⚠️  Warning: Scenario directory not found: {scenario_path}")
            continue

        # Collect all image files
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images = list(scenario_path.glob(ext))
            scenario_images[scenario].extend([str(p) for p in images])

        print(f"✓ Collected {len(scenario_images[scenario])} images from {scenario}")

    return dict(scenario_images)


def generate_labels(image_paths: List[str], label_mode: str = 'placeholder') -> Dict[str, Dict]:
    """
    Generate 5-dimensional safety labels for images.

    Args:
        image_paths: List of image file paths
        label_mode: 'placeholder' (random for now) or 'manual' (requires annotation tool)

    Returns:
        Dict mapping image filename to label dict
    """
    labels = {}
    dimension_names = ['fall_hazard', 'collision_risk', 'equipment_hazard',
                      'environmental_risk', 'protective_gear']

    print(f"\n⚠️  Generating {label_mode} labels...")
    print("⚠️  WARNING: Placeholder labels are RANDOM and for TESTING ONLY!")
    print("⚠️  For actual research, please use properly annotated labels.\n")

    for img_path in image_paths:
        filename = Path(img_path).name

        if label_mode == 'placeholder':
            # Generate random labels (FOR TESTING ONLY)
            # In production, this should read from annotation files
            dim_labels = {dim: random.randint(0, 1) for dim in dimension_names}

            # Overall safety: 0 if any dimension is 1 (unsafe), else 1 (safe)
            overall = 0 if any(dim_labels.values()) else 1

            labels[filename] = {
                **dim_labels,
                'overall_safety': overall,
                'source': 'placeholder_random',  # Mark as placeholder
                'scenario': extract_scenario_from_filename(filename)
            }
        else:
            # TODO: Implement manual annotation loading
            raise NotImplementedError("Manual label mode requires annotation tool integration")

    return labels


def extract_scenario_from_filename(filename: str) -> str:
    """Extract scenario name (SO-XX) from filename."""
    # Example filename: H-220713_G16_SO-35_001_0001.jpg
    parts = filename.split('_')
    for part in parts:
        if part.startswith('SO-'):
            return part
    return 'unknown'


def organize_dataset(
    scenario_images: Dict[str, List[str]],
    labels: Dict[str, Dict],
    output_dir: str,
    train_scenarios: List[str],
    val_scenarios: List[str],
    test_scenarios: List[str]
):
    """
    Organize images into train/val/test directories and save labels.

    Args:
        scenario_images: Dict mapping scenario to image paths
        labels: Dict mapping filename to labels
        output_dir: Output directory path
        train/val/test_scenarios: Scenario lists for each split
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    splits = {
        'train': train_scenarios,
        'val': val_scenarios,
        'test': test_scenarios
    }

    split_labels = {split: {} for split in splits}
    split_counts = {split: 0 for split in splits}

    for split_name, scenarios in splits.items():
        split_dir = output_path / split_name
        split_dir.mkdir(exist_ok=True)

        for scenario in scenarios:
            if scenario not in scenario_images:
                print(f"⚠️  Warning: Scenario {scenario} not found in collected images")
                continue

            images = scenario_images[scenario]

            for img_path in images:
                filename = Path(img_path).name

                # Copy image to split directory
                dest_path = split_dir / filename
                shutil.copy2(img_path, dest_path)

                # Add to split labels
                if filename in labels:
                    split_labels[split_name][filename] = labels[filename]
                    split_counts[split_name] += 1

        print(f"✓ {split_name.upper()}: {split_counts[split_name]} images from scenarios {scenarios}")

    # Save labels for each split
    for split_name, split_label_dict in split_labels.items():
        labels_file = output_path / f'{split_name}_labels.json'
        with open(labels_file, 'w') as f:
            json.dump(split_label_dict, f, indent=2)
        print(f"✓ Saved {split_name} labels to {labels_file}")

    # Save combined labels
    all_labels_file = output_path / 'labels.json'
    with open(all_labels_file, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"✓ Saved combined labels to {all_labels_file}")

    # Save dataset info
    dataset_info = {
        'total_images': sum(split_counts.values()),
        'splits': {
            'train': {'count': split_counts['train'], 'scenarios': train_scenarios},
            'val': {'count': split_counts['val'], 'scenarios': val_scenarios},
            'test': {'count': split_counts['test'], 'scenarios': test_scenarios}
        },
        'dimensions': ['fall_hazard', 'collision_risk', 'equipment_hazard',
                      'environmental_risk', 'protective_gear'],
        'label_mode': 'placeholder' if 'placeholder_random' in next(iter(labels.values())).get('source', '') else 'manual'
    }

    info_file = output_path / 'dataset_info.json'
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    print(f"✓ Saved dataset info to {info_file}")

    return dataset_info


def main():
    parser = argparse.ArgumentParser(description='Prepare SafetyKnob dataset')
    parser.add_argument('--scenarios-dir', type=str, default='data/',
                       help='Directory containing SO-XX scenario folders')
    parser.add_argument('--output', type=str, default='data/processed/',
                       help='Output directory for processed dataset')
    parser.add_argument('--train-scenarios', type=str,
                       default='SO-35,SO-36,SO-37,SO-38,SO-41,SO-42,SO-43',
                       help='Comma-separated list of training scenarios')
    parser.add_argument('--val-scenarios', type=str,
                       default='SO-44,SO-45,SO-46',
                       help='Comma-separated list of validation scenarios')
    parser.add_argument('--test-scenarios', type=str,
                       default='SO-47',
                       help='Comma-separated list of test scenarios')
    parser.add_argument('--label-mode', type=str, default='placeholder',
                       choices=['placeholder', 'manual'],
                       help='Label generation mode')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Parse scenario lists
    train_scenarios = [s.strip() for s in args.train_scenarios.split(',')]
    val_scenarios = [s.strip() for s in args.val_scenarios.split(',')]
    test_scenarios = [s.strip() for s in args.test_scenarios.split(',')]

    all_scenarios = train_scenarios + val_scenarios + test_scenarios

    print("=" * 60)
    print("SafetyKnob Dataset Preparation")
    print("=" * 60)
    print(f"Source: {args.scenarios_dir}")
    print(f"Output: {args.output}")
    print(f"Train scenarios ({len(train_scenarios)}): {', '.join(train_scenarios)}")
    print(f"Val scenarios ({len(val_scenarios)}): {', '.join(val_scenarios)}")
    print(f"Test scenarios ({len(test_scenarios)}): {', '.join(test_scenarios)}")
    print("=" * 60)

    # Step 1: Collect images
    print("\n[1/3] Collecting images from scenarios...")
    scenario_images = collect_scenario_images(args.scenarios_dir, all_scenarios)

    total_images = sum(len(imgs) for imgs in scenario_images.values())
    print(f"\nTotal images collected: {total_images}")

    if total_images == 0:
        print("\n❌ Error: No images found. Please check scenarios-dir path.")
        return

    # Step 2: Generate labels
    print("\n[2/3] Generating labels...")
    all_image_paths = []
    for imgs in scenario_images.values():
        all_image_paths.extend(imgs)

    labels = generate_labels(all_image_paths, label_mode=args.label_mode)
    print(f"✓ Generated labels for {len(labels)} images")

    # Step 3: Organize dataset
    print("\n[3/3] Organizing dataset into train/val/test...")
    dataset_info = organize_dataset(
        scenario_images,
        labels,
        args.output,
        train_scenarios,
        val_scenarios,
        test_scenarios
    )

    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print("=" * 60)
    print(f"Total images: {dataset_info['total_images']}")
    print(f"Train: {dataset_info['splits']['train']['count']} images")
    print(f"Val: {dataset_info['splits']['val']['count']} images")
    print(f"Test: {dataset_info['splits']['test']['count']} images")
    print(f"\nOutput directory: {args.output}")
    print(f"Labels file: {args.output}/labels.json")
    print("\n⚠️  IMPORTANT: If using placeholder labels, replace with actual annotations!")
    print("=" * 60)


if __name__ == '__main__':
    main()
