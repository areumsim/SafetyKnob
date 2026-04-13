#!/usr/bin/env python3
"""
Create random split EXCLUDING caution (safe/danger only)

This script creates train/val/test splits using ONLY safe and danger images.
Caution images are stored separately for analysis purposes.

Split strategy:
- Train/Val/Test: Safe + Danger only (70/15/15)
- Caution: Separate folder (caution_analysis/) - NOT used in training
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    random.seed(seed)

def get_safety_label_from_class(class_name):
    """Convert class folder name to binary safety label"""
    if class_name == 'safe':
        return 1
    elif class_name == 'danger':
        return 0
    else:
        raise ValueError(f"Unknown class: {class_name}")

def main():
    set_seed(42)

    source_dir = Path('data/raw/danger_al')
    output_dir = Path('data_caution_excluded')

    print("=" * 70)
    print("SPLIT CREATION EXCLUDING CAUTION (safe/danger only)")
    print("=" * 70)
    print(f"\nSource: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"\nSplit configuration:")
    print(f"  Train/Val/Test: Safe + Danger (70/15/15)")
    print(f"  Caution: Separate analysis folder")

    # Create output directories
    for split in ['train', 'val', 'test', 'caution_analysis']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # Collect safe and danger images
    print("\n" + "=" * 70)
    print("COLLECTING SAFE AND DANGER IMAGES")
    print("=" * 70)

    all_images = []
    all_labels = []

    for class_name in ['safe', 'danger']:
        class_path = source_dir / class_name
        if not class_path.exists():
            print(f"Warning: {class_name} folder not found, skipping...")
            continue

        images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
        safety_label = get_safety_label_from_class(class_name)

        print(f"\n{class_name}: {len(images)} images (label={safety_label})")

        for img_path in images:
            all_images.append({
                'path': img_path,
                'class': class_name,
                'label': safety_label
            })
            all_labels.append(safety_label)

    print(f"\nTotal safe+danger images: {len(all_images)}")

    # Stratified split: 70% train, 30% temp
    print("\n" + "=" * 70)
    print("SPLITTING SAFE+DANGER DATA (stratified)")
    print("=" * 70)

    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        all_images, all_labels,
        test_size=0.30,
        stratify=all_labels,
        random_state=42
    )

    # Split temp into val and test (50/50 of temp = 15/15 of total)
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels,
        test_size=0.50,
        stratify=temp_labels,
        random_state=42
    )

    print(f"\nTrain: {len(train_imgs)} images")
    print(f"Val:   {len(val_imgs)} images")
    print(f"Test:  {len(test_imgs)} images")

    # Copy images and create labels
    print("\n" + "=" * 70)
    print("COPYING SAFE+DANGER IMAGES")
    print("=" * 70)

    labels = {}
    stats = defaultdict(lambda: defaultdict(int))

    splits_data = {
        'train': train_imgs,
        'val': val_imgs,
        'test': test_imgs
    }

    for split_name, split_imgs in splits_data.items():
        print(f"\nProcessing {split_name}...")

        for img_info in split_imgs:
            img_path = img_info['path']
            safety_label = img_info['label']
            class_name = img_info['class']

            # Copy to split folder
            dst_path = output_dir / split_name / img_path.name

            # Handle potential filename conflicts
            if dst_path.exists():
                base_name = img_path.stem
                ext = img_path.suffix
                counter = 1
                while dst_path.exists():
                    dst_path = output_dir / split_name / f"{base_name}_{counter}{ext}"
                    counter += 1

            shutil.copy2(img_path, dst_path)

            # Add to labels
            labels[f"{split_name}/{dst_path.name}"] = {
                'overall_safety': safety_label,
                'class': class_name
            }

            stats[split_name]['total'] += 1
            stats[split_name]['unsafe'] += (1 - safety_label)
            stats[split_name]['safe'] += safety_label

    # Process caution images (separate analysis folder)
    print("\n" + "=" * 70)
    print("COPYING CAUTION IMAGES TO ANALYSIS FOLDER")
    print("=" * 70)

    caution_path = source_dir / 'caution'
    if caution_path.exists():
        caution_images = list(caution_path.glob('*.jpg')) + list(caution_path.glob('*.png'))
        print(f"\nCaution: {len(caution_images)} images (for analysis only)")

        for img_path in caution_images:
            dst_path = output_dir / 'caution_analysis' / img_path.name

            # Handle potential filename conflicts
            if dst_path.exists():
                base_name = img_path.stem
                ext = img_path.suffix
                counter = 1
                while dst_path.exists():
                    dst_path = output_dir / 'caution_analysis' / f"{base_name}_{counter}{ext}"
                    counter += 1

            shutil.copy2(img_path, dst_path)

            # Add to labels
            labels[f"caution_analysis/{dst_path.name}"] = {
                'overall_safety': -1,  # Special marker for caution
                'class': 'caution'
            }

            stats['caution_analysis']['total'] += 1
            stats['caution_analysis']['caution'] += 1

    # Save labels
    labels_file = output_dir / 'labels.json'
    print(f"\n" + "=" * 70)
    print("SAVING LABELS")
    print("=" * 70)
    print(f"\nSaving labels to {labels_file}...")

    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=2)

    print(f"✓ Saved {len(labels)} labels")

    # Print statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    for split in ['train', 'val', 'test']:
        total = stats[split]['total']
        safe = stats[split]['safe']
        unsafe = stats[split]['unsafe']
        if total > 0:
            print(f"\n{split.upper()}:")
            print(f"  Total:  {total} images")
            print(f"  Safe:   {safe} ({safe/total*100:.1f}%)")
            print(f"  Unsafe: {unsafe} ({unsafe/total*100:.1f}%)")

    if stats['caution_analysis']['total'] > 0:
        print(f"\nCAUTION ANALYSIS:")
        print(f"  Total:  {stats['caution_analysis']['total']} images")
        print(f"  (NOT used in training)")

    print("\n" + "=" * 70)
    print("SPLIT CREATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"  train/: {stats['train']['total']} images")
    print(f"  val/:   {stats['val']['total']} images")
    print(f"  test/:  {stats['test']['total']} images")
    print(f"  caution_analysis/: {stats['caution_analysis']['total']} images")
    print(f"  labels.json: {len(labels)} entries")

if __name__ == '__main__':
    main()
