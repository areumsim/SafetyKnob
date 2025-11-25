"""
Prepare dataset from 3-class structure (danger/safe/caution)
Maps to binary classification + generates 5-dimensional labels
"""

import json
import shutil
import random
from pathlib import Path
from collections import defaultdict
import argparse
from typing import Dict, List, Tuple

def count_images(base_dir: Path) -> Dict[str, int]:
    """Count images in each class folder"""
    counts = {}
    for class_name in ['danger', 'safe', 'caution']:
        class_dir = base_dir / class_name
        if class_dir.exists():
            counts[class_name] = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
    return counts

def map_to_binary(class_name: str) -> int:
    """Map 3-class to binary

    Args:
        class_name: 'danger', 'safe', or 'caution'

    Returns:
        0 (unsafe) or 1 (safe)
    """
    if class_name == 'safe':
        return 1  # Safe
    else:
        return 0  # Unsafe (danger or caution)

def generate_5d_labels_heuristic(class_name: str) -> Dict[str, float]:
    """Generate 5-dimensional labels using heuristics

    For now, use simple heuristics:
    - danger: high risk on all dimensions
    - caution: medium risk
    - safe: low risk

    Args:
        class_name: 'danger', 'safe', or 'caution'

    Returns:
        Dict with 5 dimension scores (0=high risk, 1=safe)
    """
    if class_name == 'danger':
        # High risk on all dimensions
        return {
            'fall_hazard': random.uniform(0.1, 0.4),  # High risk
            'collision_risk': random.uniform(0.1, 0.4),
            'equipment_hazard': random.uniform(0.1, 0.4),
            'environmental_risk': random.uniform(0.1, 0.4),
            'protective_gear': random.uniform(0.1, 0.4)
        }
    elif class_name == 'caution':
        # Medium risk
        return {
            'fall_hazard': random.uniform(0.4, 0.6),
            'collision_risk': random.uniform(0.4, 0.6),
            'equipment_hazard': random.uniform(0.4, 0.6),
            'environmental_risk': random.uniform(0.4, 0.6),
            'protective_gear': random.uniform(0.4, 0.6)
        }
    else:  # safe
        # Low risk
        return {
            'fall_hazard': random.uniform(0.7, 0.9),  # Low risk
            'collision_risk': random.uniform(0.7, 0.9),
            'equipment_hazard': random.uniform(0.7, 0.9),
            'environmental_risk': random.uniform(0.7, 0.9),
            'protective_gear': random.uniform(0.7, 0.9)
        }

def collect_all_images(source_dir: Path) -> List[Tuple[Path, str]]:
    """Collect all images with their class labels

    Returns:
        List of (image_path, class_name) tuples
    """
    images = []

    for class_name in ['danger', 'safe', 'caution']:
        class_dir = source_dir / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found, skipping")
            continue

        for img_path in class_dir.glob('*.jpg'):
            images.append((img_path, class_name))
        for img_path in class_dir.glob('*.png'):
            images.append((img_path, class_name))

    return images

def split_data(images: List[Tuple[Path, str]],
               train_ratio: float = 0.7,
               val_ratio: float = 0.15,
               test_ratio: float = 0.15,
               seed: int = 42) -> Tuple[List, List, List]:
    """Split data into train/val/test sets

    Args:
        images: List of (image_path, class_name)
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_images, val_images, test_images)
    """
    random.seed(seed)

    # Stratified split by class
    class_images = defaultdict(list)
    for img_path, class_name in images:
        class_images[class_name].append(img_path)

    train_images = []
    val_images = []
    test_images = []

    for class_name, class_imgs in class_images.items():
        random.shuffle(class_imgs)

        n = len(class_imgs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_class = [(img, class_name) for img in class_imgs[:n_train]]
        val_class = [(img, class_name) for img in class_imgs[n_train:n_train + n_val]]
        test_class = [(img, class_name) for img in class_imgs[n_train + n_val:]]

        train_images.extend(train_class)
        val_images.extend(val_class)
        test_images.extend(test_class)

        print(f"{class_name}: {len(train_class)} train, {len(val_class)} val, {len(test_class)} test")

    return train_images, val_images, test_images

def copy_images_and_create_labels(images: List[Tuple[Path, str]],
                                  output_dir: Path,
                                  split_name: str) -> Dict:
    """Copy images to output directory and create labels

    Args:
        images: List of (image_path, class_name)
        output_dir: Output directory
        split_name: 'train', 'val', or 'test'

    Returns:
        Dict of labels for this split
    """
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    labels = {}

    for img_path, class_name in images:
        # Copy image
        dest_path = split_dir / img_path.name
        shutil.copy2(img_path, dest_path)

        # Create label
        binary_label = map_to_binary(class_name)
        dim_labels = generate_5d_labels_heuristic(class_name)

        labels[img_path.name] = {
            'overall_safety': binary_label,
            'class': class_name,
            **dim_labels
        }

    print(f"  Copied {len(images)} images to {split_dir}")
    return labels

def main():
    parser = argparse.ArgumentParser(description='Prepare 3-class data for SafetyKnob')
    parser.add_argument('--source', type=str,
                       default='/workspace/data1/arsim/danger_al',
                       help='Source directory with danger/safe/caution folders')
    parser.add_argument('--output', type=str,
                       default='/workspace/arsim/EmoKnob/data',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                       help='Test set ratio')

    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    print("=" * 60)
    print("SAFETYKNOB DATA PREPARATION (3-CLASS)")
    print("=" * 60)

    # Count images
    print("\n1. Counting images...")
    counts = count_images(source_dir)
    for class_name, count in counts.items():
        print(f"  {class_name}: {count} images")
    total = sum(counts.values())
    print(f"  TOTAL: {total} images")

    # Collect all images
    print("\n2. Collecting images...")
    all_images = collect_all_images(source_dir)
    print(f"  Collected {len(all_images)} images")

    # Split data
    print("\n3. Splitting data...")
    train_images, val_images, test_images = split_data(
        all_images,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    print(f"\n  Total: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

    # Copy images and create labels
    print("\n4. Copying images and creating labels...")

    all_labels = {}

    print("  Processing train set...")
    train_labels = copy_images_and_create_labels(train_images, output_dir, 'train')
    all_labels.update({f"train/{k}": v for k, v in train_labels.items()})

    print("  Processing val set...")
    val_labels = copy_images_and_create_labels(val_images, output_dir, 'val')
    all_labels.update({f"val/{k}": v for k, v in val_labels.items()})

    print("  Processing test set...")
    test_labels = copy_images_and_create_labels(test_images, output_dir, 'test')
    all_labels.update({f"test/{k}": v for k, v in test_labels.items()})

    # Save labels
    print("\n5. Saving labels...")
    labels_file = output_dir / 'labels.json'
    with open(labels_file, 'w') as f:
        json.dump(all_labels, f, indent=2)
    print(f"  Saved labels to {labels_file}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"\nData split:")
    print(f"  Train: {len(train_images)} images ({len(train_images)/total*100:.1f}%)")
    print(f"  Val:   {len(val_images)} images ({len(val_images)/total*100:.1f}%)")
    print(f"  Test:  {len(test_images)} images ({len(test_images)/total*100:.1f}%)")
    print(f"\nLabels:")
    print(f"  Binary: overall_safety (0=unsafe, 1=safe)")
    print(f"  5D: fall_hazard, collision_risk, equipment_hazard, environmental_risk, protective_gear")
    print(f"  Method: Heuristic based on original class (danger/safe/caution)")
    print("\n⚠️  NOTE: 5D labels are HEURISTIC approximations.")
    print("   For production use, consider manual annotation or CLIP zero-shot labeling.")
    print("=" * 60)

if __name__ == '__main__':
    main()
