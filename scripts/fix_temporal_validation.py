"""
Quick fix: Add validation split to temporal data

Takes 15% of training data randomly as validation set.
"""

import json
import random
import shutil
from pathlib import Path

def main():
    print("=" * 60)
    print("Adding Validation Split to Temporal Data")
    print("=" * 60)

    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data_temporal'

    # Create val directory
    (data_dir / 'val').mkdir(exist_ok=True)

    # Load labels
    labels_file = data_dir / 'labels.json'
    with open(labels_file, 'r') as f:
        all_labels = json.load(f)

    labels_5d_file = data_dir / 'labels_5d.json'
    with open(labels_5d_file, 'r') as f:
        all_labels_5d = json.load(f)

    # Get train images
    train_images = [k for k in all_labels.keys() if k.startswith('train/')]

    print(f"\nOriginal train images: {len(train_images)}")

    # Randomly select 15% for validation
    random.seed(42)
    val_count = int(len(train_images) * 0.15)
    val_images = random.sample(train_images, val_count)

    print(f"Moving {val_count} images to validation set...")

    # Move files and update labels
    new_labels = {}
    new_labels_5d = {}

    for img_path in all_labels.keys():
        filename = img_path.split('/')[-1]

        if img_path in val_images:
            # Move to val
            source = data_dir / 'train' / filename
            target = data_dir / 'val' / filename

            if source.exists():
                shutil.move(source, target)

            new_path = f"val/{filename}"
            new_labels[new_path] = all_labels[img_path]

            if img_path in all_labels_5d:
                new_labels_5d[new_path] = all_labels_5d[img_path]

        else:
            # Keep as is
            new_labels[img_path] = all_labels[img_path]

            if img_path in all_labels_5d:
                new_labels_5d[img_path] = all_labels_5d[img_path]

    # Save updated labels
    with open(labels_file, 'w') as f:
        json.dump(new_labels, f, indent=2)

    with open(labels_5d_file, 'w') as f:
        json.dump(new_labels_5d, f, indent=2)

    # Print stats
    train_count = len([k for k in new_labels if k.startswith('train/')])
    val_count = len([k for k in new_labels if k.startswith('val/')])
    test_count = len([k for k in new_labels if k.startswith('test/')])

    print("\nUpdated split:")
    print(f"  Train: {train_count}")
    print(f"  Val:   {val_count}")
    print(f"  Test:  {test_count}")

    print("\n✅ Validation split added!")

if __name__ == '__main__':
    main()
