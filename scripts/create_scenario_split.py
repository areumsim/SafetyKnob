#!/usr/bin/env python3
"""
Create sequence-level split for AI Hub data (safe/danger/caution)

CRITICAL FIX: Previous version split at individual frame level, causing
98.1% of test sequences to also appear in training (data leakage).

This version groups frames by video sequence ID and ensures all frames
from the same sequence go to the same split.

Filename pattern: H-{YYMMDD}_{AccidentCode}_{ScenarioID}_{SequenceNum}_{FrameID}.jpg
Sequence key:     H-{YYMMDD}_{AccidentCode}_{ScenarioID}_{SequenceNum}

Split strategy (sequence-level):
- Train: 70% of sequences
- Val: 15% of sequences
- Test: 15% of sequences
"""

import json
import re
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
    elif class_name in ['danger', 'caution']:
        return 0
    else:
        raise ValueError(f"Unknown class: {class_name}")


def extract_sequence_key(filename):
    """
    Extract sequence key from filename.

    Pattern: H-{YYMMDD}_{Code}_{ScenarioID}_{SeqNum}_{FrameID}.jpg
    Example: H-220607_B16_Y-14_001_0001.jpg -> H-220607_B16_Y-14_001

    All frames sharing the same sequence key are from the same CCTV clip
    and must go to the same split to avoid data leakage.
    """
    # Match everything up to the last _NNNN before .jpg
    match = re.match(r'^(H-\d{6}_[A-Z]\d{2}_[YN]-\d{2}_\d{3})_\d+\.jpg$', filename)
    if match:
        return match.group(1)

    # Fallback: try splitting by underscore and removing last part (frame ID)
    parts = filename.rsplit('.', 1)[0].rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]

    # If no pattern match, use filename itself as key (1 frame per "sequence")
    return filename.rsplit('.', 1)[0]


def main():
    set_seed(42)

    # Collect all images from existing split directories
    source_dir = Path('data_scenario')
    output_dir = Path('data_scenario_v2')

    print("=" * 70)
    print("SEQUENCE-LEVEL SPLIT CREATION (fixing data leakage)")
    print("=" * 70)

    # Load existing labels to get class info
    labels_file = source_dir / 'labels.json'
    with open(labels_file, 'r') as f:
        existing_labels = json.load(f)

    # Collect all images from all existing splits
    print("\nCollecting images from existing splits...")
    all_images = []
    class_lookup = {}

    for split in ['train', 'val', 'test']:
        split_dir = source_dir / split
        if not split_dir.exists():
            continue
        for img_path in sorted(split_dir.glob('*.jpg')):
            label_key = f"{split}/{img_path.name}"
            if label_key in existing_labels:
                class_name = existing_labels[label_key]['class']
            else:
                # Try to infer from filename: Y- prefix = safe, N- prefix = danger
                if '_Y-' in img_path.name:
                    class_name = 'safe'
                elif '_N-' in img_path.name:
                    class_name = 'danger'
                else:
                    print(f"  Warning: no label for {label_key}, skipping")
                    continue

            safety_label = get_safety_label_from_class(class_name)
            all_images.append({
                'path': img_path,
                'class': class_name,
                'label': safety_label,
                'filename': img_path.name,
            })
            class_lookup[img_path.name] = class_name

    print(f"Total images collected: {len(all_images)}")

    # Group images by sequence key
    print("\n" + "=" * 70)
    print("GROUPING BY VIDEO SEQUENCE")
    print("=" * 70)

    sequences = defaultdict(list)
    for img in all_images:
        seq_key = extract_sequence_key(img['filename'])
        sequences[seq_key].append(img)

    print(f"Total unique sequences: {len(sequences)}")
    frames_per_seq = [len(v) for v in sequences.values()]
    print(f"Frames per sequence: min={min(frames_per_seq)}, "
          f"max={max(frames_per_seq)}, "
          f"mean={sum(frames_per_seq)/len(frames_per_seq):.1f}, "
          f"median={sorted(frames_per_seq)[len(frames_per_seq)//2]}")

    # Determine majority class for each sequence (for stratification)
    seq_keys = sorted(sequences.keys())
    seq_classes = []
    for key in seq_keys:
        imgs = sequences[key]
        # All frames in a sequence should have the same class
        classes = set(img['class'] for img in imgs)
        if len(classes) > 1:
            print(f"  Warning: mixed classes in sequence {key}: {classes}")
        # Use majority class
        class_counts = defaultdict(int)
        for img in imgs:
            class_counts[img['class']] += 1
        majority_class = max(class_counts, key=class_counts.get)
        seq_classes.append(majority_class)

    print(f"\nSequence class distribution:")
    class_seq_counts = defaultdict(int)
    for c in seq_classes:
        class_seq_counts[c] += 1
    for c in sorted(class_seq_counts):
        print(f"  {c}: {class_seq_counts[c]} sequences")

    # Stratified split at SEQUENCE level (70/15/15)
    print("\n" + "=" * 70)
    print("SPLITTING AT SEQUENCE LEVEL (stratified)")
    print("=" * 70)

    train_seqs, temp_seqs, train_cls, temp_cls = train_test_split(
        seq_keys, seq_classes,
        test_size=0.30,
        stratify=seq_classes,
        random_state=42
    )

    val_seqs, test_seqs, _, _ = train_test_split(
        temp_seqs, temp_cls,
        test_size=0.50,
        stratify=temp_cls,
        random_state=42
    )

    train_seqs = set(train_seqs)
    val_seqs = set(val_seqs)
    test_seqs = set(test_seqs)

    print(f"\nSequence counts: train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}")

    # Verify zero overlap
    assert len(train_seqs & val_seqs) == 0, "Train/val sequence overlap!"
    assert len(train_seqs & test_seqs) == 0, "Train/test sequence overlap!"
    assert len(val_seqs & test_seqs) == 0, "Val/test sequence overlap!"
    print("✓ Zero sequence overlap between splits (verified)")

    # Assign images to splits based on their sequence
    print("\n" + "=" * 70)
    print("COPYING IMAGES AND CREATING LABELS")
    print("=" * 70)

    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    labels = {}
    stats = defaultdict(lambda: defaultdict(int))

    for img in all_images:
        seq_key = extract_sequence_key(img['filename'])

        if seq_key in train_seqs:
            split_name = 'train'
        elif seq_key in val_seqs:
            split_name = 'val'
        elif seq_key in test_seqs:
            split_name = 'test'
        else:
            print(f"  Warning: orphan sequence {seq_key}")
            continue

        src_path = img['path']
        dst_path = output_dir / split_name / img['filename']

        # Handle potential filename conflicts
        if dst_path.exists():
            base_name = Path(img['filename']).stem
            ext = Path(img['filename']).suffix
            counter = 1
            while dst_path.exists():
                dst_path = output_dir / split_name / f"{base_name}_{counter}{ext}"
                counter += 1

        shutil.copy2(src_path, dst_path)

        labels[f"{split_name}/{dst_path.name}"] = {
            'overall_safety': img['label'],
            'class': img['class']
        }

        stats[split_name]['total'] += 1
        stats[split_name]['unsafe'] += (1 - img['label'])
        stats[split_name]['safe'] += img['label']

    # Save labels
    labels_file = output_dir / 'labels.json'
    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"\n✓ Saved {len(labels)} labels to {labels_file}")

    # Print statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS (sequence-level split)")
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

    # Final verification: check no sequence overlap
    print("\n" + "=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    for split_a, split_b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        files_a = set(f.name for f in (output_dir / split_a).glob('*.jpg'))
        files_b = set(f.name for f in (output_dir / split_b).glob('*.jpg'))

        seqs_a = set(extract_sequence_key(f) for f in files_a)
        seqs_b = set(extract_sequence_key(f) for f in files_b)

        overlap = seqs_a & seqs_b
        print(f"  {split_a}/{split_b} sequence overlap: {len(overlap)} (must be 0)")
        assert len(overlap) == 0, f"LEAK DETECTED: {split_a}/{split_b} share {len(overlap)} sequences!"

    print("\n✓ ALL CHECKS PASSED - No data leakage detected")
    print(f"\nOutput directory: {output_dir}")
    print(f"  train/: {stats['train']['total']} images")
    print(f"  val/:   {stats['val']['total']} images")
    print(f"  test/:  {stats['test']['total']} images")
    print(f"  labels.json: {len(labels)} entries")


if __name__ == '__main__':
    main()
