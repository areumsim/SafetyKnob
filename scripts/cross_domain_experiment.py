#!/usr/bin/env python3
"""
Cross-Domain Safety Classification Experiment

Tests whether foundation model probes trained on AI Hub construction safety data
generalize to the Construction-PPE dataset (different domain, different labeling).

Experiments:
  1. Zero-shot transfer: Apply scenario-trained probe directly to PPE dataset
  2. Fine-tune on PPE: Train probe on PPE train, evaluate on PPE test
  3. Combined: Pre-train on AI Hub, fine-tune on PPE

Usage:
    python scripts/cross_domain_experiment.py \
        --model siglip \
        --ppe-data data_construction_ppe \
        --source-embeddings embeddings/scenario_v2/siglip \
        --output results/cross_domain/siglip
"""

import argparse
import json
import re
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.train_from_embeddings import BinarySafetyClassifier, load_split, set_seed


# PPE violation classes
VIOLATION_CLASSES = {5, 7, 8, 9, 10}  # none, no_helmet, no_goggle, no_gloves, no_boots


def get_embedder(model_name, device):
    """Load foundation model for embedding extraction."""
    if model_name == 'siglip':
        from transformers import SiglipVisionModel, SiglipImageProcessor
        model = SiglipVisionModel.from_pretrained('google/siglip-so400m-patch14-384').to(device)
        processor = SiglipImageProcessor.from_pretrained('google/siglip-so400m-patch14-384')
        model.eval()
        return model, processor, 1152
    elif model_name == 'clip':
        from transformers import CLIPVisionModel, CLIPImageProcessor
        model = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
        processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14')
        model.eval()
        return model, processor, 1024
    elif model_name == 'dinov2':
        from transformers import Dinov2Model, AutoImageProcessor
        model = Dinov2Model.from_pretrained('facebook/dinov2-large').to(device)
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
        model.eval()
        return model, processor, 1024
    else:
        raise ValueError(f"Unknown model: {model_name}")


def extract_ppe_embeddings(model_name, ppe_data_dir, device, output_dir):
    """Extract embeddings from Construction-PPE images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    if (output_dir / 'train_embeddings.pt').exists():
        print(f"  Using cached embeddings from {output_dir}")
        return

    model, processor, emb_dim = get_embedder(model_name, device)
    ppe_dir = Path(ppe_data_dir)

    for split in ['train', 'val', 'test']:
        img_dir = ppe_dir / 'images' / split
        label_dir = ppe_dir / 'labels' / split

        embeddings = []
        labels = {}
        filenames = []

        img_files = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
                          + list(img_dir.glob('*.jpeg')))

        print(f"  Extracting {split}: {len(img_files)} images...", end=' ', flush=True)

        for img_path in img_files:
            # Load and embed image
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                continue

            with torch.no_grad():
                if model_name == 'siglip':
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    emb = outputs.pooler_output.cpu().squeeze()
                elif model_name == 'clip':
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    emb = outputs.pooler_output.cpu().squeeze()
                elif model_name == 'dinov2':
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    emb = outputs.last_hidden_state[:, 0].cpu().squeeze()

            # Get binary label from detection annotations
            label_file = label_dir / (img_path.stem + '.txt')
            if label_file.exists():
                with open(label_file) as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                classes = {int(l.split()[0]) for l in lines}
                is_unsafe = bool(classes & VIOLATION_CLASSES)
                safety_label = 0 if is_unsafe else 1
            else:
                safety_label = 1  # No annotations = assume safe

            fname = img_path.name
            embeddings.append(emb)
            labels[fname] = {'overall_safety': safety_label, 'class': 'safe' if safety_label else 'danger'}
            filenames.append(fname)

        emb_tensor = torch.stack(embeddings)
        print(f"{len(embeddings)} embedded (dim={emb_tensor.shape[1]})")

        torch.save({'embeddings': emb_tensor, 'filenames': filenames, 'embedding_dim': emb_tensor.shape[1],
                    'model': model_name}, output_dir / f'{split}_embeddings.pt')
        torch.save({'labels': labels, 'filenames': filenames}, output_dir / f'{split}_labels.pt')

    del model
    torch.cuda.empty_cache()


def evaluate(model, emb, labels, device):
    """Evaluate model on embeddings."""
    model.eval()
    with torch.no_grad():
        scores = model(emb.to(device)).cpu().numpy()
    preds = (scores > 0.5).astype(int)
    targets = (labels.numpy() > 0.5).astype(int)

    return {
        'f1': float(f1_score(targets, preds, zero_division=0)),
        'accuracy': float(accuracy_score(targets, preds)),
        'precision': float(precision_score(targets, preds, zero_division=0)),
        'recall': float(recall_score(targets, preds, zero_division=0)),
        'auc_roc': float(roc_auc_score(targets, scores)) if len(set(targets.tolist())) > 1 else 0.0,
    }


def train_probe(train_emb, train_labels, val_emb, val_labels, embedding_dim,
                device, epochs=50, lr=1e-3, pretrained_state=None):
    """Train a probe, optionally starting from pretrained weights."""
    model = BinarySafetyClassifier(embedding_dim, hidden_dim=512, probe_depth='2layer').to(device)

    if pretrained_state:
        model.load_state_dict(pretrained_state)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(train_emb, train_labels)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    best_val_f1 = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for emb_batch, label_batch in train_loader:
            emb_batch, label_batch = emb_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(emb_batch), label_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(val_emb.to(device)).cpu()
            val_preds = (val_out > 0.5).numpy().astype(int)
            val_f1 = f1_score(val_labels.numpy() > 0.5, val_preds, zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_f1


def main():
    parser = argparse.ArgumentParser(description='Cross-domain safety classification')
    parser.add_argument('--model', type=str, default='siglip',
                       choices=['siglip', 'clip', 'dinov2'])
    parser.add_argument('--ppe-data', type=str, default='data_construction_ppe')
    parser.add_argument('--source-embeddings', type=str, default=None,
                       help='Pre-extracted source (AI Hub) embeddings')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--seeds', type=str, default='42,123,456,789,2024')
    parser.add_argument('--epochs', type=int, default=50)

    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"CROSS-DOMAIN SAFETY CLASSIFICATION")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Source: AI Hub (scenario_v2)")
    print(f"Target: Construction-PPE")

    # Step 1: Extract PPE embeddings
    ppe_emb_dir = Path(f'embeddings/ppe/{args.model}')
    print(f"\n--- Extracting PPE embeddings ---")
    extract_ppe_embeddings(args.model, args.ppe_data, device, ppe_emb_dir)

    # Load PPE embeddings
    ppe_train_emb, ppe_train_labels, _ = load_split(ppe_emb_dir, 'train')
    ppe_val_emb, ppe_val_labels, _ = load_split(ppe_emb_dir, 'val')
    ppe_test_emb, ppe_test_labels, _ = load_split(ppe_emb_dir, 'test')

    n_safe = int(ppe_test_labels.sum())
    n_unsafe = len(ppe_test_labels) - n_safe
    print(f"\nPPE test: {len(ppe_test_labels)} images (safe={n_safe}, unsafe={n_unsafe})")

    # Load source embeddings (AI Hub)
    if args.source_embeddings:
        source_dir = Path(args.source_embeddings)
    else:
        source_dir = Path(f'embeddings/scenario_v2/{args.model}')

    src_train_emb, src_train_labels, _ = load_split(source_dir, 'train')
    src_val_emb, src_val_labels, _ = load_split(source_dir, 'val')
    src_test_emb, src_test_labels, _ = load_split(source_dir, 'test')
    print(f"Source train: {len(src_train_emb)}, Source test: {len(src_test_emb)}")

    embedding_dim = src_train_emb.shape[1]

    results = {}

    # ===== Experiment 1: Zero-shot transfer (train on AI Hub, test on PPE) =====
    print(f"\n{'='*60}")
    print(f"EXP 1: Zero-shot Transfer (AI Hub → PPE)")
    print(f"{'='*60}")

    zs_f1s = []
    for seed in seeds:
        set_seed(seed)
        model, _ = train_probe(src_train_emb, src_train_labels,
                               src_val_emb, src_val_labels,
                               embedding_dim, device, epochs=args.epochs)
        # Test on source
        src_metrics = evaluate(model, src_test_emb, src_test_labels, device)
        # Test on PPE (zero-shot)
        ppe_metrics = evaluate(model, ppe_test_emb, ppe_test_labels, device)
        zs_f1s.append(ppe_metrics['f1'] * 100)
        print(f"  Seed {seed}: Source F1={src_metrics['f1']*100:.2f}%, PPE F1={ppe_metrics['f1']*100:.2f}%")

    results['zero_shot'] = {
        'description': 'Train on AI Hub, test on Construction-PPE (no adaptation)',
        'ppe_f1_mean': float(np.mean(zs_f1s)),
        'ppe_f1_std': float(np.std(zs_f1s)),
    }

    # ===== Experiment 2: Train from scratch on PPE =====
    print(f"\n{'='*60}")
    print(f"EXP 2: Train from Scratch on PPE")
    print(f"{'='*60}")

    scratch_f1s = []
    for seed in seeds:
        set_seed(seed)
        model, _ = train_probe(ppe_train_emb, ppe_train_labels,
                               ppe_val_emb, ppe_val_labels,
                               embedding_dim, device, epochs=args.epochs)
        ppe_metrics = evaluate(model, ppe_test_emb, ppe_test_labels, device)
        scratch_f1s.append(ppe_metrics['f1'] * 100)
        print(f"  Seed {seed}: PPE F1={ppe_metrics['f1']*100:.2f}%")

    results['scratch'] = {
        'description': 'Train on PPE train, test on PPE test',
        'ppe_f1_mean': float(np.mean(scratch_f1s)),
        'ppe_f1_std': float(np.std(scratch_f1s)),
    }

    # ===== Experiment 3: Pre-train on AI Hub, fine-tune on PPE =====
    print(f"\n{'='*60}")
    print(f"EXP 3: Pre-train on AI Hub → Fine-tune on PPE")
    print(f"{'='*60}")

    ft_f1s = []
    for seed in seeds:
        set_seed(seed)
        # Pre-train on AI Hub
        pretrained_model, _ = train_probe(src_train_emb, src_train_labels,
                                          src_val_emb, src_val_labels,
                                          embedding_dim, device, epochs=args.epochs)
        pretrained_state = {k: v.cpu().clone() for k, v in pretrained_model.state_dict().items()}

        # Fine-tune on PPE
        model, _ = train_probe(ppe_train_emb, ppe_train_labels,
                               ppe_val_emb, ppe_val_labels,
                               embedding_dim, device, epochs=30, lr=1e-4,
                               pretrained_state=pretrained_state)
        ppe_metrics = evaluate(model, ppe_test_emb, ppe_test_labels, device)
        ft_f1s.append(ppe_metrics['f1'] * 100)
        print(f"  Seed {seed}: PPE F1={ppe_metrics['f1']*100:.2f}%")

    results['finetune'] = {
        'description': 'Pre-train on AI Hub, fine-tune on PPE',
        'ppe_f1_mean': float(np.mean(ft_f1s)),
        'ppe_f1_std': float(np.std(ft_f1s)),
    }

    # ===== Summary =====
    print(f"\n{'='*60}")
    print(f"CROSS-DOMAIN SUMMARY ({args.model.upper()})")
    print(f"{'='*60}")
    print(f"{'Method':<45} {'PPE Test F1':<20}")
    print("-" * 65)
    for method, r in results.items():
        print(f"{r['description']:<45} {r['ppe_f1_mean']:>6.2f}±{r['ppe_f1_std']:.2f}%")

    # Save
    with open(output_dir / 'cross_domain_results.json', 'w') as f:
        json.dump({
            'model': args.model,
            'seeds': seeds,
            'source': 'AI Hub scenario_v2',
            'target': 'Construction-PPE',
            'results': results,
        }, f, indent=2)
    print(f"\nResults saved to {output_dir}/cross_domain_results.json")


if __name__ == '__main__':
    main()
