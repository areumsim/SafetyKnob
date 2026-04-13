#!/usr/bin/env python3
"""
Clean DANN (Domain Adversarial Neural Network) for Temporal Safety Classification

Fixes data leakage in original train_dann.py by using ONLY temporal split data:
  - Source domain: temporal train (June-Sept, labeled)
  - Target domain: temporal test (Oct-Nov, unlabeled for DANN, used for eval only)

The original experiment used scenario embeddings as source and temporal as target,
but 64.3% of temporal test images appeared in scenario train (data leakage).

Usage:
    python experiments/train_dann_clean.py \
        --model siglip \
        --temporal-embeddings embeddings/temporal/siglip \
        --output results/dann_clean/siglip \
        --seeds 42,123,456,789,2024
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.train_from_embeddings import load_split
from experiments.train_dann import DANNClassifier, GradientReversalLayer, compute_alpha


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Clean DANN (no data leakage)')
    parser.add_argument('--model', type=str, default='siglip')
    parser.add_argument('--temporal-embeddings', type=str, required=True,
                       help='Temporal embeddings directory (train/val/test splits)')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--domain-weight', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=str, default=None)

    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(',')]
    else:
        seeds = [args.seed]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    temporal_dir = Path(args.temporal_embeddings)

    all_results = []

    for seed in seeds:
        set_seed(seed)

        print(f"\n{'='*60}")
        print(f"CLEAN DANN TRAINING (seed={seed})")
        print(f"{'='*60}")

        # Source = temporal train (June-Sept, labeled)
        src_train_emb, src_train_labels, _ = load_split(temporal_dir, 'train')
        src_val_emb, src_val_labels, _ = load_split(temporal_dir, 'val')

        # Target = temporal test (Oct-Nov, labels used ONLY for evaluation)
        tgt_test_emb, tgt_test_labels, tgt_test_files = load_split(temporal_dir, 'test')

        embedding_dim = src_train_emb.shape[1]

        print(f"Source (temporal train): {len(src_train_emb)} samples")
        print(f"Source val: {len(src_val_emb)} samples")
        print(f"Target (temporal test): {len(tgt_test_emb)} samples")
        print(f"Embedding dim: {embedding_dim}")

        # Create model
        model = DANNClassifier(embedding_dim).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")

        class_criterion = nn.BCELoss()
        domain_criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Dataloaders
        src_dataset = TensorDataset(src_train_emb, src_train_labels)
        # Target: use test embeddings for domain adaptation (labels NOT used in training)
        tgt_dataset = TensorDataset(tgt_test_emb, torch.zeros(len(tgt_test_emb)))  # dummy labels

        src_loader = DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        tgt_loader = DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        # Training
        best_src_val_f1 = 0
        best_state = None
        start_time = time.time()

        for epoch in range(args.epochs):
            model.train()
            alpha = compute_alpha(epoch, args.epochs)

            total_class_loss = 0
            total_domain_loss = 0
            n_batches = 0

            tgt_iter = iter(tgt_loader)

            for src_emb, src_labels in src_loader:
                try:
                    tgt_emb, _ = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_loader)
                    tgt_emb, _ = next(tgt_iter)

                src_emb = src_emb.to(device)
                src_labels = src_labels.to(device)
                tgt_emb = tgt_emb.to(device)

                # Source: class loss + domain loss (domain=0)
                src_class_out, src_domain_out, _ = model(src_emb, alpha)
                class_loss = class_criterion(src_class_out, src_labels)
                src_domain_loss = domain_criterion(
                    src_domain_out,
                    torch.zeros(len(src_emb), device=device)
                )

                # Target: domain loss only (domain=1), NO class labels
                _, tgt_domain_out, _ = model(tgt_emb, alpha)
                tgt_domain_loss = domain_criterion(
                    tgt_domain_out,
                    torch.ones(len(tgt_emb), device=device)
                )

                domain_loss = (src_domain_loss + tgt_domain_loss) / 2
                total_loss = class_loss + args.domain_weight * domain_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                total_class_loss += class_loss.item()
                total_domain_loss += domain_loss.item()
                n_batches += 1

            scheduler.step()

            # Evaluate periodically
            if (epoch + 1) % 10 == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    # Source validation F1 - used for model selection
                    src_val_out, _, _ = model(src_val_emb.to(device), alpha=0)
                    src_val_preds = (src_val_out.cpu() > 0.5).numpy().astype(int)
                    src_val_targets = (src_val_labels.numpy() > 0.5).astype(int)
                    src_val_f1 = f1_score(src_val_targets, src_val_preds, zero_division=0)

                    # Target test F1 - for monitoring only
                    tgt_out, _, _ = model(tgt_test_emb.to(device), alpha=0)
                    tgt_preds = (tgt_out.cpu() > 0.5).numpy().astype(int)
                    tgt_targets = (tgt_test_labels.numpy() > 0.5).astype(int)
                    tgt_f1 = f1_score(tgt_targets, tgt_preds, zero_division=0)

                avg_cls = total_class_loss / max(1, n_batches)
                avg_dom = total_domain_loss / max(1, n_batches)
                print(f"Epoch {epoch+1:3d}: cls={avg_cls:.4f}, dom={avg_dom:.4f}, "
                      f"alpha={alpha:.3f}, val_f1={src_val_f1*100:.2f}%, "
                      f"tgt_f1={tgt_f1*100:.2f}%")

                # Model selection: source val F1
                if src_val_f1 > best_src_val_f1:
                    best_src_val_f1 = src_val_f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        training_time = time.time() - start_time

        # Final evaluation
        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            # Source (temporal train) test - evaluate on val since no separate source test
            src_out, _, _ = model(src_val_emb.to(device), alpha=0)
            src_scores = src_out.cpu().numpy()
            src_preds = (src_scores > 0.5).astype(int)
            src_targets = (src_val_labels.numpy() > 0.5).astype(int)

            # Target (temporal test)
            tgt_out, _, _ = model(tgt_test_emb.to(device), alpha=0)
            tgt_scores = tgt_out.cpu().numpy()
            tgt_preds = (tgt_scores > 0.5).astype(int)
            tgt_targets = (tgt_test_labels.numpy() > 0.5).astype(int)

        src_metrics = {
            'f1': float(f1_score(src_targets, src_preds, zero_division=0)),
            'accuracy': float(accuracy_score(src_targets, src_preds)),
            'auc_roc': float(roc_auc_score(src_targets, src_scores)) if len(set(src_targets.tolist())) > 1 else 0.0,
        }
        tgt_metrics = {
            'f1': float(f1_score(tgt_targets, tgt_preds, zero_division=0)),
            'accuracy': float(accuracy_score(tgt_targets, tgt_preds)),
            'auc_roc': float(roc_auc_score(tgt_targets, tgt_scores)) if len(set(tgt_targets.tolist())) > 1 else 0.0,
        }

        print(f"\n--- Final Results (seed={seed}) ---")
        print(f"Source Val F1: {src_metrics['f1']*100:.2f}%")
        print(f"Target Test F1: {tgt_metrics['f1']*100:.2f}%")
        print(f"Training time: {training_time:.1f}s")

        result = {
            'seed': seed,
            'model': args.model,
            'model_selection': 'source_val_f1',
            'best_src_val_f1': float(best_src_val_f1),
            'source_metrics': src_metrics,
            'target_metrics': tgt_metrics,
            'total_params': total_params,
            'training_time_seconds': training_time,
            'hyperparameters': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'domain_weight': args.domain_weight,
            }
        }
        all_results.append(result)

    # Summary
    tgt_f1s = [r['target_metrics']['f1'] * 100 for r in all_results]
    src_f1s = [r['source_metrics']['f1'] * 100 for r in all_results]

    print(f"\n{'='*60}")
    print(f"CLEAN DANN SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*60}")
    print(f"Source Val F1: {np.mean(src_f1s):.2f}±{np.std(src_f1s):.2f}%")
    print(f"Target Test F1: {np.mean(tgt_f1s):.2f}±{np.std(tgt_f1s):.2f}%")

    summary = {
        'model': args.model,
        'method': 'DANN_clean',
        'description': 'DANN using only temporal split (no scenario data, no leakage)',
        'model_selection': 'source_val_f1',
        'seeds': seeds,
        'source_f1_mean': float(np.mean(src_f1s)),
        'source_f1_std': float(np.std(src_f1s)),
        'target_f1_mean': float(np.mean(tgt_f1s)),
        'target_f1_std': float(np.std(tgt_f1s)),
        'per_seed': all_results,
    }

    with open(output_dir / 'dann_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {output_dir}/dann_results.json")


if __name__ == '__main__':
    main()
