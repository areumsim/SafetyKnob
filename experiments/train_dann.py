#!/usr/bin/env python3
"""
DANN (Domain Adversarial Neural Network) for Safety Classification

Uses gradient reversal to learn domain-invariant features for
scenario→temporal domain adaptation.

Architecture:
  [Cached Embedding] → Feature Extractor → Class Predictor (safe/unsafe)
                                         → Domain Predictor (scenario/temporal)
                                            ↑ gradient reversal layer

Usage:
    python experiments/train_dann.py \
        --model siglip \
        --source-embeddings embeddings/scenario/siglip \
        --target-embeddings embeddings/temporal/siglip \
        --output results/dann/siglip
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


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class DANNClassifier(nn.Module):
    """Domain Adversarial Neural Network for safety classification."""

    def __init__(self, embedding_dim, hidden_dim=512, domain_hidden=256):
        super().__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Class predictor (safe/unsafe)
        self.class_predictor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Domain predictor with gradient reversal
        self.gradient_reversal = GradientReversalLayer()
        self.domain_predictor = nn.Sequential(
            nn.Linear(256, domain_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(domain_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)

        # Class prediction
        class_output = self.class_predictor(features).squeeze()

        # Domain prediction with gradient reversal
        self.gradient_reversal.alpha = alpha
        reversed_features = self.gradient_reversal(features)
        domain_output = self.domain_predictor(reversed_features).squeeze()

        return class_output, domain_output, features


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_alpha(epoch, total_epochs):
    """Progressive alpha schedule (starts low, increases)."""
    p = epoch / total_epochs
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0


def main():
    parser = argparse.ArgumentParser(description='DANN domain adaptation')
    parser.add_argument('--model', type=str, default='siglip')
    parser.add_argument('--source-embeddings', type=str, required=True,
                       help='Source domain (scenario) embeddings directory')
    parser.add_argument('--target-embeddings', type=str, required=True,
                       help='Target domain (temporal) embeddings directory')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--domain-weight', type=float, default=0.5,
                       help='Weight for domain adversarial loss')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=str, default=None,
                       help='Comma-separated seeds for multi-seed run')

    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(',')]
    else:
        seeds = [args.seed]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source_dir = Path(args.source_embeddings)
    target_dir = Path(args.target_embeddings)

    all_results = []

    for seed in seeds:
        set_seed(seed)

        print(f"\n{'='*60}")
        print(f"DANN TRAINING (seed={seed})")
        print(f"{'='*60}")

        # Load source (scenario) data - with labels
        src_train_emb, src_train_labels, _ = load_split(source_dir, 'train')
        src_test_emb, src_test_labels, _ = load_split(source_dir, 'test')

        # Load target (temporal) data - labels only for evaluation
        tgt_train_emb, tgt_train_labels, _ = load_split(target_dir, 'train')
        tgt_test_emb, tgt_test_labels, tgt_test_files = load_split(target_dir, 'test')

        embedding_dim = src_train_emb.shape[1]

        print(f"Source train: {len(src_train_emb)}, Target train: {len(tgt_train_emb)}")
        print(f"Source test:  {len(src_test_emb)}, Target test:  {len(tgt_test_emb)}")
        print(f"Embedding dim: {embedding_dim}")

        # Create model
        model = DANNClassifier(embedding_dim).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")

        class_criterion = nn.BCELoss()
        domain_criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Prepare dataloaders
        src_dataset = TensorDataset(src_train_emb, src_train_labels)
        tgt_dataset = TensorDataset(tgt_train_emb, tgt_train_labels)

        src_loader = DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        tgt_loader = DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        # Load source validation set for proper model selection
        src_val_emb, src_val_labels, _ = load_split(source_dir, 'val')

        # Training
        best_src_val_f1 = 0
        best_tgt_f1_at_selection = 0  # for logging only, NOT used for selection
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
                # Get target batch (cycle if shorter)
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
                    torch.zeros(len(src_emb), device=device)  # source = 0
                )

                # Target: domain loss only (domain=1), no class labels used
                _, tgt_domain_out, _ = model(tgt_emb, alpha)
                tgt_domain_loss = domain_criterion(
                    tgt_domain_out,
                    torch.ones(len(tgt_emb), device=device)  # target = 1
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

                    # Target test F1 - for monitoring only, NOT used for selection
                    tgt_class_out, _, _ = model(tgt_test_emb.to(device), alpha=0)
                    tgt_preds = (tgt_class_out.cpu() > 0.5).numpy().astype(int)
                    tgt_targets = (tgt_test_labels.numpy() > 0.5).astype(int)
                    tgt_f1 = f1_score(tgt_targets, tgt_preds, zero_division=0)

                    # Source test F1 - for monitoring only
                    src_class_out, _, _ = model(src_test_emb.to(device), alpha=0)
                    src_preds = (src_class_out.cpu() > 0.5).numpy().astype(int)
                    src_targets = (src_test_labels.numpy() > 0.5).astype(int)
                    src_f1 = f1_score(src_targets, src_preds, zero_division=0)

                avg_cls = total_class_loss / max(1, n_batches)
                avg_dom = total_domain_loss / max(1, n_batches)
                print(f"Epoch {epoch+1:3d}: cls_loss={avg_cls:.4f}, dom_loss={avg_dom:.4f}, "
                      f"alpha={alpha:.3f}, src_val_f1={src_val_f1*100:.2f}%, "
                      f"src_test_f1={src_f1*100:.2f}%, tgt_f1={tgt_f1*100:.2f}%")

                # Model selection based on SOURCE VALIDATION F1 (not target test!)
                if src_val_f1 > best_src_val_f1:
                    best_src_val_f1 = src_val_f1
                    best_tgt_f1_at_selection = tgt_f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        training_time = time.time() - start_time

        # Final evaluation with best model
        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            # Source test
            src_out, _, _ = model(src_test_emb.to(device), alpha=0)
            src_scores = src_out.cpu().numpy()
            src_preds = (src_scores > 0.5).astype(int)
            src_targets = (src_test_labels.numpy() > 0.5).astype(int)

            # Target test
            tgt_out, _, _ = model(tgt_test_emb.to(device), alpha=0)
            tgt_scores = tgt_out.cpu().numpy()
            tgt_preds = (tgt_scores > 0.5).astype(int)
            tgt_targets = (tgt_test_labels.numpy() > 0.5).astype(int)

        src_metrics = {
            'f1': float(f1_score(src_targets, src_preds, zero_division=0)),
            'accuracy': float(accuracy_score(src_targets, src_preds)),
            'auc_roc': float(roc_auc_score(src_targets, src_scores)),
        }
        tgt_metrics = {
            'f1': float(f1_score(tgt_targets, tgt_preds, zero_division=0)),
            'accuracy': float(accuracy_score(tgt_targets, tgt_preds)),
            'auc_roc': float(roc_auc_score(tgt_targets, tgt_scores)),
        }

        print(f"\n--- Final Results (seed={seed}) ---")
        print(f"Model selection: source val F1 = {best_src_val_f1*100:.2f}%")
        print(f"Source (Scenario) Test F1: {src_metrics['f1']*100:.2f}%")
        print(f"Target (Temporal) Test F1: {tgt_metrics['f1']*100:.2f}%")
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

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'seed': seed,
            'source_metrics': src_metrics,
            'target_metrics': tgt_metrics,
        }, output_dir / f'best_model_seed{seed}.pt')

    # Summary across seeds
    src_f1s = [r['source_metrics']['f1'] * 100 for r in all_results]
    tgt_f1s = [r['target_metrics']['f1'] * 100 for r in all_results]

    print(f"\n{'='*60}")
    print(f"DANN SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*60}")
    print(f"Source (Scenario) F1: {np.mean(src_f1s):.2f}±{np.std(src_f1s):.2f}%")
    print(f"Target (Temporal) F1: {np.mean(tgt_f1s):.2f}±{np.std(tgt_f1s):.2f}%")

    # Save results (baseline comparison done externally to avoid hardcoded values)
    summary = {
        'model': args.model,
        'method': 'DANN',
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
    print(f"\nResults saved to {output_dir}/dann_results.json")


if __name__ == '__main__':
    main()
