#!/usr/bin/env python3
"""
Ensemble 2-model Subset Ablation

Tests all ensemble subsets on scenario test set:
  - SigLIP+CLIP, SigLIP+DINOv2, CLIP+DINOv2, All 3
  - Also tests single models for comparison

Usage:
    python scripts/run_ensemble_ablation.py --embeddings-base embeddings --output results/ensemble_ablation
"""

import argparse
import json
import sys
import time
from pathlib import Path
from itertools import combinations

import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.train_from_embeddings import BinarySafetyClassifier, load_split, set_seed


def load_trained_model(model_name, split, embeddings_base, seed=42, probe_depth='2layer', epochs=50):
    """Train and return a probe model."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    emb_dir = Path(embeddings_base) / split / model_name
    train_emb, train_labels, _ = load_split(emb_dir, 'train')
    embedding_dim = train_emb.shape[1]

    model = BinarySafetyClassifier(embedding_dim, hidden_dim=512, probe_depth=probe_depth).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    has_val = (emb_dir / 'val_embeddings.pt').exists()
    if has_val:
        val_emb, val_labels, _ = load_split(emb_dir, 'val')

    best_f1 = 0
    best_state = None

    dataset = torch.utils.data.TensorDataset(train_emb, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for emb_batch, label_batch in loader:
            emb_batch, label_batch = emb_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            out = model(emb_batch)
            loss = criterion(out, label_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                val_out = model(val_emb.to(device)).cpu()
                val_preds = (val_out > 0.5).numpy().astype(int)
                val_f1 = f1_score(val_labels.numpy() > 0.5, val_preds, zero_division=0)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    return model, emb_dir, device


def get_test_predictions(model, emb_dir, device, category=None):
    """Get per-image predictions on test set."""
    emb, labels, filenames = load_split(emb_dir, 'test', category)

    model.eval()
    with torch.no_grad():
        scores = model(emb.to(device)).cpu().numpy()

    return filenames, labels.numpy(), scores


def main():
    parser = argparse.ArgumentParser(description='Ensemble subset ablation')
    parser.add_argument('--embeddings-base', type=str, default='embeddings')
    parser.add_argument('--output', type=str, default='results/ensemble_ablation')
    parser.add_argument('--split', type=str, default='scenario')
    parser.add_argument('--seeds', type=str, default='42,123,456,789,2024')

    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = ['siglip', 'clip', 'dinov2']
    model_combos = []
    for r in range(1, len(models) + 1):
        for combo in combinations(models, r):
            model_combos.append(list(combo))

    print(f"{'='*70}")
    print(f"ENSEMBLE SUBSET ABLATION ({args.split})")
    print(f"{'='*70}")
    print(f"Seeds: {seeds}")
    print(f"Combinations: {['+'.join(c) for c in model_combos]}")
    print()

    all_results = {}
    start_time = time.time()

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # Train all individual models
        trained = {}
        for model_name in models:
            print(f"  Training {model_name}...", end=' ', flush=True)
            model, emb_dir, device = load_trained_model(
                model_name, args.split, args.embeddings_base, seed=seed
            )
            filenames, labels, scores = get_test_predictions(model, emb_dir, device)
            trained[model_name] = {
                'filenames': filenames,
                'labels': labels,
                'scores': scores,
            }

            f1 = f1_score(labels > 0.5, scores > 0.5, zero_division=0)
            print(f"F1={f1*100:.2f}%")

        # Test all combinations
        for combo in model_combos:
            combo_key = '+'.join(combo)

            if len(combo) == 1:
                # Single model
                d = trained[combo[0]]
                avg_scores = d['scores']
                labels = d['labels']
            else:
                # Ensemble: average scores across models
                # Need to align by filename
                ref = trained[combo[0]]
                fname_to_idx = {fn: i for i, fn in enumerate(ref['filenames'])}

                # Collect scores for common filenames
                common_fnames = set(ref['filenames'])
                for m in combo[1:]:
                    common_fnames &= set(trained[m]['filenames'])

                common_fnames = sorted(common_fnames)
                ensemble_scores = np.zeros(len(common_fnames))
                labels = np.zeros(len(common_fnames))

                for m in combo:
                    m_fname_to_idx = {fn: i for i, fn in enumerate(trained[m]['filenames'])}
                    for i, fn in enumerate(common_fnames):
                        ensemble_scores[i] += trained[m]['scores'][m_fname_to_idx[fn]]
                        if m == combo[0]:
                            labels[i] = trained[m]['labels'][m_fname_to_idx[fn]]

                avg_scores = ensemble_scores / len(combo)

            preds = (avg_scores > 0.5).astype(int)
            targets_binary = labels > 0.5

            f1 = f1_score(targets_binary, preds, zero_division=0)
            acc = accuracy_score(targets_binary, preds)
            try:
                auc = roc_auc_score(targets_binary, avg_scores)
            except Exception:
                auc = 0.0

            if combo_key not in all_results:
                all_results[combo_key] = []
            all_results[combo_key].append({
                'seed': seed,
                'f1': float(f1),
                'accuracy': float(acc),
                'auc_roc': float(auc),
                'n_samples': len(labels),
            })

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*70}")
    print(f"ENSEMBLE ABLATION SUMMARY ({args.split}, {len(seeds)} seeds)")
    print(f"{'='*70}")
    print(f"{'Combination':<30} {'F1 (mean±std)':>20} {'AUC':>15}")
    print("-" * 70)

    summary = {}
    for combo_key, results in sorted(all_results.items(),
                                      key=lambda x: -np.mean([r['f1'] for r in x[1]])):
        f1s = [r['f1'] * 100 for r in results]
        aucs = [r['auc_roc'] * 100 for r in results]
        f1_mean, f1_std = np.mean(f1s), np.std(f1s)
        auc_mean, auc_std = np.mean(aucs), np.std(aucs)

        print(f"{combo_key:<30} {f1_mean:>10.2f}±{f1_std:.2f}% {auc_mean:>8.2f}±{auc_std:.2f}%")

        summary[combo_key] = {
            'f1_mean': float(f1_mean),
            'f1_std': float(f1_std),
            'auc_mean': float(auc_mean),
            'auc_std': float(auc_std),
            'per_seed': results,
        }

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    output_file = output_dir / 'ensemble_ablation.json'
    with open(output_file, 'w') as f:
        json.dump({
            'split': args.split,
            'seeds': seeds,
            'elapsed_seconds': elapsed,
            'results': summary,
        }, f, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
