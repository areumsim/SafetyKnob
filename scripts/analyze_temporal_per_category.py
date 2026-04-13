#!/usr/bin/env python3
"""
Per-Category Temporal Shift Analysis

Analyzes which hazard categories (A-E) are most vulnerable to temporal
distribution shift by comparing scenario vs temporal performance per category.

Usage:
    python scripts/analyze_temporal_per_category.py \
        --embeddings-base embeddings \
        --output results/temporal_per_category
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score


sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.train_from_embeddings import BinarySafetyClassifier, load_split, set_seed


CAT_NAMES = {
    'A': 'Fall Hazard',
    'B': 'Collision Risk',
    'C': 'Equipment Hazard',
    'D': 'Environmental Risk',
    'E': 'Protective Gear'
}


def evaluate_model_on_split(model, embeddings_dir, split, category, device):
    """Evaluate a trained model on a specific split and category."""
    try:
        emb, labels, filenames = load_split(embeddings_dir, split, category)
    except ValueError:
        return None

    model.eval()
    with torch.no_grad():
        emb_device = emb.to(device)
        scores = model(emb_device).cpu().numpy()
        preds = (scores > 0.5).astype(int)
        targets = labels.numpy()

    n_samples = len(targets)
    n_unsafe = int((targets == 0).sum()) if targets.dtype == float else int((targets < 0.5).sum())
    n_safe = n_samples - n_unsafe

    try:
        f1 = f1_score(targets > 0.5, preds, zero_division=0)
        acc = accuracy_score(targets > 0.5, preds)
        auc = roc_auc_score(targets, scores) if len(set(targets.tolist())) > 1 else 0.0
    except Exception:
        f1, acc, auc = 0.0, 0.0, 0.0

    return {
        'f1': float(f1),
        'accuracy': float(acc),
        'auc_roc': float(auc),
        'n_samples': n_samples,
        'n_safe': n_safe,
        'n_unsafe': n_unsafe,
    }


def train_and_evaluate(model_name, category, split, embeddings_base, seed=42, epochs=50):
    """Train a probe on scenario data for a category, evaluate on both scenario and temporal test."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train on scenario
    scenario_emb_dir = Path(embeddings_base) / 'scenario' / model_name
    train_emb, train_labels, _ = load_split(scenario_emb_dir, 'train', category)
    embedding_dim = train_emb.shape[1]

    model = BinarySafetyClassifier(embedding_dim, hidden_dim=512, probe_depth='2layer').to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Try to load val
    has_val = (scenario_emb_dir / 'val_embeddings.pt').exists()
    if has_val:
        val_emb, val_labels, _ = load_split(scenario_emb_dir, 'val', category)

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

    # Evaluate on scenario test
    scenario_result = evaluate_model_on_split(model, scenario_emb_dir, 'test', category, device)

    # Evaluate on temporal test
    temporal_emb_dir = Path(embeddings_base) / 'temporal' / model_name
    temporal_result = evaluate_model_on_split(model, temporal_emb_dir, 'test', category, device)

    return scenario_result, temporal_result


def main():
    parser = argparse.ArgumentParser(description='Per-category temporal shift analysis')
    parser.add_argument('--embeddings-base', type=str, default='embeddings')
    parser.add_argument('--output', type=str, default='results/temporal_per_category')
    parser.add_argument('--model', type=str, default='siglip',
                       choices=['siglip', 'clip', 'dinov2'])
    parser.add_argument('--seeds', type=str, default='42,123,456,789,2024')

    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"PER-CATEGORY TEMPORAL SHIFT ANALYSIS")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Seeds: {seeds}")
    print()

    results = {}
    start_time = time.time()

    for cat_code, cat_name in CAT_NAMES.items():
        print(f"\n--- Category {cat_code}: {cat_name} ---")
        scenario_f1s = []
        temporal_f1s = []

        for seed in seeds:
            print(f"  Seed {seed}...", end=' ', flush=True)
            scenario_res, temporal_res = train_and_evaluate(
                args.model, cat_code, 'scenario', args.embeddings_base, seed=seed
            )

            if scenario_res and temporal_res:
                scenario_f1s.append(scenario_res['f1'] * 100)
                temporal_f1s.append(temporal_res['f1'] * 100)
                print(f"Scenario={scenario_res['f1']*100:.2f}%, Temporal={temporal_res['f1']*100:.2f}%")
            else:
                print("FAILED")

        if scenario_f1s and temporal_f1s:
            s_mean, s_std = np.mean(scenario_f1s), np.std(scenario_f1s)
            t_mean, t_std = np.mean(temporal_f1s), np.std(temporal_f1s)
            delta = t_mean - s_mean

            results[cat_code] = {
                'name': cat_name,
                'scenario_f1_mean': float(s_mean),
                'scenario_f1_std': float(s_std),
                'temporal_f1_mean': float(t_mean),
                'temporal_f1_std': float(t_std),
                'delta': float(delta),
                'n_scenario_test': scenario_res['n_samples'] if scenario_res else 0,
                'n_temporal_test': temporal_res['n_samples'] if temporal_res else 0,
                'per_seed_scenario': scenario_f1s,
                'per_seed_temporal': temporal_f1s,
            }

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*70}")
    print(f"PER-CATEGORY TEMPORAL SHIFT SUMMARY ({args.model.upper()}, {len(seeds)} seeds)")
    print(f"{'='*70}")
    print(f"{'Category':<22} {'Scenario F1':>18} {'Temporal F1':>18} {'Delta':>10} {'N(test)':>10}")
    print("-" * 80)

    sorted_cats = sorted(results.items(), key=lambda x: x[1]['delta'])
    for cat_code, r in sorted_cats:
        print(f"{r['name']:<22} "
              f"{r['scenario_f1_mean']:>8.2f}±{r['scenario_f1_std']:.2f}% "
              f"{r['temporal_f1_mean']:>8.2f}±{r['temporal_f1_std']:.2f}% "
              f"{r['delta']:>+8.2f}%p "
              f"{r['n_temporal_test']:>8d}")

    print(f"\nMost vulnerable: {sorted_cats[0][1]['name']} ({sorted_cats[0][1]['delta']:+.2f}%p)")
    print(f"Most robust:     {sorted_cats[-1][1]['name']} ({sorted_cats[-1][1]['delta']:+.2f}%p)")
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save
    output_file = output_dir / 'temporal_per_category.json'
    with open(output_file, 'w') as f:
        json.dump({
            'model': args.model,
            'seeds': seeds,
            'elapsed_seconds': elapsed,
            'categories': results
        }, f, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
