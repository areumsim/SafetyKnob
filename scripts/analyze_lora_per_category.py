#!/usr/bin/env python3
"""
LoRA Per-Category Analysis (W5 Response)

Loads a saved LoRA checkpoint, runs per-category inference on temporal
test images, and correlates LoRA improvement with MMD per category.

Usage:
    python scripts/analyze_lora_per_category.py \
        --checkpoint results/lora_rank_ablation/r16/best_model_seed42.pt \
        --data-dir data_temporal \
        --output results/lora_per_category_analysis
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score
from scipy import stats

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'figure.dpi': 150, 'font.family': 'DejaVu Sans',
})

CAT_NAMES = {
    'A': 'Fall Hazard',
    'B': 'Collision Risk',
    'C': 'Equipment Hazard',
    'D': 'Environmental Risk',
    'E': 'Protective Gear',
}

CATEGORIES = ['A', 'B', 'C', 'D', 'E']


def extract_category(filename):
    """Extract category letter from filename."""
    m = re.search(r'_([A-F])\d{2}_', filename)
    return m.group(1) if m else None


class SigLIPWithHead(nn.Module):
    """SigLIP vision model + classification head (must match training code)."""
    def __init__(self, base_model, embedding_dim=1152):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values=pixel_values)
        embeddings = outputs.pooler_output
        return self.classifier(embeddings).squeeze(-1)


def load_lora_model(checkpoint_path, lora_r, lora_alpha, device):
    """Reconstruct and load a LoRA model from checkpoint."""
    from transformers import SiglipVisionModel
    from peft import LoraConfig, get_peft_model

    base_model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")

    if lora_r > 0:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        base_model = get_peft_model(base_model, lora_config)

    model = SigLIPWithHead(base_model, embedding_dim=1152)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def load_frozen_model(checkpoint_path, device):
    """Load a frozen (head-only, r=0) model from checkpoint."""
    from transformers import SiglipVisionModel

    base_model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
    for param in base_model.parameters():
        param.requires_grad = False

    model = SigLIPWithHead(base_model, embedding_dim=1152)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def run_per_category_inference(model, data_dir, processor, device, batch_size=8):
    """Run inference on test images, grouping results by category."""
    test_dir = Path(data_dir) / "test"
    labels_path = Path(data_dir) / "labels.json"

    with open(labels_path) as f:
        labels_json = json.load(f)

    # Group test images by category
    cat_data = defaultdict(lambda: {"images": [], "labels": []})
    for img_path in sorted(test_dir.glob("*.jpg")):
        key = f"test/{img_path.name}"
        if key not in labels_json:
            continue
        cat = extract_category(img_path.name)
        if cat and cat in CAT_NAMES:
            cat_data[cat]["images"].append(str(img_path))
            cat_data[cat]["labels"].append(float(labels_json[key]["overall_safety"]))

    # Run inference per category
    cat_results = {}
    for cat in CATEGORIES:
        if cat not in cat_data or not cat_data[cat]["images"]:
            continue

        images = cat_data[cat]["images"]
        labels = cat_data[cat]["labels"]
        all_preds = []

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            pixel_values_list = []
            for img_path in batch_imgs:
                img = Image.open(img_path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt")
                pixel_values_list.append(inputs["pixel_values"].squeeze(0))

            pixel_values = torch.stack(pixel_values_list).to(device)
            with torch.no_grad():
                outputs = model(pixel_values)
                preds = (outputs.cpu() > 0.5).numpy().astype(int)
                all_preds.extend(preds)

        f1 = f1_score(labels, all_preds, zero_division=0)
        cat_results[cat] = {
            "name": CAT_NAMES[cat],
            "f1": float(f1),
            "n_test": len(labels),
            "n_positive": int(sum(labels)),
            "n_negative": len(labels) - int(sum(labels)),
        }

    return cat_results


def main():
    parser = argparse.ArgumentParser(description="LoRA per-category analysis")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to LoRA model checkpoint")
    parser.add_argument("--frozen-checkpoint", type=str, default=None,
                       help="Path to frozen (r=0) model checkpoint for comparison")
    parser.add_argument("--data-dir", type=str, default="data_temporal")
    parser.add_argument("--output", type=str, default="results/lora_per_category_analysis")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--mmd-json", type=str, default="results/figures_final/temporal_shift_analysis.json",
                       help="Path to MMD analysis JSON")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from transformers import SiglipImageProcessor
    processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    # Load LoRA model and run per-category inference
    print(f"\nLoading LoRA model (r={args.lora_r}) from {args.checkpoint}")
    lora_model = load_lora_model(args.checkpoint, args.lora_r, args.lora_alpha, device)
    lora_results = run_per_category_inference(lora_model, args.data_dir, processor, device, args.batch_size)
    del lora_model
    torch.cuda.empty_cache()

    print("\nLoRA per-category F1:")
    for cat in CATEGORIES:
        if cat in lora_results:
            r = lora_results[cat]
            print(f"  {cat} ({r['name']}): F1={r['f1']*100:.2f}% (n={r['n_test']})")

    # Load frozen baseline if checkpoint provided
    frozen_results = None
    if args.frozen_checkpoint:
        print(f"\nLoading frozen model (r=0) from {args.frozen_checkpoint}")
        frozen_model = load_frozen_model(args.frozen_checkpoint, device)
        frozen_results = run_per_category_inference(frozen_model, args.data_dir, processor, device, args.batch_size)
        del frozen_model
        torch.cuda.empty_cache()

        print("\nFrozen per-category F1:")
        for cat in CATEGORIES:
            if cat in frozen_results:
                r = frozen_results[cat]
                print(f"  {cat} ({r['name']}): F1={r['f1']*100:.2f}% (n={r['n_test']})")

    # Load MMD values
    mmd_data = None
    mmd_path = Path(args.mmd_json)
    if mmd_path.exists():
        with open(mmd_path) as f:
            mmd_data = json.load(f)
        print(f"\nLoaded MMD data from {mmd_path}")

    # Correlation analysis (LoRA improvement vs MMD)
    correlation = None
    if frozen_results and mmd_data:
        deltas = []
        mmds = []
        cats_used = []

        for cat in CATEGORIES:
            if cat in lora_results and cat in frozen_results and cat in mmd_data["mmd_per_category"]:
                delta = (lora_results[cat]["f1"] - frozen_results[cat]["f1"]) * 100
                mmd = mmd_data["mmd_per_category"][cat]["mmd"]
                deltas.append(delta)
                mmds.append(mmd)
                cats_used.append(cat)

        if len(deltas) >= 3:
            pearson_r, pearson_p = stats.pearsonr(mmds, deltas)
            spearman_r, spearman_p = stats.spearmanr(mmds, deltas)
            correlation = {
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "categories": cats_used,
                "mmds": [float(m) for m in mmds],
                "deltas": [float(d) for d in deltas],
            }
            print(f"\n--- MMD vs LoRA Improvement Correlation ---")
            print(f"  Pearson:  r={pearson_r:.3f}, p={pearson_p:.3f}")
            print(f"  Spearman: r={spearman_r:.3f}, p={spearman_p:.3f}")

    # Generate figure
    if frozen_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Grouped bar chart (frozen vs LoRA per category)
        ax = axes[0]
        x = np.arange(len(CATEGORIES))
        width = 0.35

        frozen_f1s = [frozen_results.get(c, {}).get("f1", 0) * 100 for c in CATEGORIES]
        lora_f1s = [lora_results.get(c, {}).get("f1", 0) * 100 for c in CATEGORIES]

        bars1 = ax.bar(x - width/2, frozen_f1s, width, label='Frozen (r=0)', color='#4ECDC4', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, lora_f1s, width, label=f'LoRA (r={args.lora_r})', color='#FF6B6B', edgecolor='black', linewidth=0.5)

        # Value annotations
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Category')
        ax.set_ylabel('Test F1 (%)')
        ax.set_title('Per-Category F1: Frozen vs LoRA')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n{CAT_NAMES[c]}" for c in CATEGORIES], fontsize=8)
        ax.legend()
        ax.set_ylim(0, 105)

        # Panel 2: MMD vs LoRA improvement scatter
        ax = axes[1]
        if correlation:
            ax.scatter(correlation["mmds"], correlation["deltas"],
                      s=100, c='#FF6B6B', edgecolors='black', zorder=5)
            for i, cat in enumerate(correlation["categories"]):
                ax.annotate(f"{cat}: {CAT_NAMES[cat]}",
                           (correlation["mmds"][i], correlation["deltas"][i]),
                           textcoords="offset points", xytext=(8, 5), fontsize=9)

            # Trend line
            z = np.polyfit(correlation["mmds"], correlation["deltas"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(correlation["mmds"]), max(correlation["mmds"]), 100)
            ax.plot(x_line, p(x_line), "--", color='gray', alpha=0.7)

            ax.set_xlabel('MMD (Distribution Shift)')
            ax.set_ylabel('LoRA F1 Improvement (%p)')
            ax.set_title(f'MMD vs LoRA Improvement\n(Spearman r={correlation["spearman_r"]:.2f}, p={correlation["spearman_p"]:.3f})')
        else:
            ax.text(0.5, 0.5, "No frozen baseline or\nMMD data available",
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title('MMD vs LoRA Improvement')

        plt.tight_layout()
        out_path = output_dir / "fig_lora_per_category.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nFigure saved: {out_path}")

    # Save results
    analysis = {
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_per_category": lora_results,
    }
    if frozen_results:
        analysis["frozen_per_category"] = frozen_results
    if correlation:
        analysis["mmd_correlation"] = correlation

    out_json = output_dir / "lora_per_category_results.json"
    with open(out_json, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Results saved: {out_json}")


if __name__ == "__main__":
    main()
