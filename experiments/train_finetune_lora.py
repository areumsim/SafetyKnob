#!/usr/bin/env python3
"""
SigLIP LoRA Fine-tuning Baseline

Compares frozen embedding + probe vs LoRA fine-tuning to establish
whether fine-tuning provides meaningful improvement over frozen probing.

Usage:
    python3 experiments/train_finetune_lora.py \
        --data-dir data_scenario_v2 \
        --output results/finetune_lora/siglip \
        --seed 42
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

from peft import LoraConfig, get_peft_model


class SafetyImageDataset(Dataset):
    """Dataset loading images directly for fine-tuning."""

    def __init__(self, data_dir, split, processor, labels_json, transform=None):
        self.processor = processor
        self.transform = transform
        self.images = []
        self.labels = []

        split_dir = Path(data_dir) / split
        for img_path in sorted(split_dir.glob("*.jpg")):
            key = f"{split}/{img_path.name}"
            if key in labels_json:
                self.images.append(str(img_path))
                self.labels.append(float(labels_json[key]["overall_safety"]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return pixel_values, label


class SigLIPWithHead(nn.Module):
    """SigLIP vision model + classification head."""

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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_on_loader(model, loader, device):
    """Run inference on a DataLoader, return metrics dict."""
    model.eval()
    preds, targets, scores = [], [], []
    with torch.no_grad():
        for pv, lb in loader:
            pv = pv.to(device)
            out = model(pv)
            scores.extend(out.cpu().numpy())
            preds.extend((out.cpu() > 0.5).numpy().astype(int))
            targets.extend(lb.numpy())
    return {
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "accuracy": float(accuracy_score(targets, preds)),
        "precision": float(precision_score(targets, preds, zero_division=0)),
        "recall": float(recall_score(targets, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(targets, scores)),
    }


def main():
    parser = argparse.ArgumentParser(description="SigLIP LoRA fine-tuning")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--test-data-dir", type=str, default=None,
                       help="Separate test data directory (for temporal evaluation)")
    parser.add_argument("--save-epoch-metrics", action="store_true", default=False,
                       help="Save per-epoch test metrics to epoch_metrics JSON")
    parser.add_argument("--save-checkpoints", action="store_true", default=False,
                       help="Save best model checkpoint to disk")
    parser.add_argument("--augment", action="store_true", default=False,
                       help="Apply data augmentation to training images")
    parser.add_argument("--num-workers", type=int, default=2,
                       help="Number of DataLoader workers (default: 2)")
    parser.add_argument("--full-finetune", action="store_true", default=False,
                       help="Full fine-tuning of entire backbone (no LoRA, no freezing)")

    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = [args.seed]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in seeds:
        set_seed(seed)
        print(f"\n{'='*60}")
        print(f"LoRA FINE-TUNING (seed={seed})")
        print(f"{'='*60}")

        # Load model and processor
        from transformers import SiglipImageProcessor, SiglipVisionModel
        processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        base_model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")

        # Apply LoRA, full fine-tuning, or freeze backbone (r=0 → head-only baseline)
        total_params = sum(p.numel() for p in base_model.parameters())
        if args.full_finetune:
            # Full fine-tuning: all backbone params trainable
            trainable_params = total_params
            print(f"MODE: Full fine-tuning (all backbone params trainable)")
        elif args.lora_r > 0:
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
            )
            base_model = get_peft_model(base_model, lora_config)
            trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        else:
            for param in base_model.parameters():
                param.requires_grad = False
            trainable_params = 0
        print(f"Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

        model = SigLIPWithHead(base_model, embedding_dim=1152).to(device)

        head_params = sum(p.numel() for p in model.classifier.parameters())
        print(f"Classification head params: {head_params:,}")
        print(f"Total trainable: {trainable_params + head_params:,}")

        # Load data
        data_dir = Path(args.data_dir)
        with open(data_dir / "labels.json") as f:
            labels_json = json.load(f)

        train_transform = None
        if args.augment:
            train_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ])

        train_ds = SafetyImageDataset(data_dir, "train", processor, labels_json, transform=train_transform)
        val_ds = SafetyImageDataset(data_dir, "val", processor, labels_json)

        if args.test_data_dir:
            test_data_dir = Path(args.test_data_dir)
            with open(test_data_dir / "labels.json") as f:
                test_labels = json.load(f)
            test_ds = SafetyImageDataset(test_data_dir, "test", processor, test_labels)
        else:
            test_ds = SafetyImageDataset(data_dir, "test", processor, labels_json)

        num_workers = args.num_workers
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

        # Training
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_f1 = 0
        best_state = None
        epoch_metrics_list = []
        start_time = time.time()

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            n_batches = 0

            for pixel_values, labels in train_loader:
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(pixel_values)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Validation
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for pv, lb in val_loader:
                    pv = pv.to(device)
                    out = model(pv)
                    val_preds.extend((out.cpu() > 0.5).numpy().astype(int))
                    val_targets.extend(lb.numpy())

            val_f1 = f1_score(val_targets, val_preds, zero_division=0)
            avg_loss = total_loss / max(n_batches, 1)

            # Per-epoch test metrics (optional)
            if args.save_epoch_metrics:
                test_epoch = evaluate_on_loader(model, test_loader, device)
                print(f"Epoch {epoch+1:2d}: loss={avg_loss:.4f}, val_f1={val_f1*100:.2f}%, test_f1={test_epoch['f1']*100:.2f}%")
                epoch_metrics_list.append({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "val_f1": float(val_f1),
                    "test_f1": test_epoch["f1"],
                })
                model.train()
            else:
                print(f"Epoch {epoch+1:2d}: loss={avg_loss:.4f}, val_f1={val_f1*100:.2f}%")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if args.save_checkpoints:
                    ckpt_path = output_dir / f"best_model_seed{seed}.pt"
                    torch.save(best_state, ckpt_path)
                    print(f"  Saved checkpoint: {ckpt_path}")

        training_time = time.time() - start_time

        # Save epoch metrics
        if args.save_epoch_metrics and epoch_metrics_list:
            epoch_metrics_path = output_dir / f"epoch_metrics_seed{seed}.json"
            with open(epoch_metrics_path, "w") as f:
                json.dump(epoch_metrics_list, f, indent=2)
            print(f"Epoch metrics saved to {epoch_metrics_path}")

        # Test evaluation (using best model)
        if best_state:
            model.load_state_dict(best_state)
        metrics = evaluate_on_loader(model, test_loader, device)

        print(f"\n--- Results (seed={seed}) ---")
        print(f"F1: {metrics['f1']*100:.2f}%  Acc: {metrics['accuracy']*100:.2f}%  AUC: {metrics['auc_roc']*100:.2f}%")
        print(f"Training time: {training_time:.1f}s")

        result = {
            "seed": seed,
            "method": "full_finetune" if args.full_finetune else ("head_only" if args.lora_r == 0 else "lora_finetune"),
            "model": "siglip",
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "trainable_params": trainable_params + head_params,
            "total_params": total_params + head_params,
            "training_time_seconds": training_time,
            "test_metrics": metrics,
        }
        all_results.append(result)

        # Free GPU memory for next seed
        del model, base_model
        torch.cuda.empty_cache()

    # Summary
    if len(all_results) > 1:
        f1s = [r["test_metrics"]["f1"] * 100 for r in all_results]
        print(f"\n{'='*60}")
        print(f"SUMMARY ({len(seeds)} seeds)")
        print(f"F1: {np.mean(f1s):.2f}±{np.std(f1s):.2f}%")
        print(f"{'='*60}")

    with open(output_dir / "lora_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'lora_results.json'}")


if __name__ == "__main__":
    main()
