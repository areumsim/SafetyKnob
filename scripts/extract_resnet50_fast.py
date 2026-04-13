#!/usr/bin/env python3
"""
Fast batch ResNet50 embedding extraction with DataLoader.

ResNet50 (ImageNet-pretrained) embeddings for CNN baseline comparison.
Uses batch processing (~50ms/img) vs single-image processing (~8s/img).

Usage:
    python3 scripts/extract_resnet50_fast.py \
        --data-dir data_scenario_v2 \
        --output embeddings/scenario_v2/resnet50 \
        --batch-size 64
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image), idx


def main():
    parser = argparse.ArgumentParser(description="Fast ResNet50 embedding extraction")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ResNet50 without final FC → 2048-dim embeddings
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model = nn.Sequential(*list(resnet.children())[:-1]).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

    # Load labels
    with open(data_dir / "labels.json") as f:
        all_labels = json.load(f)

    splits = sorted(set(k.split("/")[0] for k in all_labels))
    print(f"Model: ResNet50 (ImageNet-V2), Data: {data_dir}, Splits: {splits}")
    print(f"Device: {device}, Batch size: {args.batch_size}")

    total_start = time.time()

    for split in splits:
        split_labels = {k: v for k, v in all_labels.items() if k.startswith(f"{split}/")}
        filenames = []
        image_paths = []
        labels = {}

        for img_key, label_data in sorted(split_labels.items()):
            fname = img_key.split("/")[-1]
            img_path = data_dir / split / fname
            if img_path.exists():
                filenames.append(fname)
                image_paths.append(str(img_path))
                labels[fname] = label_data

        print(f"\n{'='*60}")
        print(f"Extracting {split}: {len(filenames)} images")

        dataset = ImageDataset(image_paths, transform)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                          num_workers=args.num_workers, pin_memory=True)

        all_embeddings = []
        start = time.time()

        with torch.no_grad():
            for i, (batch_imgs, batch_idx) in enumerate(loader):
                batch_imgs = batch_imgs.to(device)
                emb = model(batch_imgs).squeeze(-1).squeeze(-1)  # (B, 2048)
                all_embeddings.append(emb.cpu())
                if (i + 1) % 10 == 0:
                    done = (i + 1) * args.batch_size
                    elapsed = time.time() - start
                    print(f"  {done}/{len(filenames)} ({elapsed:.1f}s, {elapsed/done*1000:.1f}ms/img)")

        emb_tensor = torch.cat(all_embeddings, dim=0)
        elapsed = time.time() - start

        print(f"  Done: {emb_tensor.shape} in {elapsed:.1f}s ({elapsed/len(filenames)*1000:.1f}ms/img)")

        # Save in same format as other embedders
        torch.save({
            "embeddings": emb_tensor,
            "filenames": filenames,
            "embedding_dim": 2048,
            "model": "resnet50",
        }, output_dir / f"{split}_embeddings.pt")

        torch.save({
            "labels": {f: labels[f] for f in filenames},
            "filenames": filenames,
        }, output_dir / f"{split}_labels.pt")

        print(f"  Saved: {output_dir / f'{split}_embeddings.pt'}")

    total_elapsed = time.time() - total_start
    print(f"\nTotal: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
