#!/usr/bin/env python3
"""
Embedding Pre-extraction Script

Extracts and caches embeddings from foundation models (SigLIP, CLIP, DINOv2)
so that probe training runs in seconds instead of hours.

Usage:
    python3 scripts/extract_embeddings.py \
        --model siglip \
        --data-dir data_scenario \
        --output embeddings/scenario/siglip
"""

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel


# ========== Embedders ==========
# Compatible with transformers 5.x where get_image_features returns
# BaseModelOutputWithPooling instead of a raw tensor.

def _to_tensor(output):
    """Extract tensor from model output (handles both old and new transformers API)."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, 'image_embeds'):
        return output.image_embeds
    raise ValueError(f"Cannot extract tensor from {type(output)}")


class SigLIPEmbedder:
    def __init__(self, device="cuda"):
        from transformers import SiglipImageProcessor, SiglipVisionModel
        self.device = device
        self.model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
        self.processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        self.model.eval()
        self.embedding_dim = 1152

    @torch.no_grad()
    def extract_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs.pooler_output.cpu().squeeze(0)


class CLIPEmbedder:
    def __init__(self, device="cuda"):
        from transformers import CLIPVisionModel, CLIPImageProcessor
        self.device = device
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.eval()
        self.embedding_dim = 768

    @torch.no_grad()
    def extract_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs.pooler_output.cpu().squeeze(0)


class DINOv2Embedder:
    def __init__(self, device="cuda"):
        from transformers import AutoImageProcessor, Dinov2Model
        self.device = device
        self.model = Dinov2Model.from_pretrained("facebook/dinov2-large").to(device)
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
        self.model.eval()
        self.embedding_dim = 1024

    @torch.no_grad()
    def extract_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs).last_hidden_state
        return outputs.mean(dim=1).cpu().squeeze(0)


class ResNet50Embedder:
    """ResNet50 (ImageNet pretrained) as CNN baseline embedder."""
    def __init__(self, device="cuda"):
        import torchvision.models as models
        from torchvision import transforms
        self.device = device
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove final FC layer to get 2048-dim embeddings
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
        self.model.eval()
        self.embedding_dim = 2048
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.transform(image).unsqueeze(0).to(self.device)
        outputs = self.model(inputs)
        return outputs.cpu().squeeze()


def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from foundation models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['siglip', 'clip', 'dinov2', 'resnet50'],
                       help='Foundation model to use')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Data directory with train/val/test subdirs')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for embeddings')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    labels_file = data_dir / 'labels.json'
    with open(labels_file) as f:
        all_labels = json.load(f)

    # Detect splits present in the data
    splits = set()
    for key in all_labels:
        split = key.split('/')[0]
        splits.add(split)
    splits = sorted(splits)

    print(f"Model: {args.model}, Data: {data_dir}, Splits: {splits}")
    print(f"Output: {output_dir}")

    # Create embedder
    if args.model == 'siglip':
        embedder = SigLIPEmbedder(device=device)
    elif args.model == 'clip':
        embedder = CLIPEmbedder(device=device)
    elif args.model == 'dinov2':
        embedder = DINOv2Embedder(device=device)
    elif args.model == 'resnet50':
        embedder = ResNet50Embedder(device=device)

    print(f"Embedding dim: {embedder.embedding_dim}")

    # Extract embeddings for each split
    total_start = time.time()

    for split in splits:
        split_labels = {k: v for k, v in all_labels.items() if k.startswith(f'{split}/')}
        print(f"\n{'='*60}")
        print(f"Extracting {split}: {len(split_labels)} images")
        print(f"{'='*60}")

        embeddings = {}
        labels = {}
        errors = 0

        for img_key, label_data in tqdm(split_labels.items(), desc=split):
            filename = img_key.split('/')[-1]
            # Try to find the image in the split subdir
            img_path = data_dir / split / filename

            if not img_path.exists():
                # Fallback: check all subdirs
                found = False
                for subdir in ['train', 'val', 'test']:
                    alt = data_dir / subdir / filename
                    if alt.exists():
                        img_path = alt
                        found = True
                        break
                if not found:
                    errors += 1
                    continue

            try:
                emb = embedder.extract_embedding(str(img_path))
                embeddings[filename] = emb
                labels[filename] = label_data
            except Exception as e:
                print(f"\nError: {img_path}: {e}")
                errors += 1

        if errors > 0:
            print(f"  Errors: {errors}")

        # Save
        emb_file = output_dir / f'{split}_embeddings.pt'
        labels_file_out = output_dir / f'{split}_labels.pt'

        # Stack embeddings into tensor + keep filename order
        filenames = list(embeddings.keys())
        emb_tensor = torch.stack([embeddings[f] for f in filenames])

        torch.save({
            'embeddings': emb_tensor,
            'filenames': filenames,
            'embedding_dim': embedder.embedding_dim,
            'model': args.model,
        }, emb_file)

        torch.save({
            'labels': {f: labels[f] for f in filenames},
            'filenames': filenames,
        }, labels_file_out)

        print(f"  Saved: {emb_file} ({emb_tensor.shape})")
        print(f"  Saved: {labels_file_out} ({len(filenames)} labels)")

    total_time = time.time() - total_start
    print(f"\nTotal extraction time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == '__main__':
    main()
