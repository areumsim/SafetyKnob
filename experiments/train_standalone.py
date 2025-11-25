#!/usr/bin/env python3
"""
Standalone Training Script (No torchvision dependency)

Trains a model without relying on existing src imports that have torchvision issues.
"""

import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel


# ========== Embedders ==========
class SigLIPEmbedder:
    def __init__(self, model_name="google/siglip-so400m-patch14-384", device="cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim = 1152

    @torch.no_grad()
    def extract_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        return outputs.cpu().numpy().flatten()


class CLIPEmbedder:
    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim = 768

    @torch.no_grad()
    def extract_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        return outputs.cpu().numpy().flatten()


class DINOv2Embedder:
    def __init__(self, model_name="facebook/dinov2-large", device="cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim = 1024

    @torch.no_grad()
    def extract_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs).last_hidden_state
        return outputs.mean(dim=1).cpu().numpy().flatten()


# ========== Neural Classifier ==========
class SafetyClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=512, num_dimensions=5):
        super().__init__()
        self.dimension_names = ['fall_hazard', 'collision_risk', 'equipment_hazard',
                               'environmental_risk', 'protective_gear']

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Overall safety head
        self.safety_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Dimension heads
        self.dimension_heads = nn.ModuleDict({
            dim: nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ) for dim in self.dimension_names
        })

    def forward(self, x):
        features = self.feature_extractor(x)
        overall = self.safety_head(features).squeeze()

        dimensions = {}
        for dim_name, head in self.dimension_heads.items():
            dimensions[dim_name] = head(features).squeeze()

        return {'overall_safety': overall, 'dimensions': dimensions}


# ========== Dataset ==========
class ImageDatasetFromLabels(Dataset):
    def __init__(self, image_dir, labels_dict):
        self.image_dir = Path(image_dir)
        self.labels_dict = labels_dict
        self.image_files = list(labels_dict.keys())
        self.dimension_names = ['fall_hazard', 'collision_risk', 'equipment_hazard',
                               'environmental_risk', 'protective_gear']

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = self.image_dir / filename
        labels = self.labels_dict[filename]

        overall = float(labels['overall_safety'])
        dim_labels = [float(labels[dim]) for dim in self.dimension_names]

        return str(img_path), overall, torch.tensor(dim_labels, dtype=torch.float32)


# ========== Training Functions ==========
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_embeddings_batch(embedder, image_paths, device):
    embeddings = []
    for img_path in image_paths:
        try:
            emb = embedder.extract_embedding(img_path)
            embeddings.append(torch.tensor(emb, dtype=torch.float32))
        except Exception as e:
            print(f"Warning: Failed to process {img_path}: {e}")
            embeddings.append(torch.zeros(embedder.embedding_dim, dtype=torch.float32))

    return torch.stack(embeddings).to(device)


def train_epoch(model, dataloader, embedder, criterion, optimizer, device):
    model.train()
    total_loss = 0
    overall_preds, overall_targets = [], []

    for batch_idx, (image_paths, overall_labels, dim_labels) in enumerate(dataloader):
        embeddings = extract_embeddings_batch(embedder, image_paths, device)
        overall_labels = overall_labels.to(device).float()
        dim_labels = dim_labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(embeddings)

        overall_loss = criterion(outputs['overall_safety'], overall_labels)

        dim_loss = 0
        for i, dim_name in enumerate(['fall_hazard', 'collision_risk', 'equipment_hazard',
                                      'environmental_risk', 'protective_gear']):
            dim_loss += criterion(outputs['dimensions'][dim_name], dim_labels[:, i])

        loss = overall_loss + 0.5 * dim_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        overall_preds.extend(outputs['overall_safety'].detach().cpu().numpy())
        overall_targets.extend(overall_labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    overall_preds_binary = (np.array(overall_preds) > 0.5).astype(int)
    acc = accuracy_score(overall_targets, overall_preds_binary)

    return avg_loss, acc


def evaluate(model, dataloader, embedder, device):
    model.eval()
    overall_preds, overall_targets, overall_scores = [], [], []
    dim_preds = {dim: [] for dim in ['fall_hazard', 'collision_risk', 'equipment_hazard',
                                     'environmental_risk', 'protective_gear']}
    dim_targets = {dim: [] for dim in dim_preds.keys()}

    with torch.no_grad():
        for image_paths, overall_labels, dim_labels in dataloader:
            embeddings = extract_embeddings_batch(embedder, image_paths, device)

            outputs = model(embeddings)

            overall_scores.extend(outputs['overall_safety'].cpu().numpy())
            overall_preds.extend((outputs['overall_safety'] > 0.5).cpu().numpy().astype(int))
            overall_targets.extend(overall_labels.numpy())

            for i, dim_name in enumerate(dim_preds.keys()):
                dim_preds[dim_name].extend((outputs['dimensions'][dim_name] > 0.5).cpu().numpy().astype(int))
                dim_targets[dim_name].extend(dim_labels[:, i].numpy())

    metrics = {
        'overall': {
            'accuracy': float(accuracy_score(overall_targets, overall_preds)),
            'f1': float(f1_score(overall_targets, overall_preds, zero_division=0)),
            'precision': float(precision_score(overall_targets, overall_preds, zero_division=0)),
            'recall': float(recall_score(overall_targets, overall_preds, zero_division=0)),
            'auc_roc': float(roc_auc_score(overall_targets, overall_scores))
        },
        'dimensions': {}
    }

    for dim_name in dim_preds.keys():
        metrics['dimensions'][dim_name] = {
            'f1': float(f1_score(dim_targets[dim_name], dim_preds[dim_name], zero_division=0)),
            'accuracy': float(accuracy_score(dim_targets[dim_name], dim_preds[dim_name]))
        }

    metrics['avg_dimension_f1'] = float(np.mean([m['f1'] for m in metrics['dimensions'].values()]))

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Standalone training')
    parser.add_argument('--model', type=str, required=True, choices=['siglip', 'clip', 'dinov2'])
    parser.add_argument('--data-dir', type=str, default='data/processed/')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Training: {args.model.upper()}")
    print("=" * 60)

    # Load labels
    data_path = Path(args.data_dir)
    with open(data_path / 'train_labels.json') as f:
        train_labels = json.load(f)
    with open(data_path / 'val_labels.json') as f:
        val_labels = json.load(f)
    with open(data_path / 'test_labels.json') as f:
        test_labels = json.load(f)

    print(f"Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")

    # Create datasets
    train_dataset = ImageDatasetFromLabels(data_path / 'train', train_labels)
    val_dataset = ImageDatasetFromLabels(data_path / 'val', val_labels)
    test_dataset = ImageDatasetFromLabels(data_path / 'test', test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create embedder
    print(f"\nLoading {args.model} embedder...")
    if args.model == 'siglip':
        embedder = SigLIPEmbedder(device=device)
    elif args.model == 'clip':
        embedder = CLIPEmbedder(device=device)
    elif args.model == 'dinov2':
        embedder = DINOv2Embedder(device=device)

    print(f"✓ Embedder loaded (dim={embedder.embedding_dim})")

    # Create model
    model = SafetyClassifier(embedding_dim=embedder.embedding_dim, hidden_dim=512, num_dimensions=5)
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    print("\nStarting training...")
    best_val_f1 = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, embedder, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, embedder, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val - F1: {val_metrics['overall']['f1']:.4f}, AUC: {val_metrics['overall']['auc_roc']:.4f}, Avg Dim F1: {val_metrics['avg_dimension_f1']:.4f}")

        if val_metrics['overall']['f1'] > best_val_f1:
            best_val_f1 = val_metrics['overall']['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': best_val_f1,
            }, output_path / 'best_model.pt')
            print(f"✓ Saved best model (F1: {best_val_f1:.4f})")

    training_time = time.time() - start_time

    # Test evaluation
    print("\n" + "=" * 60)
    print("Testing...")
    checkpoint = torch.load(output_path / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, embedder, device)

    print(f"\nTest Results:")
    print(f"Overall - Acc: {test_metrics['overall']['accuracy']:.4f}, F1: {test_metrics['overall']['f1']:.4f}, AUC: {test_metrics['overall']['auc_roc']:.4f}")
    print(f"Dimensions:")
    for dim_name, metrics in test_metrics['dimensions'].items():
        print(f"  {dim_name}: F1={metrics['f1']:.4f}")
    print(f"Avg Dimension F1: {test_metrics['avg_dimension_f1']:.4f}")

    # Save results
    results = {
        'model': args.model,
        'embedding_dim': embedder.embedding_dim,
        'training_time_seconds': training_time,
        'test_metrics': test_metrics
    }

    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {output_path / 'results.json'}")


if __name__ == '__main__':
    main()
