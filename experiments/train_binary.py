#!/usr/bin/env python3
"""
Binary-only Training Script for danger_al dataset

Trains models on binary classification (safe vs unsafe) without 5D dimension labels.
Simplified version for faster training.
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


# ========== Binary Classifier (Simplified) ==========
class BinarySafetyClassifier(nn.Module):
    """Binary-only safety classifier (no dimension heads)"""

    def __init__(self, embedding_dim, hidden_dim=512):
        super().__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Binary safety head
        self.safety_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        overall_safety = self.safety_head(features).squeeze()
        return overall_safety


# ========== Dataset ==========
class BinaryImageDataset(Dataset):
    """Dataset for binary classification"""

    def __init__(self, image_dir, labels_dict):
        self.image_dir = Path(image_dir)
        self.labels_dict = labels_dict
        self.image_files = list(labels_dict.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = self.image_dir / filename
        label = float(self.labels_dict[filename]['overall_safety'])

        return str(img_path), label


# ========== Training Functions ==========
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_embeddings_batch(embedder, image_paths, device):
    """Extract embeddings for a batch of images"""
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
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    for batch_idx, (image_paths, labels) in enumerate(dataloader):
        # Extract embeddings
        embeddings = extract_embeddings_batch(embedder, image_paths, device)
        labels = labels.to(device).float()

        # Forward pass
        optimizer.zero_grad()
        outputs = model(embeddings)

        # Calculate loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    acc = accuracy_score(all_targets, preds_binary)

    return avg_loss, acc


def evaluate(model, dataloader, embedder, device):
    """Evaluate model on dataset"""
    model.eval()
    all_preds, all_targets, all_scores = [], [], []

    with torch.no_grad():
        for image_paths, labels in dataloader:
            embeddings = extract_embeddings_batch(embedder, image_paths, device)
            outputs = model(embeddings)

            all_scores.extend(outputs.cpu().numpy())
            all_preds.extend((outputs > 0.5).cpu().numpy().astype(int))
            all_targets.extend(labels.numpy())

    metrics = {
        'accuracy': float(accuracy_score(all_targets, all_preds)),
        'f1': float(f1_score(all_targets, all_preds, zero_division=0)),
        'precision': float(precision_score(all_targets, all_preds, zero_division=0)),
        'recall': float(recall_score(all_targets, all_preds, zero_division=0)),
        'auc_roc': float(roc_auc_score(all_targets, all_scores))
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Binary-only training for danger_al')
    parser.add_argument('--model', type=str, required=True,
                       choices=['siglip', 'clip', 'dinov2'],
                       help='Model to train')
    parser.add_argument('--data-dir', type=str, default='/workspace/arsim/EmoKnob/data',
                       help='Data directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for checkpoints and results')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"BINARY TRAINING: {args.model.upper()}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {output_path}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    # Load labels
    data_path = Path(args.data_dir)
    labels_file = data_path / 'labels.json'

    print(f"\nLoading labels from {labels_file}...")
    with open(labels_file) as f:
        all_labels = json.load(f)

    # Split by prefix
    train_labels = {k.replace('train/', ''): v for k, v in all_labels.items() if k.startswith('train/')}
    val_labels = {k.replace('val/', ''): v for k, v in all_labels.items() if k.startswith('val/')}
    test_labels = {k.replace('test/', ''): v for k, v in all_labels.items() if k.startswith('test/')}

    print(f"  Train: {len(train_labels)} images")
    print(f"  Val:   {len(val_labels)} images")
    print(f"  Test:  {len(test_labels)} images")

    # Count safe/unsafe
    train_safe = sum(1 for v in train_labels.values() if v['overall_safety'] == 1)
    val_safe = sum(1 for v in val_labels.values() if v['overall_safety'] == 1)
    test_safe = sum(1 for v in test_labels.values() if v['overall_safety'] == 1)

    print(f"\nClass distribution:")
    print(f"  Train: {train_safe} safe ({train_safe/len(train_labels)*100:.1f}%), {len(train_labels)-train_safe} unsafe")
    print(f"  Val:   {val_safe} safe ({val_safe/len(val_labels)*100:.1f}%), {len(val_labels)-val_safe} unsafe")
    print(f"  Test:  {test_safe} safe ({test_safe/len(test_labels)*100:.1f}%), {len(test_labels)-test_safe} unsafe")

    # Create datasets
    train_dataset = BinaryImageDataset(data_path / 'train', train_labels)
    val_dataset = BinaryImageDataset(data_path / 'val', val_labels)
    test_dataset = BinaryImageDataset(data_path / 'test', test_labels)

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

    print(f"✓ Embedder loaded (embedding_dim={embedder.embedding_dim})")

    # Create model
    print(f"\nCreating binary classifier...")
    model = BinarySafetyClassifier(embedding_dim=embedder.embedding_dim, hidden_dim=512)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created ({total_params:,} parameters)")

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_f1 = 0
    patience = 5
    patience_counter = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, embedder, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, embedder, device)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc_roc']:.4f}")

        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_metrics': val_metrics
            }, output_path / 'best_model.pt')

            print(f"  ✓ Saved best model (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    training_time = time.time() - start_time

    # Test evaluation
    print("\n" + "=" * 60)
    print("TESTING")
    print("=" * 60)

    checkpoint = torch.load(output_path / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, embedder, device)

    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score:  {test_metrics['f1']*100:.2f}%")
    print(f"  Precision: {test_metrics['precision']*100:.2f}%")
    print(f"  Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"  AUC-ROC:   {test_metrics['auc_roc']*100:.2f}%")

    # Save results
    results = {
        'model': args.model,
        'dataset': 'danger_al',
        'task': 'binary_classification',
        'embedding_dim': embedder.embedding_dim,
        'training_time_seconds': training_time,
        'epochs_trained': epoch + 1,
        'best_val_f1': float(best_val_f1),
        'test_metrics': test_metrics,
        'hyperparameters': {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'hidden_dim': 512,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR'
        }
    }

    results_file = output_path / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {results_file}")
    print(f"✅ Best model saved to {output_path / 'best_model.pt'}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {training_time:.1f}s ({training_time/60:.1f} minutes)")


if __name__ == '__main__':
    main()
