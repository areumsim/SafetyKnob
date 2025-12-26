#!/usr/bin/env python3
"""
Baseline CNN Training Script for danger_al dataset

Trains traditional CNN models (ResNet-50, EfficientNet-B0) end-to-end
for binary classification to establish performance baselines.
"""

import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


# ========== Dataset ==========
class BaselineImageDataset(Dataset):
    """Dataset for baseline CNN models with image transformations"""

    def __init__(self, image_dir, labels_dict, transform=None):
        self.image_dir = Path(image_dir)
        self.labels_dict = labels_dict
        self.image_files = list(labels_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = self.image_dir / filename
        label = float(self.labels_dict[filename]['overall_safety'])

        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            # Return a dummy black image
            image = torch.zeros(3, 224, 224)

        return image, label


# ========== Model Architectures ==========
class BaselineCNN(nn.Module):
    """Wrapper for baseline CNN models with binary classification head"""

    def __init__(self, model_name='resnet50', pretrained=True):
        super().__init__()
        self.model_name = model_name

        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            # Replace final FC layer for binary classification
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )

        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            # Replace final classifier for binary classification
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def forward(self, x):
        return self.backbone(x).squeeze()


# ========== Training Functions ==========
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device).float()

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

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


def evaluate(model, dataloader, device):
    """Evaluate model on dataset"""
    model.eval()
    all_preds, all_targets, all_scores = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)

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
    parser = argparse.ArgumentParser(description='Baseline CNN training for danger_al')
    parser.add_argument('--model', type=str, required=True,
                       choices=['resnet50', 'efficientnet_b0'],
                       help='Baseline model to train')
    parser.add_argument('--data-dir', type=str, default='/workspace/arsim/EmoKnob/data',
                       help='Data directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for checkpoints and results')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use ImageNet pretrained weights')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"BASELINE TRAINING: {args.model.upper()}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {output_path}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"Pretrained: {args.pretrained}")

    # Data transformations (standard ImageNet preprocessing)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

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
    train_dataset = BaselineImageDataset(data_path / 'train', train_labels, transform=train_transform)
    val_dataset = BaselineImageDataset(data_path / 'val', val_labels, transform=val_transform)
    test_dataset = BaselineImageDataset(data_path / 'test', test_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Create model
    print(f"\nCreating {args.model} model...")
    model = BaselineCNN(model_name=args.model, pretrained=args.pretrained)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_f1 = 0
    patience = 7
    patience_counter = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
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
    test_metrics = evaluate(model, test_loader, device)

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
        'model_type': 'baseline_cnn',
        'pretrained': args.pretrained,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'training_time_seconds': training_time,
        'epochs_trained': epoch + 1,
        'best_val_f1': float(best_val_f1),
        'test_metrics': test_metrics,
        'hyperparameters': {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
            'weight_decay': 0.01,
            'patience': patience
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
