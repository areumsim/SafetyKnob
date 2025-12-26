"""
Multi-task Safety Classification Training

Trains a multi-task model with:
- 5 independent dimension heads (Fall, Collision, Equipment, Environment, PPE)
- 1 overall safety head
- Shared feature extractor with balanced multi-task loss
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel


# ========== Embedders ==========
class SigLIPEmbedder:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
        self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        self.embedding_dim = 1152
        self.model.eval()

    def extract_embedding(self, image_path):
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs.cpu().numpy().flatten()


class CLIPEmbedder:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.embedding_dim = 768
        self.model.eval()

    def extract_embedding(self, image_path):
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs.cpu().numpy().flatten()


class DINOv2Embedder:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = AutoModel.from_pretrained("facebook/dinov2-large").to(device)
        self.processor = AutoProcessor.from_pretrained("facebook/dinov2-large")
        self.embedding_dim = 1024
        self.model.eval()

    def extract_embedding(self, image_path):
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state
            embedding = outputs.mean(dim=1)
        return embedding.cpu().numpy().flatten()


# ========== Multi-task Model ==========
class MultiTaskSafetyClassifier(nn.Module):
    """Multi-task safety classifier with 5 dimension heads + 1 overall head"""

    def __init__(self, embedding_dim, hidden_dim=512):
        super().__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Dimension heads (5 independent binary classifiers)
        self.dimension_heads = nn.ModuleDict({
            'fall_hazard': self._make_head(hidden_dim),
            'collision_risk': self._make_head(hidden_dim),
            'equipment_hazard': self._make_head(hidden_dim),
            'environmental_risk': self._make_head(hidden_dim),
            'protective_gear': self._make_head(hidden_dim)
        })

        # Overall safety head
        self.overall_head = self._make_head(hidden_dim)

    def _make_head(self, hidden_dim):
        """Create a single binary classification head"""
        return nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Returns:
            dict with keys: 'fall_hazard', 'collision_risk', 'equipment_hazard',
                           'environmental_risk', 'protective_gear', 'overall_safety'
        """
        features = self.feature_extractor(x)

        outputs = {}
        for dim_name, head in self.dimension_heads.items():
            outputs[dim_name] = head(features).squeeze()

        outputs['overall_safety'] = self.overall_head(features).squeeze()

        return outputs


# ========== Dataset ==========
class MultiTaskDataset(Dataset):
    """Dataset for multi-task classification with 5D labels"""

    def __init__(self, image_dir, labels_dict):
        self.image_dir = Path(image_dir)
        self.labels_dict = labels_dict
        self.image_files = list(labels_dict.keys())

        self.dimension_names = [
            'fall_hazard', 'collision_risk', 'equipment_hazard',
            'environmental_risk', 'protective_gear'
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = self.image_dir / filename
        label_data = self.labels_dict[filename]

        # Extract labels
        overall_safety = float(label_data['overall_safety'])
        dimensions = label_data['dimensions']

        # Convert dimension dict to tensor
        dim_labels = torch.tensor([
            dimensions['fall_hazard'],
            dimensions['collision_risk'],
            dimensions['equipment_hazard'],
            dimensions['environmental_risk'],
            dimensions['protective_gear']
        ], dtype=torch.float32)

        return str(img_path), overall_safety, dim_labels


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


def compute_multitask_loss(outputs, overall_labels, dim_labels, alpha=0.5, beta=0.1):
    """
    Compute multi-task loss with weighting

    Args:
        outputs: dict with dimension predictions and overall prediction
        overall_labels: ground truth for overall safety [batch_size]
        dim_labels: ground truth for dimensions [batch_size, 5]
        alpha: weight for overall safety loss
        beta: weight for each dimension loss

    Loss = alpha * L_overall + sum(beta * L_dimension_i for applicable dimensions)

    Only compute dimension loss when label is not 0.9 (not applicable)
    """
    bce = nn.BCELoss(reduction='none')

    # Overall safety loss
    loss_overall = bce(outputs['overall_safety'], overall_labels).mean()

    # Dimension losses (only for applicable dimensions)
    dimension_names = ['fall_hazard', 'collision_risk', 'equipment_hazard',
                      'environmental_risk', 'protective_gear']

    loss_dims = 0
    applicable_count = 0

    for i, dim_name in enumerate(dimension_names):
        dim_target = dim_labels[:, i]
        dim_pred = outputs[dim_name]

        # Only compute loss where label is not 0.9 (applicable dimensions)
        applicable_mask = (dim_target != 0.9).float()

        if applicable_mask.sum() > 0:
            dim_loss = bce(dim_pred, dim_target)
            # Apply mask and normalize
            masked_loss = (dim_loss * applicable_mask).sum() / applicable_mask.sum()
            loss_dims += masked_loss
            applicable_count += 1

    # Average dimension loss
    if applicable_count > 0:
        loss_dims = loss_dims / applicable_count

    # Total loss
    total_loss = alpha * loss_overall + beta * loss_dims

    return total_loss, loss_overall.item(), loss_dims.item() if applicable_count > 0 else 0.0


def train_epoch(model, dataloader, embedder, optimizer, device, alpha=0.5, beta=0.1):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_overall_loss = 0
    total_dim_loss = 0
    all_preds, all_targets = [], []

    for batch_idx, (image_paths, overall_labels, dim_labels) in enumerate(dataloader):
        # Extract embeddings
        embeddings = extract_embeddings_batch(embedder, image_paths, device)
        overall_labels = overall_labels.to(device).float()
        dim_labels = dim_labels.to(device).float()

        # Forward pass
        optimizer.zero_grad()
        outputs = model(embeddings)

        # Calculate loss
        loss, loss_overall, loss_dims = compute_multitask_loss(
            outputs, overall_labels, dim_labels, alpha, beta
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_overall_loss += loss_overall
        total_dim_loss += loss_dims
        all_preds.extend(outputs['overall_safety'].detach().cpu().numpy())
        all_targets.extend(overall_labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f} "
                  f"(Overall: {loss_overall:.4f}, Dims: {loss_dims:.4f})")

    avg_loss = total_loss / len(dataloader)
    avg_overall_loss = total_overall_loss / len(dataloader)
    avg_dim_loss = total_dim_loss / len(dataloader)
    preds_binary = (np.array(all_preds) > 0.5).astype(int)
    acc = accuracy_score(all_targets, preds_binary)

    return avg_loss, avg_overall_loss, avg_dim_loss, acc


def evaluate(model, dataloader, embedder, device):
    """Evaluate model on dataset"""
    model.eval()

    dimension_names = ['fall_hazard', 'collision_risk', 'equipment_hazard',
                      'environmental_risk', 'protective_gear']

    # Collect predictions and targets
    overall_preds, overall_targets, overall_scores = [], [], []
    dim_preds = {dim: [] for dim in dimension_names}
    dim_targets = {dim: [] for dim in dimension_names}
    dim_scores = {dim: [] for dim in dimension_names}

    with torch.no_grad():
        for image_paths, overall_labels, dim_labels in dataloader:
            embeddings = extract_embeddings_batch(embedder, image_paths, device)
            outputs = model(embeddings)

            # Overall safety
            overall_scores.extend(outputs['overall_safety'].cpu().numpy())
            overall_preds.extend((outputs['overall_safety'] > 0.5).cpu().numpy().astype(int))
            overall_targets.extend(overall_labels.numpy())

            # Dimensions (only collect applicable ones)
            for i, dim_name in enumerate(dimension_names):
                dim_target = dim_labels[:, i].numpy()
                dim_score = outputs[dim_name].cpu().numpy()
                dim_pred = (dim_score > 0.5).astype(int)

                # Only add applicable samples (label != 0.9)
                applicable_mask = dim_target != 0.9

                if applicable_mask.sum() > 0:
                    # Convert targets to int for binary classification metrics
                    dim_targets[dim_name].extend(dim_target[applicable_mask].astype(int))
                    dim_scores[dim_name].extend(dim_score[applicable_mask])
                    dim_preds[dim_name].extend(dim_pred[applicable_mask])

    # Overall safety metrics
    metrics = {
        'overall': {
            'accuracy': float(accuracy_score(overall_targets, overall_preds)),
            'f1': float(f1_score(overall_targets, overall_preds, zero_division=0)),
            'precision': float(precision_score(overall_targets, overall_preds, zero_division=0)),
            'recall': float(recall_score(overall_targets, overall_preds, zero_division=0)),
            'auc_roc': float(roc_auc_score(overall_targets, overall_scores))
        }
    }

    # Dimension-specific metrics
    for dim_name in dimension_names:
        if len(dim_targets[dim_name]) > 0:
            metrics[dim_name] = {
                'accuracy': float(accuracy_score(dim_targets[dim_name], dim_preds[dim_name])),
                'f1': float(f1_score(dim_targets[dim_name], dim_preds[dim_name], zero_division=0)),
                'precision': float(precision_score(dim_targets[dim_name], dim_preds[dim_name], zero_division=0)),
                'recall': float(recall_score(dim_targets[dim_name], dim_preds[dim_name], zero_division=0)),
                'auc_roc': float(roc_auc_score(dim_targets[dim_name], dim_scores[dim_name])),
                'sample_count': len(dim_targets[dim_name])
            }
        else:
            metrics[dim_name] = None

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Multi-task training for 5D safety classification')
    parser.add_argument('--model', type=str, required=True,
                       choices=['siglip', 'clip', 'dinov2'],
                       help='Model to train')
    parser.add_argument('--data-dir', type=str, default='data_scenario',
                       help='Data directory')
    parser.add_argument('--labels-file', type=str, default='data_scenario/labels_5d.json',
                       help='5D labels JSON file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for checkpoints and results')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Weight for overall safety loss')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='Weight for dimension losses')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"MULTI-TASK TRAINING: {args.model.upper()}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Data: {args.data_dir}")
    print(f"Labels: {args.labels_file}")
    print(f"Output: {output_path}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"Loss weights: α={args.alpha} (overall), β={args.beta} (dimensions)")

    # Load labels
    labels_file = Path(args.labels_file)
    print(f"\nLoading 5D labels from {labels_file}...")
    with open(labels_file) as f:
        all_labels = json.load(f)

    # Split by prefix
    train_labels = {k.replace('train/', ''): v for k, v in all_labels.items() if k.startswith('train/')}
    val_labels = {k.replace('val/', ''): v for k, v in all_labels.items() if k.startswith('val/')}
    test_labels = {k.replace('test/', ''): v for k, v in all_labels.items() if k.startswith('test/')}

    print(f"Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")

    # Create datasets
    data_path = Path(args.data_dir)
    train_dataset = MultiTaskDataset(data_path / 'train', train_labels)
    val_dataset = MultiTaskDataset(data_path / 'val', val_labels)
    test_dataset = MultiTaskDataset(data_path / 'test', test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize embedder
    print(f"\nLoading {args.model.upper()} embedder...")
    if args.model == 'siglip':
        embedder = SigLIPEmbedder(device)
    elif args.model == 'clip':
        embedder = CLIPEmbedder(device)
    elif args.model == 'dinov2':
        embedder = DINOv2Embedder(device)

    print(f"Embedding dimension: {embedder.embedding_dim}")

    # Initialize model
    model = MultiTaskSafetyClassifier(
        embedding_dim=embedder.embedding_dim,
        hidden_dim=512
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("\n" + "=" * 60)
    print("TRAINING START")
    print("=" * 60)

    best_val_f1 = 0
    best_epoch = 0
    train_history = []

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        print(f"\n[Epoch {epoch+1}/{args.epochs}]")

        # Train
        train_loss, train_overall_loss, train_dim_loss, train_acc = train_epoch(
            model, train_loader, embedder, optimizer, device, args.alpha, args.beta
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, embedder, device)

        epoch_time = time.time() - epoch_start

        # Log results
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (Overall: {train_overall_loss:.4f}, Dims: {train_dim_loss:.4f})")
        print(f"  Train Acc: {train_acc:.4f}")
        print(f"  Val Overall - Acc: {val_metrics['overall']['accuracy']:.4f}, "
              f"F1: {val_metrics['overall']['f1']:.4f}, "
              f"AUC: {val_metrics['overall']['auc_roc']:.4f}")

        print(f"  Val Dimensions:")
        for dim_name in ['fall_hazard', 'collision_risk', 'equipment_hazard',
                         'environmental_risk', 'protective_gear']:
            if val_metrics[dim_name] is not None:
                m = val_metrics[dim_name]
                print(f"    {dim_name:24s}: F1={m['f1']:.4f}, Acc={m['accuracy']:.4f}, "
                      f"AUC={m['auc_roc']:.4f} (n={m['sample_count']})")

        print(f"  Time: {epoch_time:.1f}s")

        # Save history
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_overall_loss': train_overall_loss,
            'train_dim_loss': train_dim_loss,
            'train_acc': train_acc,
            'val_metrics': val_metrics,
            'epoch_time': epoch_time
        })

        # Save best model
        val_f1 = val_metrics['overall']['f1']
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'embedding_dim': embedder.embedding_dim
            }, output_path / 'best_model.pt')

            print(f"  ✅ Best model saved (F1: {val_f1:.4f})")

    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time/3600:.2f}h")
    print(f"Best epoch: {best_epoch}, F1: {best_val_f1:.4f}")

    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(output_path / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, embedder, device)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Overall Safety:")
    print(f"  Accuracy: {test_metrics['overall']['accuracy']:.4f}")
    print(f"  F1 Score: {test_metrics['overall']['f1']:.4f}")
    print(f"  Precision: {test_metrics['overall']['precision']:.4f}")
    print(f"  Recall: {test_metrics['overall']['recall']:.4f}")
    print(f"  AUC-ROC: {test_metrics['overall']['auc_roc']:.4f}")

    print(f"\nDimension-Specific Results:")
    for dim_name in ['fall_hazard', 'collision_risk', 'equipment_hazard',
                     'environmental_risk', 'protective_gear']:
        if test_metrics[dim_name] is not None:
            m = test_metrics[dim_name]
            print(f"\n{dim_name.upper().replace('_', ' ')}:")
            print(f"  Accuracy: {m['accuracy']:.4f}")
            print(f"  F1 Score: {m['f1']:.4f}")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall: {m['recall']:.4f}")
            print(f"  AUC-ROC: {m['auc_roc']:.4f}")
            print(f"  Samples: {m['sample_count']}")

    # Save results
    final_results = {
        'model': args.model,
        'training_time_hours': total_time / 3600,
        'best_epoch': best_epoch,
        'test_metrics': test_metrics,
        'train_history': train_history,
        'config': vars(args)
    }

    with open(output_path / 'results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to {output_path / 'results.json'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
