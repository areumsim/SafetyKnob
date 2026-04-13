"""
t-SNE Visualization of Safety Classification Embeddings

Visualizes the embedding space to demonstrate linear separability
of Safe/Unsafe classes using Vision Foundation Models.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel


class SafetyImageDataset(Dataset):
    """Dataset for loading safety images with labels"""

    def __init__(self, data_dir: Path, labels_file: Path, split='test', max_samples=None):
        self.data_dir = data_dir

        with open(labels_file, 'r') as f:
            all_labels = json.load(f)

        # Filter by split
        split_labels = {k.replace(f'{split}/', ''): v for k, v in all_labels.items() if k.startswith(f'{split}/')}

        self.image_paths = [data_dir / split / img_name for img_name in split_labels.keys()]
        self.labels = [split_labels[img_name]['overall_safety'] for img_name in split_labels.keys()]
        self.classes = [split_labels[img_name]['class'] for img_name in split_labels.keys()]

        # Limit samples if specified
        if max_samples and max_samples < len(self.image_paths):
            indices = np.random.choice(len(self.image_paths), max_samples, replace=False)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            self.classes = [self.classes[i] for i in indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return {
            'image_path': str(self.image_paths[idx]),
            'label': self.labels[idx],
            'class': self.classes[idx]
        }


class EmbedderWrapper:
    """Wrapper for different vision model embedders"""

    def __init__(self, model_name: str, device='cuda'):
        self.device = device
        self.model_name = model_name

        if model_name == 'siglip':
            self.model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
            self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
            self.embedding_dim = 1152
        elif model_name == 'clip':
            self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.embedding_dim = 768
        elif model_name == 'dinov2':
            self.model = AutoModel.from_pretrained("facebook/dinov2-large").to(device)
            self.processor = AutoProcessor.from_pretrained("facebook/dinov2-large")
            self.embedding_dim = 1024
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model.eval()

    @torch.no_grad()
    def embed_image(self, image_path: str):
        """Extract embedding from image"""
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        if self.model_name == 'siglip':
            outputs = self.model.get_image_features(**inputs)
        elif self.model_name == 'clip':
            outputs = self.model.get_image_features(**inputs)
        elif self.model_name == 'dinov2':
            outputs = self.model(**inputs).last_hidden_state.mean(dim=1)

        return outputs.cpu().numpy().flatten()


def extract_embeddings(model_name: str, data_dir: Path, labels_file: Path,
                       split='test', max_samples=1000, device='cuda'):
    """Extract embeddings for all images"""
    print(f"\n{'='*60}")
    print(f"Extracting {model_name.upper()} Embeddings")
    print(f"{'='*60}\n")

    # Load dataset
    dataset = SafetyImageDataset(data_dir, labels_file, split, max_samples)
    print(f"Loaded {len(dataset)} images")
    print(f"  Safe: {sum(dataset.labels)} ({sum(dataset.labels)/len(dataset.labels)*100:.1f}%)")
    print(f"  Unsafe: {len(dataset.labels) - sum(dataset.labels)} ({(len(dataset.labels)-sum(dataset.labels))/len(dataset.labels)*100:.1f}%)")

    # Initialize embedder
    embedder = EmbedderWrapper(model_name, device)
    print(f"Embedding dim: {embedder.embedding_dim}")

    # Extract embeddings
    embeddings = []
    labels = []
    classes = []

    print("\nExtracting embeddings...")
    for item in tqdm(dataset, desc="Processing"):
        try:
            embedding = embedder.embed_image(item['image_path'])
            embeddings.append(embedding)
            labels.append(item['label'])
            classes.append(item['class'])
        except Exception as e:
            print(f"Error with {item['image_path']}: {e}")
            continue

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    print(f"\nExtracted {len(embeddings)} embeddings")
    print(f"  Shape: {embeddings.shape}")

    return embeddings, labels, classes


def create_tsne_visualization(embeddings, labels, classes, model_name,
                              output_dir: Path, n_components=2, perplexity=30):
    """Create t-SNE visualization"""
    print(f"\n{'='*60}")
    print(f"Creating t-SNE Visualization ({n_components}D)")
    print(f"{'='*60}\n")

    # Run t-SNE
    print(f"Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42,
        n_iter=1000,
        verbose=1
    )
    embeddings_tsne = tsne.fit_transform(embeddings)

    print(f"t-SNE complete. Shape: {embeddings_tsne.shape}")

    # Prepare data for plotting
    safe_mask = labels == 1
    unsafe_mask = labels == 0

    caution_mask = np.array([c == 'caution' for c in classes])
    safe_caution_mask = safe_mask & caution_mask
    unsafe_caution_mask = unsafe_mask & caution_mask
    safe_normal_mask = safe_mask & ~caution_mask
    unsafe_normal_mask = unsafe_mask & ~caution_mask

    if n_components == 2:
        # 2D visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Plot safe/unsafe points
        ax.scatter(embeddings_tsne[safe_normal_mask, 0], embeddings_tsne[safe_normal_mask, 1],
                  c='green', marker='o', s=30, alpha=0.6, label=f'Safe ({sum(safe_normal_mask)})')
        ax.scatter(embeddings_tsne[unsafe_normal_mask, 0], embeddings_tsne[unsafe_normal_mask, 1],
                  c='red', marker='o', s=30, alpha=0.6, label=f'Unsafe ({sum(unsafe_normal_mask)})')

        # Plot caution points if present
        if sum(caution_mask) > 0:
            ax.scatter(embeddings_tsne[safe_caution_mask, 0], embeddings_tsne[safe_caution_mask, 1],
                      c='lightgreen', marker='x', s=50, alpha=0.8, label=f'Caution→Safe ({sum(safe_caution_mask)})')
            ax.scatter(embeddings_tsne[unsafe_caution_mask, 0], embeddings_tsne[unsafe_caution_mask, 1],
                      c='orange', marker='x', s=50, alpha=0.8, label=f'Caution→Unsafe ({sum(unsafe_caution_mask)})')

        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title(f'{model_name.upper()} Embedding Space (t-SNE 2D)\nLinear Separability of Safe/Unsafe Classes',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = output_dir / f'tsne_2d_{model_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved 2D visualization to {output_file}")
        plt.close()

    elif n_components == 3:
        # 3D visualization
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot safe/unsafe points
        ax.scatter(embeddings_tsne[safe_normal_mask, 0],
                  embeddings_tsne[safe_normal_mask, 1],
                  embeddings_tsne[safe_normal_mask, 2],
                  c='green', marker='o', s=20, alpha=0.6, label=f'Safe ({sum(safe_normal_mask)})')
        ax.scatter(embeddings_tsne[unsafe_normal_mask, 0],
                  embeddings_tsne[unsafe_normal_mask, 1],
                  embeddings_tsne[unsafe_normal_mask, 2],
                  c='red', marker='o', s=20, alpha=0.6, label=f'Unsafe ({sum(unsafe_normal_mask)})')

        # Plot caution points if present
        if sum(caution_mask) > 0:
            ax.scatter(embeddings_tsne[safe_caution_mask, 0],
                      embeddings_tsne[safe_caution_mask, 1],
                      embeddings_tsne[safe_caution_mask, 2],
                      c='lightgreen', marker='x', s=40, alpha=0.8, label=f'Caution→Safe ({sum(safe_caution_mask)})')
            ax.scatter(embeddings_tsne[unsafe_caution_mask, 0],
                      embeddings_tsne[unsafe_caution_mask, 1],
                      embeddings_tsne[unsafe_caution_mask, 2],
                      c='orange', marker='x', s=40, alpha=0.8, label=f'Caution→Unsafe ({sum(unsafe_caution_mask)})')

        ax.set_xlabel('t-SNE Dim 1', fontsize=10)
        ax.set_ylabel('t-SNE Dim 2', fontsize=10)
        ax.set_zlabel('t-SNE Dim 3', fontsize=10)
        ax.set_title(f'{model_name.upper()} Embedding Space (t-SNE 3D)\nLinear Separability',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)

        plt.tight_layout()
        output_file = output_dir / f'tsne_3d_{model_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved 3D visualization to {output_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='t-SNE visualization of safety embeddings')
    parser.add_argument('--model', type=str, choices=['siglip', 'clip', 'dinov2'],
                       default='siglip', help='Model to use')
    parser.add_argument('--data-dir', type=str, default='data_scenario',
                       help='Data directory')
    parser.add_argument('--output', type=str, default='results/visualization',
                       help='Output directory')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum number of samples to use')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--dimensions', type=str, default='2,3',
                       help='Comma-separated list of dimensions (2 and/or 3)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    labels_file = data_dir / 'labels.json'
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"t-SNE Visualization - {args.model.upper()}")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max samples: {args.max_samples}")
    print(f"Perplexity: {args.perplexity}")
    print(f"Device: {args.device}")

    # Extract embeddings
    embeddings, labels, classes = extract_embeddings(
        args.model, data_dir, labels_file,
        max_samples=args.max_samples,
        device=args.device
    )

    # Create visualizations
    dimensions = [int(d) for d in args.dimensions.split(',')]

    for n_components in dimensions:
        if n_components in [2, 3]:
            create_tsne_visualization(
                embeddings, labels, classes,
                args.model, output_dir,
                n_components=n_components,
                perplexity=args.perplexity
            )

    print(f"\n{'='*60}")
    print("t-SNE Visualization Complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
