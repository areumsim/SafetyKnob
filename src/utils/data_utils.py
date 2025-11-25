"""
Data utility functions for image loading and dataset management
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms


class ImageDataset(Dataset):
    """Dataset for safety assessment images"""
    
    def __init__(self, image_dir: Path, labels_file: Optional[Path] = None):
        """
        Initialize dataset
        
        Args:
            image_dir: Directory containing images
            labels_file: Optional JSON file with labels
        """
        self.image_dir = Path(image_dir)
        self.images = []
        
        # Collect all image files
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.images.extend(list(self.image_dir.glob(f"**/{ext}")))
        
        # Load labels if provided
        self.labels = {}
        if labels_file and labels_file.exists():
            with open(labels_file, 'r') as f:
                self.labels = json.load(f)
        
        # Default transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels.get(str(image_path), {
            "is_safe": True,
            "dimensions": {
                "fall_hazard": 0.0,
                "collision_risk": 0.0,
                "equipment_hazard": 0.0,
                "environmental_risk": 0.0,
                "protective_gear": 0.0
            }
        })
        
        return image, label, str(image_path)


def load_image(image_path: str) -> Image.Image:
    """Load and preprocess a single image"""
    return Image.open(image_path).convert('RGB')


def prepare_dataloader(dataset: Dataset, batch_size: int = 32, 
                      shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """Create a DataLoader from dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )