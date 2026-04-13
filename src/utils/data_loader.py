"""
Data loading utilities for image datasets.

This module provides functions for loading and organizing image datasets
for safety classification training and evaluation.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import glob
from PIL import Image
import numpy as np


logger = logging.getLogger(__name__)


def load_image_dataset(
    data_dir: str,
    categories: List[str] = None,
    extensions: List[str] = None,
    validate_images: bool = True
) -> Dict[str, List[str]]:
    """
    Load image dataset from directory structure.
    
    Args:
        data_dir: Root directory containing category subdirectories
        categories: List of category names to load (default: all)
        extensions: List of file extensions to include
        validate_images: Whether to validate images can be opened
        
    Returns:
        Dictionary mapping category names to lists of image paths
    """
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Get categories
    if categories is None:
        categories = [
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ]
    
    dataset = {}
    
    for category in categories:
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            logger.warning(f"Category directory not found: {category_dir}")
            continue
        
        # Find all images
        image_paths = []
        for ext in extensions:
            pattern = os.path.join(category_dir, f"**/*{ext}")
            paths = glob.glob(pattern, recursive=True)
            image_paths.extend(paths)
        
        # Validate images if requested
        if validate_images:
            valid_paths = []
            for path in image_paths:
                try:
                    Image.open(path).verify()
                    valid_paths.append(path)
                except Exception as e:
                    logger.warning(f"Invalid image {path}: {e}")
            image_paths = valid_paths
        
        dataset[category] = sorted(image_paths)
        logger.info(f"Loaded {len(image_paths)} images for category '{category}'")
    
    return dataset


def organize_by_scenario(image_paths: List[str]) -> Dict[str, List[str]]:
    """
    Organize images by scenario based on filename pattern.
    
    Expected pattern: "{type}_{scenario}_{case}.jpg"
    Example: "화재_01_001.jpg" -> scenario "화재_01"
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        Dictionary mapping scenarios to lists of image paths
    """
    scenarios = {}
    
    for path in image_paths:
        filename = os.path.basename(path)
        parts = filename.split('_')
        
        if len(parts) >= 3:
            # Extract scenario (type + scenario number)
            scenario = f"{parts[0]}_{parts[1]}"
            
            if scenario not in scenarios:
                scenarios[scenario] = []
            scenarios[scenario].append(path)
        else:
            logger.warning(f"Unexpected filename format: {filename}")
    
    return scenarios


def split_dataset(
    dataset: Dict[str, List[str]],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Dictionary mapping categories to image paths
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    np.random.seed(random_state)
    
    train_dataset = {}
    val_dataset = {}
    test_dataset = {}
    
    for category, paths in dataset.items():
        # Shuffle paths
        paths = np.array(paths)
        np.random.shuffle(paths)
        
        # Calculate split indices
        n_samples = len(paths)
        n_test = int(n_samples * test_size)
        n_val = int(n_samples * val_size)
        n_train = n_samples - n_test - n_val
        
        # Split
        train_dataset[category] = paths[:n_train].tolist()
        val_dataset[category] = paths[n_train:n_train + n_val].tolist()
        test_dataset[category] = paths[n_train + n_val:].tolist()
        
        logger.info(
            f"{category}: {n_train} train, {n_val} val, {n_test} test samples"
        )
    
    return train_dataset, val_dataset, test_dataset


def load_paired_dataset(
    danger_dir: str,
    safe_dir: str,
    caution_dir: Optional[str] = None
) -> Dict[str, Dict[str, List[str]]]:
    """
    Load paired dataset where danger/safe images share scenarios.
    
    Args:
        danger_dir: Directory containing danger images
        safe_dir: Directory containing safe images
        caution_dir: Optional directory containing caution images
        
    Returns:
        Dictionary mapping scenarios to category lists
    """
    # Load all categories
    categories = {
        'danger': load_image_dataset(danger_dir, categories=[''])[''],
        'safe': load_image_dataset(safe_dir, categories=[''])['']
    }
    
    if caution_dir and os.path.exists(caution_dir):
        categories['caution'] = load_image_dataset(caution_dir, categories=[''])['']
    
    # Organize by scenario
    paired_data = {}
    
    for category, paths in categories.items():
        scenarios = organize_by_scenario(paths)
        
        for scenario, scenario_paths in scenarios.items():
            if scenario not in paired_data:
                paired_data[scenario] = {}
            paired_data[scenario][category] = scenario_paths
    
    # Filter to only scenarios with both danger and safe
    complete_scenarios = {
        scenario: data
        for scenario, data in paired_data.items()
        if 'danger' in data and 'safe' in data
    }
    
    logger.info(f"Found {len(complete_scenarios)} paired scenarios")
    
    return complete_scenarios


def create_balanced_dataset(
    dataset: Dict[str, List[str]],
    target_size: Optional[int] = None,
    sampling_strategy: str = "undersample"
) -> Dict[str, List[str]]:
    """
    Create a balanced dataset by sampling.
    
    Args:
        dataset: Dictionary mapping categories to image paths
        target_size: Target number of samples per category
        sampling_strategy: "undersample" or "oversample"
        
    Returns:
        Balanced dataset
    """
    if target_size is None:
        if sampling_strategy == "undersample":
            target_size = min(len(paths) for paths in dataset.values())
        else:
            target_size = max(len(paths) for paths in dataset.values())
    
    balanced_dataset = {}
    
    for category, paths in dataset.items():
        n_samples = len(paths)
        
        if n_samples == target_size:
            balanced_dataset[category] = paths
        elif n_samples > target_size:
            # Undersample
            indices = np.random.choice(n_samples, target_size, replace=False)
            balanced_dataset[category] = [paths[i] for i in indices]
        else:
            # Oversample
            indices = np.random.choice(n_samples, target_size, replace=True)
            balanced_dataset[category] = [paths[i] for i in indices]
        
        logger.info(
            f"{category}: {n_samples} -> {len(balanced_dataset[category])} samples"
        )
    
    return balanced_dataset


def save_dataset_info(dataset: Dict[str, List[str]], output_path: str):
    """Save dataset information to JSON file."""
    info = {
        "categories": list(dataset.keys()),
        "samples_per_category": {
            cat: len(paths) for cat, paths in dataset.items()
        },
        "total_samples": sum(len(paths) for paths in dataset.values()),
        "file_lists": {
            cat: [os.path.basename(p) for p in paths[:10]]  # Sample files
            for cat, paths in dataset.items()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Dataset info saved to {output_path}")