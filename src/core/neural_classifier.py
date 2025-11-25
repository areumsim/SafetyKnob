"""
Neural network based safety classifier for embeddings with dynamic dimensions
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, List


class SafetyClassifier(nn.Module):
    """Neural network classifier for safety assessment"""
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 256, num_dimensions: int = 5, dimension_names: List[str] = None):
        """
        Initialize safety classifier
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Hidden layer dimension
            num_dimensions: Number of safety dimensions to predict
            dimension_names: Names of dimensions (optional)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_dimensions = num_dimensions
        
        # Use provided dimension names or default names
        if dimension_names:
            self.dimension_names = dimension_names
        else:
            self.dimension_names = [
                "fall_hazard",
                "collision_risk", 
                "equipment_hazard",
                "environmental_risk",
                "protective_gear"
            ][:num_dimensions]
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Overall safety predictor
        self.safety_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Individual dimension predictors
        self.dimension_heads = nn.ModuleDict({
            dim_name: nn.Sequential(
                nn.Linear(hidden_dim // 2, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            for dim_name in self.dimension_names
        })
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input embeddings [batch_size, embedding_dim]
            
        Returns:
            Tuple of:
                - overall_safety: Overall safety score [batch_size, 1]
                - dimension_scores: Dict of dimension scores
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Predict overall safety
        overall_safety = self.safety_head(features)
        
        # Predict dimension scores
        dimension_scores = {}
        for dim_name, head in self.dimension_heads.items():
            dimension_scores[dim_name] = head(features)
        
        return overall_safety, dimension_scores