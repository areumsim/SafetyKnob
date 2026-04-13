"""
Safety Dimensions and Assessment Results for Industrial Safety Analysis
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


class SafetyDimension:
    """Dynamic safety dimensions loaded from configuration"""
    
    def __init__(self, config_dimensions: Dict[str, Dict[str, any]]):
        """
        Initialize safety dimensions from config
        
        Args:
            config_dimensions: Dictionary with dimension names as keys and
                             dict with 'weight' and 'description' as values
        """
        self.dimensions = {}
        self.weights = {}
        self.descriptions = {}
        
        for dim_name, dim_info in config_dimensions.items():
            self.dimensions[dim_name] = dim_name
            self.weights[dim_name] = dim_info.get('weight', 1.0)
            self.descriptions[dim_name] = dim_info.get('description', dim_name)
    
    def get_all(self) -> List[str]:
        """Get all safety dimension names"""
        return list(self.dimensions.keys())
    
    def get_weights(self) -> Dict[str, float]:
        """Get weights for all dimensions"""
        return self.weights.copy()
    
    def get_description(self, dimension: str) -> str:
        """Get description for a dimension"""
        return self.descriptions.get(dimension, dimension)


@dataclass
class SafetyAssessmentResult:
    """Result of safety assessment for an image"""
    image_path: str
    overall_safety_score: float  # 0-1, where 1 is completely safe
    is_safe: bool
    dimension_scores: Dict[str, float]
    confidence: float
    method_used: str  # 'similarity', 'classifier', 'ensemble'
    model_name: str  # Which vision model was used
    processing_time: float  # Time taken in seconds
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "image_path": self.image_path,
            "overall_safety_score": self.overall_safety_score,
            "is_safe": self.is_safe,
            "dimension_scores": self.dimension_scores,
            "confidence": self.confidence,
            "method_used": self.method_used,
            "model_name": self.model_name,
            "processing_time": self.processing_time
        }
    
    def get_risk_summary(self) -> str:
        """Get human-readable risk summary"""
        if self.is_safe:
            return "Image assessed as SAFE"
        
        # Find highest risk dimensions
        risky_dims = []
        for dim, score in self.dimension_scores.items():
            if score < 0.5:  # Lower score means higher risk
                risky_dims.append((dim, score))
        
        risky_dims.sort(key=lambda x: x[1])  # Sort by risk level
        
        if risky_dims:
            summary = "Image assessed as UNSAFE. Risk factors: "
            risks = [f"{dim.replace('_', ' ').title()} ({1-score:.0%} risk)" 
                    for dim, score in risky_dims[:3]]  # Top 3 risks
            summary += ", ".join(risks)
        else:
            summary = "Image assessed as UNSAFE (general safety concerns)"
            
        return summary


class DimensionAnalyzer:
    """Analyze safety dimensions from embeddings"""
    
    def __init__(self, safety_dimension: SafetyDimension, 
                 dimension_weights: Optional[Dict[str, float]] = None):
        self.safety_dimension = safety_dimension
        self.dimension_weights = dimension_weights or safety_dimension.get_weights()
        self.dimension_prototypes = {dim: [] for dim in safety_dimension.get_all()}
        
    def add_training_sample(self, embedding: np.ndarray, 
                          dimension_labels: Dict[str, float]):
        """Add a training sample for dimension analysis"""
        for dim, risk_level in dimension_labels.items():
            if dim in self.dimension_prototypes:
                self.dimension_prototypes[dim].append({
                    'embedding': embedding,
                    'risk_level': risk_level
                })
    
    def compute_dimension_scores(self, embedding: np.ndarray) -> Dict[str, float]:
        """Compute safety scores for each dimension"""
        scores = {}
        
        for dim_name in self.safety_dimension.get_all():
            if not self.dimension_prototypes[dim_name]:
                scores[dim_name] = 0.5  # Neutral if no training data
                continue
                
            # Compute similarity to safe vs unsafe prototypes
            safe_similarities = []
            unsafe_similarities = []
            
            for prototype in self.dimension_prototypes[dim_name]:
                similarity = self._cosine_similarity(embedding, prototype['embedding'])
                if prototype['risk_level'] < 0.5:  # Safe
                    safe_similarities.append(similarity)
                else:  # Unsafe
                    unsafe_similarities.append(similarity)
            
            # Average similarities
            safe_score = np.mean(safe_similarities) if safe_similarities else 0.5
            unsafe_score = np.mean(unsafe_similarities) if unsafe_similarities else 0.5
            
            # Normalize to 0-1 where 1 is safe
            scores[dim_name] = (safe_score - unsafe_score + 1) / 2
            
        return scores
    
    def compute_overall_safety(self, dimension_scores: Dict[str, float]) -> float:
        """Compute weighted overall safety score"""
        total_weight = sum(self.dimension_weights.values())
        weighted_sum = sum(
            dimension_scores.get(dim, 0.5) * self.dimension_weights.get(dim, 1.0)
            for dim in dimension_scores
        )
        return weighted_sum / total_weight
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)