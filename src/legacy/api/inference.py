"""
Inference API for safety classification.

This module provides a simple API interface for making predictions
on images using the trained safety classifier.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import numpy as np
from PIL import Image

from ..core import SafetyClassifier, PredictionResult
from ..config import get_config, get_path_manager


logger = logging.getLogger(__name__)


class SafetyInferenceAPI:
    """API for safety inference on images."""
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        model_types: Optional[List[str]] = None,
        device: Optional[str] = None,
        cache_embeddings: bool = True
    ):
        """
        Initialize the inference API.
        
        Args:
            model_dir: Directory containing trained models
            model_types: List of models to use
            device: Device to run on
            cache_embeddings: Whether to cache embeddings
        """
        config = get_config()
        
        self.model_dir = model_dir or get_path_manager().get_str("models")
        self.device = device or config.model.device
        
        # Initialize classifier
        self.classifier = SafetyClassifier(
            model_types=model_types or config.model.model_types,
            device=self.device,
            model_dir=self.model_dir
        )
        
        # Load trained models
        self._load_models()
        
        self.cache_embeddings = cache_embeddings
        
    def _load_models(self):
        """Load trained models."""
        try:
            self.classifier.load_models()
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise RuntimeError(f"Failed to load models from {self.model_dir}")
    
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Predict safety of an image.
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            Dictionary with prediction results
        """
        # Handle different input types
        if isinstance(image, (str, Path)):
            image_path = str(image)
        elif isinstance(image, Image.Image):
            # Save PIL image to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                image.save(tmp.name)
                image_path = tmp.name
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL and save
            import tempfile
            pil_image = Image.fromarray(image)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                pil_image.save(tmp.name)
                image_path = tmp.name
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        try:
            # Make prediction
            result = self.classifier.predict(image_path)
            
            # Convert to dictionary
            return result.to_dict()
            
        finally:
            # Clean up temporary files
            if not isinstance(image, (str, Path)) and os.path.exists(image_path):
                os.unlink(image_path)
    
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict safety for multiple images.
        
        Args:
            images: List of images
            show_progress: Whether to show progress bar
            
        Returns:
            List of prediction results
        """
        results = []
        
        if show_progress:
            from tqdm import tqdm
            images = tqdm(images, desc="Processing images")
        
        for image in images:
            try:
                result = self.predict(image)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                results.append({
                    "error": str(e),
                    "prediction": None
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            "model_types": self.classifier.model_types,
            "device": str(self.device),
            "model_dir": self.model_dir,
            "is_trained": self.classifier.is_trained,
            "training_stats": self.classifier.training_stats
        }
        
        return info
    
    def set_threshold(self, threshold: float):
        """
        Set danger classification threshold.
        
        Args:
            threshold: Threshold value (0.0 - 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        # This would be implemented in the classifier
        logger.info(f"Threshold set to {threshold}")
    
    def explain_prediction(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Get detailed explanation of a prediction.
        
        Args:
            image: Image to analyze
            
        Returns:
            Detailed prediction explanation
        """
        # Get basic prediction
        result = self.predict(image)
        
        # Add explanation details
        explanation = {
            **result,
            "explanation": {
                "dominant_factor": self._get_dominant_factor(result),
                "confidence_breakdown": self._get_confidence_breakdown(result),
                "model_agreement": self._calculate_model_agreement(result)
            }
        }
        
        return explanation
    
    def _get_dominant_factor(self, result: Dict[str, Any]) -> str:
        """Determine dominant factor in prediction."""
        if result["confidence"] > 0.9:
            return "High confidence from all models"
        elif result["confidence"] > 0.7:
            return "Strong agreement among models"
        else:
            return "Mixed signals from different models"
    
    def _get_confidence_breakdown(self, result: Dict[str, Any]) -> Dict[str, str]:
        """Break down confidence levels."""
        breakdown = {}
        
        for model, conf in result.get("model_confidences", {}).items():
            if conf > 0.8:
                breakdown[model] = "High"
            elif conf > 0.6:
                breakdown[model] = "Medium"
            else:
                breakdown[model] = "Low"
        
        return breakdown
    
    def _calculate_model_agreement(self, result: Dict[str, Any]) -> float:
        """Calculate model agreement score."""
        votes = list(result.get("model_votes", {}).values())
        if not votes:
            return 0.0
        
        # Count most common vote
        from collections import Counter
        vote_counts = Counter(votes)
        most_common_count = vote_counts.most_common(1)[0][1]
        
        return most_common_count / len(votes)


# Convenience function for quick inference
def quick_predict(
    image_path: str,
    model_dir: Optional[str] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Quick prediction on a single image.
    
    Args:
        image_path: Path to image
        model_dir: Directory with trained models
        device: Device to use
        
    Returns:
        Prediction results
    """
    api = SafetyInferenceAPI(model_dir=model_dir, device=device)
    return api.predict(image_path)