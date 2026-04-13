"""
Main Safety Assessment System integrating all components
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm

from .embedders import create_embedder, BaseEmbedder
from .neural_classifier import SafetyClassifier
from .ensemble import EnsembleClassifier, ModelPrediction
from .safety_dimensions import (
    SafetyAssessmentResult, 
    SafetyDimension,
    DimensionAnalyzer
)
from ..config.settings import SystemConfig
from ..utils import ImageDataset

logger = logging.getLogger(__name__)


class SafetyAssessmentSystem:
    """
    Main system for industrial image safety assessment
    Integrates multiple vision models with ensemble methods
    """
    
    def __init__(self, config: SystemConfig):
        """Initialize safety assessment system"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize dimension analyzer first
        # First create SafetyDimension instance from config
        dimensions_config = {}
        if hasattr(config, 'safety'):
            if hasattr(config.safety, 'dimensions'):
                dimensions_config = config.safety.dimensions
        elif isinstance(config, dict) and 'safety' in config:
            safety_config = config['safety']
            if 'dimensions' in safety_config:
                dimensions_config = safety_config['dimensions']
        
        # Create SafetyDimension instance
        self.safety_dimension = SafetyDimension(dimensions_config)
        
        # Initialize embedders for each model
        self.embedders = {}
        for model_config in config.models:
            # Handle both dict and object formats
            if isinstance(model_config, dict):
                model_name = model_config.get("name")
                model_type = model_config.get("model_type")
                cache_dir = model_config.get("cache_dir", None)
                checkpoint = model_config.get("checkpoint", None)
            else:
                model_name = model_config.name
                model_type = model_config.model_type
                cache_dir = getattr(model_config, 'cache_dir', None)
                checkpoint = getattr(model_config, 'checkpoint', None)

            logger.info(f"Loading {model_name} embedder with checkpoint: {checkpoint}...")
            self.embedders[model_name] = create_embedder(
                model_type=model_type,
                device=str(self.device),
                cache_dir=cache_dir,
                checkpoint=checkpoint
            )
        
        # Initialize classifiers for each model
        self.classifiers = {}
        for model_config in config.models:
            # Handle both dict and object formats
            if isinstance(model_config, dict):
                model_name = model_config.get("name")
                embedding_dim = model_config.get("embedding_dim", 768)
            else:
                model_name = model_config.name
                embedding_dim = model_config.embedding_dim
                
            # Get batch size from config
            if hasattr(config, 'training') and hasattr(config.training, 'batch_size'):
                batch_size = config.training.batch_size
            else:
                batch_size = 32  # default
                
            self.classifiers[model_name] = SafetyClassifier(
                embedding_dim=embedding_dim,
                hidden_dim=batch_size * 8,
                num_dimensions=len(self.safety_dimension.get_all()),
                dimension_names=self.safety_dimension.get_all()
            ).to(self.device)
        
        # Get thresholds from config (early for ensemble initialization)
        if hasattr(config, 'safety'):
            safety_threshold = getattr(config.safety, 'safety_threshold', 0.5)
            confidence_threshold = getattr(config.safety, 'confidence_threshold', 0.7)
        elif isinstance(config, dict) and 'safety' in config:
            safety_threshold = config['safety'].get('safety_threshold', 0.5)
            confidence_threshold = config['safety'].get('confidence_threshold', 0.7)
        else:
            safety_threshold = 0.5
            confidence_threshold = 0.7

        # Initialize ensemble classifier
        self.ensemble = EnsembleClassifier(
            models=config.models,
            strategy=getattr(config, 'ensemble_strategy', 'weighted_vote'),
            safety_threshold=safety_threshold
        )
        
        # Extract dimension weights
        dimension_weights = {}
        for dim_name, dim_info in dimensions_config.items():
            if isinstance(dim_info, dict):
                dimension_weights[dim_name] = dim_info.get('weight', 1.0)
        
        self.dimension_analyzer = DimensionAnalyzer(
            safety_dimension=self.safety_dimension,
            dimension_weights=dimension_weights
        )
        
        # Performance tracking
        self.model_performances = {}

        # Store thresholds
        self.safety_threshold = safety_threshold
        self.confidence_threshold = confidence_threshold

        logger.info(f"Safety threshold: {self.safety_threshold}, Confidence threshold: {self.confidence_threshold}")

    def assess_image(self, image_path: str) -> SafetyAssessmentResult:
        """
        Assess safety of a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Safety assessment result
        """
        start_time = time.time()
        
        if getattr(self.config, 'assessment_method', 'single') == "ensemble":
            # Get predictions from all models
            model_predictions = []
            
            for model_name, embedder in self.embedders.items():
                # Extract embedding
                embedding = embedder.extract_single_embedding(image_path)
                embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(self.device)
                
                # Get classifier prediction
                classifier = self.classifiers[model_name]
                classifier.eval()
                
                with torch.no_grad():
                    if embedding_tensor.dim() == 1:
                        embedding_tensor = embedding_tensor.unsqueeze(0)
                    
                    overall_safety, dimension_scores = classifier(embedding_tensor)
                    
                    # Create model prediction
                    pred = ModelPrediction(
                        model_name=model_name,
                        is_safe=overall_safety.item() > self.safety_threshold,
                        safety_score=overall_safety.item(),
                        confidence=abs(overall_safety.item() - self.safety_threshold) * 2,
                        dimension_scores={
                            dim: score.item()
                            for dim, score in zip(self.safety_dimension.get_all(), dimension_scores.values())
                        },
                        embedding=embedding
                    )
                    model_predictions.append(pred)
            
            # Combine predictions using ensemble
            result = self.ensemble.predict(model_predictions)
            result.image_path = image_path
            
        else:
            # Single model assessment
            # Get first model name handling both dict and object formats
            if isinstance(self.config.models[0], dict):
                model_name = self.config.models[0]["name"]
            else:
                model_name = self.config.models[0].name
            embedder = self.embedders[model_name]
            classifier = self.classifiers[model_name]
            
            # Extract embedding and classify
            embedding = embedder.extract_single_embedding(image_path)
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(self.device)
            
            classifier.eval()
            with torch.no_grad():
                if embedding_tensor.dim() == 1:
                    embedding_tensor = embedding_tensor.unsqueeze(0)
                
                overall_safety, dimension_scores = classifier(embedding_tensor)
                
                result = SafetyAssessmentResult(
                    image_path=image_path,
                    overall_safety_score=overall_safety.item(),
                    is_safe=overall_safety.item() > self.safety_threshold,
                    dimension_scores={
                        dim: score.item()
                        for dim, score in dimension_scores.items()
                    },
                    confidence=abs(overall_safety.item() - self.safety_threshold) * 2,
                    method_used=getattr(self.config, 'assessment_method', 'single'),
                    model_name=model_name,
                    processing_time=0.0
                )
        
        result.processing_time = time.time() - start_time
        return result
    
    def evaluate_dataset(self, dataset: ImageDataset) -> Dict:
        """
        Evaluate system performance on a dataset
        
        Args:
            dataset: Test dataset
            
        Returns:
            Performance metrics
        """
        all_results = []
        all_labels = []
        
        # Track individual model performances
        model_predictions = {name: [] for name in self.embedders.keys()}
        model_labels = {name: [] for name in self.embedders.keys()}
        
        logger.info(f"Evaluating on {len(dataset)} images...")
        
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            image, label, path = dataset[idx]
            
            # Get ensemble prediction
            result = self.assess_image(path)
            all_results.append(result)
            all_labels.append(label["is_safe"])
            
            # Get individual model predictions for comparison
            if getattr(self.config, 'assessment_method', 'single') == "ensemble":
                for model_name, embedder in self.embedders.items():
                    embedding = embedder.extract_single_embedding(path)
                    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(self.device)
                    
                    classifier = self.classifiers[model_name]
                    classifier.eval()
                    
                    with torch.no_grad():
                        if embedding_tensor.dim() == 1:
                            embedding_tensor = embedding_tensor.unsqueeze(0)
                        
                        overall_safety, _ = classifier(embedding_tensor)
                        is_safe = overall_safety.item() > 0.5
                        
                        model_predictions[model_name].append(is_safe)
                        model_labels[model_name].append(label["is_safe"])
        
        # Calculate ensemble metrics
        ensemble_metrics = self._calculate_metrics(
            [r.is_safe for r in all_results],
            all_labels
        )
        
        # Calculate individual model metrics
        individual_metrics = {}
        for model_name in self.embedders.keys():
            if model_predictions[model_name]:
                individual_metrics[model_name] = self._calculate_metrics(
                    model_predictions[model_name],
                    model_labels[model_name]
                )
        
        # Update model performances for ensemble weighting
        for model_name, metrics in individual_metrics.items():
            self.model_performances[model_name] = metrics["f1_score"]
        
        # Update ensemble weights based on performance
        if hasattr(self.ensemble, 'update_weights'):
            self.ensemble.update_weights(self.model_performances)
        
        return {
            "ensemble_metrics": ensemble_metrics,
            "individual_metrics": individual_metrics,
            "model_performances": self.model_performances,
            "best_individual_model": max(
                individual_metrics.items(), 
                key=lambda x: x[1]["f1_score"]
            )[0] if individual_metrics else None,
            "ensemble_improvement": (
                ensemble_metrics["f1_score"] - 
                max(m["f1_score"] for m in individual_metrics.values())
            ) if individual_metrics else 0
        }
    
    def _calculate_metrics(self, predictions: List[bool], labels: List[bool]) -> Dict:
        """Calculate performance metrics"""
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        tp = np.sum((predictions == True) & (labels == True))
        tn = np.sum((predictions == False) & (labels == False))
        fp = np.sum((predictions == True) & (labels == False))
        fn = np.sum((predictions == False) & (labels == True))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": {
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn)
            }
        }
    
    def train(self, dataset: ImageDataset, training_config=None):
        """Train all classifiers"""
        training_config = training_config or self.config.training
        
        logger.info("Training classifiers for all models...")
        
        # First, extract all embeddings for efficiency
        embeddings_cache = {name: [] for name in self.embedders.keys()}
        labels_list = []
        
        logger.info("Extracting embeddings...")
        for idx in tqdm(range(len(dataset)), desc="Extracting embeddings"):
            image, label, path = dataset[idx]
            labels_list.append(label)
            
            for model_name, embedder in self.embedders.items():
                embedding = embedder.extract_single_embedding(path)
                embeddings_cache[model_name].append(embedding)
        
        # Train each model's classifier
        for model_name, embeddings in embeddings_cache.items():
            logger.info(f"Training classifier for {model_name}...")
            
            classifier = self.classifiers[model_name]
            optimizer = torch.optim.Adam(
                classifier.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay
            )
            
            # Convert to tensors
            X = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
            y_safety = torch.tensor(
                [label["is_safe"] for label in labels_list],
                dtype=torch.float32
            ).to(self.device)
            
            y_dimensions = {}
            for dim in self.safety_dimension.get_all():
                y_dimensions[dim] = torch.tensor(
                    [1 - label["dimensions"].get(dim, 0.5) for label in labels_list],
                    dtype=torch.float32
                ).to(self.device)
            
            # Training loop
            classifier.train()
            for epoch in range(training_config.epochs):
                optimizer.zero_grad()
                
                overall_pred, dimension_preds = classifier(X)
                
                # Calculate losses
                loss = torch.nn.functional.binary_cross_entropy(
                    overall_pred.squeeze(), y_safety
                )
                
                for dim in self.safety_dimension.get_all():
                    dim_loss = torch.nn.functional.binary_cross_entropy(
                        dimension_preds[dim].squeeze(),
                        y_dimensions[dim]
                    )
                    loss += dim_loss * 0.2
                
                loss.backward()
                optimizer.step()
                
                if epoch % 5 == 0:
                    logger.info(f"{model_name} - Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # If using ensemble with stacking, train meta-classifier
        if self.config.ensemble_strategy == "stacking":
            logger.info("Training ensemble meta-classifier...")
            self._train_meta_classifier(embeddings_cache, labels_list)
    
    def _train_meta_classifier(self, embeddings_cache: Dict, labels: List):
        """Train the ensemble meta-classifier for stacking"""
        training_data = []
        
        # Generate predictions from all models
        for idx in range(len(labels)):
            model_preds = []
            
            for model_name, embeddings in embeddings_cache.items():
                embedding = torch.tensor(
                    embeddings[idx], 
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                classifier = self.classifiers[model_name]
                classifier.eval()
                
                with torch.no_grad():
                    overall_safety, dimension_scores = classifier(embedding)
                    
                    pred = ModelPrediction(
                        model_name=model_name,
                        is_safe=overall_safety.item() > 0.5,
                        safety_score=overall_safety.item(),
                        confidence=abs(overall_safety.item() - 0.5) * 2,
                        dimension_scores={
                            dim: score.item() 
                            for dim, score in dimension_scores.items()
                        }
                    )
                    model_preds.append(pred)
            
            training_data.append((model_preds, labels[idx]["is_safe"]))
        
        self.ensemble.train_meta_classifier(training_data)
    
    def save_models(self, checkpoint_dir: Optional[str] = None):
        """Save all trained models"""
        checkpoint_dir = Path(checkpoint_dir or self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save classifiers
        for model_name, classifier in self.classifiers.items():
            torch.save(
                classifier.state_dict(),
                checkpoint_dir / f"{model_name}_classifier.pth"
            )
        
        # Save ensemble weights
        with open(checkpoint_dir / "ensemble_weights.json", "w") as f:
            json.dump({
                "model_weights": self.ensemble.model_weights,
                "model_performances": self.model_performances
            }, f, indent=2)
        
        logger.info(f"Models saved to {checkpoint_dir}")
    
    def load_models(self, checkpoint_dir: Optional[str] = None):
        """Load saved models"""
        checkpoint_dir = Path(checkpoint_dir or self.config.checkpoint_dir)
        
        # Load classifiers
        for model_name, classifier in self.classifiers.items():
            checkpoint_path = checkpoint_dir / f"{model_name}_classifier.pth"
            if checkpoint_path.exists():
                classifier.load_state_dict(torch.load(checkpoint_path))
                logger.info(f"Loaded {model_name} classifier")
        
        # Load ensemble weights
        weights_path = checkpoint_dir / "ensemble_weights.json"
        if weights_path.exists():
            with open(weights_path, "r") as f:
                data = json.load(f)
                self.ensemble.model_weights = data["model_weights"]
                self.model_performances = data["model_performances"]