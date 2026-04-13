"""
Ensemble methods for combining predictions from multiple models.

This module provides various ensemble strategies for aggregating predictions
from different vision models to improve overall classification accuracy.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import Counter


@dataclass
class ModelPrediction:
    """Prediction from a single model"""
    model_name: str
    is_safe: bool
    safety_score: float
    confidence: float
    dimension_scores: Dict[str, float]
    embedding: Optional[np.ndarray] = None


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    
    method: str = "weighted_voting"  # voting, weighted_voting, averaging, stacking
    weights: Optional[Dict[str, float]] = None  # Model-specific weights
    confidence_threshold: float = 0.6
    min_models_required: int = 2


class EnsembleClassifier:
    """Ensemble classifier for combining multiple model predictions."""

    def __init__(self, models: List, strategy: str = "weighted_vote", safety_threshold: float = 0.5):
        """
        Initialize ensemble classifier.

        Args:
            models: List of model configurations
            strategy: Ensemble strategy
            safety_threshold: Threshold for safety classification
        """
        self.models = models
        self.strategy = strategy
        self.safety_threshold = safety_threshold
        # Handle both dict and object formats
        self.model_weights = {}
        for m in models:
            if isinstance(m, dict):
                self.model_weights[m.get("name")] = 1.0
            else:
                self.model_weights[m.name] = 1.0
        self.meta_classifier = None
    
    def predict(self, model_predictions: List[ModelPrediction]):
        """Combine predictions from multiple models"""
        from .safety_dimensions import SafetyAssessmentResult
        
        if self.strategy == "weighted_vote":
            # Weighted voting based on model performance
            total_weight = sum(self.model_weights.values())
            weighted_safe_score = sum(
                pred.safety_score * self.model_weights[pred.model_name]
                for pred in model_predictions
            ) / total_weight
            
            is_safe = weighted_safe_score > self.safety_threshold

            # Average dimension scores
            dimension_scores = {}
            for dim in model_predictions[0].dimension_scores.keys():
                dimension_scores[dim] = sum(
                    pred.dimension_scores[dim] * self.model_weights[pred.model_name]
                    for pred in model_predictions
                ) / total_weight

            # Calculate confidence
            confidence = abs(weighted_safe_score - self.safety_threshold) * 2
            
            return SafetyAssessmentResult(
                image_path="",
                overall_safety_score=weighted_safe_score,
                is_safe=is_safe,
                dimension_scores=dimension_scores,
                confidence=confidence,
                method_used="ensemble",
                model_name="ensemble_all",
                processing_time=0.0
            )
        
        elif self.strategy == "stacking":
            # Use meta-classifier to combine predictions
            if self.meta_classifier is None:
                raise ValueError("Meta-classifier not trained for stacking")
            
            # Prepare features for meta-classifier
            features = []
            for pred in model_predictions:
                features.extend([pred.safety_score, pred.confidence])
                features.extend(pred.dimension_scores.values())
            
            # Predict using meta-classifier
            meta_pred = self.meta_classifier.predict([features])[0]
            meta_prob = self.meta_classifier.predict_proba([features])[0]
            
            # Average dimension scores
            dimension_scores = {}
            for dim in model_predictions[0].dimension_scores.keys():
                dimension_scores[dim] = np.mean([
                    pred.dimension_scores[dim] for pred in model_predictions
                ])
            
            return SafetyAssessmentResult(
                image_path="",
                overall_safety_score=meta_prob[1],  # Probability of safe
                is_safe=bool(meta_pred),
                dimension_scores=dimension_scores,
                confidence=abs(meta_prob[1] - 0.5) * 2,
                method_used="ensemble_stacking",
                model_name="ensemble_meta",
                processing_time=0.0
            )
    
    def update_weights(self, model_performances: Dict[str, float]):
        """Update model weights based on performance"""
        # Normalize weights based on F1 scores
        total_perf = sum(model_performances.values())
        if total_perf > 0:
            self.model_weights = {
                name: perf / total_perf * len(model_performances)
                for name, perf in model_performances.items()
            }
    
    def train_meta_classifier(self, training_data: List[Tuple[List[ModelPrediction], bool]]):
        """Train meta-classifier for stacking"""
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = [], []
        for model_preds, label in training_data:
            features = []
            for pred in model_preds:
                features.extend([pred.safety_score, pred.confidence])
                features.extend(pred.dimension_scores.values())
            X.append(features)
            y.append(label)
        
        self.meta_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.meta_classifier.fit(X, y)


class EnsemblePredictor:
    """Ensemble predictor for combining multiple model predictions."""
    
    def __init__(self, config: EnsembleConfig = None):
        """
        Initialize ensemble predictor.
        
        Args:
            config: Ensemble configuration
        """
        self.config = config or EnsembleConfig()
    
    def predict(
        self,
        model_predictions: Dict[str, Dict[str, float]],
        model_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Combine predictions from multiple models.
        
        Args:
            model_predictions: Dict mapping model names to their predictions
                              Format: {model_name: {"safe": prob, "danger": prob}}
            model_weights: Optional weights for each model
            
        Returns:
            Tuple of (prediction, confidence, class_probabilities)
        """
        if len(model_predictions) < self.config.min_models_required:
            raise ValueError(
                f"At least {self.config.min_models_required} models required, "
                f"got {len(model_predictions)}"
            )
        
        weights = model_weights or self.config.weights or {}
        
        if self.config.method == "voting":
            return self._voting_ensemble(model_predictions)
        elif self.config.method == "weighted_voting":
            return self._weighted_voting_ensemble(model_predictions, weights)
        elif self.config.method == "averaging":
            return self._averaging_ensemble(model_predictions, weights)
        else:
            raise ValueError(f"Unknown ensemble method: {self.config.method}")
    
    def _voting_ensemble(
        self, 
        model_predictions: Dict[str, Dict[str, float]]
    ) -> Tuple[str, float, Dict[str, float]]:
        """Simple majority voting."""
        votes = []
        
        for model_name, probs in model_predictions.items():
            if probs["danger"] > probs["safe"]:
                votes.append("danger")
            else:
                votes.append("safe")
        
        vote_counts = Counter(votes)
        prediction = vote_counts.most_common(1)[0][0]
        confidence = vote_counts[prediction] / len(votes)
        
        # Average probabilities for final scores
        avg_probs = self._average_probabilities(model_predictions)
        
        return prediction, confidence, avg_probs
    
    def _weighted_voting_ensemble(
        self,
        model_predictions: Dict[str, Dict[str, float]],
        weights: Dict[str, float]
    ) -> Tuple[str, float, Dict[str, float]]:
        """Weighted voting based on model performance."""
        weighted_probs = {"safe": 0.0, "danger": 0.0}
        total_weight = 0.0
        
        for model_name, probs in model_predictions.items():
            weight = weights.get(model_name, 1.0)
            weighted_probs["safe"] += probs["safe"] * weight
            weighted_probs["danger"] += probs["danger"] * weight
            total_weight += weight
        
        # Normalize
        weighted_probs["safe"] /= total_weight
        weighted_probs["danger"] /= total_weight
        
        # Determine prediction
        if weighted_probs["danger"] > weighted_probs["safe"]:
            prediction = "danger"
            confidence = weighted_probs["danger"]
        else:
            prediction = "safe"
            confidence = weighted_probs["safe"]
        
        return prediction, confidence, weighted_probs
    
    def _averaging_ensemble(
        self,
        model_predictions: Dict[str, Dict[str, float]],
        weights: Dict[str, float]
    ) -> Tuple[str, float, Dict[str, float]]:
        """Average probabilities from all models."""
        if weights:
            return self._weighted_voting_ensemble(model_predictions, weights)
        
        avg_probs = self._average_probabilities(model_predictions)
        
        if avg_probs["danger"] > avg_probs["safe"]:
            prediction = "danger"
            confidence = avg_probs["danger"]
        else:
            prediction = "safe"
            confidence = avg_probs["safe"]
        
        return prediction, confidence, avg_probs
    
    def _average_probabilities(
        self,
        model_predictions: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate average probabilities across models."""
        avg_probs = {"safe": 0.0, "danger": 0.0}
        
        for probs in model_predictions.values():
            avg_probs["safe"] += probs["safe"]
            avg_probs["danger"] += probs["danger"]
        
        n_models = len(model_predictions)
        avg_probs["safe"] /= n_models
        avg_probs["danger"] /= n_models
        
        return avg_probs
    
    def calculate_model_weights(
        self,
        model_performances: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate optimal weights based on model performances.
        
        Args:
            model_performances: Dict mapping model names to performance metrics
                               Format: {model_name: {"accuracy": 0.95, "auc": 0.97}}
                               
        Returns:
            Dictionary of model weights
        """
        weights = {}
        
        # Use accuracy as the primary metric
        accuracies = {
            model: perf.get("accuracy", 0.5) 
            for model, perf in model_performances.items()
        }
        
        # Convert accuracies to weights (higher accuracy = higher weight)
        min_acc = min(accuracies.values())
        max_acc = max(accuracies.values())
        
        if max_acc - min_acc < 0.01:  # All models perform similarly
            # Equal weights
            for model in accuracies:
                weights[model] = 1.0
        else:
            # Normalize weights based on performance
            for model, acc in accuracies.items():
                # Scale weights between 0.5 and 1.5
                normalized = (acc - min_acc) / (max_acc - min_acc)
                weights[model] = 0.5 + normalized
        
        return weights
    
    def get_confidence_metrics(
        self,
        model_predictions: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate confidence metrics for the ensemble prediction.
        
        Args:
            model_predictions: Model predictions
            
        Returns:
            Dictionary of confidence metrics
        """
        # Agreement score (how much models agree)
        predictions = []
        for probs in model_predictions.values():
            predictions.append("danger" if probs["danger"] > probs["safe"] else "safe")
        
        agreement_score = Counter(predictions).most_common(1)[0][1] / len(predictions)
        
        # Certainty score (average max probability)
        certainties = [
            max(probs["safe"], probs["danger"]) 
            for probs in model_predictions.values()
        ]
        avg_certainty = np.mean(certainties)
        
        # Variance in predictions
        danger_probs = [probs["danger"] for probs in model_predictions.values()]
        prediction_variance = np.var(danger_probs)
        
        return {
            "agreement_score": float(agreement_score),
            "average_certainty": float(avg_certainty),
            "prediction_variance": float(prediction_variance),
            "ensemble_confidence": float(agreement_score * avg_certainty)
        }