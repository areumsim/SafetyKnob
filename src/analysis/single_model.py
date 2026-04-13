"""
Single model analysis module.

This module provides functionality for analyzing individual vision models
for safety classification performance.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from ..core import create_embedder, SafetyClassifier
from ..utils.data_loader import load_image_dataset
from ..utils.visualization import plot_confusion_matrix, plot_roc_curve


logger = logging.getLogger(__name__)


class SingleModelAnalyzer:
    """Analyzer for single model performance evaluation."""
    
    def __init__(
        self,
        model_type: str,
        device: str = "cuda",
        output_dir: str = "results/single_model"
    ):
        """
        Initialize single model analyzer.
        
        Args:
            model_type: Type of model to analyze
            device: Device to run on
            output_dir: Directory for saving results
        """
        self.model_type = model_type
        self.device = device
        self.output_dir = os.path.join(output_dir, model_type)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.embedder = create_embedder(
            model_type=model_type,
            device=device,
            cache_dir="data/cache"
        )
        
        self.results = {
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "analysis": {}
        }
    
    def analyze(
        self,
        danger_paths: List[str],
        safe_paths: List[str],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of single model.
        
        Args:
            danger_paths: Paths to danger images
            safe_paths: Paths to safe images
            test_size: Fraction for test set
            random_state: Random seed
            
        Returns:
            Dictionary of analysis results
        """
        logger.info(f"Starting analysis for {self.model_type}")
        
        # Split data
        danger_train, danger_test = train_test_split(
            danger_paths, test_size=test_size, random_state=random_state
        )
        safe_train, safe_test = train_test_split(
            safe_paths, test_size=test_size, random_state=random_state
        )
        
        # Training
        classifier = SafetyClassifier(
            model_types=[self.model_type],
            device=self.device
        )
        classifier.train(danger_train, safe_train)
        
        # Extract test embeddings
        logger.info("Extracting test embeddings...")
        danger_test_emb = self.embedder.extract_embeddings(danger_test)
        safe_test_emb = self.embedder.extract_embeddings(safe_test)
        
        # Save cache
        self.embedder.save_cache()
        
        # Prepare test data
        X_test = np.vstack([danger_test_emb, safe_test_emb])
        y_test = np.array([1] * len(danger_test_emb) + [0] * len(safe_test_emb))
        test_paths = danger_test + safe_test
        
        # Get predictions
        y_pred = []
        y_prob = []
        
        for i, path in enumerate(test_paths):
            result = classifier.predict(path)
            y_pred.append(1 if result.prediction == "danger" else 0)
            y_prob.append(result.danger_score)
        
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        # Calculate metrics
        self.results["metrics"] = self._calculate_metrics(y_test, y_pred, y_prob)
        
        # Perform additional analysis
        self.results["analysis"] = self._perform_analysis(
            X_test, y_test, y_pred, y_prob, classifier
        )
        
        # Generate visualizations
        self._generate_visualizations(y_test, y_pred, y_prob)
        
        # Save results
        self._save_results()
        
        logger.info(f"Analysis complete for {self.model_type}")
        return self.results
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary'
        )
        
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc_score": float(roc_auc_score(y_true, y_prob)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Calculate per-class metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics["true_positive_rate"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
        metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0
        metrics["true_negative_rate"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0
        metrics["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0
        
        return metrics
    
    def _perform_analysis(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        classifier: SafetyClassifier
    ) -> Dict[str, Any]:
        """Perform additional analysis."""
        analysis = {}
        
        # Embedding statistics
        analysis["embedding_stats"] = {
            "dimension": int(X_test.shape[1]),
            "n_test_samples": int(X_test.shape[0]),
            "n_danger": int(np.sum(y_test == 1)),
            "n_safe": int(np.sum(y_test == 0))
        }
        
        # Error analysis
        errors = y_test != y_pred
        error_indices = np.where(errors)[0]
        
        analysis["error_analysis"] = {
            "n_errors": int(np.sum(errors)),
            "error_rate": float(np.mean(errors)),
            "false_positives": int(np.sum((y_test == 0) & (y_pred == 1))),
            "false_negatives": int(np.sum((y_test == 1) & (y_pred == 0)))
        }
        
        # Confidence analysis
        analysis["confidence_stats"] = {
            "mean_confidence": float(np.mean(np.maximum(y_prob, 1 - y_prob))),
            "min_confidence": float(np.min(np.maximum(y_prob, 1 - y_prob))),
            "max_confidence": float(np.max(np.maximum(y_prob, 1 - y_prob))),
            "std_confidence": float(np.std(np.maximum(y_prob, 1 - y_prob)))
        }
        
        # Threshold analysis
        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_metrics = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_prob >= threshold).astype(int)
            acc = accuracy_score(y_test, y_pred_thresh)
            threshold_metrics.append({
                "threshold": float(threshold),
                "accuracy": float(acc)
            })
        
        analysis["threshold_analysis"] = threshold_metrics
        
        return analysis
    
    def _generate_visualizations(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ):
        """Generate visualization plots."""
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        plot_confusion_matrix(
            y_true, y_pred,
            classes=['Safe', 'Danger'],
            title=f'{self.model_type} Confusion Matrix'
        )
        plt.savefig(
            os.path.join(self.output_dir, 'confusion_matrix.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        # ROC Curve
        plt.figure(figsize=(8, 6))
        plot_roc_curve(
            y_true, y_prob,
            title=f'{self.model_type} ROC Curve'
        )
        plt.savefig(
            os.path.join(self.output_dir, 'roc_curve.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        # Probability distribution
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.hist(y_prob[y_true == 0], bins=20, alpha=0.5, label='Safe', color='green')
        plt.hist(y_prob[y_true == 1], bins=20, alpha=0.5, label='Danger', color='red')
        plt.xlabel('Danger Probability')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Probability Distribution by True Class')
        
        plt.subplot(1, 2, 2)
        correct = y_true == y_pred
        plt.hist(y_prob[correct], bins=20, alpha=0.5, label='Correct', color='blue')
        plt.hist(y_prob[~correct], bins=20, alpha=0.5, label='Incorrect', color='orange')
        plt.xlabel('Danger Probability')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Probability Distribution by Prediction Correctness')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, 'probability_distribution.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
    
    def _save_results(self):
        """Save analysis results."""
        # Save JSON results
        results_path = os.path.join(self.output_dir, 'analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate summary report
        report_path = os.path.join(self.output_dir, 'summary_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Single Model Analysis Report: {self.model_type}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Performance Metrics:\n")
            f.write("-" * 30 + "\n")
            for metric, value in self.results["metrics"].items():
                if metric != "confusion_matrix":
                    f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\nError Analysis:\n")
            f.write("-" * 30 + "\n")
            error_stats = self.results["analysis"]["error_analysis"]
            for key, value in error_stats.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nConfidence Statistics:\n")
            f.write("-" * 30 + "\n")
            conf_stats = self.results["analysis"]["confidence_stats"]
            for key, value in conf_stats.items():
                f.write(f"{key}: {value:.4f}\n")
        
        logger.info(f"Results saved to {self.output_dir}")