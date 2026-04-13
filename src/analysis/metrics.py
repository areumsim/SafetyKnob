"""
Performance metrics calculation and analysis.

This module provides comprehensive metrics for evaluating
safety classification performance.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score
)


class MetricsCalculator:
    """Calculator for comprehensive classification metrics."""
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        pos_label: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate all available metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            pos_label: Positive class label
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, pos_label=pos_label))
        metrics["recall"] = float(recall_score(y_true, y_pred, pos_label=pos_label))
        metrics["f1_score"] = float(f1_score(y_true, y_pred, pos_label=pos_label))
        
        # Additional metrics
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["matthews_corrcoef"] = float(matthews_corrcoef(y_true, y_pred))
        metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Extract confusion matrix values
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_positives"] = int(tp)
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            
            # Rates
            metrics["true_positive_rate"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
            metrics["true_negative_rate"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0
            metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0
            metrics["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0
            
            # Additional ratios
            metrics["positive_predictive_value"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0
            metrics["negative_predictive_value"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0
        
        # Probability-based metrics (if probabilities provided)
        if y_prob is not None:
            metrics["auc_score"] = float(roc_auc_score(y_true, y_prob))
            metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
            
            # Probability statistics
            metrics["prob_stats"] = {
                "mean": float(np.mean(y_prob)),
                "std": float(np.std(y_prob)),
                "min": float(np.min(y_prob)),
                "max": float(np.max(y_prob)),
                "median": float(np.median(y_prob))
            }
            
            # Probability distribution by class
            if pos_label == 1:
                danger_probs = y_prob[y_true == 1]
                safe_probs = y_prob[y_true == 0]
            else:
                danger_probs = y_prob[y_true == 0]
                safe_probs = y_prob[y_true == 1]
            
            metrics["class_prob_stats"] = {
                "danger": {
                    "mean": float(np.mean(danger_probs)) if len(danger_probs) > 0 else 0,
                    "std": float(np.std(danger_probs)) if len(danger_probs) > 0 else 0,
                    "median": float(np.median(danger_probs)) if len(danger_probs) > 0 else 0
                },
                "safe": {
                    "mean": float(np.mean(safe_probs)) if len(safe_probs) > 0 else 0,
                    "std": float(np.std(safe_probs)) if len(safe_probs) > 0 else 0,
                    "median": float(np.median(safe_probs)) if len(safe_probs) > 0 else 0
                }
            }
        
        # Classification report
        metrics["classification_report"] = classification_report(
            y_true, y_pred, output_dict=True
        )
        
        return metrics
    
    @staticmethod
    def calculate_threshold_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: Optional[List[float]] = None
    ) -> List[Dict[str, float]]:
        """
        Calculate metrics at different probability thresholds.
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            thresholds: List of thresholds to evaluate
            
        Returns:
            List of metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            metrics = {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, zero_division=0))
            }
            
            # Add confusion matrix values
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics["true_positive_rate"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
                metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0
            
            results.append(metrics)
        
        return results
    
    @staticmethod
    def find_optimal_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metric: str = "f1_score"
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal threshold based on specified metric.
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            metric: Metric to optimize
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        threshold_metrics = MetricsCalculator.calculate_threshold_metrics(
            y_true, y_prob
        )
        
        # Find best threshold
        best_threshold = None
        best_score = -1
        best_metrics = None
        
        for tm in threshold_metrics:
            if tm[metric] > best_score:
                best_score = tm[metric]
                best_threshold = tm["threshold"]
                best_metrics = tm
        
        return best_threshold, best_metrics
    
    @staticmethod
    def calculate_class_weights(y_true: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets.
        
        Args:
            y_true: True labels
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_true)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_true
        )
        
        return {cls: weight for cls, weight in zip(classes, weights)}
    
    @staticmethod
    def calculate_reliability_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate reliability/calibration metrics.
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary of reliability metrics
        """
        from sklearn.calibration import calibration_curve
        
        # Calculate calibration curve
        fraction_pos, mean_predicted = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper, frac, pred in zip(
            bin_lowers, bin_uppers, fraction_pos, mean_predicted
        ):
            # Proportion of samples in bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.sum() / len(y_prob)
            
            if prop_in_bin > 0:
                # Calibration error for this bin
                bin_error = np.abs(frac - pred)
                ece += prop_in_bin * bin_error
        
        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(fraction_pos - mean_predicted))
        
        return {
            "expected_calibration_error": float(ece),
            "maximum_calibration_error": float(mce),
            "calibration_curve": {
                "fraction_positive": fraction_pos.tolist(),
                "mean_predicted": mean_predicted.tolist()
            }
        }