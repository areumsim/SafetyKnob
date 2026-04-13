"""
Multi-model comparison and analysis module.

This module provides functionality for comparing and analyzing multiple
vision models for safety classification.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from ..core import SafetyClassifier, create_embedder
from ..utils.data_loader import load_image_dataset
from .single_model import SingleModelAnalyzer


logger = logging.getLogger(__name__)


class MultiModelAnalyzer:
    """Analyzer for comparing multiple models."""
    
    def __init__(
        self,
        model_types: List[str] = None,
        device: str = "cuda",
        output_dir: str = "results/multi_model"
    ):
        """
        Initialize multi-model analyzer.
        
        Args:
            model_types: List of models to compare
            device: Device to run on
            output_dir: Directory for saving results
        """
        self.model_types = model_types or ["siglip", "clip", "dinov2", "evaclip"]
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.single_analyzers = {
            model_type: SingleModelAnalyzer(model_type, device, output_dir)
            for model_type in self.model_types
        }
        
        self.results = {
            "model_types": self.model_types,
            "timestamp": datetime.now().isoformat(),
            "individual_results": {},
            "comparison": {},
            "ensemble_results": {}
        }
    
    def analyze(
        self,
        danger_paths: List[str],
        safe_paths: List[str],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Perform comprehensive multi-model analysis.
        
        Args:
            danger_paths: Paths to danger images
            safe_paths: Paths to safe images
            test_size: Fraction for test set
            random_state: Random seed
            
        Returns:
            Dictionary of analysis results
        """
        logger.info("Starting multi-model analysis")
        
        # Split data (same split for all models)
        danger_train, danger_test = train_test_split(
            danger_paths, test_size=test_size, random_state=random_state
        )
        safe_train, safe_test = train_test_split(
            safe_paths, test_size=test_size, random_state=random_state
        )
        
        # Analyze individual models
        logger.info("Analyzing individual models...")
        for model_type in self.model_types:
            analyzer = self.single_analyzers[model_type]
            result = analyzer.analyze(
                danger_paths, safe_paths, test_size, random_state
            )
            self.results["individual_results"][model_type] = result
        
        # Compare models
        logger.info("Comparing models...")
        self.results["comparison"] = self._compare_models()
        
        # Analyze ensemble
        logger.info("Analyzing ensemble performance...")
        self.results["ensemble_results"] = self._analyze_ensemble(
            danger_train, safe_train, danger_test, safe_test
        )
        
        # Generate comparison visualizations
        self._generate_comparison_plots()
        
        # Save results
        self._save_results()
        
        logger.info("Multi-model analysis complete")
        return self.results
    
    def _compare_models(self) -> Dict[str, Any]:
        """Compare performance across models."""
        comparison = {}
        
        # Collect metrics
        metrics_df = pd.DataFrame()
        for model_type, result in self.results["individual_results"].items():
            metrics = result["metrics"].copy()
            metrics.pop("confusion_matrix", None)
            metrics["model"] = model_type
            metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])])
        
        metrics_df = metrics_df.set_index("model")
        
        # Best model by metric
        comparison["best_by_metric"] = {
            metric: metrics_df[metric].idxmax()
            for metric in metrics_df.columns
        }
        
        # Statistical comparison
        comparison["metrics_stats"] = {
            "mean": metrics_df.mean().to_dict(),
            "std": metrics_df.std().to_dict(),
            "min": metrics_df.min().to_dict(),
            "max": metrics_df.max().to_dict()
        }
        
        # Ranking
        rankings = {}
        for metric in metrics_df.columns:
            rankings[metric] = metrics_df[metric].sort_values(ascending=False).index.tolist()
        comparison["rankings"] = rankings
        
        # Save metrics dataframe
        metrics_df.to_csv(os.path.join(self.output_dir, "metrics_comparison.csv"))
        
        return comparison
    
    def _analyze_ensemble(
        self,
        danger_train: List[str],
        safe_train: List[str],
        danger_test: List[str],
        safe_test: List[str]
    ) -> Dict[str, Any]:
        """Analyze ensemble performance."""
        # Train ensemble classifier
        ensemble_classifier = SafetyClassifier(
            model_types=self.model_types,
            device=self.device,
            ensemble_method="voting"
        )
        ensemble_classifier.train(danger_train, safe_train)
        
        # Test ensemble
        test_paths = danger_test + safe_test
        y_true = np.array([1] * len(danger_test) + [0] * len(safe_test))
        
        y_pred = []
        y_prob = []
        model_agreements = []
        
        for path in test_paths:
            result = ensemble_classifier.predict(path)
            y_pred.append(1 if result.prediction == "danger" else 0)
            y_prob.append(result.danger_score)
            
            # Calculate model agreement
            votes = list(result.model_votes.values())
            agreement = votes.count(result.prediction) / len(votes)
            model_agreements.append(agreement)
        
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        model_agreements = np.array(model_agreements)
        
        # Calculate ensemble metrics
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        ensemble_results = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "auc_score": float(roc_auc_score(y_true, y_prob)),
            "mean_agreement": float(np.mean(model_agreements)),
            "agreement_std": float(np.std(model_agreements))
        }
        
        # Compare to individual models
        individual_accuracies = [
            self.results["individual_results"][model]["metrics"]["accuracy"]
            for model in self.model_types
        ]
        
        ensemble_results["improvement_over_mean"] = (
            ensemble_results["accuracy"] - np.mean(individual_accuracies)
        )
        ensemble_results["improvement_over_best"] = (
            ensemble_results["accuracy"] - np.max(individual_accuracies)
        )
        
        return ensemble_results
    
    def _generate_comparison_plots(self):
        """Generate comparison visualizations."""
        # Metrics comparison bar plot
        metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "auc_score"]
        
        plt.figure(figsize=(12, 8))
        
        data = []
        for model in self.model_types:
            metrics = self.results["individual_results"][model]["metrics"]
            for metric in metrics_to_plot:
                data.append({
                    "Model": model,
                    "Metric": metric,
                    "Score": metrics[metric]
                })
        
        df = pd.DataFrame(data)
        
        sns.barplot(data=df, x="Metric", y="Score", hue="Model")
        plt.title("Model Performance Comparison")
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45)
        
        # Add ensemble results if available
        if "ensemble_results" in self.results and self.results["ensemble_results"]:
            ensemble_acc = self.results["ensemble_results"]["accuracy"]
            plt.axhline(y=ensemble_acc, color='red', linestyle='--', 
                       label=f'Ensemble Accuracy: {ensemble_acc:.3f}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "metrics_comparison.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        # Radar chart for model comparison
        self._create_radar_chart()
        
        # Correlation heatmap
        self._create_correlation_heatmap()
    
    def _create_radar_chart(self):
        """Create radar chart for model comparison."""
        metrics = ["accuracy", "precision", "recall", "f1_score", "auc_score"]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for model in self.model_types:
            values = [
                self.results["individual_results"][model]["metrics"][metric]
                for metric in metrics
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title("Model Performance Radar Chart", y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "radar_chart.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
    
    def _create_correlation_heatmap(self):
        """Create correlation heatmap of model predictions.

        Loads per-image predictions from scenario results and computes
        pairwise prediction score correlations between models.
        """
        results_base = os.path.join(os.path.dirname(self.output_dir), "scenario")

        model_predictions = {}
        for model in self.model_types:
            result_path = os.path.join(results_base, f"{model}_2layer", "results.json")
            if not os.path.exists(result_path):
                result_path = os.path.join(results_base, model, "results.json")
            if not os.path.exists(result_path):
                continue

            with open(result_path) as f:
                data = json.load(f)

            preds = data.get("per_image_predictions", {})
            if preds:
                model_predictions[model] = preds

        if len(model_predictions) < 2:
            logger.warning("Not enough models with per-image predictions for correlation heatmap")
            return

        common_images = set.intersection(*[set(p.keys()) for p in model_predictions.values()])
        if not common_images:
            logger.warning("No common images found across models")
            return

        common_images = sorted(common_images)

        score_matrix = {}
        for model, preds in model_predictions.items():
            score_matrix[model] = [preds[img]["pred_score"] for img in common_images]

        df = pd.DataFrame(score_matrix)
        corr = df.corr(method="pearson")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr, annot=True, fmt=".4f", cmap="RdYlBu_r",
            vmin=0.5, vmax=1.0, square=True,
            xticklabels=corr.columns, yticklabels=corr.columns,
            ax=ax
        )
        ax.set_title("Model Prediction Score Correlation (Pearson)")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "correlation_heatmap.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close()

        corr_path = os.path.join(self.output_dir, "prediction_correlations.json")
        with open(corr_path, 'w') as f:
            json.dump({
                "method": "pearson",
                "n_images": len(common_images),
                "correlation_matrix": corr.to_dict(),
                "models": list(model_predictions.keys())
            }, f, indent=2)

        logger.info(f"Correlation heatmap saved ({len(common_images)} images, {len(model_predictions)} models)")
    
    def _save_results(self):
        """Save analysis results."""
        # Save JSON results
        results_path = os.path.join(self.output_dir, 'multi_model_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate comparison report
        self._generate_comparison_report()
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        report_path = os.path.join(self.output_dir, 'comparison_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Multi-Model Comparison Report\n\n")
            f.write(f"Generated: {self.results['timestamp']}\n\n")
            
            # Model summary
            f.write("## Model Performance Summary\n\n")
            f.write("| Model | Accuracy | Precision | Recall | F1 Score | AUC |\n")
            f.write("|-------|----------|-----------|--------|----------|-----|\n")
            
            for model in self.model_types:
                metrics = self.results["individual_results"][model]["metrics"]
                f.write(f"| {model} | {metrics['accuracy']:.4f} | "
                       f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                       f"{metrics['f1_score']:.4f} | {metrics['auc_score']:.4f} |\n")
            
            # Best models
            f.write("\n## Best Models by Metric\n\n")
            for metric, best_model in self.results["comparison"]["best_by_metric"].items():
                f.write(f"- **{metric}**: {best_model}\n")
            
            # Ensemble results
            if self.results.get("ensemble_results"):
                f.write("\n## Ensemble Performance\n\n")
                ens = self.results["ensemble_results"]
                f.write(f"- **Accuracy**: {ens['accuracy']:.4f}\n")
                f.write(f"- **AUC Score**: {ens['auc_score']:.4f}\n")
                f.write(f"- **Mean Model Agreement**: {ens['mean_agreement']:.4f}\n")
                f.write(f"- **Improvement over mean**: {ens['improvement_over_mean']:.4f}\n")
                f.write(f"- **Improvement over best**: {ens['improvement_over_best']:.4f}\n")
            
            # Recommendations
            f.write("\n## Recommendations\n\n")
            best_acc_model = self.results["comparison"]["best_by_metric"]["accuracy"]
            f.write(f"1. For highest accuracy, use **{best_acc_model}**\n")
            f.write("2. For robust predictions, use the ensemble approach\n")
            f.write("3. Consider the trade-off between precision and recall for your use case\n")