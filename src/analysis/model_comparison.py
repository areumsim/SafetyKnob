"""
Model Performance Comparison and Analysis Tools
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
import logging

logger = logging.getLogger(__name__)


class ModelPerformanceAnalyzer:
    """Analyze and compare performance across multiple models"""
    
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def compare_models(self, evaluation_results: Dict) -> Dict:
        """
        Compare performance across models
        
        Args:
            evaluation_results: Results from SafetyAssessmentSystem.evaluate_dataset()
            
        Returns:
            Comprehensive comparison report
        """
        ensemble_metrics = evaluation_results["ensemble_metrics"]
        individual_metrics = evaluation_results["individual_metrics"]
        
        # Create comparison dataframe
        comparison_data = []
        
        # Add individual model results
        for model_name, metrics in individual_metrics.items():
            comparison_data.append({
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1_score"],
                "Type": "Individual"
            })
        
        # Add ensemble result
        comparison_data.append({
            "Model": "Ensemble",
            "Accuracy": ensemble_metrics["accuracy"],
            "Precision": ensemble_metrics["precision"],
            "Recall": ensemble_metrics["recall"],
            "F1 Score": ensemble_metrics["f1_score"],
            "Type": "Ensemble"
        })
        
        df = pd.DataFrame(comparison_data)
        
        # Calculate improvements
        best_individual_f1 = max(m["f1_score"] for m in individual_metrics.values())
        ensemble_improvement = ensemble_metrics["f1_score"] - best_individual_f1
        
        # Statistical analysis
        individual_f1_scores = [m["f1_score"] for m in individual_metrics.values()]
        f1_mean = np.mean(individual_f1_scores)
        f1_std = np.std(individual_f1_scores)
        
        report = {
            "comparison_table": df.to_dict('records'),
            "best_individual_model": evaluation_results.get("best_individual_model"),
            "ensemble_improvement": {
                "absolute": ensemble_improvement,
                "percentage": (ensemble_improvement / best_individual_f1) * 100
            },
            "individual_model_stats": {
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "f1_range": (min(individual_f1_scores), max(individual_f1_scores))
            },
            "ranking": df.sort_values("F1 Score", ascending=False)["Model"].tolist()
        }
        
        # Save report
        self._save_report(report, "model_comparison_report.json")
        
        return report
    
    def visualize_comparison(self, evaluation_results: Dict, save_path: Optional[str] = None):
        """Create visualization of model comparisons"""
        comparison = self.compare_models(evaluation_results)
        df = pd.DataFrame(comparison["comparison_table"])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # 1. Bar plot of all metrics
        ax1 = axes[0, 0]
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        x = np.arange(len(df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax1.bar(x + i*width, df[metric], width, label=metric)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics by Model')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(df['Model'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. F1 Score comparison with improvement highlight
        ax2 = axes[0, 1]
        colors = ['skyblue' if t == 'Individual' else 'lightcoral' for t in df['Type']]
        bars = ax2.bar(df['Model'], df['F1 Score'], color=colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score Comparison')
        ax2.set_xticklabels(df['Model'], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add improvement annotation
        if comparison["ensemble_improvement"]["absolute"] > 0:
            ax2.annotate(
                f'+{comparison["ensemble_improvement"]["percentage"]:.1f}%',
                xy=(len(df)-1, df.iloc[-1]['F1 Score']),
                xytext=(len(df)-1, df.iloc[-1]['F1 Score'] + 0.05),
                ha='center',
                fontsize=12,
                color='green',
                weight='bold'
            )
        
        # 3. Confusion matrices heatmap
        ax3 = axes[1, 0]
        ensemble_cm = evaluation_results["ensemble_metrics"]["confusion_matrix"]
        cm_data = [[ensemble_cm["tn"], ensemble_cm["fp"]], 
                   [ensemble_cm["fn"], ensemble_cm["tp"]]]
        
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted Safe', 'Predicted Unsafe'],
                   yticklabels=['Actual Safe', 'Actual Unsafe'],
                   ax=ax3)
        ax3.set_title('Ensemble Confusion Matrix')
        
        # 4. Model ranking
        ax4 = axes[1, 1]
        ranking_df = df.sort_values('F1 Score', ascending=True)
        y_pos = np.arange(len(ranking_df))
        
        bars = ax4.barh(y_pos, ranking_df['F1 Score'])
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(ranking_df['Model'])
        ax4.set_xlabel('F1 Score')
        ax4.set_title('Model Ranking')
        
        # Color code bars
        for i, bar in enumerate(bars):
            if ranking_df.iloc[i]['Type'] == 'Ensemble':
                bar.set_color('lightcoral')
            else:
                bar.set_color('skyblue')
        
        # Add values
        for i, v in enumerate(ranking_df['F1 Score']):
            ax4.text(v + 0.005, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.results_dir / "model_comparison_plot.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        logger.info(f"Visualization saved to {save_path}")
        
    def generate_performance_report(self, evaluation_results: Dict) -> str:
        """Generate a detailed performance report"""
        comparison = self.compare_models(evaluation_results)
        
        report_lines = [
            "# Model Performance Analysis Report",
            "=" * 60,
            "",
            "## Executive Summary",
            f"- Best Individual Model: {comparison['best_individual_model']}",
            f"- Ensemble Improvement: {comparison['ensemble_improvement']['percentage']:.1f}%",
            f"- Total Models Evaluated: {len(evaluation_results['individual_metrics'])}",
            "",
            "## Detailed Performance Metrics",
            ""
        ]
        
        # Add comparison table
        df = pd.DataFrame(comparison["comparison_table"])
        report_lines.append(df.to_string(index=False))
        report_lines.append("")
        
        # Individual model analysis
        report_lines.extend([
            "## Individual Model Analysis",
            f"- Mean F1 Score: {comparison['individual_model_stats']['f1_mean']:.3f}",
            f"- Std Dev: {comparison['individual_model_stats']['f1_std']:.3f}",
            f"- Range: {comparison['individual_model_stats']['f1_range'][0]:.3f} - {comparison['individual_model_stats']['f1_range'][1]:.3f}",
            ""
        ])
        
        # Model rankings
        report_lines.extend([
            "## Model Rankings (by F1 Score)",
        ])
        for i, model in enumerate(comparison["ranking"], 1):
            report_lines.append(f"{i}. {model}")
        
        report_lines.append("")
        
        # Ensemble benefits
        if comparison["ensemble_improvement"]["absolute"] > 0:
            report_lines.extend([
                "## Ensemble Benefits",
                f"The ensemble approach improved F1 score by {comparison['ensemble_improvement']['absolute']:.3f} ",
                f"({comparison['ensemble_improvement']['percentage']:.1f}% improvement) over the best individual model.",
                "",
                "This demonstrates the value of combining multiple vision models for safety assessment."
            ])
        
        # Save report
        report_text = "\n".join(report_lines)
        report_path = self.results_dir / "performance_report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)
        
        logger.info(f"Performance report saved to {report_path}")
        return report_text
    
    def analyze_model_agreements(self, predictions_log: List[Dict]) -> Dict:
        """Analyze how often models agree/disagree"""
        # Count agreements between models
        model_names = list(predictions_log[0]["individual_predictions"].keys())
        n_models = len(model_names)
        
        agreement_matrix = np.zeros((n_models, n_models))
        
        for pred in predictions_log:
            individual_preds = pred["individual_predictions"]
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if individual_preds[model1] == individual_preds[model2]:
                        agreement_matrix[i, j] += 1
        
        # Normalize by number of predictions
        agreement_matrix /= len(predictions_log)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(agreement_matrix, annot=True, fmt='.2f', 
                   xticklabels=model_names, yticklabels=model_names,
                   cmap='YlOrRd', vmin=0, vmax=1)
        plt.title('Model Agreement Matrix')
        plt.tight_layout()
        
        save_path = self.results_dir / "model_agreement_heatmap.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        return {
            "agreement_matrix": agreement_matrix.tolist(),
            "model_names": model_names,
            "average_agreement": np.mean(agreement_matrix[np.triu_indices(n_models, k=1)])
        }
    
    def _save_report(self, report: Dict, filename: str):
        """Save report to JSON file"""
        save_path = self.results_dir / filename
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {save_path}")