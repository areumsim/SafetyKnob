"""
Visualization utilities for safety classification analysis.

This module provides functions for creating various plots and visualizations
for model performance analysis and embedding visualization.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pandas as pd


def setup_plotting_style():
    """Set up consistent plotting style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    normalize: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: Class names
        title: Plot title
        cmap: Colormap
        normalize: Whether to normalize values
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve"
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_embedding_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "pca",
    title: str = "Embedding Space Visualization",
    class_names: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize high-dimensional embeddings in 2D.
    
    Args:
        embeddings: Embedding vectors
        labels: Class labels
        method: Reduction method ("pca", "tsne", "umap")
        title: Plot title
        class_names: Mapping of label to class name
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Reduce dimensions
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = reducer.fit_transform(embeddings)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each class
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        class_name = class_names.get(label, f"Class {label}") if class_names else f"Class {label}"
        
        ax.scatter(reduced[mask, 0], reduced[mask, 1],
                  c=[color], label=class_name, alpha=0.6, s=50)
    
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    title: str = "Model Performance Comparison"
) -> plt.Figure:
    """
    Create bar plot comparing multiple models.
    
    Args:
        results: Dictionary mapping model names to metrics
        metrics: List of metrics to plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1_score", "auc_score"]
    
    # Prepare data
    models = list(results.keys())
    n_models = len(models)
    n_metrics = len(metrics)
    
    # Create bar positions
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars for each model
    for i, model in enumerate(models):
        values = [results[model].get(metric, 0) for metric in metrics]
        offset = (i - n_models/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model)
    
    # Customize plot
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
    
    fig.tight_layout()
    return fig


def plot_threshold_analysis(
    threshold_results: List[Dict[str, float]],
    metric: str = "f1_score",
    title: str = "Threshold Analysis"
) -> plt.Figure:
    """
    Plot metric values across different thresholds.
    
    Args:
        threshold_results: List of metrics for each threshold
        metric: Metric to plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    thresholds = [r["threshold"] for r in threshold_results]
    values = [r[metric] for r in threshold_results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, values, 'b-', linewidth=2, marker='o')
    
    # Mark optimal threshold
    max_idx = np.argmax(values)
    ax.axvline(x=thresholds[max_idx], color='r', linestyle='--', alpha=0.7,
               label=f'Optimal: {thresholds[max_idx]:.2f}')
    ax.scatter(thresholds[max_idx], values[max_idx], color='r', s=100, zorder=5)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    
    return fig


def create_performance_report(
    results: Dict[str, Any],
    output_dir: str,
    model_name: str = "Model"
):
    """
    Create comprehensive performance report with multiple plots.
    
    Args:
        results: Dictionary containing all analysis results
        output_dir: Directory to save plots
        model_name: Name of the model
    """
    os.makedirs(output_dir, exist_ok=True)
    setup_plotting_style()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Confusion Matrix
    if "confusion_matrix" in results and "y_true" in results and "y_pred" in results:
        ax1 = plt.subplot(2, 3, 1)
        cm = confusion_matrix(results["y_true"], results["y_pred"])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Safe', 'Danger'],
                    yticklabels=['Safe', 'Danger'])
        ax1.set_title(f'{model_name} - Confusion Matrix')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
    
    # 2. ROC Curve
    if "y_true" in results and "y_prob" in results:
        ax2 = plt.subplot(2, 3, 2)
        fpr, tpr, _ = roc_curve(results["y_true"], results["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, 'b-', linewidth=2,
                label=f'AUC = {roc_auc:.3f}')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title(f'{model_name} - ROC Curve')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
    
    # 3. Probability Distribution
    if "y_true" in results and "y_prob" in results:
        ax3 = plt.subplot(2, 3, 3)
        y_true = results["y_true"]
        y_prob = results["y_prob"]
        
        ax3.hist(y_prob[y_true == 0], bins=20, alpha=0.5,
                label='Safe', color='green', density=True)
        ax3.hist(y_prob[y_true == 1], bins=20, alpha=0.5,
                label='Danger', color='red', density=True)
        ax3.set_xlabel('Predicted Probability (Danger)')
        ax3.set_ylabel('Density')
        ax3.set_title(f'{model_name} - Probability Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Metrics Bar Chart
    if "metrics" in results:
        ax4 = plt.subplot(2, 3, 4)
        metrics = results["metrics"]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0)
        ]
        
        bars = ax4.bar(metric_names, metric_values, color='skyblue')
        ax4.set_ylim(0, 1.1)
        ax4.set_ylabel('Score')
        ax4.set_title(f'{model_name} - Performance Metrics')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # 5. Class Balance
    if "y_true" in results:
        ax5 = plt.subplot(2, 3, 5)
        y_true = results["y_true"]
        unique, counts = np.unique(y_true, return_counts=True)
        
        ax5.pie(counts, labels=['Safe', 'Danger'], autopct='%1.1f%%',
                colors=['green', 'red'], startangle=90)
        ax5.set_title(f'{model_name} - Class Distribution')
    
    # 6. Performance Summary Text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"{model_name} Performance Summary\n"
    summary_text += "=" * 40 + "\n\n"
    
    if "metrics" in results:
        metrics = results["metrics"]
        summary_text += f"Accuracy:  {metrics.get('accuracy', 0):.3f}\n"
        summary_text += f"Precision: {metrics.get('precision', 0):.3f}\n"
        summary_text += f"Recall:    {metrics.get('recall', 0):.3f}\n"
        summary_text += f"F1-Score:  {metrics.get('f1_score', 0):.3f}\n"
        summary_text += f"AUC Score: {metrics.get('auc_score', 0):.3f}\n\n"
    
    if "training_time" in results:
        summary_text += f"Training Time: {results['training_time']:.2f}s\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'{model_name}_performance_report.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path