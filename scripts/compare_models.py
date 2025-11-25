"""
Model Comparison and Analysis Script
Compares performance of all trained models and generates comprehensive reports
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Constants
RESULTS_DIR = Path(__file__).parent.parent / "results" / "single_models"
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "comparison"


def load_model_results() -> Dict:
    """Load results from all trained models"""
    models = {}

    for model_dir in RESULTS_DIR.iterdir():
        if model_dir.is_dir():
            results_file = model_dir / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    models[data['model']] = data

    return models


def create_comparison_table(models: Dict) -> str:
    """Create markdown comparison table"""
    lines = []
    lines.append("# Model Performance Comparison\n")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## Overall Performance\n")
    lines.append("| Model | Accuracy | F1 Score | Precision | Recall | AUC-ROC | Avg Dim F1 | Training Time (hrs) |")
    lines.append("|-------|----------|----------|-----------|--------|---------|------------|---------------------|")

    for model_name, data in sorted(models.items(), key=lambda x: x[1]['test_metrics']['overall']['f1'], reverse=True):
        metrics = data['test_metrics']['overall']
        dim_f1 = data['test_metrics']['avg_dimension_f1']
        train_time = data['training_time_seconds'] / 3600

        lines.append(
            f"| **{model_name.upper()}** | "
            f"{metrics['accuracy']:.4f} | "
            f"{metrics['f1']:.4f} | "
            f"{metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | "
            f"{metrics['auc_roc']:.4f} | "
            f"{dim_f1:.4f} | "
            f"{train_time:.2f} |"
        )

    lines.append("\n## Dimension-wise Performance\n")

    # Collect all dimensions
    dimensions = list(next(iter(models.values()))['test_metrics']['dimensions'].keys())

    for dim in dimensions:
        lines.append(f"\n### {dim.replace('_', ' ').title()}\n")
        lines.append("| Model | F1 Score | Accuracy |")
        lines.append("|-------|----------|----------|")

        dim_scores = []
        for model_name, data in models.items():
            dim_metrics = data['test_metrics']['dimensions'][dim]
            dim_scores.append((model_name, dim_metrics['f1'], dim_metrics['accuracy']))

        # Sort by F1
        dim_scores.sort(key=lambda x: x[1], reverse=True)

        for model_name, f1, acc in dim_scores:
            lines.append(f"| **{model_name.upper()}** | {f1:.4f} | {acc:.4f} |")

    return '\n'.join(lines)


def create_visualizations(models: Dict):
    """Create comparison visualizations"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Overall Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    model_names = list(models.keys())
    metrics_to_plot = ['accuracy', 'f1', 'precision', 'recall']
    metric_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall']

    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        values = [models[m]['test_metrics']['overall'][metric] for m in model_names]

        bars = ax.bar(model_names, values, color=['#2E86AB', '#A23B72', '#F18F01'])
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label} by Model', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'overall_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Dimension Performance Heatmap
    dimensions = list(next(iter(models.values()))['test_metrics']['dimensions'].keys())
    dim_labels = [d.replace('_', ' ').title() for d in dimensions]

    # Create F1 matrix
    f1_matrix = []
    for model in model_names:
        row = [models[model]['test_metrics']['dimensions'][dim]['f1'] for dim in dimensions]
        f1_matrix.append(row)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(f1_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(dimensions)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels(dim_labels, rotation=45, ha='right')
    ax.set_yticklabels([m.upper() for m in model_names])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('F1 Score', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(dimensions)):
            text = ax.text(j, i, f'{f1_matrix[i][j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title('Dimension-wise F1 Scores by Model', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dimension_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Training Time vs Performance
    fig, ax = plt.subplots(figsize=(10, 6))

    train_times = [models[m]['training_time_seconds'] / 3600 for m in model_names]
    f1_scores = [models[m]['test_metrics']['overall']['f1'] for m in model_names]

    scatter = ax.scatter(train_times, f1_scores, s=200, alpha=0.6,
                        c=['#2E86AB', '#A23B72', '#F18F01'])

    for i, model in enumerate(model_names):
        ax.annotate(model.upper(), (train_times[i], f1_scores[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')

    ax.set_xlabel('Training Time (hours)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Training Efficiency: Time vs Performance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Dimension Average Comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(dimensions))
    width = 0.25

    for i, model in enumerate(model_names):
        values = [models[model]['test_metrics']['dimensions'][dim]['f1'] for dim in dimensions]
        offset = (i - 1) * width
        ax.bar(x + offset, values, width, label=model.upper(), alpha=0.8)

    ax.set_xlabel('Safety Dimension', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Dimension Performance by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dimension_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Visualizations saved to {OUTPUT_DIR}")


def generate_insights(models: Dict) -> str:
    """Generate insights from model comparison"""
    lines = []
    lines.append("\n## Key Insights\n")

    # Best overall model
    best_model = max(models.items(), key=lambda x: x[1]['test_metrics']['overall']['f1'])
    lines.append(f"### Best Overall Performance")
    lines.append(f"- **{best_model[0].upper()}** achieves the highest F1 score of **{best_model[1]['test_metrics']['overall']['f1']:.4f}**")
    lines.append(f"- AUC-ROC: **{best_model[1]['test_metrics']['overall']['auc_roc']:.4f}** (near-perfect classification ability)")

    # Training efficiency
    fastest_model = min(models.items(), key=lambda x: x[1]['training_time_seconds'])
    lines.append(f"\n### Training Efficiency")
    lines.append(f"- **{fastest_model[0].upper()}** trains fastest in **{fastest_model[1]['training_time_seconds']/3600:.2f} hours**")

    # Dimension analysis
    dimensions = list(next(iter(models.values()))['test_metrics']['dimensions'].keys())
    lines.append(f"\n### Dimension-wise Analysis")

    for dim in dimensions:
        best_for_dim = max(models.items(),
                          key=lambda x: x[1]['test_metrics']['dimensions'][dim]['f1'])
        f1_score = best_for_dim[1]['test_metrics']['dimensions'][dim]['f1']

        lines.append(f"- **{dim.replace('_', ' ').title()}**: {best_for_dim[0].upper()} ({f1_score:.4f})")

    # Overall statistics
    lines.append(f"\n### Statistical Summary")
    all_f1s = [m['test_metrics']['overall']['f1'] for m in models.values()]
    lines.append(f"- Mean F1 Score: **{np.mean(all_f1s):.4f}**")
    lines.append(f"- Std Dev: **{np.std(all_f1s):.4f}**")
    lines.append(f"- Range: **{np.min(all_f1s):.4f} - {np.max(all_f1s):.4f}**")

    # Dimension performance gap
    avg_dim_f1s = [m['test_metrics']['avg_dimension_f1'] for m in models.values()]
    avg_overall_f1s = [m['test_metrics']['overall']['f1'] for m in models.values()]
    gap = np.mean(avg_overall_f1s) - np.mean(avg_dim_f1s)

    lines.append(f"\n### Performance Gap Analysis")
    lines.append(f"- Overall F1 (mean): **{np.mean(avg_overall_f1s):.4f}**")
    lines.append(f"- Dimension F1 (mean): **{np.mean(avg_dim_f1s):.4f}**")
    lines.append(f"- **Gap**: {gap:.4f} ({gap/np.mean(avg_overall_f1s)*100:.1f}%)")
    lines.append(f"\n**Interpretation**: Dimension-specific tasks are more challenging than binary safety classification.")

    return '\n'.join(lines)


def save_json_summary(models: Dict):
    """Save comprehensive JSON summary"""
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_models": len(models),
        "models": models,
        "best_overall": max(models.items(), key=lambda x: x[1]['test_metrics']['overall']['f1'])[0],
        "summary_statistics": {
            "mean_f1": float(np.mean([m['test_metrics']['overall']['f1'] for m in models.values()])),
            "std_f1": float(np.std([m['test_metrics']['overall']['f1'] for m in models.values()])),
            "mean_auc": float(np.mean([m['test_metrics']['overall']['auc_roc'] for m in models.values()])),
            "mean_dim_f1": float(np.mean([m['test_metrics']['avg_dimension_f1'] for m in models.values()]))
        }
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ JSON summary saved to {OUTPUT_DIR / 'comparison_summary.json'}")


def main():
    """Main execution"""
    print("=" * 60)
    print("MODEL COMPARISON AND ANALYSIS")
    print("=" * 60)

    # Load results
    print("\n📂 Loading model results...")
    models = load_model_results()
    print(f"✅ Loaded {len(models)} models: {', '.join(models.keys())}")

    # Create comparison table
    print("\n📊 Creating comparison table...")
    table = create_comparison_table(models)
    insights = generate_insights(models)

    # Save markdown report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / 'comparison_report.md', 'w') as f:
        f.write(table)
        f.write(insights)

    print(f"✅ Markdown report saved to {OUTPUT_DIR / 'comparison_report.md'}")

    # Create visualizations
    print("\n📈 Creating visualizations...")
    create_visualizations(models)

    # Save JSON summary
    print("\n💾 Saving JSON summary...")
    save_json_summary(models)

    print("\n" + "=" * 60)
    print("✅ COMPARISON COMPLETE!")
    print(f"📁 All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
