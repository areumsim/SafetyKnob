"""
Ensemble Experiment Script
Tests ensemble methods (Weighted Vote, Stacking) using trained models
"""

import sys
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import SafetyAssessmentSystem
from src.config.settings import SystemConfig
from src.utils import ImageDataset
from tqdm import tqdm


# Legacy SafetyClassifier architecture (used for trained models)
class LegacySafetyClassifier(nn.Module):
    """Original SafetyClassifier architecture used for training"""

    def __init__(self, embedding_dim, hidden_dim=512, num_dimensions=5):
        super().__init__()
        self.dimension_names = ['fall_hazard', 'collision_risk', 'equipment_hazard',
                               'environmental_risk', 'protective_gear']

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Overall safety head
        self.safety_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Dimension heads
        self.dimension_heads = nn.ModuleDict({
            dim_name: nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            for dim_name in self.dimension_names
        })

    def forward(self, x):
        features = self.feature_extractor(x)
        overall_safety = self.safety_head(features)

        dimension_scores = {}
        for dim_name, head in self.dimension_heads.items():
            dimension_scores[dim_name] = head(features)

        return overall_safety, dimension_scores

def load_test_dataset(data_dir: Path, labels_file: Path):
    """Load test dataset"""
    print(f"Loading test data from {data_dir}")

    # Load labels
    with open(labels_file, 'r') as f:
        all_labels = json.load(f)

    # Filter test labels
    test_labels = {k.replace('test/', ''): v for k, v in all_labels.items() if k.startswith('test/')}

    # Get image paths
    image_paths = [data_dir / 'test' / img_name for img_name in test_labels.keys()]
    labels = [test_labels[img_name]['overall_safety'] for img_name in test_labels.keys()]

    print(f"Loaded {len(image_paths)} test images")
    print(f"  Safe: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"  Unsafe: {len(labels) - sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")

    return image_paths, labels, test_labels

def evaluate_ensemble(system: SafetyAssessmentSystem,
                     image_paths: list,
                     labels: list,
                     test_labels: dict):
    """Evaluate ensemble on test set"""
    print("\nEvaluating ensemble...")

    predictions = []
    true_labels = []
    dimension_preds = {dim: [] for dim in ['fall_hazard', 'collision_risk', 'equipment_hazard', 'environmental_risk', 'protective_gear']}
    dimension_true = {dim: [] for dim in dimension_preds.keys()}

    start_time = time.time()

    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Get ensemble prediction
            result = system.assess_image(str(img_path))

            predictions.append(1 if result.is_safe else 0)
            true_label = labels[image_paths.index(img_path)]
            true_labels.append(true_label)

            # Dimension predictions
            img_name = img_path.name
            for dim in dimension_preds.keys():
                dimension_preds[dim].append(result.dimension_scores.get(dim, 0.5))
                dimension_true[dim].append(test_labels[img_name].get(dim, 0.5))

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue

    processing_time = time.time() - start_time

    # Calculate metrics
    print("\nCalculating metrics...")

    # Overall metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    overall_metrics = {
        'accuracy': float(accuracy_score(true_labels, predictions)),
        'f1': float(f1_score(true_labels, predictions)),
        'precision': float(precision_score(true_labels, predictions)),
        'recall': float(recall_score(true_labels, predictions)),
        'auc_roc': float(roc_auc_score(true_labels, predictions))
    }

    # Dimension metrics
    dimension_metrics = {}
    for dim in dimension_preds.keys():
        dim_pred_binary = [1 if p > 0.5 else 0 for p in dimension_preds[dim]]
        dim_true_binary = [1 if t > 0.5 else 0 for t in dimension_true[dim]]

        dimension_metrics[dim] = {
            'f1': float(f1_score(dim_true_binary, dim_pred_binary, zero_division=0)),
            'accuracy': float(accuracy_score(dim_true_binary, dim_pred_binary))
        }

    avg_dim_f1 = np.mean([m['f1'] for m in dimension_metrics.values()])

    results = {
        'model': 'ensemble',
        'method': system.config.ensemble_strategy,
        'num_models': len(system.config.models),
        'models_used': [m.name if hasattr(m, 'name') else m['name'] for m in system.config.models],
        'test_size': len(true_labels),
        'processing_time_seconds': processing_time,
        'avg_time_per_image': processing_time / len(true_labels),
        'test_metrics': {
            'overall': overall_metrics,
            'dimensions': dimension_metrics,
            'avg_dimension_f1': float(avg_dim_f1)
        }
    }

    return results

def print_results(results: dict):
    """Print results in a formatted way"""
    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION RESULTS")
    print("="*60)
    print(f"\nModel: {results['model'].upper()}")
    print(f"Method: {results['method']}")
    print(f"Models Used: {', '.join(results['models_used'])}")
    print(f"Test Size: {results['test_size']} images")
    print(f"Processing Time: {results['processing_time_seconds']:.1f}s ({results['avg_time_per_image']*1000:.1f}ms/image)")

    print("\n" + "-"*60)
    print("OVERALL PERFORMANCE")
    print("-"*60)
    overall = results['test_metrics']['overall']
    print(f"Accuracy:  {overall['accuracy']*100:.2f}%")
    print(f"F1 Score:  {overall['f1']*100:.2f}%")
    print(f"Precision: {overall['precision']*100:.2f}%")
    print(f"Recall:    {overall['recall']*100:.2f}%")
    print(f"AUC-ROC:   {overall['auc_roc']*100:.2f}%")

    print("\n" + "-"*60)
    print("DIMENSION PERFORMANCE")
    print("-"*60)
    dimensions = results['test_metrics']['dimensions']
    for dim, metrics in dimensions.items():
        print(f"{dim:25s}: F1 {metrics['f1']*100:5.2f}%  Acc {metrics['accuracy']*100:5.2f}%")
    print(f"{'Average Dimension F1':25s}: {results['test_metrics']['avg_dimension_f1']*100:5.2f}%")

    print("\n" + "="*60)

def main():
    print("="*60)
    print("ENSEMBLE EXPERIMENT")
    print("="*60)

    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    labels_file = data_dir / 'labels.json'
    config_file = project_root / 'config.json'
    output_dir = project_root / 'results' / 'ensemble'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    print("\n1. Loading configuration...")
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    config = SystemConfig.from_dict(config_dict)

    # Make sure we're in ensemble mode
    config.assessment_method = 'ensemble'
    print(f"  Assessment method: {config.assessment_method}")
    print(f"  Ensemble strategy: {config.ensemble_strategy}")
    print(f"  Models: {[m.name if hasattr(m, 'name') else m['name'] for m in config.models]}")

    # Initialize system
    print("\n2. Initializing safety assessment system...")
    system = SafetyAssessmentSystem(config)

    # Replace classifiers with legacy architecture and load trained weights
    print("\n3. Loading trained model checkpoints...")
    for model_name in system.classifiers.keys():
        checkpoint_path = project_root / 'results' / 'single_models' / model_name / 'best_model.pt'
        if checkpoint_path.exists():
            print(f"  Loading {model_name} from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=system.device)

            # Extract model state dict if checkpoint contains metadata
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Detect embedding_dim from checkpoint (in case config is wrong)
            if 'feature_extractor.0.weight' in state_dict:
                embedding_dim = state_dict['feature_extractor.0.weight'].shape[1]
                print(f"    Detected embedding_dim: {embedding_dim}")
            else:
                # Fallback to config
                model_config = next(m for m in config.models if (m.get('name') if isinstance(m, dict) else m.name) == model_name)
                embedding_dim = model_config.get('embedding_dim') if isinstance(model_config, dict) else model_config.embedding_dim

            # Create legacy classifier and load weights
            legacy_classifier = LegacySafetyClassifier(
                embedding_dim=embedding_dim,
                hidden_dim=512,
                num_dimensions=5
            ).to(system.device)
            legacy_classifier.load_state_dict(state_dict)
            legacy_classifier.eval()

            # Replace in system
            system.classifiers[model_name] = legacy_classifier
        else:
            print(f"  WARNING: Checkpoint not found for {model_name}")

    # Load test dataset
    print("\n4. Loading test dataset...")
    image_paths, labels, test_labels = load_test_dataset(data_dir, labels_file)

    # Evaluate ensemble
    print("\n5. Running ensemble evaluation...")
    results = evaluate_ensemble(system, image_paths, labels, test_labels)

    # Print results
    print_results(results)

    # Save results
    print("\n6. Saving results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'ensemble_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {results_file}")

    # Also save as latest
    latest_file = output_dir / 'results.json'
    with open(latest_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {latest_file}")

    print("\n" + "="*60)
    print("ENSEMBLE EXPERIMENT COMPLETE!")
    print("="*60)

    return results

if __name__ == '__main__':
    main()
