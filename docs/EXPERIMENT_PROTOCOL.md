# EXPERIMENT_PROTOCOL.md

**Version**: 1.0
**Last Updated**: 2025-10-01
**Status**: Active Experimental Protocol

본 문서는 SafetyKnob 프로젝트의 실험 프로토콜을 정의합니다. 모든 실험은 이 프로토콜을 따라 수행되어야 하며, 재현성과 비교 가능성을 보장하기 위해 표준화된 절차를 따릅니다.

---

## Table of Contents

1. [실험 환경](#1-실험-환경)
2. [Baseline 모델 정의](#2-baseline-모델-정의)
3. [실험 설계](#3-실험-설계)
4. [평가 프로토콜](#4-평가-프로토콜)
5. [통계적 유의성 검정](#5-통계적-유의성-검정)
6. [실험 체크리스트](#6-실험-체크리스트)

---

## 1. 실험 환경

### 1.1 하드웨어 스펙

**표준 실험 환경**:
```yaml
GPU:
  - Model: NVIDIA A100 (40GB) 또는 동등 이상
  - CUDA Compute Capability: >= 8.0
  - Recommended: Multi-GPU setup for ensemble training

CPU:
  - Model: Intel Xeon or AMD EPYC
  - Cores: >= 16 cores
  - RAM: >= 64GB (128GB recommended for full ensemble)

Storage:
  - Type: NVMe SSD
  - Capacity: >= 500GB for dataset + checkpoints + results
  - IOPS: >= 100K for efficient data loading
```

**최소 요구사항** (Single Model Experiments):
```yaml
GPU: NVIDIA RTX 3090 (24GB) or equivalent
CPU: 8 cores
RAM: 32GB
Storage: 200GB SSD
```

**환경 검증 스크립트**:
```bash
# Run environment verification
python scripts/verify_environment.py

# Expected output:
# ✓ GPU: NVIDIA A100 (40GB) - CUDA 11.8
# ✓ CPU: 32 cores detected
# ✓ RAM: 128GB available
# ✓ Disk: 1.2TB free on /workspace
# ✓ PyTorch: 2.0.1+cu118
# ✓ All dependencies satisfied
```

### 1.2 소프트웨어 버전

**Python Environment**:
```yaml
Python: 3.10.12
CUDA: 11.8
cuDNN: 8.7.0
```

**Core Dependencies** (정확한 버전):
```txt
torch==2.0.1+cu118
torchvision==0.15.2+cu118
transformers==4.35.2
open_clip_torch==2.23.0
timm==0.9.12
pillow==10.1.0
numpy==1.24.3
scikit-learn==1.3.2
pandas==2.1.3
matplotlib==3.8.2
seaborn==0.13.0
fastapi==0.104.1
uvicorn==0.24.0
```

**재현성 보장**:
```bash
# 정확한 패키지 버전으로 환경 구축
pip install -r requirements-exact.txt

# 환경 고정 (새로운 실험 시작 시)
pip freeze > experiments/exp_YYYYMMDD/environment.txt
```

**Random Seed 설정** (모든 실험에서 필수):
```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic operations (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Usage in all experiment scripts
set_seed(42)
```

### 1.3 데이터 준비

**Dataset Download and Preparation**:
```bash
# Step 1: Download from AI Hub
# URL: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71407
# Manual download required (requires AI Hub account)

# Step 2: Extract and organize
cd /workspace/arsim/SafetyKnob
mkdir -p data/raw
unzip Construction_Safety_Images.zip -d data/raw/

# Step 3: Run preprocessing
python scripts/prepare_dataset.py \
  --input data/raw/ \
  --output data/processed/ \
  --train-scenarios SO-35,SO-36,SO-37,SO-38,SO-41,SO-42,SO-43 \
  --val-scenarios SO-44,SO-45,SO-46 \
  --test-scenarios SO-47

# Step 4: Verify data integrity
python scripts/verify_dataset.py --data-dir data/processed/
# Expected: 11,583 images, 5 labels per image, train/val/test split correct
```

**Data Split Verification**:
```python
# scripts/verify_dataset.py output example:
"""
Dataset Split Summary:
- Train: 7,128 images (61.5%) - Scenarios: SO-35 to SO-43
- Val:   2,872 images (24.8%) - Scenarios: SO-44 to SO-46
- Test:  1,583 images (13.7%) - Scenarios: SO-47

Label Distribution (Overall):
- fall_hazard:        0: 6,234 (53.8%), 1: 5,349 (46.2%)
- collision_risk:     0: 7,891 (68.1%), 1: 3,692 (31.9%)
- equipment_hazard:   0: 8,234 (71.1%), 1: 3,349 (28.9%)
- environmental_risk: 0: 9,012 (77.8%), 1: 2,571 (22.2%)
- protective_gear:    0: 4,123 (35.6%), 1: 7,460 (64.4%)

✓ No data leakage detected (scenario-level split)
✓ All images have 5-dimensional labels
✓ No corrupted images found
"""
```

---

## 2. Baseline 모델 정의

모든 실험은 다음 baseline과 비교되어야 합니다. Baseline 성능은 제안 방법의 유효성을 검증하는 기준이 됩니다.

### 2.1 Baseline 1: Random Classifier

**목적**: Lower bound 성능 측정

**구현**:
```python
class RandomBaseline:
    """Random classifier baseline for lower bound."""

    def __init__(self, p_positive=0.5, seed=42):
        """
        Args:
            p_positive: Probability of predicting positive class
            seed: Random seed for reproducibility
        """
        self.p_positive = p_positive
        self.rng = np.random.RandomState(seed)

    def predict(self, images):
        """Predict random labels."""
        n = len(images)
        predictions = {
            'overall_safety': self.rng.rand(n),
            'dimensions': {
                'fall_hazard': self.rng.rand(n),
                'collision_risk': self.rng.rand(n),
                'equipment_hazard': self.rng.rand(n),
                'environmental_risk': self.rng.rand(n),
                'protective_gear': self.rng.rand(n),
            },
            'is_safe': self.rng.rand(n) > 0.5
        }
        return predictions

# Expected Performance (averaged over 10 runs):
# Accuracy: ~50% (± 2%)
# F1-Score: ~50% (± 2%)
# AUC-ROC: ~0.50 (± 0.01)
```

**실행 방법**:
```bash
python experiments/run_baseline_random.py \
  --data-dir data/processed/test/ \
  --num-runs 10 \
  --output results/baselines/random_baseline.json
```

### 2.2 Baseline 2: ResNet-50 Fine-tuned

**목적**: 전통적인 supervised learning 성능 측정

**구현**:
```python
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Baseline(nn.Module):
    """ResNet-50 fine-tuned on construction safety dataset."""

    def __init__(self, num_dimensions=5, pretrained=True):
        super().__init__()

        # Load ImageNet pretrained ResNet-50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.backbone = resnet50(weights=weights)
        else:
            self.backbone = resnet50(weights=None)

        # Replace final layer with safety prediction heads
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove original fc

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Overall safety head
        self.safety_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Per-dimension heads
        self.dimension_heads = nn.ModuleDict({
            'fall_hazard': self._make_head(512),
            'collision_risk': self._make_head(512),
            'equipment_hazard': self._make_head(512),
            'environmental_risk': self._make_head(512),
            'protective_gear': self._make_head(512),
        })

    def _make_head(self, in_features):
        return nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)  # (B, 2048)
        features = self.feature_extractor(features)  # (B, 512)

        # Predict overall safety
        overall = self.safety_head(features)  # (B, 1)

        # Predict per-dimension scores
        dimensions = {
            name: head(features)
            for name, head in self.dimension_heads.items()
        }

        return {
            'overall_safety': overall.squeeze(1),
            'dimensions': dimensions
        }
```

**훈련 설정**:
```yaml
Training Config:
  epochs: 50
  batch_size: 32
  optimizer: AdamW
  learning_rate: 1e-4
  weight_decay: 0.01
  scheduler: CosineAnnealingLR
  early_stopping:
    patience: 10
    metric: val_f1_macro

Loss Function:
  overall_loss_weight: 1.0
  dimension_loss_weight: 0.5
  loss_type: BCELoss

Augmentation:
  RandomHorizontalFlip: p=0.5
  RandomRotation: degrees=15
  ColorJitter: brightness=0.2, contrast=0.2
  RandomResizedCrop: size=224, scale=(0.8, 1.0)
```

**실행 방법**:
```bash
# Train ResNet-50 baseline
python experiments/train_resnet50_baseline.py \
  --config configs/baseline_resnet50.yaml \
  --data-dir data/processed/ \
  --output checkpoints/baselines/resnet50/ \
  --seed 42

# Evaluate on test set
python experiments/evaluate_baseline.py \
  --model resnet50 \
  --checkpoint checkpoints/baselines/resnet50/best_model.pt \
  --data-dir data/processed/test/ \
  --output results/baselines/resnet50_results.json
```

**예상 성능** (⚠️ Preliminary - Under Validation):
```yaml
ResNet-50 Baseline Expected Results:
  Overall Safety:
    Accuracy: 78-82%
    F1-Score: 0.76-0.80
    AUC-ROC: 0.84-0.88

  Per-Dimension (Average):
    fall_hazard: F1 0.74-0.78
    collision_risk: F1 0.68-0.72
    equipment_hazard: F1 0.71-0.75
    environmental_risk: F1 0.65-0.69
    protective_gear: F1 0.79-0.83

  Training Time: ~4 hours (A100 GPU)
  Inference: ~15ms/image (batch=32)
```

### 2.3 Baseline 3: CLIP Zero-shot

**목적**: Pre-trained vision-language model의 zero-shot 성능 측정

**구현**:
```python
import open_clip

class CLIPZeroShotBaseline:
    """CLIP zero-shot classifier using text prompts."""

    def __init__(self, model_name='ViT-L-14', pretrained='openai'):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        # Safety prompts for zero-shot classification
        self.prompts = {
            'overall_safety': [
                "a safe construction site with no hazards",
                "a dangerous construction site with safety hazards"
            ],
            'fall_hazard': [
                "workers at safe height with proper fall protection",
                "workers at dangerous height without fall protection"
            ],
            'collision_risk': [
                "construction site with clear paths and no collision risk",
                "construction site with collision hazards between workers and equipment"
            ],
            'equipment_hazard': [
                "construction equipment properly maintained and safely operated",
                "dangerous construction equipment or improper operation"
            ],
            'environmental_risk': [
                "safe construction environment with good conditions",
                "hazardous construction environment with dangerous conditions"
            ],
            'protective_gear': [
                "workers wearing complete safety equipment and protective gear",
                "workers without proper safety equipment or protective gear"
            ]
        }

    @torch.no_grad()
    def predict(self, images):
        """
        Zero-shot prediction using CLIP text-image similarity.

        Args:
            images: List of PIL Images or torch.Tensor

        Returns:
            dict: Predictions with overall_safety and dimensions
        """
        # Preprocess images
        if isinstance(images[0], Image.Image):
            image_inputs = torch.stack([self.preprocess(img) for img in images])
        else:
            image_inputs = images

        # Encode images
        image_features = self.model.encode_image(image_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        predictions = {'dimensions': {}}

        # Predict for each dimension
        for dimension, prompts in self.prompts.items():
            # Encode text prompts
            text_tokens = self.tokenizer(prompts)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate similarity
            similarity = (image_features @ text_features.T)  # (N, 2)

            # Convert to probability (softmax over safe/dangerous)
            probs = similarity.softmax(dim=-1)

            # Score = P(dangerous) for risk dimensions
            if dimension == 'overall_safety':
                predictions['overall_safety'] = probs[:, 0].cpu().numpy()  # P(safe)
            else:
                predictions['dimensions'][dimension] = probs[:, 1].cpu().numpy()  # P(danger)

        # Binary decision (threshold=0.5)
        predictions['is_safe'] = predictions['overall_safety'] > 0.5

        return predictions
```

**실행 방법**:
```bash
# Run CLIP zero-shot evaluation
python experiments/run_clip_zeroshot.py \
  --model ViT-L-14 \
  --pretrained openai \
  --data-dir data/processed/test/ \
  --output results/baselines/clip_zeroshot.json \
  --batch-size 64
```

**예상 성능** (⚠️ Preliminary - Based on Initial Tests):
```yaml
CLIP Zero-shot Expected Results:
  Overall Safety:
    Accuracy: 68-72%
    F1-Score: 0.65-0.69
    AUC-ROC: 0.74-0.78

  Per-Dimension:
    fall_hazard: F1 0.62-0.66 (sensitive to "height" keywords)
    collision_risk: F1 0.58-0.62 (struggles with implicit risks)
    equipment_hazard: F1 0.64-0.68 (good at equipment recognition)
    environmental_risk: F1 0.55-0.59 (environmental context is hard)
    protective_gear: F1 0.71-0.75 (best performance - visible features)

  Inference: ~8ms/image (batch=64, A100 GPU)

Note: Zero-shot performance heavily depends on prompt engineering.
      These numbers are for the prompts defined above.
```

### 2.4 Baseline Summary

**성능 비교표** (⚠️ Preliminary Estimates):

| Baseline | Overall Acc | Overall F1 | Avg Dim F1 | Training Time | Inference Speed |
|----------|-------------|------------|------------|---------------|-----------------|
| Random | ~50% | ~0.50 | ~0.50 | N/A | <1ms/img |
| ResNet-50 | 78-82% | 0.76-0.80 | 0.71-0.75 | ~4 hours | ~15ms/img |
| CLIP Zero-shot | 68-72% | 0.65-0.69 | 0.62-0.68 | N/A | ~8ms/img |
| **SafetyKnob (Target)** | **>85%** | **>0.83** | **>0.78** | ~2 hours | ~12ms/img |

**Baseline 실행 우선순위**:
1. **Random**: 모든 실험에서 필수 (sanity check)
2. **CLIP Zero-shot**: 빠른 실행, pre-trained model 효과 검증
3. **ResNet-50**: 전통적 supervised learning 비교

---

## 3. 실험 설계

### 3.1 Experiment 1: Single Model Performance

**가설**: H1 (Linear Separability) 검증

**목적**: 각 pre-trained 모델의 embedding space에서 안전/위험이 선형 분리 가능한지 검증

**실험 절차**:

**Step 1: Extract Embeddings**
```bash
# Extract embeddings for all models
for model in siglip clip dinov2 evaclip; do
  python scripts/extract_embeddings.py \
    --model $model \
    --data-dir data/processed/ \
    --output embeddings/${model}/ \
    --batch-size 64
done

# Expected output structure:
# embeddings/
#   siglip/
#     train_embeddings.npy  # (7128, 1152)
#     val_embeddings.npy    # (2872, 1152)
#     test_embeddings.npy   # (1583, 1152)
#   clip/
#     train_embeddings.npy  # (7128, 768)
#   ...
```

**Step 2: Train Linear Probes**
```bash
# Train linear classifier on frozen embeddings
python experiments/exp1_single_model.py \
  --embeddings-dir embeddings/siglip/ \
  --labels data/processed/labels.json \
  --model-name siglip \
  --output results/exp1/siglip/ \
  --seed 42

# Repeat for all models
for model in clip dinov2 evaclip; do
  python experiments/exp1_single_model.py \
    --embeddings-dir embeddings/${model}/ \
    --labels data/processed/labels.json \
    --model-name $model \
    --output results/exp1/${model}/ \
    --seed 42
done
```

**Linear Probe Configuration**:
```python
# Simple linear classifier for linear separability test
class LinearProbe(nn.Module):
    def __init__(self, embedding_dim, num_dimensions=5):
        super().__init__()
        self.overall_head = nn.Linear(embedding_dim, 1)
        self.dimension_heads = nn.ModuleDict({
            'fall_hazard': nn.Linear(embedding_dim, 1),
            'collision_risk': nn.Linear(embedding_dim, 1),
            'equipment_hazard': nn.Linear(embedding_dim, 1),
            'environmental_risk': nn.Linear(embedding_dim, 1),
            'protective_gear': nn.Linear(embedding_dim, 1),
        })

    def forward(self, x):
        overall = torch.sigmoid(self.overall_head(x))
        dimensions = {
            name: torch.sigmoid(head(x))
            for name, head in self.dimension_heads.items()
        }
        return {'overall_safety': overall.squeeze(), 'dimensions': dimensions}

# Training config
config = {
    'epochs': 50,
    'batch_size': 128,
    'optimizer': 'AdamW',
    'lr': 1e-3,
    'weight_decay': 0.01,
    'early_stopping_patience': 10
}
```

**평가 메트릭**:
```python
# Compute metrics for H1 validation
metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'f1_macro': f1_score(y_true, y_pred, average='macro'),
    'auc_roc': roc_auc_score(y_true, y_scores),
    'precision': precision_score(y_true, y_pred),
    'recall': recall_score(y_true, y_pred),

    # Per-dimension metrics
    'dimension_metrics': {
        dim: {
            'f1': f1_score(y_true_dim, y_pred_dim),
            'auc': roc_auc_score(y_true_dim, y_scores_dim)
        }
        for dim in dimension_names
    },

    # Linear separability metric (margin analysis)
    'separability_margin': compute_margin(embeddings, labels)
}
```

**예상 결과** (⚠️ Under Investigation):
```yaml
SigLIP:
  Overall: Acc 93.2%, F1 0.921, AUC 0.967
  Avg Dimension F1: 0.887
  Separability Margin: 1.42

CLIP (ViT-L/14):
  Overall: Acc 89.1%, F1 0.883, AUC 0.948
  Avg Dimension F1: 0.841
  Separability Margin: 1.18

DINOv2 (ViT-L/14):
  Overall: Acc 86.7%, F1 0.854, AUC 0.931
  Avg Dimension F1: 0.812
  Separability Margin: 1.05

EVA-CLIP:
  Overall: Acc 91.4%, F1 0.907, AUC 0.956
  Avg Dimension F1: 0.868
  Separability Margin: 1.31

Hypothesis H1 Validation:
  Status: Under Investigation
  Expected: All models achieve >85% accuracy with linear probe
  Indicates: Safety features are linearly separable in embedding space
```

**시각화**:
```bash
# Generate t-SNE and PCA plots for embedding space
python scripts/visualize_embeddings.py \
  --embeddings embeddings/siglip/test_embeddings.npy \
  --labels data/processed/test_labels.json \
  --output figures/exp1/siglip_tsne.png \
  --method tsne

# Generate separability analysis plots
python scripts/analyze_separability.py \
  --results-dir results/exp1/ \
  --output figures/exp1/separability_comparison.png
```

### 3.2 Experiment 2: Ensemble Strategies

**가설**: H2 (Ensemble Robustness) 검증

**목적**: Multi-model ensemble이 단일 모델보다 성능과 robustness에서 우수한지 검증

**실험 절차**:

**Step 1: Train Individual Models**
```bash
# Train neural classifiers for each model
for model in siglip clip dinov2 evaclip; do
  python main.py train \
    --config configs/single_model_${model}.json \
    --data-dir data/processed/ \
    --output checkpoints/single/${model}/ \
    --seed 42
done
```

**Step 2: Test Ensemble Strategies**
```bash
# Strategy 1: Weighted Vote
python experiments/exp2_ensemble.py \
  --strategy weighted_vote \
  --models siglip clip dinov2 evaclip \
  --checkpoints checkpoints/single/ \
  --data-dir data/processed/test/ \
  --output results/exp2/weighted_vote/ \
  --seed 42

# Strategy 2: Stacking (Meta-learner)
python experiments/exp2_ensemble.py \
  --strategy stacking \
  --models siglip clip dinov2 evaclip \
  --checkpoints checkpoints/single/ \
  --data-dir data/processed/ \
  --output results/exp2/stacking/ \
  --meta-learner logistic_regression \
  --seed 42

# Strategy 3: Average (uniform weights)
python experiments/exp2_ensemble.py \
  --strategy average \
  --models siglip clip dinov2 evaclip \
  --checkpoints checkpoints/single/ \
  --data-dir data/processed/test/ \
  --output results/exp2/average/ \
  --seed 42
```

**Ensemble 설정**:
```yaml
Weighted Vote:
  weight_optimization: validation_f1
  initial_weights: [0.25, 0.25, 0.25, 0.25]
  optimization_metric: f1_macro
  search_method: grid_search
  weight_range: [0.0, 1.0]
  step: 0.05

Stacking:
  meta_learner: LogisticRegression
  meta_features: [model_outputs, confidence_scores]
  cv_folds: 5
  meta_learner_params:
    C: 1.0
    max_iter: 1000

Average:
  weights: [0.25, 0.25, 0.25, 0.25]
  aggregation: mean
```

**Distribution Shift 테스트**:
```bash
# Test ensemble robustness under distribution shift
python experiments/test_distribution_shift.py \
  --ensemble-checkpoint checkpoints/ensemble/weighted_vote/ \
  --test-scenarios SO-47 \
  --shift-types weather,lighting,camera_angle \
  --output results/exp2/distribution_shift/ \
  --seed 42
```

**예상 결과** (⚠️ Under Investigation):
```yaml
In-Distribution (Test Set SO-47):
  Best Single Model (SigLIP): F1 0.921, AUC 0.967
  Average Ensemble: F1 0.928, AUC 0.971
  Weighted Vote: F1 0.936, AUC 0.974
  Stacking: F1 0.941, AUC 0.976

  Improvement: +1.5% to +2.0% F1 over best single model

Out-of-Distribution (Simulated Shift):
  Weather Shift (Rain/Fog):
    Single Model Avg: -8.2% F1 drop
    Ensemble Avg: -4.1% F1 drop
    Robustness Gain: 2x

  Lighting Shift (Low-light):
    Single Model Avg: -6.7% F1 drop
    Ensemble Avg: -3.2% F1 drop
    Robustness Gain: 2.1x

  Camera Angle Shift:
    Single Model Avg: -5.1% F1 drop
    Ensemble Avg: -2.4% F1 drop
    Robustness Gain: 2.1x

Hypothesis H2 Validation:
  Status: Under Investigation
  Expected: Ensemble outperforms single models by >1.5% F1
  Expected: Distribution shift robustness 2x better
```

**Agreement Analysis**:
```python
# Measure ensemble agreement/disagreement
def compute_agreement_metrics(predictions_dict):
    """
    Args:
        predictions_dict: {model_name: predictions_array}
    Returns:
        agreement_metrics: dict with ensemble consensus stats
    """
    metrics = {
        'full_agreement': np.mean(all_agree),  # All models agree
        'majority_agreement': np.mean(majority_agree),  # >50% agree
        'entropy': np.mean(prediction_entropy),  # Uncertainty
        'variance': np.mean(prediction_variance)  # Spread
    }
    return metrics

# Expected agreement patterns:
# - High-confidence correct: >95% full agreement
# - Low-confidence: <60% agreement, high entropy
# - Misclassified cases: 40-60% agreement (review candidates)
```

### 3.3 Experiment 3: Ablation Study

**목적**: 시스템 각 컴포넌트의 기여도 분석

**Ablation 항목**:

1. **Model Ablation**: 각 모델 제거 시 성능 변화
2. **Dimension Ablation**: 각 dimension 제거 시 overall safety 성능 변화
3. **Architecture Ablation**: Neural classifier vs Linear probe 비교
4. **Training Data Ablation**: 데이터 양에 따른 성능 변화

**실행 방법**:
```bash
# Ablation 1: Remove each model from ensemble
python experiments/exp3_ablation.py \
  --ablation-type model \
  --ensemble-config configs/ensemble_config.json \
  --data-dir data/processed/ \
  --output results/exp3/model_ablation/ \
  --seed 42

# Ablation 2: Remove each dimension
python experiments/exp3_ablation.py \
  --ablation-type dimension \
  --full-model-checkpoint checkpoints/ensemble/best_model.pt \
  --data-dir data/processed/ \
  --output results/exp3/dimension_ablation/ \
  --seed 42

# Ablation 3: Architecture comparison
python experiments/exp3_ablation.py \
  --ablation-type architecture \
  --architectures linear_probe,mlp_1layer,mlp_2layer,mlp_3layer \
  --data-dir data/processed/ \
  --output results/exp3/architecture_ablation/ \
  --seed 42

# Ablation 4: Training data size
python experiments/exp3_ablation.py \
  --ablation-type data_size \
  --data-fractions 0.1,0.25,0.5,0.75,1.0 \
  --data-dir data/processed/ \
  --output results/exp3/data_ablation/ \
  --seed 42
```

**예상 결과** (⚠️ Under Investigation):
```yaml
Model Ablation (Remove one model from 4-model ensemble):
  Full Ensemble (4 models): F1 0.936
  - SigLIP removed: F1 0.912 (-2.4%, largest drop)
  - CLIP removed: F1 0.927 (-0.9%)
  - DINOv2 removed: F1 0.931 (-0.5%)
  - EVA-CLIP removed: F1 0.920 (-1.6%)

  Finding: SigLIP contributes most, DINOv2 least

Dimension Ablation (Remove one dimension from overall prediction):
  Full (5 dimensions): Overall F1 0.936
  - fall_hazard removed: 0.921 (-1.5%)
  - collision_risk removed: 0.929 (-0.7%)
  - equipment_hazard removed: 0.932 (-0.4%)
  - environmental_risk removed: 0.934 (-0.2%)
  - protective_gear removed: 0.918 (-1.8%, largest drop)

  Finding: protective_gear most informative for overall safety

Architecture Ablation:
  Linear Probe: F1 0.887, Params 1.2K
  MLP (1 layer, 256 hidden): F1 0.921, Params 320K
  MLP (2 layers, 512→256): F1 0.936, Params 850K
  MLP (3 layers, 1024→512→256): F1 0.938, Params 2.1M

  Finding: 2-layer MLP optimal (diminishing returns at 3 layers)

Data Size Ablation:
  10% data (712 samples): F1 0.742
  25% data (1,782 samples): F1 0.834
  50% data (3,564 samples): F1 0.897
  75% data (5,346 samples): F1 0.921
  100% data (7,128 samples): F1 0.936

  Finding: Performance scales log-linearly with data size
```

### 3.4 Experiment 4: Threshold Analysis

**목적**: 최적 threshold 선택 및 confidence calibration 검증

**실험 절차**:

**Step 1: ROC/PR Curve Analysis**
```bash
# Generate ROC and PR curves for threshold selection
python experiments/exp4_threshold_analysis.py \
  --model-checkpoint checkpoints/ensemble/best_model.pt \
  --data-dir data/processed/val/ \
  --output results/exp4/ \
  --metrics roc,pr,calibration \
  --seed 42
```

**Step 2: Threshold Optimization**
```python
# Find optimal threshold for each dimension
def optimize_threshold(y_true, y_scores, metric='f1'):
    """
    Find threshold that maximizes the given metric.

    Args:
        y_true: Ground truth labels (binary)
        y_scores: Predicted scores (continuous [0,1])
        metric: 'f1', 'balanced_accuracy', 'youden_j'

    Returns:
        best_threshold: Optimal threshold value
        best_score: Best metric score
    """
    thresholds = np.linspace(0, 1, 1000)
    scores = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == 'youden_j':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            score = sensitivity + specificity - 1
        scores.append(score)

    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]

# Expected optimal thresholds (validation set):
optimal_thresholds = {
    'overall_safety': 0.52,
    'fall_hazard': 0.48,
    'collision_risk': 0.55,
    'equipment_hazard': 0.51,
    'environmental_risk': 0.58,
    'protective_gear': 0.45
}
```

**Step 3: Calibration Analysis**
```python
# Measure calibration quality
from sklearn.calibration import calibration_curve

def analyze_calibration(y_true, y_scores, n_bins=10):
    """
    Compute calibration metrics.

    Returns:
        ece: Expected Calibration Error
        mce: Maximum Calibration Error
        brier: Brier score
    """
    # Expected Calibration Error
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=n_bins)
    ece = np.mean(np.abs(prob_true - prob_pred))

    # Maximum Calibration Error
    mce = np.max(np.abs(prob_true - prob_pred))

    # Brier Score
    brier = np.mean((y_scores - y_true) ** 2)

    return {'ECE': ece, 'MCE': mce, 'Brier': brier}

# Expected calibration (before temperature scaling):
# ECE: 0.08-0.12 (moderate calibration error)
# MCE: 0.15-0.20
# Brier: 0.10-0.14
```

**Step 4: Temperature Scaling**
```python
# Apply temperature scaling for calibration
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels):
        """Fit temperature on validation set."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                logits / self.temperature, labels
            )
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return self.temperature.item()

# Expected temperature: T ≈ 1.2-1.5
# After scaling: ECE < 0.05, MCE < 0.10
```

**예상 결과** (⚠️ Under Investigation):
```yaml
Threshold Analysis Results:
  Overall Safety:
    Optimal Threshold: 0.52 (F1-maximizing)
    At threshold 0.52: Precision 0.94, Recall 0.93, F1 0.936
    At threshold 0.50: Precision 0.92, Recall 0.95, F1 0.934

  Per-Dimension Optimal Thresholds:
    fall_hazard: 0.48 (balanced, high recall priority)
    collision_risk: 0.55 (conservative, high precision)
    equipment_hazard: 0.51 (balanced)
    environmental_risk: 0.58 (conservative)
    protective_gear: 0.45 (liberal, prioritize detection)

Calibration Results:
  Before Temperature Scaling:
    ECE: 0.102 (moderate calibration error)
    MCE: 0.183
    Brier Score: 0.118

  After Temperature Scaling (T=1.38):
    ECE: 0.042 (well-calibrated)
    MCE: 0.091
    Brier Score: 0.095

  Improvement: ECE reduced by 59%, Brier improved by 19%
```

---

## 4. 평가 프로토콜

### 4.1 Standard Metrics

모든 실험은 다음 메트릭을 보고해야 합니다:

**Classification Metrics**:
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, classification_report
)

def compute_standard_metrics(y_true, y_pred, y_scores):
    """
    Compute all standard classification metrics.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        y_scores: Predicted probabilities/scores

    Returns:
        dict: All metrics
    """
    metrics = {
        # Threshold-dependent metrics (at 0.5 or optimal threshold)
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),

        # Threshold-independent metrics
        'auc_roc': roc_auc_score(y_true, y_scores),
        'auc_pr': average_precision_score(y_true, y_scores),

        # Confusion matrix components
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),

        # Detailed classification report
        'classification_report': classification_report(
            y_true, y_pred, output_dict=True
        )
    }

    # Add TN, FP, FN, TP explicitly
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    })

    return metrics
```

**Per-Dimension Metrics**:
```python
def compute_dimension_metrics(predictions, ground_truth):
    """Compute metrics for each safety dimension."""
    dimension_names = ['fall_hazard', 'collision_risk', 'equipment_hazard',
                       'environmental_risk', 'protective_gear']

    results = {}
    for dim in dimension_names:
        y_true = ground_truth[dim]
        y_scores = predictions['dimensions'][dim]
        y_pred = (y_scores >= 0.5).astype(int)

        results[dim] = compute_standard_metrics(y_true, y_pred, y_scores)

    # Compute average metrics across dimensions
    results['dimension_average'] = {
        'f1': np.mean([results[dim]['f1'] for dim in dimension_names]),
        'auc_roc': np.mean([results[dim]['auc_roc'] for dim in dimension_names]),
        'precision': np.mean([results[dim]['precision'] for dim in dimension_names]),
        'recall': np.mean([results[dim]['recall'] for dim in dimension_names])
    }

    return results
```

### 4.2 실험 보고 형식

모든 실험 결과는 다음 형식으로 저장되어야 합니다:

**results/exp{N}/{experiment_name}/results.json**:
```json
{
  "experiment_id": "exp1_siglip_linear_probe",
  "timestamp": "2025-10-01T14:32:10Z",
  "config": {
    "model": "siglip",
    "architecture": "linear_probe",
    "seed": 42,
    "hyperparameters": {
      "epochs": 50,
      "batch_size": 128,
      "learning_rate": 0.001
    }
  },
  "environment": {
    "gpu": "NVIDIA A100 40GB",
    "cuda_version": "11.8",
    "pytorch_version": "2.0.1"
  },
  "data": {
    "train_samples": 7128,
    "val_samples": 2872,
    "test_samples": 1583
  },
  "results": {
    "overall_safety": {
      "accuracy": 0.932,
      "f1": 0.921,
      "auc_roc": 0.967,
      "precision": 0.928,
      "recall": 0.914
    },
    "dimensions": {
      "fall_hazard": {"f1": 0.891, "auc_roc": 0.954},
      "collision_risk": {"f1": 0.867, "auc_roc": 0.931},
      "equipment_hazard": {"f1": 0.879, "auc_roc": 0.942},
      "environmental_risk": {"f1": 0.853, "auc_roc": 0.918},
      "protective_gear": {"f1": 0.945, "auc_roc": 0.981}
    },
    "dimension_average": {
      "f1": 0.887,
      "auc_roc": 0.945
    }
  },
  "training_metrics": {
    "total_time_seconds": 1247,
    "best_epoch": 38,
    "final_train_loss": 0.142,
    "final_val_loss": 0.198
  }
}
```

### 4.3 시각화 요구사항

모든 실험은 다음 시각화를 생성해야 합니다:

1. **ROC Curves** (`figures/exp{N}/roc_curves.png`)
2. **PR Curves** (`figures/exp{N}/pr_curves.png`)
3. **Confusion Matrix** (`figures/exp{N}/confusion_matrix.png`)
4. **Training Curves** (`figures/exp{N}/training_curves.png`)
5. **Per-Dimension Performance** (`figures/exp{N}/dimension_performance.png`)

**생성 스크립트**:
```bash
python scripts/generate_experiment_figures.py \
  --results-file results/exp1/siglip/results.json \
  --output-dir figures/exp1/siglip/
```

---

## 5. 통계적 유의성 검정

### 5.1 다중 실행 (Multiple Runs)

모든 주요 실험은 **최소 5회 반복** (서로 다른 seed)해야 하며, 평균과 표준편차를 보고해야 합니다.

```bash
# Run experiment with multiple seeds
for seed in 42 123 456 789 1024; do
  python experiments/exp1_single_model.py \
    --model siglip \
    --data-dir data/processed/ \
    --output results/exp1/siglip/seed_${seed}/ \
    --seed $seed
done

# Aggregate results
python scripts/aggregate_multi_run_results.py \
  --results-dir results/exp1/siglip/ \
  --output results/exp1/siglip/aggregated_results.json
```

**보고 형식**:
```json
{
  "experiment": "exp1_siglip_linear_probe",
  "num_runs": 5,
  "seeds": [42, 123, 456, 789, 1024],
  "aggregated_results": {
    "overall_f1": {
      "mean": 0.921,
      "std": 0.007,
      "min": 0.912,
      "max": 0.929,
      "median": 0.922,
      "confidence_interval_95": [0.913, 0.929]
    },
    "overall_auc": {
      "mean": 0.967,
      "std": 0.004,
      "confidence_interval_95": [0.961, 0.973]
    }
  }
}
```

### 5.2 Paired t-test

모델 간 성능 비교 시 **paired t-test** 수행:

```python
from scipy.stats import ttest_rel

def compare_models_significance(results_a, results_b, metric='f1'):
    """
    Test if model A is significantly better than model B.

    Args:
        results_a: List of metric values from model A (multiple runs)
        results_b: List of metric values from model B (multiple runs)
        metric: Metric name for reporting

    Returns:
        dict: Statistical test results
    """
    # Paired t-test (same seeds for both models)
    t_stat, p_value = ttest_rel(results_a, results_b)

    # Effect size (Cohen's d)
    mean_diff = np.mean(results_a) - np.mean(results_b)
    pooled_std = np.sqrt((np.var(results_a) + np.var(results_b)) / 2)
    cohens_d = mean_diff / pooled_std

    return {
        'metric': metric,
        'model_a_mean': np.mean(results_a),
        'model_b_mean': np.mean(results_b),
        'mean_difference': mean_diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant_at_0.05': p_value < 0.05,
        'significant_at_0.01': p_value < 0.01
    }

# Example usage:
siglip_f1 = [0.921, 0.918, 0.924, 0.919, 0.922]
clip_f1 = [0.883, 0.879, 0.887, 0.881, 0.885]

result = compare_models_significance(siglip_f1, clip_f1, metric='F1')
# Expected: p < 0.01, Cohen's d ≈ 5.2 (very large effect)
```

### 5.3 Bootstrap Confidence Intervals

AUC 등 복잡한 메트릭에 대해서는 **bootstrap** 사용:

```python
from sklearn.utils import resample

def bootstrap_confidence_interval(y_true, y_scores, metric_func, n_iterations=1000, alpha=0.05):
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: Ground truth labels
        y_scores: Predicted scores
        metric_func: Function that computes metric (e.g., roc_auc_score)
        n_iterations: Number of bootstrap samples
        alpha: Significance level (0.05 for 95% CI)

    Returns:
        dict: Mean, CI lower, CI upper
    """
    scores = []
    n_samples = len(y_true)

    for _ in range(n_iterations):
        # Resample with replacement
        indices = resample(range(n_samples), n_samples=n_samples)
        y_true_boot = y_true[indices]
        y_scores_boot = y_scores[indices]

        # Compute metric on bootstrap sample
        score = metric_func(y_true_boot, y_scores_boot)
        scores.append(score)

    # Compute percentiles
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))

    return {
        'mean': np.mean(scores),
        'ci_lower': lower,
        'ci_upper': upper,
        'alpha': alpha
    }

# Example:
ci = bootstrap_confidence_interval(
    y_true, y_scores,
    metric_func=roc_auc_score,
    n_iterations=1000
)
# Output: {'mean': 0.967, 'ci_lower': 0.961, 'ci_upper': 0.973, 'alpha': 0.05}
```

---

## 6. 실험 체크리스트

모든 실험 수행 시 다음 체크리스트를 확인하세요:

### 6.1 실험 시작 전

- [ ] 실험 목적과 가설이 명확하게 정의됨
- [ ] 데이터셋 준비 완료 및 무결성 검증 (`verify_dataset.py` 통과)
- [ ] Random seed 설정됨 (기본값: 42)
- [ ] 환경 검증 완료 (`verify_environment.py` 통과)
- [ ] Baseline 결과 확보 (비교 대상 준비)
- [ ] 실험 설정이 `config.json` 또는 별도 config 파일에 저장됨
- [ ] 결과 저장 경로 확인 (`results/exp{N}/` 생성됨)

### 6.2 실험 실행 중

- [ ] 훈련 progress 모니터링 (loss, metrics)
- [ ] Early stopping 작동 확인 (overfitting 방지)
- [ ] GPU 메모리 사용량 모니터링
- [ ] 중간 checkpoint 저장 확인

### 6.3 실험 완료 후

- [ ] 테스트 셋 결과 생성 (`results.json` 저장됨)
- [ ] 모든 필수 메트릭 계산됨 (accuracy, F1, AUC, precision, recall)
- [ ] Per-dimension 메트릭 계산됨
- [ ] 시각화 생성됨 (ROC, PR, confusion matrix 등)
- [ ] 다중 실행 수행됨 (최소 5회, 서로 다른 seed)
- [ ] 통계적 유의성 검정 수행됨 (baseline 대비)
- [ ] 실험 환경 정보 저장됨 (`environment.txt`)
- [ ] Checkpoint와 config 백업됨
- [ ] 결과가 문서화됨 (README 또는 experiment report에 기록)

### 6.4 재현성 체크리스트

- [ ] 동일한 seed로 재실행 시 동일한 결과 생성 확인
- [ ] `requirements-exact.txt`로 환경 재구축 가능 확인
- [ ] 데이터 전처리 스크립트 포함 (`prepare_dataset.py`)
- [ ] 모든 하이퍼파라미터가 config 파일에 명시됨
- [ ] 실험 실행 명령어가 문서화됨

---

## 부록 A: 실험 스크립트 예시

### A.1 실험 실행 마스터 스크립트

```bash
#!/bin/bash
# scripts/run_all_experiments.sh

set -e  # Exit on error

echo "=== SafetyKnob Experiment Pipeline ==="
echo "Starting at: $(date)"

# Step 1: Environment verification
echo "[Step 1/6] Verifying environment..."
python scripts/verify_environment.py || exit 1

# Step 2: Data preparation
echo "[Step 2/6] Preparing dataset..."
python scripts/prepare_dataset.py \
  --input data/raw/ \
  --output data/processed/ \
  --train-scenarios SO-35,SO-36,SO-37,SO-38,SO-41,SO-42,SO-43 \
  --val-scenarios SO-44,SO-45,SO-46 \
  --test-scenarios SO-47

python scripts/verify_dataset.py --data-dir data/processed/

# Step 3: Baseline experiments
echo "[Step 3/6] Running baseline experiments..."
python experiments/run_baseline_random.py --output results/baselines/random.json
python experiments/run_clip_zeroshot.py --output results/baselines/clip_zeroshot.json
python experiments/train_resnet50_baseline.py --output checkpoints/baselines/resnet50/

# Step 4: Experiment 1 (Single Model Performance)
echo "[Step 4/6] Experiment 1: Single Model Performance..."
for model in siglip clip dinov2 evaclip; do
  for seed in 42 123 456 789 1024; do
    python experiments/exp1_single_model.py \
      --model $model \
      --data-dir data/processed/ \
      --output results/exp1/${model}/seed_${seed}/ \
      --seed $seed
  done
done

# Step 5: Experiment 2 (Ensemble Strategies)
echo "[Step 5/6] Experiment 2: Ensemble Strategies..."
python experiments/exp2_ensemble.py \
  --strategy weighted_vote \
  --output results/exp2/weighted_vote/

python experiments/exp2_ensemble.py \
  --strategy stacking \
  --output results/exp2/stacking/

# Step 6: Generate summary report
echo "[Step 6/6] Generating summary report..."
python scripts/generate_experiment_summary.py \
  --results-dir results/ \
  --output reports/experiment_summary_$(date +%Y%m%d).md

echo "=== All experiments completed at: $(date) ==="
```

### A.2 결과 집계 스크립트

```python
# scripts/generate_experiment_summary.py

import json
import glob
from pathlib import Path
import numpy as np

def aggregate_experiment_results(results_dir):
    """Aggregate all experiment results into a summary."""

    summary = {
        'baselines': {},
        'single_models': {},
        'ensembles': {},
        'comparisons': []
    }

    # Collect baseline results
    baseline_files = glob.glob(f"{results_dir}/baselines/*.json")
    for file in baseline_files:
        with open(file) as f:
            data = json.load(f)
            model_name = Path(file).stem
            summary['baselines'][model_name] = {
                'f1': data['results']['overall_safety']['f1'],
                'auc': data['results']['overall_safety']['auc_roc']
            }

    # Collect single model results (multi-run aggregation)
    for model in ['siglip', 'clip', 'dinov2', 'evaclip']:
        model_results = []
        result_files = glob.glob(f"{results_dir}/exp1/{model}/seed_*/results.json")
        for file in result_files:
            with open(file) as f:
                data = json.load(f)
                model_results.append(data['results']['overall_safety']['f1'])

        if model_results:
            summary['single_models'][model] = {
                'f1_mean': np.mean(model_results),
                'f1_std': np.std(model_results),
                'f1_ci_95': [
                    np.mean(model_results) - 1.96 * np.std(model_results),
                    np.mean(model_results) + 1.96 * np.std(model_results)
                ]
            }

    # Collect ensemble results
    ensemble_files = glob.glob(f"{results_dir}/exp2/*/results.json")
    for file in ensemble_files:
        with open(file) as f:
            data = json.load(f)
            strategy = Path(file).parent.name
            summary['ensembles'][strategy] = {
                'f1': data['results']['overall_safety']['f1'],
                'auc': data['results']['overall_safety']['auc_roc']
            }

    return summary

def generate_markdown_report(summary, output_file):
    """Generate markdown summary report."""

    report = f"""# SafetyKnob Experiment Summary

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Baseline Performance

| Model | F1-Score | AUC-ROC |
|-------|----------|---------|
"""

    for model, metrics in summary['baselines'].items():
        report += f"| {model} | {metrics['f1']:.3f} | {metrics['auc']:.3f} |\n"

    report += """
## Single Model Performance (Multi-run Average)

| Model | F1-Score (Mean ± Std) | 95% CI |
|-------|----------------------|--------|
"""

    for model, metrics in summary['single_models'].items():
        report += f"| {model} | {metrics['f1_mean']:.3f} ± {metrics['f1_std']:.3f} | [{metrics['f1_ci_95'][0]:.3f}, {metrics['f1_ci_95'][1]:.3f}] |\n"

    report += """
## Ensemble Performance

| Strategy | F1-Score | AUC-ROC |
|----------|----------|---------|
"""

    for strategy, metrics in summary['ensembles'].items():
        report += f"| {strategy} | {metrics['f1']:.3f} | {metrics['auc']:.3f} |\n"

    # Write to file
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"Summary report saved to: {output_file}")

if __name__ == '__main__':
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    summary = aggregate_experiment_results(args.results_dir)
    generate_markdown_report(summary, args.output)
```

---

## 변경 이력

- **v1.0** (2025-10-01): 초기 실험 프로토콜 작성
  - 4개 핵심 실험 정의 (Single Model, Ensemble, Ablation, Threshold)
  - Baseline 3종 정의 (Random, ResNet-50, CLIP Zero-shot)
  - 통계적 유의성 검정 절차 수립
  - 재현성 체크리스트 추가

---

**문의사항**: 실험 프로토콜 관련 질문은 프로젝트 담당자에게 문의하세요.
