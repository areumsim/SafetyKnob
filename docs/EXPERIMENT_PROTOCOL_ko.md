# 실험 프로토콜

**버전**: 1.3
**최종 수정일**: 2026-03-24
**상태**: 활성 실험 프로토콜

본 문서는 SafetyKnob 프로젝트의 실험 프로토콜을 정의합니다. 모든 실험은 이 프로토콜을 따라 수행되어야 하며, 재현성과 비교 가능성을 보장하기 위해 표준화된 절차를 따릅니다.

---

## 목차

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

**최소 요구사항** (단일 모델 실험):
```yaml
GPU: NVIDIA RTX 3090 (24GB) or equivalent
CPU: 8 cores
RAM: 32GB
Storage: 200GB SSD
```

**환경 검증 스크립트**:
```bash
# 환경 검증 실행
python scripts/verify_environment.py

# 예상 출력:
# ✓ GPU: NVIDIA A100 (40GB) - CUDA 11.8
# ✓ CPU: 32 cores detected
# ✓ RAM: 128GB available
# ✓ Disk: 1.2TB free on /workspace
# ✓ PyTorch: 2.0.1+cu118
# ✓ All dependencies satisfied
```

### 1.2 소프트웨어 버전

**Python 환경**:
```yaml
Python: 3.10.12
CUDA: 11.8
cuDNN: 8.7.0
```

**핵심 의존성** (정확한 버전):
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
    """재현성을 위한 random seed 설정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 결정적 연산 (성능 저하 가능)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 모든 실험 스크립트에서 사용
set_seed(42)
```

### 1.3 데이터 준비

**데이터셋 다운로드 및 전처리**:
```bash
# 1단계: AI Hub에서 다운로드
# URL: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71407
# 수동 다운로드 필요 (AI Hub 계정 필요)

# 2단계: 압축 해제 및 정리
cd <프로젝트경로>
mkdir -p data/raw
unzip Construction_Safety_Images.zip -d data/raw/

# 3단계: 전처리 실행
python scripts/prepare_dataset.py \
  --input data/raw/ \
  --output data/processed/ \
  --train-scenarios SO-35,SO-36,SO-37,SO-38,SO-41,SO-42,SO-43 \
  --val-scenarios SO-44,SO-45,SO-46 \
  --test-scenarios SO-47

# 4단계: 데이터 무결성 검증
python scripts/verify_dataset.py --data-dir data/processed/
# 예상: 11,583개 이미지, 이미지당 5개 라벨, train/val/test split 정상
```

**데이터 Split 검증**:
```python
# scripts/verify_dataset.py 출력 예시:
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

**목적**: 하한 성능 측정

**구현**:
```python
class RandomBaseline:
    """하한 성능 측정을 위한 랜덤 분류기 baseline."""

    def __init__(self, p_positive=0.5, seed=42):
        """
        Args:
            p_positive: 양성 클래스 예측 확률
            seed: 재현성을 위한 random seed
        """
        self.p_positive = p_positive
        self.rng = np.random.RandomState(seed)

    def predict(self, images):
        """랜덤 라벨 예측."""
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

# 예상 성능 (10회 평균):
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
    """건설 현장 안전 데이터셋으로 fine-tuning된 ResNet-50."""

    def __init__(self, num_dimensions=5, pretrained=True):
        super().__init__()

        # ImageNet 사전학습 ResNet-50 로드
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.backbone = resnet50(weights=weights)
        else:
            self.backbone = resnet50(weights=None)

        # 최종 레이어를 안전 예측 헤드로 교체
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 원본 fc 제거

        # 공유 feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 전체 안전도 헤드
        self.safety_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 차원별 헤드
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
        # 특징 추출
        features = self.backbone(x)  # (B, 2048)
        features = self.feature_extractor(features)  # (B, 512)

        # 전체 안전도 예측
        overall = self.safety_head(features)  # (B, 1)

        # 차원별 점수 예측
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
# ResNet-50 baseline 학습
python experiments/train_resnet50_baseline.py \
  --config configs/baseline_resnet50.yaml \
  --data-dir data/processed/ \
  --output checkpoints/baselines/resnet50/ \
  --seed 42

# 테스트 셋 평가
python experiments/evaluate_baseline.py \
  --model resnet50 \
  --checkpoint checkpoints/baselines/resnet50/best_model.pt \
  --data-dir data/processed/test/ \
  --output results/baselines/resnet50_results.json
```

**예상 성능** (주의: 예비 결과 - 검증 진행 중):
```yaml
ResNet-50 Baseline 예상 결과:
  전체 안전도:
    Accuracy: 78-82%
    F1-Score: 0.76-0.80
    AUC-ROC: 0.84-0.88

  차원별 (평균):
    fall_hazard: F1 0.74-0.78
    collision_risk: F1 0.68-0.72
    equipment_hazard: F1 0.71-0.75
    environmental_risk: F1 0.65-0.69
    protective_gear: F1 0.79-0.83

  학습 시간: ~4시간 (A100 GPU)
  추론 속도: ~15ms/이미지 (batch=32)
```

### 2.3 Baseline 3: CLIP Zero-shot

**목적**: 사전학습 vision-language 모델의 zero-shot 성능 측정

**구현**:
```python
import open_clip

class CLIPZeroShotBaseline:
    """텍스트 프롬프트를 이용한 CLIP zero-shot 분류기."""

    def __init__(self, model_name='ViT-L-14', pretrained='openai'):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        # Zero-shot 분류를 위한 안전 관련 프롬프트
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
        CLIP 텍스트-이미지 유사도를 이용한 zero-shot 예측.

        Args:
            images: PIL Image 리스트 또는 torch.Tensor

        Returns:
            dict: overall_safety 및 dimensions 포함 예측 결과
        """
        # 이미지 전처리
        if isinstance(images[0], Image.Image):
            image_inputs = torch.stack([self.preprocess(img) for img in images])
        else:
            image_inputs = images

        # 이미지 인코딩
        image_features = self.model.encode_image(image_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        predictions = {'dimensions': {}}

        # 각 차원에 대해 예측
        for dimension, prompts in self.prompts.items():
            # 텍스트 프롬프트 인코딩
            text_tokens = self.tokenizer(prompts)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # 유사도 계산
            similarity = (image_features @ text_features.T)  # (N, 2)

            # 확률로 변환 (safe/dangerous에 대한 softmax)
            probs = similarity.softmax(dim=-1)

            # 점수 = P(dangerous) (위험 차원)
            if dimension == 'overall_safety':
                predictions['overall_safety'] = probs[:, 0].cpu().numpy()  # P(safe)
            else:
                predictions['dimensions'][dimension] = probs[:, 1].cpu().numpy()  # P(danger)

        # 이진 판정 (threshold=0.5)
        predictions['is_safe'] = predictions['overall_safety'] > 0.5

        return predictions
```

**실행 방법**:
```bash
# CLIP zero-shot 평가 실행
python experiments/run_clip_zeroshot.py \
  --model ViT-L-14 \
  --pretrained openai \
  --data-dir data/processed/test/ \
  --output results/baselines/clip_zeroshot.json \
  --batch-size 64
```

**예상 성능** (주의: 예비 결과 - 초기 테스트 기반):
```yaml
CLIP Zero-shot 예상 결과:
  전체 안전도:
    Accuracy: 68-72%
    F1-Score: 0.65-0.69
    AUC-ROC: 0.74-0.78

  차원별:
    fall_hazard: F1 0.62-0.66 ("height" 키워드에 민감)
    collision_risk: F1 0.58-0.62 (암묵적 위험 인식 어려움)
    equipment_hazard: F1 0.64-0.68 (장비 인식 양호)
    environmental_risk: F1 0.55-0.59 (환경 맥락 인식 어려움)
    protective_gear: F1 0.71-0.75 (최고 성능 - 가시적 특징)

  추론 속도: ~8ms/이미지 (batch=64, A100 GPU)

참고: Zero-shot 성능은 프롬프트 엔지니어링에 크게 좌우됨.
      위 수치는 상기 정의된 프롬프트 기준.
```

### 2.4 Baseline 요약

**성능 비교표** (주의: 예비 추정치):

| Baseline | 전체 Acc | 전체 F1 | 평균 차원 F1 | 학습 시간 | 추론 속도 |
|----------|----------|---------|-------------|----------|----------|
| Random | ~50% | ~0.50 | ~0.50 | 해당없음 | <1ms/이미지 |
| ResNet-50 | 78-82% | 0.76-0.80 | 0.71-0.75 | ~4시간 | ~15ms/이미지 |
| CLIP Zero-shot | 68-72% | 0.65-0.69 | 0.62-0.68 | 해당없음 | ~8ms/이미지 |
| **SafetyKnob (목표)** | **>85%** | **>0.83** | **>0.78** | ~2시간 | ~12ms/이미지 |

**Baseline 실행 우선순위**:
1. **Random**: 모든 실험에서 필수 (sanity check)
2. **CLIP Zero-shot**: 빠른 실행, 사전학습 모델 효과 검증
3. **ResNet-50**: 전통적 supervised learning 비교

---

## 3. 실험 설계

### 3.1 실험 1: 단일 모델 성능

**가설**: H1 (선형 분리 가능성) 검증

**목적**: 각 사전학습 모델의 embedding space에서 안전/위험이 선형 분리 가능한지 검증

**실험 절차**:

**1단계: Embedding 추출**
```bash
# 모든 모델의 embedding 추출
for model in siglip clip dinov2 evaclip; do
  python scripts/extract_embeddings.py \
    --model $model \
    --data-dir data/processed/ \
    --output embeddings/${model}/ \
    --batch-size 64
done

# 예상 출력 구조:
# embeddings/
#   siglip/
#     train_embeddings.npy  # (7128, 1152)
#     val_embeddings.npy    # (2872, 1152)
#     test_embeddings.npy   # (1583, 1152)
#   clip/
#     train_embeddings.npy  # (7128, 768)
#   ...
```

**2단계: Linear Probe 학습**
```bash
# Frozen embedding 위에 선형 분류기 학습
python experiments/exp1_single_model.py \
  --embeddings-dir embeddings/siglip/ \
  --labels data/processed/labels.json \
  --model-name siglip \
  --output results/exp1/siglip/ \
  --seed 42

# 모든 모델에 대해 반복
for model in clip dinov2 evaclip; do
  python experiments/exp1_single_model.py \
    --embeddings-dir embeddings/${model}/ \
    --labels data/processed/labels.json \
    --model-name $model \
    --output results/exp1/${model}/ \
    --seed 42
done
```

**Linear Probe 구성**:
```python
# 선형 분리 가능성 테스트를 위한 간단한 선형 분류기
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

# 학습 설정
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
# H1 검증을 위한 메트릭 계산
metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'f1_macro': f1_score(y_true, y_pred, average='macro'),
    'auc_roc': roc_auc_score(y_true, y_scores),
    'precision': precision_score(y_true, y_pred),
    'recall': recall_score(y_true, y_pred),

    # 차원별 메트릭
    'dimension_metrics': {
        dim: {
            'f1': f1_score(y_true_dim, y_pred_dim),
            'auc': roc_auc_score(y_true_dim, y_scores_dim)
        }
        for dim in dimension_names
    },

    # 선형 분리 가능성 메트릭 (margin 분석)
    'separability_margin': compute_margin(embeddings, labels)
}
```

**예상 결과** (주의: 조사 중):
```yaml
SigLIP:
  전체: Acc 93.2%, F1 0.921, AUC 0.967
  평균 차원 F1: 0.887
  Separability Margin: 1.42

CLIP (ViT-L/14):
  전체: Acc 89.1%, F1 0.883, AUC 0.948
  평균 차원 F1: 0.841
  Separability Margin: 1.18

DINOv2 (ViT-L/14):
  전체: Acc 86.7%, F1 0.854, AUC 0.931
  평균 차원 F1: 0.812
  Separability Margin: 1.05

EVA-CLIP:
  전체: Acc 91.4%, F1 0.907, AUC 0.956
  평균 차원 F1: 0.868
  Separability Margin: 1.31

가설 H1 검증:
  상태: 조사 중
  예상: 모든 모델이 linear probe로 85% 이상 정확도 달성
  의미: Embedding space에서 안전 관련 특징이 선형 분리 가능
```

**시각화**:
```bash
# Embedding space의 t-SNE 및 PCA 플롯 생성
python scripts/visualize_embeddings.py \
  --embeddings embeddings/siglip/test_embeddings.npy \
  --labels data/processed/test_labels.json \
  --output figures/exp1/siglip_tsne.png \
  --method tsne

# 분리 가능성 분석 플롯 생성
python scripts/analyze_separability.py \
  --results-dir results/exp1/ \
  --output figures/exp1/separability_comparison.png
```

### 3.2 실험 2: 앙상블 전략

**가설**: H2 (앙상블 강건성) 검증

**목적**: Multi-model 앙상블이 단일 모델보다 성능과 강건성에서 우수한지 검증

**실험 절차**:

**1단계: 개별 모델 학습**
```bash
# 각 모델별 neural classifier 학습
for model in siglip clip dinov2 evaclip; do
  python main.py train \
    --config configs/single_model_${model}.json \
    --data-dir data/processed/ \
    --output checkpoints/single/${model}/ \
    --seed 42
done
```

**2단계: 앙상블 전략 테스트**
```bash
# 전략 1: 가중 투표
python experiments/exp2_ensemble.py \
  --strategy weighted_vote \
  --models siglip clip dinov2 evaclip \
  --checkpoints checkpoints/single/ \
  --data-dir data/processed/test/ \
  --output results/exp2/weighted_vote/ \
  --seed 42

# 전략 2: Stacking (Meta-learner)
python experiments/exp2_ensemble.py \
  --strategy stacking \
  --models siglip clip dinov2 evaclip \
  --checkpoints checkpoints/single/ \
  --data-dir data/processed/ \
  --output results/exp2/stacking/ \
  --meta-learner logistic_regression \
  --seed 42

# 전략 3: 평균 (균등 가중치)
python experiments/exp2_ensemble.py \
  --strategy average \
  --models siglip clip dinov2 evaclip \
  --checkpoints checkpoints/single/ \
  --data-dir data/processed/test/ \
  --output results/exp2/average/ \
  --seed 42
```

**앙상블 설정**:
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
# Distribution shift 하에서 앙상블 강건성 테스트
python experiments/test_distribution_shift.py \
  --ensemble-checkpoint checkpoints/ensemble/weighted_vote/ \
  --test-scenarios SO-47 \
  --shift-types weather,lighting,camera_angle \
  --output results/exp2/distribution_shift/ \
  --seed 42
```

**예상 결과** (주의: 조사 중):
```yaml
In-Distribution (테스트 셋 SO-47):
  최고 단일 모델 (SigLIP): F1 0.921, AUC 0.967
  평균 앙상블: F1 0.928, AUC 0.971
  가중 투표: F1 0.936, AUC 0.974
  Stacking: F1 0.941, AUC 0.976

  개선폭: 최고 단일 모델 대비 F1 +1.5% ~ +2.0%

Out-of-Distribution (시뮬레이션 shift):
  기상 변화 (비/안개):
    단일 모델 평균: F1 -8.2% 하락
    앙상블 평균: F1 -4.1% 하락
    강건성 향상: 2배

  조명 변화 (저조도):
    단일 모델 평균: F1 -6.7% 하락
    앙상블 평균: F1 -3.2% 하락
    강건성 향상: 2.1배

  카메라 각도 변화:
    단일 모델 평균: F1 -5.1% 하락
    앙상블 평균: F1 -2.4% 하락
    강건성 향상: 2.1배

가설 H2 검증:
  상태: 조사 중
  예상: 앙상블이 단일 모델 대비 F1 >1.5% 향상
  예상: Distribution shift 강건성 2배 향상
```

**합의 분석**:
```python
# 앙상블 합의/불합의 측정
def compute_agreement_metrics(predictions_dict):
    """
    Args:
        predictions_dict: {model_name: predictions_array}
    Returns:
        agreement_metrics: 앙상블 합의 통계를 담은 dict
    """
    metrics = {
        'full_agreement': np.mean(all_agree),  # 모든 모델 합의
        'majority_agreement': np.mean(majority_agree),  # 50% 이상 합의
        'entropy': np.mean(prediction_entropy),  # 불확실성
        'variance': np.mean(prediction_variance)  # 분산
    }
    return metrics

# 예상 합의 패턴:
# - 고신뢰 정답: 95% 이상 완전 합의
# - 저신뢰: 60% 미만 합의, 높은 엔트로피
# - 오분류 사례: 40-60% 합의 (검토 대상 후보)
```

### 3.3 실험 3: Ablation Study

**목적**: 시스템 각 컴포넌트의 기여도 분석

**Ablation 항목**:

1. **모델 Ablation**: 각 모델 제거 시 성능 변화
2. **차원 Ablation**: 각 dimension 제거 시 전체 안전도 성능 변화
3. **아키텍처 Ablation**: Neural classifier vs Linear probe 비교
4. **학습 데이터 Ablation**: 데이터 양에 따른 성능 변화

**실행 방법**:
```bash
# Ablation 1: 앙상블에서 각 모델 제거
python experiments/exp3_ablation.py \
  --ablation-type model \
  --ensemble-config configs/ensemble_config.json \
  --data-dir data/processed/ \
  --output results/exp3/model_ablation/ \
  --seed 42

# Ablation 2: 각 차원 제거
python experiments/exp3_ablation.py \
  --ablation-type dimension \
  --full-model-checkpoint checkpoints/ensemble/best_model.pt \
  --data-dir data/processed/ \
  --output results/exp3/dimension_ablation/ \
  --seed 42

# Ablation 3: 아키텍처 비교
python experiments/exp3_ablation.py \
  --ablation-type architecture \
  --architectures linear_probe,mlp_1layer,mlp_2layer,mlp_3layer \
  --data-dir data/processed/ \
  --output results/exp3/architecture_ablation/ \
  --seed 42

# Ablation 4: 학습 데이터 크기
python experiments/exp3_ablation.py \
  --ablation-type data_size \
  --data-fractions 0.1,0.25,0.5,0.75,1.0 \
  --data-dir data/processed/ \
  --output results/exp3/data_ablation/ \
  --seed 42
```

**예상 결과** (주의: 조사 중):
```yaml
모델 Ablation (4-모델 앙상블에서 1개 제거):
  전체 앙상블 (4 모델): F1 0.936
  - SigLIP 제거: F1 0.912 (-2.4%, 최대 하락)
  - CLIP 제거: F1 0.927 (-0.9%)
  - DINOv2 제거: F1 0.931 (-0.5%)
  - EVA-CLIP 제거: F1 0.920 (-1.6%)

  발견: SigLIP 기여도 최대, DINOv2 기여도 최소

차원 Ablation (전체 예측에서 1개 차원 제거):
  전체 (5차원): 전체 F1 0.936
  - fall_hazard 제거: 0.921 (-1.5%)
  - collision_risk 제거: 0.929 (-0.7%)
  - equipment_hazard 제거: 0.932 (-0.4%)
  - environmental_risk 제거: 0.934 (-0.2%)
  - protective_gear 제거: 0.918 (-1.8%, 최대 하락)

  발견: protective_gear가 전체 안전도에 가장 유의미한 정보

아키텍처 Ablation:
  Linear Probe: F1 0.887, 파라미터 1.2K
  MLP (1 layer, 256 hidden): F1 0.921, 파라미터 320K
  MLP (2 layers, 512→256): F1 0.936, 파라미터 850K
  MLP (3 layers, 1024→512→256): F1 0.938, 파라미터 2.1M

  발견: 2-layer MLP가 최적 (3-layer에서 수확 체감)

데이터 크기 Ablation:
  10% 데이터 (712개): F1 0.742
  25% 데이터 (1,782개): F1 0.834
  50% 데이터 (3,564개): F1 0.897
  75% 데이터 (5,346개): F1 0.921
  100% 데이터 (7,128개): F1 0.936

  발견: 성능이 데이터 크기에 대해 로그-선형적으로 증가
```

### 3.4 실험 4: Threshold 분석

**목적**: 최적 threshold 선택 및 confidence calibration 검증

**실험 절차**:

**1단계: ROC/PR Curve 분석**
```bash
# Threshold 선택을 위한 ROC 및 PR curve 생성
python experiments/exp4_threshold_analysis.py \
  --model-checkpoint checkpoints/ensemble/best_model.pt \
  --data-dir data/processed/val/ \
  --output results/exp4/ \
  --metrics roc,pr,calibration \
  --seed 42
```

**2단계: Threshold 최적화**
```python
# 각 차원별 최적 threshold 탐색
def optimize_threshold(y_true, y_scores, metric='f1'):
    """
    주어진 메트릭을 최대화하는 threshold 탐색.

    Args:
        y_true: 실제 라벨 (이진)
        y_scores: 예측 점수 (연속 [0,1])
        metric: 'f1', 'balanced_accuracy', 'youden_j'

    Returns:
        best_threshold: 최적 threshold 값
        best_score: 최고 메트릭 점수
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

# 예상 최적 threshold (validation 셋):
optimal_thresholds = {
    'overall_safety': 0.52,
    'fall_hazard': 0.48,
    'collision_risk': 0.55,
    'equipment_hazard': 0.51,
    'environmental_risk': 0.58,
    'protective_gear': 0.45
}
```

**3단계: Calibration 분석**
```python
# Calibration 품질 측정
from sklearn.calibration import calibration_curve

def analyze_calibration(y_true, y_scores, n_bins=10):
    """
    Calibration 메트릭 계산.

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

# 예상 calibration (temperature scaling 이전):
# ECE: 0.08-0.12 (중간 수준의 calibration error)
# MCE: 0.15-0.20
# Brier: 0.10-0.14
```

**4단계: Temperature Scaling**
```python
# Calibration을 위한 temperature scaling 적용
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, logits, labels):
        """Validation 셋에서 temperature 학습."""
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

# 예상 temperature: T ≈ 1.2-1.5
# Scaling 후: ECE < 0.05, MCE < 0.10
```

**예상 결과** (주의: 조사 중):
```yaml
Threshold 분석 결과:
  전체 안전도:
    최적 Threshold: 0.52 (F1 최대화)
    Threshold 0.52: Precision 0.94, Recall 0.93, F1 0.936
    Threshold 0.50: Precision 0.92, Recall 0.95, F1 0.934

  차원별 최적 Threshold:
    fall_hazard: 0.48 (균형, 높은 재현율 우선)
    collision_risk: 0.55 (보수적, 높은 정밀도)
    equipment_hazard: 0.51 (균형)
    environmental_risk: 0.58 (보수적)
    protective_gear: 0.45 (관대, 감지 우선)

Calibration 결과:
  Temperature Scaling 이전:
    ECE: 0.102 (중간 수준의 calibration error)
    MCE: 0.183
    Brier Score: 0.118

  Temperature Scaling 이후 (T=1.38):
    ECE: 0.042 (잘 보정됨)
    MCE: 0.091
    Brier Score: 0.095

  개선폭: ECE 59% 감소, Brier 19% 개선
```

---

## 4. 평가 프로토콜

### 4.1 표준 메트릭

모든 실험은 다음 메트릭을 보고해야 합니다:

**분류 메트릭**:
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, classification_report
)

def compute_standard_metrics(y_true, y_pred, y_scores):
    """
    표준 분류 메트릭 일괄 계산.

    Args:
        y_true: 실제 이진 라벨
        y_pred: 예측 이진 라벨
        y_scores: 예측 확률/점수

    Returns:
        dict: 모든 메트릭
    """
    metrics = {
        # Threshold 의존 메트릭 (0.5 또는 최적 threshold)
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),

        # Threshold 독립 메트릭
        'auc_roc': roc_auc_score(y_true, y_scores),
        'auc_pr': average_precision_score(y_true, y_scores),

        # 혼동 행렬 구성요소
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),

        # 상세 분류 보고서
        'classification_report': classification_report(
            y_true, y_pred, output_dict=True
        )
    }

    # TN, FP, FN, TP 명시적 추가
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    })

    return metrics
```

**차원별 메트릭**:
```python
def compute_dimension_metrics(predictions, ground_truth):
    """각 안전 차원별 메트릭 계산."""
    dimension_names = ['fall_hazard', 'collision_risk', 'equipment_hazard',
                       'environmental_risk', 'protective_gear']

    results = {}
    for dim in dimension_names:
        y_true = ground_truth[dim]
        y_scores = predictions['dimensions'][dim]
        y_pred = (y_scores >= 0.5).astype(int)

        results[dim] = compute_standard_metrics(y_true, y_pred, y_scores)

    # 차원 평균 메트릭 계산
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
3. **혼동 행렬** (`figures/exp{N}/confusion_matrix.png`)
4. **학습 곡선** (`figures/exp{N}/training_curves.png`)
5. **차원별 성능** (`figures/exp{N}/dimension_performance.png`)

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
# 여러 seed로 실험 실행
for seed in 42 123 456 789 1024; do
  python experiments/exp1_single_model.py \
    --model siglip \
    --data-dir data/processed/ \
    --output results/exp1/siglip/seed_${seed}/ \
    --seed $seed
done

# 결과 집계
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
    모델 A가 모델 B보다 유의하게 우수한지 검정.

    Args:
        results_a: 모델 A의 메트릭 값 리스트 (다중 실행)
        results_b: 모델 B의 메트릭 값 리스트 (다중 실행)
        metric: 보고용 메트릭 이름

    Returns:
        dict: 통계 검정 결과
    """
    # Paired t-test (동일 seed 사용)
    t_stat, p_value = ttest_rel(results_a, results_b)

    # 효과 크기 (Cohen's d)
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

# 사용 예시:
siglip_f1 = [0.921, 0.918, 0.924, 0.919, 0.922]
clip_f1 = [0.883, 0.879, 0.887, 0.881, 0.885]

result = compare_models_significance(siglip_f1, clip_f1, metric='F1')
# 예상: p < 0.01, Cohen's d ≈ 5.2 (매우 큰 효과 크기)
```

### 5.3 Bootstrap 신뢰구간

AUC 등 복잡한 메트릭에 대해서는 **bootstrap** 사용:

```python
from sklearn.utils import resample

def bootstrap_confidence_interval(y_true, y_scores, metric_func, n_iterations=1000, alpha=0.05):
    """
    메트릭에 대한 bootstrap 신뢰구간 계산.

    Args:
        y_true: 실제 라벨
        y_scores: 예측 점수
        metric_func: 메트릭 계산 함수 (예: roc_auc_score)
        n_iterations: Bootstrap 샘플 수
        alpha: 유의 수준 (95% CI의 경우 0.05)

    Returns:
        dict: 평균, CI 하한, CI 상한
    """
    scores = []
    n_samples = len(y_true)

    for _ in range(n_iterations):
        # 복원 추출
        indices = resample(range(n_samples), n_samples=n_samples)
        y_true_boot = y_true[indices]
        y_scores_boot = y_scores[indices]

        # Bootstrap 샘플에서 메트릭 계산
        score = metric_func(y_true_boot, y_scores_boot)
        scores.append(score)

    # 백분위수 계산
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))

    return {
        'mean': np.mean(scores),
        'ci_lower': lower,
        'ci_upper': upper,
        'alpha': alpha
    }

# 사용 예시:
ci = bootstrap_confidence_interval(
    y_true, y_scores,
    metric_func=roc_auc_score,
    n_iterations=1000
)
# 출력: {'mean': 0.967, 'ci_lower': 0.961, 'ci_upper': 0.973, 'alpha': 0.05}
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

- [ ] 훈련 진행 모니터링 (loss, metrics)
- [ ] Early stopping 작동 확인 (과적합 방지)
- [ ] GPU 메모리 사용량 모니터링
- [ ] 중간 checkpoint 저장 확인

### 6.3 실험 완료 후

- [ ] 테스트 셋 결과 생성 (`results.json` 저장됨)
- [ ] 모든 필수 메트릭 계산됨 (accuracy, F1, AUC, precision, recall)
- [ ] 차원별 메트릭 계산됨
- [ ] 시각화 생성됨 (ROC, PR, 혼동 행렬 등)
- [ ] 다중 실행 수행됨 (최소 5회, 서로 다른 seed)
- [ ] 통계적 유의성 검정 수행됨 (baseline 대비)
- [ ] 실험 환경 정보 저장됨 (`environment.txt`)
- [ ] Checkpoint와 config 백업됨
- [ ] 결과가 문서화됨 (실험 보고서에 기록)

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

set -e  # 오류 시 종료

echo "=== SafetyKnob 실험 파이프라인 ==="
echo "시작 시각: $(date)"

# 1단계: 환경 검증
echo "[Step 1/6] 환경 검증 중..."
python scripts/verify_environment.py || exit 1

# 2단계: 데이터 준비
echo "[Step 2/6] 데이터셋 준비 중..."
python scripts/prepare_dataset.py \
  --input data/raw/ \
  --output data/processed/ \
  --train-scenarios SO-35,SO-36,SO-37,SO-38,SO-41,SO-42,SO-43 \
  --val-scenarios SO-44,SO-45,SO-46 \
  --test-scenarios SO-47

python scripts/verify_dataset.py --data-dir data/processed/

# 3단계: Baseline 실험
echo "[Step 3/6] Baseline 실험 실행 중..."
python experiments/run_baseline_random.py --output results/baselines/random.json
python experiments/run_clip_zeroshot.py --output results/baselines/clip_zeroshot.json
python experiments/train_resnet50_baseline.py --output checkpoints/baselines/resnet50/

# 4단계: 실험 1 (단일 모델 성능)
echo "[Step 4/6] 실험 1: 단일 모델 성능..."
for model in siglip clip dinov2 evaclip; do
  for seed in 42 123 456 789 1024; do
    python experiments/exp1_single_model.py \
      --model $model \
      --data-dir data/processed/ \
      --output results/exp1/${model}/seed_${seed}/ \
      --seed $seed
  done
done

# 5단계: 실험 2 (앙상블 전략)
echo "[Step 5/6] 실험 2: 앙상블 전략..."
python experiments/exp2_ensemble.py \
  --strategy weighted_vote \
  --output results/exp2/weighted_vote/

python experiments/exp2_ensemble.py \
  --strategy stacking \
  --output results/exp2/stacking/

# 6단계: 요약 보고서 생성
echo "[Step 6/6] 요약 보고서 생성 중..."
python scripts/generate_experiment_summary.py \
  --results-dir results/ \
  --output reports/experiment_summary_$(date +%Y%m%d).md

echo "=== 모든 실험 완료 시각: $(date) ==="
```

### A.2 결과 집계 스크립트

```python
# scripts/generate_experiment_summary.py

import json
import glob
from pathlib import Path
import numpy as np

def aggregate_experiment_results(results_dir):
    """모든 실험 결과를 요약으로 집계."""

    summary = {
        'baselines': {},
        'single_models': {},
        'ensembles': {},
        'comparisons': []
    }

    # Baseline 결과 수집
    baseline_files = glob.glob(f"{results_dir}/baselines/*.json")
    for file in baseline_files:
        with open(file) as f:
            data = json.load(f)
            model_name = Path(file).stem
            summary['baselines'][model_name] = {
                'f1': data['results']['overall_safety']['f1'],
                'auc': data['results']['overall_safety']['auc_roc']
            }

    # 단일 모델 결과 수집 (다중 실행 집계)
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

    # 앙상블 결과 수집
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
    """마크다운 요약 보고서 생성."""

    report = f"""# SafetyKnob 실험 요약

생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Baseline 성능

| 모델 | F1-Score | AUC-ROC |
|------|----------|---------|
"""

    for model, metrics in summary['baselines'].items():
        report += f"| {model} | {metrics['f1']:.3f} | {metrics['auc']:.3f} |\n"

    report += """
## 단일 모델 성능 (다중 실행 평균)

| 모델 | F1-Score (평균 ± 표준편차) | 95% CI |
|------|--------------------------|--------|
"""

    for model, metrics in summary['single_models'].items():
        report += f"| {model} | {metrics['f1_mean']:.3f} ± {metrics['f1_std']:.3f} | [{metrics['f1_ci_95'][0]:.3f}, {metrics['f1_ci_95'][1]:.3f}] |\n"

    report += """
## 앙상블 성능

| 전략 | F1-Score | AUC-ROC |
|------|----------|---------|
"""

    for strategy, metrics in summary['ensembles'].items():
        report += f"| {strategy} | {metrics['f1']:.3f} | {metrics['auc']:.3f} |\n"

    # 파일 저장
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"요약 보고서 저장 완료: {output_file}")

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

## 보완 실험 프로토콜 (2026-03 추가)

### 실험 A: ResNet50-Frozen Baseline (공정 비교)

**목적**: Foundation model의 representation 품질이 ImageNet 사전학습 CNN보다 우수함을 입증

**실행 방법**:
```bash
# Scenario split에서 ResNet50-Frozen 실험
python experiments/train_baseline.py \
  --model resnet50 --frozen \
  --data-dir data_scenario/ \
  --output results/scenario/baseline_frozen/ \
  --epochs 30 --lr 1e-3

# Temporal split에서 ResNet50-Frozen/Finetuned 실험
python experiments/train_baseline.py \
  --model resnet50 --frozen \
  --data-dir data_temporal/ \
  --output results/temporal/baseline_frozen/

python experiments/train_baseline.py \
  --model resnet50 \
  --data-dir data_temporal/ \
  --output results/temporal/baseline_finetuned/
```

**비교표 (실측)**:

| 모델 | 모드 | 학습 파라미터 수 | Scenario F1 | Temporal F1 |
|------|------|-----------------|-------------|-------------|
| ResNet50 | Frozen | 1,049,601 | 78.42% | 57.33% |
| ResNet50 | Finetuned | 24.6M | 95.49% | N/A (미실행) |
| SigLIP | Frozen+2layer | 656,129 | 96.13% | 68.08% |

**판정**: Foundation model(SigLIP) >> ResNet50-Frozen (+17.71%p). Frozen representation 품질 격차 명확.

### Embedding 추출 워크플로우

**목적**: 모델 임베딩을 사전 추출하여 probe 실험을 초 단위로 반복 실행

**실행 방법**:
```bash
# 1. Scenario split 임베딩 추출 (모델당 ~30분, GPU 1개)
for model in siglip clip dinov2; do
  python scripts/extract_embeddings.py \
    --model $model \
    --data-dir data_scenario \
    --output embeddings/scenario/$model
done

# 2. Temporal split 임베딩 추출
for model in siglip clip dinov2; do
  python scripts/extract_embeddings.py \
    --model $model \
    --data-dir data_temporal \
    --output embeddings/temporal/$model
done

# 3. 캐시 임베딩 기반 probe 학습 (~3초/실험)
python experiments/train_from_embeddings.py \
  --model siglip \
  --embeddings-dir embeddings/scenario/siglip \
  --probe-depth 2layer \
  --output results/scenario/siglip_2layer
```

**출력 구조**:
```
embeddings/
├── scenario/{siglip,clip,dinov2}/
│   ├── train_embeddings.pt   # (N, D) 텐서 + filenames
│   ├── train_labels.pt       # labels dict
│   ├── val_embeddings.pt
│   ├── val_labels.pt
│   ├── test_embeddings.pt
│   └── test_labels.pt
└── temporal/{siglip,clip,dinov2}/
    └── (동일 구조)
```

**시간 절감**: SigLIP 기준 11시간/실험 → 3초/실험 (임베딩 추출 1회 30분 + probe 학습 3초 x N실험)

### 실험 B: Probe Depth Ablation (선형 분리 검증)

**목적**: RESEARCH_METHODOLOGY.md에 명시된 "linear probe baseline" + "비선형 MLP 비교" 검증

**실행 방법**:
```bash
for depth in linear 1layer 2layer; do
  python experiments/train_binary.py \
    --model siglip \
    --probe-depth $depth \
    --data-dir data_scenario/ \
    --output results/scenario/siglip_${depth}/ \
    --epochs 20
done
```

**판정 기준**:
- Linear F1 vs 2-layer F1 차이 < 3%p → "선형 분리 가능" 확인
- Linear F1 vs 2-layer F1 차이 > 5%p → 비선형 특성 중요, 주장 수정 필요

### 실험 C: Temporal Ensemble (RQ2 보완)

**목적**: Distribution shift 하에서 앙상블의 강건성 검증

**실행 방법**:
```bash
python scripts/run_ensemble_binary.py \
  --data-dir data_temporal/ \
  --results-dir results/temporal/binary/
```

### 실험 D: 5D 독립 vs Multi-task (RQ3 검증)

**목적**: 공유 feature extractor의 positive transfer 여부 확인

**실행 방법**:
```bash
for cat in A B C D E; do
  python experiments/train_binary.py \
    --model siglip \
    --category $cat \
    --data-dir data_scenario/ \
    --output results/scenario/siglip_independent_${cat}/
done
```

**비교**: 각 카테고리별 독립 F1 vs multi-task 모델의 해당 차원 F1

---

## 변경 이력

- **v1.3** (2026-03-24): Multi-seed 검증, DANN, Scaling curve, Ensemble ablation 결과 추가
  - 전체 실험 5-seed 반복 (mean±std 제공)
  - DANN domain adaptation: ~~99.24% (무효화, 데이터 누출)~~ → clean 65.28% (효과 없음). LoRA: 77.77%
  - 데이터 scaling curve: 10%~100% 5단계
  - 앙상블 2-모델 ablation: SigLIP+DINOv2 최적
  - 카테고리별 temporal 분석: 카테고리별 shift 분석

- **v1.2** (2026-03-24): 실험 결과 반영 및 임베딩 워크플로우 추가
  - 실험 A: ResNet50-Frozen 실측값 반영 (Scenario 78.42%, Temporal 57.33%)
  - 임베딩 사전추출 워크플로우 섹션 추가
  - SigLIP Scenario F1 95.73% → 96.13% (2-layer probe 정정)

- **v1.1** (2026-03-20): 보완 실험 프로토콜 추가
  - 실험 A: ResNet50-Frozen baseline (공정 비교)
  - 실험 B: Probe depth ablation (선형 분리 검증)
  - 실험 C: Temporal ensemble (RQ2 보완)
  - 실험 D: 5D 독립 vs multi-task (RQ3 검증)
  - `train_baseline.py`에 `--frozen` 옵션, `train_binary.py`에 `--probe-depth`/`--category` 옵션 추가
  - `run_ensemble_binary.py`에 `--data-dir`/`--results-dir` 옵션 추가

- **v1.0** (2025-10-01): 초기 실험 프로토콜 작성
  - 4개 핵심 실험 정의 (단일 모델, 앙상블, Ablation, Threshold)
  - Baseline 3종 정의 (Random, ResNet-50, CLIP Zero-shot)
  - 통계적 유의성 검정 절차 수립
  - 재현성 체크리스트 추가

---

**문의사항**: 실험 프로토콜 관련 질문은 프로젝트 담당자에게 문의하세요.
