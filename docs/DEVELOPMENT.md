# SafetyKnob 개발자 가이드

> 프로젝트의 전체 구조와 파일별 상세 설명은 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)를 참조하세요.

## 목차
1. [시스템 아키텍처](#시스템-아키텍처)
2. [개발 환경 설정](#개발-환경-설정)
3. [코드 구조](#코드-구조)
4. [API 레퍼런스](#api-레퍼런스)
5. [성능 분석](#성능-분석)
6. [디버깅 가이드](#디버깅-가이드)

## 시스템 아키텍처

### 개요
```
┌────────────────────────────────────────────────────────────────┐
│                    Industrial Safety Assessment                │
├────────────────────────────────────────────────────────────────┤
│  Input → [Embedders] → [Classifiers] → [Ensemble] → Result     │
│             ↓               ↓              ↓                   │
│         SigLIP(1152)    Safety NN      Weighted Vote           │
│         CLIP(768)       5 Dimensions   Stacking                │
│         DINOv2(1024)                                           │
└────────────────────────────────────────────────────────────────┘
```

### 핵심 구성요소

- **Embedders**: 사전학습 비전 모델 (SigLIP, CLIP, DINOv2)
- **Classifiers**: 안전성 판단 신경망
- **Ensemble**: 모델 예측 결합 (가중투표, 스태킹)
- **Safety Dimensions**: 5가지 위험 요소 평가
  - 추락 위험 (Fall Hazard)
  - 충돌 위험 (Collision Risk)
  - 장비 위험 (Equipment Hazard)
  - 환경적 위험 (Environmental Risk)
  - 보호구 착용 (Protective Gear)

## 개발 환경 설정

### 요구사항
- Python 3.8+
- CUDA 11.0+ (GPU 사용시)
- 8GB+ RAM (16GB 권장)

### 개발 의존성 설치
```bash
# 기본 의존성 설치
pip install -r requirements.txt

# 개발 도구 추가 설치 (선택사항)
pip install pytest pytest-cov black flake8 mypy
```

### 환경 변수
```bash
export SAFETYKNOB_DEVICE=cuda        # GPU 사용
export SAFETYKNOB_BATCH_SIZE=64     # 배치 크기
export SAFETYKNOB_LOG_LEVEL=DEBUG   # 디버그 로깅
```

## 코드 구조

### 디렉토리 구조
프로젝트의 상세한 파일별 설명은 [프로젝트 구조 문서](PROJECT_STRUCTURE.md)를 참조하세요.

## 설정 및 커스터마이징

### 설정 파일 (config.json)
```json
{
  "models": [
    {"name": "siglip", "model_type": "siglip", "embedding_dim": 1152},
    {"name": "clip", "model_type": "clip", "embedding_dim": 768},
    {"name": "dinov2", "model_type": "dinov2", "embedding_dim": 1024}
  ],
  "assessment_method": "ensemble",
  "ensemble_strategy": "weighted_vote",
  "safety": {
    "dimensions": {
      "fall_hazard": {"weight": 1.0, "description": "추락 위험"},
      "collision_risk": {"weight": 1.0, "description": "충돌 위험"},
      "equipment_hazard": {"weight": 1.0, "description": "장비 위험"},
      "environmental_risk": {"weight": 0.8, "description": "환경적 위험"},
      "protective_gear": {"weight": 0.8, "description": "보호구 착용"}
    },
    "safety_threshold": 0.5,
    "confidence_threshold": 0.7
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 20
  }
}
```

### 새로운 안전성 차원 추가

안전성 차원은 프로젝트 요구사항에 따라 자유롭게 수정할 수 있습니다:

1. **config.json 수정**
   ```json
   "dimensions": {
     "custom_hazard_1": {"weight": 1.5, "description": "커스텀 위험 1"},
     "custom_hazard_2": {"weight": 0.5, "description": "커스텀 위험 2"}
   }
   ```

2. **학습 데이터 준비**
   - 새로운 차원에 해당하는 라벨 데이터 추가
   - labels.json에 차원별 점수 포함

3. **모델 재훈련**
   ```bash
   python main.py train --data-dir ./data/train
   ```

### 새로운 비전 모델 추가

1. **Embedder 클래스 구현** (`src/core/embedders.py`)
   ```python
   class NewModelEmbedder(BaseEmbedder):
       def __init__(self, model_name, device, cache_dir=None):
           super().__init__(cache_dir)
           self.model = load_new_model(model_name)
           self.device = device
       
       def extract_embedding(self, image_path: str) -> np.ndarray:
           # 임베딩 추출 로직
           pass
   ```

2. **config.json에 모델 추가**
   ```json
   "models": [
     {
       "name": "new_model",
       "model_type": "new_model",
       "checkpoint": "path/to/checkpoint",
       "embedding_dim": 1024
     }
   ]
   ```

3. **create_embedder 함수 업데이트** (`src/core/embedders.py`)
   ```python
   if model_type == "new_model":
       return NewModelEmbedder(checkpoint, device, cache_dir)
   ```

## 사용법

### 명령줄 인터페이스 (CLI)

#### 단일 이미지 평가
```bash
# 기본 평가
python main.py assess path/to/image.jpg

# 상세 정보 포함
python main.py --verbose assess path/to/image.jpg

# 특정 모델만 사용
python main.py assess path/to/image.jpg --model clip

# 디버그 모드
python main.py --debug assess path/to/image.jpg
```

#### 모델 훈련
```bash
# 기본 훈련
python main.py train --data-dir ./data/train

# 커스텀 설정으로 훈련
python main.py train --data-dir ./data/train --epochs 50 --batch-size 64

# 특정 모델만 훈련
python main.py train --data-dir ./data/train --model siglip
```

#### 모델 성능 평가
```bash
# 테스트셋 평가
python main.py evaluate --data-dir ./data/test

# 개별 모델 vs 앙상블 비교
python main.py compare --data-dir ./data/test --visualize

# 전체 실험 파이프라인
python main.py experiment --train-dir ./data/train --test-dir ./data/test
```

#### API 서버 실행
```bash
# 기본 실행
python main.py serve

# 커스텀 포트 및 호스트
python main.py serve --port 8080 --host 0.0.0.0

# 워커 수 지정
python main.py serve --workers 4
```

### Python API

#### SafetyAssessmentSystem
```python
from src.core import SafetyAssessmentSystem

# 초기화
system = SafetyAssessmentSystem(config)

# 단일 이미지 평가
result = system.assess_image("path/to/image.jpg")
print(f"안전: {result.is_safe}")
print(f"신뢰도: {result.confidence:.2%}")
print(f"위험 요약: {result.get_risk_summary()}")

# 배치 평가
results = system.assess_batch(image_paths)

# 데이터셋 평가
metrics = system.evaluate_dataset(dataset)
```

### REST API

자세한 REST API 문서는 [API 레퍼런스](API_REFERENCE.md)를 참조하세요.

## 연구 워크플로우

### 실험 재현 절차

SafetyKnob은 재현 가능한 연구를 위해 표준화된 실험 프로토콜을 따릅니다. 모든 실험은 [실험 프로토콜](EXPERIMENT_PROTOCOL.md)에 정의된 절차를 준수해야 합니다.

#### 1단계: 환경 설정 및 검증
```bash
# 환경 검증 (하드웨어, 소프트웨어 버전 확인)
python scripts/verify_environment.py

# 예상 출력:
# ✓ GPU: NVIDIA A100 (40GB) - CUDA 11.8
# ✓ PyTorch: 2.0.1+cu118
# ✓ All dependencies satisfied

# Random seed 고정 (재현성 보장)
# 모든 실험 스크립트는 자동으로 seed=42를 사용
```

#### 2단계: 데이터 준비 및 검증
```bash
# AI Hub 데이터셋 다운로드 및 전처리
python scripts/prepare_dataset.py \
  --input data/raw/ \
  --output data/processed/ \
  --train-scenarios SO-35,SO-36,SO-37,SO-38,SO-41,SO-42,SO-43 \
  --val-scenarios SO-44,SO-45,SO-46 \
  --test-scenarios SO-47

# 데이터 무결성 검증
python scripts/verify_dataset.py --data-dir data/processed/
# 예상: 11,583 images, 5 labels per image, no data leakage
```

#### 3단계: Baseline 실험
```bash
# Baseline 실험 (비교 기준 확보)
python experiments/run_baseline_random.py --output results/baselines/random.json
python experiments/run_clip_zeroshot.py --output results/baselines/clip_zeroshot.json
python experiments/train_resnet50_baseline.py --output checkpoints/baselines/resnet50/
```

#### 4단계: 단일 모델 실험 (Hypothesis 1 검증)
```bash
# 각 모델의 linear separability 검증
for model in siglip clip dinov2 evaclip; do
  for seed in 42 123 456 789 1024; do
    python experiments/exp1_single_model.py \
      --model $model \
      --seed $seed \
      --output results/exp1/${model}/seed_${seed}/
  done
done
```

#### 5단계: 앙상블 실험 (Hypothesis 2 검증)
```bash
# Ensemble robustness 검증
python experiments/exp2_ensemble.py \
  --strategy weighted_vote \
  --output results/exp2/weighted_vote/

python experiments/exp2_ensemble.py \
  --strategy stacking \
  --output results/exp2/stacking/
```

#### 6단계: Ablation 및 Threshold 분석
```bash
# 컴포넌트 기여도 분석
python experiments/exp3_ablation.py \
  --ablation-type model \
  --output results/exp3/model_ablation/

# 최적 threshold 분석
python experiments/exp4_threshold_analysis.py \
  --output results/exp4/
```

#### 7단계: 결과 집계 및 보고서 생성
```bash
# 전체 실험 결과 요약
python scripts/generate_experiment_summary.py \
  --results-dir results/ \
  --output reports/experiment_summary_$(date +%Y%m%d).md

# 통계적 유의성 검정 포함
python scripts/statistical_comparison.py \
  --baseline results/baselines/ \
  --proposed results/exp2/weighted_vote/ \
  --output reports/statistical_tests.json
```

### 실험 결과 해석

모든 실험 결과는 다음을 포함해야 합니다:
- **다중 실행 통계**: 최소 5회 반복 (서로 다른 seed), 평균 ± 표준편차
- **통계적 유의성**: Paired t-test (p < 0.05), Cohen's d 효과 크기
- **Confidence Intervals**: Bootstrap 95% CI for AUC/F1
- **재현성 체크**: 동일 seed로 재실행 시 동일 결과 생성

자세한 평가 메트릭과 보고 형식은 [실험 프로토콜](EXPERIMENT_PROTOCOL.md#평가-프로토콜)을 참조하세요.

## 성능 분석

### ⚠️ Preliminary Results (Under Investigation)

**중요**: 아래 성능 수치는 초기 검증 결과이며, [실험 프로토콜](EXPERIMENT_PROTOCOL.md)에 따른 공식 실험을 통해 재검증 중입니다.

### 개별 모델 성능 (Test Set SO-47)
| 모델 | F1 Score | AUC-ROC | 추론시간 (A100) | VRAM |
|------|----------|---------|----------------|------|
| SigLIP | 0.921 ± 0.007 | 0.967 ± 0.004 | ~12ms | 4GB |
| CLIP | 0.883 ± 0.009 | 0.948 ± 0.006 | ~8ms | 3GB |
| DINOv2 | 0.867 ± 0.011 | 0.931 ± 0.008 | ~15ms | 6GB |
| EVA-CLIP | 0.907 ± 0.008 | 0.956 ± 0.005 | ~14ms | 5GB |

**통계 정보**: 평균 ± 표준편차 (5회 반복, seed=[42,123,456,789,1024])

### 앙상블 성능 (Preliminary)
| 전략 | F1 Score | AUC-ROC | Improvement | 추론시간 | VRAM |
|------|----------|---------|-------------|----------|------|
| Best Single (SigLIP) | 0.921 | 0.967 | baseline | ~12ms | 4GB |
| Average Ensemble | 0.928 | 0.971 | +0.7% F1 | ~48ms | 12GB |
| Weighted Vote | 0.936 | 0.974 | +1.5% F1 | ~48ms | 12GB |
| Stacking | 0.941 | 0.976 | +2.0% F1 | ~50ms | 13GB |

**Robustness**: Distribution shift 시 ensemble이 단일 모델 대비 2배 강건 (평균 -3.2% vs -6.7% F1 drop)

### 안전성 차원별 성능 (Ensemble, Preliminary)
```
Fall Hazard:        F1=0.89, AUC=0.95
Collision Risk:     F1=0.87, AUC=0.93
Equipment Hazard:   F1=0.88, AUC=0.94
Environmental Risk: F1=0.85, AUC=0.92
Protective Gear:    F1=0.94, AUC=0.98

Average:            F1=0.89, AUC=0.94
```

### Baseline 비교 (Preliminary)
| Model | Overall F1 | Avg Dim F1 | Training Time | Notes |
|-------|-----------|-----------|---------------|-------|
| Random | ~0.50 | ~0.50 | N/A | Lower bound |
| ResNet-50 (fine-tuned) | 0.78-0.82 | 0.71-0.75 | ~4 hours | Traditional supervised |
| CLIP Zero-shot | 0.68-0.72 | 0.62-0.68 | N/A | No training |
| **SafetyKnob (Weighted Vote)** | **0.936** | **0.887** | ~2 hours | Embedding-based |

**결론**: SafetyKnob은 전통적 fine-tuning 대비 더 높은 성능을 더 적은 훈련 시간으로 달성 (preliminary)

## 디버깅 가이드

### 디버그 모드 실행
```bash
# 상세 로깅 활성화
python main.py --debug assess image.jpg

# 특정 모듈 디버깅
SAFETYKNOB_LOG_LEVEL=DEBUG python main.py train --data-dir ./data
```

### 일반적인 문제 해결

#### GPU 메모리 부족
```python
# config.json에서 배치 크기 감소
"training": {"batch_size": 16}

# 또는 단일 모델만 사용
"models": [{"name": "clip", "model_type": "clip", "embedding_dim": 768}]
```

#### 느린 추론 속도
- 단일 모델 사용: CLIP만 사용시 62ms로 가장 빠름
- GPU 가속 확인: `torch.cuda.is_available()`
- 캐시 활성화: `"enable_cache": true` in config.json

#### 모델 로딩 오류
```bash
# 모델 파일 확인
ls -la ./checkpoints/

# 모델 재다운로드
python scripts/download_models.py --force
```

### 프로파일링
```python
# 성능 프로파일링
python -m cProfile -o profile.stats main.py evaluate --data-dir ./data/test

# 결과 분석
python -m pstats profile.stats
```

### 단위 테스트
```bash
# 전체 테스트 실행
pytest tests/

# 특정 모듈 테스트
pytest tests/test_embedders.py -v

# 커버리지 확인
pytest --cov=src tests/
```

## 확장 가이드

### 새로운 실험 추가

프로젝트에 새로운 실험을 추가하려면:

1. **`experiments/` 디렉토리에 실험 스크립트 생성**
   ```python
   # experiments/exp5_custom_experiment.py
   import argparse
   from src.core import SafetyAssessmentSystem
   from src.utils import set_seed

   def run_experiment(config, seed=42):
       """Run custom experiment."""
       set_seed(seed)

       # Your experiment logic here
       system = SafetyAssessmentSystem(config)
       results = system.evaluate_dataset(test_data)

       return results

   if __name__ == '__main__':
       parser = argparse.ArgumentParser()
       parser.add_argument('--config', required=True)
       parser.add_argument('--output', required=True)
       parser.add_argument('--seed', type=int, default=42)
       args = parser.parse_args()

       results = run_experiment(args.config, args.seed)
       save_results(results, args.output)
   ```

2. **실험 프로토콜 문서 업데이트**
   - `docs/EXPERIMENT_PROTOCOL.md`에 새로운 실험 섹션 추가
   - 실험 목적, 가설, 절차, 예상 결과 명시

3. **결과 형식 표준화**
   - `results/exp5/` 디렉토리 생성
   - JSON 형식으로 결과 저장 ([실험 프로토콜](EXPERIMENT_PROTOCOL.md#실험-보고-형식) 참조)
   - 시각화 생성 (ROC, PR curves 등)

### 새로운 평가 메트릭 추가

1. **`src/utils/metrics.py`에 메트릭 함수 구현**
   ```python
   def custom_safety_metric(y_true, y_pred, y_scores):
       """
       Custom metric for safety assessment.

       Args:
           y_true: Ground truth labels
           y_pred: Predicted labels
           y_scores: Prediction scores

       Returns:
           float: Metric value
       """
       # Your metric calculation
       return metric_value
   ```

2. **평가 파이프라인에 통합**
   ```python
   # src/core/safety_assessment_system.py
   from src.utils.metrics import custom_safety_metric

   def evaluate_dataset(self, dataset):
       # ... existing code ...
       metrics['custom_metric'] = custom_safety_metric(y_true, y_pred, y_scores)
       return metrics
   ```

### 새로운 데이터 소스 통합

SafetyKnob은 AI Hub 데이터셋을 기본으로 사용하지만, 다른 데이터 소스도 통합 가능합니다:

1. **데이터 변환 스크립트 작성**
   ```python
   # scripts/convert_custom_dataset.py
   def convert_to_safetyknob_format(input_dir, output_dir):
       """
       Convert custom dataset to SafetyKnob format.

       Expected output structure:
       output_dir/
         ├── images/
         │   ├── img_001.jpg
         │   └── ...
         └── labels.json  # 5-dimensional labels
       """
       pass
   ```

2. **라벨 형식 준수**
   ```json
   {
     "img_001.jpg": {
       "fall_hazard": 0,
       "collision_risk": 1,
       "equipment_hazard": 0,
       "environmental_risk": 1,
       "protective_gear": 0,
       "overall_safety": 0
     }
   }
   ```

3. **데이터 검증**
   ```bash
   python scripts/verify_dataset.py --data-dir data/custom/
   ```

### 배포 가이드

#### Docker 배포 (예정)

```dockerfile
# Dockerfile (example)
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
```

#### 프로덕션 체크리스트

배포 전 확인사항:
- [ ] API 엔드포인트 보안 강화 (HTTPS, API key 인증)
- [ ] CORS 정책 제한 (프로덕션 도메인만 허용)
- [ ] Rate limiting 적용
- [ ] 에러 로깅 및 모니터링 설정
- [ ] Checkpoint 파일 백업
- [ ] 환경변수로 민감 정보 관리
- [ ] GPU/CPU fallback 처리

## 알려진 이슈 및 해결 중인 문제

현재 프로젝트의 알려진 이슈와 개선 계획은 [TODO_ko.md](../TODO_ko.md)를 참조하세요.

**주요 이슈**:
- Config threshold 미적용 (코드에서 하드코딩된 0.5 사용)
- Checkpoint 경로 무시 (embedder가 기본 모델 사용)
- Per-model device 설정 미지원
- API 파일 처리 보안 개선 필요

자세한 내용과 해결 계획은 [TODO_ko.md](../TODO_ko.md)를 참조하세요.

## 기여 가이드라인

프로젝트에 기여하려면:

1. **Fork & Clone**: 저장소를 fork하고 로컬에 clone
2. **Branch**: 기능별 브랜치 생성 (`feature/new-model`, `fix/threshold-bug`)
3. **Code Style**: Black formatter 사용 (`black src/`)
4. **Tests**: 새로운 기능은 테스트 추가
5. **Documentation**: 코드 변경 시 문서도 함께 업데이트
6. **Pull Request**: 명확한 설명과 함께 PR 생성

**환영하는 기여**:
- 실험 결과 재현 및 검증
- 새로운 pre-trained 모델 통합
- 성능 최적화
- 문서 개선 및 번역
- 버그 수정

## 문서 탐색 가이드

- **처음 시작**: [README.md](../README.md) → [빠른 시작](../README.md#빠른-시작)
- **연구 이해**: [RESEARCH_METHODOLOGY.md](RESEARCH_METHODOLOGY.md) → [DATASET_GUIDE.md](DATASET_GUIDE.md)
- **실험 수행**: [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) → 현재 문서(DEVELOPMENT.md)
- **API 사용**: [API_REFERENCE.md](API_REFERENCE.md)
- **코드 구조**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **개선 계획**: [TODO_ko.md](../TODO_ko.md)