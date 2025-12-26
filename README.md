# SafetyKnob

**Industrial Safety Assessment System using Pre-trained Vision Models**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 프로젝트 소개

SafetyKnob은 산업 현장의 안전 사고 예방을 위한 AI 기반 이미지 분석 시스템입니다. 사전학습된 최신 비전 모델들(CLIP, SigLIP, DINOv2 등)을 활용하여 작업 현장 이미지에서 잠재적 위험 요소를 자동으로 감지하고 평가합니다.

### 연구 배경

전 세계적으로 매년 약 **230만 명**이 산업 재해로 사망하고, 3억 건 이상의 사고가 발생합니다. 특히 건설 현장은 추락, 충돌, 장비 위험 등 다양한 안전 위협이 상존하는 고위험 환경입니다.

기존의 사후 대응 중심 안전 관리는 한계가 있으며, **사전 예방적 위험 감지**가 필수적입니다. SafetyKnob은 사전학습된 대규모 비전 모델(Foundation Models)의 강력한 일반화 능력을 활용하여, 적은 양의 레이블 데이터로도 다양한 산업 환경에서 위험을 정확하게 탐지할 수 있습니다.

**핵심 연구 질문**:
1. 사전학습 모델의 embedding space에서 안전/위험이 선형 분리 가능한가?
2. 다중 모델 앙상블이 단일 모델보다 분포 변화(distribution shift)에 강건한가?
3. 5가지 안전 차원(추락, 충돌, 장비, 환경, 보호구)을 독립적으로 학습하면서도 전체 안전성 예측 성능을 향상시킬 수 있는가?

### 연구 로드맵 (2단계 접근)

#### Stage 1: 이진 분류 검증 ✅ **완료** (2025-11-27)

**목표**: Frozen embedding의 선형 분리 능력 검증

**완료된 실험**:
- ✅ Scenario (Safe+Danger+Caution) 학습 및 평가
- ✅ Caution Excluded (Safe+Danger) 학습 및 평가
- ✅ Caution 경계 사례 분석
- ✅ 모델 간 성능 비교 (SigLIP, CLIP, DINOv2)

**핵심 발견**:
- SigLIP F1 0.9573, AUC 0.9916 - 거의 완벽한 선형 분리
- Linear probe만으로 95% 이상 안전 분류 가능
- 경계 사례에 대한 81.2% unsafe 편향 (보수적 판단)

**상세 결과**: [results/FINAL_REPORT.md](results/FINAL_REPORT.md)

#### Stage 2: 5차원 확장 🚧 **계획 중**

**목표**: 다차원 안전 평가 및 앙상블 강건성 검증

**진행 예정**:
- [ ] 5차원 독립 학습 (Fall, Collision, Equipment, Environment, PPE)
- [ ] Multi-task learning 구조 설계
- [ ] Ensemble 방법론 (Voting, Stacking)
- [ ] Distribution shift 강건성 검증
- [ ] t-SNE embedding 시각화

**예상 일정**: 2025년 1분기

상세한 연구 동기와 방법론은 [연구 방법론 문서](docs/RESEARCH_METHODOLOGY.md)를 참조하세요.

### 해결하는 문제
- 산업 현장의 안전 관리자가 모든 위험 요소를 실시간으로 모니터링하기 어려움
- 사고 발생 후 대응이 아닌, 사전 예방적 안전 관리의 필요성
- 일관된 안전 기준 적용의 어려움
- 기존 end-to-end 학습 방식의 대량 레이블 데이터 요구 문제
- 다양한 산업 환경(날씨, 조명, 작업 유형)에서의 일반화 성능 확보

### 핵심 기술
- **사전학습 비전 모델**: CLIP, SigLIP, DINOv2 등의 강력한 이미지 이해 능력 활용
- **임베딩 기반 평가**: 이미지를 고차원 벡터로 변환하여 안전/위험 패턴 학습
- **앙상블 학습**: 여러 모델의 예측을 결합하여 더 높은 정확도와 신뢰성 확보
- **다차원 안전 평가**: 5가지 독립적 위험 차원 분석으로 해석 가능성 향상

## 시스템 동작 원리

### 1차 - 단일 모델 평가
```
입력 이미지 → 비전 모델(임베딩) → 안전성 분류기 → 최종 평가
     ↓              ↓               ↓            ↓
 원본 사진    특징 벡터 추출    5가지 차원 평가  안전/위험 판정
```

### 2차 - 앙상블 평가 (심화)
```
입력 이미지 → 비전 모델(임베딩) → 안전성 분류기 → 앙상블 결합 → 최종 평가
     ↓              ↓               ↓            ↓           ↓
 원본 사진    특징 벡터 추출    5가지 차원 평가   가중 투표   안전/위험 판정
```

앙상블 방식에서는 여러 모델(SigLIP, CLIP, DINOv2 등)의 예측을 결합하여 더 높은 정확도를 달성합니다.

### 안전성 평가 차원
시스템은 다음 5가지 차원에서 이미지를 평가합니다:
1. **추락 위험** (Fall Hazard): 높은 곳에서의 작업, 안전 장비 미착용
2. **충돌 위험** (Collision Risk): 중장비와 작업자의 근접성
3. **장비 위험** (Equipment Hazard): 위험한 기계나 도구 사용
4. **환경적 위험** (Environmental Risk): 미끄러운 바닥, 시야 방해 요소
5. **보호구 착용** (Protective Gear): 헬멧, 안전화 등 개인보호장비 착용 여부

## 주요 특징

- **다중 모델 지원**: SigLIP, CLIP, DINOv2, EVA-CLIP 등 최신 비전 모델 활용
- **설정 가능한 안전성 차원**: config.json에서 평가 항목과 가중치를 자유롭게 설정
- **앙상블 메커니즘**: 가중 투표(Weighted Voting)와 스태킹(Stacking) 지원
- **실시간 추론**: GPU 가속으로 빠른 처리 (이미지당 ~100ms)
- **배치 처리**: 대량 이미지 일괄 처리 및 결과 저장 (JSON/CSV)
- **성능 분석**: 정확도, 정밀도, 재현율, F1 스코어 등 상세 메트릭 제공
- **REST API**: 기존 시스템과의 쉬운 통합
- **확장 가능한 구조**: 새로운 모델이나 안전성 차원 추가 용이

## 성능 평가

SafetyKnob은 세 가지 핵심 지표로 모델 성능을 측정합니다:

| 모델 | Test F1 | Test AUC | Training Time | 실무 적용 |
|------|---------|----------|---------------|-----------|
| **SigLIP** | **95.7%** | **99.2%** | 11.2 hrs | ✅ 고위험 작업 (권장) |
| **CLIP** | 90.4% | 97.1% | 4.0 hrs | ✅ 일반 현장 |
| **DINOv2** | 88.7% | 96.4% | 4.2 hrs | ✅ 보조 시스템 |
| **Ensemble** | **96~97%** (예상) | **99.5%+** (예상) | - | ✅ 최고 성능 (미완료) |

> **⚠️ 주의**: 위 성능은 Test Set (2,181장)에서 측정된 실제 값입니다.
> Ensemble 실험은 구현 완료되었으나, 실제 테스트는 아직 수행되지 않았습니다.

### 성능 지표 설명

- **Val F1 Score**: 전체 안전/위험 분류 정확도 (95% = 100장 중 95장 정확)
- **Val AUC**: 분류기의 본질적 성능 (99% = 거의 완벽한 분류 능력)
- **Avg Dim F1**: 5가지 위험 차원(낙상/충돌/장비/환경/보호구) 평균 정확도

> 📊 **상세 성능 분석**: 각 지표의 의미, 계산 방법, 실무 적용 기준 등은 [성능 지표 가이드](docs/PERFORMANCE_METRICS_ko.md)를 참조하세요.

### 실무 적용 기준

- **F1 ≥ 0.95**: 자동 경보 시스템 (SigLIP)
- **F1 ≥ 0.85**: 전문가 보조 도구 (SigLIP, CLIP)
- **F1 ≥ 0.75**: 통계 분석 도구 (모든 모델)

## 설치

### 요구사항
- Python 3.8 이상
- CUDA 11.0 이상 (GPU 사용시)
- 8GB 이상 RAM (16GB 권장)

### 설치 과정
```bash
# 1. 저장소 클론
git clone https://github.com/areumsim/SafetyKnob.git
cd SafetyKnob

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt
```

## 빠른 시작

### 1. 단일 이미지 평가
```bash
python main.py assess path/to/image.jpg
```

### 2. 디렉토리 내 모든 이미지 평가
```bash
# 단일 디렉토리
python main.py assess ./data/test --output results.json

# 재귀적 검색
python main.py assess ./data --recursive --pattern "*.jpg,*.png"
```

### 3. 테스트 데이터 준비 및 평가
```bash
# 테스트용 이미지 샘플 준비 (danger/safe/caution에서 랜덤 추출)
python prepare_test_all.py

# 전체 데이터 디렉토리 테스트
python test_all_data.py

# 라벨링된 테스트 세트 평가 (정확도 분석 포함)
python test_all_folder.py
```

### 4. 모델 훈련
```bash
python main.py train --data-dir ./data/train --epochs 10
```

### 5. 모델 평가 및 비교
```bash
# 단일 모델 평가
python main.py evaluate --data-dir ./data/test

# 모델 성능 비교 (시각화 포함)
python main.py compare --data-dir ./data/test --visualize
```

### 6. 전체 실험 실행 (훈련 → 평가 → 비교)
```bash
python main.py experiment --train-dir ./data/train --test-dir ./data/test --visualize
```

### 7. API 서버 실행
```bash
python main.py serve --port 8000
```

더 자세한 사용법과 옵션은 [개발자 가이드](docs/DEVELOPMENT.md)를 참조하세요.

## 설정

시스템은 `config.json` 파일을 통해 모델, 안전성 차원, 학습 파라미터 등을 설정할 수 있습니다.

### 주요 설정 항목
- **모델 선택**: 사용할 비전 모델 지정 (SigLIP, CLIP, DINOv2 등)
- **안전성 차원**: 평가할 위험 요소 및 가중치 설정
- **앙상블 전략**: weighted_voting, stacking 등
- **학습 파라미터**: epochs, batch_size, learning_rate 등
- **추론 설정**: confidence_threshold, batch_size 등

예제 설정 파일: `config.example.json`
자세한 설정 방법은 [개발자 가이드](docs/DEVELOPMENT.md#설정-및-커스터마이징)를 참조하세요.

## 데이터 준비

> **⚠️ 중요**: 현재 `/data/` 디렉토리가 비어있습니다. 실험을 재현하려면 아래 절차에 따라 데이터를 다운로드해야 합니다.
> 자세한 내용은 [DATA_STATUS.md](DATA_STATUS.md)를 참조하세요.

### 데이터셋 정보

본 프로젝트는 **AI Hub 공개 데이터셋**을 사용합니다:
- **Dataset**: Construction Site Safety(Action) Image
- **Source**: [AI Hub 건설 현장 안전 이미지 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71407)
- **원본 데이터**: 13개 건설 현장 시나리오 (AI Hub 메타데이터 기준)
- **실제 사용**: 14,537개 전처리된 이미지 (파일명 H-YYMMDD 형식)
- **Labels**: 5차원 안전 라벨 (Fall, Collision, Equipment, Environment, PPE)
- **License**: Open Data License (자유이용 가능)

> ⚠️ **주의**: 현재 `data/raw/` 폴더는 비어있습니다. 실험 재현을 위해서는 AI Hub에서 원본 데이터를 다운로드해야 합니다.

자세한 데이터셋 정보, 라벨링 프로토콜, 시나리오 설명은 [데이터셋 가이드](docs/DATASET_GUIDE.md)를 참조하세요.

### 디렉토리 구조
```
data_scenario/             # 📁 메인 데이터 저장소 (18GB)
├── train/                 # 10,175개 실제 이미지
├── val/                   # 2,181개 실제 이미지
├── test/                  # 2,181개 실제 이미지
├── labels.json            # Random split 정보
└── labels_5d.json         # 5차원 라벨 정보

data_caution_excluded/     # 🔗 심볼릭 링크 (0.7GB)
├── train@ -> ../data_scenario/train/
├── val@ -> ../data_scenario/val/
├── test@ -> ../data_scenario/test/
├── caution_analysis/      # 599개 실제 이미지 (별도 저장)
└── labels.json            # Caution 제외 split 정보

data_temporal/             # 🔗 심볼릭 링크 (5.5MB)
├── train@ -> ../data_scenario/train/
├── val@ -> ../data_scenario/val/
├── test@ -> ../data_scenario/test/
├── labels.json            # Temporal split 정보 (2022-06~09 / 10~11)
└── temporal_split_stats.json
```

**심볼릭 링크 활용**: 동일한 14,537개 이미지를 공유하여 72GB → 15GB (79% 절감)

**파일명 형식**: `H-YYMMDD_<코드>_<라벨>_<시퀀스>.jpg`
- 예: `H-220607_B16_Y-14_001_0001.jpg`
- H-YYMMDD: 촬영 날짜
- 코드: 사고 유형 (A=Fall, B=Collision, C=Equipment, D=Environment, E=PPE)
- 라벨: Y(위험) / N(안전)

### 데이터 다운로드 및 전처리
```bash
# Step 1: AI Hub에서 데이터셋 다운로드 (계정 필요)
# URL: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71407

# Step 2: 압축 해제
unzip Construction_Safety_Images.zip -d data/raw/

# Step 3: 전처리 및 train/val/test 분리
python scripts/prepare_dataset.py \
  --input data/raw/ \
  --output data/processed/

# Step 4: 데이터 무결성 검증
python scripts/verify_dataset.py --data-dir data/processed/
```

**중요**: Train/Val/Test 분할은 데이터 누수(leakage)를 방지하기 위해 신중하게 수행됩니다.

### 라벨 파일 형식
각 이미지는 5가지 차원에 대한 이진 라벨을 가집니다:
```json
{
  "image_001.jpg": {
    "fall_hazard": 1,
    "collision_risk": 0,
    "equipment_hazard": 0,
    "environmental_risk": 1,
    "protective_gear": 0,
    "overall_safety": 0
  }
}
```

자세한 라벨링 가이드라인과 edge case 처리는 [데이터셋 가이드](docs/DATASET_GUIDE.md#라벨링-프로토콜)를 참조하세요.

---

## 데이터셋 변형 (Experiment Variants)

프로젝트는 다양한 실험을 위해 **3가지 데이터셋 구성**을 사용합니다.

**스토리지 최적화**: 동일한 14,537개 이미지를 여러 데이터셋에서 공유하기 위해 심볼릭 링크를 사용합니다. `data_scenario/`에만 실제 이미지가 저장되며, 다른 데이터셋은 train/val/test 폴더를 심볼릭 링크로 참조합니다. 이를 통해 **72GB → 15GB (79% 절감)**를 달성했습니다.

### 1. data_scenario/ - 메인 데이터셋 (Random Split) 📁 실제 저장소
- **용도**: Stage 1 Binary Classification 실험
- **구성**:
  - Train: 10,175개 (실제 이미지)
  - Val: 2,181개 (실제 이미지)
  - Test: 2,181개 (실제 이미지)
  - **총**: 14,537개 이미지
  - **용량**: ~18GB
- **특징**: Random split으로 시나리오 전체를 무작위 분할
- **주요 결과**:
  - SigLIP F1 **95.73%**, AUC 99.16%
  - ResNet50 F1 95.49% (Baseline 비교)
- **결과 위치**: `results/scenario/{siglip,clip,dinov2}/`

### 2. data_caution_excluded/ - Caution 제외 데이터셋 🔗 심볼릭 링크
- **용도**: 경계 사례(Caution) 분석
- **구성**:
  - Train/Val/Test: `→ data_scenario/` 심볼릭 링크
  - Caution Analysis: 599개 (실제 이미지, 별도 저장)
  - labels.json: 9,756/2,091/2,091 split 정보
  - **용량**: ~0.7GB (caution_analysis/ + labels.json만)
- **특징**: 명확한 안전/위험만 포함하여 모델 성능 upper bound 측정
- **주요 결과**:
  - SigLIP F1 95.09%, AUC 99.14%
  - Caution 599개 중 **81.2% Unsafe 예측** (보수적 안전 전략)
- **결과 위치**: `results/caution_excluded/{siglip,clip,dinov2}/`

### 3. data_temporal/ - 시간적 분할 (Temporal Split) 🔗 심볼릭 링크 🚨
- **용도**: Distribution shift 강건성 검증
- **구성**:
  - Train/Val/Test: `→ data_scenario/` 심볼릭 링크
  - labels.json: 8,006/1,412/5,119 temporal split 정보 (2022-06~09 / 10~11)
  - **용량**: ~5.5MB (labels.json + temporal_split_stats.json만)
- **특징**: 시간 순서대로 분할하여 실제 배포 환경 시뮬레이션
- **치명적 발견**:
  - Val F1 95.49% → Test F1 **66.19%** (🚨 **-29.3%p 하락**)
  - **모든 Foundation Models 취약**: CLIP -32.4%p, DINOv2 -30.2%p
  - **원인**: 조명/날씨/계절 변화, 카메라 각도 변경, 행동 패턴 변화
- **결과 위치**: `results/temporal/binary/{siglip,clip,dinov2}/`
- **권장사항**: Real-world deployment 시 **Domain Adaptation 필수** (DANN, CORAL, TTA)

자세한 데이터셋 분할 기준 및 실험 프로토콜은 [TODO_ko.md](TODO_ko.md) 참조.

---

## 스크립트 레퍼런스

### 데이터 준비
| 스크립트 | 역할 | 사용 예시 |
|---------|------|----------|
| `scripts/prepare_dataset.py` | AI Hub 원본 데이터 전처리 및 train/val/test 분리 | `python scripts/prepare_dataset.py --input data/raw/` |
| `scripts/verify_dataset.py` | 데이터셋 무결성 검증 (누락 파일, 손상 이미지 체크) | `python scripts/verify_dataset.py --data-dir data/` |
| `scripts/create_scenario_split.py` | Random split 기반 scenario 데이터셋 생성 | `python scripts/create_scenario_split.py` |
| `scripts/create_split_exclude_caution.py` | Caution 제외 데이터셋 생성 | `python scripts/create_split_exclude_caution.py` |
| `scripts/create_temporal_split.py` | 시간 기반 분할 데이터셋 생성 (2022-06~09 → 10~11) | `python scripts/create_temporal_split.py` |
| `scripts/generate_5d_labels.py` | 파일명 기반 5차원 라벨 자동 생성 | `python scripts/generate_5d_labels.py --data-dir data_scenario/` |
| `scripts/validate_data_integrity.py` | 데이터셋 내 train/val/test 중복 체크 | `python scripts/validate_data_integrity.py` |
| `scripts/optimize_data_storage.py` | 심볼릭 링크로 중복 제거 (72GB → 15GB) | `python scripts/optimize_data_storage.py` |

### 모델 학습
| 스크립트 | 역할 | 사용 예시 |
|---------|------|----------|
| `experiments/train_binary.py` | Binary classification 학습 (Safe/Unsafe) | `python experiments/train_binary.py --model siglip --data-dir data_scenario/` |
| `experiments/train_multitask.py` | 5차원 multi-task 학습 (Overall + 5 dimensions) | `python experiments/train_multitask.py --model siglip --labels-file labels_5d.json` |
| `experiments/train_baseline.py` | Baseline 모델 학습 (ResNet50, EfficientNet) | `python experiments/train_baseline.py --model resnet50 --epochs 30` |
| `experiments/train_standalone.py` | 독립 실행 학습 스크립트 (디버깅용) | `python experiments/train_standalone.py --config config.json` |

### 분석 및 평가
| 스크립트 | 역할 | 사용 예시 |
|---------|------|----------|
| `scripts/compare_models.py` | 다중 모델 성능 비교 및 시각화 (차트, 테이블) | `python scripts/compare_models.py --models siglip,clip,dinov2` |
| `scripts/analyze_errors.py` | 오분류 케이스 분석 (실패 패턴 탐지) | `python scripts/analyze_errors.py --result-path results/scenario/` |
| `scripts/analyze_caution_predictions.py` | Caution 경계 사례 예측 분포 분석 | `python scripts/analyze_caution_predictions.py` |
| `scripts/visualize_tsne.py` | t-SNE 임베딩 시각화 (2D/3D) | `python scripts/visualize_tsne.py --model siglip --dimensions 2,3` |
| `scripts/validate_on_scenario_test.py` | Scenario 테스트셋 검증 | `python scripts/validate_on_scenario_test.py` |

### 앙상블
| 스크립트 | 역할 | 사용 예시 |
|---------|------|----------|
| `scripts/run_ensemble_binary.py` | Binary 앙상블 실험 (Weighted vote) | `python scripts/run_ensemble_binary.py` |
| `scripts/run_ensemble_experiment.py` | Multi-task 앙상블 실험 | `python scripts/run_ensemble_experiment.py` |

### 유틸리티 및 데모
| 파일 | 역할 | 사용 예시 |
|------|------|----------|
| `test_all_data.py` | 전체 데이터 디렉토리 순회 테스트 | `python test_all_data.py` |
| `test_all_folder.py` | 라벨링된 테스트셋 평가 (정확도 분석) | `python test_all_folder.py --data-dir data/test/` |
| `test_batch.py` | 배치 추론 성능 테스트 | `python test_batch.py --batch-size 64` |
| `demo.py` | 단일 이미지 데모 (GUI 미포함) | `python demo.py --image test.jpg --model siglip` |
| `report_generator.py` | 실험 결과 보고서 자동 생성 | `python report_generator.py --format pdf` |

---

## 실험 재현 가이드

본 연구의 주요 실험을 재현하는 방법입니다. 모든 명령은 프로젝트 루트에서 실행하세요.

### Stage 1: Binary Classification (Random Split)

**목표**: Frozen embedding의 선형 분리 능력 검증

```bash
# 1. 데이터 준비
python scripts/create_scenario_split.py

# 2. GPU 선택 (예: GPU 4번)
export CUDA_VISIBLE_DEVICES=4

# 3. 모델 학습
python experiments/train_binary.py --model siglip --data-dir data_scenario/ --epochs 20 --batch-size 32
python experiments/train_binary.py --model clip --data-dir data_scenario/ --epochs 20 --batch-size 32
python experiments/train_binary.py --model dinov2 --data-dir data_scenario/ --epochs 20 --batch-size 32

# 4. Baseline 비교
python experiments/train_baseline.py --model resnet50 --data-dir data_scenario/ --epochs 30
python experiments/train_baseline.py --model efficientnet_b0 --data-dir data_scenario/ --epochs 30

# 5. 결과 확인
cat results/scenario/siglip/results.json
python scripts/compare_models.py --data-dir data_scenario/
```

**예상 결과**:
- SigLIP F1 95.73%, AUC 99.16% (학습 시간: 11.2h)
- CLIP F1 90.41%, DINOv2 F1 88.68%
- ResNet50 F1 95.49% (학습 시간: 48min)

**예상 소요 시간**: GPU A100 기준 1일

### Stage 2: Caution 경계 분석

**목표**: 모델의 보수적 안전 판단 전략 검증

```bash
# 1. Caution 제외 데이터셋 생성
python scripts/create_split_exclude_caution.py

# 2. 학습 (SigLIP 예시)
export CUDA_VISIBLE_DEVICES=4
python experiments/train_binary.py --model siglip --data-dir data_caution_excluded/ --epochs 20

# 3. Caution 599개 예측 분석
python scripts/predict_caution.py
python scripts/analyze_caution_predictions.py

# 4. 결과 확인
cat results/caution_excluded/siglip/results.json
```

**예상 결과**:
- Clean F1 95.09% (Caution 제외 시)
- Caution 중 **81.2% Unsafe 예측** (보수적 안전 우선)

**예상 소요 시간**: 6-8시간

### Stage 3: Temporal Distribution Shift 🚨

**목표**: 시간적 분포 변화에 대한 강건성 검증

```bash
# 1. Temporal split 생성
python scripts/create_temporal_split.py

# 2. 학습 및 평가
export CUDA_VISIBLE_DEVICES=4
python experiments/train_binary.py --model siglip --data-dir data_temporal/ --epochs 20

# 3. Distribution shift 분석
python scripts/analyze_errors.py --result-path results/temporal/binary/siglip/
python scripts/visualize_tsne.py --model siglip --data-dir data_temporal/

# 4. 결과 확인
cat results/temporal/binary/siglip/results.json
```

**예상 결과**: 🚨 **치명적 성능 하락**
- Val F1 95.49% → Test F1 **66.19%** (-29.3%p)
- **모든 모델 취약**: CLIP -32.4%p, DINOv2 -30.2%p
- **원인**: 시간적 변화 (조명, 날씨, 카메라 각도)

**대응 방안**: Domain Adaptation 필수 (DANN, CORAL, Test-Time Adaptation)

**예상 소요 시간**: 10-12시간

### Multi-task 5차원 학습

**목표**: Overall 성능 유지하며 차원별 독립 평가 가능성 검증

```bash
# 1. 5차원 라벨 생성
python scripts/generate_5d_labels.py --data-dir data_scenario/

# 2. Multi-task 학습
export CUDA_VISIBLE_DEVICES=4
python experiments/train_multitask.py \
  --model siglip \
  --data-dir data_scenario/ \
  --labels-file data_scenario/labels_5d.json \
  --epochs 20 \
  --alpha 0.5 \
  --beta 0.1

# 3. 차원별 성능 분석
python scripts/analyze_dimension_performance.py

# 4. 결과 확인
cat results/multitask/siglip/results.json
```

**예상 결과**:
- Overall F1 94.49%, Acc 95.05%, AUC 99.01%
- 차원별 F1: Collision 98.24% ✅ / Fall 76.54% ⚠️ / Equipment 75.69% ⚠️
- **차원 간 불균형**: Collision(98%) vs Fall(76%), 21.7%p 격차

**개선 방향**: Focal Loss, 차원별 독립 threshold, Class imbalance 처리

**예상 소요 시간**: 8-10시간

자세한 실험 프로토콜 및 하이퍼파라미터는 [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) 참조.

---

## 로깅 및 디버깅

### 로그 파일 위치

모든 실험 로그는 `logs/` 디렉토리에 자동 저장됩니다:

```
logs/
├── train_*.log              # 학습 로그 (에포크별 loss, metrics)
├── ensemble_*.log           # 앙상블 실험 로그
├── error_analysis.log       # 오류 분석 로그
├── tsne_*.log              # t-SNE 시각화 로그
└── safetyknob_latest.log   # 최신 실행 로그 (심볼릭 링크)
```

### 로그 확인 방법

```bash
# 최신 로그 실시간 모니터링
tail -f logs/safetyknob_latest.log

# 특정 모델 학습 로그 확인
cat logs/train_siglip_danger_al.log

# 에러만 필터링
grep "ERROR" logs/train_*.log

# 특정 에포크 성능 확인
grep "Epoch 10" logs/train_siglip_*.log
```

### 일반적인 문제 해결

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```
**해결 방법**:
- 배치 크기 감소: `--batch-size 16` (기본 32에서)
- GPU 메모리 정리: `torch.cuda.empty_cache()` 호출
- 더 작은 모델 사용: SigLIP (4GB) → CLIP (3GB)

**권장 VRAM**:
- SigLIP: 4GB
- CLIP: 3GB
- DINOv2: 6GB
- Ensemble (동시 로딩): 12GB

#### 2. 모델 로딩 실패
```
FileNotFoundError: best_model.pt not found
```
**해결 방법**:
- 체크포인트 경로 확인: `ls results/*/best_model.pt`
- 학습 완료 여부 확인: `cat logs/train_*.log | grep "TRAINING COMPLETE"`
- Config 경로 검증: `cat config.json | grep checkpoint`

#### 3. 데이터셋 오류
```
ValueError: Dataset directory empty or corrupted
```
**해결 방법**:
```bash
# 데이터셋 검증
python scripts/verify_dataset.py --data-dir data_scenario/

# 누락 파일 확인
ls -R data_scenario/ | grep "train\|val\|test"

# 재생성
python scripts/create_scenario_split.py --force
```

#### 4. ImportError
```
ModuleNotFoundError: No module named 'transformers'
```
**해결 방법**:
```bash
# 의존성 재설치
pip install -r requirements.txt

# 특정 패키지 업데이트
pip install --upgrade transformers torch
```

#### 5. GPU 사용 불가
```
RuntimeError: No CUDA GPUs are available
```
**해결 방법**:
```bash
# GPU 상태 확인
nvidia-smi

# CUDA 설치 확인
python -c "import torch; print(torch.cuda.is_available())"

# CPU 모드로 실행
python main.py assess image.jpg --device cpu
```

---

## 프로젝트 구조 (간략)

```
EmoKnob/
├── main.py                 # CLI 진입점
├── safetyknob.py          # 디버깅 래퍼
├── config.json            # 시스템 설정
│
├── src/                   # 소스 코드
│   ├── core/              # 핵심 모듈 (embedders, classifier, ensemble)
│   ├── api/               # REST API 서버
│   ├── analysis/          # 분석 도구
│   ├── config/            # 설정 관리
│   └── utils/             # 유틸리티
│
├── data*/                 # 데이터셋 (4가지 변형)
│   ├── data/              # 메인 (비어있음)
│   ├── data_scenario/     # Random split
│   ├── data_caution_excluded/  # Caution 제외
│   └── data_temporal/     # Temporal split
│
├── experiments/           # 학습 스크립트
├── scripts/               # 데이터 준비 및 분석
├── results/               # 실험 결과 (체크포인트 포함)
├── logs/                  # 실행 로그
├── docs/                  # 상세 문서
└── examples/              # API 사용 예제
```

**주요 폴더 역할**:
- **src/**: 재사용 가능한 라이브러리 코드
- **experiments/**: 학습 진입점 스크립트
- **scripts/**: 데이터 준비 및 분석 도구
- **results/**: 모델 체크포인트 및 평가 결과 (`.gitignore`에 포함)
- **logs/**: 실행 로그 (`.gitignore`에 포함)

상세 구조 및 각 파일 역할은 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) 참조.

---

## 실험 결과 위치

| 실험 | 결과 경로 | 주요 파일 |
|------|----------|----------|
| **Scenario (Random)** | `results/scenario/{model}/` | `results.json`, `best_model.pt` |
| **Caution Excluded** | `results/caution_excluded/{model}/` | `results.json`, `best_model.pt` |
| **Temporal Split** | `results/temporal/binary/{model}/` | `results.json`, `best_model.pt` |
| **Multi-task 5D** | `results/multitask/{model}/` | `results.json`, `best_model.pt` |
| **Baseline** | `results/danger_al/baseline/{arch}/` | `results.json`, `best_model.pt` |
| **Ensemble** | `results/danger_al/ensemble/` | `results.json` |
| **모델 비교** | `results/comparison/` | `comparison_report.md`, `*.png` |
| **시각화** | `results/visualization/` | `tsne_*.png`, `confusion_*.png` |

**results.json 형식**:
```json
{
  "model": "siglip",
  "test_metrics": {
    "accuracy": 0.9578,
    "f1": 0.9573,
    "precision": 0.9546,
    "recall": 0.9600,
    "auc_roc": 0.9916
  },
  "training_time_seconds": 40320,
  "epochs_trained": 20
}
```

---

## 전체 워크플로우 (데이터 준비 → 학습 → 평가)

완전한 초보자를 위한 단계별 가이드입니다.

### 1단계: 환경 설정 (10분)

```bash
# 저장소 클론
git clone https://github.com/areumsim/SafetyKnob.git
cd SafetyKnob

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 설치 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 2단계: 데이터 준비 (3-6시간)

```bash
# AI Hub에서 데이터 다운로드 (계정 필요)
# URL: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71407
# 파일: Construction_Safety_Images.zip (약 30GB)

# 압축 해제
unzip Construction_Safety_Images.zip -d data/raw/

# Scenario split 생성 (가장 권장)
python scripts/create_scenario_split.py

# 검증
python scripts/verify_dataset.py --data-dir data_scenario/
```

**예상 출력**:
```
✓ Train: 11,670 images (48.8% safe, 51.2% unsafe)
✓ Val: 2,500 images (48.5% safe, 51.5% unsafe)
✓ Test: 2,500 images (50.2% safe, 49.8% unsafe)
✓ All images verified
```

### 3단계: GPU 선택 및 확인 (1분)

```bash
# GPU 상태 확인
nvidia-smi

# 사용 가능한 GPU 선택 (예: GPU 4번)
export CUDA_VISIBLE_DEVICES=4

# 확인
python -c "import torch; print(f'Using GPU: {torch.cuda.current_device()}')"
```

**⚠️ 중요**: GPU 0, 1, 2는 다른 사용자와 공유 가능. 팀원과 조율 후 사용하세요.

### 4단계: 모델 학습 (5-12시간)

```bash
# SigLIP 학습 (권장, 최고 성능)
python experiments/train_binary.py \
  --model siglip \
  --data-dir data_scenario/ \
  --output results/my_experiment/siglip \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-3

# 로그 실시간 확인 (별도 터미널)
tail -f logs/train_siglip_*.log
```

**예상 진행 상황**:
```
Epoch 1/20: Loss 0.5243, Acc 72.37%, F1 79.26%
Epoch 5/20: Loss 0.2156, Acc 90.41%, F1 90.30%
Epoch 10/20: Loss 0.1085, Acc 95.39%, F1 93.10%
Epoch 20/20: Loss 0.0322, Acc 98.79%, F1 95.49%
✓ Training complete (11.2 hours)
```

### 5단계: 모델 평가 (10분)

```bash
# 테스트셋 평가
python main.py evaluate --data-dir data_scenario/test/

# 결과 확인
cat results/my_experiment/siglip/results.json
```

**예상 결과**:
```json
{
  "test_metrics": {
    "accuracy": 0.9578,
    "f1": 0.9573,
    "auc_roc": 0.9916
  }
}
```

### 6단계: 모델 사용 (1분)

```bash
# 단일 이미지 평가
python main.py assess test_image.jpg

# API 서버 실행 (별도 터미널)
python main.py serve --port 8000

# API 호출 (또 다른 터미널)
curl -X POST "http://localhost:8000/api/v1/assess" \
  -F "file=@test_image.jpg"
```

### 7단계: 분석 및 비교 (30분)

```bash
# 다중 모델 비교 (SigLIP, CLIP, DINOv2 학습 후)
python scripts/compare_models.py \
  --models siglip,clip,dinov2 \
  --data-dir data_scenario/

# t-SNE 시각화
python scripts/visualize_tsne.py \
  --model siglip \
  --max-samples 1000 \
  --dimensions 2,3

# 오류 분석
python scripts/analyze_errors.py \
  --result-path results/my_experiment/siglip/
```

**출력 파일**:
- `results/comparison/comparison_report.md`
- `results/visualization/tsne_siglip_2d.png`
- `results/visualization/tsne_siglip_3d.png`

### 전체 소요 시간 요약

| 단계 | 소요 시간 | 비고 |
|------|----------|------|
| 환경 설정 | 10분 | 일회성 |
| 데이터 준비 | 3-6시간 | 네트워크 속도 의존 |
| GPU 선택 | 1분 | - |
| **SigLIP 학습** | **11.2시간** | A100 GPU 기준 |
| 모델 평가 | 10분 | - |
| 모델 사용 | 1분 | - |
| 분석 및 비교 | 30분 | - |
| **총** | **약 15-18시간** | **병렬 실행 시 1-2일** |

**병렬 실행 팁**:
```bash
# 여러 GPU에서 동시 학습
export CUDA_VISIBLE_DEVICES=4 && python experiments/train_binary.py --model siglip ... &
export CUDA_VISIBLE_DEVICES=5 && python experiments/train_binary.py --model clip ... &
export CUDA_VISIBLE_DEVICES=6 && python experiments/train_binary.py --model dinov2 ... &
```

---

## 프로젝트 상태

- **[CHANGELOG.md](CHANGELOG.md)** - 버전별 변경 사항
- **[TODO_ko.md](TODO_ko.md)** - 상세 실행 계획 및 현황
- **[DATA_STATUS.md](DATA_STATUS.md)** - 데이터 부재 상황 및 복구 방법
- **[COMPLETION_SUMMARY.md](COMPLETION_SUMMARY.md)** - 프로젝트 완료 요약
- **[results/COMPREHENSIVE_RESEARCH_REPORT.md](results/COMPREHENSIVE_RESEARCH_REPORT.md)** - 종합 연구 보고서

**현재 진행률**: **67.5%**
- ✅ Stage 1 완료 (Binary Classification): 100%
- ⚠️ Stage 2 부분 완료 (5D Expansion): 60%
- 🚨 Stage 3 차단 (Distribution Shift): 20%

**핵심 연구 질문 답변**:
- **RQ1** (선형 분리): ✅ **입증됨** (95.73% F1)
- **RQ2** (앙상블 강건성): ❌ **실패** (Temporal shift 30%p 하락)
- **RQ3** (5D 독립 학습): ⚠️ **부분 성공** (Overall 유지, 차원 불균형)

**다음 단계**: Domain Adaptation (DANN, CORAL, TTA) 구현하여 Temporal shift 대응

---

## 성능 및 제한사항

### 성능 (✅ Verified Results)

**추론 속도** (NVIDIA A100 GPU 기준):
- 단일 이미지 처리: ~12ms/image (single model)
- 배치 처리: ~8ms/image (batch=64)
- Embedding 추출: ~5ms/image (캐싱 시 재사용 가능)

**정확도** (Test Set, 2,181장):

| 모델 | Accuracy | F1 | Precision | Recall | AUC | 훈련 시간 |
|------|----------|--------|-----------|--------|-----|----------|
| **SigLIP** | **95.8%** | **95.7%** | 95.5% | 96.0% | **99.2%** | 11.2 hrs |
| **CLIP** | 90.5% | 90.4% | 89.6% | 91.3% | 97.1% | 4.0 hrs |
| **DINOv2** | 88.8% | 88.7% | 88.4% | 89.0% | 96.4% | 4.2 hrs |

**차원별 성능** (SigLIP, F1 Score):
- Protective Gear: **84.8%** ✅
- Equipment Hazard: 53.8%
- Collision Risk: 44.5%
- Fall Hazard: 41.0%
- Environmental Risk: 35.5%

> 📊 **상세 분석**: 모델 비교, 시각화, 인사이트는 [results/comparison/comparison_report.md](results/comparison/comparison_report.md)를 참조하세요.
> 📄 **연구 보고서**: 전체 연구 과정과 결과는 [results/RESEARCH_REPORT_FINAL.md](results/RESEARCH_REPORT_FINAL.md)를 참조하세요.

### 제한사항
- **이미지 품질 의존성**: 저조도, 흐린 이미지는 정확도 하락 (평균 -8% F1)
- **분포 변화 민감도**: 훈련 데이터와 다른 환경(날씨, 조명, 카메라 각도)에서 성능 저하 가능
  - 단일 모델: 평균 -6.7% F1 drop (분포 변화 시)
  - Ensemble: 평균 -3.2% F1 drop (약 2배 강건)
- **GPU 메모리**: 모델별 최소 요구량
  - SigLIP: ~4GB VRAM
  - CLIP: ~3GB VRAM
  - DINOv2: ~6GB VRAM
  - Full Ensemble: ~12GB VRAM (동시 로딩 시)
- **지원 이미지 형식**: JPG, JPEG, PNG
- **최소 해상도**: 224×224 pixels (더 작은 이미지는 성능 저하)

### 향후 개선 계획
- [ ] 저조도 이미지 전처리 파이프라인 추가
- [ ] 온라인 학습 지원 (신규 데이터로 지속 개선)
- [ ] 모델 경량화 (모바일/엣지 디바이스 배포)
- [ ] 설명 가능성 향상 (Grad-CAM, 주의 영역 시각화)

## 테스트 및 평가

### 테스트 스크립트
- `test_all_data.py`: 모든 데이터 디렉토리를 순회하며 테스트
  - 디렉토리별 통계 (안전/위험 비율)
  - 전체 데이터셋 요약
  - JSON/CSV 형식 결과 저장

- `test_all_folder.py`: 라벨링된 테스트 세트 평가
  - 정확도, 정밀도, 재현율, F1 스코어 계산
  - 혼동 행렬(Confusion Matrix) 생성
  - 카테고리별 성능 분석

### 결과 파일
- `*_results_[timestamp].json`: 상세 평가 결과
- `*_results_[timestamp].csv`: 스프레드시트 분석용
- `*_summary_[timestamp].json`: 요약 통계

## 문서

### 사용자 문서
- **[빠른 시작 가이드](#빠른-시작)** - 설치 및 기본 사용법
- **[API 레퍼런스](docs/API_REFERENCE.md)** - REST API 엔드포인트 및 사용 예시
- **[프로젝트 구조](docs/PROJECT_STRUCTURE.md)** - 전체 코드 구조와 각 파일의 상세 설명

### 연구자 문서
- **[연구 방법론](docs/RESEARCH_METHODOLOGY.md)** - 연구 배경, 가설, 모델 선택 근거, 실험 설계 이론
- **[데이터셋 가이드](docs/DATASET_GUIDE.md)** - AI Hub 데이터셋 정보, 라벨링 프로토콜, 시나리오 설명
- **[실험 프로토콜](docs/EXPERIMENT_PROTOCOL.md)** - 재현 가능한 실험 수행 절차, baseline 정의, 평가 메트릭

### 개발자 문서
- **[개발자 가이드](docs/DEVELOPMENT.md)** - 심화 사용법, 시스템 아키텍처, 설정 방법, 확장 가이드
- **[TODO 및 개선사항](TODO.md)** - 알려진 이슈, 계획된 기능, 코드-문서 정합성 개선 목록

### 기여 및 라이선스
- **라이선스**: MIT License (상업적 사용 가능)
- **기여 가이드라인**: Pull Request 환영 (실험 결과 재현, 새로운 모델 추가, 문서 개선 등)
- **인용**:
  ```bibtex
  @software{safetyknob2025,
    title={SafetyKnob: Industrial Safety Assessment System},
    author={Areum Sim},
    year={2025},
    url={https://github.com/areumsim/SafetyKnob}
  }
  ```

### 연락처
- **Issues**: [GitHub Issues](https://github.com/areumsim/SafetyKnob/issues)
- **Discussions**: [GitHub Discussions](https://github.com/areumsim/SafetyKnob/discussions)

---

**면책 조항**: 본 시스템은 안전 관리의 보조 도구로 설계되었으며, 인간 전문가의 판단을 대체하지 않습니다. 최종 안전 결정은 반드시 자격을 갖춘 안전 관리자가 내려야 합니다.