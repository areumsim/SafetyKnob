# SafetyKnob - Industrial Image Safety Assessment System

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

| 모델 | Test F1 | Test AUC | Avg Dim F1 | Training Time | 실무 적용 |
|------|---------|----------|------------|---------------|-----------|
| **SigLIP** | **95.5%** | **99.1%** | 51.9% | 5.1 hrs | ✅ 고위험 작업 (권장) |
| **CLIP** | 87.4% | 94.6% | 47.3% | 2.9 hrs | ✅ 일반 현장 |
| **DINOv2** | 86.0% | 94.3% | 42.3% | 3.1 hrs | ✅ 보조 시스템 |
| **Ensemble** | **96~97%** (예상) | **99.5%+** (예상) | **55~60%** (예상) | - | ✅ 최고 성능 (미완료) |

> **⚠️ 주의**: 위 성능은 Test Set (SO-47 시나리오, 1,737장)에서 측정된 실제 값입니다.
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
git clone https://github.com/yourusername/safetyknob.git
cd safetyknob

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
- **Size**: 11,583 labeled images
- **Scenarios**: 13 construction scenarios (SO-35 to SO-47)
- **Labels**: 5-dimensional safety annotations per image
- **License**: Open Data License (자유이용 가능)

자세한 데이터셋 정보, 라벨링 프로토콜, 시나리오 설명은 [데이터셋 가이드](docs/DATASET_GUIDE.md)를 참조하세요.

### 디렉토리 구조
```
data/
├── raw/                    # AI Hub에서 다운로드한 원본 데이터
│   ├── SO-35/             # 비계 조립 (Scaffold Assembly)
│   ├── SO-41/             # 굴착기 작업 (Excavator Operation)
│   └── ...                # 기타 시나리오 (SO-36~SO-47)
├── processed/             # 전처리된 데이터 (train/val/test 분리)
│   ├── train/             # 훈련 세트 (7,128 images, SO-35~SO-43)
│   ├── val/               # 검증 세트 (2,872 images, SO-44~SO-46)
│   ├── test/              # 테스트 세트 (1,583 images, SO-47)
│   └── labels.json        # 5차원 라벨 정보
├── test_all/              # 통합 테스트용 (prepare_test_all.py로 생성)
└── embeddings/            # 사전 추출된 embedding (optional, 캐싱용)
    ├── siglip/
    ├── clip/
    └── dinov2/
```

### 데이터 다운로드 및 전처리
```bash
# Step 1: AI Hub에서 데이터셋 다운로드 (계정 필요)
# URL: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71407

# Step 2: 압축 해제
unzip Construction_Safety_Images.zip -d data/raw/

# Step 3: 전처리 및 train/val/test 분리
python scripts/prepare_dataset.py \
  --input data/raw/ \
  --output data/processed/ \
  --train-scenarios SO-35,SO-36,SO-37,SO-38,SO-41,SO-42,SO-43 \
  --val-scenarios SO-44,SO-45,SO-46 \
  --test-scenarios SO-47

# Step 4: 데이터 무결성 검증
python scripts/verify_dataset.py --data-dir data/processed/
```

**중요**: Train/Val/Test 분할은 **시나리오 단위**로 수행되어 데이터 누수(leakage)를 방지합니다. 동일 시나리오 내 이미지는 하나의 세트에만 포함됩니다.

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

## 성능 및 제한사항

### 성능 (✅ Verified Results)

**추론 속도** (NVIDIA A100 GPU 기준):
- 단일 이미지 처리: ~12ms/image (single model)
- 배치 처리: ~8ms/image (batch=64)
- Embedding 추출: ~5ms/image (캐싱 시 재사용 가능)

**정확도** (Test Set SO-47, 1,737장):

| 모델 | Accuracy | F1 | Precision | Recall | AUC | 훈련 시간 |
|------|----------|--------|-----------|--------|-----|----------|
| **SigLIP** | **95.3%** | **95.5%** | 94.4% | 96.6% | **99.1%** | 5.1 hrs |
| **CLIP** | 87.2% | 87.4% | 88.4% | 86.5% | 94.6% | 2.9 hrs |
| **DINOv2** | 85.7% | 86.0% | 86.0% | 86.0% | 94.3% | 3.1 hrs |

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
    title={SafetyKnob: Industrial Image Safety Assessment System},
    author={Your Team},
    year={2025},
    url={https://github.com/yourusername/safetyknob}
  }
  ```

### 연락처
- **Issues**: [GitHub Issues](https://github.com/yourusername/safetyknob/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/safetyknob/discussions)

---

**면책 조항**: 본 시스템은 안전 관리의 보조 도구로 설계되었으며, 인간 전문가의 판단을 대체하지 않습니다. 최종 안전 결정은 반드시 자격을 갖춘 안전 관리자가 내려야 합니다.