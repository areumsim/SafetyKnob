# SafetyKnob 프로젝트 구조 및 파일 설명

## 프로젝트 개요
SafetyKnob은 산업 현장 이미지의 안전성을 평가하는 AI 시스템입니다. 사전학습된 비전 모델(CLIP, SigLIP, DINOv2 등)을 활용하여 이미지에서 안전/위험 요소를 자동으로 분석합니다.

## 디렉토리 구조
```
safetyknob/
├── main.py                # 메인 진입점 (CLI 인터페이스)
├── safetyknob.py             # 디버깅/프로파일링 래퍼
├── setup.py               # 패키지 설치 설정
├── pyproject.toml         # 현대적 Python 패키지 설정
├── requirements.txt       # 프로젝트 의존성
├── config.json            # 시스템 설정 파일
├── config.example.json    # 설정 파일 예시
├── report_generator.py    # 보고서 생성 유틸리티
├── README.md              # 프로젝트 소개 및 사용법
├── TODO_ko.md             # 개선 계획 및 알려진 이슈 (한국어)
│
├── docs/                  # 문서
│   ├── README.md → ../README.md
│   ├── RESEARCH_METHODOLOGY.md    # 연구 방법론 (가설, 모델 선택 근거)
│   ├── RESEARCH_METHODOLOGY_ko.md # 연구 방법론 (한국어)
│   ├── DATASET_GUIDE.md           # 데이터셋 가이드 (AI Hub)
│   ├── EXPERIMENT_PROTOCOL.md     # 실험 프로토콜 (재현성 보장)
│   ├── DEVELOPMENT.md             # 개발자 가이드
│   ├── PROJECT_STRUCTURE.md       # 이 파일
│   └── API_REFERENCE.md           # REST API 문서
│
├── scripts/               # 유틸리티 및 분석 스크립트
│   ├── prepare_dataset.py              # 데이터 전처리
│   ├── create_scenario_split.py        # Scenario (Random) split 생성
│   ├── create_temporal_split.py        # Temporal split 생성
│   ├── create_split_exclude_caution.py # Caution 제외 split
│   ├── generate_5d_labels.py           # 5차원 안전 라벨 생성
│   ├── extract_embeddings.py           # SigLIP/CLIP/DINOv2 임베딩 추출
│   ├── verify_dataset.py              # 데이터 무결성 검증
│   ├── run_multiseed.py               # Multi-seed 실험 자동화
│   ├── run_ensemble_ablation.py       # 앙상블 조합 ablation
│   ├── run_scaling_curve.py           # 데이터 스케일링 곡선
│   ├── run_full_experiment.sh         # 전체 실험 재현 스크립트 (12단계)
│   ├── statistical_tests.py           # 통계적 유의성 검정
│   ├── generate_paper_figures.py      # 논문용 Figure 9개 생성
│   ├── analyze_temporal_shift_cause.py # Temporal shift 근본원인 분석
│   ├── analyze_errors.py              # 오류 패턴 분석
│   ├── compare_experiments.py         # 실험 비교
│   └── visualize_tsne.py              # t-SNE 시각화
│
├── experiments/           # 실험 스크립트
│   ├── train_from_embeddings.py  # 캐싱 임베딩 기반 probe 학습
│   ├── train_standalone.py       # 독립형 학습
│   ├── train_binary.py           # Binary classification 학습
│   ├── train_baseline.py         # CNN baseline (ResNet50, EfficientNet)
│   ├── train_dann.py             # DANN domain adaptation
│   └── train_multitask.py        # Multi-task 5차원 학습
│
├── data/                  # 데이터셋
│   ├── raw/                      # AI Hub 원본 데이터
│   ├── processed/                # 전처리된 데이터 (train/val/test)
│   └── cache/                    # Embedding 캐시
│
├── results/               # 실험 결과 및 분석 보고서
│   ├── scenario/                 # Scenario split 결과
│   ├── temporal/                 # Temporal split 결과
│   ├── multiseed/                # Multi-seed 검증 (87실험 × 5시드)
│   ├── dann/                     # DANN domain adaptation 결과
│   ├── figures/                  # 논문용 Figure 9개
│   ├── statistical_tests/        # 통계 검정 결과
│   ├── STATISTICAL_ANALYSIS.md   # 통계 분석 보고서
│   ├── TEMPORAL_SHIFT_ANALYSIS.md # Temporal shift 근본원인 분석
│   └── PAPER_DRAFT_ELEMENTS.md   # 논문 핵심 요소 정리
│
├── reports/               # 분석 보고서 (디렉토리 생성 예정)
│   ├── RESEARCH_ANALYSIS.md      # 연구 방향 분석
│   ├── experiment_results_YYYYMMDD.json
│   └── figures/                  # 시각화 그래프
│
├── checkpoints/           # 모델 체크포인트 (디렉토리 생성 예정)
│   ├── baselines/
│   ├── single_models/
│   └── ensemble/
│
├── logs/                  # 로그 파일
│
├── src/                   # 소스 코드
│   ├── __init__.py
│   ├── core/              # 핵심 모듈
│   ├── api/               # REST API
│   ├── analysis/          # 분석 도구
│   ├── config/            # 설정 관리
│   └── utils/             # 유틸리티
│
```

**범례**:
- 존재하는 디렉토리/파일
- 계획됨: 계획되었으나 아직 생성되지 않음

## 주요 파일 설명

### 루트 디렉토리 파일

#### `main.py`
- **기능**: 프로그램의 메인 진입점
- **주요 명령어**:
  - `assess`: 단일 이미지 안전성 평가
  - `train`: 모델 훈련
  - `evaluate`: 모델 성능 평가
  - `compare`: 멀티 모델 성능 비교
  - `experiment`: 전체 실험 파이프라인
  - `serve`: REST API 서버 시작
- **사용 예시**: `python main.py assess image.jpg`

#### `safetyknob.py`
- **기능**: main.py를 감싸는 래퍼 스크립트
- **특징**:
  - 디버그 모드 지원 (`--debug`)
  - 성능 프로파일링 (`--profile`)
  - 상세 로깅 (`--verbose`)

#### `config.json`
- **기능**: 시스템 전체 설정 파일
- **주요 설정**:
  - 사용할 모델 목록 (SigLIP, CLIP, DINOv2 등)
  - 안전성 평가 차원 및 가중치
  - 훈련 하이퍼파라미터
  - API 서버 설정

#### `report_generator.py`
- **기능**: 분석 결과를 다양한 형식으로 출력
- **지원 형식**: HTML, PDF, Markdown
- **주요 기능**: 성능 메트릭 시각화, 혼동 행렬, ROC 곡선

## src/ 디렉토리 구조

### src/core/ - 핵심 모듈

#### `safety_assessment_system.py`
- **기능**: 안전성 평가 시스템의 메인 클래스
- **주요 클래스**: `SafetyAssessmentSystem`
- **책임**:
  - 이미지 안전성 평가 오케스트레이션
  - 모델 로딩 및 관리
  - 훈련/평가 프로세스 관리

#### `embedders.py`
- **기능**: 사전학습 비전 모델 관리
- **지원 모델**:
  - SigLIP (google/siglip-so400m-patch14-384)
  - CLIP (openai/clip-vit-large-patch14)
  - DINOv2 (facebook/dinov2-large)
  - EVA-CLIP
- **주요 메서드**: 이미지를 임베딩 벡터로 변환

#### `classifier.py`
- **기능**: 안전성 분류기 구현
- **주요 클래스**: `SafetyClassifier`
- **특징**:
  - 임베딩 기반 분류
  - 5가지 안전성 차원별 예측
  - 신경망 기반 분류기

#### `ensemble.py`
- **기능**: 멀티 모델 앙상블 메서드
- **앙상블 전략**:
  - 가중 투표 (Weighted Voting)
  - 스태킹 (Stacking)
- **효과**: 개별 모델보다 높은 정확도 달성

#### `safety_dimensions.py`
- **기능**: 안전성 평가 차원 정의 및 관리
- **주요 클래스**:
  - `SafetyDimension`: 동적 차원 관리
  - `SafetyAssessmentResult`: 평가 결과 데이터 구조
  - `DimensionAnalyzer`: 차원별 점수 계산
- **특징**: config.json에서 차원을 동적으로 로드

### src/api/ - REST API

#### `server.py`
- **기능**: FastAPI 기반 웹 서버
- **엔드포인트**:
  - `GET /api/v1/health`: 서버 상태 확인
  - `GET /api/v1/info`: 시스템 정보 및 사용 가능한 모델 목록 (server.py:75)
  - `POST /api/v1/assess`: 단일 이미지 안전성 평가 (server.py:96)
  - `POST /api/v1/assess/batch`: 배치 이미지 평가 (구현 완료, server.py:140)
- **참고**: `/api/v1/models`는 `/api/v1/info`의 alias로 계획됨

#### `inference.py`
- **기능**: API용 추론 로직 (legacy)
- **상태**: 현재 `server.py`가 주요 API 구현
- **특징**: 단순 추론 wrapper

### src/analysis/ - 분석 도구

#### `single_model.py`
- **기능**: 단일 모델 성능 분석
- **분석 내용**: 정확도, F1 스코어, 혼동 행렬

#### `multi_model.py`
- **기능**: 여러 모델 동시 분석
- **특징**: 병렬 처리로 효율적 분석

#### `model_comparison.py`
- **기능**: 모델 간 성능 비교
- **주요 클래스**: `ModelPerformanceAnalyzer`
- **출력**: 비교 차트, 성능 보고서

#### `metrics.py`
- **기능**: 평가 메트릭 계산
- **메트릭**: Accuracy, Precision, Recall, F1, AUC-ROC

### src/config/ - 설정 관리

#### `settings.py`
- **기능**: 시스템 설정 클래스
- **주요 클래스**: `SystemConfig`
- **특징**: config.json 파싱 및 검증

#### `paths.py`
- **기능**: 프로젝트 경로 관리
- **관리 경로**: 데이터, 체크포인트, 로그, 결과

### src/utils/ - 유틸리티

#### `data_loader.py`
- **기능**: 이미지 데이터 로딩
- **주요 클래스**: `ImageDataset`
- **특징**: PyTorch Dataset 인터페이스

#### `cache_manager.py`
- **기능**: 임베딩 캐시 관리
- **효과**: 반복 계산 방지로 속도 향상

#### `visualization.py`
- **기능**: 결과 시각화
- **시각화 종류**:
  - 성능 메트릭 차트
  - 혼동 행렬
  - ROC 곡선
  - t-SNE 임베딩 시각화

#### `logger.py`
- **기능**: 로깅 설정 및 관리
- **특징**: 컬러 로깅, 파일 출력

#### `analysis_utils.py`
- **기능**: 분석 관련 유틸리티 함수
- **내용**: 데이터 전처리, 통계 계산

#### `data_utils.py`
- **기능**: 데이터 처리 유틸리티
- **내용**: 이미지 전처리, 라벨 처리

#### `visualization_config.py`
- **기능**: 시각화 설정 및 스타일
- **내용**:
  - 플롯 스타일 상수 (PLOT_STYLE, PLOT_CONFIG)
  - 색상 정의 (SAFE_COLOR, DANGER_COLOR)
  - 차트 설정 함수 (get_cluster_fig_config, get_confusion_matrix_config 등)

## 실행 흐름

### 1. 이미지 평가 흐름
```
main.py (assess 명령)
  ↓
SafetyAssessmentSystem.assess_image()
  ↓
Embedders (SigLIP/CLIP/DINOv2)로 임베딩 생성
  ↓
SafetyClassifier 또는 유사도 기반 평가
  ↓
Ensemble로 결과 결합
  ↓
SafetyAssessmentResult 반환
```

### 2. 모델 훈련 흐름
```
main.py (train 명령)
  ↓
ImageDataset으로 데이터 로드
  ↓
SafetyAssessmentSystem.train()
  ↓
각 모델별 SafetyClassifier 훈련
  ↓
체크포인트 저장
```

### 3. API 서버 흐름
```
main.py (serve 명령)
  ↓
FastAPI 서버 시작
  ↓
/api/v1/assess 엔드포인트
  ↓
이미지 업로드 → 평가 → JSON 응답
```

## 확장 방법

### 새로운 비전 모델 추가
1. `src/core/embedders.py`에 새 Embedder 클래스 추가
2. `config.json`의 models 섹션에 모델 정보 추가

### 새로운 안전성 차원 추가
1. `config.json`의 safety.dimensions에 새 차원 추가
2. 가중치와 설명 포함

### 새로운 앙상블 전략 추가
1. `src/core/ensemble.py`에 새 전략 구현
2. `config.json`의 ensemble_strategy 업데이트
