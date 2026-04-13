# Changelog

All notable changes to SafetyKnob project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-04-06

### Critical Fix
- **Scenario split frame-level data leakage discovered**: `data_scenario/` (old) used frame-level split, causing 98.1% test sequence overlap with train. All `results/multiseed/scenario_*` results invalidated.
- Corrected results use `data_scenario_v2/` (sequence-level split) via `results/multiseed_v2/`
- SigLIP 2-layer: 96.11% → **87.28%** (corrected, -8.83%p)
- CLIP 2-layer: 91.54% → **78.91%** (corrected, -12.63%p)
- DINOv2 2-layer: 90.92% → **77.94%** (corrected, -12.98%p)
- **1-layer ≈ 2-layer** (gap 0.12%p, was 0.96%p) → 1-layer MLP sufficient
- All documentation updated to scenario_v2 numbers

### Added
- 3-layer probe experiments (`results/scenario_v2/scenario_v2_siglip_3layer_seed*/`)
- Ensemble stacking (LR, optimized weights) (`results/ensemble_stacking/`)
- Data scaling curve on scenario_v2 (`results/scaling_v2/`)
- Temporal per-month analysis (Oct vs Nov) (`results/temporal_error_analysis/`)
- Temporal calibration ECE (0.2816 vs scenario 0.0993)
- Temporal error analysis with FN categorization
- Full fine-tuning ceiling experiment (`results/full_finetune_temporal/`)
- Inference benchmarking (probe <0.001ms/image)
- `results/RESULT_INDEX.md` — canonical result mapping
- `results/CRITICAL_FINDING_SCENARIO_LEAKAGE.md` — leakage analysis
- `results/COMPREHENSIVE_REVIEW_2026_04_06.md` — full research review

### Changed
- `experiments/train_from_embeddings.py`: Added 3-layer probe, --train-fraction arg
- `experiments/train_finetune_lora.py`: Added --full-finetune option
- All documentation updated with corrected numbers (scenario_v2 baseline)
- `results/TEMPORAL_SHIFT_ANALYSIS.md` fully replaced with corrected analysis
- DANN results marked as invalidated across all docs (취소선 + 수정값)

### Preserved (이전 코드/결과 보존)
- `results/multiseed/` — old (leaked) results preserved for audit trail
- `experiments/train_dann.py` — original DANN script preserved (deprecated, use train_dann_clean.py)
- `data_scenario/` — old frame-level split preserved
- `embeddings/scenario/` — old embeddings preserved

---

## [1.6.0] - 2026-04-03

### Critical Fix
- **Data leakage discovered and corrected**: scenario_v2 and temporal datasets share 64.3% of test images in train → All cross-dataset experiments invalidated and re-run with clean temporal-only splits
- DANN SigLIP: 99.19% → 65.28% (corrected, -33.91%p)
- Per-category temporal: "+1~4%p" → "-21~30%p" (corrected, sign reversed)

### Added
- `experiments/train_dann_clean.py` — Clean DANN using temporal-only splits (no data leakage)
- `scripts/label_shift_correction.py` — Category-aware sample reweighting experiment
- `scripts/cross_domain_experiment.py` — Cross-domain transfer (AI Hub → Construction-PPE)
- `scripts/analyze_temporal_shift_visual.py` — t-SNE + MMD temporal shift analysis
- `scripts/generate_final_figures.py` — Final paper figures (temporal methods, cross-domain, LoRA gap)
- LoRA fine-tuning temporal: **F1 77.77±1.02%** (5-seed, +11.59%p, only meaningful improvement)
- Cross-domain zero-shot: SigLIP 84.44%, CLIP 75.05%, DINOv2 63.36% (all 5-seed)
- Statistical test: LoRA vs Baseline paired t-test (p=0.000031, Cohen's d=10.48)
- t-SNE temporal shift visualization + per-category MMD analysis
- 6 new figures in `results/figures_final/`

### Changed
- `experiments/train_finetune_lora.py` — num_workers=0 (deadlock fix)
- `docs/RESEARCH_AUDIT.md` — Full revision with leakage discovery, corrected results, LoRA/cross-domain
- `results/PAPER_DRAFT_ELEMENTS.md` — 4 contributions (updated), honest limitations

### Key Findings (Revised)
- Temporal shift is genuine **feature-level shift**, NOT label shift
- DANN, label reweighting, hierarchical: all ineffective on clean splits
- LoRA recovers 38.7% of temporal gap (only method that works)
- MMD analysis: Equipment Hazard (C) has largest shift, Fall Hazard (A) smallest

---

## [1.5.0] - 2026-03-24

### Added
- `scripts/statistical_tests.py` — 통계적 유의성 검정 (paired t-test, McNemar, Bootstrap CI, Bonferroni/Holm-Bonferroni 보정)
- `scripts/generate_paper_figures.py` — 논문용 Figure 9개 자동 생성 (Confusion Matrix, ROC, PR, Error 분포, 상관 히트맵 등)
- `scripts/analyze_temporal_shift_cause.py` — Temporal shift 근본원인 분석 (Feature Distribution Shift 규명)
- `scripts/run_full_experiment.sh` — 전체 실험 재현 자동화 스크립트 (12단계)
- `results/STATISTICAL_ANALYSIS.md` — 통계 분석 보고서 (Bonferroni 보정 후 110/136쌍 유의)
- `results/TEMPORAL_SHIFT_ANALYSIS.md` — Temporal shift 근본원인 분석 보고서
- `results/PAPER_DRAFT_ELEMENTS.md` — 논문 핵심 요소 정리 (Contributions, Limitations, RQ 결과)
- `results/figures/` — 논문용 고해상도 Figure 9개

### Changed
- `src/analysis/multi_model.py` — `_create_correlation_heatmap()` 구현 완료
- `src/core/embedders.py` — DINOv2 기본 체크포인트 `dinov2-giant` → `dinov2-large` 통일
- `config.example.json` — 3모델(SigLIP, CLIP, DINOv2) 전체 포함으로 완성
- 전체 문서 DINOv2 참조 `1536-d` → `1024-d` 통일
- 하드코딩된 절대경로 12곳 상대경로로 변경
- 삭제 파일 참조(main_v2, benchmark_v2, src_v2, TODO.md) 전부 제거

### Fixed
- `scripts/generate_5d_labels.py:212` — `sum()` key 인자 버그 수정
- `docs/DATASET_GUIDE_ko.md` — 앵커 링크 불일치 수정
- `docs/EXPERIMENT_PROTOCOL.md` — 프로젝트명 SafetyKnob→EmoKnob, 버전 1.3 업데이트

### Added (한국어 문서)
- 10개 한국어 문서 완성 (총 6,965줄)
- `docs/DEVELOPMENT_ko.md`, `docs/API_REFERENCE_ko.md`, `docs/PROJECT_STRUCTURE_ko.md` — 스텁→완전 번역
- `docs/AIHUB_DOWNLOAD_GUIDE_ko.md`, `docs/DATA_ANALYSIS_ko.md`, `docs/RESEARCH_AUDIT_ko.md`, `docs/EXPERIMENT_PROTOCOL_ko.md` — 신규 작성

---

## [1.4.0] - 2026-03-24

### Added

#### Multi-seed 검증 및 논문 수준 실험

1. **Multi-seed 전체 실험 (85 runs × 5 seeds)**
   - `scripts/run_multiseed.py`: 모든 probe 실험 자동화 (seed={42,123,456,789,2024})
   - 전 결과에 mean±std 제공. SigLIP 2-layer: 96.11±0.30% (std < 0.5%p)
   - 결과: `results/multiseed/multiseed_summary.json`

2. **DANN Domain Adaptation**
   - `experiments/train_dann.py`: Gradient reversal layer 기반 domain adaptation
   - ~~원래 결과: Temporal F1 99.24% — 데이터 누출로 무효화 (2026-04-01)~~
   - **수정된 결과** (`experiments/train_dann_clean.py`): Temporal F1 65.28±0.88% (효과 없음)
   - LoRA fine-tuning이 유일한 부분 해결책: 66.11% → **77.77±1.02%** (+11.59%p)
   - 결과: `results/dann_clean/`, `results/lora_temporal/`

3. **Data Scaling Curve**
   - `scripts/run_scaling_curve.py`: 10/25/50/75/100% 데이터 비율별 실험
   - 25%만으로 88.94% F1 달성. 50% 이상에서 수확체감.
   - 결과: `results/scaling_curve/scaling_curve.json`, `scaling_curve.png`

4. **Ensemble 2-model Subset Ablation**
   - `scripts/run_ensemble_ablation.py`: 모든 model subset 조합 비교
   - SigLIP+DINOv2(97.11%) > 3-model 앙상블(95.88%) → 2-model이 최적
   - 결과: `results/ensemble_ablation/ensemble_ablation.json`

5. **Per-Category Temporal Shift Analysis**
   - `scripts/analyze_temporal_per_category.py`: 카테고리(A-E)별 temporal 취약도 분석
   - 놀라운 발견: 카테고리별 독립 분류기는 temporal shift에 거의 영향 없음
   - PPE만 -0.24%p 하락, 나머지 4개 카테고리는 오히려 향상
   - 결과: `results/temporal_per_category/temporal_per_category.json`

6. **t-SNE 시각화 (캐시 임베딩 기반)**
   - `scripts/visualize_tsne_cached.py`: GPU 불필요, 사전추출 임베딩 사용
   - 6개 plot 생성: safe/unsafe, category별, domain shift 비교
   - 결과: `results/visualization/tsne_*.png`

### Changed

- README.md: 성능표에 mean±std 추가, DANN 결과, scaling curve 추가
- RESEARCH_AUDIT.md: Limitations 섹션 대폭 보강, 새 실험 결과 통합
- RESEARCH_METHODOLOGY.md: multi-seed 결과 반영, DANN/ensemble ablation 추가
- EXPERIMENT_PROTOCOL.md: TBD 제거, 임베딩 워크플로우, 버전 v1.3
- TODO_ko.md: 보완 실험 완료 상태 반영

---

## [1.3.0] - 2026-03-24

### Added

#### 보완 실험 완료 (RQ1-RQ3 검증)

1. **임베딩 사전 추출 인프라**
   - `scripts/extract_embeddings.py`: Foundation model 임베딩 1회 추출 후 캐시
   - `experiments/train_from_embeddings.py`: 캐시 기반 경량 probe 학습 (~3초/실험)
   - transformers 5.x API 호환성 수정 (SiglipVisionModel, Dinov2Model, _to_tensor helper)

2. **RQ1 Probe Depth Ablation 실행 (9실험)**
   - SigLIP: Linear 80.43% / 1-Layer 95.42% / 2-Layer 96.13%
   - CLIP: Linear 76.63% / 1-Layer 89.68% / 2-Layer 91.52%
   - DINOv2: Linear 75.19% / 1-Layer 90.41% / 2-Layer 90.79%
   - **판정**: Gap > 5%p → 비선형 변환 필수, "선형 분리" 주장 수정

3. **RQ1 ResNet50-Frozen Baseline 실행**
   - ResNet50-Frozen: F1 78.42% (vs SigLIP-Frozen 96.13%) → Foundation model 결정적 우위

4. **RQ2 Temporal Ensemble 실험 실행**
   - Temporal 재학습: SigLIP 68.08%, CLIP 61.84%, DINOv2 55.98%
   - Ensemble(avg) 63.96% < SigLIP 68.08% → **앙상블 실패**
   - 에러 상관 분석: corr 0.34-0.44, All wrong 17.1%
   - ResNet50-Frozen temporal: 57.33%

5. **RQ3 Independent vs Multi-task 비교 실행 (5실험)**
   - 5차원 중 4차원에서 독립 학습이 multi-task보다 우수
   - Fall: 94.24% vs 76.54% (-17.69%p), Equipment: 91.47% vs 75.69% (-15.78%p)
   - **판정**: 부정적 전이 확인, multi-task → category-specific 재정의

6. **data_temporal/ 재생성**
   - `create_temporal_split.py`: val split 추가 (85/15 stratified)
   - train 8,006 / val 1,412 / test 5,119

### Fixed

1. **아키텍처 불일치 버그** (Critical)
   - `train_binary.py`: `load_checkpoint_compat()` 추가 (old→new state dict 변환)
   - `run_ensemble_binary.py`: `BinarySafetyClassifier` 신규 아키텍처 동기화
   - 체크포인트 저장 시 `probe_depth`, `embedding_dim` 메타데이터 추가

2. **transformers 5.x 호환성**
   - SigLIP: `AutoModel` → `SiglipVisionModel` + `SiglipImageProcessor`
   - DINOv2: `AutoModel` → `Dinov2Model` + `AutoImageProcessor`
   - CLIP: `get_image_features()` 반환 타입 변경 대응

### Added (Documentation)

- `docs/RESEARCH_AUDIT.md`: 코드 감사 결과 및 보완 실험 전체 보고서
- README.md: 실제 실험 결과로 성능표 갱신
- RESEARCH_METHODOLOGY.md: RQ1-RQ3 검증 결과 반영

---

## [1.2.0] - 2026-03-20

### Added

#### 실험 보완 (연구 주장 검증 강화)

1. **ResNet50-Frozen 실험 지원** (`experiments/train_baseline.py`)
   - `--frozen` 옵션 추가: backbone freeze 후 probe head만 학습
   - Trainable params: ~1.0M (vs Finetuned 24.6M)
   - 목적: Frozen SigLIP vs Frozen ResNet50 공정 비교 (RQ1)

2. **Probe Depth Ablation** (`experiments/train_binary.py`)
   - `--probe-depth` 옵션 추가: `linear` (1,153 params), `1layer` (~591K), `2layer` (~656K, 기존)
   - 목적: True linear probe 성능 측정으로 선형 분리 주장 검증 (RQ1)

3. **Temporal Ensemble 지원** (`scripts/run_ensemble_binary.py`)
   - `--data-dir`, `--results-dir`, `--output-dir` CLI 인자 추가
   - 기존 하드코딩 경로 제거, 임의 데이터/결과 디렉토리 지원
   - 목적: Temporal split에서 앙상블 강건성 검증 (RQ2)

4. **5D Independent Training** (`experiments/train_binary.py`)
   - `--category A/B/C/D/E` 옵션 추가: 카테고리별 이미지 필터링
   - 목적: Per-dimension 독립 classifier vs shared multi-task 비교 (RQ3)

#### 문서 갱신

- **README.md**: 성능표에 ResNet50-Frozen, Probe Depth Ablation 행 추가
- **RESEARCH_METHODOLOGY.md**: Section 2.1~2.3 실제 결과 및 검증 상태 업데이트
- **EXPERIMENT_PROTOCOL.md**: 보완 실험 A~D 프로토콜 추가
- **TODO_ko.md**: Stage 2 보완 실험 항목 추가, 실행 상태 반영

---

## [1.1.0] - 2025-11-24

### ✅ Added

#### 문서화
- **연구 보고서**: 전체 연구 과정과 결과를 포함한 최종 보고서 (`results/RESEARCH_REPORT_FINAL.md`)
- **모델 비교 분석**: 3개 모델 성능 비교 리포트 및 시각화 (`results/comparison/`)
- **데이터 상태 문서**: 현재 데이터 부재 상황과 복구 절차 설명 (`DATA_STATUS.md`)

#### 분석 도구
- **모델 비교 스크립트**: `scripts/compare_models.py`
  - 전체 성능 비교 테이블
  - 차원별 성능 히트맵
  - 훈련 효율성 분석
  - 4가지 시각화 차트 자동 생성

### 🐛 Fixed

#### 버그 수정 (Critical)

1. **Config 반영 문제** ✅
   - **문제**: `config.json`의 `checkpoint` 파라미터가 코드에서 무시됨
   - **해결**:
     - `embedders.py`: 모든 Embedder 클래스에 `checkpoint` 파라미터 추가
     - `safety_assessment_system.py`: checkpoint를 `create_embedder()`에 전달
   - **영향**: 이제 config.json에서 모델 체크포인트를 자유롭게 변경 가능
   - **파일**:
     - `src/core/embedders.py:49,82,119,151,203-225`
     - `src/core/safety_assessment_system.py:53-74`

2. **Threshold 하드코딩 문제** ✅
   - **문제**: `safety_threshold`, `confidence_threshold`가 코드에 0.5로 고정됨
   - **해결**:
     - `SafetyAssessmentSystem`: config에서 threshold 읽어서 사용
     - `EnsembleClassifier`: threshold를 파라미터로 받음
     - 모든 판단 로직에 설정값 반영
   - **영향**: config.json에서 임계값 조정 가능
   - **파일**:
     - `src/core/safety_assessment_system.py:100-136,167,169,206,211`
     - `src/core/ensemble.py:38-49,71,82`

#### 버그 수정 (API 안정성)

3. **API 보안 및 안정성 개선** ✅
   - **문제**:
     - 파일 저장 시 동시성 문제 (`/tmp/{filename}` 사용)
     - 파일 크기 제한 없음
     - CORS `*` 설정 (보안 위험)
   - **해결**:
     - `tempfile.NamedTemporaryFile()` 사용 → UUID 기반 고유 파일명
     - 파일 크기 제한: 10MB
     - 허용 확장자 검증: `.jpg, .jpeg, .png, .bmp`
     - 이미지 유효성 검증: `Image.verify()`
     - 배치 크기 제한: 50개 (100→50)
     - 에러 메시지 표준화 (JSON 형식)
     - try-finally로 임시 파일 정리 보장
   - **영향**: 운영 환경에서 안전하게 사용 가능
   - **파일**: `src/api/server.py:1-294`

### 📝 Changed

#### 문서 업데이트

1. **README.md** ✅
   - 성능 지표를 실제 측정값으로 업데이트
   - 예상치 제거, 검증된 결과로 교체
   - 데이터 부재 경고 추가
   - 연구 보고서 및 비교 분석 링크 추가

2. **성능 수치 정정**
   - **기존**: 예상치 (F1 95.7%, AUC 99.2%)
   - **변경**: 실제 측정치 (F1 95.5%, AUC 99.1%)
   - **출처**: `results/single_models/*/results.json`

### 📊 Results

#### 모델 성능 (Test Set, 1,737장)

| Model | Accuracy | F1 | Precision | Recall | AUC | Training Time |
|-------|----------|-----|-----------|--------|-----|---------------|
| **SigLIP** | **95.3%** | **95.5%** | 94.4% | 96.6% | **99.1%** | 5.1 hrs |
| **CLIP** | 87.2% | 87.4% | 88.4% | 86.5% | 94.6% | 2.9 hrs |
| **DINOv2** | 85.7% | 86.0% | 86.0% | 86.0% | 94.3% | 3.1 hrs |

#### 핵심 발견

1. **SigLIP 최고 성능**: 모든 지표에서 압도적 (F1 95.5%, AUC 99.1%)
2. **CLIP 최고 효율**: 가장 빠른 학습 (2.9시간)
3. **차원별 성능 격차**: Protective Gear (85%) vs Environmental Risk (35%)
4. **전체 vs 차원 격차**: 89.7% vs 47.2% (42.5%p 차이)

### ⚠️ Known Issues

1. **데이터 부재** ❌
   - `/data/` 디렉토리 비어있음
   - 훈련 재현 불가능
   - 해결: AI Hub에서 데이터셋 재다운로드 필요
   - 문서: `DATA_STATUS.md` 참조

2. **앙상블 실험 미완료** ⚠️
   - 코드 구현 완료, 테스트 필요
   - 예상 성능: F1 96-97%, AUC 99.5%+
   - 데이터 확보 후 진행 가능

3. **Baseline 비교 없음** ⚠️
   - ResNet-50 end-to-end 학습 미수행
   - 데이터 효율성 주장 근거 부족

4. **DINOv2 Embedding Dim 불일치** ✅ **수정 완료**
   - 이전: 문서에서 1536 dim (dinov2-giant) 참조
   - 실제: 1024 dim (dinov2-large) 사용
   - 수정: config.json, embedders.py, 모든 문서를 dinov2-large (1024-d)로 통일 (2026-03-24)

### 🔬 Research Status

#### 가설 검증

1. **가설 1: 임베딩 공간 선형 분리** ✅ **확인됨**
   - F1 95.5%, AUC 99.1% 달성
   - Linear Probe만으로 충분

2. **가설 2: 앙상블 강건성** ⚠️ **부분 검증**
   - 모델별 강점 차이 확인
   - 실제 앙상블 테스트 필요

3. **가설 3: 차원 독립성** ⚠️ **부분 검증**
   - 독립 학습 가능
   - 성능 불균형 심각 (26-85%)

---

## [1.0.0] - 2024-10-02

### Added
- 초기 프로젝트 구조
- 3개 모델 (SigLIP, CLIP, DINOv2) 구현
- 신경망 분류기 (multi-head architecture)
- 앙상블 메커니즘 (Weighted Vote, Stacking)
- FastAPI 서버
- CLI 인터페이스
- 기본 문서화

### Trained
- SigLIP 모델 훈련 완료 (5.1 hrs)
- CLIP 모델 훈련 완료 (2.9 hrs)
- DINOv2 모델 훈련 완료 (3.1 hrs)

---

## [Unreleased]

### To Do (High Priority)
- [ ] 데이터셋 재확보 (AI Hub)
- [ ] 앙상블 실험 수행
- [ ] Baseline 모델 비교 (ResNet-50)
- [ ] 확률 보정 (Temperature Scaling)
- [ ] DINOv2 Embedding Dim 수정

### To Do (Medium Priority)
- [ ] 차원별 성능 개선 (SMOTE, 재라벨링)
- [ ] 분포 변화 강건성 테스트
- [ ] Grad-CAM 시각화
- [ ] 단위 테스트 추가

### To Do (Low Priority)
- [ ] 모델 경량화 (Quantization)
- [ ] 온라인 학습 지원
- [ ] 다중 모달 확장

---

## Version History

- **1.1.0** (2025-11-24): 버그 수정, 문서화, 비교 분석
- **1.0.0** (2024-10-02): 초기 릴리스, 3개 모델 훈련 완료
