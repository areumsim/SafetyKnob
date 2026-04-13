# Research Audit Report

## Overview

This document records the code audit findings and supplementary experiment results
for the SafetyKnob research project. The audit identified gaps between research
claims and supporting evidence, which were then addressed through additional experiments.

**Audit Date**: 2026-03-20
**Experiments Completed**: 2026-03-24
**Multi-seed Validation**: 2026-03-24 (5 seeds: {42, 123, 456, 789, 2024})
**Critical Revision**: 2026-04-01 (Data leakage discovered and corrected)
**Ablation Analysis**: 2026-04-06 (LoRA rank ablation, augmentation, per-category analysis)

> **⚠ IMPORTANT (2026-04-01)**: A critical data leakage was discovered in the DANN
> experiment and per-category temporal analysis. The scenario and temporal datasets
> share the same 14,537 images with different splits. 64.3% of temporal test images
> appeared in scenario train data. All experiments using scenario data as source
> and temporal data as target have been invalidated and re-run with clean splits.
> See "Critical Revision: Data Leakage Discovery" section below.

---

## Audit Findings

### A. Architecture Mismatch Bug (FIXED)

**Problem**: `train_binary.py` used `classifier.*` keys while `run_ensemble_binary.py` and
existing checkpoints used `feature_extractor.*` + `safety_head.*` keys. Cross-loading was impossible.

**Fix**: Added `load_checkpoint_compat()` to auto-detect and convert old-format state dicts.
New checkpoints save `probe_depth` and `embedding_dim` metadata.

### B. Duplicate Result Sets (RESOLVED)

| Path | SigLIP F1 | Source |
|------|-----------|--------|
| `results/scenario/` | 95.73% | `data_scenario/` random split (canonical) |
| `results/danger_al/` | 94.61% | Different data split (deprecated) |

**Decision**: `results/scenario/` is the canonical result set.

### C. Temporal Split Regeneration (COMPLETED)

- `data_temporal/` was empty; `create_temporal_split.py` had no val split
- Fixed: added stratified 85/15 train/val split from June-Sept 2022 data
- Result: train=8,006 / val=1,412 / test=5,119

### D. Embedding Extraction Performance (FIXED)

- All training scripts extracted embeddings per-epoch (SigLIP: 11h per run)
- Created `scripts/extract_embeddings.py` for one-time extraction
- Created `experiments/train_from_embeddings.py` for fast probe training (~3s per experiment)
- Fixed transformers 5.x API compatibility (BaseModelOutputWithPooling changes)

### E. transformers 5.x API Changes (FIXED)

- `AutoProcessor`/`AutoModel` broke for SigLIP and DINOv2
- SigLIP: switched to `SiglipVisionModel` + `SiglipImageProcessor`
- DINOv2: switched to `Dinov2Model` + `AutoImageProcessor`
- CLIP: `get_image_features()` returns `BaseModelOutputWithPooling` — added `_to_tensor()` helper

---

## Supplementary Experiment Results

### RQ1: Feature Quality of Foundation Model Representations

**Original claim**: "Linear separability of foundation model features"
**Problem**: Used 2-layer MLP (656K params), not a true linear probe

#### Probe Depth Ablation Results (5-seed mean±std)

| Model | Linear F1 | 1-Layer F1 | 2-Layer F1 | Gap |
|-------|-----------|------------|------------|-----|
| SigLIP | 80.37±0.16% | 95.15±0.24% | 96.11±0.30% | 15.74%p |
| CLIP | 76.43±0.20% | 89.93±0.59% | 91.54±0.49% | 15.10%p |
| DINOv2 | 74.47±0.58% | 89.97±0.30% | 90.92±0.42% | 16.46%p |

#### Baseline Comparison

| Model | Mode | Probe Params | F1 |
|-------|------|-------------|-----|
| ResNet50 | Frozen | 1,049,601 | 78.42% |
| ResNet50 | Finetuned | 24.6M | 95.49% (prior result) |
| SigLIP | Frozen (2-layer) | 656,129 | 96.13% |
| SigLIP | Frozen (linear) | 1,153 | 80.43% |

#### RQ1 Verdict

- **Linear probe gap > 5%p**: Features are NOT linearly separable. Non-linear transformation is critical.
- **SigLIP-Frozen (2-layer) >> ResNet50-Frozen**: Foundation model representations are decisively superior to ImageNet representations (+17.71%p).
- **SigLIP-Frozen (2-layer) > ResNet50-Finetuned**: Foundation model + lightweight probe surpasses full end-to-end finetuning (+0.64%p with 38x fewer trainable parameters).
- **Revised claim**: "Foundation model features enable high accuracy with lightweight non-linear probes, eliminating the need for end-to-end finetuning."

### RQ2: Ensemble Robustness Under Distribution Shift

**Original claim**: "Ensemble improves robustness to distribution shift"
**Problem**: Ensemble was never tested on temporal split; on scenario split, ensemble < SigLIP

#### Temporal Distribution Shift Results (5-seed mean±std)

| Model | Scenario F1 | Temporal F1 | Delta |
|-------|------------|------------|-------|
| SigLIP (2-layer) | 96.11±0.30% | 66.11±0.72% | -30.00%p |
| CLIP (2-layer) | 91.54±0.49% | 60.72±0.51% | -30.81%p |
| DINOv2 (2-layer) | 90.92±0.42% | 56.65±0.83% | -34.27%p |
| **Ensemble (avg)** | 95.88±0.21% | 63.96% | -31.92%p |
| ResNet50-Frozen | 78.42% | 57.33% | -21.09%p |
| **DANN (SigLIP)** | **96.70±0.12%** | **99.24±0.05%** | **+2.54%p** |

#### Error Correlation (Temporal Test Set)

| Pair | Correlation |
|------|------------|
| SigLIP-CLIP | 0.4382 |
| SigLIP-DINOv2 | 0.3369 |
| CLIP-DINOv2 | 0.3891 |

- All correct: 39.0% of images
- All wrong: 17.1% of images

#### Ensemble Effect Separation (Scenario vs Temporal)

Scenario split에서의 앙상블 결과를 분리 검증하여 distribution shift 효과와 앙상블 효과를 구분:

| Model | Scenario F1 | Temporal F1 |
|-------|------------|------------|
| SigLIP (best single) | 96.13% | 68.08% |
| Ensemble (avg) | 95.81% | 63.96% |
| **Delta (Ensemble - Single)** | **-0.32%p** | **-4.12%p** |

- **Scenario에서도 앙상블 실패** (-0.32%p): distribution shift와 무관하게 앙상블이 최고 단일 모델보다 열등
- **Temporal에서 격차 확대** (-4.12%p): distribution shift 하에서 앙상블의 약점이 더 두드러짐
- **결론**: 앙상블 실패는 distribution shift의 부산물이 아니라, 근본적으로 모델 간 에러 다양성 부족에 기인

#### RQ2 Verdict

- **Ensemble failed**: On both splits, ensemble F1 < best single model (SigLIP).
- **Error correlations are moderate** (0.34-0.44), suggesting some diversity exists but is insufficient.
- **All models collapse under temporal shift**: 28-35%p degradation across all approaches.
- **Foundation models NOT more robust than ResNet50**: ResNet50-Frozen actually showed smaller degradation (21%p) despite lower absolute performance. This may indicate that ImageNet features, being more generic (texture/edge-based), are paradoxically more robust to scene-level distribution shift than semantically-specialized foundation model features.
- **Root cause hypothesis**: Models overfit to scene-level visual patterns (construction site layout, lighting) that change across months, rather than learning invariant safety-relevant features.

#### DANN Domain Adaptation Results

**⚠ INVALIDATED (2026-04-01)**: Original DANN results below had severe data leakage.
See "Critical Revision" section for corrected results.

~~| Method | Source (Scenario) F1 | Target (Temporal) F1 |~~
~~|--------|---------------------|---------------------|~~
~~| No adaptation (baseline) | 96.11±0.30% | 66.11±0.72% |~~
~~| **DANN** | **96.70±0.12%** | **99.24±0.05%** |~~

**Corrected DANN Results** (clean, temporal-only split, 5-seed mean±std):

| Method | Model | Val F1 | Target (Temporal Test) F1 |
|--------|-------|--------|--------------------------|
| Baseline (no DA) | SigLIP | ~96% | 66.11±0.72% |
| DANN Clean | SigLIP | 96.40±0.22% | 65.28±0.88% |
| DANN Clean | CLIP | 89.37±0.42% | 60.31±0.94% |
| DANN Clean | DINOv2 | 91.25±0.34% | 55.87±0.55% |

- **DANN provides NO improvement**: All models show marginal degradation (-0.4 to -0.8%p) with DANN.
- Domain discriminator loss converged to ~0.693 (= -ln(0.5)), indicating inability to distinguish domains in embedding space.
- The temporal shift is not addressable by standard domain adaptation in frozen embedding space.

#### Ensemble 2-model Subset Ablation (Scenario, 5-seed mean±std)

| Combination | F1 | AUC |
|-------------|-----|-----|
| SigLIP+DINOv2 | 97.11±0.12% | 99.41±0.03% |
| SigLIP+CLIP | 96.66±0.20% | 99.42±0.04% |
| SigLIP (single) | 96.30±0.26% | 99.39±0.04% |
| SigLIP+CLIP+DINOv2 | 95.88±0.21% | 99.42±0.04% |
| CLIP+DINOv2 | 93.53±0.34% | 98.25±0.06% |

- **SigLIP+DINOv2 pair outperforms full 3-model ensemble** (+1.23%p).
- Adding CLIP degrades the ensemble (CLIP is the weakest model and introduces noise).
- **Optimal ensemble is 2-model**: SigLIP+DINOv2, not all three.

#### Per-Category Temporal Shift Analysis (SigLIP, 5-seed mean±std)

**⚠ INVALIDATED (2026-04-01)**: Original per-category results below had data leakage
(trained on scenario data which contained 64.3% of temporal test images).

~~| Category | Scenario F1 | Temporal F1 | Delta |~~
~~|----------|------------|------------|-------|~~
~~| Collision Risk (B) | 98.18% | 99.74% | +1.55%p |~~
~~| Fall Hazard (A) | 95.09% | 99.17% | +4.08%p |~~

**Corrected Per-Category Results** (clean, trained on temporal train only, 5-seed mean±std):

| Category | Scenario F1 (clean) | Temporal F1 (clean) | Delta |
|----------|--------------------|--------------------| ------|
| Fall Hazard (A) | 95.23% | 74.02±0.16% | -21.21%p |
| Collision Risk (B) | 98.24% | 69.33±2.50% | -28.91%p |
| Equipment Hazard (C) | 93.10% | 64.34±0.55% | -28.76%p |
| Environmental Risk (D) | 96.06% | 66.99±0.27% | -29.07%p |
| Protective Gear (E) | 98.00% | 67.58±1.02% | -30.42%p |
| **Global Binary** | **96.11%** | **66.11±0.72%** | **-30.00%p** |

- **Revised finding**: Per-category classifiers are NOT robust to temporal shift. All categories show 21-30%p degradation.
- The original +1~4%p improvement was entirely an artifact of data leakage.
- Fall Hazard (A) is the most robust (-21%p), possibly because fall hazard visual patterns are more time-invariant.
- **Implication**: The temporal shift is a genuine feature-level distribution change, not merely label shift.

#### Additional Temporal Shift Correction Experiments (2026-04-01)

| Method | SigLIP F1 | CLIP F1 | DINOv2 F1 |
|--------|-----------|---------|-----------|
| Baseline (no correction) | 66.38±0.71% | 60.77±0.50% | 56.90±0.58% |
| Category reweighting | 66.50±0.69% | 60.30±0.54% | 54.92±0.46% |
| DANN Clean | 65.28±0.88% | 60.31±0.94% | 55.87±0.55% |
| Hierarchical (cat+safety) | 67.56±0.28% | - | - |

- **Label shift correction is ineffective**: Category-aware reweighting provides <0.5%p improvement.
- **Hierarchical classifier provides marginal benefit**: +1.45%p over flat binary, suggesting the category classification step helps slightly but cannot overcome the feature shift.
- **Conclusion**: The temporal shift is dominated by feature-level changes (lighting, weather, vegetation, camera angles across seasons), not by label/category distribution shift.

#### Data Scaling Curve (SigLIP 2-layer, 5-seed mean±std)

| Fraction | N_train | F1 |
|----------|---------|-----|
| 10% | 1,017 | 82.68±0.46% |
| 25% | 2,543 | 88.94±0.46% |
| 50% | 5,087 | 93.41±0.23% |
| 75% | 7,631 | 95.01±0.18% |
| 100% | 10,175 | 96.11±0.28% |

- Diminishing returns beyond 50% data. The 50%→100% gain is only 2.70%p.
- 25% data (2,543 samples) already achieves 88.94% — practical for data-scarce domains.

### RQ3: Multi-dimensional Safety Assessment

**Original claim**: "Independent dimension heads with shared feature extractor"
**Problem**: No comparison between independent and multi-task training

#### Independent vs Multi-task Results (SigLIP, 2-layer probe, 5-seed mean±std for Independent)

| Dimension | Independent F1 | Multi-task F1 | Delta |
|-----------|---------------|--------------|-------|
| Fall Hazard (A) | 95.23±0.25% | 76.54% | -18.69%p |
| Collision Risk (B) | 98.24±0.27% | 98.24% | ±0.00%p |
| Equipment Hazard (C) | 93.10±0.61% | 75.69% | -17.41%p |
| Environmental Risk (D) | 96.06±0.23% | 94.74% | -1.32%p |
| Protective Gear (E) | 98.00±0.19% | 90.50% | -7.50%p |

#### RQ3 Verdict

- **Multi-task hurts 4 of 5 dimensions**: Shared feature extractor causes negative transfer for Fall Hazard (-17.69%p) and Equipment Hazard (-15.78%p).
- **Only Collision Risk benefits**: +1.02%p, likely due to its already strong signal.

#### Label Design Flaw Analysis

현재 데이터셋의 라벨 구조는 진정한 multi-task/multi-label 학습에 부적합하며, 이는 부정적 전이의 근본 원인:

1. **1이미지 = 1차원 라벨**: 각 이미지는 파일명 카테고리 코드(A-E)로 단 1개의 활성 차원만 보유. 나머지 4차원은 0.9("not applicable")로 기본 설정.
2. **Multi-task 실패는 설계적 한계**: Shared feature extractor가 5개 차원을 동시에 학습할 때, 4/5 차원의 라벨이 "해당 없음"이므로 유의미한 gradient signal이 없음. 이는 multi-task learning이 실패하도록 사전 결정된 구조.
3. **부정적 전이 메커니즘**: Fall(A) 이미지에서 Collision(B) head는 항상 0.9(안전) 라벨을 학습 → shared backbone이 Fall-specific feature를 억제하는 방향으로 편향.
4. **Collision이 유일하게 양의 전이를 보인 이유**: Collision(B)은 데이터셋에서 가장 큰 비중을 차지하여 shared backbone이 Collision-friendly feature를 학습. 다른 차원은 이 편향의 피해자.

**정확한 용어 재정의**:
- ❌ "Multi-dimensional safety assessment" → 실제로는 하나의 이미지에 하나의 위험 유형만 라벨링
- ✅ "Category-specific binary classification" → 각 카테고리별 독립적 safe/unsafe 분류
- 진정한 multi-dimensional assessment를 위해서는 1이미지에 대해 5차원 동시 라벨링이 필요 (현재 데이터셋에 부재)

---

## Canonical Result Reference

All canonical results use `data_scenario/` (random split, 14,537 images) as the primary evaluation dataset.
Temporal results use `data_temporal/` (June-Sept train, Oct-Nov test) for distribution shift evaluation.

Results directories:
- `results/scenario/{model}_{depth}/`: Probe depth ablation
- `results/scenario/resnet50_frozen/`: ResNet50 baseline
- `results/temporal/{model}/`: Temporal split experiments
- `results/temporal/ensemble/`: Temporal ensemble results
- `results/scenario/siglip_independent_{A-E}/`: Per-dimension independent classifiers
- `results/multitask/siglip/`: Multi-task classifier

---

## Limitations and Threats to Validity

### Statistical Rigor

1. **신뢰구간**: 모든 핵심 실험에 5-seed 반복(seed={42,123,456,789,2024}) mean±std 제공 완료.
2. **통계적 유의성**: SigLIP(96.11±0.30%) vs CLIP(91.54±0.49%) 차이(4.57%p)는 표준편차 대비 충분히 크므로 유의. 그러나 정식 paired t-test / McNemar's test는 미수행.
3. **Seed 수 한계**: 5 seeds는 정확한 95% CI를 제공하기에 불충분. 방향성 확인 수준의 신뢰도.
4. **초기 실험과의 차이**: Multi-seed 평균은 초기 단일 seed 결과와 소폭 차이 존재 (예: SigLIP 2-layer 96.13% → 96.11±0.30%). 이는 seed 의존성이 낮음을 시사.

### Data and Label Design

5. **단일 데이터셋**: AI Hub 건설 현장 안전 이미지 1종만 사용. 다른 산업 도메인(제조, 물류)으로의 일반화 미검증.
6. **라벨 설계 결함**: 1이미지 = 1활성차원 구조로 인해 multi-task 실험이 구조적으로 불리. 진정한 multi-label annotation 부재.
7. **자동 라벨링**: 파일명 기반 자동 라벨링(카테고리 코드 A-E). 인간 검수 미수행으로 라벨 노이즈 가능성.

### Experimental Design

8. **RQ2 교란 요인**: 앙상블 효과와 distribution shift 효과가 혼재. Scenario split에서의 분리 검증으로 부분 해소 (앙상블이 scenario에서도 -0.32%p).
9. **Temporal shift 분석 피상적**: 어떤 시나리오/카테고리가 가장 취약한지에 대한 세분화 분석 부족. ResNet50-Frozen의 상대적 강건성(-21%p vs -28%p) 원인 미탐구.
10. **Ablation 미완성**: 앙상블 내 2-model subset ablation, 데이터 크기 scaling curve, 3-layer/4-layer probe depth saturation 확인 미수행.

### Reproducibility

11. **GPU 의존성**: 학습 결과는 NVIDIA RTX 4090 기준. 다른 GPU에서의 재현성은 cudnn 결정성 설정에 의존.
12. **transformers 버전**: transformers 5.x API 변경으로 인한 호환성 이슈 존재. 정확한 라이브러리 버전 고정 필요.

---

## Critical Revision: Data Leakage Discovery (2026-04-01)

### Overview

A thorough research review discovered that the `scenario_v2` and `temporal` datasets
contain the **same 14,537 images** with different train/test splits. Any experiment
that trains on scenario data and evaluates on temporal data has data leakage:

```
Scenario train ∩ Temporal test: 3,291 images (64.3% of temporal test)
Scenario train ∩ Temporal train: 5,791 images
```

### Affected Experiments

| Experiment | Original Result | Clean Result | Impact |
|-----------|----------------|-------------|--------|
| DANN (SigLIP) | Target F1 99.19% | 65.28% | **INVALIDATED** (-33.91%p) |
| DANN (CLIP) | Target F1 93.12% | 60.31% | **INVALIDATED** (-32.81%p) |
| DANN (DINOv2) | Target F1 98.11% | 55.87% | **INVALIDATED** (-42.24%p) |
| Per-cat temporal (A) | +4.08%p | -21.21%p | **INVALIDATED** (sign reversed) |
| Per-cat temporal (B) | +1.55%p | -28.91%p | **INVALIDATED** (sign reversed) |
| Per-cat temporal (C) | +2.81%p | -28.76%p | **INVALIDATED** (sign reversed) |
| Per-cat temporal (D) | +2.82%p | -29.07%p | **INVALIDATED** (sign reversed) |
| Per-cat temporal (E) | -0.24%p | -30.42%p | **INVALIDATED** |

### Root Cause

`train_dann.py` (line 161) loads scenario embeddings as source domain and temporal
embeddings as target domain. Since both datasets are different splits of the same
14,537 images, the model has already "seen" 64.3% of temporal test images during
training via the scenario train split.

### Corrections Applied

1. Created `experiments/train_dann_clean.py`: Uses ONLY temporal split data
   - Source = temporal train (June-Sept)
   - Target = temporal test (Oct-Nov, labels hidden from DANN)
   - Train-test overlap: 0 images (verified)

2. Re-ran per-category analysis using only temporal embeddings for both training and evaluation.

3. Added `scripts/label_shift_correction.py`: Tests whether category distribution
   reweighting can correct the temporal shift (result: negligible effect).

### Revised Research Conclusions

**Before correction:**
- "DANN completely resolves temporal shift (99.24% F1)"
- "Per-category classifiers are robust to temporal shift (+1~4%p)"
- "Temporal shift is primarily label shift (Simpson's Paradox)"

**After correction:**
- DANN provides NO improvement on temporal shift (-0.83%p)
- Per-category classifiers suffer the same ~30%p degradation as global classifiers
- Temporal shift is a **genuine feature-level distribution change**, not label shift
- Neither DANN, label reweighting, nor hierarchical classification meaningfully addresses it
- The shift likely stems from seasonal visual changes (lighting, weather, vegetation)

### Unaffected Experiments

The following experiments are NOT affected by this leakage:
- All scenario_v2 experiments (train/test from same split, no cross-dataset evaluation)
- Probe depth ablation (scenario_v2 only)
- Data scaling curve (scenario_v2 only)
- Ensemble ablation on scenario_v2
- Temporal baseline (trains and tests within temporal split)

### New Clean Results

All corrected results are stored in:
- `results/temporal_clean/` — Clean temporal baselines
- `results/temporal_per_category_clean/` — Clean per-category temporal results
- `results/dann_clean/` — Clean DANN results
- `results/label_shift/` — Label shift correction experiments
- `results/hierarchical_clean/` — Clean hierarchical classifier results
- `results/lora_temporal/` — LoRA fine-tuning on temporal split
- `results/cross_domain/` — Cross-domain transfer experiments (AI Hub → Construction-PPE)

---

## New Experiments (2026-04-01)

### LoRA Fine-tuning for Temporal Shift

LoRA fine-tuning (r=16, alpha=32, 0.46% of params trainable) partially recovers temporal
performance, confirming that the temporal shift requires backbone modification.

| Method | Temporal Test F1 | vs Frozen Baseline |
|--------|-----------------|-------------------|
| Frozen 2-layer probe | 66.18±0.60% | - |
| LoRA fine-tune (r=16, 5 epochs) | **77.77±1.02%** | **+11.59%p** |

- **5-seed validated** (seeds: 42, 123, 456, 789, 2024)
- LoRA recovers **38.7%** of the temporal gap (66→78 out of 66→96 possible)
- Val F1 reaches ~98% (near-perfect on train distribution) but test drops to ~78%
- Indicates the temporal shift is addressable with backbone adaptation but not fully
- Per-seed F1 range: 75.99% ~ 79.07% (consistent improvement across all seeds)

### Cross-Domain Generalization (AI Hub → Construction-PPE)

Tested generalization from AI Hub construction safety dataset to Ultralytics
Construction-PPE dataset (1,416 images, 11 PPE classes, different labeling scheme).

| Method | SigLIP F1 | CLIP F1 | DINOv2 F1 |
|--------|-----------|---------|-----------|
| Zero-shot transfer | 84.44±2.21% | 75.05±6.28% | 63.36±5.13% |
| PPE from scratch | 94.97±0.93% | 96.04±0.26% | 95.82±0.70% |
| Pre-train AI Hub → fine-tune PPE | 94.65±0.72% | 93.68±0.27% | 92.17±0.59% |

Key findings:
- **SigLIP zero-shot transfers best (84.44%)**: Vision-language pre-training captures safety-relevant concepts
- **CLIP zero-shot moderate (75.05%)**: Good transfer but high variance (±6.28%)
- **DINOv2 transfers poorly (63.36%)**: Self-supervised features less transferable for safety
- **PPE from scratch matches/beats pre-trained**: PPE dataset (1,132 train) is sufficient to learn safety features independently
- **Model ranking for transfer**: SigLIP > CLIP > DINOv2 (consistent with scenario performance ranking)

---

## Ablation Analysis (2026-04-06)

비판적 분석에서 제기된 6가지 약점(W1-W6)에 대한 실험적 대응.
전체 상세 보고서: `results/ABLATION_ANALYSIS_REPORT.md`

### W1: LoRA Rank Ablation — Val-Test Gap은 과적합인가?

r=0(head-only), 4, 8, 16, 32에 대한 ablation (seed=42, 5 epochs):

| Rank | Method | Test F1 | Val F1 | Val-Test Gap |
|------|--------|---------|--------|-------------|
| 0 | Head-only | 56.15% | 83.96% | 27.8%p |
| 4 | LoRA | 72.40% | 97.66% | 25.3%p |
| 8 | LoRA | 77.32% | 97.96% | 20.6%p |
| 16 | LoRA | 76.53% | 98.83% | 22.3%p |
| 32 | LoRA | 76.57% | 98.25% | 21.7%p |

**결론**: Val-test gap은 과적합이 아닌 temporal distribution shift.
- Head-only(backbone 0 params trainable)에서 gap이 최대(27.8%p)
- Rank 증가 시 test F1은 r=8에서 saturation(plateau), 하락 없음
- Epoch별 test F1이 monotonic 하락하지 않음 (과적합 패턴 아님)

### W2: Data Augmentation 2×2 Ablation

| | No Augmentation | With Augmentation | Delta |
|--|-----------------|-------------------|-------|
| r=0 (Head-only) | 56.15% | 59.31% | +3.16%p |
| r=16 (LoRA) | 76.53% | 77.39% | +0.86%p |

**결론**: Pixel-level augmentation(+3%p)은 LoRA backbone 적응(+21%p)의 1/7 수준.
Temporal shift는 semantic-level 변화이므로 pixel augmentation으로 해결 불가.

### W3: Temporal Shift 시각적 증거

- 카테고리별 train vs test 이미지 비교 그리드 생성 (`fig_temporal_samples_grid.png`)
- 기존 t-SNE, MMD per-category, monthly distribution 분석과 결합

### W4: PPE Test n=141 통계 분석

| Method | Mean F1 | Bootstrap 95% CI | Wilson CI (n=141) |
|--------|---------|-------------------|-------------------|
| Zero-shot | 84.44% | [82.41, 85.92] | [77.55, 89.50] |
| Scratch | 94.97% | [94.07, 95.60] | [90.02, 97.53] |
| Fine-tune | 94.65% | [94.00, 95.13] | [89.62, 97.32] |

- Scratch vs Fine-tune CI 겹침 → 통계적으로 구분 불가
- Wilson CI width ~7.5%p → n=141의 한계 정직 인정
- Negative transfer 없음: task 불일치(scene-level vs equipment-level)로 설명

### W5: LoRA Per-Category Analysis

| Category | MMD | Frozen F1 | LoRA F1 | Delta |
|----------|-----|-----------|---------|-------|
| A (Fall Hazard) | 0.031 | 59.02% | 76.70% | +17.68%p |
| B (Collision Risk) | 0.042 | 59.09% | 81.10% | +22.01%p |
| C (Equipment) | 0.155 | 64.55% | 77.93% | +13.38%p |
| D (Environmental) | 0.049 | 59.30% | 86.02% | +26.72%p |
| E (Protective Gear) | 0.138 | 29.79% | 60.36% | +30.57%p |

- LoRA는 모든 카테고리에서 +13~31%p 일관된 개선
- MMD↔LoRA Delta 상관 없음 (Spearman r=0.000) — 초기 가설과 불일치, 정직 보고
- 대신 frozen F1이 낮은 카테고리(E)에서 LoRA 개선이 가장 큼

### W6: Reproducibility

- `requirements.txt`에 `peft>=0.6.0` 추가
- `--num-workers` CLI arg 추가

### Ablation Results Directories

- `results/lora_rank_ablation/r{0,4,8,16,32}/` — Rank ablation 실험 결과
- `results/augmentation_ablation/{r0,r16}_augment/` — Augmentation ablation 결과
- `results/lora_per_category_analysis/` — Per-category LoRA vs Frozen 분석
- `results/bootstrap_ci_analysis/` — PPE bootstrap CI 분석
- `results/figures_final/fig_lora_*.png, fig_augmentation_*.png, fig_temporal_samples_grid.png` — 생성 Figure

---

## Critical Revision 2: Scenario Split Frame-Level Leakage (2026-04-06)

### Overview

`data_scenario/`의 split이 frame-level로 수행되어, 같은 CCTV 영상의 연속 프레임이 train/test에 분산됨.
`scripts/create_scenario_split.py` 주석: "98.1% of test sequences also appear in training".
`data_scenario_v2/`에서 sequence-level split으로 수정됨.

**그러나 `results/multiseed/` (논문 기준 결과)는 old `embeddings/scenario/`를 사용.**
corrected 결과는 `results/multiseed_v2/`에 존재.

### Affected Results

| Model | Old (leaked, `results/multiseed/`) | Clean (`results/multiseed_v2/`) | Delta |
|-------|-----------------------------------|--------------------------------|-------|
| SigLIP linear | 80.37±0.16% | **76.81±0.28%** | -3.56%p |
| SigLIP 1-layer | 95.15±0.24% | **87.16±0.26%** | -7.99%p |
| SigLIP 2-layer | 96.11±0.30% | **87.28±0.45%** | -8.83%p |
| CLIP 2-layer | 91.54±0.49% | **78.91±0.48%** | -12.63%p |
| DINOv2 2-layer | 90.92±0.42% | **77.94±0.46%** | -12.98%p |

### Impact on Research Conclusions

1. **"Non-linear essential" 주장 수정**: 1-layer vs 2-layer gap이 0.12%p로 축소. 1-layer로 충분.
2. **"Foundation > ResNet50 finetuned" 재검증 필요**: ResNet50-finetuned (95.49%)는 old split. scenario_v2에서 재실험 필요.
3. **Temporal degradation 수정**: 87.28→66.11 = -21.17%p (이전 보고 -30.00%p에서 축소).
4. **절대 성능 87%**: "즉시 배포 가능" 주장 철회. "보조 도구 수준"으로 조정.

### New 3-Layer Probe Results (2026-04-06)

| Depth | Scenario_v2 F1 | Temporal F1 |
|-------|---------------|-------------|
| 1-layer | 87.16±0.26% | - |
| 2-layer | 87.28±0.45% | 66.11±0.72% |
| **3-layer** | **86.81±0.94%** | **66.98±0.60%** |

3-layer에서 scenario 성능 저하, temporal 동등. **2-layer에서 saturation 확인.**

### Ensemble Stacking Results (2026-04-06, seed=42)

| Method | Scenario_v2 F1 | 비고 |
|--------|---------------|------|
| SigLIP single | 87.50% | baseline |
| **Optimized weights** | **88.40%** | SigLIP:0.50, DINOv2:0.45, CLIP:0.05 |
| LR Stacking | 87.78% | SigLIP coefficient dominant (1.87) |
| SigLIP+DINOv2 pair | 87.67% | |
| Naive 3-model avg | 84.80% | worst |

Optimized weights: +0.90%p over single SigLIP. Modest improvement.

### Unaffected Experiments

- Temporal baselines (train/test within temporal split) — correct
- LoRA experiments (temporal split only) — correct
- DANN clean (temporal split only) — correct
- Cross-domain (separate dataset) — correct
- Per-category scenario_v2 results in `results/multiseed_v2/` — correct

### Clean Result Directories

- `results/multiseed_v2/scenario_v2_*` — Corrected scenario results (5-seed)
- `results/scenario_v2/scenario_v2_siglip_3layer_*` — 3-layer probe results
- `results/ensemble_stacking/` — Stacking and optimized weights
- `results/CRITICAL_FINDING_SCENARIO_LEAKAGE.md` — Detailed analysis
- `results/COMPREHENSIVE_REVIEW_2026_04_06.md` — Full review report

### Full Fine-Tuning Ceiling Results (2026-04-06)

SigLIP 428M params full FT on temporal train (lr=1e-5, batch=2, 3 epochs):

| Method | Temporal Test F1 | Trainable Params |
|--------|-----------------|-----------------|
| Frozen probe | 66.11±0.72% | 656K |
| **Full FT (best epoch 2)** | **72.62%** | **428.8M** |
| **LoRA (r=16, 5-seed)** | **77.77±1.02%** | **1.97M** |

- **Full FT < LoRA by 5.15%p**: Overfitting with 428M params on 8K data
- **LoRA is the practical ceiling**: Parameter-efficient fine-tuning superior to full FT in data-scarce setting
- Epoch 2→3: val F1 92→96% but test F1 72.6→71.6% (classic overfitting)
- Results: `results/full_finetune_temporal_v3/`
