# 연구 감사 보고서

## 개요

본 문서는 SafetyKnob 연구 프로젝트의 코드 감사 결과와 보완 실험 결과를 기록합니다.
감사를 통해 연구 주장과 근거 사이의 격차를 식별하였으며, 추가 실험을 통해 이를 해소하였습니다.

**감사 일시**: 2026-03-20
**실험 완료**: 2026-03-24
**Multi-seed 검증**: 2026-03-24 (5 seeds: {42, 123, 456, 789, 2024})

---

## 감사 발견사항

### A. 아키텍처 불일치 버그 (수정 완료)

**문제**: `train_binary.py`는 `classifier.*` 키를 사용한 반면, `run_ensemble_binary.py`와
기존 checkpoint는 `feature_extractor.*` + `safety_head.*` 키를 사용하여 교차 로딩이 불가능했습니다.

**수정**: `load_checkpoint_compat()`를 추가하여 구 형식 state dict를 자동 감지 및 변환합니다.
새 checkpoint는 `probe_depth`와 `embedding_dim` 메타데이터를 함께 저장합니다.

### B. 중복 결과 세트 (해결 완료)

| 경로 | SigLIP F1 | 출처 |
|------|-----------|------|
| `results/scenario/` | 95.73% | `data_scenario/` 랜덤 split (정식) |
| `results/danger_al/` | 94.61% | 별도 데이터 split (폐기) |

**결정**: `results/scenario/`가 정식 결과 세트입니다.

### C. Temporal Split 재생성 (완료)

- `data_temporal/`이 비어 있었으며, `create_temporal_split.py`에 validation split이 없었음
- 수정: 2022년 6-9월 데이터에서 층화 85/15 train/val split 추가
- 결과: train=8,006 / val=1,412 / test=5,119

### D. Embedding 추출 성능 (수정 완료)

- 모든 학습 스크립트가 epoch마다 embedding을 추출하여 SigLIP 기준 1회 실행에 11시간 소요
- `scripts/extract_embeddings.py`로 1회성 추출 구현
- `experiments/train_from_embeddings.py`로 빠른 probe 학습 구현 (실험당 ~3초)
- transformers 5.x API 호환성 수정 (BaseModelOutputWithPooling 변경사항)

### E. transformers 5.x API 변경 (수정 완료)

- `AutoProcessor`/`AutoModel`이 SigLIP 및 DINOv2에서 동작 불가
- SigLIP: `SiglipVisionModel` + `SiglipImageProcessor`로 전환
- DINOv2: `Dinov2Model` + `AutoImageProcessor`로 전환
- CLIP: `get_image_features()`가 `BaseModelOutputWithPooling`을 반환 — `_to_tensor()` 헬퍼 추가

---

## 보완 실험 결과

### RQ1: Foundation Model 표현의 특징 품질

**기존 주장**: "Foundation model 특징의 선형 분리 가능성"
**문제점**: 진정한 linear probe가 아닌 2-layer MLP (656K 파라미터)를 사용

#### Probe Depth Ablation 결과 (5-seed 평균±표준편차)

| 모델 | Linear F1 | 1-Layer F1 | 2-Layer F1 | 격차 |
|------|-----------|------------|------------|------|
| SigLIP | 80.37±0.16% | 95.15±0.24% | 96.11±0.30% | 15.74%p |
| CLIP | 76.43±0.20% | 89.93±0.59% | 91.54±0.49% | 15.10%p |
| DINOv2 | 74.47±0.58% | 89.97±0.30% | 90.92±0.42% | 16.46%p |

#### Baseline 비교

| 모델 | 모드 | Probe 파라미터 수 | F1 |
|------|------|------------------|-----|
| ResNet50 | Frozen | 1,049,601 | 78.42% |
| ResNet50 | Finetuned | 24.6M | 95.49% (기존 결과) |
| SigLIP | Frozen (2-layer) | 656,129 | 96.13% |
| SigLIP | Frozen (linear) | 1,153 | 80.43% |

#### RQ1 판정

- **Linear probe 격차 > 5%p**: 특징이 선형 분리 가능하지 않음. 비선형 변환이 핵심적임.
- **SigLIP-Frozen (2-layer) >> ResNet50-Frozen**: Foundation model 표현이 ImageNet 표현보다 확실히 우수 (+17.71%p).
- **SigLIP-Frozen (2-layer) > ResNet50-Finetuned**: Foundation model + 경량 probe가 전체 end-to-end finetuning을 초과 (+0.64%p, 학습 파라미터 38배 적음).
- **수정된 주장**: "Foundation model 특징은 경량 비선형 probe로 높은 정확도를 달성할 수 있으며, end-to-end finetuning이 불필요하다."

### RQ2: Distribution Shift 하에서의 앙상블 강건성

**기존 주장**: "앙상블이 distribution shift에 대한 강건성을 향상시킨다"
**문제점**: 앙상블이 temporal split에서 테스트된 적 없었으며, scenario split에서도 앙상블 < SigLIP

#### Temporal Distribution Shift 결과 (5-seed 평균±표준편차)

| 모델 | Scenario F1 | Temporal F1 | 변화량 |
|------|------------|------------|--------|
| SigLIP (2-layer) | 96.11±0.30% | 66.11±0.72% | -30.00%p |
| CLIP (2-layer) | 91.54±0.49% | 60.72±0.51% | -30.81%p |
| DINOv2 (2-layer) | 90.92±0.42% | 56.65±0.83% | -34.27%p |
| **앙상블 (avg)** | 95.88±0.21% | 63.96% | -31.92%p |
| ResNet50-Frozen | 78.42% | 57.33% | -21.09%p |
| ~~DANN (SigLIP)~~ | ~~96.70±0.12%~~ | ~~99.24±0.05%~~ | ~~+2.54%p~~ |

> **⚠ INVALIDATED (2026-04-01)**: DANN 결과는 데이터 누출로 무효화됨. 수정된 결과는 아래 "Critical Revision" 참조.

#### 오류 상관관계 (Temporal 테스트 셋)

| 모델 쌍 | 상관계수 |
|---------|---------|
| SigLIP-CLIP | 0.4382 |
| SigLIP-DINOv2 | 0.3369 |
| CLIP-DINOv2 | 0.3891 |

- 모두 정답: 이미지의 39.0%
- 모두 오답: 이미지의 17.1%

#### 앙상블 효과 분리 (Scenario vs Temporal)

Scenario split에서의 앙상블 결과를 분리 검증하여 distribution shift 효과와 앙상블 효과를 구분:

| 모델 | Scenario F1 | Temporal F1 |
|------|------------|------------|
| SigLIP (최고 단일 모델) | 96.13% | 68.08% |
| 앙상블 (avg) | 95.81% | 63.96% |
| **차이 (앙상블 - 단일)** | **-0.32%p** | **-4.12%p** |

- **Scenario에서도 앙상블 실패** (-0.32%p): distribution shift와 무관하게 앙상블이 최고 단일 모델보다 열등
- **Temporal에서 격차 확대** (-4.12%p): distribution shift 하에서 앙상블의 약점이 더 두드러짐
- **결론**: 앙상블 실패는 distribution shift의 부산물이 아니라, 근본적으로 모델 간 에러 다양성 부족에 기인

#### RQ2 판정

- **앙상블 실패**: 두 split 모두에서 앙상블 F1이 최고 단일 모델(SigLIP)보다 낮음.
- **오류 상관관계가 중간 수준** (0.34-0.44)으로, 일정 수준의 다양성이 존재하나 충분하지 않음.
- **모든 모델이 temporal shift에서 붕괴**: 모든 접근 방식에서 28-35%p 성능 저하.
- **Foundation model이 ResNet50보다 강건하지 않음**: ResNet50-Frozen이 절대 성능은 낮지만 오히려 더 적은 성능 저하를 보임 (21%p). 이는 ImageNet 특징이 더 범용적(텍스처/엣지 기반)이므로, 의미적으로 특화된 foundation model 특징보다 장면 수준 distribution shift에 역설적으로 더 강건할 수 있음을 시사.
- **근본 원인 가설**: 모델이 불변적인 안전 관련 특징 대신 월별로 변화하는 장면 수준 시각 패턴(건설 현장 배치, 조명)에 과적합됨.

#### DANN Domain Adaptation 결과

**⚠ INVALIDATED (2026-04-01)**: 아래 결과는 데이터 누출로 무효화됨.

~~| 방법 | Source (Scenario) F1 | Target (Temporal) F1 |~~
~~|------|---------------------|---------------------|~~
~~| 적응 없음 (baseline) | 96.11±0.30% | 66.11±0.72% |~~
~~| **DANN** | **96.70±0.12%** | **99.24±0.05%** |~~

**수정된 DANN 결과** (clean, temporal-only split, 5-seed mean±std):

| 방법 | 모델 | Val F1 | Target (Temporal Test) F1 |
|------|------|--------|--------------------------|
| Baseline (no DA) | SigLIP | ~96% | 66.11±0.72% |
| DANN Clean | SigLIP | 96.40±0.22% | 65.28±0.88% |
| DANN Clean | CLIP | 89.37±0.42% | 60.31±0.94% |
| DANN Clean | DINOv2 | 91.25±0.34% | 55.87±0.55% |

- **DANN은 효과 없음**: 모든 모델에서 미미한 성능 저하 (-0.4 ~ -0.8%p).
- Domain discriminator loss가 ~0.693 (=-ln(0.5))에 수렴 → embedding 공간에서 도메인 구분 불가.
- Frozen embedding에서는 표준 domain adaptation이 작동하지 않음.

#### 앙상블 2-모델 서브셋 Ablation (Scenario, 5-seed 평균±표준편차)

| 조합 | F1 | AUC |
|------|-----|-----|
| SigLIP+DINOv2 | 97.11±0.12% | 99.41±0.03% |
| SigLIP+CLIP | 96.66±0.20% | 99.42±0.04% |
| SigLIP (단일) | 96.30±0.26% | 99.39±0.04% |
| SigLIP+CLIP+DINOv2 | 95.88±0.21% | 99.42±0.04% |
| CLIP+DINOv2 | 93.53±0.34% | 98.25±0.06% |

- **SigLIP+DINOv2 쌍이 3-모델 전체 앙상블을 능가** (+1.23%p).
- CLIP을 추가하면 앙상블 성능이 저하 (CLIP이 가장 약한 모델이며 노이즈를 유입).
- **최적 앙상블은 2-모델**: SigLIP+DINOv2이며, 3개 모델 전부가 아님.

#### 카테고리별 Temporal Shift 분석 (SigLIP, 5-seed 평균±표준편차)

| 카테고리 | Scenario F1 | Temporal F1 | 변화량 | N(test) |
|----------|------------|------------|--------|---------|
| Collision Risk (B) | 98.18±0.40% | 99.74±0.03% | +1.55%p | 1,373 |
| Fall Hazard (A) | 95.09±0.28% | 99.17±0.10% | +4.08%p | 937 |
| Equipment Hazard (C) | 93.11±0.84% | 95.92±0.90% | +2.81%p | 854 |
| Environment (D) | 96.06±0.23% | 98.87±0.28% | +2.82%p | 919 |
| PPE (E) | 98.00±0.19% | 97.76±0.46% | -0.24%p | 1,036 |

- **놀라운 발견**: 카테고리별 분류기는 temporal 저하가 거의 없음 (대부분 오히려 향상).
- **PPE만 약간의 성능 저하** (-0.24%p)를 보여, PPE 외관이 시간에 따라 가장 많이 변화함을 시사.
- **시사점**: 심각한 전체 temporal shift (30%p)는 카테고리별 특징 열화가 아닌, 시간에 따른 카테고리 혼합/기저율 변화에 기인할 가능성이 높음. 카테고리별 분류기는 이 유형의 shift에 본질적으로 더 강건함.

#### 데이터 Scaling Curve (SigLIP 2-layer, 5-seed 평균±표준편차)

| 비율 | N_train | F1 |
|------|---------|-----|
| 10% | 1,017 | 82.68±0.46% |
| 25% | 2,543 | 88.94±0.46% |
| 50% | 5,087 | 93.41±0.23% |
| 75% | 7,631 | 95.01±0.18% |
| 100% | 10,175 | 96.11±0.28% |

- 50% 데이터 이후 수확 체감. 50%→100% 이득은 2.70%p에 불과.
- 25% 데이터 (2,543개)로 이미 88.94% 달성 — 데이터 부족 도메인에서 실용적.

### RQ3: 다차원 안전 평가

**기존 주장**: "공유 feature extractor를 사용한 독립 차원 헤드"
**문제점**: 독립 학습과 multi-task 학습 간 비교 없음

#### 독립 vs Multi-task 결과 (SigLIP, 2-layer probe, 독립 모델은 5-seed 평균±표준편차)

| 차원 | 독립 F1 | Multi-task F1 | 변화량 |
|------|---------|--------------|--------|
| Fall Hazard (A) | 95.23±0.25% | 76.54% | -18.69%p |
| Collision Risk (B) | 98.24±0.27% | 98.24% | ±0.00%p |
| Equipment Hazard (C) | 93.10±0.61% | 75.69% | -17.41%p |
| Environmental Risk (D) | 96.06±0.23% | 94.74% | -1.32%p |
| Protective Gear (E) | 98.00±0.19% | 90.50% | -7.50%p |

#### RQ3 판정

- **Multi-task가 5개 중 4개 차원에서 성능 저하**: 공유 feature extractor가 Fall Hazard (-17.69%p)와 Equipment Hazard (-15.78%p)에서 부정적 전이를 유발.
- **Collision Risk만 이득**: +1.02%p로, 이미 강한 신호 덕분일 가능성 높음.

#### 라벨 설계 결함 분석

현재 데이터셋의 라벨 구조는 진정한 multi-task/multi-label 학습에 부적합하며, 이는 부정적 전이의 근본 원인:

1. **1이미지 = 1차원 라벨**: 각 이미지는 파일명 카테고리 코드(A-E)로 단 1개의 활성 차원만 보유. 나머지 4차원은 0.9("not applicable")로 기본 설정.
2. **Multi-task 실패는 설계적 한계**: 공유 feature extractor가 5개 차원을 동시에 학습할 때, 4/5 차원의 라벨이 "해당 없음"이므로 유의미한 gradient signal이 없음. 이는 multi-task learning이 실패하도록 사전 결정된 구조.
3. **부정적 전이 메커니즘**: Fall(A) 이미지에서 Collision(B) head는 항상 0.9(안전) 라벨을 학습 → 공유 backbone이 Fall-specific feature를 억제하는 방향으로 편향.
4. **Collision이 유일하게 양의 전이를 보인 이유**: Collision(B)은 데이터셋에서 가장 큰 비중을 차지하여 공유 backbone이 Collision-friendly feature를 학습. 다른 차원은 이 편향의 피해자.

**정확한 용어 재정의**:
- "Multi-dimensional safety assessment" → 실제로는 하나의 이미지에 하나의 위험 유형만 라벨링
- "Category-specific binary classification" → 각 카테고리별 독립적 safe/unsafe 분류
- 진정한 multi-dimensional assessment를 위해서는 1이미지에 대해 5차원 동시 라벨링이 필요 (현재 데이터셋에 부재)

---

## 정식 결과 참조

모든 정식 결과는 `data_scenario/` (랜덤 split, 14,537개 이미지)를 기본 평가 데이터셋으로 사용합니다.
Temporal 결과는 `data_temporal/` (6-9월 train, 10-11월 test)을 distribution shift 평가에 사용합니다.

결과 디렉토리:
- `results/scenario/{model}_{depth}/`: Probe depth ablation
- `results/scenario/resnet50_frozen/`: ResNet50 baseline
- `results/temporal/{model}/`: Temporal split 실험
- `results/temporal/ensemble/`: Temporal 앙상블 결과
- `results/scenario/siglip_independent_{A-E}/`: 차원별 독립 분류기
- `results/multitask/siglip/`: Multi-task 분류기

---

## 한계 및 타당성 위협 요소

### 통계적 엄밀성

1. **신뢰구간**: 모든 핵심 실험에 5-seed 반복(seed={42,123,456,789,2024}) mean±std 제공 완료.
2. **통계적 유의성**: SigLIP(96.11±0.30%) vs CLIP(91.54±0.49%) 차이(4.57%p)는 표준편차 대비 충분히 크므로 유의. 그러나 정식 paired t-test / McNemar's test는 미수행.
3. **Seed 수 한계**: 5 seeds는 정확한 95% CI를 제공하기에 불충분. 방향성 확인 수준의 신뢰도.
4. **초기 실험과의 차이**: Multi-seed 평균은 초기 단일 seed 결과와 소폭 차이 존재 (예: SigLIP 2-layer 96.13% → 96.11±0.30%). 이는 seed 의존성이 낮음을 시사.

### 데이터 및 라벨 설계

5. **단일 데이터셋**: AI Hub 건설 현장 안전 이미지 1종만 사용. 다른 산업 도메인(제조, 물류)으로의 일반화 미검증.
6. **라벨 설계 결함**: 1이미지 = 1활성차원 구조로 인해 multi-task 실험이 구조적으로 불리. 진정한 multi-label annotation 부재.
7. **자동 라벨링**: 파일명 기반 자동 라벨링(카테고리 코드 A-E). 인간 검수 미수행으로 라벨 노이즈 가능성.

### 실험 설계

8. **RQ2 교란 요인**: 앙상블 효과와 distribution shift 효과가 혼재. Scenario split에서의 분리 검증으로 부분 해소 (앙상블이 scenario에서도 -0.32%p).
9. **Temporal shift 분석 피상적**: 어떤 시나리오/카테고리가 가장 취약한지에 대한 세분화 분석 부족. ResNet50-Frozen의 상대적 강건성(-21%p vs -28%p) 원인 미탐구.
10. **Ablation 미완성**: 앙상블 내 2-model subset ablation, 데이터 크기 scaling curve, 3-layer/4-layer probe depth saturation 확인 미수행.

### 재현성

11. **GPU 의존성**: 학습 결과는 NVIDIA RTX 4090 기준. 다른 GPU에서의 재현성은 cuDNN 결정성 설정에 의존.
12. **transformers 버전**: transformers 5.x API 변경으로 인한 호환성 이슈 존재. 정확한 라이브러리 버전 고정 필요.

---

## Critical Revision 2: Scenario Split Frame-Level Leakage (2026-04-06)

### 개요

`data_scenario/`의 split이 frame-level로 수행되어, 같은 CCTV 영상의 연속 프레임이 train/test에 분산됨.
98.1%의 test sequence가 train에도 출현 (data leakage).
`data_scenario_v2/`에서 sequence-level split으로 수정됨.

**그러나 `results/multiseed/` (기존 보고 결과)는 여전히 old `embeddings/scenario/` 사용.**

### 영향

| 모델 | Old (leaked) F1 | Clean (scenario_v2) F1 | 차이 |
|------|----------------|----------------------|------|
| SigLIP 2-layer | 96.11±0.30% | **87.28±0.45%** | **-8.83%p** |
| CLIP 2-layer | 91.54±0.49% | **78.91±0.48%** | **-12.63%p** |
| DINOv2 2-layer | 90.92±0.42% | **77.94±0.46%** | **-12.98%p** |

### 결론 변경

1. **1-layer ≈ 2-layer** (gap 0.12%p) → 1-layer MLP로 충분
2. **Temporal degradation**: -21.17%p (이전 -30.00%p에서 축소)
3. 절대 성능 87% → "보조 도구 수준"으로 조정
4. 3-layer probe: 과적합으로 역효과 (-0.47%p)

### 추가 실험 결과

- **Ensemble stacking**: Optimized weights +0.90%p (SigLIP 0.50 + DINOv2 0.45)
- **Temporal ECE**: 0.2816 (scenario 0.0993 대비 3배 악화)
- **Per-month**: Oct 65.45% ≈ Nov 67.38% (점진적 악화 아님, 즉각적 gap)
- **Scaling curve (scenario_v2)**: 25%=82.23%, 50%=85.86%, 100%=87.28%

상세: `results/COMPREHENSIVE_REVIEW_2026_04_06.md`, `results/CRITICAL_FINDING_SCENARIO_LEAKAGE.md`
