# SafetyKnob — 프로젝트 현황 및 실행 계획

**최종 업데이트**: 2025-11-28
**프로젝트 단계**: Stage 1 완료, Stage 2 부분 완료, Stage 3 계획 중

---

## 📋 프로젝트 개요

### 연구 동기
- 이미지에서 잠재적 위험을 탐지하여 산업 재해를 사전에 방지
- 사전 학습된 Vision Foundation Models(CLIP, SigLIP, DINOv2)로 적은 데이터로 높은 성능 달성
- 이진 분류를 넘어 5차원 안전 평가(낙상/충돌/장비/환경/보호구)로 설명 가능성 제공

### 핵심 연구 질문
- **RQ1**: 사전학습 임베딩 공간에서 안전/위험이 선형 분리 가능한가? → ✅ **입증됨** (95.73% F1)
- **RQ2**: 다중 모델 앙상블이 distribution shift에 강건한가? → ❌ **실패** (Temporal split에서 미검증)
- **RQ3**: 5차원 독립 학습이 전체 안전성 예측을 향상시키는가? → ⚠️ **부분 성공** (Overall 유지, 차원 간 불균형 존재)

### 설계 원칙
- **구성 우선**(config.json): 모델/체크포인트/차원/임계값/학습 파라미터
- **확장성**: 모델/차원 추가 용이, 코드 수정 없이 체크포인트 교체
- **성능**: GPU 가속, 캐싱, 배치 평가
- **가시성**: health/info 엔드포인트, 지표, 재현 가능한 벤치마크

---

## 🎯 현재 프로젝트 상태

### ✅ 완료된 작업 (Stage 1 - 기본 Binary Classification)

#### 1. Scenario 실험 (Random Split)
- [x] **데이터셋 구성**: Train 11,670 / Val 2,500 / Test 2,500개
- [x] **SigLIP 학습 완료**: F1 95.73%, Acc 95.78%, AUC 99.16% (학습 시간: 11.2h)
- [x] **CLIP 학습 완료**: F1 90.41%, Acc 90.46%, AUC 97.06% (학습 시간: 3.95h)
- [x] **DINOv2 학습 완료**: F1 88.68%, Acc 88.81%, AUC 96.41% (학습 시간: 4.21h)
- [x] **ResNet50 Baseline**: F1 95.49%, Acc 95.56%, AUC 99.22% (학습 시간: 48min)
- [x] **EfficientNet Baseline**: F1 94.70%, Acc 94.73%, AUC 99.10% (학습 시간: 55min)
- [x] **결과 분석**: Foundation model과 Baseline 성능 근소한 차이 (0.24%p)

**주요 발견**:
- ✅ RQ1 답변: Frozen embedding + Linear probe만으로 95.73% F1 달성
- ⚠️ Trade-off: SigLIP(11.2h) vs ResNet50(48min) - 학습 시간 대비 성능 향상 미미

#### 2. Caution 경계 분석
- [x] **Caution Excluded 데이터셋**: Train 9,756 / Val 2,091 / Test 2,091개
- [x] **SigLIP**: F1 95.09%, Acc 94.98%, AUC 99.14%
- [x] **CLIP**: F1 90.45%, Acc 90.05%, AUC 97.03%
- [x] **DINOv2**: F1 89.07%, Acc 88.86%, AUC 96.20%
- [x] **599개 Caution 이미지 분석**: 81.2% Unsafe 예측, 18.8% Safe 예측

**주요 발견**:
- 모델들은 애매한 상황을 **Unsafe로 보수적 판단** (안전 우선 전략)
- 평균 Confidence 91-95%로 높은 확신도

#### 3. Ensemble 실험
- [x] **Weighted Vote 앙상블**: SigLIP + CLIP + DINOv2
- [x] **결과**: F1 94.37%, Acc 94.46%

**주요 발견**:
- ❌ 앙상블(94.37%) < 최고 단일 모델 SigLIP(95.73%)
- 원인: 모든 모델이 이미 높은 성능 → 에러 다양성 부족
- 결론: Random split에서는 단일 최고 모델이 더 효율적

#### 4. Multi-task Learning 실험 (5차원 안전 평가)
- [x] **5D 라벨 자동 생성**: 파일명 기반 14,537개 이미지에 5차원 라벨 부여
- [x] **SigLIP 5D 모델 학습**: Overall F1 94.95%, Acc 95.05%, AUC 99.01%
- [x] **차원별 성능**:
  - Collision Risk: F1 98.24%, Acc 98.63% ✅
  - Environmental Risk: F1 94.74%, Acc 95.54% ✅
  - Protective Gear: F1 90.50%, Acc 91.91% ⚠️
  - Fall Hazard: F1 76.54%, Acc 75.11% ⚠️
  - Equipment Hazard: F1 75.69%, Acc 73.72% ⚠️

**주요 발견**:
- ✅ Overall 성능 유지: 5D(95.05%) ≈ Binary(95.73%)
- ✅ 차원별 해석 가능성 향상
- ⚠️ 차원 간 불균형: Collision(98.24%) vs Fall(76.54%), 21.7%p 격차
- ⚠️ High Recall 전략: Fall/Equipment에서 recall >91% (위험 최소화)

#### 5. 종합 연구 보고서 작성
- [x] **COMPREHENSIVE_RESEARCH_REPORT.md 작성**: 8개 섹션, ~70KB
- [x] **모든 실험 결과 검증**: 20개 results.json 파일 확인
- [x] **RQ별 평가 완료**: RQ1(✅), RQ2(❌), RQ3(⚠️)
- [x] **README 정렬도 평가**: 67.5% 달성
- [x] **12개월 연구 로드맵 작성**: Domain Adaptation, Temporal Ensemble 등

---

### ⚠️ 진행 중/부분 완료 작업 (Stage 2)

#### 1. Temporal Split 실험 (Distribution Shift 검증)
- [x] **데이터셋 구성**: Train 8,006개(2022-06~09) / Val 1,412개 / Test 5,119개(2022-10~11)
- [x] **SigLIP**: Val F1 95.49%, **Test F1 66.19%** (Gap: -28.88%p) 🚨
- [x] **CLIP**: Val F1 90.51%, **Test F1 58.18%** (Gap: -31.16%p) 🚨
- [x] **DINOv2**: Val F1 88.94%, **Test F1 59.05%** (Gap: -31.24%p) 🚨

**치명적 발견**:
- 🚨 **RQ2 실패**: Distribution shift에 매우 취약 (평균 30%p 성능 하락)
- 🚨 **Real-world Deployment 불가**: 현재 모델로는 시간적 변화에 대응 불가
- ❌ **Temporal Ensemble 미검증**: 단일 모델도 성능 저하 심각, 앙상블 실험 미수행

**원인 분석**:
1. 시간적 변화: 조명, 날씨, 계절 변화
2. 행동 패턴 변화: 작업자 행동, 안전 장비 사용 변화
3. 카메라 각도/위치: 설치 위치 조정
4. 일반화 실패: Foundation model도 domain shift에 취약

#### 2. 5D 라벨 품질 검증
- [x] 파일명 기반 자동 라벨링 완료
- [ ] **인간 검수 미수행** ⚠️
- [ ] 실제 안전 평가와 정확도 차이 측정 필요

---

### ❌ 미완료/차단된 작업

#### 1. RQ2 완전 검증 (앙상블의 강건성)
- [ ] **Temporal split에서 앙상블 실험** (Critical)
  - 현재 상태: 단일 모델도 60-67% F1로 실패
  - 필요 작업: Domain adaptation 후 앙상블 재시도
  - 차단 이유: 모든 모델이 temporal shift에서 성능 급락

#### 2. Real-world Deployment
- [ ] **Distribution shift 대응** (Blocker)
  - 현재 상태: 30%p 성능 하락으로 배포 불가
  - 필요 작업: Domain adaptation 기법 적용
  - 우선순위: **최고 우선순위**

---

## 🚀 다음 단계 실행 계획

### Phase 1: 보고서 검토 및 피드백 (1주)

**목표**: 연구 결과 검증 및 추가 분석 방향 결정

- [ ] **종합 연구 보고서 리뷰**
  - `results/COMPREHENSIVE_RESEARCH_REPORT.md` 검토
  - RQ1, RQ2, RQ3 평가 재확인
  - Temporal shift 원인 분석 심화

- [ ] **추가 데이터 분석**
  - [ ] Temporal split 오류 케이스 분석 (scripts/analyze_errors.py 활용)
  - [ ] Caution 이미지 599개 오분류 패턴 분석
  - [ ] 5D 차원별 혼동 행렬 생성 및 불균형 원인 파악

- [ ] **시각화 보완**
  - [ ] t-SNE: Temporal train vs test 임베딩 공간 시각화
  - [ ] Confusion matrix: 각 실험별 생성
  - [ ] ROC curve: 모델 간 비교 차트

**성공 기준**:
- Temporal shift 원인 3가지 이상 구체화
- 5D 차원 불균형 해결 방안 도출
- 다음 연구 우선순위 확정

---

### Phase 2: 추가 분석 및 수정 (2주)

**목표**: 현재 모델의 한계 극복 및 개선 방안 실험

#### 높은 우선순위 (Week 1-2)

- [ ] **Distribution Shift 대응 실험** 🚨 (Critical Path)
  - [ ] **Domain Adaptation 기법 적용**
    - [ ] DANN (Domain Adversarial Neural Network) 구현
    - [ ] CORAL (Correlation Alignment) 구현
    - [ ] Test-Time Adaptation (TTA) 실험
    - 목표: Temporal test F1 66% → 80%+ 향상

  - [ ] **Temporal Augmentation**
    - [ ] 조명 변화 시뮬레이션 (brightness, contrast)
    - [ ] 계절 변화 시뮬레이션 (color shift)
    - [ ] 카메라 각도 변화 augmentation
    - 목표: Val-Test gap 28%p → 10%p 이하 감소

  - [ ] **Continual Learning**
    - [ ] EWC (Elastic Weight Consolidation) 적용
    - [ ] Progressive learning: Train(06-09) → Fine-tune(10월 일부) → Test(11월)
    - 목표: Catastrophic forgetting 방지

- [ ] **5D 라벨 품질 개선**
  - [ ] **인간 검수**
    - [ ] 전문가 검수: 차원별 100개씩 샘플링 (총 500개)
    - [ ] Inter-annotator agreement 측정 (Cohen's Kappa)
    - [ ] 불일치 케이스 분석 및 라벨 규칙 개선

  - [ ] **Active Learning**
    - [ ] 모델 불확실성 높은 100개 이미지 선정
    - [ ] 인간 라벨링 후 재학습
    - 목표: Fall/Equipment F1 76% → 85%+ 향상

  - [ ] **Class Imbalance 처리**
    - [ ] Focal Loss 적용 (차원별 독립 weight)
    - [ ] 차원별 threshold 독립 조정 (ROC curve 기반)
    - [ ] SMOTE/oversampling: Fall/Equipment 샘플 증강

- [ ] **Ensemble 재실험**
  - [ ] **Random split 앙상블 개선**
    - [ ] Stacking 전략 (Meta-learner: XGBoost)
    - [ ] Boosting 전략 (AdaBoost, Gradient Boosting)
    - [ ] Uncertainty-based voting (entropy weighting)
    - 목표: 94.37% → 96%+ (단일 최고 모델 초과)

  - [ ] **Temporal split 앙상블** (Domain adaptation 후)
    - [ ] 3-model ensemble: SigLIP + CLIP + DINOv2
    - [ ] Temporal-aware weighting (시간 가중치)
    - 목표: RQ2 재검증 (앙상블 > 단일 모델 입증)

#### 중간 우선순위 (Week 2)

- [ ] **설명 가능성 향상**
  - [ ] Grad-CAM 시각화: 차원별 attention map 생성
  - [ ] SHAP values: 픽셀별 기여도 분석
  - [ ] 오분류 케이스 리포트: Top 50 실패 케이스 시각화

- [ ] **Few-shot Learning 탐색**
  - [ ] Prototypical Networks 구현
  - [ ] MAML (Model-Agnostic Meta-Learning) 실험
  - 목표: 차원별 소량 데이터(50개)로 학습 가능성 검증

**성공 기준**:
- Temporal test F1 80%+ 달성 (Domain adaptation)
- 5D 차원별 F1 85%+ 달성 (라벨 품질 + Class imbalance)
- Ensemble이 단일 모델보다 2%p 이상 우수

---

### Phase 3: 향후 연구 (3-12개월)

**목표**: Real-world deployment 및 연구 기여 극대화

#### 3-6개월 (Short-term)

- [ ] **Multi-domain Training**
  - [ ] 다양한 산업 현장 데이터 수집 (건설/제조/물류)
  - [ ] Domain-specific head 추가
  - [ ] Domain generalization 검증

- [ ] **Temporal Ensemble 완전 검증**
  - [ ] 12개월 이상 장기 데이터셋 구성
  - [ ] Seasonal drift 분석 (봄/여름/가을/겨울)
  - [ ] RQ2 최종 답변: 앙상블의 시간적 강건성 입증

- [ ] **Real-time Inference 최적화**
  - [ ] TensorRT/ONNX 변환
  - [ ] Knowledge distillation (SigLIP → MobileNet)
  - 목표: Latency 500ms → 50ms 감소

#### 6-12개월 (Long-term)

- [ ] **Foundation Model Fine-tuning**
  - [ ] PEFT (Parameter-Efficient Fine-Tuning): LoRA, Adapter
  - [ ] Prompt tuning: Vision-language prompting
  - 목표: Frozen embedding 한계 극복

- [ ] **Video-based Safety Assessment**
  - [ ] Temporal modeling: 3D CNN, Transformer
  - [ ] Action recognition: 위험 행동 탐지
  - [ ] Trajectory prediction: 사고 예측

- [ ] **Human-in-the-Loop System**
  - [ ] 저신뢰 예측 시 인간 검토 요청
  - [ ] Feedback loop: 검수 결과로 재학습
  - [ ] Continual learning pipeline 구축

**성공 기준**:
- 3개 이상 도메인에서 F1 90%+ (Domain generalization)
- RQ2 완전 해결: Temporal ensemble 2%p 이상 우수
- Real-time inference 50ms 이하

---

## 🛠️ 시스템 개선 작업 (Engineering)

### 높은 우선순위

- [ ] **구성 임계값 반영**
  - [ ] `assess_image`의 고정 0.5 제거, `config.safety.safety_threshold` 적용
  - [ ] `config.safety.confidence_threshold` 의미 정의/적용 (저신뢰 플래그)
  - 파일: `src/core/safety_assessment_system.py:SafetyAssessmentSystem.assess_image`

- [ ] **모델 체크포인트/임베딩 차원 반영**
  - [ ] `src/core/embedders.py`에서 per-model `checkpoint` 로드 지원
  - [ ] 실제 임베딩 차원과 `embedding_dim` 일치 검사 (경고/오류)
  - [ ] CLIP 모델 ID 정합: 코드(`-336`) vs 설정 통일
  - [ ] 임베딩 실패 시 모델별 차원 맞춤 또는 오류 전파

- [ ] **분류기 네이밍 충돌 해결**
  - [ ] `src/core/classifier.py:SafetyClassifier` → `SafetyClassifierLegacy` rename
  - [ ] `src/core/__init__.py`는 NN 분류기만 기본 export
  - [ ] Legacy는 `src/legacy/`로 이동

- [ ] **API 엔드포인트/문서 일관화**
  - [ ] `/api/v1/models` 추가 또는 `/api/v1/info`로 표준화
  - [ ] `docs/API_REFERENCE.md`에서 배치 평가 "지원" 업데이트
  - [ ] 앙상블 명칭 `weighted_vote`로 통일

- [ ] **API 보안/안정화**
  - [ ] `/tmp/{filename}` → UUID `NamedTemporaryFile` 또는 메모리 처리
  - [ ] 파일 크기/타입 엄격 검증, JSON 에러 표준화
  - [ ] 선택적 API key 인증/레이트리밋, CORS 제한

- [ ] **평가 효율성 개선**
  - [ ] (모델,이미지) 키의 임베딩 캐시로 재추출 방지
  - [ ] `ImageDataset`과 임베딩 경로 정렬 (파일 경로 우선)

### 중간 우선순위

- [ ] **차원 점수 방법론 통합**
  - [ ] `dimension_scoring: nn|similarity|hybrid` 모드 추가
  - [ ] 소규모 라벨셋으로 성능 비교 (AUC/AP 플롯)

- [ ] **확률 보정 및 불확실성 지표**
  - [ ] Temperature/Platt 보정 적용, 파라미터 저장
  - [ ] ECE/Brier 산출, `confidence`를 보정 확률 기반 제공

- [ ] **Typed config/유효성 검사**
  - [ ] `config.models`용 typed `ModelConfig` 도입 및 검증
  - [ ] `model_type` 표준화 맵 (eva_clip vs evaclip)
  - [ ] per-model vs 전역 디바이스 정책 결정/문서화

- [ ] **벤치마킹/로깅**
  - [ ] `scripts/benchmark.py`: 모델/앙상블 지연/처리량/메모리 측정
  - [ ] 구조적 로깅 (지연/메모리) 및 로그 레벨 구성화

- [ ] **테스트/재현성 확보**
  - [ ] 단위테스트: config 파싱, 임계값, 앙상블, API, MockEmbedder
  - [ ] 회귀테스트: 소형 라벨셋 end-to-end, CSV/JSON 검증
  - [ ] torch/numpy/python seed 고정, 결정적 플래그 문서화

### 낮은 우선순위

- [ ] **레거시 코드 정리**
  - [ ] `src/core/classifier.py`, `src/api/inference.py` → `src/legacy/`
  - [ ] 중복 앙상블 코드 경로 제거

- [ ] **임베딩 처리 품질 강화**
  - [ ] EXIF 회전/손상 이미지 처리
  - [ ] DINOv2 기본 백본 크기 조정 (giant → base)
  - [ ] 오프라인 체크포인트 가이드

- [ ] **문서 업데이트**
  - [ ] 5차원 `labels.json` 현실적 예시 (경계 사례)
  - [ ] "임계값 선택 방법" 섹션 (ROC/PR, 보정 플롯)
  - [ ] Dockerfile/환경변수 구성 가이드

---

## 📊 성공 기준

### Phase 1 (1주)
- ✅ Temporal shift 원인 3가지 구체화
- ✅ 5D 차원 불균형 해결 방안 도출
- ✅ 우선순위 확정 (Domain adaptation 최우선)

### Phase 2 (2주)
- ✅ **Temporal test F1 80%+** (Domain adaptation)
- ✅ **5D 차원별 F1 85%+** (라벨 품질 + Class imbalance)
- ✅ **Ensemble > 단일 모델** (2%p 이상 우수)

### Phase 3 (3-12개월)
- ✅ **Multi-domain F1 90%+** (3개 도메인)
- ✅ **RQ2 완전 해결** (Temporal ensemble 입증)
- ✅ **Real-time inference 50ms** (TensorRT 최적화)

### Engineering (2-4주)
- ✅ 구성 임계값 적용, 체크포인트 로딩
- ✅ API 보안/안정화 (UUID, 검증, 인증)
- ✅ 단위테스트 통과, 벤치마크 재현

---

## 📈 프로젝트 진행률

### README 정렬도 평가

| 목표 | 달성률 | 상태 |
|------|--------|------|
| **Stage 1**: Binary Classification | 100% | ✅ 완료 |
| **Stage 2**: 5D Expansion | 60% | ⚠️ 진행 중 |
| **Stage 3**: Distribution Shift | 20% | ❌ 차단됨 |
| **전체** | **67.5%** | ⚠️ 진행 중 |

### 핵심 가치 달성도

| 가치 | 달성률 | 근거 |
|------|--------|------|
| 적은 데이터로 높은 성능 | 100% | ✅ 14,537개로 95.73% F1 달성 |
| 다양한 환경 일반화 | 30% | ❌ Temporal shift 30%p 하락 |
| 설명 가능성 | 80% | ✅ 5D 차원별 독립 평가 가능 |

---

## 🔍 위험 관리

### 높은 위험 (Critical)

1. **Temporal Shift 30%p 하락** 🚨
   - 영향: Real-world deployment 불가
   - 완화: Domain adaptation 필수 (DANN, CORAL, TTA)
   - 대응: Phase 2에서 최우선 해결

2. **RQ2 미검증** ⚠️
   - 영향: 연구 기여 불완전
   - 완화: Temporal ensemble 실험 (Domain adaptation 후)
   - 대응: Phase 3에서 완전 해결

### 중간 위험

3. **5D 라벨 품질 미검증** ⚠️
   - 영향: 차원별 성능 신뢰도 하락
   - 완화: 인간 검수 500개 + Active learning
   - 대응: Phase 2에서 해결

4. **차원 간 불균형 (21.7%p)** ⚠️
   - 영향: Fall/Equipment 탐지 신뢰도 저하
   - 완화: Focal Loss + 독립 threshold
   - 대응: Phase 2에서 해결

### 낮은 위험

5. **Ensemble 효과 부족** (Random split)
   - 영향: 계산 비용 대비 이득 미미
   - 완화: Stacking/Boosting 시도
   - 대응: Phase 2 중간 우선순위

---

## 📝 실험 통계

### 총 GPU 시간
- **~120 GPU-hours**
- 모델 학습: 15개 모델 × 평균 6시간
- 데이터 준비: 5D 라벨 생성, Temporal split

### 사용 GPU
- NVIDIA GPU (CUDA 11.x)
- 동시 실행: 최대 4개 GPU (4, 5, 6, 7번)

### 최종 결과물
- 학습된 모델: 15개
- 실험 결과 JSON: 20개
- 문서: 5개 (COMPREHENSIVE_RESEARCH_REPORT, DATA_ANALYSIS, etc.)
- 스크립트: 20+ (train, evaluate, visualize, etc.)

---

## 🔗 참고 문서

- **종합 연구 보고서**: `results/COMPREHENSIVE_RESEARCH_REPORT.md`
- **README**: `README.md` (연구 목표 및 방향성)
- **API 문서**: `docs/API_REFERENCE.md`
- **실험 로그**: `logs/*.log`

---

## 📌 주요 파일 위치

### 실험 결과
- Scenario: `results/scenario/{siglip,clip,dinov2}/results.json`
- Baseline: `results/danger_al/baseline/{resnet50,efficientnet_b0}/results.json`
- Ensemble: `results/danger_al/ensemble/results.json`
- Temporal: `results/temporal/binary/{siglip,clip,dinov2}/results.json`
- Multi-task 5D: `results/multitask/siglip/results.json`
- Caution Excluded: `results/caution_excluded/{siglip,clip,dinov2}/results.json`

### 학습된 모델
- `results/*/best_model.pt`

### 로그 파일
- `logs/train_*.log`
- `logs/evaluate_*.log`

---

**최종 권장사항**:
1. **Random split** 환경: **SigLIP 단일 모델** 사용 (95.73% F1, 빠른 추론)
2. **Temporal shift** 환경: **Domain adaptation 필수** (현재 60-67% F1 → 80%+ 목표)
3. **설명 가능성** 필요: **Multi-task 5D** 모델 사용 (95.05% F1 + 차원별 해석)

SafetyKnob은 controlled 환경에서는 높은 성능을 보이나, **real-world deployment에는 Domain Adaptation이 필수**입니다.
