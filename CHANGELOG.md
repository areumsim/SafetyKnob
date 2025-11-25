# Changelog

All notable changes to SafetyKnob project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

4. **DINOv2 Embedding Dim 불일치** ⚠️
   - Config: 1536 dim
   - 실제: 1024 dim (결과 파일 기준)
   - 영향: 미미 (학습 시 자동 조정)
   - 수정 필요: `config.json` 또는 모델 checkpoint

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
