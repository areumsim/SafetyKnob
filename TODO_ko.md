## SafetyKnob — 의도, 검토, 실행 계획 (KR)

### 프로젝트 의도 요약

- 연구 동기
  - 이미지에서 잠재적 위험을 탐지하여 산업 재해를 사전에 방지.
  - 사전 학습 비전 백본(CLIP, SigLIP, DINOv2, EVA-CLIP)으로 일반화/데이터 효율성 확보.
  - 이진 분류를 넘어 낙상/충돌/장비/환경/보호구 등 다차원 안전 평가와 해석 가능성 제공.

- 실질적 엔지니어링 목표
  - 다음을 지원하는 구성 가능 시스템 구축:
    - 다중 백본 임베딩 추출(+캐싱) 및 경량 분류기 학습(차원별/전체 안전도 예측)
    - 앙상블(Weighted Vote, Stacking)로 정확도/강건성 향상
    - CLI(assess/train/evaluate/compare/experiment), FastAPI(실시간/배치)
    - 지표 산출 및 결과(JSON/CSV) 내보내기, 보고서/시각화(선택)

- 설계 원칙
  - 구성 우선(config.json): 모델/체크포인트/차원/임계값/학습 파라미터
  - 확장성: 모델/차원 추가 용이, 코드 수정 없이 체크포인트 교체
  - 성능: GPU 가속, 캐싱, 배치 평가
  - 가시성: health/info 엔드포인트, 지표, 재현 가능한 벤치마크

### 코드 vs. 의도 검토

- 강한 일치점
  - 오케스트레이션: `SafetyAssessmentSystem`이 임베더 → 모델별 NN 분류기 → 앙상블(`weighted_vote`, `stacking`)을 통합
  - 구성 기반: `SystemConfig.from_dict`가 `config.json`(모델/차원/전략/학습)을 런타임에 반영
  - CLI/API: `main.py` 전체 워크플로우, `src/api/server.py`의 `health/info/assess/assess-batch`
  - 캐싱 훅: 임베더의 파일 경로 기반 캐싱 옵션

- 부족/불일치
  - API 문서 vs 구현
    - 문서: `/api/v1/models` 언급 vs 서버: `/api/v1/info`로 모델 목록 제공. 문서엔 배치가 “계획”이지만 `/api/v1/assess/batch`는 이미 구현됨.
  - 임계값 사용
    - `assess_image`에서 고정 `> 0.5` 사용. `config.safety.safety_threshold/confidence_threshold` 미반영.
  - 체크포인트 처리
    - 임베더가 기본 체크포인트를 하드코딩, `config.json`의 `checkpoint` 무시 → 구성 불일치.
    - CLIP ID 불일치: 코드(`openai/clip-vit-large-patch14-336`) vs 설정(`openai/clip-vit-large-patch14`).
  - 클래스 이름 충돌
    - NN(신규)과 legacy 두 개의 `SafetyClassifier` 공존, `__init__.py`에서 모두 export → 혼란.
  - 차원 점수 통합 미흡
    - `DimensionAnalyzer`(유사도 기반)는 생성되나 추론 경로에 미연결. NN 헤드만 실제 사용.
  - 평가 비효율
    - `evaluate_dataset`에서 임베딩 재계산(assess + per-model loop 중복). `ImageDataset` 텐서 변환은 임베딩 경로에서 미사용.
  - 앙상블 명명 불일치
    - `weighted_vote` vs `weighted_voting` 혼용. `EnsembleConfig.method`는 `weighted_voting`을 참조.
  - API 파일 처리/보안
    - `/tmp/{filename}` 저장 시 동시성 충돌 위험. 크기/타입 검증 최소. CORS `*`, 인증/레이트리밋 부재.
  - 디바이스 구성
    - `config.models`의 per-model `device`가 무시되고 단일 디바이스가 적용됨.
  - 에러 처리/임베딩 차원
    - `extract_single_embedding` 실패 시 항상 768-dim zero 반환(모델별 임베딩 차원 무시).

### 피드백 (연구 + 엔지니어링 정렬)

- AI 연구 관점
  - 점수/임계값 정의
    - “안전 점수” vs “위험 점수(=1-risk)”를 명시하고 코드/지표/시각화 일치화.
    - ROC/PR 기반 임계값 탐색 스크립트 제공(전역/차원별 권장값 산출).
  - 확률 보정/불확실성
    - Temperature/Platt 보정 및 ECE/Brier 보고. “confidence”를 보정 확률에 연계.
    - 앙상블 합의도/분산/엔트로피 보고, 저합의 시 휴먼리뷰 정책 정의.
  - 차원 점수 방법론
    - `dimension_scoring: nn | similarity | hybrid` 모드 지원 및 소규모 라벨셋에서 비교.
  - 데이터/라벨 전략
    - 5차원 라벨링 가이드(경계 사례 포함). CLIP 프롬프트 기반 pseudo-label 옵션.
  - 설명가능성
    - Grad-CAM 등 시각화 및 차원별 설명을 리포트에 포함.

- 소프트웨어 엔지니어링 관점
  - 구성 충실도
    - `checkpoint` 반영 및 `embedding_dim` 검증(불일치 경고/오류). 임계값 일관 적용. per-model vs 전역 디바이스 결정/문서화.
  - 레거시 정리
    - legacy 분류기 rename(`SafetyClassifierLegacy`) 및 `src/legacy/`로 이동. 기본 export는 NN.
  - API 안정성/보안
    - `NamedTemporaryFile`/메모리 처리, 고유 파일명, 파일 크기/타입 검증, JSON 에러 표준화. 선택적 API key/레이트리밋, 운영 CORS 제한.
  - 평가 효율
    - (모델,이미지)별 임베딩 캐시로 중복 제거. `ImageDataset`을 파일 경로 중심으로 정렬.
  - 테스트/벤치마크/재현성
    - 단위테스트(config/threshold/ensemble/API), 소형 라벨셋, seed 고정. 성능 벤치마크 스크립트와 환경 메타데이터 기록.

### 실행 가능한 TODO

#### 높은 우선순위

- [ ] 구성 임계값 반영
  - [ ] `assess_image`의 고정 0.5 제거, `config.safety.safety_threshold` 적용.
  - [ ] `config.safety.confidence_threshold` 의미 정의/적용(저신뢰 플래그/게이팅 등) 및 평가/출력에 반영.

- [ ] 모델 체크포인트/임베딩 차원 반영
  - [ ] `src/core/embedders.py`에서 per-model `checkpoint` 로드 지원(SigLIP/CLIP/DINOv2/EVA-CLIP).
  - [ ] 실제 임베딩 차원과 `embedding_dim` 일치 검사(경고/오류).
  - [ ] CLIP 모델 ID 정합(설정과 코드 통일) 또는 로드된 모델에서 `embedding_dim` 자동 결정.
  - [ ] 임베딩 실패 시 zero 벡터는 모델별 차원에 맞추거나 오류 전파; 로깅 강화.

- [ ] 분류기 네이밍 충돌 해결
  - [ ] `src/core/classifier.py:SafetyClassifier` → `SafetyClassifierLegacy`로 rename(또는 `src/legacy/`로 이동).
  - [ ] `src/core/__init__.py`는 기본적으로 NN 분류기만 export(legacy는 새 이름으로 선택 export).

- [ ] API 엔드포인트/문서 일관화
  - [ ] 모델 정보 엔드포인트를 `/api/v1/models`로 추가하거나 `/api/v1/info`로 표준화하고 문서 갱신.
  - [ ] `docs/API_REFERENCE.md`에서 배치 평가를 “지원”으로 업데이트하고 실제 요청/응답/에러 예시 반영.
  - [ ] 앙상블 명칭을 `weighted_vote`로 통일(`EnsembleConfig.method` 포함).

- [ ] API 보안/안정화
  - [ ] `/tmp/{filename}` 대신 UUID `NamedTemporaryFile` 또는 메모리 처리 사용, 파일명 sanitize.
  - [ ] 파일 크기/타입 엄격 검증 및 표준화된 JSON 에러.
  - [ ] 선택적 API key 인증/레이트리밋 추가, 운영 환경 CORS 제한/HTTPS 권고 문서화.

- [ ] 평가 효율성 개선
  - [ ] 실행 중 (모델,이미지) 키의 임베딩 캐시로 재추출 방지.
  - [ ] `ImageDataset`과 임베딩 경로를 정렬(임베딩 필요 시 파일 경로 사용)하여 이중 로딩 제거.

#### 높은 우선순위 (현재 진행 상황 - 모델 훈련)

- [x] **SigLIP 모델 훈련 완료**
  - 훈련 시간: 약 5시간 (18,249초)
  - 테스트 성능: Accuracy: 95.3%, F1: 95.5%, AUC: 99.1%
  - 차원별 점수: protective_gear (84.8%), equipment_hazard (53.8%), collision_risk (44.5%), fall_hazard (41.0%), environmental_risk (35.5%)
  - 결과 저장 위치: `results/single_models/siglip/`

**향후 자동 실행 작업** (SigLIP 훈련 이후 순차적으로 실행 예정):
- [ ] CLIP 모델 훈련
- [ ] DINOv2 모델 훈련
- [ ] Ensemble 실험 (훈련된 모델 조합)
- [ ] 연구 분석 보고서 생성

#### 중간 우선순위

- [ ] 차원 점수 방법론 통합
  - [ ] `dimension_scoring: nn|similarity|hybrid` 모드 추가 및 시스템 연결, 소규모 라벨셋으로 성능 비교(AUC/AP 플롯).

- [ ] 확률 보정 및 불확실성 지표
  - [ ] 학습 후 Temperature/Platt 보정 적용, 파라미터 저장.
  - [ ] ECE/Brier 산출 및 `confidence`를 보정 확률 기반으로 제공(옵션).

- [ ] Typed config/유효성 검사
  - [ ] `config.models`용 typed `ModelConfig`(name, model_type, checkpoint, embedding_dim, device, cache_dir) 도입 및 검증.
  - [ ] `model_type` 표준화 맵(eva_clip vs evaclip, dinov2 vs dino) 적용 및 검증.
  - [ ] per-model vs 전역 디바이스 정책 결정/문서화 및 일관 적용.

- [ ] 벤치마킹/로깅
  - [ ] `scripts/benchmark.py`로 모델/앙상블 지연/처리량/메모리 측정, GPU/버전/설정 메타데이터와 함께 저장.
  - [ ] 지연/메모리 구조적 로깅 및 로그 레벨 구성화.

- [ ] 테스트/재현성 확보
  - [ ] 단위테스트: config 파싱, 임계값 로직, 앙상블 가중 업데이트, API 검증(정상/에러), MockEmbedder 결정론.
  - [ ] 회귀테스트: 소형 라벨셋으로 end-to-end assess/evaluate, CSV/JSON 출력 검증.
  - [ ] torch/numpy/python seed 고정 및 가능한 결정적 플래그 문서화.

#### 낮은 우선순위

- [ ] 레거시 코드 정리/구조화
  - [ ] `src/core/classifier.py`, `src/api/inference.py`의 미사용 부분을 `src/legacy/`로 이동하거나 Deprecated 표기.
  - [ ] 중복 앙상블 코드 경로 제거, 단일 구현 유지.

- [ ] 임베딩 처리 품질/예외 강화
  - [ ] EXIF 회전/손상 이미지 처리, 예외 메시지 개선.
  - [ ] DINOv2는 실용성을 위해 기본을 더 작은 백본으로, giant는 opt-in.
  - [ ] 오프라인 체크포인트/HF 캐시 경로 가이드 제공.

- [ ] 문서 업데이트
  - [ ] 5차원 `labels.json` 현실적 예시(경계 사례/가이드 포함).
  - [ ] “임계값 선택 방법” 섹션(ROC/PR, 보정 플롯)과 권고 워크플로 추가.
  - [ ] 배포용 Dockerfile/환경변수 구성 가이드, 운영 체크리스트(HTTPS, API keys, rate limiting, CORS).

### 구현 노트/파일 포인터

- 임계값
  - `src/core/safety_assessment_system.py`: `config.safety.safety_threshold/confidence_threshold` 적용, 지표/출력 반영.

- 임베더
  - `src/core/embedders.py`: `checkpoint` 사용, `embedding_dim` 검증, CLIP ID 정합, EXIF/에러 처리, zero 벡터 크기 수정.
  - `create_embedder`/`get_embedder`: `checkpoint`/`cache_dir` 전달, `model_type` 표준화.

- 분류기 네이밍/Export
  - `src/core/classifier.py`(legacy rename), `src/core/__init__.py`(export 정책).

- 앙상블 명칭/로직
  - `src/core/ensemble.py`: `weighted_vote`로 통일, `update_weights`/stacking 경로 정합성 확인.

- API/문서
  - `src/api/server.py`: 고유 임시파일/메모리 처리, `/api/v1/models`(옵션), 엄격 검증, 에러 표준화.
  - `docs/API_REFERENCE.md`: 엔드포인트/배치/에러 스키마 갱신. `README`/`DEVELOPMENT`의 성능 수치는 벤치마크 결과로 갱신.

- 평가 효율
  - `src/core/safety_assessment_system.py:evaluate_dataset`: 실행 내 임베딩 캐시, 중복 제거.
  - `src/utils/data_utils.py:ImageDataset`: 임베딩 워크플로에 맞게 파일 경로 우선.

### 성공 기준

- 기능
  - 구성 임계값 적용, 체크포인트 로딩, 동시성 안전한 추론, API/문서 일치.

- 연구
  - 보정된 확률/불확실성 지표 제공, 차원 점수 모드 비교/플롯, 임계값 추천 스크립트 산출.

- 엔지니어링
  - 네이밍 충돌 제거, 결정론적 테스트 통과, 벤치마크 재현, 구조적 로깅으로 운영 가시성 확보.
