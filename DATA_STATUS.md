# 데이터 상태 보고서

**작성일**: 2025-11-24
**프로젝트**: SafetyKnob

---

## 현재 상태

### ❌ 데이터 부재

프로젝트 디렉토리 `/workspace/arsim/EmoKnob/data/`가 **비어있습니다**.

```bash
$ ls -la data/
total 8
drwxr-xr-x  2 root root 4096 Oct 22 08:08 .
drwxrwxrwx 12 root root 4096 Oct  2 04:46 ..
```

### ✅ 보유 자산

다음 자산들은 정상적으로 보존되어 있습니다:

1. **훈련된 모델 체크포인트** (3개)
   - `/results/single_models/siglip/best_model.pt`
   - `/results/single_models/clip/best_model.pt`
   - `/results/single_models/dinov2/best_model.pt`

2. **테스트 결과** (JSON)
   - `/results/single_models/siglip/results.json`
   - `/results/single_models/clip/results.json`
   - `/results/single_models/dinov2/results.json`

3. **비교 분석 결과**
   - `/results/comparison/comparison_report.md`
   - `/results/comparison/comparison_summary.json`
   - `/results/comparison/*.png` (시각화 4개)

---

## 영향 분석

### ⚠️ 불가능한 작업

데이터 부재로 인해 다음 작업들을 수행할 수 없습니다:

1. **모델 재훈련**
   - 새로운 하이퍼파라미터로 실험 불가
   - 재현성 검증 불가

2. **추가 실험**
   - 앙상블 테스트 불가 (구현은 완료, 데이터 필요)
   - Baseline 모델 비교 불가
   - Cross-validation 불가

3. **데이터 증강**
   - SMOTE, Mixup 등 적용 불가
   - 차원별 성능 개선 실험 불가

### ✅ 가능한 작업

데이터 없이도 다음 작업들은 수행 가능합니다:

1. **추론 (Inference)**
   - 체크포인트로 새 이미지 평가 가능
   - API 서버 실행 가능
   - Demo 실행 가능

2. **분석**
   - 기존 결과 분석 및 시각화
   - 보고서 작성
   - 성능 비교

3. **코드 개선**
   - 버그 수정
   - 리팩토링
   - 문서화

---

## 원인 및 권장 조치

### 원인 추정

1. **데이터 삭제**: 의도적 또는 실수로 삭제됨
2. **미다운로드**: 처음부터 로컬에 저장하지 않음
3. **다른 경로**: 다른 위치에 저장되어 있을 수 있음

### 🔧 복구 절차

#### 1단계: 데이터셋 다운로드

AI Hub에서 공식 데이터셋을 다운로드하세요:

```bash
# AI Hub 웹사이트 방문
# URL: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71407
# "Construction Site Safety(Action) Image" 데이터셋 다운로드

# 다운로드 후 압축 해제
unzip Construction_Safety_Images.zip -d data/raw/
```

**필요 정보:**
- **데이터셋 이름**: Construction Site Safety(Action) Image
- **크기**: 약 20GB (압축), 30GB (압축 해제)
- **파일 수**: 52,418장 (원본)
- **라벨**: labels.json (별도 다운로드 필요)

#### 2단계: 데이터 전처리

```bash
# 전처리 스크립트 실행
python scripts/prepare_dataset.py \
  --input data/raw/ \
  --output data/processed/ \
  --train-scenarios SO-35,SO-36,SO-37,SO-38,SO-41,SO-42,SO-43 \
  --val-scenarios SO-44,SO-45,SO-46 \
  --test-scenarios SO-47

# 예상 결과:
# data/processed/
# ├── train/       # 8,108장
# ├── val/         # 1,738장
# ├── test/        # 1,737장
# └── labels.json  # 5차원 라벨
```

#### 3단계: 데이터 검증

```bash
# 무결성 확인
python scripts/verify_dataset.py --data-dir data/processed/

# 예상 출력:
# ✅ Train: 8,108 images
# ✅ Val: 1,738 images
# ✅ Test: 1,737 images
# ✅ Labels: 11,583 entries
# ✅ All files accessible
```

---

## 대안: 체크포인트로 추론 가능

데이터 없이도 훈련된 모델로 **새로운 이미지 평가**는 가능합니다.

### 단일 이미지 평가

```bash
# 체크포인트 로드 후 추론
python main.py assess path/to/new_image.jpg
```

### API 서버 실행

```bash
# API 서버 시작 (체크포인트 자동 로드)
python main.py serve --port 8000

# cURL로 테스트
curl -X POST "http://localhost:8000/api/v1/assess" \
  -F "file=@test_image.jpg"
```

### Demo 실행

```bash
# 데모 스크립트 실행
python demo.py --image path/to/image.jpg --model siglip
```

---

## 타임라인

| 단계 | 예상 시간 | 비고 |
|------|----------|------|
| 데이터셋 다운로드 | 1-3 시간 | 네트워크 속도에 따름 |
| 압축 해제 | 10-30분 | 52,418 파일 |
| 전처리 | 1-2시간 | 이미지 검증 포함 |
| **총 소요 시간** | **3-6시간** | - |

---

## 문의 사항

데이터 복구 중 문제가 발생하면:

1. **AI Hub 로그인 문제**: 회원가입 또는 비밀번호 재설정
2. **다운로드 실패**: 브라우저 변경 (Chrome → Firefox)
3. **전처리 오류**: `scripts/prepare_dataset.py --help` 확인
4. **파일 누락**: `scripts/verify_dataset.py` 실행 후 로그 확인

---

**작성자**: SafetyKnob Team
**마지막 업데이트**: 2025-11-24
