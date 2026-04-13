# AI Hub 데이터셋 다운로드 가이드

SafetyKnob 프로젝트의 기존 실험을 재현하기 위해 AI Hub "Construction Site Safety(Action) Image" 데이터셋을 다운로드해야 합니다.

---

## 1. 데이터셋 정보

- **데이터셋 이름**: Construction Site Safety(Action) Image
- **제공기관**: 한국정보화진흥원 (NIA)
- **URL**: https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71407
- **크기**: 약 20GB (압축), 30GB (압축 해제)
- **파일 수**: 52,418장 원본 이미지
- **라이선스**: 오픈 데이터 라이선스 (자유이용 가능)

---

## 2. 다운로드 절차

### 2.1 AI Hub 회원가입

1. AI Hub 웹사이트 방문: https://www.aihub.or.kr
2. 우측 상단 "회원가입" 클릭
3. 개인정보 입력 및 이용약관 동의
4. 이메일 인증 완료

### 2.2 데이터셋 검색 및 다운로드

1. **데이터셋 페이지 접속**:
   https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71407

2. **로그인** 후 "데이터 다운로드" 버튼 클릭

3. **다운로드 신청**:
   - 이용목적: 연구용
   - 활용분야: 안전/보안
   - 기타 정보 입력

4. **승인 대기**:
   - 일반적으로 1-2일 소요
   - 승인 완료 시 이메일 알림

5. **다운로드 실행**:
   - 승인 후 "다운로드" 버튼 활성화
   - 파일 선택:
     ```
     ✅ Training Images (SO-35~SO-43)
     ✅ Validation Images (SO-44~SO-46)
     ✅ Test Images (SO-47)
     ✅ Labels (labels.json)
     ```

### 2.3 압축 해제

다운로드한 파일을 프로젝트 디렉토리로 이동하고 압축 해제:

```bash
# 다운로드 위치에서 실행 (예: ~/Downloads/)
cd ~/Downloads

# 프로젝트 data/raw/ 디렉토리로 이동
mv Construction_Safety_*.zip <프로젝트경로>/data/raw/

# 압축 해제
cd <프로젝트경로>/data/raw/
unzip Construction_Safety_Images.zip

# 결과 확인
ls -lh
# 예상 출력:
# SO-35/ (비계 조립)
# SO-36/
# ...
# SO-47/ (현장 준비)
# labels.json
```

---

## 3. 데이터 전처리

압축 해제 후 전처리 스크립트 실행:

```bash
cd <프로젝트경로>

# 전처리 스크립트 실행
python scripts/prepare_dataset.py \
  --input data/raw/ \
  --output data/processed/ \
  --train-scenarios SO-35,SO-36,SO-37,SO-38,SO-41,SO-42,SO-43 \
  --val-scenarios SO-44,SO-45,SO-46 \
  --test-scenarios SO-47

# 예상 소요 시간: 10-15분
```

### 3.1 예상 결과

```
data/processed/
├── train/       # 8,108장
│   ├── H-220607_E16_N-42_001_0001.jpg
│   ├── H-220607_E16_N-42_001_0011.jpg
│   └── ...
├── val/         # 1,738장
│   └── ...
├── test/        # 1,737장
│   └── ...
└── labels.json  # 5차원 라벨 정보
```

### 3.2 데이터 검증

```bash
# 데이터 개수 확인
python -c "
import json
from pathlib import Path

data_dir = Path('data/processed')
print(f'Train: {len(list((data_dir / \"train\").glob(\"*.jpg\")))}')
print(f'Val:   {len(list((data_dir / \"val\").glob(\"*.jpg\")))}')
print(f'Test:  {len(list((data_dir / \"test\").glob(\"*.jpg\")))}')

# 라벨 확인
with open(data_dir / 'labels.json') as f:
    labels = json.load(f)
print(f'Total labels: {len(labels)}')
"

# 예상 출력:
# Train: 8108
# Val:   1738
# Test:  1737
# Total labels: 11583
```

---

## 4. 문제 해결

### Q1: 다운로드 승인이 안 됩니다
**A**: AI Hub 고객센터에 문의 (help@aihub.or.kr)
- 승인은 보통 영업일 기준 1-2일 소요
- 주말/공휴일 제외

### Q2: 파일이 너무 큽니다
**A**: 시나리오별로 나누어 다운로드 가능
- 최소 요구사항:
  - Training: SO-35, SO-41, SO-42, SO-43 (핵심 시나리오)
  - Validation: SO-44
  - Test: SO-47
  - 약 10GB 정도로 축소 가능

### Q3: labels.json이 없습니다
**A**: 별도로 요청해야 할 수 있음
- AI Hub 데이터셋 페이지에서 "라벨 데이터" 별도 다운로드
- 또는 고객센터 문의

### Q4: 압축 해제 공간이 부족합니다
**A**: 최소 50GB 여유 공간 필요
```bash
# 현재 디스크 사용량 확인
df -h /workspace

# 불필요한 파일 정리
# (예: 다른 프로젝트 데이터, 캐시 등)
```

---

## 5. 데이터 다운로드 완료 후

다음 명령으로 다음 단계 진행:

```bash
# 기존 모델로 테스트셋 검증
python experiments/validate_checkpoint.py \
  --model siglip \
  --checkpoint results/single_models/siglip/best_model.pt \
  --data-dir data/processed/test

# 예상 결과: F1 95.5%, AUC 99.1%
```

---

## 6. 대안: 샘플 데이터로 테스트

전체 데이터셋 다운로드가 어려운 경우, 샘플 데이터로 테스트 가능:

```bash
# danger_al 데이터로 먼저 실험

# Binary classification으로 재훈련
python experiments/train_standalone.py \
  --model siglip \
  --data-dir data \
  --binary-only

# Phase 2로 진행
```

---

## 요약

1. ✅ AI Hub 회원가입
2. ✅ 데이터셋 다운로드 신청
3. ⏳ 승인 대기 (1-2일)
4. ✅ 다운로드 및 압축 해제
5. ✅ 전처리 스크립트 실행
6. ✅ 데이터 검증
7. ✅ 다음 단계 진행

**예상 소요 시간**: 승인 1-2일 + 다운로드 1-2시간 + 전처리 10-15분
