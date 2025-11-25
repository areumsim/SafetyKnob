# SafetyKnob API 레퍼런스

## 개요

SafetyKnob은 FastAPI 기반의 REST API를 제공합니다. 모든 응답은 JSON 형식입니다.

## 기본 정보

- **Base URL**: `http://localhost:8000`
- **API Version**: v1
- **API Prefix**: `/api/v1`

## 인증

현재 버전은 인증을 요구하지 않습니다. 프로덕션 환경에서는 적절한 인증 메커니즘을 추가하세요.

## 엔드포인트

### 1. 서버 상태 확인

서버의 건강 상태를 확인합니다.

**요청**
```http
GET /api/v1/health
```

**응답**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "version": "1.0.0",
  "models_loaded": 3
}
```

### 2. 이미지 안전성 평가

이미지를 업로드하여 안전성을 평가합니다.

**요청**
```http
POST /api/v1/assess
Content-Type: multipart/form-data

file: <image_file>
```

**매개변수**
- `file` (required): 평가할 이미지 파일 (JPEG, PNG, BMP 지원)

**응답 (성공)**
```json
{
  "success": true,
  "result": {
    "image_path": "uploaded_image.jpg",
    "is_safe": false,
    "overall_safety_score": 0.23,
    "confidence": 0.89,
    "risk_summary": "Image assessed as UNSAFE. Risk factors: Fall Hazard (78% risk)",
    "dimension_scores": {
      "fall_hazard": 0.22,
      "collision_risk": 0.45,
      "equipment_hazard": 0.35,
      "environmental_risk": 0.68,
      "protective_gear": 0.15
    },
    "method_used": "ensemble",
    "model_name": "ensemble_all",
    "processing_time": 0.284
  }
}
```

**응답 (실패)**
```json
{
  "success": false,
  "error": "Invalid image format",
  "detail": "Supported formats: JPEG, PNG, BMP"
}
```

**상태 코드**
- `200 OK`: 성공적으로 평가됨
- `400 Bad Request`: 잘못된 요청 (파일 없음, 잘못된 형식 등)
- `500 Internal Server Error`: 서버 오류

### 3. 시스템 정보 및 모델 목록

시스템 정보와 현재 로드된 모델 목록을 반환합니다.

**요청**
```http
GET /api/v1/info
```

**별칭 (계획됨)**
```http
GET /api/v1/models
```
*참고*: `/api/v1/models`는 `/api/v1/info`의 alias로 제공될 예정입니다. 현재는 `/api/v1/info`를 사용하세요.

**응답**
```json
{
  "system": {
    "name": "industrial_safety_assessment",
    "version": "1.0.0",
    "description": "Industrial image safety assessment using pre-trained vision models"
  },
  "models": [
    {
      "name": "siglip",
      "model_type": "siglip",
      "checkpoint": "google/siglip-so400m-patch14-384",
      "embedding_dim": 1152,
      "device": "cuda",
      "status": "loaded"
    },
    {
      "name": "clip",
      "model_type": "clip",
      "checkpoint": "openai/clip-vit-large-patch14",
      "embedding_dim": 768,
      "device": "cuda",
      "status": "loaded"
    },
    {
      "name": "dinov2",
      "model_type": "dinov2",
      "checkpoint": "facebook/dinov2-giant",
      "embedding_dim": 1536,
      "device": "cuda",
      "status": "loaded"
    }
  ],
  "assessment_method": "ensemble",
  "ensemble_strategy": "weighted_vote",
  "safety_dimensions": {
    "fall_hazard": {"weight": 1.0, "description": "추락 위험"},
    "collision_risk": {"weight": 1.0, "description": "충돌 위험"},
    "equipment_hazard": {"weight": 1.0, "description": "장비 위험"},
    "environmental_risk": {"weight": 0.8, "description": "환경적 위험"},
    "protective_gear": {"weight": 0.8, "description": "보호구 착용"}
  }
}
```

### 4. 배치 평가 (✅ 구현 완료)

여러 이미지를 한 번에 평가합니다.

**요청**
```http
POST /api/v1/assess/batch
Content-Type: multipart/form-data

files: <multiple_image_files>
```

**매개변수**
- `files` (required): 평가할 이미지 파일 목록 (여러 파일 업로드)

**응답 (성공)**
```json
{
  "success": true,
  "total_count": 3,
  "safe_count": 1,
  "unsafe_count": 2,
  "results": [
    {
      "filename": "image1.jpg",
      "is_safe": true,
      "overall_safety_score": 0.85,
      "confidence": 0.92
    },
    {
      "filename": "image2.jpg",
      "is_safe": false,
      "overall_safety_score": 0.25,
      "confidence": 0.88
    }
  ],
  "total_processed": 2,
  "total_time": 0.568
}
```

## 에러 처리

모든 에러 응답은 다음 형식을 따릅니다:

```json
{
  "success": false,
  "error": "에러 타입",
  "detail": "상세 에러 메시지",
  "timestamp": "2024-01-15T10:30:00"
}
```

### 일반적인 에러 코드

| 코드 | 설명 |
|------|------|
| 400 | 잘못된 요청 (파일 없음, 형식 오류 등) |
| 413 | 파일 크기 초과 (기본 제한: 10MB) |
| 415 | 지원하지 않는 미디어 타입 |
| 500 | 서버 내부 오류 |
| 503 | 서비스 이용 불가 (모델 로딩 중 등) |

## 사용 예시

### Python (requests)
```python
import requests

# 서버 상태 확인
response = requests.get("http://localhost:8000/api/v1/health")
print(response.json())

# 이미지 평가
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/api/v1/assess", 
        files=files
    )
    result = response.json()
    
    if result["success"]:
        assessment = result["result"]
        print(f"안전성: {'안전' if assessment['is_safe'] else '위험'}")
        print(f"신뢰도: {assessment['confidence']:.2%}")
```

### cURL
```bash
# 서버 상태 확인
curl http://localhost:8000/api/v1/health

# 이미지 평가
curl -X POST \
  -F "file=@test_image.jpg" \
  http://localhost:8000/api/v1/assess
```

### JavaScript (fetch)
```javascript
// 이미지 평가
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/api/v1/assess', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    if (data.success) {
        const assessment = data.result;
        console.log(`안전성: ${assessment.is_safe ? '안전' : '위험'}`);
        console.log(`신뢰도: ${(assessment.confidence * 100).toFixed(1)}%`);
    }
});
```

## 성능 고려사항

### 권장 사항
1. **이미지 크기**: 최적 성능을 위해 1920x1080 이하 권장
2. **파일 형식**: JPEG 형식이 가장 빠른 처리 속도
3. **배치 처리**: 대량 처리시 배치 API 사용 권장
4. **연결 재사용**: Keep-Alive 헤더 사용으로 성능 향상

### 제한 사항
- 최대 파일 크기: 10MB (설정 가능)
- 동시 요청 수: 기본 100개 (워커 수에 따라 조정)
- 타임아웃: 30초

## 웹소켓 지원 (계획 중)

실시간 스트리밍 평가를 위한 웹소켓 엔드포인트:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream');

ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    // 실시간 결과 처리
};
```

## 보안 고려사항

프로덕션 환경에서는 다음을 고려하세요:

1. **HTTPS 사용**: SSL/TLS 인증서 적용
2. **API 키 인증**: 헤더 기반 인증 추가
3. **Rate Limiting**: 과도한 요청 방지
4. **CORS 설정**: 허용된 도메인만 접근
5. **입력 검증**: 파일 타입 및 크기 엄격히 검증