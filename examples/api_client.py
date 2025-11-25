#!/usr/bin/env python3
"""
SafetyKnob API 클라이언트 예시

이 스크립트는 SafetyKnob REST API를 사용하는 방법을 보여줍니다.
"""

import requests
import json
import sys
from pathlib import Path
import time


class SafetyKnobClient:
    """SafetyKnob API 클라이언트"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
    
    def check_health(self):
        """서버 상태 확인"""
        try:
            response = requests.get(f"{self.api_url}/health")
            return response.json()
        except requests.exceptions.ConnectionError:
            return {"status": "offline", "error": "서버에 연결할 수 없습니다"}
    
    def assess_image(self, image_path):
        """이미지 안전성 평가"""
        if not Path(image_path).exists():
            return {"error": f"이미지 파일을 찾을 수 없습니다: {image_path}"}
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{self.api_url}/assess", files=files)
        
        return response.json()
    
    def batch_assess(self, image_paths):
        """여러 이미지 일괄 평가"""
        results = []
        
        for path in image_paths:
            print(f"평가 중: {path}")
            result = self.assess_image(path)
            results.append({
                "image": path,
                "result": result
            })
            time.sleep(0.1)  # 서버 부하 방지
        
        return results
    
    def get_models(self):
        """사용 가능한 모델 목록 조회"""
        response = requests.get(f"{self.api_url}/models")
        return response.json()


def main():
    """메인 함수"""
    
    # 클라이언트 생성
    client = SafetyKnobClient()
    
    # 1. 서버 상태 확인
    print("=== 서버 상태 확인 ===")
    health = client.check_health()
    print(f"서버 상태: {health.get('status', 'unknown')}")
    
    if health.get('status') != 'healthy':
        print("서버가 실행 중이 아닙니다. 'python main.py serve'로 서버를 시작하세요.")
        return
    
    # 2. 사용 가능한 모델 확인
    print("\n=== 사용 가능한 모델 ===")
    models = client.get_models()
    for model in models.get('models', []):
        print(f"- {model['name']}: {model['type']} (dim={model['embedding_dim']})")
    
    # 3. 단일 이미지 평가
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\n=== 이미지 평가: {image_path} ===")
        
        result = client.assess_image(image_path)
        
        if 'error' in result:
            print(f"오류: {result['error']}")
        else:
            assessment = result.get('result', {})
            print(f"안전성: {'안전' if assessment.get('is_safe') else '위험'}")
            print(f"신뢰도: {assessment.get('confidence', 0):.2%}")
            print(f"전체 안전 점수: {assessment.get('overall_safety_score', 0):.2%}")
            
            print("\n차원별 점수:")
            for dim, score in assessment.get('dimension_scores', {}).items():
                risk_level = "안전" if score > 0.5 else "위험"
                print(f"  - {dim}: {score:.2f} ({risk_level})")
            
            print(f"\n위험 요약: {assessment.get('risk_summary', 'N/A')}")
            print(f"처리 시간: {assessment.get('processing_time', 0):.3f}초")
    
    # 4. 배치 처리 예시
    print("\n=== 배치 처리 예시 ===")
    print("여러 이미지를 처리하려면:")
    print("python api_client.py image1.jpg image2.jpg image3.jpg")


if __name__ == "__main__":
    main()