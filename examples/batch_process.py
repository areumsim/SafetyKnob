#!/usr/bin/env python3
"""
SafetyKnob 배치 처리 예시

대량의 이미지를 처리하고 결과를 CSV/JSON으로 저장하는 예시입니다.
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime
import argparse

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.safety_assessment_system import SafetyAssessmentSystem
from src.config.settings import SystemConfig


def process_directory(directory_path, output_format='csv'):
    """디렉토리 내 모든 이미지 처리"""
    
    # 설정 로드
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = SystemConfig.from_dict(config_dict)
    else:
        print("config.json을 찾을 수 없습니다.")
        return
    
    # 시스템 초기화
    print("시스템 초기화 중...")
    system = SafetyAssessmentSystem(config)
    
    # 이미지 파일 찾기
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(directory_path).glob(f"**/*{ext}"))
    
    print(f"{len(image_files)}개의 이미지를 찾았습니다.")
    
    # 결과 저장
    results = []
    start_time = datetime.now()
    
    for i, image_path in enumerate(image_files, 1):
        print(f"처리 중 [{i}/{len(image_files)}]: {image_path.name}")
        
        try:
            # 이미지 평가
            assessment = system.assess_image(str(image_path))
            
            # 결과 저장
            result = {
                'filename': image_path.name,
                'path': str(image_path),
                'is_safe': assessment.is_safe,
                'safety_score': assessment.overall_safety_score,
                'confidence': assessment.confidence,
                'processing_time': assessment.processing_time,
                'risk_summary': assessment.get_risk_summary(),
                'timestamp': datetime.now().isoformat()
            }
            
            # 차원별 점수 추가
            for dim, score in assessment.dimension_scores.items():
                result[f'dim_{dim}'] = score
            
            results.append(result)
            
        except Exception as e:
            print(f"  오류 발생: {e}")
            results.append({
                'filename': image_path.name,
                'path': str(image_path),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    # 처리 시간 계산
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n총 처리 시간: {total_time:.2f}초")
    print(f"평균 처리 시간: {total_time/len(image_files):.3f}초/이미지")
    
    # 결과 저장
    output_filename = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if output_format == 'csv':
        save_to_csv(results, f"{output_filename}.csv")
    elif output_format == 'json':
        save_to_json(results, f"{output_filename}.json")
    else:
        save_to_csv(results, f"{output_filename}.csv")
        save_to_json(results, f"{output_filename}.json")
    
    # 요약 통계
    print_summary(results)


def save_to_csv(results, filename):
    """결과를 CSV로 저장"""
    if not results:
        return
    
    # 모든 키 수집 (차원별 점수 포함)
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())
    
    # 정렬된 키 목록
    fieldnames = sorted(all_keys)
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"CSV 결과 저장: {filename}")


def save_to_json(results, filename):
    """결과를 JSON으로 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"JSON 결과 저장: {filename}")


def print_summary(results):
    """결과 요약 출력"""
    print("\n=== 처리 결과 요약 ===")
    
    # 성공/실패 카운트
    success_count = sum(1 for r in results if 'error' not in r)
    error_count = len(results) - success_count
    
    print(f"총 이미지: {len(results)}")
    print(f"성공: {success_count}")
    print(f"실패: {error_count}")
    
    if success_count > 0:
        # 안전/위험 카운트
        safe_count = sum(1 for r in results if r.get('is_safe', False))
        danger_count = success_count - safe_count
        
        print(f"\n안전: {safe_count} ({safe_count/success_count*100:.1f}%)")
        print(f"위험: {danger_count} ({danger_count/success_count*100:.1f}%)")
        
        # 평균 점수
        avg_safety_score = sum(r.get('safety_score', 0) for r in results if 'error' not in r) / success_count
        avg_confidence = sum(r.get('confidence', 0) for r in results if 'error' not in r) / success_count
        
        print(f"\n평균 안전 점수: {avg_safety_score:.2%}")
        print(f"평균 신뢰도: {avg_confidence:.2%}")
        
        # 가장 위험한 이미지들
        dangerous_images = sorted(
            [r for r in results if 'error' not in r and not r.get('is_safe', True)],
            key=lambda x: x.get('safety_score', 1)
        )[:5]
        
        if dangerous_images:
            print("\n가장 위험한 이미지들:")
            for img in dangerous_images:
                print(f"  - {img['filename']}: {img.get('safety_score', 0):.2%}")


def main():
    parser = argparse.ArgumentParser(description='SafetyKnob 배치 처리')
    parser.add_argument('directory', help='처리할 이미지 디렉토리')
    parser.add_argument('--format', choices=['csv', 'json', 'both'], 
                       default='csv', help='출력 형식')
    
    args = parser.parse_args()
    
    if not Path(args.directory).exists():
        print(f"디렉토리를 찾을 수 없습니다: {args.directory}")
        return
    
    process_directory(args.directory, args.format)


if __name__ == "__main__":
    main()