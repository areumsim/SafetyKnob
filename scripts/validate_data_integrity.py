#!/usr/bin/env python3
"""데이터 무결성 검증 - 심볼릭 링크 변환 전 체크"""

import json
from pathlib import Path
from collections import Counter

def check_dataset_integrity():
    """각 데이터셋의 train/val/test 중복 확인"""

    datasets = ['data_scenario', 'data_caution_excluded', 'data_temporal']
    results = {}

    for ds in datasets:
        labels_file = Path(ds) / 'labels.json'
        if not labels_file.exists():
            print(f"⚠️  {ds}/labels.json 파일 없음 - 스킵")
            continue

        with open(labels_file) as f:
            labels = json.load(f)

        # train/val/test 파일 분리
        splits = {'train': [], 'val': [], 'test': [], 'other': []}
        for img_path in labels.keys():
            if img_path.startswith('train/'):
                splits['train'].append(img_path)
            elif img_path.startswith('val/'):
                splits['val'].append(img_path)
            elif img_path.startswith('test/'):
                splits['test'].append(img_path)
            elif img_path.startswith('caution_analysis/'):
                splits['other'].append(img_path)

        # 중복 체크
        all_files = splits['train'] + splits['val'] + splits['test']
        duplicates = [f for f, cnt in Counter(all_files).items() if cnt > 1]

        results[ds] = {
            'train_count': len(splits['train']),
            'val_count': len(splits['val']),
            'test_count': len(splits['test']),
            'other_count': len(splits['other']),
            'duplicates': len(duplicates),
            'status': '✅ Clean' if len(duplicates) == 0 else '❌ Error'
        }

    # 결과 출력
    print("=" * 60)
    print("데이터셋 무결성 검증 결과")
    print("=" * 60)
    for ds, info in results.items():
        print(f"\n{ds}:")
        print(f"  Train: {info['train_count']:,}개")
        print(f"  Val:   {info['val_count']:,}개")
        print(f"  Test:  {info['test_count']:,}개")
        if info['other_count'] > 0:
            print(f"  Other: {info['other_count']:,}개 (caution 등)")
        print(f"  중복:  {info['duplicates']}개")
        print(f"  상태:  {info['status']}")

    # 전체 통과 여부
    all_clean = all(r['duplicates'] == 0 for r in results.values())
    print("\n" + "=" * 60)
    if all_clean:
        print("✅ 모든 데이터셋 무결성 확인 완료!")
        print("심볼릭 링크 변환을 안전하게 진행할 수 있습니다.")
    else:
        print("❌ 중복 파일 발견! 데이터 정리 필요")
    print("=" * 60)

    return all_clean

if __name__ == '__main__':
    check_dataset_integrity()
