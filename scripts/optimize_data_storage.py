#!/usr/bin/env python3
"""데이터 스토리지 최적화 - 심볼릭 링크로 중복 제거"""

import os
import shutil
from pathlib import Path

def create_symlinks_for_dataset(dataset_name, base_dataset='data_scenario'):
    """데이터셋의 train/val/test를 심볼릭 링크로 변경"""

    dataset_path = Path(dataset_name)
    base_path = Path(base_dataset)

    if not dataset_path.exists():
        print(f"⚠️  {dataset_name} 폴더가 존재하지 않습니다. 스킵.")
        return 0

    print(f"\n처리 중: {dataset_name}")
    print("=" * 60)

    # 백업할 파일들 (labels.json 등)
    backup_files = []
    for item in dataset_path.iterdir():
        if item.is_file():
            backup_files.append(item)

    # 특수 폴더 백업 (caution_analysis 등)
    special_dirs = []
    for item in dataset_path.iterdir():
        if item.is_dir() and item.name not in ['train', 'val', 'test']:
            special_dirs.append(item)

    # 백업 생성
    backup_dir = Path(f'.backup_{dataset_name}')
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(exist_ok=True)

    print(f"\n[1/5] 백업 생성 중...")
    for f in backup_files:
        shutil.copy2(f, backup_dir / f.name)
        print(f"  ✓ 백업: {f.name}")

    for d in special_dirs:
        if d.exists():
            shutil.copytree(d, backup_dir / d.name, dirs_exist_ok=True)
            print(f"  ✓ 백업: {d.name}/ (폴더)")

    # 기존 train/val/test 폴더 용량 계산 및 삭제
    print(f"\n[2/5] 기존 이미지 폴더 삭제 중...")
    saved_space = 0
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        if split_path.exists() and not split_path.is_symlink():
            # 용량 계산
            size = sum(f.stat().st_size for f in split_path.rglob('*') if f.is_file())
            saved_space += size

            shutil.rmtree(split_path)
            print(f"  ✓ 삭제: {split}/ ({size / 1e9:.2f} GB)")
        elif split_path.exists() and split_path.is_symlink():
            print(f"  ⚠️  {split}/ 이미 심볼릭 링크입니다. 스킵.")

    # 심볼릭 링크 생성
    print(f"\n[3/5] 심볼릭 링크 생성 중...")
    for split in ['train', 'val', 'test']:
        source = base_path / split
        target = dataset_path / split

        if target.exists():
            continue

        # 상대 경로로 링크 생성
        rel_source = os.path.relpath(source, dataset_path)
        os.symlink(rel_source, target)
        print(f"  ✓ 링크: {split}/ → {rel_source}")

    # 백업 복원
    print(f"\n[4/5] labels.json 및 특수 폴더 복원 중...")
    for f in backup_files:
        shutil.copy2(backup_dir / f.name, dataset_path / f.name)
        print(f"  ✓ 복원: {f.name}")

    for d in special_dirs:
        target_dir = dataset_path / d.name
        if target_dir.exists() and not target_dir.is_symlink():
            shutil.rmtree(target_dir)
        if backup_dir.joinpath(d.name).exists():
            shutil.copytree(backup_dir / d.name, target_dir, dirs_exist_ok=True)
            print(f"  ✓ 복원: {d.name}/")

    # 백업 삭제
    print(f"\n[5/5] 임시 백업 삭제 중...")
    shutil.rmtree(backup_dir)
    print(f"  ✓ 백업 폴더 삭제: .backup_{dataset_name}/")

    print(f"\n절감 용량: {saved_space / 1e9:.2f} GB")
    print("✅ 완료!")

    return saved_space

def main():
    print("=" * 60)
    print("데이터 스토리지 최적화 시작")
    print("=" * 60)
    print("\n⚠️  주의: 이 작업은 실제 이미지 파일을 삭제합니다!")
    print("labels.json은 보존되며 모든 실험은 기존과 동일하게 작동합니다.\n")

    total_saved = 0

    # data_caution_excluded 최적화
    saved = create_symlinks_for_dataset('data_caution_excluded')
    total_saved += saved

    # data_temporal 최적화
    saved = create_symlinks_for_dataset('data_temporal')
    total_saved += saved

    # data 폴더 삭제 (사용 안함)
    if Path('data').exists():
        print("\n처리 중: data/")
        print("=" * 60)
        print("\n[1/2] data/ 폴더 용량 계산 중...")
        size = sum(f.stat().st_size for f in Path('data').rglob('*') if f.is_file())
        print(f"  용량: {size / 1e9:.2f} GB")

        print("\n[2/2] data/ 폴더 삭제 중...")
        shutil.rmtree('data')
        total_saved += size
        print(f"  ✓ 삭제 완료: data/ ({size / 1e9:.2f} GB)")
        print("✅ 완료!")

    print("\n" + "=" * 60)
    print(f"총 절감 용량: {total_saved / 1e9:.2f} GB")
    print("=" * 60)
    print("\n✅ 데이터 최적화 완료!")
    print("\n다음 명령으로 링크 확인:")
    print("  ls -l data_caution_excluded/")
    print("  ls -l data_temporal/")
    print("\n모든 실험은 기존과 동일하게 실행 가능합니다.")

if __name__ == '__main__':
    # 안전 확인
    print("\n" + "=" * 60)
    print("데이터 최적화 확인")
    print("=" * 60)
    print("\n작업 내용:")
    print("  1. data_caution_excluded/의 train/val/test → 심볼릭 링크로 변경")
    print("  2. data_temporal/의 train/val/test → 심볼릭 링크로 변경")
    print("  3. data/ 폴더 완전 삭제")
    print("\n보존 항목:")
    print("  - data_scenario/ (메인, 실제 이미지)")
    print("  - 모든 labels.json 파일")
    print("  - data_caution_excluded/caution_analysis/ (599개 이미지)")
    print("\n예상 절감: 약 53GB")

    response = input("\n데이터 최적화를 시작하시겠습니까? (yes/no): ")
    if response.lower() == 'yes':
        main()
    else:
        print("\n취소되었습니다.")
