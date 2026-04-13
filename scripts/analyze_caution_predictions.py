#!/usr/bin/env python3
"""
Caution Predictions 상세 분석 및 시각화

Caution 이미지에 대한 3개 모델의 예측 결과를 상세 분석하고 시각화합니다.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_predictions(json_path):
    """예측 결과 로드"""
    with open(json_path, 'r') as f:
        return json.load(f)

def analyze_predictions(data):
    """예측 결과 상세 분석"""
    models = data['models']
    predictions = data['predictions']
    num_images = data['num_images']

    analysis = {
        'models': models,
        'num_images': num_images,
        'per_model': {},
        'agreement': {},
        'confidence_analysis': {}
    }

    # 모델별 분석
    for model in models:
        model_preds = predictions[model]

        safe_preds = [p for p in model_preds if p['prediction'] == 'safe']
        unsafe_preds = [p for p in model_preds if p['prediction'] == 'unsafe']

        analysis['per_model'][model] = {
            'safe_count': len(safe_preds),
            'safe_pct': len(safe_preds) / num_images * 100,
            'unsafe_count': len(unsafe_preds),
            'unsafe_pct': len(unsafe_preds) / num_images * 100,
            'avg_confidence': np.mean([p['confidence'] for p in model_preds]),
            'safe_confidences': [p['confidence'] for p in safe_preds],
            'unsafe_confidences': [p['confidence'] for p in unsafe_preds],
            'avg_safe_conf': np.mean([p['confidence'] for p in safe_preds]) if safe_preds else 0,
            'avg_unsafe_conf': np.mean([p['confidence'] for p in unsafe_preds]) if unsafe_preds else 0
        }

    # 모델 간 일치도 분석
    agreement_patterns = []
    for i in range(num_images):
        preds = [predictions[m][i]['prediction'] for m in models]
        safe_count = preds.count('safe')

        if safe_count == 3:
            agreement_patterns.append('all_safe')
        elif safe_count == 0:
            agreement_patterns.append('all_unsafe')
        elif safe_count == 2:
            agreement_patterns.append('majority_safe')
        else:  # safe_count == 1
            agreement_patterns.append('majority_unsafe')

    counter = Counter(agreement_patterns)
    analysis['agreement'] = {
        'all_safe': counter['all_safe'],
        'all_unsafe': counter['all_unsafe'],
        'majority_safe': counter['majority_safe'],
        'majority_unsafe': counter['majority_unsafe'],
        'all_safe_pct': counter['all_safe'] / num_images * 100,
        'all_unsafe_pct': counter['all_unsafe'] / num_images * 100,
        'majority_safe_pct': counter['majority_safe'] / num_images * 100,
        'majority_unsafe_pct': counter['majority_unsafe'] / num_images * 100
    }

    # Confidence 범위별 분포
    for model in models:
        model_preds = predictions[model]
        confidences = [p['confidence'] for p in model_preds]

        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(confidences, bins=bins)

        analysis['confidence_analysis'][model] = {
            'bins': bins,
            'counts': hist.tolist(),
            'min': min(confidences),
            'max': max(confidences),
            'median': np.median(confidences),
            'std': np.std(confidences)
        }

    return analysis

def create_visualizations(analysis, output_dir):
    """시각화 생성"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = analysis['models']

    # 1. 예측 분포 비교 (Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    safe_counts = [analysis['per_model'][m]['safe_pct'] for m in models]
    unsafe_counts = [analysis['per_model'][m]['unsafe_pct'] for m in models]

    ax.bar(x - width/2, safe_counts, width, label='Safe', color='#2ecc71')
    ax.bar(x + width/2, unsafe_counts, width, label='Unsafe', color='#e74c3c')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Caution Images: Prediction Distribution by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'caution_prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Confidence 분포 (Histograms)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, model in enumerate(models):
        ax = axes[idx]
        per_model = analysis['per_model'][model]

        # Safe와 Unsafe 별로 histogram
        if per_model['safe_confidences']:
            ax.hist(per_model['safe_confidences'], bins=20, alpha=0.6,
                   color='#2ecc71', label='Safe', range=(0.5, 1.0))
        if per_model['unsafe_confidences']:
            ax.hist(per_model['unsafe_confidences'], bins=20, alpha=0.6,
                   color='#e74c3c', label='Unsafe', range=(0.5, 1.0))

        ax.set_xlabel('Confidence Score', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{model.upper()}\nAvg: {per_model["avg_confidence"]:.3f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlim(0.5, 1.0)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Confidence Score Distribution by Model', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'caution_confidence_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 모델 일치도 분석 (Stacked Bar)
    fig, ax = plt.subplots(figsize=(10, 6))

    agreement = analysis['agreement']
    categories = ['All 3 Agree\nSafe', 'All 3 Agree\nUnsafe',
                 '2 Models\nAgree Safe', '2 Models\nAgree Unsafe']
    values = [
        agreement['all_safe'],
        agreement['all_unsafe'],
        agreement['majority_safe'],
        agreement['majority_unsafe']
    ]
    percentages = [
        agreement['all_safe_pct'],
        agreement['all_unsafe_pct'],
        agreement['majority_safe_pct'],
        agreement['majority_unsafe_pct']
    ]
    colors = ['#27ae60', '#c0392b', '#52be80', '#e67e73']

    bars = ax.bar(categories, values, color=colors, alpha=0.8)

    # 값 표시
    for bar, val, pct in zip(bars, values, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Model Agreement on Caution Images', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'caution_model_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 모델별 Safe/Unsafe Confidence 비교
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    safe_confs = [analysis['per_model'][m]['avg_safe_conf'] for m in models]
    unsafe_confs = [analysis['per_model'][m]['avg_unsafe_conf'] for m in models]

    bars1 = ax.bar(x - width/2, safe_confs, width, label='Safe Predictions', color='#2ecc71')
    bars2 = ax.bar(x + width/2, unsafe_confs, width, label='Unsafe Predictions', color='#e74c3c')

    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Confidence', fontsize=12, fontweight='bold')
    ax.set_title('Average Confidence by Prediction Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'caution_confidence_by_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 시각화 저장 완료: {output_dir}")

def print_analysis_summary(analysis):
    """분석 결과 요약 출력"""
    print("\n" + "="*70)
    print("CAUTION 예측 결과 상세 분석")
    print("="*70)

    print(f"\n총 이미지 수: {analysis['num_images']}")
    print(f"분석 모델: {', '.join([m.upper() for m in analysis['models']])}")

    print("\n" + "-"*70)
    print("1. 모델별 예측 분포")
    print("-"*70)

    for model in analysis['models']:
        pm = analysis['per_model'][model]
        print(f"\n{model.upper()}:")
        print(f"  Safe:   {pm['safe_count']:4d} ({pm['safe_pct']:5.1f}%)")
        print(f"  Unsafe: {pm['unsafe_count']:4d} ({pm['unsafe_pct']:5.1f}%)")
        print(f"  평균 Confidence: {pm['avg_confidence']:.4f}")
        print(f"    - Safe 예측 평균 Confidence:   {pm['avg_safe_conf']:.4f}")
        print(f"    - Unsafe 예측 평균 Confidence: {pm['avg_unsafe_conf']:.4f}")

    print("\n" + "-"*70)
    print("2. 모델 간 일치도 분석")
    print("-"*70)

    agr = analysis['agreement']
    print(f"\n3개 모델 모두 Safe 예측:   {agr['all_safe']:4d} ({agr['all_safe_pct']:5.1f}%)")
    print(f"3개 모델 모두 Unsafe 예측: {agr['all_unsafe']:4d} ({agr['all_unsafe_pct']:5.1f}%)")
    print(f"2개 모델 Safe 예측:        {agr['majority_safe']:4d} ({agr['majority_safe_pct']:5.1f}%)")
    print(f"2개 모델 Unsafe 예측:      {agr['majority_unsafe']:4d} ({agr['majority_unsafe_pct']:5.1f}%)")

    print("\n" + "-"*70)
    print("3. Confidence 통계")
    print("-"*70)

    for model in analysis['models']:
        conf = analysis['confidence_analysis'][model]
        print(f"\n{model.upper()}:")
        print(f"  최소: {conf['min']:.4f}")
        print(f"  최대: {conf['max']:.4f}")
        print(f"  중앙값: {conf['median']:.4f}")
        print(f"  표준편차: {conf['std']:.4f}")

    print("\n" + "="*70)
    print("주요 발견 사항")
    print("="*70)

    # 자동 인사이트 생성
    models = analysis['models']

    # 가장 보수적인 모델 (가장 많이 unsafe 예측)
    unsafe_pcts = {m: analysis['per_model'][m]['unsafe_pct'] for m in models}
    most_conservative = max(unsafe_pcts, key=unsafe_pcts.get)
    least_conservative = min(unsafe_pcts, key=unsafe_pcts.get)

    print(f"\n1. 가장 보수적 (Unsafe 많이 예측): {most_conservative.upper()} ({unsafe_pcts[most_conservative]:.1f}%)")
    print(f"   가장 관대함 (Safe 많이 예측): {least_conservative.upper()} ({100-unsafe_pcts[least_conservative]:.1f}% safe)")

    # 일치도 분석
    total_agree = agr['all_safe'] + agr['all_unsafe']
    agree_pct = total_agree / analysis['num_images'] * 100
    print(f"\n2. 전체 일치율: {agree_pct:.1f}% ({total_agree}/{analysis['num_images']})")
    print(f"   - 모두 Safe 일치: {agr['all_safe_pct']:.1f}%")
    print(f"   - 모두 Unsafe 일치: {agr['all_unsafe_pct']:.1f}%")

    # Confidence 분석
    avg_confs = {m: analysis['per_model'][m]['avg_confidence'] for m in models}
    highest_conf_model = max(avg_confs, key=avg_confs.get)
    print(f"\n3. 가장 높은 평균 Confidence: {highest_conf_model.upper()} ({avg_confs[highest_conf_model]:.4f})")

    # 안전성 편향
    unsafe_bias = sum(unsafe_pcts.values()) / len(models)
    print(f"\n4. Unsafe 편향: 평균 {unsafe_bias:.1f}%가 Unsafe로 예측됨")
    print(f"   → Caution은 모호한 경우임에도 모델들이 Unsafe로 예측하는 경향")

    print("\n" + "="*70)

def main():
    # 경로 설정
    predictions_file = Path('results/caution_excluded/caution_predictions.json')
    output_dir = Path('results/caution_excluded/analysis')

    print("Caution 예측 결과 분석 시작...")

    # 데이터 로드
    print(f"\n예측 결과 로드: {predictions_file}")
    data = load_predictions(predictions_file)

    # 분석 수행
    print("\n분석 수행 중...")
    analysis = analyze_predictions(data)

    # 결과 출력
    print_analysis_summary(analysis)

    # 시각화 생성
    print(f"\n시각화 생성 중...")
    create_visualizations(analysis, output_dir)

    # 분석 결과 JSON 저장
    analysis_file = output_dir / 'caution_analysis_summary.json'
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n✅ 분석 결과 저장: {analysis_file}")

    print("\n" + "="*70)
    print("분석 완료!")
    print("="*70)
    print(f"\n생성된 파일:")
    print(f"  - {output_dir}/caution_prediction_distribution.png")
    print(f"  - {output_dir}/caution_confidence_histograms.png")
    print(f"  - {output_dir}/caution_model_agreement.png")
    print(f"  - {output_dir}/caution_confidence_by_prediction.png")
    print(f"  - {analysis_file}")

if __name__ == '__main__':
    main()
