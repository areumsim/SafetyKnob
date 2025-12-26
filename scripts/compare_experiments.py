#!/usr/bin/env python3
"""
실험 결과 비교 시각화

Scenario (Safe+Danger+Caution) vs Caution Excluded (Safe+Danger) 실험 비교
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_results(results_dir, models):
    """결과 로드"""
    results = {}
    for model in models:
        result_file = Path(results_dir) / model / 'results.json'
        if result_file.exists():
            with open(result_file, 'r') as f:
                results[model] = json.load(f)
    return results

def create_comparison_visualizations(scenario_results, caution_excluded_results, output_dir):
    """비교 시각화 생성"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = ['siglip', 'clip', 'dinov2']
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc_roc']

    # 1. Overall Performance Comparison (with proper Y-axis)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        scenario_values = [scenario_results[m]['test_metrics'][metric] for m in models]
        caution_values = [caution_excluded_results[m]['test_metrics'][metric] for m in models]

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax.bar(x - width/2, scenario_values, width,
                       label='Scenario\n(Safe+Danger+Caution)', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, caution_values, width,
                       label='Caution Excluded\n(Safe+Danger)', color='#e74c3c', alpha=0.8)

        # 값 표시
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Model', fontsize=10, fontweight='bold')
        ax.set_ylabel(metric.upper().replace('_', '-'), fontsize=10, fontweight='bold')
        ax.set_title(f'{metric.upper().replace("_", "-")} Comparison',
                    fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models])

        # IMPORTANT: Set Y-axis from 0 to 1
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    # 마지막 subplot 사용 안함
    axes[5].axis('off')

    plt.suptitle('Experiment Comparison: Scenario vs Caution Excluded',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'experiment_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Model Performance Overview (Heatmap)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scenario heatmap
    scenario_matrix = np.array([
        [scenario_results[m]['test_metrics'][metric] for metric in metrics]
        for m in models
    ])

    sns.heatmap(scenario_matrix, annot=True, fmt='.4f', cmap='YlGnBu',
               xticklabels=[m.upper().replace('_', '-') for m in metrics],
               yticklabels=[m.upper() for m in models],
               vmin=0, vmax=1, cbar_kws={'label': 'Score'},
               ax=ax1)
    ax1.set_title('Scenario (Safe+Danger+Caution)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('Model', fontsize=10, fontweight='bold')

    # Caution Excluded heatmap
    caution_matrix = np.array([
        [caution_excluded_results[m]['test_metrics'][metric] for metric in metrics]
        for m in models
    ])

    sns.heatmap(caution_matrix, annot=True, fmt='.4f', cmap='YlGnBu',
               xticklabels=[m.upper().replace('_', '-') for m in metrics],
               yticklabels=[m.upper() for m in models],
               vmin=0, vmax=1, cbar_kws={'label': 'Score'},
               ax=ax2)
    ax2.set_title('Caution Excluded (Safe+Danger)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('')

    plt.suptitle('Model Performance Heatmap Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. F1 Score Detailed Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    scenario_f1 = [scenario_results[m]['test_metrics']['f1'] for m in models]
    caution_f1 = [caution_excluded_results[m]['test_metrics']['f1'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, scenario_f1, width,
                   label='Scenario (Safe+Danger+Caution)', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, caution_f1, width,
                   label='Caution Excluded (Safe+Danger)', color='#e74c3c', alpha=0.8)

    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score Comparison: Scenario vs Caution Excluded',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models], fontsize=11)
    ax.set_ylim(0, 1.0)  # Y축 0-1
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Training Time Comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    scenario_times = [scenario_results[m]['training_time_seconds'] / 3600 for m in models]
    caution_times = [caution_excluded_results[m]['training_time_seconds'] / 3600 for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, scenario_times, width,
                   label='Scenario', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, caution_times, width,
                   label='Caution Excluded', color='#e74c3c', alpha=0.8)

    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}h',
                   ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Time (hours)', fontsize=12, fontweight='bold')
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 비교 시각화 저장 완료: {output_dir}")

def print_comparison_summary(scenario_results, caution_excluded_results):
    """비교 요약 출력"""
    print("\n" + "="*70)
    print("실험 결과 비교 요약")
    print("="*70)

    models = ['siglip', 'clip', 'dinov2']
    metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc_roc']

    print("\n" + "-"*70)
    print("Scenario (Safe+Danger+Caution) 결과")
    print("-"*70)

    for model in models:
        print(f"\n{model.upper()}:")
        for metric in metrics:
            value = scenario_results[model]['test_metrics'][metric]
            print(f"  {metric.upper():10s}: {value:.4f}")
        print(f"  Training time: {scenario_results[model]['training_time_seconds']/3600:.2f}h")

    print("\n" + "-"*70)
    print("Caution Excluded (Safe+Danger) 결과")
    print("-"*70)

    for model in models:
        print(f"\n{model.upper()}:")
        for metric in metrics:
            value = caution_excluded_results[model]['test_metrics'][metric]
            print(f"  {metric.upper():10s}: {value:.4f}")
        print(f"  Training time: {caution_excluded_results[model]['training_time_seconds']/3600:.2f}h")

    print("\n" + "-"*70)
    print("성능 차이 분석 (Scenario - Caution Excluded)")
    print("-"*70)

    for model in models:
        print(f"\n{model.upper()}:")
        for metric in metrics:
            scenario_val = scenario_results[model]['test_metrics'][metric]
            caution_val = caution_excluded_results[model]['test_metrics'][metric]
            diff = scenario_val - caution_val
            print(f"  {metric.upper():10s}: {diff:+.4f} ({'+' if diff >= 0 else ''}{diff*100:.2f}%)")

    print("\n" + "="*70)
    print("주요 발견 사항")
    print("="*70)

    # F1 Score 비교
    print("\n1. F1 Score 비교:")
    for model in models:
        scenario_f1 = scenario_results[model]['test_metrics']['f1']
        caution_f1 = caution_excluded_results[model]['test_metrics']['f1']
        diff = scenario_f1 - caution_f1

        comparison = "향상" if diff > 0 else "감소"
        print(f"   {model.upper():8s}: Scenario {scenario_f1:.4f} vs Caution Excluded {caution_f1:.4f} "
              f"(차이: {diff:+.4f}, {comparison})")

    # 최고 성능 모델
    print("\n2. 최고 F1 Score:")
    scenario_best = max(models, key=lambda m: scenario_results[m]['test_metrics']['f1'])
    scenario_best_f1 = scenario_results[scenario_best]['test_metrics']['f1']

    caution_best = max(models, key=lambda m: caution_excluded_results[m]['test_metrics']['f1'])
    caution_best_f1 = caution_excluded_results[caution_best]['test_metrics']['f1']

    print(f"   Scenario: {scenario_best.upper()} ({scenario_best_f1:.4f})")
    print(f"   Caution Excluded: {caution_best.upper()} ({caution_best_f1:.4f})")

    # 학습 시간 비교
    print("\n3. 학습 시간:")
    for model in models:
        scenario_time = scenario_results[model]['training_time_seconds'] / 3600
        caution_time = caution_excluded_results[model]['training_time_seconds'] / 3600
        diff_time = scenario_time - caution_time

        print(f"   {model.upper():8s}: Scenario {scenario_time:.2f}h vs "
              f"Caution Excluded {caution_time:.2f}h (차이: {diff_time:+.2f}h)")

    print("\n" + "="*70)

def main():
    # 경로 설정
    scenario_dir = Path('results/scenario')
    caution_excluded_dir = Path('results/caution_excluded')
    output_dir = Path('results/comparison')

    models = ['siglip', 'clip', 'dinov2']

    print("실험 결과 비교 시작...")

    # 결과 로드
    print(f"\nScenario 결과 로드: {scenario_dir}")
    scenario_results = load_results(scenario_dir, models)

    print(f"Caution Excluded 결과 로드: {caution_excluded_dir}")
    caution_excluded_results = load_results(caution_excluded_dir, models)

    # 결과 확인
    if len(scenario_results) != len(models):
        print(f"\n⚠️  Warning: Scenario 결과가 불완전합니다 (발견: {len(scenario_results)}/{len(models)})")

    if len(caution_excluded_results) != len(models):
        print(f"\n⚠️  Warning: Caution Excluded 결과가 불완전합니다 (발견: {len(caution_excluded_results)}/{len(models)})")

    # 비교 요약 출력
    print_comparison_summary(scenario_results, caution_excluded_results)

    # 시각화 생성
    print(f"\n시각화 생성 중...")
    create_comparison_visualizations(scenario_results, caution_excluded_results, output_dir)

    print("\n" + "="*70)
    print("비교 완료!")
    print("="*70)
    print(f"\n생성된 파일:")
    print(f"  - {output_dir}/experiment_comparison.png")
    print(f"  - {output_dir}/performance_heatmap.png")
    print(f"  - {output_dir}/f1_comparison.png")
    print(f"  - {output_dir}/training_time_comparison.png")

if __name__ == '__main__':
    main()
