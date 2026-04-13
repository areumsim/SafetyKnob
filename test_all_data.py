#!/usr/bin/env python3
"""
Test all data directories in the SafetyKnob project
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.safety_assessment_system import SafetyAssessmentSystem
from src.config.settings import SystemConfig


def test_all_data():
    """Test all image directories and generate comprehensive report"""
    
    # Load configuration
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = SystemConfig.from_dict(config_dict)
    else:
        print("Error: config.json not found")
        return
    
    # Initialize system
    print("Initializing safety assessment system...")
    system = SafetyAssessmentSystem(config)
    
    # Find all directories with images
    data_dir = Path("data")
    test_dirs = []
    
    # Common image directories
    for dir_name in ["danger", "safe", "caution"]:
        dir_path = data_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            test_dirs.append(dir_path)
    
    # SO-XX directories
    for dir_path in data_dir.glob("SO-*"):
        if dir_path.is_dir():
            test_dirs.append(dir_path)
    
    print(f"\nFound {len(test_dirs)} directories to test")
    
    # Test each directory
    all_results = []
    summary_by_dir = {}
    
    for dir_path in sorted(test_dirs):
        print(f"\n{'='*60}")
        print(f"Testing directory: {dir_path.name}")
        print(f"{'='*60}")
        
        # Get all image files
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_files.extend(dir_path.glob(ext))
        
        if not image_files:
            print(f"No images found in {dir_path}")
            continue
        
        # Limit to reasonable number for testing
        max_images = 50
        if len(image_files) > max_images:
            print(f"Found {len(image_files)} images, testing first {max_images}")
            image_files = image_files[:max_images]
        else:
            print(f"Found {len(image_files)} images")
        
        # Process images
        dir_results = []
        safe_count = 0
        danger_count = 0
        
        for i, img_path in enumerate(image_files, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(image_files)}")
            
            try:
                result = system.assess_image(str(img_path))
                
                dir_results.append({
                    'directory': dir_path.name,
                    'file_path': str(img_path),
                    'file_name': img_path.name,
                    'is_safe': result.is_safe,
                    'safety_score': result.overall_safety_score,
                    'confidence': result.confidence,
                    'processing_time': result.processing_time,
                    **{f'dim_{k}': v for k, v in result.dimension_scores.items()}
                })
                
                if result.is_safe:
                    safe_count += 1
                else:
                    danger_count += 1
                    
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
                dir_results.append({
                    'directory': dir_path.name,
                    'file_path': str(img_path),
                    'file_name': img_path.name,
                    'error': str(e)
                })
        
        # Directory summary
        summary_by_dir[dir_path.name] = {
            'total_images': len(image_files),
            'processed': len([r for r in dir_results if 'error' not in r]),
            'safe_count': safe_count,
            'danger_count': danger_count,
            'safe_percentage': (safe_count / len(dir_results) * 100) if dir_results else 0,
            'avg_safety_score': sum(r.get('safety_score', 0) for r in dir_results if 'error' not in r) / len(dir_results) if dir_results else 0,
            'avg_confidence': sum(r.get('confidence', 0) for r in dir_results if 'error' not in r) / len(dir_results) if dir_results else 0
        }
        
        print(f"\nDirectory Summary:")
        print(f"  Total processed: {summary_by_dir[dir_path.name]['processed']}")
        print(f"  Safe: {safe_count} ({summary_by_dir[dir_path.name]['safe_percentage']:.1f}%)")
        print(f"  Danger: {danger_count} ({100 - summary_by_dir[dir_path.name]['safe_percentage']:.1f}%)")
        print(f"  Avg Safety Score: {summary_by_dir[dir_path.name]['avg_safety_score']:.3f}")
        print(f"  Avg Confidence: {summary_by_dir[dir_path.name]['avg_confidence']:.3f}")
        
        all_results.extend(dir_results)
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = f"all_data_test_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'summary_by_directory': summary_by_dir,
            'detailed_results': all_results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n\nDetailed results saved to: {results_file}")
    
    # Save CSV
    csv_file = results_file.replace('.json', '.csv')
    df = pd.DataFrame(all_results)
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"CSV saved to: {csv_file}")
    
    # Save summary
    summary_file = f"all_data_test_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_by_dir, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_file}")
    
    # Print overall summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    total_processed = sum(s['processed'] for s in summary_by_dir.values())
    total_safe = sum(s['safe_count'] for s in summary_by_dir.values())
    total_danger = sum(s['danger_count'] for s in summary_by_dir.values())
    
    print(f"Total images processed: {total_processed}")
    print(f"Total safe: {total_safe} ({total_safe/total_processed*100:.1f}%)")
    print(f"Total danger: {total_danger} ({total_danger/total_processed*100:.1f}%)")
    
    print("\nBy Directory:")
    for dir_name, summary in sorted(summary_by_dir.items()):
        print(f"\n{dir_name}:")
        print(f"  Processed: {summary['processed']}")
        print(f"  Safe: {summary['safe_count']} ({summary['safe_percentage']:.1f}%)")
        print(f"  Danger: {summary['danger_count']} ({100-summary['safe_percentage']:.1f}%)")
        print(f"  Avg Score: {summary['avg_safety_score']:.3f}")


if __name__ == "__main__":
    test_all_data()