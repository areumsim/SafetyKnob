#!/usr/bin/env python3
"""
Test all images in test_all folder and generate comprehensive report
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from collections import defaultdict

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.safety_assessment_system import SafetyAssessmentSystem
from src.config.settings import SystemConfig


def test_all_folder():
    """Test all images in test_all folder"""
    
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
    print("="*60)
    print("SafetyKnob Safety Assessment - Test All Folder")
    print("="*60)
    print("\nInitializing safety assessment system...")
    system = SafetyAssessmentSystem(config)
    
    # Get all images from test_all folder
    test_all_dir = Path("data/test_all")
    if not test_all_dir.exists():
        print("Error: test_all directory not found")
        return
    
    image_files = list(test_all_dir.glob("*.jpg"))
    if not image_files:
        print("Error: No images found in test_all directory")
        return
    
    print(f"\nFound {len(image_files)} images in test_all directory")
    
    # Categorize files
    categories = defaultdict(list)
    for img_file in image_files:
        if img_file.name.startswith("danger_"):
            categories["danger"].append(img_file)
        elif img_file.name.startswith("safe_"):
            categories["safe"].append(img_file)
        elif img_file.name.startswith("caution_"):
            categories["caution"].append(img_file)
        else:
            categories["unknown"].append(img_file)
    
    print("\nImage distribution:")
    for category, files in categories.items():
        print(f"  {category}: {len(files)} images")
    
    # Process all images
    print("\n" + "="*60)
    print("Processing images...")
    print("="*60)
    
    results = []
    category_stats = defaultdict(lambda: {"total": 0, "safe": 0, "danger": 0})
    
    for i, img_path in enumerate(sorted(image_files), 1):
        # Determine actual category
        actual_category = "unknown"
        if img_path.name.startswith("danger_"):
            actual_category = "danger"
        elif img_path.name.startswith("safe_"):
            actual_category = "safe"
        elif img_path.name.startswith("caution_"):
            actual_category = "caution"
        
        print(f"\n[{i}/{len(image_files)}] Processing: {img_path.name}")
        print(f"  Actual category: {actual_category}")
        
        try:
            # Assess image
            result = system.assess_image(str(img_path))
            
            # Store result
            result_data = {
                'file_name': img_path.name,
                'actual_category': actual_category,
                'predicted_safe': result.is_safe,
                'predicted_category': 'safe' if result.is_safe else 'danger',
                'safety_score': result.overall_safety_score,
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'correct_prediction': (actual_category == 'safe' and result.is_safe) or 
                                    (actual_category == 'danger' and not result.is_safe),
                'risk_summary': result.get_risk_summary()
            }
            
            # Add dimension scores
            for dim_name, score in result.dimension_scores.items():
                result_data[f'dim_{dim_name}'] = score
            
            results.append(result_data)
            
            # Update statistics
            category_stats[actual_category]["total"] += 1
            if result.is_safe:
                category_stats[actual_category]["safe"] += 1
            else:
                category_stats[actual_category]["danger"] += 1
            
            # Print result
            print(f"  Predicted: {'SAFE' if result.is_safe else 'DANGER'}")
            print(f"  Safety Score: {result.overall_safety_score:.3f}")
            print(f"  Confidence: {result.confidence:.3f}")
            if not result.is_safe:
                print(f"  Risk: {result.get_risk_summary()}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'file_name': img_path.name,
                'actual_category': actual_category,
                'error': str(e)
            })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results as JSON
    json_file = f"test_all_results_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'total_images': len(image_files),
            'category_statistics': dict(category_stats),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    # Save as CSV
    csv_file = f"test_all_results_{timestamp}.csv"
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False, encoding='utf-8')
    
    # Calculate and display statistics
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    # Overall accuracy
    correct_predictions = sum(1 for r in results if 'correct_prediction' in r and r['correct_prediction'])
    total_valid = sum(1 for r in results if 'error' not in r and r['actual_category'] in ['safe', 'danger'])
    
    if total_valid > 0:
        accuracy = correct_predictions / total_valid * 100
        print(f"\nOverall Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_valid})")
    
    # Category-wise statistics
    print("\nCategory-wise Analysis:")
    print("-" * 50)
    print(f"{'Category':<15} {'Total':<8} {'Pred Safe':<10} {'Pred Danger':<12} {'Accuracy':<10}")
    print("-" * 50)
    
    for category in ['danger', 'safe', 'caution']:
        stats = category_stats[category]
        if stats['total'] > 0:
            if category == 'danger':
                correct = stats['danger']
            elif category == 'safe':
                correct = stats['safe']
            else:
                correct = '-'
            
            if category in ['danger', 'safe'] and isinstance(correct, int):
                acc = f"{correct/stats['total']*100:.1f}%"
            else:
                acc = "N/A"
            
            print(f"{category:<15} {stats['total']:<8} {stats['safe']:<10} {stats['danger']:<12} {acc:<10}")
    
    # Confusion matrix for safe/danger categories
    print("\nConfusion Matrix (for safe/danger only):")
    print("-" * 40)
    print("                Predicted")
    print("Actual         Safe    Danger")
    print("-" * 40)
    
    # Calculate confusion matrix
    tp = category_stats['safe']['safe']  # True Positive (actual safe, predicted safe)
    fn = category_stats['safe']['danger']  # False Negative (actual safe, predicted danger)
    fp = category_stats['danger']['safe']  # False Positive (actual danger, predicted safe)
    tn = category_stats['danger']['danger']  # True Negative (actual danger, predicted danger)
    
    print(f"Safe           {tp:<8}{fn:<8}")
    print(f"Danger         {fp:<8}{tn:<8}")
    
    # Performance metrics
    print("\nPerformance Metrics:")
    print("-" * 30)
    
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
        print(f"Precision: {precision:.3f}")
    
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
        print(f"Recall: {recall:.3f}")
    
    if 'precision' in locals() and 'recall' in locals() and (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"F1 Score: {f1:.3f}")
    
    # Average scores by category
    print("\nAverage Scores by Category:")
    print("-" * 40)
    
    for category in ['danger', 'safe', 'caution']:
        cat_results = [r for r in results if r.get('actual_category') == category and 'safety_score' in r]
        if cat_results:
            avg_score = sum(r['safety_score'] for r in cat_results) / len(cat_results)
            avg_conf = sum(r['confidence'] for r in cat_results) / len(cat_results)
            print(f"{category}: Score={avg_score:.3f}, Confidence={avg_conf:.3f}")
    
    print("\n" + "="*60)
    print(f"Results saved to:")
    print(f"  - {json_file}")
    print(f"  - {csv_file}")
    print("="*60)


if __name__ == "__main__":
    test_all_folder()