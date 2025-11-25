#!/usr/bin/env python3
"""
Batch testing script for SafetyKnob safety assessment
Tests multiple images and generates results report
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.safety_assessment_system import SafetyAssessmentSystem
from src.config.settings import SystemConfig


def test_batch_images(image_paths: List[str], output_file: str = None):
    """Test batch of images and save results"""
    
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
    
    # Process images
    results = []
    print(f"\nProcessing {len(image_paths)} images...")
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {Path(image_path).name}")
        
        try:
            # Assess image
            result = system.assess_image(image_path)
            
            # Store result
            results.append({
                'file_path': image_path,
                'file_name': Path(image_path).name,
                'is_safe': result.is_safe,
                'safety_score': result.overall_safety_score,
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'risk_summary': result.get_risk_summary(),
                **{f'dim_{k}': v for k, v in result.dimension_scores.items()}
            })
            
            # Print summary
            print(f"  Result: {'SAFE' if result.is_safe else 'DANGER'}")
            print(f"  Safety Score: {result.overall_safety_score:.3f}")
            print(f"  Confidence: {result.confidence:.3f}")
            if not result.is_safe:
                print(f"  Risk: {result.get_risk_summary()}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'file_path': image_path,
                'file_name': Path(image_path).name,
                'error': str(e)
            })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    json_file = output_file or f"batch_test_results_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'total_images': len(image_paths),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {json_file}")
    
    # Save as CSV
    csv_file = json_file.replace('.json', '.csv')
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"CSV saved to: {csv_file}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    successful = [r for r in results if 'error' not in r]
    if successful:
        safe_count = sum(1 for r in successful if r['is_safe'])
        danger_count = len(successful) - safe_count
        
        print(f"Total processed: {len(successful)}/{len(image_paths)}")
        print(f"Safe images: {safe_count} ({safe_count/len(successful)*100:.1f}%)")
        print(f"Danger images: {danger_count} ({danger_count/len(successful)*100:.1f}%)")
        print(f"Average safety score: {sum(r['safety_score'] for r in successful)/len(successful):.3f}")
        print(f"Average confidence: {sum(r['confidence'] for r in successful)/len(successful):.3f}")
        print(f"Average processing time: {sum(r['processing_time'] for r in successful)/len(successful):.3f}s")


def main():
    parser = argparse.ArgumentParser(description='Batch test images for safety assessment')
    parser.add_argument('--dir', help='Directory containing images')
    parser.add_argument('--files', nargs='+', help='List of image files')
    parser.add_argument('--pattern', default='*.jpg', help='File pattern for directory search')
    parser.add_argument('--output', help='Output file name')
    parser.add_argument('--limit', type=int, help='Limit number of images to process')
    
    args = parser.parse_args()
    
    # Collect image paths
    image_paths = []
    
    if args.dir:
        # Process directory
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"Error: Directory not found: {args.dir}")
            return
        
        for pattern in args.pattern.split(','):
            image_paths.extend(str(p) for p in dir_path.glob(pattern))
        
    elif args.files:
        # Process specific files
        image_paths = args.files
    else:
        # Default: test with sample images from danger and safe directories
        data_dir = Path("data")
        if (data_dir / "danger").exists() and (data_dir / "safe").exists():
            # Get 5 samples from each
            danger_images = list((data_dir / "danger").glob("*.jpg"))[:5]
            safe_images = list((data_dir / "safe").glob("*.jpg"))[:5]
            image_paths = [str(p) for p in danger_images + safe_images]
            print("Testing with sample images from danger and safe directories")
        else:
            print("Error: No input specified. Use --dir or --files")
            return
    
    # Apply limit if specified
    if args.limit and len(image_paths) > args.limit:
        image_paths = image_paths[:args.limit]
    
    if not image_paths:
        print("Error: No images found")
        return
    
    # Run batch test
    test_batch_images(image_paths, args.output)


if __name__ == "__main__":
    main()