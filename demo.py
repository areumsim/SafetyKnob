#!/usr/bin/env python3
"""
SafetyKnob Demo - Test safety assessment on sample images
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_batch import test_batch_images


def main():
    print("="*60)
    print("SafetyKnob Safety Assessment Demo")
    print("="*60)
    
    # Get sample images
    data_dir = Path("data")
    
    # Get 3 danger and 3 safe images
    danger_images = list((data_dir / "danger").glob("*.jpg"))[:3]
    safe_images = list((data_dir / "safe").glob("*.jpg"))[:3]
    
    if not danger_images or not safe_images:
        print("Error: Could not find sample images")
        return
    
    all_images = [str(p) for p in danger_images + safe_images]
    
    print(f"\nTesting with {len(danger_images)} danger images and {len(safe_images)} safe images")
    print("\nDanger images:")
    for img in danger_images:
        print(f"  - {img.name}")
    print("\nSafe images:")
    for img in safe_images:
        print(f"  - {img.name}")
    
    # Run test
    print("\n" + "="*60)
    test_batch_images(all_images, "demo_results.json")
    
    print("\n" + "="*60)
    print("Demo completed!")
    print(f"Results saved to: demo_results.json and demo_results.csv")


if __name__ == "__main__":
    main()