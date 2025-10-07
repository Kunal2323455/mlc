#!/usr/bin/env python3
"""
Simple script to run the improved tulsi disease detection training
"""

import os
import sys

def main():
    print("🌿 Starting Tulsi Disease Detection Model Training...")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists('dataset.zip'):
        print("❌ Error: dataset.zip not found!")
        print("Please ensure the dataset.zip file is in the current directory.")
        return 1
    
    # Run the improved training script
    try:
        print("🚀 Running improved tulsi detection training...")
        exec(open('improved_tulsi_detection.py').read())
        print("\n✅ Training completed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)