#!/usr/bin/env python3
"""
Quick test script to verify the Tulsi disease detection system
"""

import os
import sys
import json
from pathlib import Path

def check_dataset():
    """Check if dataset exists and is structured correctly"""
    print("🔍 Checking dataset...")
    
    dataset_path = Path("dataset")
    if not dataset_path.exists():
        print("❌ Dataset folder not found!")
        return False
    
    expected_classes = ['bacterial', 'fungal', 'healthy', 'pests']
    found_classes = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    found_classes.sort()
    
    print(f"   Found classes: {found_classes}")
    
    for cls in expected_classes:
        cls_path = dataset_path / cls
        if cls_path.exists():
            count = len(list(cls_path.glob('*.jpg')) + list(cls_path.glob('*.png')) + 
                       list(cls_path.glob('*.jpeg')))
            print(f"   ✅ {cls}: {count} images")
        else:
            print(f"   ❌ {cls}: not found")
            return False
    
    return True

def check_dependencies():
    """Check if all dependencies are installed"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'sklearn', 'cv2', 'PIL', 'tensorflow'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'cv2':
                __import__('cv2')
            elif package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    return len(missing) == 0

def check_model_files():
    """Check if model files exist"""
    print("\n🔍 Checking model files...")
    
    files_to_check = {
        'tulsi_disease_detection_best_model.h5': 'Best model file',
        'model_config.json': 'Model configuration'
    }
    
    all_exist = True
    for file, description in files_to_check.items():
        if Path(file).exists():
            print(f"   ✅ {file} ({description})")
        else:
            print(f"   ⚠️  {file} - Not found (will be created during training)")
            all_exist = False
    
    return all_exist

def test_prediction():
    """Test prediction on a sample image"""
    print("\n🔍 Testing prediction system...")
    
    # Check if model exists
    if not Path('tulsi_disease_detection_best_model.h5').exists():
        print("   ⚠️  Model not trained yet. Run train_model.py first.")
        return False
    
    try:
        from detector import TulsiDiseaseDetector
        
        detector = TulsiDiseaseDetector(
            'tulsi_disease_detection_best_model.h5',
            'model_config.json'
        )
        
        # Find a test image
        test_image = None
        for cls in ['bacterial', 'fungal', 'healthy', 'pests']:
            cls_path = Path('dataset') / cls
            if cls_path.exists():
                images = list(cls_path.glob('*.jpg')) + list(cls_path.glob('*.png'))
                if images:
                    test_image = str(images[0])
                    break
        
        if test_image:
            print(f"   Testing with: {test_image}")
            result = detector.predict(test_image)
            print(f"   ✅ Prediction: {result['predicted_disease']}")
            print(f"   ✅ Confidence: {result['confidence_percentage']}")
            return True
        else:
            print("   ⚠️  No test images found")
            return False
            
    except Exception as e:
        print(f"   ❌ Error during prediction: {e}")
        return False

def main():
    """Run all checks"""
    print("="*70)
    print("🌿 TULSI DISEASE DETECTION - SYSTEM CHECK")
    print("="*70)
    
    checks = [
        ("Dataset Structure", check_dataset()),
        ("Dependencies", check_dependencies()),
        ("Model Files", check_model_files()),
    ]
    
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:25s}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    
    if all_passed:
        print("🎉 All checks passed! System is ready.")
        print("\n📝 Next steps:")
        print("   1. Run: python train_model.py  (to train the models)")
        print("   2. Run: python test_system.py  (to test predictions)")
        print("   3. Run: python api.py          (to start the API server)")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\n📝 To fix:")
        print("   1. Install dependencies: pip install -r requirements_full.txt")
        print("   2. Make sure dataset.zip is extracted: unzip dataset.zip")
        print("   3. Train the model: python train_model.py")
    
    print("="*70)
    
    # Try prediction if model exists
    if Path('tulsi_disease_detection_best_model.h5').exists():
        test_prediction()

if __name__ == "__main__":
    main()