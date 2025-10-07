#!/usr/bin/env python3
"""
Test script to validate the enhanced main.py functionality
"""

import os
import sys
import importlib.util

def test_imports():
    """Test if all required libraries can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        import tensorflow as tf
        print("✅ Core libraries imported successfully")
        print(f"   - NumPy: {np.__version__}")
        print(f"   - Pandas: {pd.__version__}")
        print(f"   - TensorFlow: {tf.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_dataset_structure():
    """Test dataset structure"""
    print("\n🔍 Testing dataset structure...")
    
    if not os.path.exists('dataset.zip'):
        print("❌ dataset.zip not found")
        return False
    
    if os.path.exists('dataset'):
        classes = sorted(os.listdir('dataset'))
        expected_classes = ['bacterial', 'fungal', 'healthy', 'pests']
        
        print(f"   Found classes: {classes}")
        
        # Check if we have the right classes (allowing for fungi/fungal variation)
        if 'fungi' in classes:
            classes = [cls if cls != 'fungi' else 'fungal' for cls in classes]
        
        missing_classes = set(expected_classes) - set(classes)
        if missing_classes:
            print(f"❌ Missing classes: {missing_classes}")
            return False
        
        # Count images in each class
        for cls in classes:
            cls_path = os.path.join('dataset', cls)
            if os.path.isdir(cls_path):
                count = len([f for f in os.listdir(cls_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
                print(f"   {cls}: {count} images")
        
        print("✅ Dataset structure is valid")
        return True
    else:
        print("⚠️ Dataset not extracted yet (this is normal)")
        return True

def test_model_functions():
    """Test if model creation functions are defined correctly"""
    print("\n🔍 Testing model function definitions...")
    
    # Load the main.py module
    spec = importlib.util.spec_from_file_location("main", "main.py")
    main_module = importlib.util.module_from_spec(spec)
    
    try:
        # Test if we can load the module without executing it
        with open('main.py', 'r') as f:
            content = f.read()
        
        # Check for key function definitions
        required_functions = [
            'create_custom_cnn',
            'create_vgg16_model', 
            'create_mobilenet_model',
            'create_efficientnet_model',
            'create_resnet_model',
            'compile_and_train_model',
            'plot_training_history',
            'evaluate_model',
            'create_ensemble_predictions'
        ]
        
        missing_functions = []
        for func in required_functions:
            if f'def {func}(' not in content:
                missing_functions.append(func)
        
        if missing_functions:
            print(f"❌ Missing functions: {missing_functions}")
            return False
        
        print("✅ All required functions are defined")
        return True
        
    except Exception as e:
        print(f"❌ Error testing functions: {e}")
        return False

def test_file_structure():
    """Test if all required files are present"""
    print("\n🔍 Testing file structure...")
    
    required_files = ['main.py', 'detector.py', 'api.py', 'requirements.txt']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"   ✅ {file}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files are present")
    return True

def main():
    """Run all tests"""
    print("🧪 TESTING ENHANCED TULSI DISEASE DETECTION SYSTEM")
    print("="*60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Library Imports", test_imports),
        ("Dataset Structure", test_dataset_structure),
        ("Model Functions", test_model_functions)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("📊 TEST SUMMARY:")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Your enhanced main.py is ready to run.")
        print("\n🚀 To start training:")
        print("   python3 main.py")
        print("\n📚 For more information, see README_IMPROVED.md")
    else:
        print(f"\n⚠️ {len(tests) - passed} test(s) failed. Please check the issues above.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)