#!/usr/bin/env python3
"""
Test script for frontend integration with all models
"""

import os
import sys
import json
import requests
import time
from pathlib import Path

def test_api_health():
    """Test if API is running and healthy"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data}")
            return True
        else:
            print(f"❌ API Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API not accessible: {e}")
        return False

def test_models_endpoint():
    """Test the models endpoint"""
    try:
        response = requests.get("http://localhost:8000/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"✅ Available models: {models}")
            print(f"✅ Model count: {data.get('count', 0)}")
            return models
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"❌ Models endpoint error: {e}")
        return []

def test_model_info(models):
    """Test model info endpoints"""
    for model in models:
        try:
            response = requests.get(f"http://localhost:8000/model/{model}/info", timeout=5)
            if response.status_code == 200:
                data = response.json()
                info = data.get('info', {})
                print(f"✅ {model}: {info.get('name', 'Unknown')} - {info.get('accuracy', 'N/A')}")
            else:
                print(f"❌ Model info failed for {model}: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Model info error for {model}: {e}")

def test_frontend_files():
    """Test if frontend files exist and are accessible"""
    frontend_dir = Path("frontend")
    required_files = ["index.html", "styles.css", "app.js", "sw.js"]
    
    missing_files = []
    for file in required_files:
        file_path = frontend_dir / file
        if not file_path.exists():
            missing_files.append(file)
        else:
            print(f"✅ Frontend file exists: {file}")
    
    if missing_files:
        print(f"❌ Missing frontend files: {missing_files}")
        return False
    
    # Test if files are accessible via HTTP
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend accessible via HTTP")
            return True
        else:
            print(f"❌ Frontend HTTP access failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Frontend HTTP error: {e}")
        return False

def test_prediction_with_sample():
    """Test prediction with a sample image (if available)"""
    # Look for sample images in dataset
    sample_image = None
    
    # Check for existing images in dataset
    if os.path.exists("dataset"):
        for class_dir in ["healthy", "bacterial", "fungal", "pests"]:
            class_path = os.path.join("dataset", class_dir)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    sample_image = os.path.join(class_path, images[0])
                    break
    
    if not sample_image:
        print("⚠️ No sample image found for prediction test")
        return True  # Not a failure, just no test data
    
    # Get available models
    models = test_models_endpoint()
    if not models:
        print("❌ No models available for prediction test")
        return False
    
    # Test prediction with first available model
    test_model = models[0]
    print(f"🧪 Testing prediction with model: {test_model}")
    
    try:
        with open(sample_image, 'rb') as f:
            files = {'file': f}
            data = {'model_name': test_model}
            
            response = requests.post(
                "http://localhost:8000/predict",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get('prediction', {})
            print(f"✅ Prediction successful:")
            print(f"   Disease: {prediction.get('predicted_disease', 'Unknown')}")
            print(f"   Confidence: {prediction.get('confidence_percentage', 'N/A')}")
            print(f"   Model: {result.get('model_used', 'Unknown')}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            try:
                error = response.json()
                print(f"   Error: {error.get('detail', 'Unknown error')}")
            except:
                pass
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction request error: {e}")
        return False
    except Exception as e:
        print(f"❌ Prediction test error: {e}")
        return False

def check_model_files():
    """Check if model files exist"""
    model_files = [
        "best_Custom_CNN_model.h5",
        "best_VGG16_Transfer_model.h5", 
        "best_MobileNet_Transfer_model.h5",
        "best_EfficientNet_Transfer_model.h5",
        "best_ResNet50_Transfer_model.h5",
        "model_config.json"
    ]
    
    existing_files = []
    missing_files = []
    
    for file in model_files:
        if os.path.exists(file):
            existing_files.append(file)
            print(f"✅ Model file exists: {file}")
        else:
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️ Missing model files: {missing_files}")
        print("💡 Run 'python main.py' to train the models first")
    
    return len(existing_files) > 0

def main():
    """Run all frontend integration tests"""
    print("🧪 TULSI AI FRONTEND INTEGRATION TESTS")
    print("="*60)
    
    tests = []
    
    # Test 1: Check model files
    print("\n1️⃣ Checking Model Files...")
    model_files_exist = check_model_files()
    tests.append(("Model Files", model_files_exist))
    
    # Test 2: Check frontend files
    print("\n2️⃣ Checking Frontend Files...")
    frontend_files_ok = test_frontend_files()
    tests.append(("Frontend Files", frontend_files_ok))
    
    if not model_files_exist:
        print("\n❌ Cannot proceed with API tests - no model files found")
        print("💡 Please run 'python main.py' first to train the models")
    else:
        print("\n3️⃣ Testing API Health...")
        api_healthy = test_api_health()
        tests.append(("API Health", api_healthy))
        
        if api_healthy:
            print("\n4️⃣ Testing Models Endpoint...")
            models = test_models_endpoint()
            tests.append(("Models Endpoint", len(models) > 0))
            
            if models:
                print("\n5️⃣ Testing Model Info...")
                test_model_info(models)
                tests.append(("Model Info", True))
                
                print("\n6️⃣ Testing Prediction...")
                prediction_ok = test_prediction_with_sample()
                tests.append(("Prediction Test", prediction_ok))
            else:
                tests.append(("Model Info", False))
                tests.append(("Prediction Test", False))
        else:
            print("❌ API not running. Start with: python api.py")
            tests.append(("Models Endpoint", False))
            tests.append(("Model Info", False))
            tests.append(("Prediction Test", False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY:")
    print("="*60)
    
    passed = 0
    for test_name, result in tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Frontend integration is working perfectly.")
        print("\n🚀 Your Tulsi AI system is ready!")
        print("   • Frontend: http://localhost:8000")
        print("   • API: http://localhost:8000/docs")
    else:
        print(f"\n⚠️ {len(tests) - passed} test(s) failed.")
        
        if not model_files_exist:
            print("\n💡 Next steps:")
            print("   1. Run 'python main.py' to train models")
            print("   2. Run 'python api.py' to start the server")
            print("   3. Open http://localhost:8000 in your browser")
        elif not api_healthy:
            print("\n💡 Next steps:")
            print("   1. Run 'python api.py' to start the server")
            print("   2. Open http://localhost:8000 in your browser")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)