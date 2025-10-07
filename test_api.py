# Test script for the Tulsi Disease Detection API
import requests
import os
import json

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 TESTING TULSI DISEASE DETECTION API")
    print("="*50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check: PASSED")
            print(f"   Response: {response.json()}")
        else:
            print("❌ Health check: FAILED")
            return False
    except Exception as e:
        print(f"❌ Health check: ERROR - {e}")
        return False
    
    # Test model info endpoint
    try:
        response = requests.get(f"{base_url}/model-info")
        if response.status_code == 200:
            print("✅ Model info: PASSED")
            print(f"   Response: {response.json()}")
        else:
            print("❌ Model info: FAILED")
    except Exception as e:
        print(f"❌ Model info: ERROR - {e}")
    
    # Test prediction endpoint with a sample image
    sample_image_path = None
    for class_name in ['bacterial', 'fungal', 'healthy', 'pests']:
        class_path = os.path.join('dataset', class_name)
        if os.path.exists(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                sample_image_path = os.path.join(class_path, images[0])
                break
    
    if sample_image_path and os.path.exists(sample_image_path):
        try:
            with open(sample_image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{base_url}/predict", files=files)
            
            if response.status_code == 200:
                print("✅ Prediction endpoint: PASSED")
                result = response.json()
                print(f"   Predicted disease: {result['prediction']['predicted_disease']}")
                print(f"   Confidence: {result['prediction']['confidence_percentage']}")
                print(f"   Treatment: {result['recommendation'][:100]}...")
            else:
                print("❌ Prediction endpoint: FAILED")
                print(f"   Status: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"❌ Prediction endpoint: ERROR - {e}")
    else:
        print("⚠️ No sample image found for prediction test")
    
    print("\n🎯 API testing completed!")

if __name__ == "__main__":
    test_api()