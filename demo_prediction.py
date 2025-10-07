#!/usr/bin/env python3
"""
Demo script showing how to use the Tulsi disease detector
"""

import os
from detector import TulsiDiseaseDetector

print("="*70)
print("🌿 TULSI DISEASE DETECTION - DEMO PREDICTION")
print("="*70)

# Initialize the detector
print("\n📥 Loading model...")
detector = TulsiDiseaseDetector(
    'tulsi_disease_detection_best_model.h5',
    'model_config.json'
)
print("✅ Model loaded successfully!")

# Find sample images from each class
sample_images = {}
for class_name in ['bacterial', 'fungal', 'healthy', 'pests']:
    class_path = os.path.join('dataset', class_name)
    if os.path.exists(class_path):
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if images:
            sample_images[class_name] = os.path.join(class_path, images[0])

print(f"\n🔍 Testing on {len(sample_images)} sample images...")
print("="*70)

# Test on each class
for true_class, image_path in sample_images.items():
    print(f"\n📸 TRUE CLASS: {true_class.upper()}")
    print(f"   Image: {os.path.basename(image_path)}")
    
    # Make prediction
    result = detector.predict(image_path)
    
    # Display results
    print(f"\n   🎯 PREDICTION: {result['predicted_disease'].upper()}")
    print(f"   📊 Confidence: {result['confidence_percentage']}")
    
    # Check if prediction is correct
    is_correct = result['predicted_disease'].lower() == true_class.lower()
    print(f"   ✅ Correct!" if is_correct else f"   ❌ Incorrect")
    
    # Show all probabilities
    print(f"\n   📈 All Probabilities:")
    for disease, prob in result['all_class_probabilities'].items():
        bar = "█" * int(prob * 50)
        print(f"      {disease:12s} {prob*100:5.1f}% {bar}")
    
    # Get treatment recommendation
    print(f"\n   💊 Treatment Recommendation:")
    recommendation = detector.get_treatment_recommendation(result)
    for line in recommendation.split('\n'):
        print(f"      {line}")
    
    print("\n" + "-"*70)

print("\n" + "="*70)
print("🎉 DEMO COMPLETED!")
print("="*70)

print("\n💡 Usage in your code:")
print("""
from detector import TulsiDiseaseDetector

detector = TulsiDiseaseDetector(
    'tulsi_disease_detection_best_model.h5',
    'model_config.json'
)

result = detector.predict('path/to/your/image.jpg')
print(f"Disease: {result['predicted_disease']}")
print(f"Confidence: {result['confidence_percentage']}")

treatment = detector.get_treatment_recommendation(result)
print(treatment)
""")