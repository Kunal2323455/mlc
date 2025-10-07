# 🚀 Quick Start Guide - Tulsi Disease Detection

## ⚡ Fastest Way to Get Started

### 1. Test the Model (Prediction)

```python
from detector import TulsiDiseaseDetector

# Initialize detector
detector = TulsiDiseaseDetector(
    'tulsi_disease_detection_best_model.h5',
    'model_config.json'
)

# Make prediction
result = detector.predict('your_image.jpg')

# View results
print(f"🔍 Prediction: {result['predicted_disease']}")
print(f"📊 Confidence: {result['confidence_percentage']}")
print(f"💚 Healthy: {result['is_healthy']}")

# Get treatment advice
advice = detector.get_treatment_recommendation(result)
print(f"\n💊 Treatment:\n{advice}")
```

### 2. Run the API Server

```bash
# Start server
python3 api.py

# Server will run at http://localhost:8000
```

Test with curl:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_image.jpg" \
  -F "model_name=mobilenet"
```

### 3. View Model Performance

Check the visualizations:
- `visualization_01_class_distribution.png` - How data is distributed
- `visualization_03_confusion_matrix.png` - Where model makes mistakes
- `visualization_04_per_class_performance.png` - Performance by disease type
- `visualization_05_model_summary.png` - Complete model info

## 📊 Model Performance at a Glance

```
✅ Accuracy:  97.67%
✅ Bacterial: 96.88% F1-score
✅ Fungal:    97.26% F1-score
✅ Healthy:   97.87% F1-score
✅ Pests:     97.93% F1-score
```

## 🔧 Installation (If Needed)

```bash
pip install tensorflow numpy pillow
```

## 📁 Essential Files

| File | Purpose |
|------|---------|
| `tulsi_disease_detection_best_model.h5` | The trained model (14MB) |
| `model_config.json` | Model configuration |
| `detector.py` | Easy-to-use prediction class |
| `api.py` | Web API for deployment |

## 🎯 Supported Disease Classes

1. **bacterial** - Bacterial infections
2. **fungal** - Fungal infections
3. **healthy** - Healthy leaves
4. **pests** - Pest infestations

## 💡 Example Output

```python
{
  'predicted_disease': 'healthy',
  'confidence': 0.975,
  'confidence_percentage': '97.5%',
  'all_class_probabilities': {
    'bacterial': 0.005,
    'fungal': 0.010,
    'healthy': 0.975,
    'pests': 0.010
  },
  'is_healthy': True,
  'needs_treatment': False
}
```

## 🆘 Common Issues

### Issue: Model file not found
```bash
# Make sure you're in the project directory
cd /workspace
ls tulsi_disease_detection_best_model.h5
```

### Issue: TensorFlow not installed
```bash
pip install tensorflow
```

### Issue: Image not loading
```python
# Check image path and format (jpg, png supported)
import os
print(os.path.exists('your_image.jpg'))
```

## 🔄 Re-training the Model

Quick training (15 epochs):
```bash
python3 train_quick.py
```

Full training (50 epochs, 5 models):
```bash
python3 train_model.py
```

## 📞 Need Help?

Check these files:
- `PROJECT_SUMMARY.md` - Complete project documentation
- `test_system.py` - System verification
- `model_evaluation_results.csv` - Detailed metrics

---

**Ready to use!** The model is trained and ready for predictions. 🌿