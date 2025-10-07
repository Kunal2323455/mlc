# 🌿 Tulsi Leaf Plant Disease Detection - Project Summary

## 📊 Project Overview

This project implements a high-accuracy machine learning model for detecting diseases in Tulsi (Holy Basil) leaves. The model can classify images into 4 categories:
- **Bacterial** infections
- **Fungal** infections
- **Healthy** leaves
- **Pest** infestations

## 🎯 Model Performance

### Best Model: MobileNetV2 (Transfer Learning)

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **97.67%** |
| **Test Precision** | 97.76% |
| **Test Recall** | 97.67% |
| **Test F1-Score** | 97.67% |
| **Total Parameters** | 2,620,356 |

### Per-Class Performance

| Disease Class | Precision | Recall | F1-Score | Test Samples |
|--------------|-----------|--------|----------|--------------|
| **Bacterial** | 93.94% | 100.00% | 96.88% | 31 |
| **Fungal** | 98.61% | 95.95% | 97.26% | 74 |
| **Healthy** | 95.83% | 100.00% | 97.87% | 115 |
| **Pests** | 100.00% | 95.93% | 97.93% | 123 |

## 📁 Dataset Information

- **Total Images**: 2,274
- **Training Set**: 1,590 images (70%)
- **Validation Set**: 341 images (15%)
- **Test Set**: 343 images (15%)

### Class Distribution

| Class | Images | Percentage |
|-------|--------|------------|
| Bacterial | 204 | 9.0% |
| Fungal | 490 | 21.5% |
| Healthy | 765 | 33.6% |
| Pests | 815 | 35.8% |

## 🏗️ Model Architecture

**Base Model**: MobileNetV2 (Pre-trained on ImageNet)
- Transfer learning approach
- Base model frozen during training
- Custom classification head added

**Input**: 224×224×3 RGB images

**Architecture Layers**:
1. MobileNetV2 base (frozen)
2. GlobalAveragePooling2D
3. Dense(256) + ReLU + BatchNormalization + Dropout(0.5)
4. Dense(128) + ReLU + Dropout(0.3)
5. Dense(4) + Softmax

## 🔬 Training Details

- **Optimizer**: Adam (learning_rate=0.0001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 15 (with early stopping)
- **Class Weights**: Applied to handle class imbalance

### Data Augmentation

Training images were augmented with:
- Rotation (±40°)
- Width/Height shifts (±20%)
- Shear transformations (±20%)
- Zoom (±30%)
- Horizontal & vertical flips
- Brightness adjustment (70-130%)
- Channel shifts

## 📊 Visualizations Generated

1. **visualization_01_class_distribution.png** - Dataset class distribution
2. **visualization_02_data_split.png** - Train/Val/Test split distribution
3. **visualization_03_confusion_matrix.png** - Model confusion matrix with percentages
4. **visualization_04_per_class_performance.png** - Per-class metrics comparison
5. **visualization_05_model_summary.png** - Complete model summary card

## 📦 Generated Files

### Model Files
- `tulsi_disease_detection_best_model.h5` - Main model file (14MB)
- `model_config.json` - Model configuration and metadata
- `best_MobileNetV2_model.h5` - Original best model backup

### Results & Visualizations
- `model_evaluation_results.csv` - Detailed per-class results
- `visualization_*.png` - 5 comprehensive visualizations

### Code Files
- `detector.py` - Prediction wrapper class
- `train_model.py` - Full training pipeline (5 models)
- `train_quick.py` - Quick training script (2 models)
- `finalize_model.py` - Model evaluation and visualization
- `test_system.py` - System verification script
- `api.py` - FastAPI deployment server

## 🚀 Usage

### 1. Single Image Prediction

```python
from detector import TulsiDiseaseDetector

# Load detector
detector = TulsiDiseaseDetector(
    'tulsi_disease_detection_best_model.h5',
    'model_config.json'
)

# Predict
result = detector.predict('path/to/leaf/image.jpg')

print(f"Disease: {result['predicted_disease']}")
print(f"Confidence: {result['confidence_percentage']}")
print(f"Healthy: {result['is_healthy']}")

# Get treatment recommendation
treatment = detector.get_treatment_recommendation(result)
print(treatment)
```

### 2. API Server

```bash
python3 api.py
```

Then access at `http://localhost:8000`

### 3. Training New Model

```bash
# Quick training (15 epochs, 2 models)
python3 train_quick.py

# Full training (50 epochs, 5 models)
python3 train_model.py
```

## 🔧 Installation

```bash
# Install dependencies
pip install -r requirements_full.txt

# Or individual packages
pip install numpy pandas matplotlib seaborn scikit-learn
pip install opencv-python pillow tensorflow
pip install fastapi uvicorn python-multipart
```

## 🧪 System Testing

```bash
python3 test_system.py
```

## 📈 Key Features

✅ **High Accuracy**: 97.67% test accuracy
✅ **Balanced Performance**: All classes > 95% F1-score
✅ **Class Imbalance Handling**: Weighted loss function
✅ **Data Augmentation**: Comprehensive augmentation pipeline
✅ **Transfer Learning**: Leverages ImageNet pre-training
✅ **Lightweight Model**: Only 2.6M parameters
✅ **Treatment Recommendations**: Actionable advice for each disease
✅ **Production Ready**: FastAPI deployment included
✅ **Comprehensive Visualizations**: 5 detailed analysis charts

## 🎨 Treatment Recommendations

The system provides specific treatment recommendations for each disease:

- **Bacterial**: Copper-based bactericide, improved air circulation
- **Fungal**: Organic fungicides, humidity reduction, drainage improvement
- **Pests**: Neem oil, insecticidal soap, beneficial insects
- **Healthy**: Maintenance and regular monitoring advice

## 🔮 Future Improvements

1. Add more disease classes
2. Implement real-time detection from camera
3. Mobile app deployment
4. Multi-language support for recommendations
5. Integration with agricultural databases
6. Severity level detection within each disease class

## 👥 Contributors

This project was developed for Tulsi plant disease detection using state-of-the-art deep learning techniques.

## 📄 License

This project is for educational and research purposes.

---

**Generated on**: 2025-10-07
**Model Version**: 1.0
**Framework**: TensorFlow 2.20.0