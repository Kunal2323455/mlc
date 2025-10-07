# 🌿 Tulsi Plant Disease Detection System

A comprehensive machine learning system for detecting diseases in Tulsi (Holy Basil) plants using deep learning and computer vision techniques.

## 📋 Overview

This project implements an end-to-end solution for detecting four types of conditions in Tulsi plants:
- **Bacterial Infections** 🦠
- **Fungal Infections** 🍄  
- **Pest Infestations** 🐛
- **Healthy Plants** ✅

## 🎯 Key Features

- **High Accuracy**: Achieves 92.1% accuracy on test dataset
- **Multiple Model Architectures**: Enhanced CNN, VGG16, MobileNetV2 implementations
- **Real-time Predictions**: FastAPI-based REST API for instant disease detection
- **Treatment Recommendations**: Automated treatment suggestions for detected diseases
- **Comprehensive Visualizations**: Confusion matrices, performance metrics, and training plots
- **Production Ready**: Containerized deployment with proper error handling

## 📊 Dataset Information

- **Total Images**: 2,274 high-quality plant images
- **Classes Distribution**:
  - Bacterial: 204 images (9.0%)
  - Fungal: 490 images (21.5%)
  - Healthy: 765 images (33.6%)
  - Pests: 815 images (35.8%)

## 🏗️ Architecture

### Model Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|----------|---------|
| Bacterial | 83.9% | 83.9% | 83.9% | 31 |
| Fungal | 91.6% | 87.8% | 89.7% | 74 |
| Healthy | 93.1% | 93.9% | 93.5% | 115 |
| Pests | 93.6% | 95.1% | 94.4% | 123 |

**Overall Accuracy: 92.1%**

### Model Architecture
- **Enhanced CNN** with residual connections
- **Transfer Learning** using pre-trained VGG16 and MobileNetV2
- **Data Augmentation** for improved generalization
- **Class Balancing** to handle imbalanced dataset

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### 1. Train the Model
```bash
python3 train_models.py
```

### 2. Evaluate Performance
```bash
python3 evaluate_model.py
```

### 3. Start the API Server
```bash
python3 api.py
```

### 4. Test the API
```bash
python3 test_api.py
```

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

### Model Information
```bash
GET /model-info
```

### Disease Prediction
```bash
POST /predict
Content-Type: multipart/form-data
Body: image file
```

**Response Example:**
```json
{
  "prediction": {
    "predicted_disease": "fungal",
    "confidence": 0.94,
    "confidence_percentage": "94.0%",
    "all_class_probabilities": {
      "bacterial": 0.02,
      "fungal": 0.94,
      "healthy": 0.03,
      "pests": 0.01
    },
    "is_healthy": false,
    "needs_treatment": true
  },
  "recommendation": "🍄 FUNGAL INFECTION detected.\n• Apply fungicide spray\n• Reduce humidity around plant\n• Ensure good drainage\n• Remove affected parts",
  "model_used": "Tulsi Disease Detection CNN"
}
```

## 💊 Treatment Recommendations

The system provides specific treatment recommendations for each detected condition:

### 🦠 Bacterial Infections
- Apply copper-based bactericide
- Improve air circulation
- Avoid overhead watering
- Remove infected leaves

### 🍄 Fungal Infections
- Apply fungicide spray
- Reduce humidity around plant
- Ensure good drainage
- Remove affected parts

### 🐛 Pest Infestations
- Apply neem oil or insecticidal soap
- Check for insects regularly
- Use yellow sticky traps
- Quarantine if necessary

### ✅ Healthy Plants
- Continue regular care and monitoring

## 📁 Project Structure

```
tulsi-disease-detection/
├── dataset/                    # Training dataset
│   ├── bacterial/
│   ├── fungal/
│   ├── healthy/
│   └── pests/
├── data_split/                 # Train/validation/test splits
├── models/                     # Saved model files
├── api.py                      # FastAPI server
├── detector.py                 # Model wrapper class
├── train_models.py            # Training script
├── evaluate_model.py          # Evaluation and visualization
├── main_optimized.py          # Comprehensive training pipeline
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🔬 Model Training Details

### Data Preprocessing
- **Image Resizing**: 224x224 pixels
- **Normalization**: Pixel values scaled to [0,1]
- **Data Augmentation**: Rotation, flipping, zoom, brightness adjustment

### Training Configuration
- **Batch Size**: 32
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Categorical crossentropy
- **Callbacks**: Early stopping, model checkpointing, learning rate reduction

### Class Balancing
- **Weighted Loss**: Applied class weights to handle imbalanced dataset
- **Stratified Splitting**: Maintains class distribution across train/val/test sets

## 📈 Performance Metrics

### Confusion Matrix
```
                Predicted
Actual    Bact  Fung  Heal  Pest
Bacterial  26    3     1     1
Fungal      2   65     4     3  
Healthy     1    2   108     4
Pests       2    1     3   117
```

### Key Insights
- **Best Performance**: Pests detection (95.1% recall)
- **Most Challenging**: Bacterial infections (83.9% recall)
- **Overall Robustness**: High precision across all classes
- **Clinical Relevance**: Low false negative rate for disease detection

## 🛠️ Technical Implementation

### Deep Learning Framework
- **TensorFlow/Keras**: Model development and training
- **Transfer Learning**: Pre-trained ImageNet weights
- **Custom Architecture**: Enhanced CNN with batch normalization

### API Framework
- **FastAPI**: High-performance async API
- **Pydantic**: Data validation and serialization
- **CORS**: Cross-origin resource sharing enabled

### Deployment Features
- **Model Versioning**: Configuration management
- **Error Handling**: Comprehensive exception handling
- **File Management**: Automatic cleanup of temporary files
- **Health Monitoring**: API health check endpoints

## 📊 Visualization Features

The system includes comprehensive visualization tools:

- **Dataset Distribution**: Class balance and sample images
- **Training Progress**: Loss and accuracy curves
- **Confusion Matrix**: Detailed prediction analysis
- **Performance Metrics**: Per-class precision, recall, F1-score
- **Sample Predictions**: Visual prediction demonstrations

## 🔧 Configuration

### Model Configuration (`model_config.json`)
```json
{
  "best_model": "Enhanced_CNN",
  "class_names": ["bacterial", "fungal", "healthy", "pests"],
  "img_height": 224,
  "img_width": 224,
  "accuracy": 0.921,
  "training_date": "2025-10-07"
}
```

## 🚀 Deployment Options

### Local Development
```bash
python3 api.py
# Server runs on http://localhost:8000
```

### Production Deployment
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 📝 Usage Examples

### Python Client
```python
import requests

# Make prediction
with open('plant_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )

result = response.json()
print(f"Disease: {result['prediction']['predicted_disease']}")
print(f"Confidence: {result['prediction']['confidence_percentage']}")
print(f"Treatment: {result['recommendation']}")
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@plant_image.jpg"
```

## 🧪 Testing

### Unit Tests
```bash
python3 -m pytest tests/
```

### API Testing
```bash
python3 test_api.py
```

### Performance Benchmarking
```bash
python3 evaluate_model.py
```

## 📚 Research and References

This project implements state-of-the-art techniques in plant disease detection:

- **Deep Learning**: Convolutional Neural Networks for image classification
- **Transfer Learning**: Leveraging pre-trained models for improved performance
- **Data Augmentation**: Techniques to improve model generalization
- **Class Imbalance**: Methods to handle unbalanced datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset contributors for providing high-quality plant images
- TensorFlow team for the deep learning framework
- FastAPI team for the excellent web framework
- Open source community for various tools and libraries

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation

---

**Built with ❤️ for sustainable agriculture and plant health monitoring**