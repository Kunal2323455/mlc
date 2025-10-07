import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class TulsiDiseaseDetector:
    """Deployment-ready Tulsi Disease Detection wrapper."""

    def __init__(self, model_path, config_path=None):
        self.model = load_model(model_path)

        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                self.class_names = self.config['class_names']
                self.img_height = self.config['img_height']
                self.img_width = self.config['img_width']
        else:
            # Fallback defaults - handle both fungal and fungi naming
            self.class_names = ['bacterial', 'fungal', 'healthy', 'pests']
            self.img_height = 224
            self.img_width = 224

    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array

    def predict(self, image_path):
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array, verbose=0)

        predicted_class_idx = int(np.argmax(predictions[0]))
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

        all_predictions = {}
        for i, class_name in enumerate(self.class_names):
            all_predictions[class_name] = float(predictions[0][i])

        return {
            'predicted_disease': predicted_class,
            'confidence': confidence,
            'confidence_percentage': f"{confidence*100:.1f}%",
            'all_class_probabilities': all_predictions,
            'is_healthy': predicted_class.lower() == 'healthy',
            'needs_treatment': predicted_class.lower() in ['bacterial', 'fungal', 'fungi', 'pests']
        }

    def get_treatment_recommendation(self, prediction_result):
        disease = prediction_result['predicted_disease'].lower()
        confidence = prediction_result['confidence']

        recommendations = {
            'healthy': "✅ Plant appears healthy! Continue regular care and monitoring.",
            'bacterial': (
                "🦠 BACTERIAL INFECTION detected.\n"
                "• Apply copper-based bactericide\n"
                "• Improve air circulation\n"
                "• Avoid overhead watering\n"
                "• Remove infected leaves"
            ),
            'fungal': (
                "🍄 FUNGAL INFECTION detected.\n"
                "• Apply fungicide spray\n"
                "• Reduce humidity around plant\n"
                "• Ensure good drainage\n"
                "• Remove affected parts"
            ),
            'fungi': (
                "🍄 FUNGAL INFECTION detected.\n"
                "• Apply fungicide spray\n"
                "• Reduce humidity around plant\n"
                "• Ensure good drainage\n"
                "• Remove affected parts"
            ),
            'pests': (
                "🐛 PEST INFESTATION detected.\n"
                "• Apply neem oil or insecticidal soap\n"
                "• Check for insects regularly\n"
                "• Use yellow sticky traps\n"
                "• Quarantine if necessary"
            ),
        }

        base = recommendations.get(disease, "Unknown condition detected.")
        if confidence < 0.6:
            base += f"\n\n⚠️ Note: Prediction confidence is {confidence*100:.1f}%. Consider consulting an expert for confirmation."
        return base