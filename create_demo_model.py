# Create a demo model for testing the API
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import json

# Create a simple model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Save the model
model.save('tulsi_disease_detection_best_model.h5')
print("✅ Demo model created and saved!")

# Create model configuration
config = {
    'best_model': 'Demo_CNN',
    'class_names': ['bacterial', 'fungal', 'healthy', 'pests'],
    'img_height': 224,
    'img_width': 224,
    'accuracy': 0.85,  # Demo accuracy
    'training_date': '2025-10-07',
    'dataset_info': {
        'total_images': 2274,
        'class_distribution': {
            'bacterial': 204,
            'fungal': 490,
            'healthy': 765,
            'pests': 815
        }
    }
}

with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Model configuration saved!")
print("🚀 Ready for API testing!")