# ====================================================================
# OPTIMIZED TULSI PLANT DISEASE DETECTION - FAST TRAINING VERSION
# Classes: Bacterial, Fungal, Pests, Healthy
# ====================================================================

import os
import shutil
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, Flatten, 
                                   Dense, BatchNormalization, GlobalAveragePooling2D,
                                   Activation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.regularizers import l2

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("✅ All libraries imported successfully!")
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ====================================================================
# DATASET SETUP AND ANALYSIS
# ====================================================================

dataset_root = "./dataset"

# Analyze dataset
def analyze_dataset(root_dir):
    classes = sorted([d for d in os.listdir(root_dir) 
                     if os.path.isdir(os.path.join(root_dir, d))])
    
    class_counts = {}
    total_images = 0
    
    for cls in classes:
        cls_path = os.path.join(root_dir, cls)
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        images = [f for f in os.listdir(cls_path) 
                 if f.lower().endswith(valid_extensions)]
        class_counts[cls] = len(images)
        total_images += len(images)
    
    return classes, class_counts, total_images

classes, class_counts, total_images = analyze_dataset(dataset_root)

print(f"\n📂 Disease Classes: {classes}")
print(f"📊 Total Images: {total_images}")
print("\n📈 Class Distribution:")
for cls, count in class_counts.items():
    percentage = (count / total_images) * 100
    print(f"  {cls}: {count} images ({percentage:.1f}%)")

# Visualize class distribution
plt.figure(figsize=(12, 5))

# Bar chart
plt.subplot(1, 2, 1)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = plt.bar(class_counts.keys(), class_counts.values(), color=colors)
plt.title('Tulsi Disease Dataset - Class Distribution', fontsize=14, fontweight='bold')
plt.ylabel('Number of Images')
plt.xlabel('Disease Classes')

for bar, count in zip(bars, class_counts.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             str(count), ha='center', va='bottom', fontweight='bold')

# Pie chart
plt.subplot(1, 2, 2)
plt.pie(class_counts.values(), labels=class_counts.keys(), colors=colors, 
        autopct='%1.1f%%', startangle=90)
plt.title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# ====================================================================
# DATA SPLITTING
# ====================================================================

def create_balanced_split(dataset_root, classes, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    train_dir = "data_split/train"
    val_dir = "data_split/validation"
    test_dir = "data_split/test"
    
    # Create directories
    for split_dir in [train_dir, val_dir, test_dir]:
        for cls in classes:
            os.makedirs(os.path.join(split_dir, cls), exist_ok=True)
    
    split_info = {}
    
    for cls in classes:
        cls_path = os.path.join(dataset_root, cls)
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(valid_extensions)]
        
        np.random.shuffle(images)
        
        total_images = len(images)
        train_split = int(total_images * train_ratio)
        val_split = int(total_images * (train_ratio + val_ratio))
        
        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]
        
        # Copy images
        for img in train_images:
            shutil.copy2(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
        for img in val_images:
            shutil.copy2(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))
        for img in test_images:
            shutil.copy2(os.path.join(cls_path, img), os.path.join(test_dir, cls, img))
        
        split_info[cls] = {
            'train': len(train_images),
            'validation': len(val_images),
            'test': len(test_images),
            'total': total_images
        }
    
    return train_dir, val_dir, test_dir, split_info

# Create splits
if not os.path.exists("data_split"):
    train_dir, val_dir, test_dir, split_info = create_balanced_split(dataset_root, classes)
    print("✅ Dataset split completed!")
    
    split_df = pd.DataFrame(split_info).T
    print("\n📊 Dataset Split Summary:")
    print(split_df)
else:
    train_dir, val_dir, test_dir = "data_split/train", "data_split/validation", "data_split/test"
    print("✅ Using existing dataset split")

# ====================================================================
# DATA GENERATORS
# ====================================================================

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
NUM_CLASSES = len(classes)

class_names = sorted(classes)
class_indices = {name: idx for idx, name in enumerate(class_names)}
print(f"🏷️ Class Mapping: {class_indices}")

# Calculate class weights
def calculate_class_weights(train_dir, classes):
    class_counts = []
    for cls in classes:
        cls_path = os.path.join(train_dir, cls)
        count = len(os.listdir(cls_path))
        class_counts.append(count)
    
    total_samples = sum(class_counts)
    class_weights = {}
    for i, count in enumerate(class_counts):
        class_weights[i] = total_samples / (len(classes) * count)
    
    return class_weights

class_weights = calculate_class_weights(train_dir, classes)
print(f"📊 Class Weights: {class_weights}")

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

validation_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

print(f"🔄 Training samples: {train_generator.samples}")
print(f"🔄 Validation samples: {validation_generator.samples}")
print(f"🔄 Test samples: {test_generator.samples}")

# ====================================================================
# MODEL ARCHITECTURES
# ====================================================================

def create_enhanced_cnn():
    """Enhanced CNN model"""
    model = Sequential([
        Conv2D(64, (3, 3), padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(512, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        GlobalAveragePooling2D(),
        Dense(512, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        
        Dense(256, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def create_vgg16_model():
    """VGG16 transfer learning model"""
    base_model = VGG16(weights='imagenet', include_top=False, 
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Freeze most layers, unfreeze last few
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(256, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def create_mobilenet_model():
    """MobileNetV2 transfer learning model"""
    base_model = MobileNetV2(weights='imagenet', include_top=False,
                           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Freeze base model initially
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(256, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

# ====================================================================
# TRAINING FUNCTION
# ====================================================================

def train_model(model, model_name, epochs=25):
    """Train model with callbacks"""
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f'best_{model_name}_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print(f"\n🚀 Training {model_name}...")
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return model, history

# ====================================================================
# TRAIN MODELS
# ====================================================================

models_to_train = {
    'Enhanced_CNN': create_enhanced_cnn(),
    'VGG16_Transfer': create_vgg16_model(),
    'MobileNet_Transfer': create_mobilenet_model()
}

trained_models = {}
training_histories = {}

for model_name, model in models_to_train.items():
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*60}")
    
    trained_model, history = train_model(model, model_name, epochs=25)
    trained_models[model_name] = trained_model
    training_histories[model_name] = history

# ====================================================================
# TRAINING VISUALIZATION
# ====================================================================

def plot_training_history(histories):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = ['blue', 'red', 'green']
    
    # Accuracy plot
    for i, (model_name, history) in enumerate(histories.items()):
        color = colors[i % len(colors)]
        axes[0].plot(history.history['accuracy'], 
                    label=f'{model_name} - Train', 
                    color=color, linestyle='-')
        axes[0].plot(history.history['val_accuracy'], 
                    label=f'{model_name} - Val', 
                    color=color, linestyle='--')
    
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    for i, (model_name, history) in enumerate(histories.items()):
        color = colors[i % len(colors)]
        axes[1].plot(history.history['loss'], 
                    label=f'{model_name} - Train', 
                    color=color, linestyle='-')
        axes[1].plot(history.history['val_loss'], 
                    label=f'{model_name} - Val', 
                    color=color, linestyle='--')
    
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_training_history(training_histories)

# ====================================================================
# MODEL EVALUATION
# ====================================================================

def evaluate_model(model, model_name, test_generator):
    """Evaluate model"""
    print(f"\n📊 Evaluating {model_name}...")
    
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f"✅ {model_name} Test Accuracy: {accuracy:.4f}")
    
    report = classification_report(true_classes, predicted_classes, 
                                 target_names=class_names, output_dict=True)
    cm = confusion_matrix(true_classes, predicted_classes)
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes,
        'classification_report': report,
        'confusion_matrix': cm
    }

# Evaluate all models
evaluation_results = {}
for model_name, model in trained_models.items():
    evaluation_results[model_name] = evaluate_model(model, model_name, test_generator)

# ====================================================================
# RESULTS VISUALIZATION
# ====================================================================

def plot_confusion_matrix(cm, class_names, model_name):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations
    annotations = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            row.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
        annotations.append(row)
    
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def create_comparison_report(evaluation_results):
    """Create model comparison"""
    comparison_data = []
    
    for model_name, results in evaluation_results.items():
        report = results['classification_report']
        
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision': report['macro avg']['precision'],
            'Recall': report['macro avg']['recall'],
            'F1-Score': report['macro avg']['f1-score']
        })
    
    return pd.DataFrame(comparison_data)

# Plot confusion matrices
for model_name, results in evaluation_results.items():
    plot_confusion_matrix(results['confusion_matrix'], class_names, model_name)

# Model comparison
comparison_df = create_comparison_report(evaluation_results)
print("\n🏆 MODEL PERFORMANCE COMPARISON:")
print("="*60)
print(comparison_df.round(4))

# Visualization
plt.figure(figsize=(12, 6))
x = np.arange(len(comparison_df))
width = 0.2

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']

for i, metric in enumerate(metrics):
    plt.bar(x + i*width, comparison_df[metric], width, 
            label=metric, color=colors[i], alpha=0.8)

plt.xlabel('Models', fontweight='bold')
plt.ylabel('Score', fontweight='bold')
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xticks(x + width*1.5, comparison_df['Model'], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Find best model
best_model_idx = comparison_df['Accuracy'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
best_model = trained_models[best_model_name]
best_accuracy = comparison_df.loc[best_model_idx, 'Accuracy']

print(f"\n🏅 Best performing model: {best_model_name}")
print(f"🎯 Best accuracy: {best_accuracy:.4f}")

# ====================================================================
# SAVE BEST MODEL
# ====================================================================

# Save the best model
best_model.save('tulsi_disease_detection_best_model.h5')
print(f"💾 Best model saved as 'tulsi_disease_detection_best_model.h5'")

# Save model configuration
model_config = {
    'best_model': best_model_name,
    'class_names': class_names,
    'img_height': IMG_HEIGHT,
    'img_width': IMG_WIDTH,
    'accuracy': float(best_accuracy),
    'class_weights': class_weights,
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_info': {
        'total_images': total_images,
        'class_distribution': class_counts,
        'train_samples': train_generator.samples,
        'val_samples': validation_generator.samples,
        'test_samples': test_generator.samples
    },
    'model_comparison': comparison_df.to_dict('records')
}

import json
with open('model_config.json', 'w') as f:
    json.dump(model_config, f, indent=2)

print("📋 Model configuration saved as 'model_config.json'")

# ====================================================================
# FINAL SUMMARY
# ====================================================================

print("\n" + "="*70)
print("🎉 TULSI DISEASE DETECTION TRAINING COMPLETED!")
print("="*70)

print(f"\n📊 DATASET SUMMARY:")
print(f"   • Total Images: {total_images}")
print(f"   • Classes: {', '.join(class_names)}")
print(f"   • Distribution: {dict(class_counts)}")

print(f"\n🏆 TRAINING RESULTS:")
print(f"   • Models Trained: {len(trained_models)}")
print(f"   • Best Model: {best_model_name}")
print(f"   • Best Accuracy: {best_accuracy:.2%}")

print(f"\n📁 FILES GENERATED:")
print(f"   ✅ tulsi_disease_detection_best_model.h5")
print(f"   ✅ model_config.json")
print(f"   ✅ Individual model checkpoints")

print(f"\n🚀 DEPLOYMENT STATUS: READY")
print("="*70)