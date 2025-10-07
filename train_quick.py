#!/usr/bin/env python3
"""
Quick training script for Tulsi disease detection (fewer epochs for testing)
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, Flatten, 
                                   Dense, BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16, MobileNetV2
import json

print("="*70)
print("🌿 TULSI PLANT DISEASE DETECTION - QUICK TRAINING")
print("="*70)
print(f"✅ TensorFlow Version: {tf.__version__}")

# Configuration
DATASET_ROOT = "./dataset"
DATA_SPLIT_DIR = "./data_split"
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 15  # Reduced for quick training
LEARNING_RATE = 0.0001
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Dataset analysis
print("\n📊 Analyzing Dataset...")
classes = sorted([d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))])
class_counts = {}
for cls in classes:
    cls_path = os.path.join(DATASET_ROOT, cls)
    images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    class_counts[cls] = len(images)
    print(f"  • {cls:12s}: {class_counts[cls]:4d} images")

# Create stratified split
print("\n📂 Creating data split...")
if os.path.exists(DATA_SPLIT_DIR):
    shutil.rmtree(DATA_SPLIT_DIR)

train_dir = os.path.join(DATA_SPLIT_DIR, "train")
val_dir = os.path.join(DATA_SPLIT_DIR, "validation")
test_dir = os.path.join(DATA_SPLIT_DIR, "test")

for split_dir in [train_dir, val_dir, test_dir]:
    for cls in classes:
        os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

split_info = {}
for cls in classes:
    cls_path = os.path.join(DATASET_ROOT, cls)
    images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    np.random.shuffle(images)
    
    total = len(images)
    train_split = int(total * TRAIN_RATIO)
    val_split = int(total * (TRAIN_RATIO + VAL_RATIO))
    
    train_images = images[:train_split]
    val_images = images[train_split:val_split]
    test_images = images[val_split:]
    
    for img in train_images:
        shutil.copy2(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
    for img in val_images:
        shutil.copy2(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))
    for img in test_images:
        shutil.copy2(os.path.join(cls_path, img), os.path.join(test_dir, cls, img))
    
    split_info[cls] = {'train': len(train_images), 'val': len(val_images), 'test': len(test_images)}

print("✅ Data split completed")

# Data generators
print("\n🔄 Creating data generators...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=True, seed=RANDOM_SEED
)

validation_generator = val_test_datagen.flow_from_directory(
    val_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

NUM_CLASSES = len(classes)
class_names = sorted(list(train_generator.class_indices.keys()))

print(f"✅ Train: {train_generator.samples}, Val: {validation_generator.samples}, Test: {test_generator.samples}")

# Class weights
class_weights_list = class_weight.compute_class_weight(
    'balanced', classes=np.unique(train_generator.classes), y=train_generator.classes
)
class_weights = dict(enumerate(class_weights_list))

# Model architectures
def create_mobilenet():
    base_model = MobileNetV2(weights='imagenet', include_top=False,
                            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ], name='MobileNetV2')
    return model

def create_vgg16():
    base_model = VGG16(weights='imagenet', include_top=False,
                      input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ], name='VGG16')
    return model

# Training function
def train_model(model, model_name):
    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*70}")
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(f'best_{model_name}_model.h5', monitor='val_accuracy', save_best_only=True, verbose=0)
    ]
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2
    )
    
    return model, history

# Train models
print("\n🚀 Starting training...")
models = {
    'MobileNetV2': create_mobilenet,
    'VGG16': create_vgg16
}

trained_models = {}
histories = {}

for name, model_fn in models.items():
    model = model_fn()
    trained_model, history = train_model(model, name)
    trained_models[name] = trained_model
    histories[name] = history
    tf.keras.backend.clear_session()

# Evaluation
print("\n📊 Evaluating models...")
results = {}

for model_name, model in trained_models.items():
    test_generator.reset()
    predictions = model.predict(test_generator, steps=test_generator.samples // BATCH_SIZE + 1, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes[:len(predicted_classes)]
    
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(true_classes, predicted_classes, average='weighted')
    
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': confusion_matrix(true_classes, predicted_classes)
    }
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

# Find best model
best_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = trained_models[best_name]
best_accuracy = results[best_name]['accuracy']

print(f"\n🏆 Best Model: {best_name} (Accuracy: {best_accuracy:.4f})")

# Save best model
best_model.save('tulsi_disease_detection_best_model.h5')
print(f"💾 Saved: tulsi_disease_detection_best_model.h5")

# Save config
config = {
    'best_model': best_name,
    'class_names': class_names,
    'img_height': IMG_HEIGHT,
    'img_width': IMG_WIDTH,
    'accuracy': float(best_accuracy),
    'precision': float(results[best_name]['precision']),
    'recall': float(results[best_name]['recall']),
    'f1_score': float(results[best_name]['f1']),
    'total_classes': NUM_CLASSES
}

with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("💾 Saved: model_config.json")

# Create visualizations
print("\n📊 Creating visualizations...")

# 1. Class distribution
plt.figure(figsize=(10, 6))
plt.bar(class_counts.keys(), class_counts.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
plt.title('Class Distribution', fontsize=14, fontweight='bold')
plt.ylabel('Number of Images')
plt.xlabel('Disease Classes')
for i, (k, v) in enumerate(class_counts.items()):
    plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('visualization_class_distribution.png', dpi=300)
plt.close()

# 2. Training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#3498db', '#e74c3c']

for i, (name, hist) in enumerate(histories.items()):
    ax1.plot(hist.history['accuracy'], label=f'{name} (Train)', color=colors[i], linestyle='-')
    ax1.plot(hist.history['val_accuracy'], label=f'{name} (Val)', color=colors[i], linestyle='--')
    ax2.plot(hist.history['loss'], label=f'{name} (Train)', color=colors[i], linestyle='-')
    ax2.plot(hist.history['val_loss'], label=f'{name} (Val)', color=colors[i], linestyle='--')

ax1.set_title('Model Accuracy', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.set_title('Model Loss', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualization_training_history.png', dpi=300)
plt.close()

# 3. Confusion matrix for best model
cm = results[best_name]['cm']
plt.figure(figsize=(8, 6))
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
annot = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - {best_name}', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')
plt.tight_layout()
plt.savefig('visualization_confusion_matrix.png', dpi=300)
plt.close()

# 4. Model comparison
comp_df = pd.DataFrame([
    {'Model': name, 'Accuracy': res['accuracy'], 'Precision': res['precision'], 
     'Recall': res['recall'], 'F1-Score': res['f1']}
    for name, res in results.items()
])

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comp_df))
width = 0.2

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

for i, metric in enumerate(metrics):
    ax.bar(x + i*width, comp_df[metric], width, label=metric, color=colors_bar[i], alpha=0.8)

ax.set_xlabel('Models', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(comp_df['Model'])
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visualization_model_comparison.png', dpi=300)
plt.close()

print("\n✅ Visualizations saved:")
print("  • visualization_class_distribution.png")
print("  • visualization_training_history.png")
print("  • visualization_confusion_matrix.png")
print("  • visualization_model_comparison.png")

print("\n" + "="*70)
print("🎉 TRAINING COMPLETED!")
print("="*70)
print(f"\n📋 Summary:")
print(f"  • Dataset: {sum(class_counts.values())} images, {len(classes)} classes")
print(f"  • Best Model: {best_name}")
print(f"  • Accuracy: {best_accuracy:.2%}")
print(f"  • Classes: {', '.join(class_names)}")
print("\n🚀 Ready for deployment!")
print("="*70)