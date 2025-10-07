#!/usr/bin/env python3
# ====================================================================
# TULSI PLANT DISEASE DETECTION - IMPROVED TRAINING PIPELINE
# Classes: bacterial, fungal, healthy, pests
# ====================================================================

import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, Flatten, 
                                   Dense, BatchNormalization, GlobalAveragePooling2D,
                                   Input, Concatenate, Activation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, EfficientNetB0
import json

print("="*70)
print("🌿 TULSI PLANT DISEASE DETECTION - TRAINING PIPELINE")
print("="*70)
print(f"✅ TensorFlow Version: {tf.__version__}")
print(f"✅ GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ====================================================================
# CONFIGURATION
# ====================================================================

# Dataset configuration
DATASET_ROOT = "./dataset"
DATA_SPLIT_DIR = "./data_split"

# Training configuration
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ====================================================================
# STEP 1: DATASET ANALYSIS
# ====================================================================

print("\n" + "="*70)
print("STEP 1: ANALYZING DATASET")
print("="*70)

# Get class folders
classes = sorted([d for d in os.listdir(DATASET_ROOT) 
                 if os.path.isdir(os.path.join(DATASET_ROOT, d))])

print(f"📂 Classes found: {classes}")

# Count images per class
class_counts = {}
for cls in classes:
    cls_path = os.path.join(DATASET_ROOT, cls)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = [f for f in os.listdir(cls_path) if f.lower().endswith(valid_extensions)]
    class_counts[cls] = len(images)

print(f"\n📊 Dataset Distribution:")
for cls, count in class_counts.items():
    print(f"  • {cls:12s}: {count:4d} images")

total_images = sum(class_counts.values())
print(f"  • {'Total':12s}: {total_images:4d} images")

# Visualize class distribution
plt.figure(figsize=(12, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = plt.bar(class_counts.keys(), class_counts.values(), color=colors)
plt.title('Tulsi Leaf Disease Dataset - Class Distribution', fontsize=16, fontweight='bold')
plt.ylabel('Number of Images', fontsize=12)
plt.xlabel('Disease Classes', fontsize=12)

for bar, count in zip(bars, class_counts.values()):
    percentage = (count / total_images) * 100
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('01_class_distribution.png', dpi=300, bbox_inches='tight')
print("📊 Saved: 01_class_distribution.png")
plt.close()

# ====================================================================
# STEP 2: STRATIFIED DATA SPLITTING
# ====================================================================

print("\n" + "="*70)
print("STEP 2: SPLITTING DATASET")
print("="*70)

def create_stratified_split():
    """Create stratified train/validation/test split"""
    
    train_dir = os.path.join(DATA_SPLIT_DIR, "train")
    val_dir = os.path.join(DATA_SPLIT_DIR, "validation")
    test_dir = os.path.join(DATA_SPLIT_DIR, "test")
    
    # Remove existing split if any
    if os.path.exists(DATA_SPLIT_DIR):
        shutil.rmtree(DATA_SPLIT_DIR)
    
    # Create directories
    for split_dir in [train_dir, val_dir, test_dir]:
        for cls in classes:
            os.makedirs(os.path.join(split_dir, cls), exist_ok=True)
    
    split_info = {}
    
    for cls in classes:
        cls_path = os.path.join(DATASET_ROOT, cls)
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(valid_extensions)]
        
        # Shuffle images
        np.random.shuffle(images)
        
        # Calculate split points
        total = len(images)
        train_split = int(total * TRAIN_RATIO)
        val_split = int(total * (TRAIN_RATIO + VAL_RATIO))
        
        # Split images
        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]
        
        # Copy images to respective directories
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
            'total': total
        }
    
    return train_dir, val_dir, test_dir, split_info

train_dir, val_dir, test_dir, split_info = create_stratified_split()

# Display split information
split_df = pd.DataFrame(split_info).T
print("\n📊 Dataset Split Summary:")
print(split_df)

# Visualize split
fig, ax = plt.subplots(figsize=(12, 6))
split_df[['train', 'validation', 'test']].plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#e74c3c'])
plt.title('Dataset Split Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Disease Classes', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.legend(['Train', 'Validation', 'Test'])
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('02_data_split_distribution.png', dpi=300, bbox_inches='tight')
print("📊 Saved: 02_data_split_distribution.png")
plt.close()

# ====================================================================
# STEP 3: DATA GENERATORS WITH AUGMENTATION
# ====================================================================

print("\n" + "="*70)
print("STEP 3: CREATING DATA GENERATORS")
print("="*70)

# Advanced augmentation for training
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
    channel_shift_range=0.2,
    fill_mode='nearest'
)

# Validation and test data (only rescaling)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=RANDOM_SEED
)

validation_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

NUM_CLASSES = len(classes)
class_names = sorted(list(train_generator.class_indices.keys()))

print(f"✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {validation_generator.samples}")
print(f"✅ Test samples: {test_generator.samples}")
print(f"✅ Class mapping: {train_generator.class_indices}")

# Calculate class weights to handle imbalance
class_weights_list = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights_list))
print(f"\n⚖️ Class Weights (to handle imbalance):")
for cls_name, cls_idx in train_generator.class_indices.items():
    print(f"  • {cls_name:12s}: {class_weights[cls_idx]:.3f}")

# Visualize sample augmented images
def show_augmented_images(generator, num_images=12):
    plt.figure(figsize=(15, 10))
    batch_images, batch_labels = next(generator)
    
    for i in range(min(num_images, len(batch_images))):
        plt.subplot(3, 4, i + 1)
        plt.imshow(batch_images[i])
        label_idx = np.argmax(batch_labels[i])
        class_name = class_names[label_idx]
        plt.title(f'{class_name}', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Augmented Training Images', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('03_augmented_samples.png', dpi=300, bbox_inches='tight')
    print("📊 Saved: 03_augmented_samples.png")
    plt.close()

show_augmented_images(train_generator)

# ====================================================================
# STEP 4: MODEL ARCHITECTURES
# ====================================================================

print("\n" + "="*70)
print("STEP 4: BUILDING MODEL ARCHITECTURES")
print("="*70)

def create_custom_cnn():
    """Enhanced Custom CNN"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ], name='Custom_CNN')
    
    return model

def create_vgg16_model():
    """VGG16 Transfer Learning"""
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
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ], name='VGG16_Transfer')
    
    return model

def create_resnet50_model():
    """ResNet50 Transfer Learning"""
    base_model = ResNet50(weights='imagenet', include_top=False,
                         input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ], name='ResNet50_Transfer')
    
    return model

def create_mobilenet_model():
    """MobileNetV2 Transfer Learning"""
    base_model = MobileNetV2(weights='imagenet', include_top=False,
                            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ], name='MobileNetV2_Transfer')
    
    return model

def create_efficientnet_model():
    """EfficientNetB0 Transfer Learning"""
    base_model = EfficientNetB0(weights='imagenet', include_top=False,
                               input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ], name='EfficientNetB0_Transfer')
    
    return model

# ====================================================================
# STEP 5: TRAINING FUNCTION
# ====================================================================

def compile_and_train_model(model, model_name, epochs=EPOCHS):
    """Compile and train model with callbacks"""
    
    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*70}")
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f'best_{model_name}_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        CSVLogger(f'{model_name}_training_log.csv')
    ]
    
    # Display summary
    print(f"\n🏗️ {model_name} Architecture Summary:")
    model.summary()
    
    # Calculate steps
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = validation_generator.samples // BATCH_SIZE
    
    # Train
    print(f"\n🚀 Starting training for {model_name}...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return model, history

# ====================================================================
# STEP 6: TRAIN ALL MODELS
# ====================================================================

print("\n" + "="*70)
print("STEP 6: TRAINING ALL MODELS")
print("="*70)

models_to_train = {
    'Custom_CNN': create_custom_cnn,
    'VGG16_Transfer': create_vgg16_model,
    'ResNet50_Transfer': create_resnet50_model,
    'MobileNetV2_Transfer': create_mobilenet_model,
    'EfficientNetB0_Transfer': create_efficientnet_model
}

trained_models = {}
training_histories = {}

for model_name, model_fn in models_to_train.items():
    model = model_fn()
    trained_model, history = compile_and_train_model(model, model_name, epochs=EPOCHS)
    trained_models[model_name] = trained_model
    training_histories[model_name] = history
    
    # Clear memory
    tf.keras.backend.clear_session()

# ====================================================================
# STEP 7: VISUALIZE TRAINING HISTORY
# ====================================================================

print("\n" + "="*70)
print("STEP 7: VISUALIZING TRAINING HISTORY")
print("="*70)

def plot_training_history(histories):
    """Plot comprehensive training history"""
    metrics = ['accuracy', 'loss', 'precision', 'recall']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for j, (model_name, history) in enumerate(histories.items()):
            color = colors[j % len(colors)]
            
            # Training metric
            if metric in history.history:
                epochs_range = range(1, len(history.history[metric]) + 1)
                ax.plot(epochs_range, history.history[metric],
                       label=f'{model_name} (Train)', color=color, linestyle='-', linewidth=2)
            
            # Validation metric
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                ax.plot(epochs_range, history.history[val_metric],
                       label=f'{model_name} (Val)', color=color, linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_title(f'{metric.capitalize()} vs Epochs', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=11)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('04_training_history_all_metrics.png', dpi=300, bbox_inches='tight')
    print("📊 Saved: 04_training_history_all_metrics.png")
    plt.close()

plot_training_history(training_histories)

# ====================================================================
# STEP 8: MODEL EVALUATION
# ====================================================================

print("\n" + "="*70)
print("STEP 8: EVALUATING MODELS ON TEST SET")
print("="*70)

def evaluate_model(model, model_name, test_gen):
    """Comprehensive model evaluation"""
    print(f"\n📊 Evaluating {model_name}...")
    
    # Reset generator
    test_gen.reset()
    
    # Get predictions
    predictions = model.predict(test_gen, steps=test_gen.samples // BATCH_SIZE + 1, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = test_gen.classes[:len(predicted_classes)]
    
    # Calculate metrics
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(true_classes, predicted_classes, average='weighted')
    
    print(f"✅ {model_name} Test Accuracy: {accuracy:.4f}")
    print(f"✅ {model_name} Precision: {precision:.4f}")
    print(f"✅ {model_name} Recall: {recall:.4f}")
    print(f"✅ {model_name} F1-Score: {f1:.4f}")
    
    # Classification report
    report = classification_report(true_classes, predicted_classes, 
                                  target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
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
# STEP 9: CONFUSION MATRICES
# ====================================================================

print("\n" + "="*70)
print("STEP 9: GENERATING CONFUSION MATRICES")
print("="*70)

def plot_confusion_matrix(cm, class_names, model_name):
    """Plot beautiful confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'05_confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    print(f"📊 Saved: 05_confusion_matrix_{model_name}.png")
    plt.close()

# Plot confusion matrices
for model_name, results in evaluation_results.items():
    plot_confusion_matrix(results['confusion_matrix'], class_names, model_name)

# ====================================================================
# STEP 10: MODEL COMPARISON
# ====================================================================

print("\n" + "="*70)
print("STEP 10: COMPARING ALL MODELS")
print("="*70)

# Create comparison DataFrame
comparison_data = []
for model_name, results in evaluation_results.items():
    comparison_data.append({
        'Model': model_name,
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1-Score': results['f1_score']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

print("\n🏆 MODEL PERFORMANCE COMPARISON:")
print("="*70)
print(comparison_df.to_string(index=False))

# Save comparison to CSV
comparison_df.to_csv('model_comparison_results.csv', index=False)
print("\n💾 Saved: model_comparison_results.csv")

# Visualize comparison
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(comparison_df))
width = 0.2

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

for i, metric in enumerate(metrics):
    ax.bar(x + i*width, comparison_df[metric], width, 
           label=metric, color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.2)

ax.set_xlabel('Models', fontsize=13, fontweight='bold')
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Model Performance Comparison - All Metrics', fontsize=16, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax.legend(fontsize=11, loc='lower right')
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('06_model_comparison.png', dpi=300, bbox_inches='tight')
print("📊 Saved: 06_model_comparison.png")
plt.close()

# ====================================================================
# STEP 11: PER-CLASS PERFORMANCE
# ====================================================================

print("\n" + "="*70)
print("STEP 11: PER-CLASS PERFORMANCE ANALYSIS")
print("="*70)

# Get best model
best_model_name = comparison_df.iloc[0]['Model']
best_results = evaluation_results[best_model_name]
best_report = best_results['classification_report']

# Create per-class comparison
per_class_data = []
for cls in class_names:
    per_class_data.append({
        'Class': cls,
        'Precision': best_report[cls]['precision'],
        'Recall': best_report[cls]['recall'],
        'F1-Score': best_report[cls]['f1-score'],
        'Support': int(best_report[cls]['support'])
    })

per_class_df = pd.DataFrame(per_class_data)
print(f"\n📊 Per-Class Performance ({best_model_name}):")
print(per_class_df.to_string(index=False))

# Visualize per-class performance
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(per_class_df))
width = 0.25

metrics = ['Precision', 'Recall', 'F1-Score']
colors = ['#3498db', '#2ecc71', '#e74c3c']

for i, metric in enumerate(metrics):
    ax.bar(x + i*width, per_class_df[metric], width, 
           label=metric, color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.2)

ax.set_xlabel('Disease Classes', fontsize=13, fontweight='bold')
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title(f'Per-Class Performance - {best_model_name}', fontsize=16, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(per_class_df['Class'])
ax.legend(fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('07_per_class_performance.png', dpi=300, bbox_inches='tight')
print("📊 Saved: 07_per_class_performance.png")
plt.close()

# ====================================================================
# STEP 12: SAVE BEST MODEL
# ====================================================================

print("\n" + "="*70)
print("STEP 12: SAVING BEST MODEL")
print("="*70)

best_model = trained_models[best_model_name]
best_accuracy = comparison_df.iloc[0]['Accuracy']

# Save model
best_model.save('tulsi_disease_detection_best_model.h5')
print(f"💾 Saved: tulsi_disease_detection_best_model.h5")

# Save model configuration
model_config = {
    'best_model': best_model_name,
    'class_names': class_names,
    'img_height': IMG_HEIGHT,
    'img_width': IMG_WIDTH,
    'accuracy': float(best_accuracy),
    'precision': float(comparison_df.iloc[0]['Precision']),
    'recall': float(comparison_df.iloc[0]['Recall']),
    'f1_score': float(comparison_df.iloc[0]['F1-Score']),
    'total_classes': NUM_CLASSES,
    'total_parameters': int(best_model.count_params())
}

with open('model_config.json', 'w') as f:
    json.dump(model_config, f, indent=2)

print("💾 Saved: model_config.json")

# ====================================================================
# FINAL SUMMARY
# ====================================================================

print("\n" + "="*70)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)

print("\n📋 FINAL SUMMARY:")
print(f"  ✅ Total Images: {total_images}")
print(f"  ✅ Classes: {', '.join(class_names)}")
print(f"  ✅ Models Trained: {len(trained_models)}")
print(f"  ✅ Best Model: {best_model_name}")
print(f"  ✅ Best Accuracy: {best_accuracy:.2%}")
print(f"  ✅ Best Precision: {comparison_df.iloc[0]['Precision']:.2%}")
print(f"  ✅ Best Recall: {comparison_df.iloc[0]['Recall']:.2%}")
print(f"  ✅ Best F1-Score: {comparison_df.iloc[0]['F1-Score']:.2%}")

print("\n📁 Generated Files:")
print("  • tulsi_disease_detection_best_model.h5 (Best model)")
print("  • model_config.json (Configuration)")
print("  • model_comparison_results.csv (All results)")
print("  • 01_class_distribution.png")
print("  • 02_data_split_distribution.png")
print("  • 03_augmented_samples.png")
print("  • 04_training_history_all_metrics.png")
print("  • 05_confusion_matrix_*.png (for each model)")
print("  • 06_model_comparison.png")
print("  • 07_per_class_performance.png")

print("\n" + "="*70)
print("🌿 Ready for deployment!")
print("="*70)