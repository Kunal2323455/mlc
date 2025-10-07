# ====================================================================
# TULSI PLANT DISEASE DETECTION - ENHANCED MAIN SCRIPT
# Classes: Bacterial, Fungal, Pests, Healthy
# Enhanced with advanced models, ensemble methods, and comprehensive visualizations
# ====================================================================

# STEP 0 — Import All Required Libraries
import os
import shutil
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout, Flatten, 
                                   Dense, BatchNormalization, GlobalAveragePooling2D,
                                   GlobalMaxPooling2D, Concatenate, Input)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, EfficientNetB0, InceptionV3
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import json
from datetime import datetime

print("✅ All libraries imported successfully!")
print(f"TensorFlow Version: {tf.__version__}")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ====================================================================
# STEP 1 — Dataset Setup and Extraction
# ====================================================================

# Dataset path configuration
zip_path = 'dataset.zip'
dataset_root = "./dataset"

# Extract dataset
if not os.path.exists(dataset_root):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    print(f"✅ Extracted '{zip_path}' → {dataset_root}")
else:
    print(f"✅ Dataset already extracted at {dataset_root}")

# ====================================================================
# STEP 2 — Dataset Analysis and Preprocessing
# ====================================================================

def list_class_folders(root_dir):
    """Returns sorted list of class folders"""
    folders = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return sorted(folders)

def count_images(folder):
    """Count images in each class folder"""
    counts = {}
    for cls in sorted(os.listdir(folder)):
        cls_path = os.path.join(folder, cls)
        if os.path.isdir(cls_path):
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            counts[cls] = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith(valid_extensions)
            ])
    return counts

# Handle nested folder structure
classes = list_class_folders(dataset_root)
if len(classes) == 1:
    inner_dir = os.path.join(dataset_root, classes[0])
    inner_classes = list_class_folders(inner_dir)
    if len(inner_classes) >= 2:
        dataset_root = inner_dir
        classes = inner_classes

if len(classes) < 2:
    raise RuntimeError(f"❌ Found {len(classes)} class folders. Expected >= 2.")

print("📂 Disease Classes Found:", classes)
print("📊 Expected classes: ['bacterial', 'fungal', 'healthy', 'pests']")

# Fix class naming inconsistency (fungi -> fungal)
if 'fungi' in classes:
    old_path = os.path.join(dataset_root, 'fungi')
    new_path = os.path.join(dataset_root, 'fungal')
    if not os.path.exists(new_path):
        os.rename(old_path, new_path)
        print("✅ Renamed 'fungi' folder to 'fungal' for consistency")
    classes = [cls if cls != 'fungi' else 'fungal' for cls in classes]

# Count images before split
counts_before = count_images(dataset_root)
print("\n📈 Image Distribution:")
for cls, count in counts_before.items():
    print(f"  {cls}: {count} images")

# Enhanced visualization of class distribution
plt.figure(figsize=(12, 8))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = plt.bar(counts_before.keys(), counts_before.values(), 
               color=colors[:len(counts_before)], alpha=0.8, edgecolor='black', linewidth=1)

plt.title('Tulsi Plant Disease Dataset - Class Distribution', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('Number of Images', fontsize=14, fontweight='bold')
plt.xlabel('Disease Classes', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar, count in zip(bars, counts_before.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
             str(count), ha='center', va='bottom', fontweight='bold', fontsize=12)

# Add percentage labels
total_images = sum(counts_before.values())
for bar, count in zip(bars, counts_before.values()):
    percentage = (count / total_images) * 100
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
             f'{percentage:.1f}%', ha='center', va='center', 
             fontweight='bold', fontsize=11, color='white')

plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ====================================================================
# STEP 3 — Advanced Stratified Data Splitting
# ====================================================================

def create_stratified_split(dataset_root, classes, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create stratified train/validation/test split"""
    
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
        
        # Shuffle images with fixed seed for reproducibility
        np.random.seed(42)
        np.random.shuffle(images)
        
        # Calculate split points
        total_images = len(images)
        train_split = int(total_images * train_ratio)
        val_split = int(total_images * (train_ratio + val_ratio))
        
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
            'total': total_images
        }
    
    return train_dir, val_dir, test_dir, split_info

# Create splits if they don't exist
if not os.path.exists("data_split"):
    train_dir, val_dir, test_dir, split_info = create_stratified_split(dataset_root, classes)
    print("✅ Dataset split completed!")
    
    # Display split information with visualization
    split_df = pd.DataFrame(split_info).T
    print("\n📊 Dataset Split Summary:")
    print(split_df)
    
    # Visualize data split
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Split distribution by class
    split_df[['train', 'validation', 'test']].plot(kind='bar', ax=ax1, 
                                                   color=['#FF9999', '#66B2FF', '#99FF99'])
    ax1.set_title('Data Split Distribution by Class', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Number of Images', fontweight='bold')
    ax1.set_xlabel('Classes', fontweight='bold')
    ax1.legend(['Train', 'Validation', 'Test'])
    ax1.tick_params(axis='x', rotation=45)
    
    # Overall split distribution
    total_split = split_df[['train', 'validation', 'test']].sum()
    ax2.pie(total_split.values, labels=total_split.index, autopct='%1.1f%%',
            colors=['#FF9999', '#66B2FF', '#99FF99'], startangle=90)
    ax2.set_title('Overall Data Split Distribution', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.show()
else:
    train_dir, val_dir, test_dir = "data_split/train", "data_split/validation", "data_split/test"
    print("✅ Using existing dataset split")

# ====================================================================
# STEP 4 — Advanced Data Augmentation and Generators
# ====================================================================

# Image parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
NUM_CLASSES = len(classes)

# Create class name to index mapping
class_names = sorted(classes)
class_indices = {name: idx for idx, name in enumerate(class_names)}
print(f"🏷️ Class Mapping: {class_indices}")

# Advanced data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    channel_shift_range=0.2,
    fill_mode='nearest'
)

# Validation and test data (only rescaling)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
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
# STEP 5 — Visualization of Augmented Images
# ====================================================================

def show_augmented_images(generator, num_images=12):
    """Display augmented images from generator"""
    plt.figure(figsize=(18, 12))
    
    batch_images, batch_labels = next(generator)
    
    for i in range(min(num_images, len(batch_images))):
        plt.subplot(3, 4, i + 1)
        plt.imshow(batch_images[i])
        
        # Get class name from label
        label_idx = np.argmax(batch_labels[i])
        class_name = class_names[label_idx]
        plt.title(f'Class: {class_name}', fontsize=11, fontweight='bold')
        plt.axis('off')
        
        # Add border color based on class
        colors = {'bacterial': 'red', 'fungal': 'orange', 'healthy': 'green', 'pests': 'purple'}
        color = colors.get(class_name, 'black')
        for spine in plt.gca().spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    plt.suptitle('Augmented Training Images', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()

print("🖼️ Sample Augmented Images:")
show_augmented_images(train_generator)

# ====================================================================
# STEP 6 — Enhanced Model Architectures
# ====================================================================

def create_custom_cnn():
    """Create enhanced custom CNN architecture"""
    model = Sequential([
        # First Convolutional Block
        Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Fourth Convolutional Block
        Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Dense Layers
        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def create_vgg16_model():
    """Create VGG16 based transfer learning model"""
    base_model = VGG16(weights='imagenet', include_top=False, 
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Freeze base model layers
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def create_mobilenet_model():
    """Create MobileNetV2 based transfer learning model"""
    base_model = MobileNetV2(weights='imagenet', include_top=False,
                           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Freeze base model layers
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def create_efficientnet_model():
    """Create EfficientNetB0 based transfer learning model"""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, 
                               input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def create_resnet_model():
    """Create ResNet50 based transfer learning model"""
    base_model = ResNet50(weights='imagenet', include_top=False,
                         input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

# ====================================================================
# STEP 7 — Advanced Model Training Setup
# ====================================================================

def lr_schedule(epoch):
    """Learning rate scheduler"""
    lr = 0.001
    if epoch > 30:
        lr *= 0.1
    elif epoch > 20:
        lr *= 0.5
    elif epoch > 10:
        lr *= 0.8
    return lr

def compile_and_train_model(model, model_name, epochs=40, fine_tune=True):
    """Compile and train the model with advanced callbacks"""
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Enhanced callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=7,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            f'best_{model_name}_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        LearningRateScheduler(lr_schedule, verbose=1)
    ]
    
    # Display model summary
    print(f"\n🏗️ {model_name} Architecture:")
    model.summary()
    
    # Initial training
    print(f"\n🚀 Training {model_name}...")
    history = model.fit(
        train_generator,
        epochs=epochs//2 if fine_tune and hasattr(model.layers[0], 'trainable') else epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning for transfer learning models
    if fine_tune and hasattr(model.layers[0], 'trainable') and len(model.layers) > 1:
        print(f"\n🔧 Fine-tuning {model_name}...")
        
        # Unfreeze the base model
        model.layers[0].trainable = True
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Continue training
        history_fine = model.fit(
            train_generator,
            epochs=epochs//2,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1,
            initial_epoch=len(history.history['loss'])
        )
        
        # Combine histories
        for key in history.history.keys():
            history.history[key].extend(history_fine.history[key])
    
    return model, history

# ====================================================================
# STEP 8 — Train Multiple Models
# ====================================================================

# Create and train models
models_to_train = {
    'Custom_CNN': create_custom_cnn(),
    'VGG16_Transfer': create_vgg16_model(),
    'MobileNet_Transfer': create_mobilenet_model(),
    'EfficientNet_Transfer': create_efficientnet_model(),
    'ResNet50_Transfer': create_resnet_model()
}

trained_models = {}
training_histories = {}

for model_name, model in models_to_train.items():
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*80}")
    
    # Determine if fine-tuning should be applied
    fine_tune = 'Transfer' in model_name
    
    trained_model, history = compile_and_train_model(model, model_name, epochs=40, fine_tune=fine_tune)
    trained_models[model_name] = trained_model
    training_histories[model_name] = history

# ====================================================================
# STEP 9 — Training Results Visualization
# ====================================================================

def plot_training_history(histories, metrics=['accuracy', 'loss']):
    """Plot training history for multiple models"""
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 6*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, metric in enumerate(metrics):
        for j, (model_name, history) in enumerate(histories.items()):
            color = colors[j % len(colors)]
            
            # Training metric
            axes[i].plot(history.history[metric], 
                        label=f'{model_name} - Train', 
                        color=color, linestyle='-', linewidth=2)
            
            # Validation metric
            val_metric = f'val_{metric}'
            if val_metric in history.history:
                axes[i].plot(history.history[val_metric], 
                            label=f'{model_name} - Val', 
                            color=color, linestyle='--', linewidth=2)
        
        axes[i].set_title(f'Model {metric.capitalize()}', fontsize=16, fontweight='bold')
        axes[i].set_xlabel('Epochs', fontsize=12)
        axes[i].set_ylabel(metric.capitalize(), fontsize=12)
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Plot training histories
plot_training_history(training_histories, ['accuracy', 'loss', 'precision', 'recall'])

# ====================================================================
# STEP 10 — Model Evaluation and Comparison
# ====================================================================

def evaluate_model(model, model_name, test_generator):
    """Evaluate model and return comprehensive metrics"""
    print(f"\n📊 Evaluating {model_name}...")
    
    # Reset test generator
    test_generator.reset()
    
    # Get predictions
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = test_generator.classes
    
    # Calculate metrics
    accuracy = accuracy_score(true_classes, predicted_classes)
    
    print(f"✅ {model_name} Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(true_classes, predicted_classes, 
                                 target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes,
        'classification_report': report,
        'confusion_matrix': cm,
        'per_class_accuracy': per_class_accuracy
    }

# Evaluate all models
evaluation_results = {}
for model_name, model in trained_models.items():
    evaluation_results[model_name] = evaluate_model(model, model_name, test_generator)

# ====================================================================
# STEP 11 — Results Visualization and Analysis
# ====================================================================

def plot_confusion_matrix(cm, class_names, model_name, per_class_acc):
    """Plot enhanced confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Create annotation matrix with counts and percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annotations = np.empty_like(cm).astype(str)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]:.2%})'
    
    # Plot heatmap
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=class_names, yticklabels=class_names, 
                square=True, linewidths=0.5)
    
    plt.title(f'Confusion Matrix - {model_name}\nOverall Accuracy: {np.trace(cm)/np.sum(cm):.3f}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def create_comparison_report(evaluation_results):
    """Create comprehensive model comparison report"""
    comparison_data = []
    
    for model_name, results in evaluation_results.items():
        report = results['classification_report']
        
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision': report['macro avg']['precision'],
            'Recall': report['macro avg']['recall'],
            'F1-Score': report['macro avg']['f1-score'],
            'Weighted_F1': report['weighted avg']['f1-score']
        })
        
        # Add per-class accuracies
        for i, class_name in enumerate(class_names):
            comparison_data[-1][f'{class_name}_Acc'] = results['per_class_accuracy'][i]
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

# Plot confusion matrices for all models
for model_name, results in evaluation_results.items():
    plot_confusion_matrix(results['confusion_matrix'], class_names, 
                         model_name, results['per_class_accuracy'])

# Create and display comparison report
comparison_df = create_comparison_report(evaluation_results)
print("\n🏆 MODEL PERFORMANCE COMPARISON:")
print("="*80)
print(comparison_df.round(4))

# Enhanced model comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Overall metrics comparison
ax1 = axes[0, 0]
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(comparison_df))
width = 0.2

colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']
for i, metric in enumerate(metrics_to_plot):
    ax1.bar(x + i*width, comparison_df[metric], width, 
            label=metric, color=colors[i], alpha=0.8)

ax1.set_xlabel('Models', fontweight='bold')
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width*1.5)
ax1.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Per-class accuracy heatmap
ax2 = axes[0, 1]
class_acc_cols = [col for col in comparison_df.columns if col.endswith('_Acc')]
class_acc_data = comparison_df[class_acc_cols]
class_acc_data.columns = [col.replace('_Acc', '') for col in class_acc_data.columns]

im = ax2.imshow(class_acc_data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax2.set_xticks(range(len(comparison_df)))
ax2.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax2.set_yticks(range(len(class_names)))
ax2.set_yticklabels(class_names)
ax2.set_title('Per-Class Accuracy Heatmap', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(len(class_names)):
    for j in range(len(comparison_df)):
        text = ax2.text(j, i, f'{class_acc_data.iloc[j, i]:.3f}',
                       ha="center", va="center", color="black", fontweight='bold')

plt.colorbar(im, ax=ax2, label='Accuracy')

# Model ranking
ax3 = axes[1, 0]
ranking_scores = comparison_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].mean(axis=1)
sorted_indices = ranking_scores.argsort()[::-1]

bars = ax3.barh(range(len(comparison_df)), ranking_scores[sorted_indices], 
                color=plt.cm.viridis(np.linspace(0, 1, len(comparison_df))))
ax3.set_yticks(range(len(comparison_df)))
ax3.set_yticklabels(comparison_df.iloc[sorted_indices]['Model'])
ax3.set_xlabel('Average Score', fontweight='bold')
ax3.set_title('Model Ranking (Average of Key Metrics)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# Add score labels
for i, (idx, score) in enumerate(zip(sorted_indices, ranking_scores[sorted_indices])):
    ax3.text(score + 0.01, i, f'{score:.3f}', va='center', fontweight='bold')

# Accuracy vs Model Complexity
ax4 = axes[1, 1]
model_params = []
for model_name, model in trained_models.items():
    model_params.append(model.count_params())

scatter = ax4.scatter(model_params, comparison_df['Accuracy'], 
                     s=100, c=range(len(comparison_df)), 
                     cmap='tab10', alpha=0.7, edgecolors='black')

ax4.set_xlabel('Model Parameters (millions)', fontweight='bold')
ax4.set_ylabel('Accuracy', fontweight='bold')
ax4.set_title('Model Efficiency (Accuracy vs Complexity)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Add model labels
for i, model in enumerate(comparison_df['Model']):
    ax4.annotate(model[:10], (model_params[i], comparison_df['Accuracy'].iloc[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()

# Find best model
best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
best_model = trained_models[best_model_name]
best_accuracy = comparison_df['Accuracy'].max()

print(f"\n🏅 Best performing model: {best_model_name}")
print(f"🎯 Best accuracy: {best_accuracy:.4f}")

# ====================================================================
# STEP 12 — Model Ensemble for Ultimate Accuracy
# ====================================================================

def create_ensemble_predictions(models, test_generator, method='weighted'):
    """Create ensemble predictions from multiple models"""
    all_predictions = []
    model_accuracies = []
    
    for model_name, model in models.items():
        test_generator.reset()
        predictions = model.predict(test_generator, verbose=0)
        all_predictions.append(predictions)
        model_accuracies.append(evaluation_results[model_name]['accuracy'])
    
    # Ensemble methods
    if method == 'average':
        ensemble_pred = np.mean(all_predictions, axis=0)
    elif method == 'weighted':
        # Weight by individual model accuracy
        weights = np.array(model_accuracies) / sum(model_accuracies)
        ensemble_pred = np.average(all_predictions, axis=0, weights=weights)
    elif method == 'voting':
        # Majority voting
        individual_preds = [np.argmax(pred, axis=1) for pred in all_predictions]
        ensemble_pred_classes = []
        for i in range(len(individual_preds[0])):
            votes = [pred[i] for pred in individual_preds]
            ensemble_pred_classes.append(max(set(votes), key=votes.count))
        return np.array(ensemble_pred_classes)
    
    return ensemble_pred

# Create ensemble with top 3 models
print("\n🤖 Creating Model Ensemble...")
top_3_models = comparison_df.nlargest(3, 'Accuracy')['Model'].tolist()
ensemble_models = {name: trained_models[name] for name in top_3_models}

print(f"Ensemble models: {top_3_models}")

# Test ensemble methods
ensemble_methods = ['average', 'weighted', 'voting']
ensemble_results = {}

for method in ensemble_methods:
    print(f"\nTesting {method} ensemble...")
    
    if method == 'voting':
        ensemble_pred_classes = create_ensemble_predictions(ensemble_models, test_generator, method)
        true_classes = test_generator.classes
        ensemble_accuracy = accuracy_score(true_classes, ensemble_pred_classes)
    else:
        ensemble_pred = create_ensemble_predictions(ensemble_models, test_generator, method)
        ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
        true_classes = test_generator.classes
        ensemble_accuracy = accuracy_score(true_classes, ensemble_pred_classes)
    
    ensemble_results[method] = ensemble_accuracy
    print(f"{method.capitalize()} Ensemble Accuracy: {ensemble_accuracy:.4f}")

# Find best ensemble method
best_ensemble_method = max(ensemble_results.keys(), key=lambda x: ensemble_results[x])
best_ensemble_accuracy = ensemble_results[best_ensemble_method]

print(f"\n🏆 Best Ensemble Method: {best_ensemble_method}")
print(f"🎯 Best Ensemble Accuracy: {best_ensemble_accuracy:.4f}")

# Compare ensemble with individual models
print(f"\nAccuracy Improvement:")
print(f"Best Individual Model: {best_accuracy:.4f}")
print(f"Best Ensemble: {best_ensemble_accuracy:.4f}")
print(f"Improvement: {((best_ensemble_accuracy - best_accuracy) / best_accuracy * 100):+.2f}%")

# ====================================================================
# STEP 13 — Prediction Function and Testing
# ====================================================================

def predict_disease(image_path, model, class_names):
    """Predict disease from image path"""
    # Load and preprocess image
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_predictions': dict(zip(class_names, predictions[0]))
    }

def visualize_predictions(image_paths, model, class_names, num_images=8):
    """Visualize predictions on sample images"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, img_path in enumerate(image_paths[:num_images]):
        if i >= 8:
            break
            
        # Load image
        img = load_img(img_path)
        axes[i].imshow(img)
        
        # Make prediction
        result = predict_disease(img_path, model, class_names)
        
        # Set title with prediction
        title = f"Pred: {result['predicted_class']}\n"
        title += f"Conf: {result['confidence']:.2%}"
        
        # Color code by confidence
        if result['confidence'] > 0.8:
            color = 'green'
        elif result['confidence'] > 0.6:
            color = 'orange'
        else:
            color = 'red'
            
        axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle(f'Disease Predictions - {best_model_name}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Test predictions on sample images
sample_images = []
for class_name in class_names:
    class_path = os.path.join(test_dir, class_name)
    if os.path.exists(class_path):
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:2]
        sample_images.extend([os.path.join(class_path, img) for img in images])

if sample_images:
    print(f"\n🔍 Testing predictions with {best_model_name}:")
    visualize_predictions(sample_images, best_model, class_names)

# ====================================================================
# STEP 14 — Save Best Model and Create Deployment Configuration
# ====================================================================

# Determine final model (ensemble if better, otherwise best individual)
if best_ensemble_accuracy > best_accuracy:
    final_model_name = f"Ensemble_{best_ensemble_method}"
    final_accuracy = best_ensemble_accuracy
    print(f"\n🎉 Final Model Selected: {final_model_name}")
    
    # Save ensemble configuration
    ensemble_config = {
        'method': best_ensemble_method,
        'models': top_3_models,
        'accuracy': best_ensemble_accuracy,
        'individual_accuracies': {name: evaluation_results[name]['accuracy'] for name in top_3_models}
    }
    
    with open('ensemble_config.json', 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    # Save individual models for ensemble
    for model_name in top_3_models:
        trained_models[model_name].save(f'ensemble_{model_name}.h5')
        
else:
    final_model_name = best_model_name
    final_accuracy = best_accuracy
    print(f"\n🎉 Final Model Selected: {final_model_name}")

# Save the best model
best_model.save('tulsi_disease_detection_best_model.h5')
print(f"💾 Best model saved as 'tulsi_disease_detection_best_model.h5'")

# Save model configuration
model_config = {
    'best_model': best_model_name,
    'final_model': final_model_name,
    'final_accuracy': final_accuracy,
    'class_names': class_names,
    'img_height': IMG_HEIGHT,
    'img_width': IMG_WIDTH,
    'num_classes': NUM_CLASSES,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'all_model_accuracies': {name: results['accuracy'] for name, results in evaluation_results.items()}
}

with open('model_config.json', 'w') as f:
    json.dump(model_config, f, indent=2)

print("📋 Model configuration saved as 'model_config.json'")

# ====================================================================
# STEP 15 — Create Deployment-Ready Prediction System
# ====================================================================

class TulsiDiseaseDetector:
    """Complete Tulsi Disease Detection System"""
    
    def __init__(self, model_path, config_path=None):
        self.model = load_model(model_path)
        
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                self.class_names = self.config['class_names']
                self.img_height = self.config['img_height']
                self.img_width = self.config['img_width']
        else:
            self.class_names = class_names
            self.img_height = IMG_HEIGHT
            self.img_width = IMG_WIDTH
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        img = load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    
    def predict(self, image_path):
        """Make prediction and return detailed results"""
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array, verbose=0)
        
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Create detailed results
        all_predictions = {}
        for i, class_name in enumerate(self.class_names):
            all_predictions[class_name] = float(predictions[0][i])
        
        return {
            'predicted_disease': predicted_class,
            'confidence': float(confidence),
            'confidence_percentage': f"{confidence*100:.1f}%",
            'all_class_probabilities': all_predictions,
            'is_healthy': predicted_class.lower() == 'healthy',
            'needs_treatment': predicted_class.lower() in ['bacterial', 'fungal', 'pests']
        }
    
    def get_treatment_recommendation(self, prediction_result):
        """Get treatment recommendations based on prediction"""
        disease = prediction_result['predicted_disease'].lower()
        confidence = prediction_result['confidence']
        
        recommendations = {
            'healthy': "✅ Plant appears healthy! Continue regular care and monitoring.",
            'bacterial': "🦠 BACTERIAL INFECTION detected.\n" + 
                        "• Apply copper-based bactericide\n" + 
                        "• Improve air circulation\n" + 
                        "• Avoid overhead watering\n" + 
                        "• Remove infected leaves",
            'fungal': "🍄 FUNGAL INFECTION detected.\n" + 
                     "• Apply fungicide spray\n" + 
                     "• Reduce humidity around plant\n" + 
                     "• Ensure good drainage\n" + 
                     "• Remove affected parts",
            'pests': "🐛 PEST INFESTATION detected.\n" + 
                    "• Apply neem oil or insecticidal soap\n" + 
                    "• Check for insects regularly\n" + 
                    "• Use yellow sticky traps\n" + 
                    "• Quarantine if necessary"
        }
        
        base_recommendation = recommendations.get(disease, "Unknown condition detected.")
        
        if confidence < 0.6:
            base_recommendation += f"\n\n⚠️ Note: Prediction confidence is {confidence*100:.1f}%. Consider consulting an expert for confirmation."
        
        return base_recommendation

# Create detector instance
detector = TulsiDiseaseDetector('tulsi_disease_detection_best_model.h5', 'model_config.json')

# Test the complete system
print("\n🌿 TULSI DISEASE DETECTION SYSTEM - READY FOR DEPLOYMENT!")
print("="*80)

# Test on a sample image
if sample_images:
    test_image = sample_images[0]
    result = detector.predict(test_image)
    recommendation = detector.get_treatment_recommendation(result)
    
    print(f"\n📸 Test Image: {os.path.basename(test_image)}")
    print(f"🏥 Diagnosis: {result['predicted_disease']}")
    print(f"🎯 Confidence: {result['confidence_percentage']}")
    print(f"📋 Treatment Recommendation:")
    print(recommendation)

print("\n" + "="*80)
print("🎉 TULSI DISEASE DETECTION PROJECT COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nProject Summary:")
print(f"✅ Models Trained: {len(trained_models)}")
print(f"✅ Best Individual Model: {best_model_name}")
print(f"✅ Best Individual Accuracy: {best_accuracy:.2%}")
print(f"✅ Final Model: {final_model_name}")
print(f"✅ Final Accuracy: {final_accuracy:.2%}")
print(f"✅ Classes Detected: {', '.join(class_names)}")
print("✅ Deployment System: Ready")
print("✅ Treatment Recommendations: Integrated")
print("\nFiles Generated:")
print("📁 tulsi_disease_detection_best_model.h5")
print("📁 model_config.json")
if best_ensemble_accuracy > best_accuracy:
    print("📁 ensemble_config.json")
    for model_name in top_3_models:
        print(f"📁 ensemble_{model_name}.h5")
for model_name in trained_models.keys():
    print(f"📁 best_{model_name}_model.h5")