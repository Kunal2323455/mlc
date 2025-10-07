#!/usr/bin/env python3
"""
Finalize the Tulsi disease detection model and create comprehensive visualizations
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import json

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("="*70)
print("🌿 TULSI DISEASE DETECTION - MODEL FINALIZATION")
print("="*70)

# Configuration
DATASET_ROOT = "./dataset"
DATA_SPLIT_DIR = "./data_split"
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# Load the test generator
val_test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATA_SPLIT_DIR, "test"),
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = sorted(list(test_generator.class_indices.keys()))
print(f"✅ Classes: {class_names}")
print(f"✅ Test samples: {test_generator.samples}")

# Load MobileNetV2 model
print("\n📥 Loading MobileNetV2 model...")
model = load_model('best_MobileNetV2_model.h5')
print("✅ Model loaded successfully")

# Evaluate on test set
print("\n📊 Evaluating model on test set...")
test_generator.reset()
predictions = model.predict(test_generator, steps=test_generator.samples // BATCH_SIZE + 1, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes[:len(predicted_classes)]

# Calculate metrics
accuracy = accuracy_score(true_classes, predicted_classes)
precision, recall, f1, _ = precision_recall_fscore_support(true_classes, predicted_classes, average='weighted')

print(f"\n🎯 Test Results:")
print(f"  • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  • Precision: {precision:.4f}")
print(f"  • Recall:    {recall:.4f}")
print(f"  • F1-Score:  {f1:.4f}")

# Classification report
report = classification_report(true_classes, predicted_classes, target_names=class_names, output_dict=True)

print("\n📋 Per-Class Performance:")
for cls in class_names:
    print(f"  {cls:12s} - Precision: {report[cls]['precision']:.3f}, Recall: {report[cls]['recall']:.3f}, F1: {report[cls]['f1-score']:.3f}")

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Save the best model with standard name
shutil.copy('best_MobileNetV2_model.h5', 'tulsi_disease_detection_best_model.h5')
print("\n💾 Saved: tulsi_disease_detection_best_model.h5")

# Save model configuration
config = {
    'best_model': 'MobileNetV2_Transfer',
    'class_names': class_names,
    'img_height': IMG_HEIGHT,
    'img_width': IMG_WIDTH,
    'test_accuracy': float(accuracy),
    'test_precision': float(precision),
    'test_recall': float(recall),
    'test_f1_score': float(f1),
    'total_classes': len(class_names),
    'total_parameters': int(model.count_params())
}

with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("💾 Saved: model_config.json")

# Create comprehensive visualizations
print("\n📊 Creating visualizations...")

# 1. Dataset Distribution
class_counts = {}
for cls in class_names:
    cls_path = os.path.join(DATASET_ROOT, cls)
    images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    class_counts[cls] = len(images)

plt.figure(figsize=(12, 7))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = plt.bar(class_counts.keys(), class_counts.values(), color=colors, edgecolor='black', linewidth=2)
plt.title('Tulsi Leaf Disease Dataset - Class Distribution', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Number of Images', fontsize=13, fontweight='bold')
plt.xlabel('Disease Classes', fontsize=13, fontweight='bold')

total = sum(class_counts.values())
for i, (bar, count) in enumerate(zip(bars, class_counts.values())):
    percentage = (count / total) * 100
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15, 
             f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('visualization_01_class_distribution.png', dpi=300, bbox_inches='tight')
print("  ✅ visualization_01_class_distribution.png")
plt.close()

# 2. Data Split Distribution
split_counts = {'Train': [], 'Validation': [], 'Test': []}
for cls in class_names:
    for split, folder in [('Train', 'train'), ('Validation', 'validation'), ('Test', 'test')]:
        path = os.path.join(DATA_SPLIT_DIR, folder, cls)
        count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        split_counts[split].append(count)

x = np.arange(len(class_names))
width = 0.25
fig, ax = plt.subplots(figsize=(12, 7))

colors_split = ['#3498db', '#2ecc71', '#e74c3c']
for i, (split, counts) in enumerate(split_counts.items()):
    ax.bar(x + i*width, counts, width, label=split, color=colors_split[i], edgecolor='black', linewidth=1.5)

ax.set_xlabel('Disease Classes', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Images', fontsize=13, fontweight='bold')
ax.set_title('Dataset Split Distribution (Train/Validation/Test)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(class_names)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('visualization_02_data_split.png', dpi=300, bbox_inches='tight')
print("  ✅ visualization_02_data_split.png")
plt.close()

# 3. Confusion Matrix
plt.figure(figsize=(10, 8))
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

annot = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Number of Predictions'}, 
            linewidths=2, linecolor='gray',
            annot_kws={'fontsize': 11, 'fontweight': 'bold'})

plt.title(f'Confusion Matrix - MobileNetV2\nTest Accuracy: {accuracy:.2%}', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('visualization_03_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("  ✅ visualization_03_confusion_matrix.png")
plt.close()

# 4. Per-Class Performance
per_class_metrics = {
    'Precision': [report[cls]['precision'] for cls in class_names],
    'Recall': [report[cls]['recall'] for cls in class_names],
    'F1-Score': [report[cls]['f1-score'] for cls in class_names]
}

x = np.arange(len(class_names))
width = 0.25
fig, ax = plt.subplots(figsize=(12, 7))

colors_metrics = ['#3498db', '#2ecc71', '#f39c12']
for i, (metric, values) in enumerate(per_class_metrics.items()):
    bars = ax.bar(x + i*width, values, width, label=metric, color=colors_metrics[i], 
                  edgecolor='black', linewidth=1.5, alpha=0.9)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Disease Classes', fontsize=13, fontweight='bold')
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(class_names)
ax.legend(fontsize=11, loc='lower right')
ax.set_ylim([0, 1.15])
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('visualization_04_per_class_performance.png', dpi=300, bbox_inches='tight')
print("  ✅ visualization_04_per_class_performance.png")
plt.close()

# 5. Model Summary Card
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Title
fig.text(0.5, 0.95, '🌿 Tulsi Leaf Disease Detection Model', 
         ha='center', fontsize=20, fontweight='bold')

# Model Information
info_text = f"""
MODEL ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Architecture:       MobileNetV2 (Transfer Learning)
• Total Parameters:   {config['total_parameters']:,}
• Input Size:         {IMG_HEIGHT}×{IMG_WIDTH}×3 (RGB)
• Output Classes:     {len(class_names)}

PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Test Accuracy:      {accuracy:.2%}
• Test Precision:     {precision:.4f}
• Test Recall:        {recall:.4f}
• Test F1-Score:      {f1:.4f}

DATASET INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Images:       {sum(class_counts.values())}
• Training Samples:   {sum(split_counts['Train'])} ({sum(split_counts['Train'])/sum(class_counts.values())*100:.0f}%)
• Validation Samples: {sum(split_counts['Validation'])} ({sum(split_counts['Validation'])/sum(class_counts.values())*100:.0f}%)
• Test Samples:       {sum(split_counts['Test'])} ({sum(split_counts['Test'])/sum(class_counts.values())*100:.0f}%)

DISEASE CLASSES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

for cls in class_names:
    info_text += f"• {cls.capitalize():13s}  {class_counts[cls]} images   (F1: {report[cls]['f1-score']:.3f})\n"

fig.text(0.1, 0.5, info_text, ha='left', va='center', 
         fontsize=12, fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig('visualization_05_model_summary.png', dpi=300, bbox_inches='tight')
print("  ✅ visualization_05_model_summary.png")
plt.close()

# Save detailed results to CSV
results_df = pd.DataFrame({
    'Class': class_names,
    'Precision': [report[cls]['precision'] for cls in class_names],
    'Recall': [report[cls]['recall'] for cls in class_names],
    'F1-Score': [report[cls]['f1-score'] for cls in class_names],
    'Support': [int(report[cls]['support']) for cls in class_names]
})

results_df.to_csv('model_evaluation_results.csv', index=False)
print("  ✅ model_evaluation_results.csv")

print("\n" + "="*70)
print("🎉 MODEL FINALIZATION COMPLETED!")
print("="*70)

print("\n📁 Generated Files:")
print("  ✅ tulsi_disease_detection_best_model.h5")
print("  ✅ model_config.json")
print("  ✅ model_evaluation_results.csv")
print("  ✅ visualization_01_class_distribution.png")
print("  ✅ visualization_02_data_split.png")
print("  ✅ visualization_03_confusion_matrix.png")
print("  ✅ visualization_04_per_class_performance.png")
print("  ✅ visualization_05_model_summary.png")

print("\n🎯 Model Performance Summary:")
print(f"  • Architecture: MobileNetV2 (Transfer Learning)")
print(f"  • Test Accuracy: {accuracy:.2%}")
print(f"  • Weighted F1-Score: {f1:.4f}")
print(f"  • Total Parameters: {config['total_parameters']:,}")

print("\n🚀 Model is ready for deployment!")
print("="*70)