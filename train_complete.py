"""
COMPLETE Training Script - Red or Ripe ML Project
This will train the model properly
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from datetime import datetime
import config

print("""
╔══════════════════════════════════════════════════════════════╗
║  🍅 RED OR RIPE: ML-BASED TOMATO QUALITY GRADING            ║
║                                                              ║
║  Starting Complete Training Process                          ║
╚══════════════════════════════════════════════════════════════╝
""")

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"🎮 GPU Available: {len(gpus)} GPU(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("💻 Running on CPU (slower but will work)")

# Data Augmentation
print("\n🔄 Setting up Data Augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load Dataset
print("\n📂 Loading Dataset...")
train_generator = train_datagen.flow_from_directory(
    config.TRAIN_DIR,
    target_size=(config.IMG_SIZE, config.IMG_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    config.VAL_DIR,
    target_size=(config.IMG_SIZE, config.IMG_SIZE),
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Get class names
class_names = list(train_generator.class_indices.keys())
print(f"\n📊 Dataset Info:")
print(f"   Training samples: {train_generator.samples:,}")
print(f"   Validation samples: {val_generator.samples:,}")
print(f"   Classes: {class_names}")
print(f"   Class indices: {train_generator.class_indices}")

# Build Model
print("\n🏗️ Building MobileNetV2 Model...")
base_model = MobileNetV2(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(config.NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print(f"   Total parameters: {model.count_params():,}")

# Compile
print("\n⚙️ Compiling model...")
model.compile(
    optimizer=Adam(learning_rate=config.LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        config.MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train
print("\n🚀 Starting Training...")
print("="*70)
start_time = datetime.now()

history = model.fit(
    train_generator,
    epochs=config.EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

end_time = datetime.now()
duration = (end_time - start_time).total_seconds() / 60

print("\n" + "="*70)
print(f"✅ Training completed in {duration:.2f} minutes")
print("="*70)

# Save Model and Class Names
print("\n💾 Saving model and metadata...")
model.save(config.MODEL_PATH)
model.save(config.MODEL_KERAS_PATH)

with open(config.CLASS_NAMES_PATH, 'w') as f:
    json.dump(class_names, f, indent=2)

print(f"   ✅ Model saved: {config.MODEL_PATH}")
print(f"   ✅ Classes saved: {config.CLASS_NAMES_PATH}")
print(f"   ✅ Class order: {class_names}")

# Evaluate
print("\n📈 Evaluating on validation set...")
val_generator.reset()
predictions = model.predict(val_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes

# Classification Report
print("\n" + "="*70)
print("📋 CLASSIFICATION REPORT")
print("="*70)
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

with open(config.CLASSIFICATION_REPORT_PATH, 'w') as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\n🔢 Confusion Matrix:")
print(cm)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(config.TRAINING_HISTORY_PATH, dpi=150, bbox_inches='tight')
print(f"   ✅ Training graphs saved: {config.TRAINING_HISTORY_PATH}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=150, bbox_inches='tight')
print(f"   ✅ Confusion matrix saved: {config.CONFUSION_MATRIX_PATH}")

# Final Results
print("\n" + "="*70)
print("🎯 FINAL RESULTS")
print("="*70)
print(f"Training Accuracy:     {history.history['accuracy'][-1]*100:.2f}%")
print(f"Validation Accuracy:   {history.history['val_accuracy'][-1]*100:.2f}%")
print(f"Training Loss:         {history.history['loss'][-1]:.4f}")
print(f"Validation Loss:       {history.history['val_loss'][-1]:.4f}")
print("="*70)

print("\n✅ TRAINING COMPLETE!")
print("🚀 Next: Run 'python app.py' to test the web application")
