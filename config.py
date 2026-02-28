"""
Configuration file for Tomato Quality & Shelf Life Detection System
"""

import os

# ============================================
# PROJECT PATHS
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================
# MODEL HYPERPARAMETERS
# ============================================
IMG_SIZE = 224          # MobileNetV2 requires 224x224
BATCH_SIZE = 32         # Images per batch
EPOCHS = 20             # Training iterations
LEARNING_RATE = 0.001   # Adam optimizer learning rate
NUM_CLASSES = 4         # Damaged, Old, Ripe, Unripe

# ============================================
# MODEL PATHS
# ============================================
MODEL_PATH = os.path.join(MODELS_DIR, 'tomato_model.h5')
MODEL_KERAS_PATH = os.path.join(MODELS_DIR, 'tomato_model.keras')
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, 'class_names.json')

# ============================================
# RESULTS PATHS
# ============================================
TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, 'training_history.png')
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
CLASSIFICATION_REPORT_PATH = os.path.join(RESULTS_DIR, 'classification_report.txt')

# ============================================
# SHELF LIFE PREDICTION MAPPING
# ============================================
SHELF_LIFE_MAP = {
    'Ripe': {
        'min_days': 3,
        'max_days': 5,
        'storage_tip': 'Store at room temperature. Refrigerate if not using within 2 days.',
        'message': 'Perfect for eating! Consume within 3-5 days.'
    },
    'Unripe': {
        'min_days': 7,
        'max_days': 10,
        'storage_tip': 'Leave at room temperature to ripen (2-3 days). Once ripe, consume within 3-5 days.',
        'message': 'Not ready yet. Will ripen in 2-3 days. Total shelf life: 7-10 days.'
    },
    'Old': {
        'min_days': 0,
        'max_days': 1,
        'storage_tip': 'Use immediately in cooked dishes. Do not store.',
        'message': 'Overripe. Use immediately or discard within 24 hours.'
    },
    'Damaged': {
        'min_days': 0,
        'max_days': 0,
        'storage_tip': 'Do not consume. Discard immediately.',
        'message': 'Not safe to eat. Discard immediately.'
    }
}

# ============================================
# FLASK CONFIGURATION
# ============================================
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# ============================================
# TRAINING CONFIGURATION
# ============================================
USE_AUGMENTATION = True
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3

# ============================================
# PRINT CONFIGURATION
# ============================================
print("✅ Configuration loaded successfully!")
print(f"📂 Base directory: {BASE_DIR}")
print(f"📂 Training data: {TRAIN_DIR}")
print(f"📂 Validation data: {VAL_DIR}")
print(f"📂 Models directory: {MODELS_DIR}")
print(f"📂 Results directory: {RESULTS_DIR}")