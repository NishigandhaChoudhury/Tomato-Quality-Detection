"""
RED OR RIPE - ADVANCED VERSION WITH TEST-TIME AUGMENTATION
===========================================================
This version uses Test-Time Augmentation (TTA) to improve accuracy
on Google images and other out-of-distribution data.

TTA makes multiple predictions with slight variations and averages them,
significantly improving robustness without retraining!
"""

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
import json
import cv2

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'tomato_model.h5'
CLASS_NAMES_PATH = 'class_names.json'

# Test-Time Augmentation settings
USE_TTA = True  # Enable TTA for better accuracy
TTA_ITERATIONS = 5  # Number of augmented predictions to average

# Confidence thresholds
MIN_CONFIDENCE = 0.45  # Lowered slightly because TTA increases confidence
TOMATO_DETECTION_THRESHOLD = 0.35

# Damage weights
DAMAGE_WEIGHTS = {
    'Damaged': 100,
    'Old': 70,
    'Ripe': 15,
    'Unripe': 5
}

# ============================================================================
# MODEL LOADING
# ============================================================================

print("🍅 Loading Red or Ripe model (TTA-Enhanced)...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

try:
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"✅ Classes loaded: {class_names}")
    print(f"✅ TTA: {'Enabled' if USE_TTA else 'Disabled'} ({TTA_ITERATIONS} iterations)")
except Exception as e:
    print(f"❌ Error loading class names: {e}")
    raise

# ============================================================================
# TEST-TIME AUGMENTATION FUNCTIONS
# ============================================================================

def apply_tta_augmentation(pil_image, augmentation_type='random'):
    """
    Apply Test-Time Augmentation to PIL image.
    Returns slightly modified version to test model robustness.
    """
    if augmentation_type == 'original':
        return pil_image
    
    aug_image = pil_image.copy()
    
    # Random brightness adjustment (±10%)
    if np.random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(aug_image)
        factor = np.random.uniform(0.9, 1.1)
        aug_image = enhancer.enhance(factor)
    
    # Random contrast adjustment (±10%)
    if np.random.random() > 0.5:
        enhancer = ImageEnhance.Contrast(aug_image)
        factor = np.random.uniform(0.9, 1.1)
        aug_image = enhancer.enhance(factor)
    
    # Random color adjustment (±5%)
    if np.random.random() > 0.5:
        enhancer = ImageEnhance.Color(aug_image)
        factor = np.random.uniform(0.95, 1.05)
        aug_image = enhancer.enhance(factor)
    
    # Random sharpness adjustment (±10%)
    if np.random.random() > 0.5:
        enhancer = ImageEnhance.Sharpness(aug_image)
        factor = np.random.uniform(0.9, 1.1)
        aug_image = enhancer.enhance(factor)
    
    return aug_image

def preprocess_image(image, target_size=224, apply_augmentation=False):
    """
    Preprocess image with optional TTA augmentation.
    """
    try:
        # Convert to PIL
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            pil_img = Image.fromarray(image)
        else:
            pil_img = image
        
        # Ensure RGB
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # Apply TTA augmentation if requested
        if apply_augmentation:
            pil_img = apply_tta_augmentation(pil_img)
        
        # Resize
        pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
        
        # Convert to numpy and normalize
        img_array = np.array(pil_img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        raise

def predict_with_tta(image, num_iterations=5):
    """
    Make predictions using Test-Time Augmentation.
    
    TTA Process:
    1. Make one prediction on original image
    2. Make multiple predictions on slightly augmented versions
    3. Average all predictions
    4. Return averaged result
    
    This significantly improves robustness and accuracy!
    """
    all_predictions = []
    
    # First prediction: original image (no augmentation)
    img_array_original = preprocess_image(image, apply_augmentation=False)
    pred_original = model.predict(img_array_original, verbose=0)[0]
    all_predictions.append(pred_original)
    
    # Additional predictions: augmented versions
    for i in range(num_iterations - 1):
        img_array_aug = preprocess_image(image, apply_augmentation=True)
        pred_aug = model.predict(img_array_aug, verbose=0)[0]
        all_predictions.append(pred_aug)
    
    # Average all predictions
    avg_predictions = np.mean(all_predictions, axis=0)
    
    # Also calculate standard deviation for confidence assessment
    std_predictions = np.std(all_predictions, axis=0)
    
    return avg_predictions, std_predictions

def is_likely_tomato(image):
    """
    Simple heuristic to check if image is likely a tomato.
    """
    try:
        if image.shape[2] == 3:
            hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
        else:
            return False
        
        # Color ranges
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        lower_orange = np.array([10, 50, 50])
        upper_orange = np.array([25, 255, 255])
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        tomato_mask = mask_red1 + mask_red2 + mask_orange + mask_green
        tomato_percentage = np.sum(tomato_mask > 0) / (image.shape[0] * image.shape[1])
        brightness = np.mean(image)
        
        has_tomato_colors = tomato_percentage > 0.15
        proper_brightness = 30 < brightness < 240
        
        return has_tomato_colors and proper_brightness
        
    except Exception as e:
        print(f"⚠️ Tomato detection error: {e}")
        return True

def calculate_damage(predictions, class_names):
    """Calculate weighted damage percentage."""
    probs = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    damage = sum(probs.get(cls, 0) * DAMAGE_WEIGHTS.get(cls, 0) for cls in class_names)
    return min(100.0, max(0.0, damage))

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict(image):
    """
    Main prediction function with TTA and robust error handling.
    """
    if image is None:
        return "⚠️ **Please upload an image**", "No image provided", 0
    
    try:
        # Validate input
        if len(image.shape) not in [2, 3]:
            return "❌ **Invalid Image Format**", "Image must be 2D or 3D array", 0
        
        # Check if tomato
        if not is_likely_tomato(image):
            return (
                "⚠️ **This doesn't appear to be a tomato**\n\n"
                "Please upload an image of a tomato.\n\n"
                "**Tips:**\n"
                "• Use clear, well-lit photos\n"
                "• Show the entire tomato\n"
                "• Avoid heavy filters",
                "Non-tomato object detected",
                0
            )
        
        # Make predictions (with or without TTA)
        if USE_TTA:
            predictions, pred_std = predict_with_tta(image, num_iterations=TTA_ITERATIONS)
            using_tta_text = f"✓ Enhanced with TTA ({TTA_ITERATIONS} predictions averaged)"
        else:
            img_array = preprocess_image(image, apply_augmentation=False)
            predictions = model.predict(img_array, verbose=0)[0]
            pred_std = np.zeros_like(predictions)
            using_tta_text = ""
        
        # Get predicted class
        class_idx = np.argmax(predictions)
        confidence = float(predictions[class_idx] * 100)
        predicted_class = class_names[class_idx]
        
        # Calculate prediction consistency (low std = more consistent)
        prediction_consistency = 100 - (np.mean(pred_std) * 100)
        
        # Check confidence
        if confidence < MIN_CONFIDENCE * 100:
            return (
                f"⚠️ **Low Confidence Detection**\n\n"
                f"Predicted: {predicted_class}\n"
                f"Confidence: {confidence:.1f}%\n\n"
                f"Please try:\n"
                f"• Better lighting\n"
                f"• Clearer photo\n"
                f"• Different angle",
                f"Low confidence: {confidence:.1f}%\nPredicted: {predicted_class}",
                50
            )
        
        # Calculate damage
        damage = calculate_damage(predictions, class_names)
        
        # Icons and messages
        icons = {'Ripe': '✅', 'Unripe': '🟢', 'Old': '⚠️', 'Damaged': '❌'}
        messages = {
            'Ripe': 'Fresh & Ready to Eat!',
            'Unripe': 'Not Ready - Wait 2-3 days',
            'Old': 'Overripe - Use immediately',
            'Damaged': 'Do Not Consume - Spoiled'
        }
        recommendations = {
            'Ripe': 'Perfect for consumption. Store in cool, dry place. Use within 3-5 days.',
            'Unripe': 'Let ripen at room temperature. Do not refrigerate. Ready in 2-3 days.',
            'Old': 'Use immediately in cooking. Suitable for sauces, soups. Do not store.',
            'Damaged': 'Discard immediately. May contain harmful bacteria. Do not consume.'
        }
        
        # Build result
        result = (
            f"{icons.get(predicted_class, '🍅')} **{predicted_class.upper()}**\n\n"
            f"{messages.get(predicted_class, '')}\n\n"
            f"**Confidence:** {confidence:.1f}%\n"
            f"{using_tta_text}"
        )
        
        # Build details
        details = (
            f"**Quality Assessment:**\n"
            f"Category: {predicted_class}\n"
            f"Confidence: {confidence:.1f}%\n"
            f"Prediction Consistency: {prediction_consistency:.1f}%\n"
            f"Damage Score: {damage:.1f}%\n\n"
            f"**Recommendation:**\n"
            f"{recommendations.get(predicted_class, '')}\n\n"
            f"**Class Probabilities:**\n"
        )
        
        for i, cls in enumerate(class_names):
            prob = predictions[i] * 100
            details += f"• {cls}: {prob:.1f}%\n"
        
        if USE_TTA:
            details += f"\n**TTA Info:**\n"
            details += f"Averaged {TTA_ITERATIONS} predictions for robustness\n"
            details += f"Prediction consistency: {prediction_consistency:.1f}%"
        
        return result, details, damage
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return (
            f"❌ **Error Processing Image**\n\n{str(e)}",
            f"Error: {str(e)}",
            0
        )

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

custom_css = """
.gradio-container {max-width: 1200px; margin: auto;}
"""

with gr.Blocks(title="🍅 Red or Ripe - TTA Enhanced", css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # 🍅 Red or Ripe: Professional Tomato Quality Assessment
        
        ### AI-Powered Quality Grading with Test-Time Augmentation
        **MobileNetV2 + TTA | 95.44% Base Accuracy | Enhanced Robustness**
        
        Upload a tomato image for professional quality assessment.
        System uses **Test-Time Augmentation** for improved accuracy on diverse images!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Upload Image")
            input_img = gr.Image(label="Tomato Image", type="numpy", height=400)
            btn = gr.Button("🔍 Analyze Quality (with TTA)", variant="primary", size="lg")
            
            gr.Markdown(
                """
                **What is TTA?**
                
                Test-Time Augmentation makes multiple predictions with slight 
                variations (brightness, contrast, etc.) and averages them.
                
                **Result:** More robust and accurate predictions, especially 
                on Google images and photos taken in different conditions!
                
                **Tips for best results:**
                - Clear, well-lit photos
                - Show entire tomato
                - Avoid shadows
                """
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Results")
            output_result = gr.Markdown(value="Upload an image to see results...")
            output_damage = gr.Slider(
                minimum=0,
                maximum=100,
                label="Damage Level (%)",
                interactive=False,
                info="0% = Perfect, 100% = Completely Damaged"
            )
            output_details = gr.Textbox(label="Detailed Analysis", lines=14, show_copy_button=True)
    
    btn.click(
        fn=predict,
        inputs=input_img,
        outputs=[output_result, output_details, output_damage]
    )
    
    gr.Markdown("---")
    gr.Markdown(
        """
        ### Technical Details
        
        **Architecture:** MobileNetV2 with Transfer Learning  
        **Training Accuracy:** 95.44%  
        **Enhancement:** Test-Time Augmentation (5 predictions averaged)  
        **Inference Time:** ~200ms with TTA (~40ms without)  
        
        **Why TTA Helps:**
        - Averages out prediction noise
        - More robust to lighting variations
        - Better generalization to Google images
        - Higher confidence in correct predictions
        
        **Team:** Nishigandha, Anika, Vaishnavi, Ashwini  
        **Mentor:** Dr. Gokul Rajan V  
        **Institution:** Alliance University
        """
    )

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 Red or Ripe - TTA Enhanced Version")
    print("="*70)
    print(f"✅ Model loaded")
    print(f"✅ Classes: {class_names}")
    print(f"✅ TTA: Enabled ({TTA_ITERATIONS} iterations)")
    print(f"✅ Tomato Detection: Enabled")
    print(f"✅ Confidence Threshold: {MIN_CONFIDENCE * 100}%")
    print("="*70)
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
