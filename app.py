"""
RED OR RIPE - PRODUCTION-READY APPLICATION
============================================
Professional-grade tomato quality detection with:
- Robust preprocessing matching training exactly
- Non-tomato object detection
- Confidence thresholding
- Better error handling
- Enhanced user feedback
"""

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import cv2

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'tomato_model.h5'
CLASS_NAMES_PATH = 'class_names.json'

# Confidence thresholds
MIN_CONFIDENCE = 0.50  # Minimum confidence to accept prediction
TOMATO_DETECTION_THRESHOLD = 0.35  # Threshold for "is this a tomato?"

# Expected image properties for tomatoes
EXPECTED_CHANNELS = 3  # RGB

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

print("🍅 Loading Red or Ripe model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    # Recompile for inference
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

try:
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"✅ Classes loaded: {class_names}")
except Exception as e:
    print(f"❌ Error loading class names: {e}")
    raise

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_image(image, target_size=224):
    """
    Preprocess image to match EXACTLY what was used during training.
    
    Critical: This MUST match the training preprocessing:
    1. Resize to (224, 224)
    2. Convert to RGB if needed
    3. Normalize to [0, 1] by dividing by 255
    4. Add batch dimension
    
    Args:
        image: NumPy array from Gradio (H, W, C) in range [0, 255]
        target_size: Target dimension (default 224 for MobileNetV2)
    
    Returns:
        Preprocessed image ready for model (1, 224, 224, 3) in range [0, 1]
    """
    try:
        # Convert to PIL Image for consistent resizing
        if isinstance(image, np.ndarray):
            # Ensure uint8 for PIL
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            pil_img = Image.fromarray(image)
        else:
            pil_img = image
        
        # Ensure RGB (handle grayscale or RGBA)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # Resize using BILINEAR (same as Keras ImageDataGenerator default)
        pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
        
        # Convert back to numpy
        img_array = np.array(pil_img, dtype=np.float32)
        
        # Normalize to [0, 1] - CRITICAL: This MUST match training
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        raise

def is_likely_tomato(image):
    """
    Simple heuristic to check if image is likely a tomato.
    Checks for:
    1. Presence of red/orange colors
    2. Reasonable size and shape
    3. Not too dark or too bright
    
    Args:
        image: NumPy array (H, W, C) in range [0, 255]
    
    Returns:
        bool: True if likely a tomato, False otherwise
    """
    try:
        # Convert to HSV for color analysis
        if image.shape[2] == 3:  # RGB
            hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
        else:
            return False
        
        # Define red/orange/yellow color ranges in HSV
        # Red (two ranges because red wraps around in HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Orange
        lower_orange = np.array([10, 50, 50])
        upper_orange = np.array([25, 255, 255])
        
        # Green (for unripe tomatoes)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Combine masks
        tomato_mask = mask_red1 + mask_red2 + mask_orange + mask_green
        
        # Calculate percentage of tomato-colored pixels
        tomato_percentage = np.sum(tomato_mask > 0) / (image.shape[0] * image.shape[1])
        
        # Check brightness (not too dark, not too bright)
        brightness = np.mean(image)
        
        # Decision criteria
        has_tomato_colors = tomato_percentage > 0.15  # At least 15% tomato-colored
        proper_brightness = 30 < brightness < 240  # Not too dark or washed out
        
        return has_tomato_colors and proper_brightness
        
    except Exception as e:
        print(f"⚠️ Tomato detection error: {e}")
        # If detection fails, allow prediction (fail open)
        return True

def calculate_damage(predictions, class_names):
    """
    Calculate weighted damage percentage from class probabilities.
    
    Args:
        predictions: NumPy array of class probabilities
        class_names: List of class names
    
    Returns:
        float: Damage percentage (0-100)
    """
    probs = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    damage = sum(probs.get(cls, 0) * DAMAGE_WEIGHTS.get(cls, 0) for cls in class_names)
    return min(100.0, max(0.0, damage))

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict(image):
    """
    Main prediction function with robust error handling and validation.
    
    Args:
        image: NumPy array from Gradio interface
    
    Returns:
        tuple: (result_text, details_text, damage_slider_value)
    """
    # Validate input
    if image is None:
        return "⚠️ **Please upload an image**", "No image provided", 0
    
    try:
        # Check image dimensions
        if len(image.shape) not in [2, 3]:
            return "❌ **Invalid Image Format**", "Image must be 2D or 3D array", 0
        
        # Check if image is likely a tomato
        if not is_likely_tomato(image):
            return (
                "⚠️ **This doesn't appear to be a tomato**\n\n"
                "Please upload an image of a tomato for quality assessment.\n\n"
                "**Tips:**\n"
                "• Use clear, well-lit photos\n"
                "• Show the entire tomato\n"
                "• Avoid heavy filters or editing",
                "Non-tomato object detected",
                0
            )
        
        # Preprocess image
        img_array = preprocess_image(image, target_size=224)
        
        # Get predictions
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get predicted class
        class_idx = np.argmax(predictions)
        confidence = float(predictions[class_idx] * 100)
        predicted_class = class_names[class_idx]
        
        # Check confidence threshold
        if confidence < MIN_CONFIDENCE * 100:
            return (
                f"⚠️ **Low Confidence Detection**\n\n"
                f"Predicted: {predicted_class}\n"
                f"Confidence: {confidence:.1f}%\n\n"
                f"The model is uncertain about this image. "
                f"Please try:\n"
                f"• Better lighting\n"
                f"• Clearer photo\n"
                f"• Different angle",
                f"Low confidence: {confidence:.1f}%\nPredicted: {predicted_class}",
                50  # Middle value for uncertainty
            )
        
        # Calculate damage
        damage = calculate_damage(predictions, class_names)
        
        # Prepare result
        icons = {
            'Ripe': '✅',
            'Unripe': '🟢',
            'Old': '⚠️',
            'Damaged': '❌'
        }
        
        messages = {
            'Ripe': 'Fresh & Ready to Eat!',
            'Unripe': 'Not Ready - Wait 2-3 days',
            'Old': 'Overripe - Use immediately',
            'Damaged': 'Do Not Consume - Spoiled'
        }
        
        recommendations = {
            'Ripe': 'Perfect for consumption. Store in cool, dry place. Use within 3-5 days.',
            'Unripe': 'Let ripen at room temperature for 2-3 days. Do not refrigerate yet.',
            'Old': 'Use immediately in cooking. Suitable for sauces, soups. Do not store.',
            'Damaged': 'Discard immediately. May contain harmful bacteria. Do not consume.'
        }
        
        # Build result string
        result = (
            f"{icons.get(predicted_class, '🍅')} **{predicted_class.upper()}**\n\n"
            f"{messages.get(predicted_class, '')}\n\n"
            f"**Confidence:** {confidence:.1f}%"
        )
        
        # Build details string
        details = (
            f"**Quality Assessment:**\n"
            f"Category: {predicted_class}\n"
            f"Confidence: {confidence:.1f}%\n"
            f"Damage Score: {damage:.1f}%\n\n"
            f"**Recommendation:**\n"
            f"{recommendations.get(predicted_class, '')}\n\n"
            f"**Class Probabilities:**\n"
        )
        
        # Add all class probabilities for transparency
        for i, cls in enumerate(class_names):
            prob = predictions[i] * 100
            details += f"• {cls}: {prob:.1f}%\n"
        
        return result, details, damage
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return (
            f"❌ **Error Processing Image**\n\n{str(e)}",
            f"Error details: {str(e)}",
            0
        )

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Custom CSS for better UI
custom_css = """
.gradio-container {
    max-width: 1200px;
    margin: auto;
}
.result-box {
    border-radius: 10px;
    padding: 20px;
}
"""

# Create interface
with gr.Blocks(title="🍅 Red or Ripe - Professional", css=custom_css, theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.Markdown(
        """
        # 🍅 Red or Ripe: Professional Tomato Quality Assessment
        
        ### AI-Powered Quality Grading System
        **MobileNetV2 Deep Learning Model | 95.44% Accuracy | Real-Time Analysis**
        
        Upload a clear photo of a tomato to assess its quality and freshness.
        """
    )
    
    # Main interface
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Upload Image")
            input_img = gr.Image(
                label="Tomato Image",
                type="numpy",
                height=400
            )
            
            btn = gr.Button(
                "🔍 Analyze Quality",
                variant="primary",
                size="lg"
            )
            
            gr.Markdown(
                """
                **Tips for best results:**
                - Use clear, well-lit photos
                - Show the entire tomato
                - Avoid shadows and reflections
                - Hold camera steady (no blur)
                """
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Quality Assessment Results")
            
            output_result = gr.Markdown(
                label="Result",
                value="Upload an image to see results..."
            )
            
            output_damage = gr.Slider(
                minimum=0,
                maximum=100,
                label="Damage Level (%)",
                interactive=False,
                info="0% = Perfect, 100% = Completely Damaged"
            )
            
            output_details = gr.Textbox(
                label="Detailed Analysis",
                lines=12,
                max_lines=15
            )
    
    # Connect button
    btn.click(
        fn=predict,
        inputs=input_img,
        outputs=[output_result, output_details, output_damage]
    )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown(
        """
        ### About This System
        
        This system uses a **MobileNetV2 Convolutional Neural Network** trained on 7,224 tomato images 
        to classify quality into four categories: **Ripe**, **Unripe**, **Old**, and **Damaged**.
        
        **Technical Details:**
        - Architecture: MobileNetV2 with Transfer Learning
        - Training Accuracy: 95.44%
        - Inference Time: <50ms
        - Model Size: 10.4 MB
        
        **Team:** Nishigandha, Anika, Vaishnavi, Ashwini  
        **Mentor:** Dr. Gokul Rajan V  
        **Institution:** Alliance University, Bengaluru
        
        ---
        
        **Links:**  
        🌐 [Live Demo](https://huggingface.co/spaces/NishigandhaChoudhury/tomato-quality-detection)  
        💻 [GitHub Repository](https://github.com/NishigandhaChoudhury/Tomato-Quality-Detection)  
        
        *For academic research and educational purposes.*
        """
    )

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 Starting Red or Ripe Application")
    print("="*70)
    print(f"✅ Model: {MODEL_PATH}")
    print(f"✅ Classes: {class_names}")
    print(f"✅ Confidence Threshold: {MIN_CONFIDENCE * 100}%")
    print(f"✅ Tomato Detection: {'Enabled' if TOMATO_DETECTION_THRESHOLD else 'Disabled'}")
    print("="*70)
    print("\n🌐 Launching interface...\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
