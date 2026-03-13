import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json

print("🍅 Loading Red or Ripe model...")
model = tf.keras.models.load_model('tomato_model.h5')
with open('class_names.json', 'r') as f:
    class_names = json.load(f)
print(f"✅ Model loaded! Classes: {class_names}")

def calculate_damage(predictions, class_names):
    probs = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    weights = {'Damaged': 100, 'Old': 70, 'Ripe': 15, 'Unripe': 5}
    damage = sum(probs.get(cls, 0) * w for cls, w in weights.items())
    return min(100.0, max(0.0, damage))

def predict(image):
    if image is None:
        return "Please upload an image", "", 0
    
    try:
        img = Image.fromarray(image).convert('RGB')
        img = img.resize((256, 256))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)
        
        predictions = model.predict(img_array, verbose=0)[0]
        class_idx = np.argmax(predictions)
        confidence = float(predictions[class_idx] * 100)
        predicted_class = class_names[class_idx]
        damage = calculate_damage(predictions, class_names)
        
        icons = {'Ripe': '✅', 'Unripe': '🟢', 'Old': '⚠️', 'Damaged': '❌'}
        messages = {
            'Ripe': 'Fresh & Ready to Eat!',
            'Unripe': 'Not Ready - Wait 2-3 days',
            'Old': 'Overripe - Use immediately',
            'Damaged': 'Do Not Consume'
        }
        
        result = f"{icons.get(predicted_class, '🍅')} **{predicted_class}**\n\n{messages.get(predicted_class, '')}"
        details = f"Quality: {predicted_class}\nDamage: {damage:.1f}%"
        
        return result, details, damage
    except Exception as e:
        return f"Error: {str(e)}", "", 0

# Create Gradio interface
with gr.Blocks(title="🍅 Red or Ripe") as demo:
    gr.Markdown("# 🍅 Red or Ripe: ML Tomato Quality Grading")
    gr.Markdown("**MobileNetV2 CNN | 95% Accuracy | Real-time Damage Assessment**")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="📸 Upload Tomato Image", type="numpy")
            btn = gr.Button("🔍 Analyze Tomato", variant="primary", size="lg")
        
        with gr.Column():
            output_result = gr.Markdown(label="Result")
            output_damage = gr.Slider(0, 100, label="Damage Percentage (%)", interactive=False)
            output_details = gr.Textbox(label="Detailed Analysis", lines=4)
    
    btn.click(
        fn=predict,
        inputs=input_img,
        outputs=[output_result, output_details, output_damage]
    )
    
    gr.Markdown("---")

# Launch the app
demo.launch()
