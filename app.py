import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json

print("🍅 Loading model...")
model = tf.keras.models.load_model('tomato_model.h5')
with open('class_names.json', 'r') as f:
    class_names = json.load(f)
print("✅ Model loaded!")

def calculate_damage(predictions, class_names):
    probs = {class_names[i]: predictions[i] for i in range(len(class_names))}
    weights = {'Damaged': 100, 'Old': 70, 'Ripe': 15, 'Unripe': 5}
    damage = sum(probs.get(cls, 0) * w for cls, w in weights.items())
    return min(100, max(0, damage))

def predict(image):
    img = Image.fromarray(image).convert('RGB')
    img = img.resize((224, 224))
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
    details = f"**Quality:** {predicted_class}\n**Damage:** {damage:.1f}%\n**Confidence:** {confidence:.1f}%"
    
    return result, details, damage

with gr.Blocks(title="🍅 Red or Ripe") as demo:
    gr.Markdown("# 🍅 Red or Ripe: ML Tomato Quality Grading")
    gr.Markdown("**MobileNetV2 CNN | 95% Accuracy | Damage Assessment**")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="📸 Upload Tomato", type="numpy")
            btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
        with gr.Column():
            output_result = gr.Markdown()
            output_damage = gr.Slider(0, 100, label="Damage %", interactive=False)
            output_details = gr.Textbox(label="Details", lines=4)
    
    btn.click(predict, input_img, [output_result, output_details, output_damage])
    gr.Markdown("---\n**Team:** Nishigandha, Anika, Vaishnavi, Ashwini | **VIT Vellore**")

demo.launch()