"""
Quick Test App - Red or Ripe
Tests the trained model
"""

from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import config

app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model(config.MODEL_PATH)
print("✅ Model loaded!")

with open(config.CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)
print(f"✅ Classes: {class_names}")

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>🍅 Red or Ripe - Test</title>
    <style>
        body { font-family: Arial; text-align: center; padding: 50px; background: #f0f0f0; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 40px; border-radius: 20px; }
        input[type="file"] { margin: 20px; }
        .btn { background: #667eea; color: white; padding: 15px 30px; border: none; border-radius: 10px; cursor: pointer; font-size: 16px; }
        .result { margin-top: 30px; padding: 20px; border-radius: 10px; font-size: 18px; }
        .good { background: #d4edda; color: #155724; }
        .bad { background: #f8d7da; color: #721c24; }
        img { max-width: 300px; margin: 20px; border-radius: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🍅 Red or Ripe</h1>
        <p>ML-Based Tomato Quality Grading</p>
        <input type="file" id="imageInput" accept="image/*">
        <button class="btn" onclick="predictImage()">Analyze Tomato</button>
        <img id="preview" style="display:none;">
        <div id="result"></div>
    </div>
    <script>
        const imageInput = document.getElementById('imageInput');
        const preview = document.getElementById('preview');
        
        imageInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        });
        
        function predictImage() {
            const file = imageInput.files[0];
            if (!file) {
                alert('Please select an image first!');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', file);
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(r => r.json())
            .then(data => {
                const resultDiv = document.getElementById('result');
                if (data.success) {
                    const className = data.quality_class;
                    const conf = data.confidence;
                    const isGood = className === 'Ripe' || className === 'Unripe';
                    
                    resultDiv.className = 'result ' + (isGood ? 'good' : 'bad');
                    resultDiv.innerHTML = `
                        <h2>${className}</h2>
                        <p>Damage: ${data.damage_percentage}%</p>
                    `;
                } else {
                    resultDiv.innerHTML = '<p>Error: ' + data.error + '</p>';
                }
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, 0)
        
        predictions = model.predict(img_array, verbose=0)[0]
        class_idx = np.argmax(predictions)
        confidence = float(predictions[class_idx] * 100)
        predicted_class = class_names[class_idx]
        
        # Calculate damage
        damage_weights = {'Damaged': 100, 'Old': 70, 'Ripe': 15, 'Unripe': 5}
        damage = float(sum(predictions[i] * damage_weights.get(class_names[i], 0) 
                   for i in range(len(class_names))))
        
        return jsonify({
            'success': True,
            'quality_class': predicted_class,
            'confidence': round(confidence, 2),
            'damage_percentage': round(damage, 1)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
