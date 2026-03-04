"""
Red or Ripe - Flask App for Render.com
"""

from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os

app = Flask(__name__)

# Load model
print("Loading model...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'tomato_model.keras')
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), 'class_names.json')

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded!")

with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = json.load(f)
print(f"✅ Classes: {class_names}")

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>🍅 Red or Ripe</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: Arial, sans-serif; 
            text-align: center; 
            padding: 50px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { 
            max-width: 700px; 
            margin: 0 auto; 
            background: white; 
            padding: 40px; 
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 { color: #333; font-size: 2.5em; margin-bottom: 10px; }
        .subtitle { color: #666; margin-bottom: 30px; }
        input[type="file"] { margin: 20px; }
        .btn { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 15px 40px; 
            border: none; 
            border-radius: 50px; 
            cursor: pointer; 
            font-size: 18px;
            font-weight: 600;
            margin-top: 15px;
        }
        .btn:hover { opacity: 0.9; }
        .result { 
            margin-top: 30px; 
            padding: 30px; 
            border-radius: 15px; 
            font-size: 18px;
        }
        .good { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; }
        .warning { background: linear-gradient(135deg, #FFD93D 0%, #F6C745 100%); color: #333; }
        .bad { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); color: white; }
        img { max-width: 400px; margin: 20px; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.2); }
        .damage-bar { 
            background: rgba(255,255,255,0.3); 
            height: 30px; 
            border-radius: 10px; 
            margin: 15px 0; 
            overflow: hidden;
        }
        .damage-fill { 
            height: 100%; 
            background: rgba(255,255,255,0.8); 
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            transition: width 1s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🍅 Red or Ripe</h1>
        <p class="subtitle">ML-Based Tomato Quality Grading</p>
        <input type="file" id="imageInput" accept="image/*">
        <br>
        <button class="btn" onclick="predictImage()">🔍 Analyze Tomato</button>
        <img id="preview" style="display:none;">
        <div id="result"></div>
        <p style="margin-top: 30px; color: #999;">95% Accuracy | MobileNetV2 CNN</p>
    </div>
    <script>
        const imageInput = document.getElementById('imageInput');
        const preview = document.getElementById('preview');
        
        imageInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });
        
        function predictImage() {
            const file = imageInput.files[0];
            if (!file) {
                alert('⚠️ Please select an image first!');
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
                    const damage = data.damage_percentage;
                    
                    let resultClass = 'good';
                    let icon = '✅';
                    let message = 'Fresh & Ready to Eat';
                    
                    if (className === 'Damaged') {
                        resultClass = 'bad';
                        icon = '❌';
                        message = 'Do Not Consume';
                    } else if (className === 'Old') {
                        resultClass = 'warning';
                        icon = '⚠️';
                        message = 'Overripe - Use Immediately';
                    } else if (className === 'Unripe') {
                        resultClass = 'warning';
                        icon = '🟢';
                        message = 'Not Ready - Wait 2-3 days';
                    }
                    
                    resultDiv.className = 'result ' + resultClass;
                    resultDiv.innerHTML = \`
                        <h2>\${icon} \${className}</h2>
                        <p>\${message}</p>
                        <div class="damage-bar">
                            <div class="damage-fill" style="width: \${damage}%">
                                \${damage}% Damage
                            </div>
                        </div>
                    \`;
                } else {
                    resultDiv.innerHTML = '<div class="result bad"><p>❌ ' + data.error + '</p></div>';
                }
            })
            .catch(err => {
                document.getElementById('result').innerHTML = '<div class="result bad"><p>❌ Error: ' + err + '</p></div>';
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
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
