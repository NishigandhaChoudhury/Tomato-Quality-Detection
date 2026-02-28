# 🍅 Red or Ripe: ML-Based Tomato Quality Grading

ML system for automated tomato quality classification with 95% accuracy.

## Features
- 4-class classification (Ripe, Unripe, Old, Damaged)
- MobileNetV2 CNN with Transfer Learning
- Damage percentage assessment
- Web-based interface

## Tech Stack
- TensorFlow/Keras, Flask, Python
- MobileNetV2 architecture
- 7,224 images dataset

## Performance
- Accuracy: 95.03%
- Model: 2.6M parameters

## Team
Nishigandha Choudhury, Anika Ramya Shetty, Vaishnavi K V, Ashwini  
Mentor: Dr. Gokul Rajan V  
Alliance University, Bengaluru

## Usage
```bash
pip install -r requirements.txt
python app_simple.py
```
Open http://localhost:5000
```

### **3. .gitignore** (IMPORTANT!)

Create this file to prevent uploading large folders:
```
# Datasets
dataset/
*.jpg
*.png
*.jpeg

# Models
models/
*.h5
*.keras

# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/

# Results
results/
```

