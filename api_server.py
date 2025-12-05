"""
Flask API сервер для Android приложения Food-101
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import io

app = Flask(__name__)
CORS(app)

# Конфигурация
CONFIDENCE_THRESHOLD = 70.0  # Минимальная уверенность в процентах

# Загрузка модели
print("Загрузка модели...")
model = keras.models.load_model('models/texture_improved_perfect.keras')
with open('class_names.json', 'r') as f:
    class_names = json.load(f)
print(f"✓ Модель загружена: Improved Texture-Aware ResNet50")
print(f"✓ Accuracy: 75.44%")
print(f"✓ Параметров: {model.count_params():,}")
print(f"✓ Классов: {len(class_names)}")
print(f"✓ Порог уверенности: {CONFIDENCE_THRESHOLD}%")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'model': 'Improved Texture-Aware ResNet50',
        'classes': len(class_names),
        'accuracy': '75.44%',
        'improvements': 'Mixup + CosineDecayRestarts + Better augmentation'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_file = request.files['image']
        image_bytes = image_file.read()
        img_array = preprocess_image(image_bytes)
        predictions = model.predict(img_array, verbose=0)
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]

        # Получаем топ-1 предсказание и его уверенность
        top1_confidence = float(predictions[0][top_5_indices[0]] * 100)

        results = []
        for idx in top_5_indices:
            dish_name = class_names[idx].replace('_', ' ').title()
            confidence = float(predictions[0][idx] * 100)
            results.append({'dish': dish_name, 'confidence': f'{confidence:.1f}'})

        # Проверка порога уверенности
        low_confidence = top1_confidence < CONFIDENCE_THRESHOLD

        response = {
            'success': True,
            'predictions': results,
            'low_confidence': low_confidence,
            'top1_confidence': round(top1_confidence, 1)
        }

        if low_confidence:
            response['message'] = 'Пожалуйста, сделайте более четкое фото еды'

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n" + "="*70)
    print("🚀 API СЕРВЕР ЗАПУЩЕН")
    print("="*70)
    print(f"\n📱 IP адрес для Android приложения:")
    print(f"   {local_ip}:5001")
    print(f"\n⚠️  Mac и телефон должны быть в одной WiFi сети!")
    print("="*70 + "\n")

    app.run(host='0.0.0.0', port=5001, debug=False)
