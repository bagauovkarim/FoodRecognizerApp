"""
Flask API сервер для Android приложения Food-101
С УЛУЧШЕНИЕМ: Test-Time Augmentation (TTA)
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageEnhance
import json
import io

app = Flask(__name__)
CORS(app)

# Конфигурация
CONFIDENCE_THRESHOLD = 70.0  # Минимальная уверенность в процентах
USE_TTA = True  # Включить Test-Time Augmentation
TTA_STEPS = 5  # Количество аугментаций (больше = точнее, но медленнее)

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
print(f"✓ Test-Time Augmentation: {'ВКЛЮЧЕН' if USE_TTA else 'ВЫКЛЮЧЕН'} ({TTA_STEPS} шагов)")

def preprocess_image(image_bytes):
    """Базовая предобработка изображения"""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('RGB')
    img = img.resize((224, 224))
    return img

def apply_augmentation(img, aug_type):
    """
    Применить аугментацию к изображению

    aug_type:
    0 - оригинал
    1 - горизонтальный flip
    2 - увеличение яркости
    3 - уменьшение яркости
    4 - увеличение контраста
    """
    if aug_type == 0:
        # Оригинал
        return img
    elif aug_type == 1:
        # Горизонтальный flip
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif aug_type == 2:
        # Увеличение яркости на 10%
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(1.1)
    elif aug_type == 3:
        # Уменьшение яркости на 10%
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(0.9)
    elif aug_type == 4:
        # Увеличение контраста на 10%
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(1.1)
    else:
        return img

def predict_with_tta(img_pil, num_augmentations=5):
    """
    Предсказание с Test-Time Augmentation

    Применяем несколько трансформаций и усредняем результаты
    """
    all_predictions = []

    for i in range(num_augmentations):
        # Применяем аугментацию
        aug_img = apply_augmentation(img_pil, i)

        # Конвертируем в numpy array
        img_array = np.array(aug_img, dtype=np.float32)

        # Применяем препроцессинг ResNet50
        img_preprocessed = keras.applications.resnet50.preprocess_input(img_array)

        # Добавляем batch dimension
        img_batch = np.expand_dims(img_preprocessed, axis=0)

        # Предсказание
        predictions = model.predict(img_batch, verbose=0)
        all_predictions.append(predictions[0])

    # Усредняем все предсказания
    avg_predictions = np.mean(all_predictions, axis=0)

    return avg_predictions

def predict_without_tta(img_pil):
    """Обычное предсказание без TTA"""
    img_array = np.array(img_pil, dtype=np.float32)
    img_preprocessed = keras.applications.resnet50.preprocess_input(img_array)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    predictions = model.predict(img_batch, verbose=0)
    return predictions[0]

@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'model': 'Improved Texture-Aware ResNet50 + TTA',
        'classes': len(class_names),
        'accuracy': '75.44%',
        'tta_enabled': USE_TTA,
        'tta_steps': TTA_STEPS if USE_TTA else 0,
        'improvements': 'Mixup + CosineDecayRestarts + TTA'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_file = request.files['image']
        image_bytes = image_file.read()

        # Предобработка
        img_pil = preprocess_image(image_bytes)

        # Предсказание с TTA или без
        if USE_TTA:
            predictions = predict_with_tta(img_pil, TTA_STEPS)
        else:
            predictions = predict_without_tta(img_pil)

        # Получаем топ-5
        top_5_indices = np.argsort(predictions)[-5:][::-1]

        # Получаем топ-1 предсказание и его уверенность
        top1_confidence = float(predictions[top_5_indices[0]] * 100)

        results = []
        for idx in top_5_indices:
            dish_name = class_names[idx].replace('_', ' ').title()
            confidence = float(predictions[idx] * 100)
            results.append({'dish': dish_name, 'confidence': f'{confidence:.1f}'})

        # Проверка порога уверенности
        low_confidence = top1_confidence < CONFIDENCE_THRESHOLD

        response = {
            'success': True,
            'predictions': results,
            'low_confidence': low_confidence,
            'top1_confidence': round(top1_confidence, 1),
            'tta_used': USE_TTA
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
    print("🚀 API СЕРВЕР ЗАПУЩЕН (С TTA УЛУЧШЕНИЕМ)")
    print("="*70)
    print(f"\n📱 IP адрес для Android приложения:")
    print(f"   {local_ip}:5001")
    print(f"\n⚡ Test-Time Augmentation: {'ВКЛЮЧЕН' if USE_TTA else 'ВЫКЛЮЧЕН'}")
    if USE_TTA:
        print(f"   Количество аугментаций: {TTA_STEPS}")
        print(f"   Ожидаемое улучшение: +2-4% точности")
        print(f"   Время обработки: ~{TTA_STEPS}x дольше")
    print(f"\n⚠️  Mac и телефон должны быть в одной WiFi сети!")
    print("="*70 + "\n")

    app.run(host='0.0.0.0', port=5001, debug=False)
