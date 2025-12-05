#!/usr/bin/env python3
"""
Оценка texture-aware модели на одном загружаемом изображении
Выводит: val_accuracy, confidence, recall

Использование:
    python evaluate_texture_model_single_image.py path/to/image.jpg true_label_index
    python evaluate_texture_model_single_image.py path/to/image.jpg  # без true label
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import tensorflow as tf
import keras
import numpy as np
import json
from sklearn.metrics import recall_score
from PIL import Image

print("=" * 80)
print("🔬 ОЦЕНКА TEXTURE-AWARE МОДЕЛИ НА ОДНОМ ИЗОБРАЖЕНИИ")
print("=" * 80)
print()

# Проверка аргументов
if len(sys.argv) < 2:
    print("❌ Ошибка: Необходимо указать путь к изображению")
    print("   Использование: python evaluate_texture_model_single_image.py path/to/image.jpg [true_label_index]")
    print()
    sys.exit(1)

image_path = sys.argv[1]
true_label = int(sys.argv[2]) if len(sys.argv) > 2 else None

# Проверка существования файла
if not os.path.exists(image_path):
    print(f"❌ Ошибка: Файл не найден: {image_path}")
    sys.exit(1)

# Загрузка модели
print("📦 Загрузка модели...")
model = keras.models.load_model('models/texture_improved_perfect.keras')
print(f"✅ Модель загружена")
print(f"   Параметров: {model.count_params():,}")
print()

# Загрузка имен классов
with open('class_names.json', 'r') as f:
    class_names = json.load(f)
print(f"📋 Загружено классов: {len(class_names)}")
print()

# Загрузка и предобработка изображения
print(f"🖼️  Загрузка изображения: {image_path}")
img = Image.open(image_path)
img = img.convert('RGB')
img = img.resize((224, 224))

# Конвертация в numpy array
img_array = np.array(img, dtype=np.float32)

# Применяем препроцессинг ResNet50
img_preprocessed = keras.applications.resnet50.preprocess_input(img_array)

# Добавляем batch dimension
img_batch = np.expand_dims(img_preprocessed, axis=0)

print("✅ Изображение загружено и предобработано")
print()

# Предсказание
print("🧪 Выполнение предсказания...")
predictions = model.predict(img_batch, verbose=0)
predictions = predictions[0]  # Убираем batch dimension

# Top-1 предсказание
predicted_class = np.argmax(predictions)
confidence = float(predictions[predicted_class])

# Top-5 предсказания для справки
top_5_indices = np.argsort(predictions)[-5:][::-1]

print("✅ Предсказание выполнено")
print()

# Вычисление метрик
if true_label is not None:
    # Если известен истинный класс
    val_accuracy = 1.0 if predicted_class == true_label else 0.0

    # Recall для бинарной классификации (правильно/неправильно)
    recall = recall_score([true_label], [predicted_class], average='macro', zero_division=0.0)

    true_class_name = class_names[true_label].replace('_', ' ').title()
else:
    # Если истинный класс неизвестен
    val_accuracy = None
    recall = None
    true_class_name = "Неизвестно"

predicted_class_name = class_names[predicted_class].replace('_', ' ').title()

print("=" * 80)
print("📊 РЕЗУЛЬТАТЫ ОЦЕНКИ")
print("=" * 80)
print()
print(f"🖼️  Изображение:        {os.path.basename(image_path)}")
print(f"🎯 Истинный класс:     {true_class_name}")
print(f"🔮 Предсказанный класс: {predicted_class_name}")
print()

if val_accuracy is not None:
    print(f"val_accuracy:       {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
else:
    print(f"val_accuracy:       N/A (истинный класс не указан)")

print(f"confidence:         {confidence:.4f} ({confidence*100:.2f}%)")

if recall is not None:
    print(f"recall:             {recall:.4f} ({recall*100:.2f}%)")
else:
    print(f"recall:             N/A (истинный класс не указан)")

print()

# Top-5 предсказаний
print("📈 Top-5 предсказаний:")
for i, idx in enumerate(top_5_indices, 1):
    dish_name = class_names[idx].replace('_', ' ').title()
    conf = predictions[idx] * 100
    marker = "✓" if idx == predicted_class else " "
    print(f"   {marker} {i}. {dish_name:30s} {conf:5.2f}%")
print()

# Дополнительная статистика
print("📈 Дополнительная статистика:")
print(f"   Уверенность (confidence): {confidence*100:.2f}%")
print(f"   Энтропия предсказания:    {-np.sum(predictions * np.log(predictions + 1e-10)):.4f}")
print(f"   Max вероятность:          {np.max(predictions)*100:.2f}%")
print(f"   Min вероятность:          {np.min(predictions)*100:.2f}%")
print()

# Сохранение результатов в файл
result_filename = 'texture_model_single_image_results.txt'
with open(result_filename, 'w') as f:
    f.write("TEXTURE-AWARE MODEL SINGLE IMAGE RESULTS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Image:             {os.path.basename(image_path)}\n")
    f.write(f"True class:        {true_class_name}\n")
    f.write(f"Predicted class:   {predicted_class_name}\n\n")

    if val_accuracy is not None:
        f.write(f"val_accuracy:  {val_accuracy:.4f}\n")
    else:
        f.write(f"val_accuracy:  N/A\n")

    f.write(f"confidence:    {confidence:.4f}\n")

    if recall is not None:
        f.write(f"recall:        {recall:.4f}\n")
    else:
        f.write(f"recall:        N/A\n")

    f.write("\nTop-5 predictions:\n")
    for i, idx in enumerate(top_5_indices, 1):
        dish_name = class_names[idx].replace('_', ' ')
        conf = predictions[idx]
        f.write(f"{i}. {dish_name:30s} {conf:.4f}\n")

print(f"💾 Результаты сохранены в: {result_filename}")
print()
print("=" * 80)
