#!/usr/bin/env python3
"""
Тестирование Best Model 81% на одном изображении
Показывает топ-5 предсказаний с confidence
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
import json

def test_single_image(image_path):
    """
    Тестирует одно изображение с помощью Best Model 81%

    Args:
        image_path: Путь к изображению
    """

    # Проверка существования файла
    if not os.path.exists(image_path):
        print(f"❌ Ошибка: Файл не найден: {image_path}")
        return

    print("=" * 80)
    print("🔬 ТЕСТИРОВАНИЕ BEST MODEL 81% НА ОДНОМ ИЗОБРАЖЕНИИ")
    print("=" * 80)
    print()

    # Загрузка модели
    print("📦 Загрузка модели...")
    model = keras.models.load_model('models/best_model_81percent.keras')
    print(f"✅ Best Model 81% загружена")
    print(f"   Параметров: {model.count_params():,}")
    print()

    # Загрузка названий классов
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)

    # Загрузка и предобработка изображения
    print(f"📷 Загрузка изображения: {image_path}")
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        print(f"✅ Изображение загружено: {img.size[0]}x{img.size[1]}")

        # Изменение размера и предобработка
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32)
        img_preprocessed = keras.applications.resnet50.preprocess_input(img_array)
        img_batch = np.expand_dims(img_preprocessed, axis=0)

        print("✅ Изображение обработано (224x224)")
        print()

    except Exception as e:
        print(f"❌ Ошибка при загрузке изображения: {e}")
        return

    # Предсказание
    print("🧪 Выполнение предсказания...")
    predictions = model.predict(img_batch, verbose=0)
    print("✅ Предсказание выполнено")
    print()

    # Получаем топ-5 предсказаний
    top5_indices = np.argsort(predictions[0])[-5:][::-1]

    print("=" * 80)
    print("📊 РЕЗУЛЬТАТЫ")
    print("=" * 80)
    print()

    print("🏆 Топ-5 предсказаний:")
    print()
    for rank, idx in enumerate(top5_indices, 1):
        dish_name = class_names[idx].replace('_', ' ').title()
        confidence = predictions[0][idx] * 100

        # Добавляем индикатор для топ-1
        indicator = "👑" if rank == 1 else f"{rank}."

        # Визуальный бар уверенности
        bar_length = int(confidence / 2)  # Масштаб 0-50 символов
        bar = "█" * bar_length + "░" * (50 - bar_length)

        print(f"   {indicator:3} {dish_name:30} | {confidence:6.2f}% | {bar}")

    print()

    # Анализ уверенности
    top1_confidence = predictions[0][top5_indices[0]] * 100

    print("📈 Анализ уверенности:")
    if top1_confidence >= 90:
        print(f"   🟢 Очень высокая уверенность ({top1_confidence:.1f}%)")
        print(f"   → Модель очень уверена в предсказании")
    elif top1_confidence >= 70:
        print(f"   🟡 Высокая уверенность ({top1_confidence:.1f}%)")
        print(f"   → Модель достаточно уверена")
    elif top1_confidence >= 50:
        print(f"   🟠 Средняя уверенность ({top1_confidence:.1f}%)")
        print(f"   → Возможно, стоит проверить топ-3 предсказания")
    else:
        print(f"   🔴 Низкая уверенность ({top1_confidence:.1f}%)")
        print(f"   → Изображение может быть нечётким или не относится к Food-101")

    print()

    # Дополнительная информация
    gap = predictions[0][top5_indices[0]] - predictions[0][top5_indices[1]]
    print(f"📊 Статистика:")
    print(f"   Разрыв между топ-1 и топ-2: {gap*100:.2f}%")
    print(f"   Сумма топ-5 вероятностей:  {np.sum(predictions[0][top5_indices])*100:.2f}%")
    print()

    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python3 test_best_model_single_image.py <путь_к_изображению>")
        print()
        print("Пример:")
        print("  python3 test_best_model_single_image.py test_images/pizza.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    test_single_image(image_path)
