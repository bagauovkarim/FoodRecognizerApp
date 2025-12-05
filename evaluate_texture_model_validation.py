#!/usr/bin/env python3
"""
Оценка texture-aware модели на случайной выборке
10 случайных классов × 10 фотографий = 100 изображений
Метрики: accuracy, recall, top-5 accuracy, confidence
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
import tensorflow_datasets as tfds
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
import json
import random

print("=" * 80)
print("🔬 ОЦЕНКА TEXTURE-AWARE МОДЕЛИ НА СЛУЧАЙНОЙ ВЫБОРКЕ")
print("=" * 80)
print()

# Загрузка модели
print("📦 Загрузка модели...")
model = keras.models.load_model('models/texture_improved_perfect.keras')
print(f"✅ Модель загружена")
print(f"   Параметров: {model.count_params():,}")
print()

# Загрузка названий классов
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

print(f"📋 Всего классов: {len(class_names)}")
print()

# Загрузка validation датасета
print("📂 Загрузка Food-101 validation датасета...")
ds_val = tfds.load(
    'food101',
    split='validation',
    as_supervised=True,
    shuffle_files=False
)
print("✅ Датасет загружен")
print()

# Препроцессинг
def preprocess(image, label):
    image = tf.image.resize(image, [224, 224])
    image = keras.applications.resnet50.preprocess_input(image)
    return image, label

# Конвертируем в список для фильтрации
print("🔄 Конвертация датасета...")
all_data = list(ds_val.as_numpy_iterator())
print(f"✅ Всего примеров: {len(all_data)}")
print()

# Выбираем 10 случайных классов
random.seed(42)  # Для воспроизводимости
selected_classes = random.sample(range(101), 10)
selected_classes_names = [class_names[i] for i in selected_classes]

print("🎲 Случайно выбранные классы:")
for i, class_idx in enumerate(selected_classes, 1):
    print(f"   {i}. {class_names[class_idx].replace('_', ' ').title()} (класс {class_idx})")
print()

# Собираем по 10 примеров для каждого класса
print("📸 Сбор по 10 фотографий для каждого класса...")
selected_data = []
class_counts = {cls: 0 for cls in selected_classes}

for image, label in all_data:
    if label in selected_classes and class_counts[label] < 10:
        selected_data.append((image, label))
        class_counts[label] += 1

        # Проверяем, собрали ли мы всё
        if all(count == 10 for count in class_counts.values()):
            break

print(f"✅ Собрано {len(selected_data)} изображений")
print()

# Создаём датасет из отобранных данных
def data_generator():
    for image, label in selected_data:
        yield image, label

ds_selected = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(), dtype=tf.int64)
    )
)

# Применяем preprocessing
ds_selected_prep = (
    ds_selected
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(10)
    .prefetch(tf.data.AUTOTUNE)
)

print("🧪 Оценка модели на выборке (100 изображений)...")
print()

# Получаем предсказания
all_predictions = []
all_top5_predictions = []
all_confidences = []
all_labels = []

for images, labels in ds_selected_prep:
    predictions = model.predict(images, verbose=0)

    # Top-1 предсказания
    predicted_classes = np.argmax(predictions, axis=1)
    all_predictions.extend(predicted_classes)

    # Top-5 предсказания
    top5_indices = np.argsort(predictions, axis=1)[:, -5:]
    all_top5_predictions.extend(top5_indices)

    # Confidence (максимальная вероятность)
    confidences = np.max(predictions, axis=1)
    all_confidences.extend(confidences)

    # Истинные метки
    all_labels.extend(labels.numpy())

# Конвертируем в numpy arrays
all_predictions = np.array(all_predictions)
all_top5_predictions = np.array(all_top5_predictions)
all_confidences = np.array(all_confidences)
all_labels = np.array(all_labels)

# Вычисляем метрики
accuracy = np.mean(all_predictions == all_labels)
average_confidence = np.mean(all_confidences)

# Recall (macro - среднее по классам)
recall_macro = recall_score(all_labels, all_predictions, labels=selected_classes, average='macro', zero_division=0)

# Recall для каждого класса отдельно
recall_per_class = recall_score(all_labels, all_predictions, labels=selected_classes, average=None, zero_division=0)

# Top-5 Accuracy
top5_correct = 0
for i, label in enumerate(all_labels):
    if label in all_top5_predictions[i]:
        top5_correct += 1
top5_accuracy = top5_correct / len(all_labels)

print("=" * 80)
print("📊 РЕЗУЛЬТАТЫ ОЦЕНКИ")
print("=" * 80)
print()
print(f"🎯 Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"🎯 Top-5 Accuracy:     {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")
print(f"📈 Recall (macro):     {recall_macro:.4f} ({recall_macro*100:.2f}%)")
print(f"💪 Avg Confidence:     {average_confidence:.4f} ({average_confidence*100:.2f}%)")
print()

# Детальная статистика
print("📈 Детальная статистика:")
print(f"   Всего изображений:        {len(all_labels)}")
print(f"   Правильно (Top-1):        {np.sum(all_predictions == all_labels)}")
print(f"   Правильно (Top-5):        {top5_correct}")
print(f"   Неправильно:              {np.sum(all_predictions != all_labels)}")
print(f"   Средняя confidence:       {average_confidence*100:.2f}%")
print(f"   Мин. confidence:          {np.min(all_confidences)*100:.2f}%")
print(f"   Макс. confidence:         {np.max(all_confidences)*100:.2f}%")
print()

# Статистика по классам
print("📊 Производительность по классам:")
print()
for i, class_idx in enumerate(selected_classes):
    class_mask = all_labels == class_idx
    class_predictions = all_predictions[class_mask]
    class_labels = all_labels[class_mask]
    class_confidences = all_confidences[class_mask]

    class_accuracy = np.mean(class_predictions == class_labels)
    class_confidence = np.mean(class_confidences)
    class_recall = recall_per_class[i]

    # Top-5 для класса
    class_top5 = all_top5_predictions[class_mask]
    class_top5_correct = np.sum([class_idx in top5 for top5 in class_top5])
    class_top5_accuracy = class_top5_correct / len(class_labels)

    class_name = class_names[class_idx].replace('_', ' ').title()
    print(f"   {class_name:25} | Acc: {class_accuracy*100:5.1f}% | Recall: {class_recall*100:5.1f}% | Top-5: {class_top5_accuracy*100:5.1f}% | Conf: {class_confidence*100:5.1f}%")

print()

# Примеры ошибок
print("❌ Примеры неправильных предсказаний:")
print()
errors_shown = 0
for i, (label, pred, conf) in enumerate(zip(all_labels, all_predictions, all_confidences)):
    if label != pred and errors_shown < 5:
        true_name = class_names[label].replace('_', ' ').title()
        pred_name = class_names[pred].replace('_', ' ').title()
        print(f"   Пример {i+1}: Истинный: {true_name} → Предсказан: {pred_name} (Conf: {conf*100:.1f}%)")
        errors_shown += 1

if errors_shown == 0:
    print("   Нет ошибок! Все предсказания верны!")

print()

# Сохраняем результаты в файл
with open('texture_model_random_sample_results.txt', 'w', encoding='utf-8') as f:
    f.write("TEXTURE-AWARE MODEL RANDOM SAMPLE EVALUATION\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Sample Size: 100 images (10 classes × 10 images)\n")
    f.write(f"Random Seed: 42\n\n")

    f.write("SELECTED CLASSES:\n")
    for i, class_idx in enumerate(selected_classes, 1):
        f.write(f"{i}. {class_names[class_idx]}\n")
    f.write("\n")

    f.write("OVERALL METRICS:\n")
    f.write(f"Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Top-5 Accuracy:     {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)\n")
    f.write(f"Recall (macro):     {recall_macro:.4f} ({recall_macro*100:.2f}%)\n")
    f.write(f"Avg Confidence:     {average_confidence:.4f} ({average_confidence*100:.2f}%)\n")
    f.write("\n")

    f.write("PER-CLASS PERFORMANCE:\n")
    for i, class_idx in enumerate(selected_classes):
        class_mask = all_labels == class_idx
        class_predictions = all_predictions[class_mask]
        class_labels = all_labels[class_mask]
        class_confidences = all_confidences[class_mask]

        class_accuracy = np.mean(class_predictions == class_labels)
        class_confidence = np.mean(class_confidences)
        class_recall = recall_per_class[i]

        class_top5 = all_top5_predictions[class_mask]
        class_top5_correct = np.sum([class_idx in top5 for top5 in class_top5])
        class_top5_accuracy = class_top5_correct / len(class_labels)

        f.write(f"{class_names[class_idx]:30} | Acc: {class_accuracy*100:5.1f}% | Recall: {class_recall*100:5.1f}% | Top-5: {class_top5_accuracy*100:5.1f}% | Conf: {class_confidence*100:5.1f}%\n")

print("💾 Результаты сохранены в: texture_model_random_sample_results.txt")
print()
print("=" * 80)
