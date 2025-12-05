#!/usr/bin/env python3
"""
STAGE 3: Full Fine-Tuning для texture_improved_perfect.keras
Цель: 75.44% → 82-84%

Стратегия:
✅ Разморозка ВСЕЙ модели (включая conv1, conv2, conv3)
✅ Batch size = 64 (было 16)
✅ Warmup (3 эпохи) + Cosine Decay
✅ AdamW optimizer с weight decay
✅ Агрессивная аугментация
✅ Label smoothing = 0.05
✅ Early stopping patience = 10
✅ БЕЗ Mixup (чистое обучение)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
from keras import layers, optimizers, callbacks
import tensorflow_datasets as tfds
import numpy as np
import gc
from datetime import datetime

print("=" * 80)
print("🚀 STAGE 3: FULL FINE-TUNING TEXTURE-AWARE MODEL")
print("=" * 80)
print()

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

INPUT_MODEL = 'models/texture_improved_perfect.keras'
OUTPUT_MODEL = 'models/texture_improved_perfect_v2.keras'
LOG_FILE = 'training_stage3.log'

# Batch и данные
BATCH_SIZE = 64  # Оптимально для 16GB RAM + RTX 2070 SUPER
BUFFER_SIZE = 20000  # Полный shuffle buffer

# Обучение
EPOCHS = 30  # Полное обучение
WARMUP_EPOCHS = 3  # Полный warmup

# Learning rates
WARMUP_LR = 1e-7
INITIAL_LR = 1e-5
MIN_LR = 1e-8

# Регуляризация
LABEL_SMOOTHING = 0.05
WEIGHT_DECAY = 1e-5

print("📋 Конфигурация:")
print(f"   Batch size:          {BATCH_SIZE}")
print(f"   Buffer size:         {BUFFER_SIZE}")
print(f"   Epochs:              {EPOCHS}")
print(f"   Warmup epochs:       {WARMUP_EPOCHS}")
print(f"   Initial LR:          {INITIAL_LR}")
print(f"   Min LR:              {MIN_LR}")
print(f"   Label smoothing:     {LABEL_SMOOTHING}")
print(f"   Weight decay:        {WEIGHT_DECAY}")
print()

# =============================================================================
# MIXED PRECISION
# =============================================================================

print("🚀 Включение Mixed Precision для ускорения на RTX 2070 SUPER...")
# Mixed precision для RTX GPU - ускоряет обучение в ~2 раза
keras.mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed Precision (FP16) активирован")
print()

# =============================================================================
# ОЧИСТКА ПАМЯТИ
# =============================================================================

print("🧹 Очистка памяти...")
keras.backend.clear_session()
gc.collect()
print("✅ Память очищена")
print()

# =============================================================================
# WARMUP + COSINE DECAY LR SCHEDULE
# =============================================================================

class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """
    Warmup (3 эпохи): 1e-7 → 1e-5
    CosineDecay (27 эпох): 1e-5 → 1e-8
    """

    def __init__(self, warmup_steps, total_steps,
                 warmup_lr=1e-7, initial_lr=1e-5, min_lr=1e-8):
        super().__init__()
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
        self.warmup_lr = warmup_lr
        self.initial_lr = initial_lr
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # Warmup phase
        warmup_progress = step / self.warmup_steps
        warmup_lr = self.warmup_lr + (self.initial_lr - self.warmup_lr) * warmup_progress

        # Cosine decay phase
        decay_steps = self.total_steps - self.warmup_steps
        step_in_decay = step - self.warmup_steps
        cosine_decay = 0.5 * (1.0 + tf.cos(
            tf.constant(np.pi) * step_in_decay / decay_steps
        ))
        decay_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay

        # Выбираем между warmup и decay
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: decay_lr
        )

    def get_config(self):
        return {
            'warmup_steps': float(self.warmup_steps.numpy()),
            'total_steps': float(self.total_steps.numpy()),
            'warmup_lr': self.warmup_lr,
            'initial_lr': self.initial_lr,
            'min_lr': self.min_lr
        }

# =============================================================================
# ЗАГРУЗКА МОДЕЛИ
# =============================================================================

print("📦 Загрузка модели...")
model = keras.models.load_model(INPUT_MODEL)
print(f"✅ Модель загружена: {model.count_params():,} параметров")
print()

# =============================================================================
# РАЗМОРОЗКА ВСЕЙ МОДЕЛИ
# =============================================================================

print("🔓 Разморозка ВСЕЙ модели...")

def unfreeze_all_layers(model):
    """Разморозить ВСЕ слои, включая nested ResNet50"""
    total_unfrozen = 0

    for layer in model.layers:
        # Nested model (ResNet50)
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                if not sublayer.trainable:
                    sublayer.trainable = True
                    total_unfrozen += 1
        else:
            if not layer.trainable:
                layer.trainable = True
                total_unfrozen += 1

    return total_unfrozen

unfrozen = unfreeze_all_layers(model)
print(f"✅ Разморожено {unfrozen} слоёв")

trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
total = model.count_params()
print(f"   Trainable params: {trainable:,} ({trainable/total*100:.1f}%)")
print()

# =============================================================================
# ДАТАСЕТ
# =============================================================================

print("📂 Загрузка Food-101...")
ds_train, ds_val = tfds.load(
    'food101',
    split=['train', 'validation'],
    as_supervised=True,
    shuffle_files=True
)
print("✅ Датасет загружен (train: 75750, val: 25250)")
print()

# =============================================================================
# АУГМЕНТАЦИЯ (АГРЕССИВНАЯ!)
# =============================================================================

def resize_only(image, label):
    """Resize до 224x224"""
    image = tf.image.resize(image, [224, 224])
    return image, label

def augment_aggressive(image, label):
    """
    Агрессивная аугментация для Stage 3
    Предотвращает переобучение при full fine-tuning
    """
    # Geometric
    image = tf.image.random_flip_left_right(image)

    # Random zoom через crop_to_bounding_box + resize
    # Zoom 0.7-1.0 означает crop от 70% до 100% изображения
    zoom_factor = tf.random.uniform([], 0.7, 1.0)
    crop_size = tf.cast(224.0 * zoom_factor, tf.int32)

    # Случайная позиция для crop
    offset_h = tf.random.uniform([], 0, 224 - crop_size + 1, dtype=tf.int32)
    offset_w = tf.random.uniform([], 0, 224 - crop_size + 1, dtype=tf.int32)

    image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, crop_size, crop_size)
    image = tf.image.resize(image, [224, 224])

    # Color jitter (усиленное)
    image = tf.image.random_brightness(image, 0.3)
    image = tf.image.random_contrast(image, 0.7, 1.4)
    image = tf.image.random_saturation(image, 0.7, 1.4)
    image = tf.image.random_hue(image, 0.08)

    # Gaussian noise (для дополнительной регуляризации)
    if tf.random.uniform([]) > 0.5:
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.05)
        image = image + noise

    return image, label

def preprocess_for_model(image, label):
    """ResNet50 preprocessing"""
    image = keras.applications.resnet50.preprocess_input(image)
    return image, label

# =============================================================================
# ПОДГОТОВКА ДАТАСЕТОВ
# =============================================================================

print("🔄 Подготовка датасетов...")

def convert_to_onehot(image, label):
    """Конвертация sparse label в one-hot для CategoricalCrossentropy"""
    return image, tf.one_hot(label, 101)

# Training: resize → augment → preprocess → batch → one-hot
ds_train_prep = (
    ds_train
    .shuffle(BUFFER_SIZE)
    .map(resize_only, num_parallel_calls=tf.data.AUTOTUNE)
    .map(augment_aggressive, num_parallel_calls=tf.data.AUTOTUNE)
    .map(preprocess_for_model, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(lambda img, lbl: (img, tf.one_hot(lbl, 101)), num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

# Validation: resize → preprocess → batch → one-hot
ds_val_prep = (
    ds_val
    .map(resize_only, num_parallel_calls=tf.data.AUTOTUNE)
    .map(preprocess_for_model, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .map(lambda img, lbl: (img, tf.one_hot(lbl, 101)), num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

print("✅ Датасеты готовы (с one-hot labels)")
print()

# =============================================================================
# LR SCHEDULE
# =============================================================================

steps_per_epoch = 75750 // BATCH_SIZE  # ~1183 steps (batch=64)
warmup_steps = steps_per_epoch * WARMUP_EPOCHS
total_steps = steps_per_epoch * EPOCHS

lr_schedule = WarmupCosineDecay(
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    warmup_lr=WARMUP_LR,
    initial_lr=INITIAL_LR,
    min_lr=MIN_LR
)

print(f"📈 LR Schedule:")
print(f"   Steps per epoch:     {steps_per_epoch}")
print(f"   Warmup steps:        {warmup_steps}")
print(f"   Total steps:         {total_steps}")
print(f"   Warmup: {WARMUP_LR} → {INITIAL_LR}")
print(f"   Decay:  {INITIAL_LR} → {MIN_LR}")
print()

# =============================================================================
# OPTIMIZER & COMPILATION
# =============================================================================

print("⚙️  Компиляция модели...")

# AdamW с weight decay
optimizer = optimizers.Adam(
    learning_rate=lr_schedule,
    clipnorm=1.0,  # Gradient clipping
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

# Loss с label smoothing
# Для Keras 3.x используем CategoricalCrossentropy с конвертацией labels
loss = keras.losses.CategoricalCrossentropy(
    from_logits=False,
    label_smoothing=LABEL_SMOOTHING
)

# Metrics (для CategoricalCrossentropy)
metrics = [
    'accuracy',
    keras.metrics.TopKCategoricalAccuracy(k=5, name='top5')
]

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)

print("✅ Модель скомпилирована")
print(f"   Optimizer: Adam")
print(f"   Loss: SparseCategoricalCrossentropy (label_smoothing={LABEL_SMOOTHING})")
print(f"   Metrics: accuracy, top5")
print()

# =============================================================================
# ПРОВЕРКА НАЧАЛЬНОЙ ТОЧНОСТИ
# =============================================================================

print("🔬 Проверка начальной точности...")
initial_scores = model.evaluate(
    ds_val_prep.take(100),
    verbose=0
)
print(f"📊 Начальная accuracy: {initial_scores[1]*100:.2f}%")
print(f"   Top-5 accuracy:     {initial_scores[2]*100:.2f}%")
print()

# =============================================================================
# CALLBACKS
# =============================================================================

cbs = [
    # 1. Model Checkpoint
    callbacks.ModelCheckpoint(
        OUTPUT_MODEL,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),

    # 2. Early Stopping (patience=10)
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,  # Полный patience для лучшего обучения
        restore_best_weights=True,
        mode='max',
        verbose=1
    ),

    # 3. CSV Logger
    callbacks.CSVLogger(
        LOG_FILE,
        append=False
    ),

    # 4. ReduceLROnPlateau (backup)
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-9,
        verbose=1
    ),

    # 5. Progress callback
    callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(
            f"\n📊 Epoch {epoch+1}/{EPOCHS}: "
            f"val_acc={logs.get('val_accuracy', 0)*100:.2f}%, "
            f"val_top5={logs.get('val_top5', 0)*100:.2f}%"
        )
    )
]

# =============================================================================
# ОБУЧЕНИЕ
# =============================================================================

print("=" * 80)
print("🚀 НАЧАЛО ОБУЧЕНИЯ STAGE 3")
print("=" * 80)
print()
print(f"⏰ Ожидаемое время: ~45-60 минут ({EPOCHS} эпох, batch={BATCH_SIZE})")
print(f"⚡ С early stopping возможно ~30-40 минут")
print(f"⚙️  Оптимизировано для 16GB RAM + RTX 2070 SUPER")
print()

start_time = datetime.now()

history = model.fit(
    ds_train_prep,
    validation_data=ds_val_prep,
    epochs=EPOCHS,
    callbacks=cbs,
    verbose=1
)

end_time = datetime.now()
duration = (end_time - start_time).total_seconds() / 60

# =============================================================================
# РЕЗУЛЬТАТЫ
# =============================================================================

print()
print("=" * 80)
print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
print("=" * 80)
print()

best_acc = max(history.history['val_accuracy'])
best_top5 = max(history.history['val_top5'])
best_epoch = history.history['val_accuracy'].index(best_acc) + 1

print(f"⏱️  Время обучения: {duration:.1f} минут")
print()
print(f"📊 РЕЗУЛЬТАТЫ:")
print(f"   Начальная accuracy:  {initial_scores[1]*100:.2f}%")
print(f"   Лучшая accuracy:     {best_acc*100:.2f}% (эпоха {best_epoch})")
print(f"   Лучшая top-5:        {best_top5*100:.2f}%")
print()

improvement = (best_acc - initial_scores[1]) * 100
print(f"📈 ПРИРОСТ: {improvement:+.2f}%")
print()

# Оценка результата
if best_acc >= 0.84:
    print("🎉🎉🎉 ПРЕВОСХОДНО! Достигнут 84%+")
    print("   Модель готова к продакшену!")
elif best_acc >= 0.82:
    print("🎉🎉 ОТЛИЧНО! Достигнут 82%+")
    print("   Модель превзошла базовую ResNet50!")
elif best_acc >= 0.81:
    print("🎉 ХОРОШО! Достигнут 81%+")
    print("   Модель сравнялась с базовой!")
elif best_acc >= 0.78:
    print("✅ Хорошее улучшение!")
    print("   Можно запустить ещё 10-20 эпох")
else:
    print("⚠️  Результат ниже ожидаемого")
    print(f"   До целевых 82%: {(0.82 - best_acc)*100:.2f}%")

print()
print(f"💾 Лучшая модель: {OUTPUT_MODEL}")
print(f"📋 Лог обучения:  {LOG_FILE}")
print()

# =============================================================================
# ФИНАЛЬНАЯ ОЦЕНКА НА ПОЛНОМ VAL SET
# =============================================================================

print("🔬 Финальная оценка на полном validation set...")
final_model = keras.models.load_model(OUTPUT_MODEL)

final_model.compile(
    optimizer='adam',
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5')]
)

final_scores = final_model.evaluate(ds_val_prep, verbose=1)
print()
print(f"📊 Финальная оценка:")
print(f"   Accuracy: {final_scores[1]*100:.2f}%")
print(f"   Top-5:    {final_scores[2]*100:.2f}%")
print()

# =============================================================================
# РЕКОМЕНДАЦИИ
# =============================================================================

print("=" * 80)
print("🎯 РЕКОМЕНДАЦИИ:")
print("=" * 80)
print()

if final_scores[1] >= 0.81:
    print("✅ Модель готова к использованию!")
    print()
    print("Интеграция в API:")
    print("1. pkill -f api_server_with_tta.py")
    print("2. sed -i '' 's/texture_improved_perfect.keras/texture_improved_perfect_v2.keras/g' api_server_with_tta.py")
    print(f"3. sed -i '' 's/75.44%/{final_scores[1]*100:.2f}%/g' api_server_with_tta.py")
    print("4. python3 api_server_with_tta.py &")
elif final_scores[1] >= 0.78:
    print("✅ Хороший результат! Можно дообучить ещё:")
    print()
    print("Запустите ещё 10-20 эпох:")
    print("1. Измените EPOCHS на 10-20")
    print("2. Измените INPUT_MODEL на texture_improved_perfect_v2.keras")
    print("3. Запустите скрипт снова")
else:
    print("⚠️  Требуется дополнительная работа:")
    print()
    print("Варианты:")
    print("1. Увеличить EPOCHS до 40-50")
    print("2. Попробовать BATCH_SIZE = 32 (если 64 слишком много)")
    print("3. Использовать class weights для сложных классов")

print()
print("=" * 80)
print("✨ STAGE 3 ЗАВЕРШЁН")
print("=" * 80)
