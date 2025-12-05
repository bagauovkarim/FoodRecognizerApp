"""
Texture-Aware ResNet50 Implementation

This module provides a texture-aware modification of the classic ResNet50 architecture
for improved food recognition performance by incorporating texture analysis capabilities.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


class ChannelAttention(layers.Layer):
    """
    Channel Attention mechanism to emphasize important feature channels.

    This helps the model focus on texture-rich channels that are crucial
    for distinguishing similar-looking food items.
    """
    def __init__(self, filters, reduction_ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.filters = filters
        self.reduction_ratio = reduction_ratio

        self.shared_dense_one = layers.Dense(
            filters // reduction_ratio,
            activation='relu',
            kernel_initializer='he_normal'
        )
        self.shared_dense_two = layers.Dense(
            filters,
            kernel_initializer='he_normal'
        )

    def call(self, inputs):
        # Global Average Pooling
        avg_pool = layers.GlobalAveragePooling2D()(inputs)
        avg_pool = layers.Reshape((1, 1, self.filters))(avg_pool)
        avg_pool = self.shared_dense_one(avg_pool)
        avg_pool = self.shared_dense_two(avg_pool)

        # Global Max Pooling
        max_pool = layers.GlobalMaxPooling2D()(inputs)
        max_pool = layers.Reshape((1, 1, self.filters))(max_pool)
        max_pool = self.shared_dense_one(max_pool)
        max_pool = self.shared_dense_two(max_pool)

        # Combine and apply sigmoid
        attention = layers.Add()([avg_pool, max_pool])
        attention = layers.Activation('sigmoid')(attention)

        return layers.Multiply()([inputs, attention])

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "reduction_ratio": self.reduction_ratio
        })
        return config


def build_texture_aware_resnet50(input_shape=(224, 224, 3), num_classes=101):
    """
    Build Texture-Aware ResNet50 model.

    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes

    Returns:
        Compiled Keras model with texture-aware modifications
    """

    # Load base ResNet50 without top layers
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling=None
    )

    # Get intermediate outputs for texture analysis
    layer_names = [
        'conv2_block3_out',   # Early texture features (56x56)
        'conv3_block4_out',   # Mid-level features (28x28)
        'conv4_block6_out',   # High-level features (14x14)
        'conv5_block3_out'    # Final features (7x7)
    ]

    outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create multi-scale feature extractor
    feature_extractor = Model(
        inputs=base_model.input,
        outputs=outputs,
        name='texture_feature_extractor'
    )

    # Build texture-aware head
    inputs = keras.Input(shape=input_shape, name='image_input')

    # Extract multi-scale features
    features = feature_extractor(inputs)

    # Apply channel attention to each scale
    conv2_features = ChannelAttention(filters=256, name='attention_conv2')(features[0])
    conv3_features = ChannelAttention(filters=512, name='attention_conv3')(features[1])
    conv4_features = ChannelAttention(filters=1024, name='attention_conv4')(features[2])
    conv5_features = ChannelAttention(filters=2048, name='attention_conv5')(features[3])

    # Depthwise convolutions for texture extraction
    # These extract local texture patterns at different scales
    texture_conv2 = layers.DepthwiseConv2D(
        kernel_size=3,
        padding='same',
        activation='relu',
        name='texture_conv2'
    )(conv2_features)

    texture_conv3 = layers.DepthwiseConv2D(
        kernel_size=3,
        padding='same',
        activation='relu',
        name='texture_conv3'
    )(conv3_features)

    texture_conv4 = layers.DepthwiseConv2D(
        kernel_size=3,
        padding='same',
        activation='relu',
        name='texture_conv4'
    )(conv4_features)

    # Resize all feature maps to same spatial size (7x7)
    texture_conv2_resized = layers.AveragePooling2D(pool_size=8, name='pool_conv2')(texture_conv2)
    texture_conv3_resized = layers.AveragePooling2D(pool_size=4, name='pool_conv3')(texture_conv3)
    texture_conv4_resized = layers.AveragePooling2D(pool_size=2, name='pool_conv4')(texture_conv4)

    # Concatenate multi-scale texture features
    multi_scale_features = layers.Concatenate(name='concat_multi_scale')([
        texture_conv2_resized,
        texture_conv3_resized,
        texture_conv4_resized,
        conv5_features
    ])

    # Final channel attention on combined features
    combined_filters = 256 + 512 + 1024 + 2048  # 3840
    multi_scale_attention = ChannelAttention(
        filters=combined_filters,
        name='attention_combined'
    )(multi_scale_features)

    # Global pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(multi_scale_attention)

    # Dense layers with dropout
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)
    x = layers.Dense(256, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create final model
    model = Model(inputs=inputs, outputs=outputs, name='texture_aware_resnet50')

    return model


def create_and_compile_model(num_classes=101, learning_rate=0.0001):
    """
    Create and compile texture-aware ResNet50 model.

    Args:
        num_classes: Number of output classes
        learning_rate: Initial learning rate

    Returns:
        Compiled model ready for training
    """
    model = build_texture_aware_resnet50(num_classes=num_classes)

    # Compile with Adam optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
    )

    return model


def get_model_summary():
    """Print model architecture summary."""
    model = build_texture_aware_resnet50()
    print("\n" + "="*80)
    print("TEXTURE-AWARE RESNET50 ARCHITECTURE")
    print("="*80 + "\n")
    model.summary()
    print("\n" + "="*80)
    print(f"Total parameters: {model.count_params():,}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Building Texture-Aware ResNet50...")
    model = create_and_compile_model()
    get_model_summary()

    # Save model architecture diagram
    try:
        keras.utils.plot_model(
            model,
            to_file='texture_aware_resnet50_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=96
        )
        print("Model architecture diagram saved to 'texture_aware_resnet50_architecture.png'")
    except Exception as e:
        print(f"Could not save architecture diagram: {e}")
