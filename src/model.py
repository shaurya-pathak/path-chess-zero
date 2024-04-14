import tensorflow as tf
from tensorflow.keras import layers, models, Input
import numpy as np

def squeeze_excitation_block(input_tensor, ratio=16):
    init = input_tensor
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)

    x = layers.multiply([init, se])
    return x

def residual_se_block(input_tensor, kernel_size=3, filters=64):
    x = layers.Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = squeeze_excitation_block(x)

    x = layers.add([x, input_tensor])
    x = tf.nn.relu(x)
    return x

import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_alpha_zero_model(input_shape=(4, 8, 8, 14), residual_blocks=5, filters=256):
    # Note: Input shape reflects the separate historical dimensions
    inputs = Input(shape=input_shape)
    
    # Adapt the input for use in 2D convolutional layers:
    # We need to reshape input to merge the history with channels: (8, 8, 4*14)
    reshaped_inputs = layers.Reshape((8, 8, 4*14))(inputs)

    # Initial convolutional layer
    x = layers.Conv2D(filters, 3, padding='same')(reshaped_inputs)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)

    # Stack of residual SE blocks
    def residual_se_block(input_layer, filters):
        # Start block
        x = layers.Conv2D(filters, 3, padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = tf.nn.relu(x)

        # Middle block
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        # Squeeze and Excitation layers
        se = tf.reduce_mean(x, [1, 2], keepdims=True)
        se = layers.Conv2D(filters // 16, 1)(se)
        se = tf.nn.relu(se)
        se = layers.Conv2D(filters, 1)(se)
        se = tf.nn.sigmoid(se)

        x = x * se

        # Residual and output
        x = layers.add([input_layer, x])
        x = tf.nn.relu(x)
        return x

    for _ in range(residual_blocks):
        x = residual_se_block(x, filters=filters)

    # Policy head
    policy_conv = layers.Conv2D(2, 1, padding='same')(x)
    policy_conv = layers.BatchNormalization()(policy_conv)
    policy_conv = tf.nn.relu(policy_conv)
    policy_flat = layers.Flatten()(policy_conv)
    policy_output = layers.Dense(4672, activation='softmax', name='policy_output')(policy_flat)

    # Value head
    value_conv = layers.Conv2D(1, 1)(x)
    value_conv = layers.BatchNormalization()(value_conv)
    value_conv = tf.nn.relu(value_conv)
    value_flat = layers.Flatten()(value_conv)
    value_dense = layers.Dense(64, activation='relu')(value_flat)
    value_output = layers.Dense(1, activation='tanh', name='value_output')(value_dense)

    # Create the model
    model = models.Model(inputs=inputs, outputs=[policy_output, value_output])
    return model

def train_model(model, game_data, n_epochs):
    # Assuming game_data is a list of tuples: (encoded_state, policy_vector, value_estimate)
    states, policy_vectors, value_estimates = zip(*game_data)

    # Convert lists to numpy arrays for training
    X = np.array(states)
    Y_policy = np.array(policy_vectors)
    Y_value = np.array(value_estimates)

    # Train the model. This assumes your model has two outputs: policy and value
    model.fit(X, [Y_policy, Y_value], batch_size=64, epochs=n_epochs)
