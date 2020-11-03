# -*- coding: utf-8 -*-
"""Convolutional autoencoder model constructor.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
import numpy as np


es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=1e-2,
    patience=2,
    verbose=1
)


def build(lr):
    """Return a convolutional autoencoder model.

    Returns:
        tensorflow.keras.Model: The autoencoder model.
    """
    input_shape = (52, 52, 1)
    filters = [512, 256, 64]
    embedding_size = 512
    activation = tf.nn.relu

    # Encoder (Image -> Embedding)
    layers = [
        L.Conv2D(filters[0], kernel_size=(3, 3), padding='same',
            activation=activation),
        L.MaxPooling2D(pool_size=(2, 2), padding='same'),
        L.Conv2D(filters[1], kernel_size=(3, 3), padding='same',
            activation=activation),
        L.MaxPooling2D(pool_size=(2, 2), padding='same'),
        L.Conv2D(filters[2], kernel_size=(3, 3), padding='same',
            activation=activation),
        L.MaxPooling2D(pool_size=(2, 2), padding='same'),
        L.Flatten(),
        L.Dense(units=embedding_size, activation=None, name='embedding')
    ]

    # Decoder (Embedding -> Image)
    layers += [
        L.Dense(units=np.prod(input_shape), activation='relu'),
        L.Reshape(input_shape),
        L.Conv2DTranspose(filters[1], kernel_size=(3, 3), padding='same',
            activation=activation),
        L.Conv2DTranspose(filters[0], kernel_size=(3, 3), padding='same',
            activation=activation),
        L.Conv2DTranspose(input_shape[2], kernel_size=(3, 3), padding='same',
            activation=None)
    ]

    autoencoder = K.Sequential()
    autoencoder.add(L.Input(shape=input_shape))
    for layer in layers:
        autoencoder.add(layer)

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=lr),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['mse']
    )

    return autoencoder


def pack_data(X):
    """Convert array of images to machine trainable data.

    Args:
        X (numpy.ndarray): Image data represented as a single image
            or array of images.

    Returns:
        numpy.ndarray: Transformed image data.

    """
    # scale image data to (0, 1)
    X = (X.astype('float32') / 255.0)
    # add channel dimension
    X = np.expand_dims(X, axis=-1)
    return X


def unpack_data(X):
    """Convert neural network output data back to images.

    Args:
        X (numpy.ndarray): Transformed image data.

    Returns:
        numpy.ndarray: Image data represented as a single image
            or array of images.

    """
    # process predictions back into usable data
    pass


def metrics(m, log):
    """Output model evaluation metrics to the logger.

    Args:
        m (tuple): Result of tensorflow.keras.Model.evaluate()
        log (logging.Logger): Logger to log the metric data to.

    Returns:
        None

    """
    log.info('Loss: {:.6f}'.format(m[0]))
    log.info('MSE: {:.6f}'.format(m[1]))


if __name__ == '__main__':
    # Test model summary
    model = build()
    model.summary()
    K.utils.plot_model(model, to_file='cae.png', show_shapes=True)
