# -*- coding: utf-8 -*-
"""Convolutional autoencoder model constructor.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
import numpy as np


def build(
    input_shape=(52, 52, 1), filters=[64, 128, 256], embedding_size=512
):
    """Return a convolutional autoencoder model.

    Args:
        input_shape (tuple):
            Tuple of integers containing shape (width, height, channel number)
            of dataset.
        filters (list):
            Integer, the dimensionality of the output space (i.e. the number of
            output filters in the convolution).
        embedding_size (int):
            Size of the autoencoder embedding (encoder output dim).

    Returns:
        tensorflow.keras.model.Model:
            The autoencoder model.
    """
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

    return autoencoder


if __name__ == '__main__':
    # Test model summary
    model = build()
    model.summary()
    K.utils.plot_model(model, to_file='cae.png', show_shapes=True)
