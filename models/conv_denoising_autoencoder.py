# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras.layers as L
import numpy as np


def build(input_shape, lr, debug=False):

    embedding_size = 512

    # Encoder (Image -> Embedding)
    layers = [
        L.Conv2D(64, kernel_size=(3, 3), padding='same', activation='elu'),
        L.MaxPooling2D(pool_size=(2, 2), padding='same'),
        L.Conv2D(128, kernel_size=(3, 3), padding='same', activation='elu'),
        L.MaxPooling2D(pool_size=(2, 2), padding='same'),
        L.Conv2D(256, kernel_size=(3, 3), padding='same', activation='elu'),
        L.MaxPooling2D(pool_size=(2, 2), padding='same'),
        L.Flatten(),
        L.Dense(embedding_size, activation='elu')
    ]

    # Decoder (Embedding -> Image)
    layers += [
        L.Dense(np.prod(input_shape), activation='elu'),
        L.Reshape(input_shape),
        L.Conv2DTranspose(128, kernel_size=(3, 3), padding='same', activation='elu'),
        L.Conv2DTranspose(64, kernel_size=(3, 3), padding='same', activation='elu'),
        L.Conv2DTranspose(input_shape[2], kernel_size=(3, 3), padding='same', activation=None)
    ]

    autoencoder = tf.keras.Sequential()
    autoencoder.add(L.Input(shape=input_shape))
    for layer in layers:
        autoencoder.add(layer)
        if debug == True:
            autoencoder.summary()

    #optimiser = tf.keras.optimizers.Adadelta(learning_rate=lr)
    optimiser = tf.keras.optimizers.Adamax(learning_rate=lr)

    #loss = tf.keras.losses.BinaryCrossentropy()
    loss = tf.keras.losses.MeanSquaredError()

    autoencoder.compile(optimizer=optimiser, loss=loss)
    return autoencoder


if __name__ == '__main__':
    # Test model summary
    build((52, 52, 1), 0.001, True)
