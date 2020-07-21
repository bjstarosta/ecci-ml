# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras.layers as L


def build(input_shape, lr, debug=False):
    # encoder
    layers = [
        L.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
        L.MaxPooling2D(pool_size=(2, 2), padding='same'),
        L.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
        L.MaxPooling2D(pool_size=(2, 2), padding='same'),
    ]

    # decoder
    layers += [
        L.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
        L.UpSampling2D(size=(2, 2)),
        L.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
        L.UpSampling2D(size=(2, 2)),
        L.Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')
    ]

    autoencoder = tf.keras.Sequential()
    autoencoder.add(L.Input(shape=input_shape))
    for layer in layers:
        autoencoder.add(layer)
        if debug == True:
            autoencoder.summary()

    optimiser = tf.keras.optimizers.Adadelta(learning_rate=lr)
    #optimiser = tf.keras.optimizers.Adam(learning_rate=lr)

    #loss = tf.keras.losses.BinaryCrossentropy()
    loss = tf.keras.losses.MeanSquaredError()

    autoencoder.compile(optimizer=optimiser, loss=loss)
    return autoencoder


if __name__ == '__main__':
    # Test model summary
    build((52, 52, 1), 0.001, True)
