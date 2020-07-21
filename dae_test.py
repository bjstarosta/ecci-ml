# -*- coding: utf-8 -*-
"""Denoising autoencoder prediction.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import logging

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import utils


def load_model():

    model_path = '/home/bjs/python/disrecog/ecci_ml/dae/dae.h5'

    if os.path.exists(model_path):
        logging.info('Loading pre-trained denoising autoencoder from "{0}".'.format(
            model_path
        ))
        model = tf.keras.models.load_model(model_path)
    else:
        logging.error('Could not find pre-trained model in "{0}".'.format(
            model_path
        ))
        quit()

    return model


def test_synthetic_data(model, dataset):

    dataset_dir = '/home/bjs/python/disrecog/ecci_ml/datasets/'

    Y = np.array(utils.load_dataset(os.path.join(dataset_dir, dataset)))
    Y = (Y.astype('float32') / 255.0)
    Y = np.expand_dims(Y, axis=-1)

    Y_decoded = model.predict(Y)

    plt.figure(figsize=(20, 4))
    n = 9
    for i in range(1, n):
        # display original
        ax = plt.subplot(2, n, i)
        plt.imshow(Y[i].reshape(Y.shape[1], Y.shape[2]), vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(Y_decoded[i].reshape(Y_decoded.shape[1], Y_decoded.shape[2]), vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


if __name__ == '__main__':
    autoencoder = load_model()
    autoencoder.summary()

    logging.info('Visualising synthetic data (clean).')
    test_synthetic_data(autoencoder, 'dipoles_test')
    logging.info('Visualising synthetic data (noisy).')
    test_synthetic_data(autoencoder, 'dipoles_test_noise')
