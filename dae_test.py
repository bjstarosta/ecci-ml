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


dataset_dir = 'datasets/'
model_path = 'dae/dae.h5'

Y_1 = utils.load_dataset(os.path.join(dataset_dir, 'dipoles_hc_noise'), 3)
Y_2 = utils.load_dataset(os.path.join(dataset_dir, 'dipoles_lc_noise'), 3)
Y_3 = utils.load_dataset(os.path.join(dataset_dir, 'dipoles_vlc_noise'), 3)
Y = np.array(Y_1 + Y_2 + Y_3)
Y = (Y.astype('float32') / 255.0)
Y = np.expand_dims(Y, axis=-1)

if os.path.exists(model_path):
    logging.info('Loading pre-trained denoising autoencoder from "{0}".'.format(
        model_path
    ))
    autoencoder = tf.keras.models.load_model(model_path)
else:
    logging.error('Could not find pre-trained model in "{0}".'.format(
        model_path
    ))

autoencoder.summary()

Y_decoded = autoencoder.predict(Y)

logging.info('Visualising.')

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
