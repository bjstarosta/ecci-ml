# -*- coding: utf-8 -*-
"""Denoising autoencoder prediction.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import logging

import click
import numpy as np
import matplotlib.pyplot as plt

import cae_train
import utils


def test_synthetic_data(model, dataset, dataset_dir='datasets/', fig_title=''):
    """Display a figure testing autoencoder prediction on images from dataset.

    Args:
        model (tensorflow.keras.model.Model): Trained model.
        dataset (str): Desired dataset subdirectory in the dataset folder.
        dataset_dir (str): Path to dataset folder.

    """
    Y = np.array(utils.load_dataset(os.path.join(dataset_dir, dataset)))
    Y = (Y.astype('float32') / 255.0)
    Y = np.expand_dims(Y, axis=-1)

    cae_train.visualise(model, Y, fig_title, 6)


@click.group()
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help="""Logs debug messages during script run."""
)
def main(**kwargs):
    """Autoencoder prediction testing script."""
    LOG_FORMAT = '[%(levelname)s] %(message)s'
    if kwargs['verbose'] is True:
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        LOG_LEVEL = 'DEBUG'
    else:
        LOG_LEVEL = 'INFO'

    logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)


@main.command()
@click.option(
    '-p',
    '--model-path',
    type=str,
    default='cae_model/cae.h5',
    show_default=True,
    help="""Path to trained model .h5 file."""
)
def comparison(**kwargs):
    """Synthetic and real data autoencoder prediction comparison."""
    autoencoder = cae_train.load_model(kwargs['model_path'])
    autoencoder.summary()

    logging.info('Visualising synthetic data (clean).')
    test_synthetic_data(autoencoder, 'dipoles_test',
        fig_title='Clean synthetic data comparison')
    logging.info('Visualising synthetic data (noisy).')
    test_synthetic_data(autoencoder, 'dipoles_test_noise',
        fig_title='Noisy synthetic data comparison')
    logging.info('Visualising ECCI data (noisy).')
    test_synthetic_data(autoencoder, 'ecci_test',
        fig_title='Experimental (ECCI) data comparison')

    plt.show()


if __name__ == '__main__':
    main()
