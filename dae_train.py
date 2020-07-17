# -*- coding: utf-8 -*-
"""Denoising autoencoder training supervisor.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import logging
import datetime

import click
import numpy as np
import tensorflow as tf
import sklearn.model_selection as sklms
import matplotlib.pyplot as plt

import utils
import models.conv_denoising_autoencoder as cdae


_DEFAULTS = {
    'model_dir': 'dae/',
    'checkpoint_dir': 'dae/checkpoints',
    'dataset_dir': 'datasets/',
    'log_dir': 'logs/dae',

    'seed': int(datetime.datetime.utcnow().strftime('%d%m%Y')),
    'batch_size': 128,
    'epochs': 100,
    'epoch_save_interval': 1,
    'learning_rate': 0.001,
    'test_size': 0.2
}


def train(
        lr, batch, epochs, epoch_interval, split, seed=None,
        model_dir=_DEFAULTS['model_dir'],
        checkpoint_dir=_DEFAULTS['checkpoint_dir'],
        dataset_dir=_DEFAULTS['dataset_dir'],
        log_dir=_DEFAULTS['log_dir'],
        overwrite_model=False, tb=False, test=False
    ):
    """Trains the Autoencoder and saves best model.

    Args:
        lr (float):
            Learning rate for minimizing loss during training.
        batch (int):
            Batch size of minibatches to use during training.
        epochs (int):
            Number of epochs for training model.
        epoch_interval (int):
            Epoch interval to save model checkpoints during training.
        split (float):
            Fraction of dataset to split off as test data.
        seed (int, optional):
            Random number generator seed. If not set it will be randomised.
        model_dir (str, optional):
            Path to directory where trained models will be saved for later use.
        checkpoint_dir (str, optional):
            Path to directory where trained models will be saved for later use.
        dataset_dir (str, optional):
            Path to dataset directory.
        tb (bool, optional):
            Flag for TensorBoard visualization. Defaults to True.
        test (bool, optional):
            Flag for code sanity test (training using a single image).
            Defaults to False.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    utils.reset_tf_session()
    utils.setup_path(model_dir)
    utils.setup_path(checkpoint_dir)

    if test == True:
        logging.info('Sanity test ENABLED: loading datasets with length=1.')

        X = np.array(utils.load_dataset(os.path.join(dataset_dir, 'dipoles_hc_noise'), limit=1))
        Y = np.array(utils.load_dataset(os.path.join(dataset_dir, 'dipoles_hc'), limit=1))
    else:
        # Loading noisy dataset
        X_1 = utils.load_dataset(os.path.join(dataset_dir, 'dipoles_hc_noise'))
        X_2 = utils.load_dataset(os.path.join(dataset_dir, 'dipoles_lc_noise'))
        X_3 = utils.load_dataset(os.path.join(dataset_dir, 'dipoles_vlc_noise'))
        X = np.array(X_1 + X_2 + X_3)

        # Loading clean dataset
        Y_1 = utils.load_dataset(os.path.join(dataset_dir, 'dipoles_hc'))
        Y_2 = utils.load_dataset(os.path.join(dataset_dir, 'dipoles_lc'))
        Y_3 = utils.load_dataset(os.path.join(dataset_dir, 'dipoles_vlc'))
        Y = np.array(Y_1 + Y_2 + Y_3)

    # scale image data to (0, 1)
    X = (X.astype('float32') / 255.0)
    Y = (Y.astype('float32') / 255.0)

    # add channel dimension to image data
    X = np.expand_dims(X, axis=-1)
    Y = np.expand_dims(Y, axis=-1)
    img_shape = X.shape[1:]

    logging.debug("X_shape={0}, Y_shape={1}, img_shape={2}".format(
        X.shape, Y.shape, img_shape
    ))

    if test == True:
        train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        val_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        X_test = X
        Y_test = Y
    else:
        X_train, X_test, Y_train, Y_test = sklms.train_test_split(X, Y, test_size=split, random_state=seed)
        X_test, X_val = np.array_split(X_test, 2)
        Y_test, Y_val = np.array_split(Y_test, 2)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        val_dataset = val_dataset.batch(batch)


    model_path = os.path.join(model_dir, 'cdae.h5')
    if overwrite_model == False and test == False and os.path.exists(model_path):
        logging.info('Loading pre-trained denoising autoencoder.')
        autoencoder = tf.keras.models.load_model(model_path)
    else:
        logging.info('Creating denoising autoencoder.')
        autoencoder = cdae.build(img_shape, lr)

    autoencoder.summary()

    callbacks = [tf.keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoint_dir, 'model.ckpt'),
        monitor='val_loss',
        save_best_only=True, # checkpoint models only when `val_loss` improves
        save_freq='epoch',
        period=epoch_interval,
        verbose=1
    )]
    """, tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', # Stop training when `val_loss` is no longer improving
        min_delta=1e-2, # "no longer improving" being defined as "no better than 1e-2 less"
        patience=2, # "no longer improving" being further defined as "for at least 2 epochs"
        verbose=1,
    )"""
    if tb == True:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            write_graph=True,
            write_images=True
        ))
    for epoch in range(epochs):
        # save training data in history object
        history = autoencoder.fit(
            train_dataset,
            #x=X,
            #y=Y,
            epochs=1,
            #batch_size=batch,
            #validation_split=split,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        logging.info('Epoch {}/{}  -  loss: {:.6f}  -  val loss: {:.6f}\n'.format(
            epoch + 1,
            epochs,
            history.history['loss'][0],
            history.history['val_loss'][0]
        ))

    logging.info('Saving denoising autoencoder to `{0}`.'.format(model_path))
    autoencoder.save(model_path)

    logging.info('Evaluating DeepConv denoising autoencoder.')
    denoising_loss = autoencoder.evaluate(
        X_test,
        Y_test,
        verbose=1
    )
    logging.info('- Denoising loss: {:.6f}.'.format(denoising_loss))

    logging.info('Visualising.')
    visualise(autoencoder, X_test)


def visualise(autoencoder, x_test, n=9):
    decoded_imgs = autoencoder.predict(x_test)

    plt.figure(figsize=(20, 4))
    for i in range(1, n):
        # display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i].reshape(x_test.shape[1], x_test.shape[2]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(decoded_imgs.shape[1], decoded_imgs.shape[2]))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


@click.command(name='Training Configuration')
@click.option(
    '-lr',
    '--learning-rate',
    type=float,
    default=_DEFAULTS['learning_rate'],
    show_default=True,
    help="""Learning rate for minimizing loss during training."""
)
@click.option(
    '-bz',
    '--batch-size',
    default=_DEFAULTS['batch_size'],
    show_default=True,
    help="""Batch size of minibatches to use during training."""
)
@click.option(
    '-ne',
    '--num-epochs',
    type=int,
    default=_DEFAULTS['epochs'],
    show_default=True,
    help="""Number of epochs for training model."""
)
@click.option(
    '-se',
    '--save-every',
    type=int,
    default=_DEFAULTS['epoch_save_interval'],
    show_default=True,
    help="""Epoch interval to save model checkpoints during training."""
)
@click.option(
    '-ts',
    '--test-size',
    type=float,
    default=_DEFAULTS['test_size'],
    show_default=True,
    help="""Fraction of loaded dataset to split off as test data."""
)
@click.option(
    '-s',
    '--seed',
    type=int,
    default=_DEFAULTS['seed'],
    show_default='current UTC date in integer format, e.g. 1012020',
    help="""Random number generator seed."""
)
@click.option(
    '-o',
    '--overwrite-model',
    is_flag=True,
    help="""Flag for overwriting previously trained model instead of adding to it."""
)
@click.option(
    '-tb',
    '--enable-tensorboard',
    is_flag=True,
    help="""Flag for TensorBoard visualization."""
)
@click.option(
    '-t',
    '--test',
    is_flag=True,
    help="""Flag for sanity test mode (training using a dataset with a single image)."""
)
@click.option(
    '--debug',
    is_flag=True,
    help="""Logs debug messages during script run."""
)
def main(**kwargs):
    """Trains the autoencoder and saves best model."""
    LOG_FORMAT = '[%(levelname)s] %(message)s'
    if kwargs['debug'] == True:
        LOG_LEVEL = 'DEBUG'
    else:
        LOG_LEVEL = 'INFO'

    logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)

    try:
        train(
            lr=kwargs['learning_rate'],
            batch=kwargs['batch_size'],
            epochs=kwargs['num_epochs'],
            epoch_interval=kwargs['save_every'],
            split=kwargs['test_size'],
            overwrite_model=kwargs['overwrite_model'],
            tb=kwargs['enable_tensorboard'],
            test=kwargs['test']
        )
    except KeyboardInterrupt:
        print('EXIT')

if __name__ == '__main__':
    main()
