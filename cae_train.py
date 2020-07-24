# -*- coding: utf-8 -*-
"""Convolutional autoencoder training supervisor.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import errno
import logging
import datetime

import click
import numpy as np
import tensorflow as tf
import sklearn.model_selection as sklms
import matplotlib.pyplot as plt

import utils
import models.cae as cae


_DEFAULTS = {
    'model_dir': 'cae_model/',
    'model_file': 'cae.h5',
    'checkpoint_dir': 'cae_model/checkpoints/',
    'checkpoint_file': 'cae.{epoch:02d}-{val_loss:.2f}.h5',
    'dataset_dir': 'datasets/',
    'log_dir': 'logs/cae',

    'seed': int(datetime.datetime.utcnow().strftime('%d%m%Y')),
    'emb_size': 512,

    'batch_size': 64,
    'epochs': 10,
    'learning_rate': 0.001,
    'test_size': 0.2
}


def train(
    lr, batch, epochs, split, seed=None, flags=[], options=_DEFAULTS
):
    """Train and save best autoencoder model.

    Args:
        lr (float):
            Learning rate for minimizing loss during training.
        batch (int):
            Batch size of minibatches to use during training.
        epochs (int):
            Number of epochs for training model.
        split (float):
            Fraction of dataset to split off as test data.
        seed (int, optional):
            Random number generator seed. If not set it will be randomised.
        flags (list of str, optional):
            A flag is set if it is present in the passed list.
            List of possible flags:
                overwrite-model: The function does not check for previously
                    trained models to add training data to, and instead starts
                    a new model and overwrites the one existing in the default
                    location.
                enable-tensorboard: Flag for TensorBoard visualization
                    functionality.
                sanity-test: Flag for code sanity test (training using a
                    single image).
        options (dict, optional):
            A dictionary of various types. If not set, the _DEFAULTS variable
            is used.

    Returns:
        tensorflow.keras.model.Model:
            Trained convolutional autoencoder model.
        tuple(numpy.ndarray, numpy.ndarray):
            Testing data for the model containing both noisy and clean input.

    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()

    # Create paths to save the model in if need be
    utils.setup_path(options['model_dir'])
    utils.setup_path(options['checkpoint_dir'])

    # Load synthetic datasets
    if 'sanity-test' in flags:
        logging.info('Sanity test ENABLED: loading datasets with length=1.')

        X = np.array(utils.load_dataset(os.path.join(options['dataset_dir'],
            'dipoles_hc_noise'), limit=1))
        Y = np.array(utils.load_dataset(os.path.join(options['dataset_dir'],
            'dipoles_hc'), limit=1))
    else:
        # Loading noisy dataset
        X = np.array(
            utils.load_dataset(os.path.join(options['dataset_dir'],
                'dipoles_hc_noise'))
            + utils.load_dataset(os.path.join(options['dataset_dir'],
                'dipoles_lc_noise'))
            + utils.load_dataset(os.path.join(options['dataset_dir'],
                'dipoles_vlc_noise'))
        )
        # Loading clean dataset
        Y = np.array(
            utils.load_dataset(os.path.join(options['dataset_dir'],
                'dipoles_hc'))
            + utils.load_dataset(os.path.join(options['dataset_dir'],
                'dipoles_lc'))
            + utils.load_dataset(os.path.join(options['dataset_dir'],
                'dipoles_vlc'))
        )

    X = preprocess_images(X)
    Y = preprocess_images(Y)

    img_shape = X.shape[1:]

    logging.debug("min(X)={0}, max(X)={1}, avg(X)={2}, var(X)={3}".format(
        np.min(X), np.max(X), np.average(X), np.var(X)
    ))
    logging.debug("min(Y)={0}, max(Y)={1}, avg(Y)={2}, var(Y)={3}".format(
        np.min(Y), np.max(Y), np.average(Y), np.var(Y)
    ))
    logging.debug("X_shape={0}, Y_shape={1}, img_shape={2}".format(
        X.shape, Y.shape, img_shape
    ))

    if 'sanity-test' in flags:
        X_train = X
        Y_train = Y
        X_test = X
        Y_test = Y
        val_dataset = (X, Y)
    else:
        X_train, X_test, Y_train, Y_test = sklms.train_test_split(X, Y,
            test_size=split, random_state=seed)
        X_test, X_val = np.array_split(X_test, 2)
        Y_test, Y_val = np.array_split(Y_test, 2)
        val_dataset = (X_val, Y_val)

    test_dataset = (X_test, Y_test)

    # Load existing model for continued training if options allow
    model_path = os.path.join(options['model_dir'], options['model_file'])
    if ('overwrite-model' not in flags
    and 'sanity-test' not in flags
    and os.path.exists(model_path)):
        autoencoder = load_model(model_path)
    else:
        logging.info('Creating autoencoder.')
        autoencoder = cae.build(
            img_shape,
            embedding_size=options['emb_size']
        )
        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=lr),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['mse']
        )

    autoencoder.summary()

    # Define callbacks
    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=1e-2,
        patience=2,
        verbose=1
    ))
    if 'sanity-test' not in flags:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            os.path.join(
                options['checkpoint_dir'], options['checkpoint_file']),
            monitor='val_loss',
            save_best_only=True,  # checkpoint only when `val_loss` improves
            save_freq='epoch',
            verbose=1
        ))
    if 'enable-tensorboard' in flags:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=options['log_dir'],
            write_graph=True,
            write_images=True
        ))

    # Train the autoencoder model
    autoencoder.fit(
        x=X_train,
        y=Y_train,
        epochs=epochs,
        batch_size=batch,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )

    if 'sanity-test' not in flags:
        logging.info('Saving model to `{0}`.'.format(model_path))
        autoencoder.save(model_path)

    logging.info('Evaluating.')
    metrics = autoencoder.evaluate(
        X_test,
        Y_test,
        verbose=1
    )

    logging.info('- Loss: {:.6f}'.format(metrics[0]))
    logging.info('- MSE: {:.6f}'.format(metrics[1]))

    return autoencoder, test_dataset


def preprocess_images(x):
    """Convert array of images to machine trainable data.

    Args:
        x (numpy.ndarray):
            Image data represented as a single image or array of images.

    Returns:
        numpy.ndarray: Transformed image data.

    """
    # scale image data to (0, 1)
    x = (x.astype('float32') / 255.0)
    # add channel dimension
    x = np.expand_dims(x, axis=-1)
    return x


def visualise(model, x_test, fig_title='', n=9, save_png=None):
    """Show matplotlib figure visualising trained autoencoder output.

    Figure will contain two rows of images: top row showing inputs into the
    model predictor, and the bottom row showing the corresponding outputs.

    Args:
        model (tensorflow.keras.model.Model):
            Keras model to use for prediction.
        x_test (numpy.ndarray):
            Numpy array containing input images for the predictor.
        fig_title (str, optional):
            Title to display on the figure.
        n (int, optional):
            Number of columns of images to display.

    Returns:
        matplotlib.figure.Figure: Current figure object.

    """
    decoded_imgs = model.predict(x_test[0:n])

    fig = plt.figure(figsize=(20, 4))
    fig.suptitle(fig_title, fontsize=16)
    for i in range(0, n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(
            x_test[i].reshape(x_test.shape[1], x_test.shape[2]),
            clim=(0.0, 1.0))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel('Original')

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(decoded_imgs.shape[1],
            decoded_imgs.shape[2]),
            clim=(0.0, 1.0))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel('Reconstruction')

    return fig


def load_model(path):
    """Load a previously trained model.

    Args:
        path (str): Path to file.

    Returns:
        tensorflow.keras.model.Model: Trained Keras model.

    """
    if os.path.exists(path):
        logging.info('Loading model from "{0}".'.format(path))
        return tf.keras.models.load_model(path)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)


@click.command(name='Training configuration')
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
    '-e',
    '--epochs',
    type=int,
    default=_DEFAULTS['epochs'],
    show_default=True,
    help="""Number of epochs for training model."""
)
@click.option(
    '-ts',
    '--split',
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
    help="""Flag for overwriting previously trained model instead of adding
        to it."""
)
@click.option(
    '-tb',
    '--enable-tensorboard',
    is_flag=True,
    help="""Flag for tensorboard visualization."""
)
@click.option(
    '-v',
    '--visualise',
    is_flag=True,
    help="""Flag for tensorboard visualization."""
)
@click.option(
    '-t',
    '--test',
    is_flag=True,
    help="""Flag for sanity test mode (training using a dataset with a single
        image)."""
)
@click.option(
    '--debug',
    is_flag=True,
    help="""Logs debug messages during script run."""
)
def main(**kwargs):
    """Trains the autoencoder and saves best model."""
    LOG_FORMAT = '[%(levelname)s] %(message)s'
    if kwargs['debug'] is True:
        LOG_LEVEL = 'DEBUG'
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
    else:
        LOG_LEVEL = 'INFO'

    logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)

    try:
        flags = []
        if kwargs['overwrite_model'] is True:
            flags.append('overwrite-model')
        if kwargs['enable_tensorboard'] is True:
            flags.append('enable-tensorboard')
        if kwargs['test'] is True:
            flags.append('sanity-test')

        autoencoder, test = train(
            lr=kwargs['learning_rate'],
            batch=kwargs['batch_size'],
            epochs=kwargs['epochs'],
            split=kwargs['split'],
            seed=kwargs['seed'],
            flags=flags
        )

        if kwargs['test'] is True:
            logging.info('Testing visualisation.')
            visualise(autoencoder, test[0], 'Visualisation test', 1)
        else:
            logging.info('Visualising noisy images using the CAE.')
            visualise(autoencoder, test[0],
                'Prediction on images from test set')

        plt.show()

    except KeyboardInterrupt:
        print('EXIT')


if __name__ == '__main__':
    main()
