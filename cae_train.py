# -*- coding: utf-8 -*-
"""Convolutional autoencoder training supervisor.

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
    XY_train, XY_val, seed=None, flags=[], options=_DEFAULTS
):
    """Train and save best autoencoder model.

    Args:
        XY_train (tuple of numpy.ndarray):
            Training dataset.
        XY_val (tuple of numpy.ndarray):
            Validation dataset.
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

    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()

    # Create paths to save the model in if need be
    utils.setup_path(options['model_dir'])
    utils.setup_path(options['checkpoint_dir'])

    XY_train = (preprocess_images(XY_train[0]), preprocess_images(XY_train[1]))
    XY_val = (preprocess_images(XY_val[0]), preprocess_images(XY_val[1]))

    img_shape = XY_train[0].shape[1:]

    # Load existing model for continued training if options allow
    model_path = os.path.join(options['model_dir'], options['model_file'])
    if ('overwrite-model' not in flags
    and 'sanity-test' not in flags
    and os.path.exists(model_path)):
        autoencoder = utils.load_model(model_path)
    else:
        logging.info('Creating autoencoder.')
        autoencoder = cae.build(
            img_shape,
            embedding_size=options['emb_size']
        )
        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adamax(
                learning_rate=options['learning_rate']),
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
        x=XY_train[0],
        y=XY_train[1],
        epochs=options['epochs'],
        batch_size=options['batch_size'],
        validation_data=XY_val,
        callbacks=callbacks,
        verbose=1
    )

    if 'sanity-test' not in flags:
        logging.info('Saving model to `{0}`.'.format(model_path))
        autoencoder.save(model_path)

    return autoencoder


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


def load_dipole_dataset(split, seed=None):
    """Load the full dipole + constant dataset and return it split.

    Args:
        split (float):
            Fraction of dataset to split off as test data.
        seed (int, optional):
            Random number generator seed. If not set it will be randomised.

    Returns:
        tuple: (tuple of X and Y training data, tuple of X and Y test data,
            tuple of X and Y validation data)

    """
    # Loading noisy dataset
    X_hc = utils.load_dataset(
        os.path.join(_DEFAULTS['dataset_dir'], 'dipoles_hc_noise'))
    X_lc = utils.load_dataset(
        os.path.join(_DEFAULTS['dataset_dir'], 'dipoles_lc_noise'))
    X_vlc = utils.load_dataset(
        os.path.join(_DEFAULTS['dataset_dir'], 'dipoles_vlc_noise'))
    X_const = utils.load_dataset(
        os.path.join(_DEFAULTS['dataset_dir'], 'constants_noise'))

    # Loading clean dataset
    Y_hc = utils.load_dataset(
        os.path.join(_DEFAULTS['dataset_dir'], 'dipoles_hc'))
    Y_lc = utils.load_dataset(
        os.path.join(_DEFAULTS['dataset_dir'], 'dipoles_lc'))
    Y_vlc = utils.load_dataset(
        os.path.join(_DEFAULTS['dataset_dir'], 'dipoles_vlc'))
    Y_const = utils.load_dataset(
        os.path.join(_DEFAULTS['dataset_dir'], 'constants'))

    # Splitting datasets
    X_tr_hc, X_test_hc, Y_tr_hc, Y_test_hc = \
        sklms.train_test_split(X_hc, Y_hc,
            test_size=split, random_state=seed)
    X_tr_lc, X_test_lc, Y_tr_lc, Y_test_lc = \
        sklms.train_test_split(X_lc, Y_lc,
            test_size=split, random_state=seed)
    X_tr_vlc, X_test_vlc, Y_tr_vlc, Y_test_vlc = \
        sklms.train_test_split(X_vlc, Y_vlc,
            test_size=split, random_state=seed)
    X_tr_const, X_test_const, Y_tr_const, Y_test_const = \
        sklms.train_test_split(X_const, Y_const,
            test_size=split, random_state=seed)

    XY_train = (
        np.array(X_tr_hc + X_tr_lc + X_tr_vlc + X_tr_const),
        np.array(Y_tr_hc + Y_tr_lc + Y_tr_vlc + Y_tr_const)
    )
    XY_test = (
        np.array(X_test_hc + X_test_lc + X_test_vlc + X_test_const),
        np.array(Y_test_hc + Y_test_lc + Y_test_vlc + Y_test_const)
    )
    X_test, X_val = np.array_split(XY_test[0], 2)
    Y_test, Y_val = np.array_split(XY_test[1], 2)
    XY_test = (X_test, Y_test)
    XY_val = (X_val, Y_val)

    return XY_train, XY_test, XY_val


@click.command(name='Training configuration')
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

        # Load synthetic datasets
        if 'sanity-test' in flags:
            logging.info(
                'Sanity test ENABLED: loading datasets with length=1.')

            X = np.array(utils.load_dataset(
                os.path.join(_DEFAULTS['dataset_dir'], 'dipoles_hc_noise'),
                limit=1))
            Y = np.array(utils.load_dataset(
                os.path.join(_DEFAULTS['dataset_dir'], 'dipoles_hc'),
                limit=1))
            XY_train = (X, Y)
            XY_test = (X, Y)
            XY_val = (X, Y)
        else:
            XY_train, XY_test, XY_val = load_dipole_dataset(
                kwargs['split'], kwargs['seed'])

        logging.debug("min(X)={0}, max(X)={1}, avg(X)={2}, var(X)={3}".format(
            np.min(XY_train[0]), np.max(XY_train[0]),
            np.average(XY_train[0]), np.var(XY_train[0])
        ))
        logging.debug("min(Y)={0}, max(Y)={1}, avg(Y)={2}, var(Y)={3}".format(
            np.min(XY_train[1]), np.max(XY_train[1]),
            np.average(XY_train[1]), np.var(XY_train[1])
        ))

        autoencoder = train(
            XY_train, XY_val,
            seed=kwargs['seed'],
            flags=flags
        )

        logging.info('Evaluating.')
        XY_test = (preprocess_images(XY_test[0]),
            preprocess_images(XY_test[1]))
        metrics = autoencoder.evaluate(
            XY_test[0],
            XY_test[1],
            verbose=1
        )

        logging.info('- Loss: {:.6f}'.format(metrics[0]))
        logging.info('- MSE: {:.6f}'.format(metrics[1]))

        if kwargs['test'] is True:
            logging.info('Testing visualisation.')
            visualise(autoencoder, XY_test[0], 'Visualisation test', 1)
        else:
            logging.info('Visualising noisy images using the CAE.')
            visualise(autoencoder, XY_test[0],
                'Prediction on images from test set')

        plt.show()

    except KeyboardInterrupt:
        print('EXIT')


if __name__ == '__main__':
    main()
