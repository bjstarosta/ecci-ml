# -*- coding: utf-8 -*-
"""Multilayer perceptron based binary classifier.

Searches for dislocations on ECCI using the synthetic data defined by a
single binary class.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import logging
import datetime

import click
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
import sklearn.model_selection as sklms
import matplotlib.pyplot as plt

import utils


_DEFAULTS = {
    'model_dir': 'bc_model/',
    'model_file': 'bc.h5',
    'checkpoint_dir': 'bc_model/checkpoints/',
    'checkpoint_file': 'bc.{epoch:02d}-{val_loss:.2f}.h5',
    'dataset_dir': 'datasets/',
    'log_dir': 'logs/bc',

    'seed': int(datetime.datetime.utcnow().strftime('%d%m%Y')),

    'batch_size': 64,
    'epochs': 15,
    'learning_rate': 0.001,
    'test_size': 0.2
}


def train_model(
    XY_train, XY_val, seed=None, flags=[], options=_DEFAULTS
):
    """Train and save best model.

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

    XY_train = (preprocess_images(XY_train[0]), XY_train[1])
    XY_val = (preprocess_images(XY_val[0]), XY_val[1])

    logging.debug("min(X)={0}, max(X)={1}, avg(X)={2}, var(X)={3}".format(
        np.min(XY_train[0]), np.max(XY_train[0]),
        np.average(XY_train[0]), np.var(XY_train[0])
    ))
    logging.debug("min(Y)={0}, max(Y)={1}, avg(Y)={2}, var(Y)={3}".format(
        np.min(XY_train[1]), np.max(XY_train[1]),
        np.average(XY_train[1]), np.var(XY_train[1])
    ))

    img_shape = XY_train[0].shape[1:]

    # Load existing model for continued training if options allow
    model_path = os.path.join(options['model_dir'], options['model_file'])
    if ('overwrite-model' not in flags and os.path.exists(model_path)):
        model = utils.load_model(model_path)
    else:
        logging.info('Creating new model.')

        model = K.Sequential()
        model.add(L.Flatten(input_shape=img_shape))
        model.add(L.Dense(300, activation="relu"))
        model.add(L.Dense(100, activation="relu"))
        model.add(L.Dense(2, activation="softmax"))

        model.compile(
            optimizer=K.optimizers.SGD(
                learning_rate=options['learning_rate']
            ),
            loss=K.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"]
        )

    model.summary()

    # Define callbacks
    callbacks = [
        K.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=1e-2,
            patience=2,
            verbose=1
        ),
        K.callbacks.ModelCheckpoint(
            os.path.join(
                options['checkpoint_dir'], options['checkpoint_file']),
            monitor='val_loss',
            save_best_only=True,  # checkpoint only when `val_loss` improves
            save_freq='epoch',
            verbose=1
        )
    ]
    if 'enable-tensorboard' in flags:
        callbacks.append(K.callbacks.TensorBoard(
            log_dir=options['log_dir'],
            write_graph=True,
            write_images=True
        ))

    # Train the model
    model.fit(
        x=XY_train[0],
        y=XY_train[1],
        epochs=options['epochs'],
        batch_size=options['batch_size'],
        validation_data=XY_val,
        callbacks=callbacks,
        verbose=1
    )

    logging.info('Saving model to `{0}`.'.format(model_path))
    model.save(model_path)

    return model


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
    # Loading clean dataset
    X_hc = utils.load_dataset(
        os.path.join(_DEFAULTS['dataset_dir'], 'dipoles_hc'))
    X_lc = utils.load_dataset(
        os.path.join(_DEFAULTS['dataset_dir'], 'dipoles_lc'))
    X_vlc = utils.load_dataset(
        os.path.join(_DEFAULTS['dataset_dir'], 'dipoles_vlc'))
    X_const = utils.load_dataset(
        os.path.join(_DEFAULTS['dataset_dir'], 'constants'))

    # Generating labels
    Y_hc = [1] * len(X_hc)
    Y_lc = [1] * len(X_lc)
    Y_vlc = [1] * len(X_vlc)
    Y_const = [0] * len(X_const)

    logging.debug("len(Y_hc)={0}".format(len(Y_hc)))
    logging.debug("len(Y_lc)={0}".format(len(Y_lc)))
    logging.debug("len(Y_vlc)={0}".format(len(Y_vlc)))
    logging.debug("len(Y_const)={0}".format(len(Y_const)))

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


@click.group()
@click.option(
    '--debug',
    is_flag=True,
    help="""Logs debug messages during script run."""
)
@click.pass_context
def main(ctx, **kwargs):
    """Use simple binary classifier to find dislocations on ECCI."""
    LOG_FORMAT = '[%(levelname)s] %(message)s'
    if kwargs['debug'] is True:
        LOG_LEVEL = 'DEBUG'
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
    else:
        LOG_LEVEL = 'INFO'

    logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)

    ctx.obj['debug'] = kwargs['debug']


@main.command()
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
@click.pass_context
def train(ctx, **kwargs):
    """Train multilayer perceptron NN with the relevant dataset."""
    try:
        flags = []
        if kwargs['overwrite_model'] is True:
            flags.append('overwrite-model')
        if kwargs['enable_tensorboard'] is True:
            flags.append('enable-tensorboard')

        # Load synthetic datasets
        XY_train, XY_test, XY_val = load_dipole_dataset(
            kwargs['split'], kwargs['seed'])

        model = train_model(
            XY_train, XY_val,
            seed=kwargs['seed'],
            flags=flags
        )

        logging.info('Evaluating.')
        XY_test = (preprocess_images(XY_test[0]), XY_test[1])
        metrics = model.evaluate(
            XY_test[0],
            XY_test[1],
            verbose=1
        )

        logging.info('- Loss: {:.6f}'.format(metrics[0]))
        logging.info('- Accuracy: {:.6f}'.format(metrics[1]))

    except KeyboardInterrupt:
        print('EXIT')


@main.command()
@click.option(
    '-n',
    '--test-num',
    type=int,
    default=6,
    show_default=True,
    help="""Number of images to test from the dataset."""
)
@click.pass_context
def test(ctx, **kwargs):
    """Test trained model."""
    try:
        model_path = os.path.join(
            _DEFAULTS['model_dir'], _DEFAULTS['model_file'])
        model = utils.load_model(model_path)

        X = np.array(utils.load_dataset(
            os.path.join(_DEFAULTS['dataset_dir'], 'ecci_test')))
        n = kwargs['test_num']

        X = preprocess_images(X)
        y_pred = model.predict(X[0:n])

        fig = plt.figure(figsize=(20, 4))
        fig.suptitle("Binary classifier test", fontsize=16)
        for i in range(0, n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(
                X[i].reshape(X.shape[1], X.shape[2]),
                clim=(0.0, 1.0))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.set_yticks([])
            ax.set_title('Pr: {0}'.format(y_pred[i]), fontsize=12)
            if i == 0:
                ax.set_ylabel('Original')

        plt.show()

    except KeyboardInterrupt:
        print('EXIT')


if __name__ == '__main__':
    main(obj={})
