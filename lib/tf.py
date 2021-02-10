# -*- coding: utf-8 -*-
"""Tensorflow interface functions.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import numpy as np
import tensorflow as tf

import lib.logger
import lib.utils
import models
import weights


logger = lib.logger.logger
train_def_options = {
    'batch_size': 32,
    'epochs': 10,
    'learning_rate': 0.001,
    'test_size': 0.2,
    'val_size': 0.5
}


def train(
    ds, model_id, seed=None, flags=[], options={}
):
    """Train and save neural network model.

    Args:
        ds (datasets.Dataset): Dataset object with loaded data.
        model_id (str): Model identifier.
        seed (int): Random number generator seed.
        flags (list): A flag is set if it is present in the passed list.
        List of possible flags:
            overwrite-model: The function does not check for previously
                trained models to add training data to, and instead starts
                a new model and overwrites the one existing in the default
                location.
            sanity-test: Flag for testing mode (model not saved).
        options (dict): Additional training options. Is intersected with the
            _train_def_options dictionary.

    Returns:
        tensorflow.keras.Model: Trained model.

    """
    options = {**train_def_options, **options}

    # Set random seeds
    if seed is None:
        seed = lib.utils.generate_seed()
        logger.info('Unique seed undefined. Setting to: {0}.'.format(seed))

    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()

    # Load model definition
    model = models.load_model(model_id)

    # Set up dataset properties
    ds.batch_size = options['batch_size']
    ds.shuffle = True
    ds.apply(model.pack_data)

    # Save some dataset statistics to debug
    batch0 = ds[0]
    logger.debug("Statistics of first data batch:")
    logger.debug("X.shape={0}".format(batch0[0].shape))
    logger.debug("min(X)={0}, max(X)={1}, avg(X)={2}, var(X)={3}".format(
        np.min(batch0[0]), np.max(batch0[0]),
        np.average(batch0[0]), np.var(batch0[0])
    ))
    logger.debug("Y.shape={0}".format(batch0[1].shape))
    logger.debug("min(Y)={0}, max(Y)={1}, avg(Y)={2}, var(Y)={3}".format(
        np.min(batch0[1]), np.max(batch0[1]),
        np.average(batch0[1]), np.var(batch0[1])
    ))

    # Apply dataset split or set up test mode
    if 'sanity-test' not in flags:
        ds_test = ds.split(options['test_size'])
        ds_val = ds_test.split(options['val_size'])
    else:
        ds_test = ds
        ds_val = ds

    # Load a model to add to or set up a new one
    if ('overwrite-model' not in flags and 'sanity-test' not in flags
    and weights.weights_exist(model_id)):
        logger.info('Pre-trained weights found. Loading latest iteration.')
        model_nn = weights.load_weights(model_id)
    else:
        logger.info(
            'Pre-trained weights not used. Building model from scratch.')
        input_shape = batch0[0][0].shape
        logger.info(
            'Using input shape: {0}.'.format(input_shape))
        model_nn = model.build(options['learning_rate'], input_shape)

    # Define callbacks
    callbacks = []
    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=lib.logger.tensorboard_log_path(model_id),
        write_graph=True,
        write_images=True
    ))
    if hasattr(model, 'es_callback'):
        callbacks.append(model.es_callback)
    # if 'sanity-test' not in flags:
    #     callbacks.append(tf.keras.callbacks.ModelCheckpoint(
    #         os.path.join(
    #             options['checkpoint_dir'], options['checkpoint_file']),
    #         monitor='val_loss',
    #         save_best_only=True,  # checkpoint only when `val_loss` improves
    #         save_freq='epoch',
    #         verbose=1
    #     ))

    # Train the autoencoder model
    model_nn.fit(
        x=ds,
        validation_data=ds_val,
        epochs=options['epochs'],
        callbacks=callbacks,
        verbose=2
    )

    # Evaluate
    logger.info('Evaluating.')
    metrics = model_nn.evaluate(
        x=ds_test,
        verbose=0
    )
    model.metrics(metrics, logger)

    # Save model to weights directory
    weights_id = weights.available(model_id, str(seed))
    weights_path = weights.path(weights_id[0], weights_id[1])
    if 'sanity-test' not in flags:
        logger.info('Saving model to `{0}`.'.format(weights_path))
        model_nn.save(weights_path)

    return model_nn


def predict(X, model_id, weights_id):
    """Output predictions for input samples using selected trained model.

    Args:
        X (numpy.ndarray): Input data to use for predictions.
        model_id (str): Model identifier in string format.
        weights_id (tuple): Weights file identifier in tuple of strings format.
            The tuple should be of the form: (model_id, iteration_id).

    Returns:
        numpy.ndarray: Predictions.

    """
    model = models.load_model(model_id)
    model_nn = weights.load_weights(weights_id[0], weights_id[1])

    single_image = False
    if len(X.shape) == 2:
        single_image = True
        X = np.array([X])

    X = model.pack_data(X)
    logger.debug(
        "after pack: min(X)={0}, max(X)={1}, avg(X)={2}, var(X)={3}".format(
            np.min(X), np.max(X), np.average(X), np.var(X)
        )
    )

    pred = model_nn.predict(X)
    logger.debug(
        "after predict: min(X)={0}, max(X)={1}, avg(X)={2}, var(X)={3}".format(
            np.min(pred), np.max(pred), np.average(pred), np.var(pred)
        )
    )

    pred = model.unpack_data(pred)

    if single_image is True:
        pred = np.squeeze(pred)

    return pred
