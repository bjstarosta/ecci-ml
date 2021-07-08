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
    'learning_rate': 0.001
}


def set_seed(seed=None):

    # Set random seeds
    if seed is None:
        seed = lib.utils.generate_seed()
        logger.info('Unique seed undefined. Setting to: {0}.'.format(seed))

    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()

    return seed


def train(
    model_id, ds, ds_test, ds_val=None,
    revision_id=None, seed=None, flags=[], options={}
):
    """Train and save neural network model.

    Args:
        model_id (str): Model identifier.
        ds (datasets.Dataset): Dataset object with loaded data.
        ds_test (datasets.Dataset): Dataset object to be used for testing.
        ds_val (datasets.Dataset): Dataset object to be used for validation.
            If None is passed, validation is not performed.
        revision_id (str): Revision identifier of the model to be updated.
            If None is passed, a new model will be created.
        seed (int): Random number generator seed.
        flags (list): A flag is set if it is present in the passed list.
        List of possible flags:
            sanity-test: Flag for testing mode (model not saved).
            no-save: Do not save this model.
            no-metrics: Do not evaluate this model.
            no-early-stopping: Always train the specfied number of epochs.
                Otherwise training stops when loss begins to stagnate.
        options (dict): Additional training options. Is intersected with the
            _train_def_options dictionary.

    Returns:
        tensorflow.keras.Model: Trained model.

    """
    options = {**train_def_options, **options}

    # Load model definition
    model = models.load_model(model_id)

    if seed is None:
        seed = set_seed()

    # Set up dataset properties
    for x in [ds, ds_test, ds_val]:
        if x is None:
            continue
        x.rs = np.random.default_rng(seed=seed)
        x.batch_size = options['batch_size']
        x.shuffle_on_epoch_end = True
        x.apply(model.pack_data)
        x.preprocess(model.preprocess_data)

    # Save some dataset statistics to stdout
    if 'log-statistics' in flags:
        logger.info("Gathering statistics (this may take a while...)")

        labels = ['training set', 'test set', 'validation set']
        for i, x in enumerate([ds, ds_test, ds_val]):
            if x is None:
                continue

            logger.info("Statistics of {0}:".format(labels[i]))
            batch0 = x[0]
            stats = x.statistics()
            logger.info("X[0].shape={0}".format(batch0[0].shape))
            logger.info((
                "avg(X)={0}, var(X)={1}, std(X)={2}, "
                "min(X)={3}, max(X)={4}, count(X)={5}, "
                "NaNcount(X)={6}"
            ).format(*stats[0]))
            logger.info("Y[0].shape={0}".format(batch0[1].shape))
            logger.info((
                "avg(Y)={0}, var(Y)={1}, std(Y)={2}, "
                "min(Y)={3}, max(Y)={4}, count(Y)={5}, "
                "NaNcount(Y)={6}"
            ).format(*stats[1]))

    # Load a model to add to or set up a new one
    if ('sanity-test' not in flags and 'no-save' not in flags
    and revision_id is not None
    and weights.weights_exist(model_id, revision_id)):
        logger.info(
            'Pre-trained weights found. Loading iteration {0}.'.format(
                revision_id
            )
        )
        model_nn = weights.load_weights(model_id, revision_id)
    else:
        logger.info(
            'Pre-trained weights not used. Building model from scratch.')
        input_shape = ds[0][0][0].shape
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
    if 'no-early-stopping' not in flags and hasattr(model, 'es_callback'):
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
    metrics = []
    if 'no-metrics' not in flags:
        logger.info('Evaluating model.')
        metrics = model_nn.evaluate(
            x=ds_test,
            verbose=0
        )
        model.metrics(metrics, logger)

    # Save model to weights directory
    if 'sanity-test' not in flags and 'no-save' not in flags:
        if revision_id is None:
            weights_id = weights.available(model_id, str(seed))
        else:
            weights_id = weights.available(model_id, revision_id)

        weights_path = weights.path(weights_id[0], weights_id[1])
        logger.info('Saving model to `{0}`.'.format(weights_path))
        model_nn.save(weights_path)

    return model_nn, metrics


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
