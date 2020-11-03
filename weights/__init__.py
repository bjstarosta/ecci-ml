# -*- coding: utf-8 -*-
"""Weights management module.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import errno
import re

import tensorflow as tf

import lib.logger


logger = lib.logger.logger
PATH_WEIGHTS = os.path.dirname(os.path.abspath(__file__))
FN_WEIGHTS = '{0}_{1}.h5'
FN_LOG = '{0}_{1}.log'

_FN_WEIGHTS_RE = re.sub(r'\{([0-9]+)\}', r'(?P<g\1>\\S+)', FN_WEIGHTS)
_FN_LOG_RE = re.sub(r'\{([0-9]+)\}', r'(?P<g\1>\\S+)', FN_LOG)


def path(model_id, iter_id, basename=False, fmt=FN_WEIGHTS):
    """Return a path to a file in the weights module with relevant identifiers.

    Args:
        model_id (str): Model identifier.
        iter_id (str): Iteration identifier.
        basename (bool): If False, an absolute path will be returned.
            If True, only the filename will be returned.
        fmt (str): Formattable string representing the filename.
            Must include two placeholders: {0} (model ident.)
            and {1} (iteration ident.).

    Returns:
        str: Path to file or full filename.

    """
    bn = fmt.format(model_id, iter_id)
    if basename is True:
        return bn
    else:
        return os.path.join(PATH_WEIGHTS, bn)


def available(model_id, iter_id='0'):
    """Return the next available iteration ID for the passed model identifier.

    Args:
        model_id (str): Model identifier.
        iter_id (str): Iteration identifier. Will use an incremental numeric
            system by default.

    Returns:
        tuple: Tuple containing the model ID and the iteration ID respectively.

    """
    decimal = iter_id.isdecimal()
    new_iter_id = iter_id
    i = 1
    while True:
        p = path(model_id, new_iter_id)
        if not os.path.exists(p):
            break
        # if decimal:
        #     new_iter_id = str(int(new_iter_id) + 1)
        # else:
        #     new_iter_id = str(iter_id) + '_' + str(i)
        #     i += 1
        new_iter_id = str(iter_id) + '_' + str(i)
        i += 1
    return model_id, new_iter_id


def load_weights(model_id, iter_id=None):
    """Load a previously trained model.

    Args:
        model_id (str): The identifier of the model used for generating
            these weights.
        iter_id (str): Iteration identifier. If not set, the latest
            iteration will be loaded.

    Returns:
        tensorflow.keras.Model: Trained Keras model.

    """
    if iter_id is None:
        iter_id = list_weights(model_id)[-1][1]

    p = path(model_id, iter_id)
    if os.path.exists(p):
        logger.info('Loading model from "{0}".'.format(p))
        return tf.keras.models.load_model(p)
    else:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), p)


def list_weights(model_id=None):
    """List all model iterations present in the directory.

    Args:
        model_id (str): Set to only list iterations belonging to a particular
            model.

    Returns:
        list: List of tuples of the form: (model id, iteration id).

    """
    lst = []
    ptn = re.compile(_FN_WEIGHTS_RE)
    for f in os.listdir(PATH_WEIGHTS):
        m = ptn.match(f)
        if m is None:
            continue
        if model_id is not None and m.group('g0') != model_id:
            continue
        lst.append((m.group('g0'), m.group('g1')))
    return sorted(lst, key=lambda k: k[0] + k[1])


def weights_exist(model_id, iter_id=None):
    """Check if a saved model with the given model and iteration exists.

    If no iteration identifier is given, the function will check if any
    iteration is present.

    Args:
        model_id (str): The identifier of the model.
        iter_id (str): Iteration identifier.

    Returns:
        bool: True if saved model exists, False otherwise.

    """
    if iter_id is not None:
        return os.path.exists(path(model_id, iter_id))
    else:
        lst = list_weights(model_id)
        if len(lst) > 0:
            return True
        else:
            return False


def save_weights(model, model_id, iter_id=None):
    """Save the weights of the passed model.

    Args:
        model (tensorflow.keras.model.Model): Trained Keras model.
        model_id (str): The identifier of the model used for generating
            these weights.
        iter_id (str): Iteration identifier. If not set, it will be generated
            to avoid overwriting already existing weight files.

    Returns:
        None

    """
    if iter_id is None:
        iter_id = available(model_id)[1]

    p = path(model_id, iter_id)
    logger.info('Saving model to `{0}`.'.format(p))
    model.save(p)
