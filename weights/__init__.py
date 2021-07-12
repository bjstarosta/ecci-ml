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
FN_WEIGHTS_NAME = '{0}_{2}_{1}.h5'
FN_LOG = '{0}_{1}.log'

_FN_WEIGHTS_RE = re.compile(
    re.sub(r'\{([0-9]+)\}', r'(?P<g\1>\\S+?)', FN_WEIGHTS))
_FN_WEIGHTS_NAME_RE = re.compile(
    re.sub(r'\{([0-9]+)\}', r'(?P<g\1>\\S+?)', FN_WEIGHTS_NAME))
_FN_LOG_RE = re.compile(
    re.sub(r'\{([0-9]+)\}', r'(?P<g\1>\\S+?)', FN_LOG))


def path(model_id, iter_id, name=None, basename=False):
    """Return a path to a file in the weights module with relevant identifiers.

    Args:
        model_id (str): Model identifier.
        iter_id (str): Iteration identifier.
        name (str): Custom identifier.
        basename (bool): If False, an absolute path will be returned.
            If True, only the filename will be returned.

    Returns:
        str: Path to file or full filename.

    """
    if name is None:
        fmt = FN_WEIGHTS
    else:
        fmt = FN_WEIGHTS_NAME
    bn = fmt.format(model_id, iter_id, name)
    if basename is True:
        return bn
    else:
        return os.path.join(PATH_WEIGHTS, bn)


def path_decode(path):
    """Decode a path to a valid weights file into relevant identifiers.

    Args:
        path (str): Path to file or full filename.

    Returns:
        tuple: Tuple consisting of the model ID and iteration ID.
            Function can also return None if the path was not valid.

    """
    rel = [_FN_WEIGHTS_NAME_RE, _FN_WEIGHTS_RE]
    for re_i in rel:
        m = re_i.match(os.path.basename(path))
        if m is None:
            continue
        mgd = m.groupdict()
        if 'g2' not in mgd.keys():
            mgd['g2'] = None
        return (mgd['g0'], mgd['g1'], mgd['g2'])
    return None


def available(model_id, iter_id='0', name=None):
    """Return the next available iteration ID for the passed model identifier.

    Args:
        model_id (str): Model identifier.
        iter_id (str): Iteration identifier.
        name (str): Custom identifier. Will use an incremental numeric
            system by default.

    Returns:
        tuple: Tuple containing the model ID, iteration ID and name
            respectively.

    """
    new_name = name
    i = 1
    while True:
        p = path(model_id, iter_id, new_name)
        if not os.path.exists(p):
            break
        if name is None:
            new_name = str(i)
        else:
            new_name = str(name) + '-' + str(i)
        i += 1
    return model_id, iter_id, new_name


def load_weights(model_id, iter_id=None, name=None):
    """Load a previously trained model.

    Args:
        model_id (str): The identifier of the model used for generating
            these weights.
        iter_id (str): Iteration identifier. If not set, the latest
            iteration will be loaded.
        name (str): Custom identifier.

    Returns:
        tensorflow.keras.Model: Trained Keras model.

    """
    if iter_id is None:
        iter_id = list_weights(model_id)[-1][1]

    p = path(model_id, iter_id, name)
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
        list: List of tuples of the form: (model id, iteration id, [name]).

    """
    lst = []
    for f in os.listdir(PATH_WEIGHTS):
        m = path_decode(f)
        if m is None or (model_id is not None and m[0] != model_id):
            continue
        lst.append(m)
    return sorted(lst, key=lambda k: k[0] + k[1])


def weights_exist(model_id, iter_id=None, name=None):
    """Check if a saved model with the given model and iteration exists.

    If no iteration identifier is given, the function will check if any
    iteration is present.

    Args:
        model_id (str): The identifier of the model.
        iter_id (str): Iteration identifier.
        name (str): Custom identifier.

    Returns:
        bool: True if saved model exists, False otherwise.

    """
    if iter_id is not None:
        return os.path.exists(path(model_id, iter_id, name))
    else:
        lst = list_weights(model_id)
        if len(lst) > 0:
            return True
        else:
            return False


def save_weights(model, model_id, iter_id=None, name=None):
    """Save the weights of the passed model.

    Args:
        model (tensorflow.keras.model.Model): Trained Keras model.
        model_id (str): The identifier of the model used for generating
            these weights.
        iter_id (str): Iteration identifier. If not set, it will be generated
            to avoid overwriting already existing weight files.
        name (str): Custom identifier. If not set, it will be generated
            to avoid overwriting already existing weight files.

    Returns:
        None

    """
    if iter_id is None or name is None:
        _, iter_id, name = available(model_id)

    p = path(model_id, iter_id, name)
    logger.info('Saving model to `{0}`.'.format(p))

    if not isinstance(model, tf.keras.Model):
        raise RuntimeError(
            'Passed model is not an instance of tf.keras.Model.'
        )
    model.save(p)
