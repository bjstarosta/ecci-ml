# -*- coding: utf-8 -*-
"""Models module.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import errno
import importlib

import lib.logger


logger = lib.logger.logger
PATH_MODELS = os.path.dirname(os.path.abspath(__file__))


def path(model_id, basename=False):
    """Return a path to a model schema with relevant identifiers.

    Args:
        model_id (str): Model identifier.
        basename (bool): If False, an absolute path will be returned.
            If True, only the filename will be returned.

    Returns:
        str: Path to file or full filename.

    """
    bn = model_id + '.py'
    if basename is True:
        return bn
    else:
        return os.path.join(PATH_MODELS, bn)


def load_model(model_id, vars={}):
    """Load a model schema using the given identifier.

    Loads the target file, runs the build() function and returns the result.
    The vars parameter gets unpacked as named arguments and passed to the
    build() function.

    Args:
        model_id (str): Model identifier.
        vars (dict): Dictionary of arguments for the build function.
            See the model schema files for argument descriptions.

    Returns:
        tensorflow.keras.Model: Built Keras model.
        Python module of the loaded model.

    """
    if model_exists(model_id):
        logger.info('Loading model `{0}`.'.format(model_id))
        mod = importlib.import_module('models.' + model_id)
        return mod.build(**vars), mod
    else:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), path(model_id))


def list_models():
    """List all model schemas present in the directory.

    Returns:
        list: List of model identifiers.

    """
    lst = []
    for f in os.listdir(PATH_MODELS):
        if not f.endswith('.py'):
            continue
        if f in ['__init__.py']:
            continue
        lst.append(os.path.splitext(f)[0])
    return sorted(lst)


def model_exists(model_id):
    """Check if a given model schema exists.

    Args:
        model_id (str): The identifier of the model.

    Returns:
        bool: True if model schema exists, False otherwise.

    """
    return os.path.exists(path(model_id))
