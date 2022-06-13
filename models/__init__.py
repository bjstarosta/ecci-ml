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


def load_model(model_id):
    """Load a model schema using the given identifier.

    Loads the target file, runs the build() function and returns the result.
    The vars parameter gets unpacked as named arguments and passed to the
    build() function.

    Args:
        model_id (str): Model identifier.

    Returns:
        module: Python module of the loaded model.

    """
    if model_exists(model_id):
        logger.info('Loading model `{0}`.'.format(model_id))
        mod = importlib.import_module('models.' + model_id)
        return mod
    else:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), path(model_id))


def list_models(with_desc=False):
    """List all model schemas present in the directory.

    Args:
        with_desc (bool): If True, the returned list will be a list of tuples
            containing dataset ID, dataset __doc__.

    Returns:
        list: List of model identifiers.

    """
    lst = []
    for f in os.listdir(PATH_MODELS):
        if not f.endswith('.py'):
            continue
        if f in ['__init__.py']:
            continue
        f_ = os.path.splitext(f)[0]
        if with_desc is True:
            mod = importlib.import_module('models.' + f_)
            lst.append((f_, mod.__doc__))
        else:
            lst.append(f_)
    return sorted(lst)


def model_exists(model_id):
    """Check if a given model schema exists.

    Args:
        model_id (str): The identifier of the model.

    Returns:
        bool: True if model schema exists, False otherwise.

    """
    return os.path.exists(path(model_id))


def model_input_shape(model_mod):
    """Return shape of the input layer for a given model.

    Args:
        model_mod (module): Model module reference as given by load_model().
            Can also be a model_id string, in which case the module will be
            loaded within the function.

    Returns:
        tuple: Tuple of ints describing the shape of the input layer.

    """
    if type(model_mod) is str:
        model_mod = load_model(model_mod)

    mlmm = model_mod.build()
    return mlmm.get_layer(index=0)._batch_input_shape[1:]
