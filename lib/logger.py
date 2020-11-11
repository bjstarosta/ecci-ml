# -*- coding: utf-8 -*-
"""Unified logger.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import datetime as dt
import logging


PATH_LOGS = os.path.dirname(os.path.abspath(__file__))
PATH_LOGS = os.path.abspath(os.path.join(PATH_LOGS, '../logs'))
PATH_CUR_LOG = 'log_{0}.log'.format(
    dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
PATH_CUR_LOG = os.path.join(PATH_LOGS, PATH_CUR_LOG)


logger = logging.getLogger('main-log')
logger.setLevel(logging.INFO)


def start_stream_log():
    """Set up the stream log handler."""
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    c_handler.setFormatter(logging.Formatter(
        '[%(levelname)s] %(asctime)s - %(message)s'))
    logger.addHandler(c_handler)


def start_file_log():
    """Set up the file log handler."""
    f_handler = logging.FileHandler(PATH_CUR_LOG)
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(logging.Formatter(
        '[%(asctime)s - %(levelname)s] %(message)s'))
    logger.addHandler(f_handler)


def tensorboard_log_path(model_id):
    """Returns a path to the log directory for the TensorBoard callback.

    If the directory doesn't exist, it is created.

    Args:
        model_id (str): Model identifier.

    Returns:
        str: Absolute path to directory.

    """
    path = os.path.join(PATH_LOGS, model_id)
    if not os.path.isdir(path):
        os.mkdir(path)
    return path
