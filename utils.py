# -*- coding: utf-8 -*-
"""Training supervisor utility functions.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import logging

import click
import numpy as np
import tensorflow as tf
import skimage.external.tifffile as tifffile


def reset_tf_session():
    """Resets the TensorFlow session for Keras `backend`.

    Returns:
        tf.Session: New tensorflow session object.
    """
    tf.keras.backend.clear_session()

def setup_path(path):
    """Checks if passed directory exists, creates it if it doesn't."""
    if not os.path.isdir(path):
        os.mkdir(path)

def load_dataset(dataset, limit=0):
    """
    Loads an image dataset with train/test split and attributes into memory.

    Args:
        dataset (str):
            Path to the folder with the dataset.
        limit (int, optional):
            If set to a positive integer, will load only the first x amount
            of files in the directory.

    Returns:
        numpy.ndarray:
            Unsplit data for the model.
    """
    X = []

    if not os.path.isdir(dataset):
        raise IOError('"{0}": Not a directory.'.format(dataset))

    logging.info("Loading image dataset from '{0}'.".format(
        click.format_filename(dataset)
    ))

    images = os.listdir(dataset)
    if limit > 0:
        images = images[0:limit]

    images_valid = []
    for im in images:
        im_path = os.path.join(dataset, im)
        if not valid_image(im_path):
            continue
        images_valid.append(im_path)

    with click.progressbar(
        label='Loading images...',
        length=len(images_valid),
        show_pos=True
    ) as pbar:
        for im in images_valid:
            im = load_image(im_path)
            X.append(im)
            pbar.update(1)

    logging.info("{0:d} images loaded from '{1}'.".format(
        len(X),
        click.format_filename(dataset)
    ))

    return X

def valid_image(path):
    if not os.path.isfile(path):
        return False
    if not path.endswith('.tif'):
        return False
    return True

def load_image(path):
    with tifffile.TiffFile(path) as tif:
        return tif.asarray()

def save_image(path, img):
    tifffile.imsave(path, img)
