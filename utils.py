# -*- coding: utf-8 -*-
"""Utility functions.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import errno
import logging

import click
import tensorflow as tf
import skimage.external.tifffile as tifffile


def setup_path(path):
    """Check if passed directory exists, create it if it doesn't.

    Args:
        path (str): Path to directory.

    """
    if not os.path.isdir(path):
        os.mkdir(path)


def load_dataset(dataset, limit=0):
    """Load an image dataset into memory.

    Args:
        dataset (str):
            Path to the folder with the dataset.
        limit (int, optional):
            If set to a positive integer, will load only the first x amount
            of files in the directory.

    Returns:
        numpy.ndarray: Unsplit data for the model.
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
    images_valid.sort()

    with click.progressbar(
        label='Loading images...',
        length=len(images_valid),
        show_pos=True
    ) as pbar:
        for im in images_valid:
            im = load_image(im)
            X.append(im)
            pbar.update(1)

    logging.info("{0:d} images loaded from '{1}'.".format(
        len(X),
        click.format_filename(dataset)
    ))

    return X


def load_model(path):
    """Load a previously trained model.

    Args:
        path (str): Path to file.

    Returns:
        tensorflow.keras.model.Model: Trained Keras model.

    """
    if os.path.exists(path):
        logging.info('Loading model from "{0}".'.format(path))
        return tf.keras.models.load_model(path)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)


def valid_image(path):
    """Determine whether passed path is an image type that we accept.

    Currently only tiff files.

    Args:
        path (str): Path to file.

    Returns:
        bool: True if file path is valid for reading, False otherwise.

    """
    if not os.path.isfile(path):
        return False
    if not path.endswith('.tif') and not path.endswith('.tiff'):
        return False
    return True


def load_image(path):
    """Read image from the passed path and return it in numpy array form.

    Args:
        path (str): Path to image file.

    Returns:
        numpy.ndarray: Image data.

    """
    with tifffile.TiffFile(path) as tif:
        return tif.asarray()


def save_image(path, img):
    """Save numpy array data as an image file in the passed path.

    Args:
        path (str): Path to image file.
        img (numpy.ndarray): Image data.

    """
    tifffile.imsave(path, img)
