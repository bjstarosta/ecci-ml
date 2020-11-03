# -*- coding: utf-8 -*-
"""Utility functions.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import datetime
try:
    import skimage.external.tifffile as tifffile
except ImportError:
    import tifffile


def generate_seed():
    """Return a semi-unique number for seeding a random state.

    Returns:
        int

    """
    return int(datetime.datetime.utcnow().strftime('%Y%m%d'))


def setup_path(path):
    """Check if passed directory exists, create it if it doesn't.

    Args:
        path (str): Path to directory.

    """
    if not os.path.isdir(path):
        os.mkdir(path)


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
    _, ext = os.path.splitext(path.lower())
    if ext != '.tif' and ext != '.tiff':
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
