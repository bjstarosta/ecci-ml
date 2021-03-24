# -*- coding: utf-8 -*-
"""Utility functions.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import datetime

import numpy as np
import imageio
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
    if ext != '.tif' and ext != '.tiff' and ext != '.png':
        return False
    return True


def load_image(path, type=None, mode=None):
    """Read image from the passed path and return it in numpy array form.

    Automatically performs channel mode conversions of the loaded image.

    Args:
        path (str): Path to image file.
        type (str): Value type to return the image as. See image_convtype()
            for documentation of accepted values.
        mode (str): Channel mode to return the image as. See image_convmode()
            for documentation of accepted values.

    Returns:
        numpy.ndarray: Image data.

    """
    if not os.path.isfile(path):
        return None
    _, ext = os.path.splitext(path.lower())

    # Image read
    if ext == '.tif' or ext == '.tiff':
        with tifffile.TiffFile(path) as tif:
            out = tif.asarray()
    elif ext == '.png':
        out = imageio.imread(path)
    else:
        return None

    if out is None:
        raise RuntimeError('File could not be loaded: {0}'.format(path))

    out = image_convmode(out, mode)
    out = image_convtype(out, type)
    return out


def save_image(path, img, type=None, mode=None):
    """Save numpy array data as an image file in the passed path.

    Args:
        path (str): Path to image file.
        img (numpy.ndarray): Image data.
        type (str): Value type to save the image with. See image_convtype()
            for documentation of accepted values.
        mode (str): Channel mode to save the image with. See image_convmode()
            for documentation of accepted values.

    Returns:
        None

    """
    img = image_convmode(img, mode)
    img = image_convtype(img, type)

    _, ext = os.path.splitext(path.lower())
    if ext == '.tif' or ext == '.tiff':
        tifffile.imsave(path, img)
    elif ext == '.png':
        imageio.imsave(path, img)
    else:
        raise RuntimeError('Type unsupported: {0}'.format(ext))


def image_convtype(img, type):
    """Convert image data type.

    Args:
        img (numpy.ndarray): Image data.
        type (str): Value type to return the image as. Available types:
            - 'uint8': returns a range of integer values (0...255).
            - 'float32': returns a range of single precision floating point
                values (0...1).
            - None: returns the image as it was loaded.

    Returns:
        numpy.ndarray: Type converted image data.

    """
    if type == 'uint8':
        if np.issubdtype(img.dtype, np.floating):
            img = img * 255.0
        img = img.astype('uint8')
    elif type == 'float32':
        if np.issubdtype(img.dtype, np.integer):
            img = img.astype('float32') / 255.0
        img = img.astype('float32')
    elif type is not None:
        raise RuntimeError('Passed type is unsupported')
    return img


def image_convmode(img, mode):
    """Convert image channel mode.

    Args:
        img (numpy.ndarray): Image data.
        mode (str): Channel mode to return the image as. Available modes:
            - 'grayscale', 'gs': returns a 2-dim array with values directly in
                the columns.
            - 'grayscale1c', 'gs1c': returns a 3-dim array with each column
                containing a vector with a single value.
            - 'rgb': returns a 3-dim array with each column containing a
                vector with three values.
            - 'rgba': returns a 3-dim array with each column containing a
                vector with four values.
            - None: returns the image as it was loaded.

    Returns:
        numpy.ndarray: Mode converted image data.

    """
    rgb2gs = [0.2989, 0.5870, 0.1140]

    if mode == 'grayscale' or mode == 'gs':
        if len(img.shape) == 3 and img.shape[2] >= 3:
            img = np.dot(img[..., :3], rgb2gs)
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = np.squeeze(img, axis=2)
        if len(img.shape) != 2:
            raise RuntimeError('Could not convert image to grayscale')

    elif mode == 'grayscale1c' or mode == 'gs1c':
        if len(img.shape) == 3 and img.shape[2] >= 3:
            img = np.dot(img[..., :3], rgb2gs)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)

    elif mode == 'rgb':
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[..., :3]
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = np.squeeze(img, axis=2)
        if len(img.shape) == 2:
            img = np.stack((img, img, img), axis=-1)

    elif mode == 'rgba':
        if np.issubdtype(img.dtype, np.integer):
            mul = 255
        elif np.issubdtype(img.dtype, np.floating):
            mul = 1.

        if len(img.shape) == 3 and img.shape[2] == 3:
            img = np.stack((img[..., 0], img[..., 1], img[..., 2],
                np.ones(img.shape) * mul), axis=-1)
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = np.squeeze(img, axis=2)
        if len(img.shape) == 2:
            img = np.stack((img, img, img, np.ones(img.shape) * mul), axis=-1)

    elif mode is not None:
        raise RuntimeError('Passed mode is unsupported')

    return img
