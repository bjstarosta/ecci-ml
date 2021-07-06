# -*- coding: utf-8 -*-
"""Image filtering functions.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os

import imageio
import numpy as np
import cv2
try:
    import skimage.external.tifffile as tifffile
except ImportError:
    import tifffile


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
        type (str): Value type to return the image as. See convtype()
            for documentation of accepted values.
        mode (str): Channel mode to return the image as. See convmode()
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

    out = convmode(out, mode)
    out = convtype(out, type)
    return out


def save_image(path, img, type=None, mode=None):
    """Save numpy array data as an image file in the passed path.

    Args:
        path (str): Path to image file.
        img (numpy.ndarray): Image data.
        type (str): Value type to save the image with. See convtype()
            for documentation of accepted values.
        mode (str): Channel mode to save the image with. See convmode()
            for documentation of accepted values.

    Returns:
        None

    """
    img = convmode(img, mode)
    img = convtype(img, type)

    _, ext = os.path.splitext(path.lower())
    if ext == '.tif' or ext == '.tiff':
        tifffile.imsave(path, img)
    elif ext == '.png':
        imageio.imsave(path, img)
    else:
        raise RuntimeError('Type unsupported: {0}'.format(ext))


def convtype(img, type):
    """Convert image data type.

    Args:
        img (numpy.ndarray): Image data.
        type (str): Value type to return the image as. Available types:
            - 'uint8': returns a range of integer values (0...255).
            - 'float32': returns a range of single precision floating point
                values (0...1).
            - 'float64': returns a range of double precision floating point
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
    elif type == 'float64':
        if np.issubdtype(img.dtype, np.integer):
            img = img.astype('float64') / 255.0
        img = img.astype('float64')
    elif type is not None:
        raise RuntimeError('Passed type is unsupported')
    return img


def convmode(img, mode):
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


def fscale(img, a, b, min=None, max=None):
    """Min-max normalises an array containing image data.

    See: https://en.wikipedia.org/wiki/Feature_scaling

    Args:
        img (numpy.ndarray): Array containing input image data.
        a (float): Minimum value.
        b (float): Maximum value.
        min (float): Minimum range for input image. If unset, the lowest
            pixel value will be used.
        max (float): Maximum range for input image. If unset, the highest
            pixel value will be used.

    Returns:
        numpy.ndarray: Array containing normalised image data.

    """
    img = np.asarray(img, dtype=np.float64)
    if min is None:
        min = np.min(img)
    if max is None:
        max = np.max(img)
    img = a + ((img - min) * (b - a)) / (max - min)
    return img


def bg_removal(im, fgblur=3, bgblur=21):
    """Remove background colour gradients from ECCI image containing TDs.

    Default values for the filter sizes should work on all ECCI images.

    Args:
        im (numpy.ndarray): Input image. Needs to be in uint8 format.
        fgblur (int): Foreground filter size.
        bgblur (int): Background filter size.

    Returns:
        numpy.ndarray: Output image in float64 format in the [0, 1] domain.

    """
    image_fg = cv2.medianBlur(im, fgblur).astype(np.float64)
    image_bg = cv2.medianBlur(im, bgblur).astype(np.float64)
    im_nobg = image_fg - image_bg
    im_nobg = fscale(im_nobg, 0, 1)
    # print(np.min(im_nobg), np.max(im_nobg), im_nobg.dtype)
    return im_nobg
