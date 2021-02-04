# -*- coding: utf-8 -*-
"""Utility functions.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import datetime

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


def load_image(path):
    """Read image from the passed path and return it in numpy array form.

    Args:
        path (str): Path to image file.

    Returns:
        numpy.ndarray: Image data.

    """
    if not os.path.isfile(path):
        return None
    _, ext = os.path.splitext(path.lower())

    if ext == '.tif' or ext == '.tiff':
        with tifffile.TiffFile(path) as tif:
            return tif.asarray()
    elif ext == '.png':
        return imageio.imread(path)
    else:
        return None


def save_image(path, img):
    """Save numpy array data as an image file in the passed path.

    Args:
        path (str): Path to image file.
        img (numpy.ndarray): Image data.

    """
    _, ext = os.path.splitext(path.lower())

    if ext == '.tif' or ext == '.tiff':
        tifffile.imsave(path, img)
    elif ext == '.png':
        imageio.imsave(path, img)


class ImageSequence(object):

    def __init__(self, pathlist):
        self.lst = pathlist

    def __getitem__(self, index):
        """Get item at specified index."""
        return self.lst[index]

    def __len__(self):
        """Return length of sequence."""
        return len(self.lst)

    def __iter__(self):
        """Create a generator that iterates over the sequence."""
        for item in (self[i] for i in range(len(self))):
            yield load_image(item)
