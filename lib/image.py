# -*- coding: utf-8 -*-
"""Image filtering functions.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import math

import imageio
import numpy as np
import scipy.ndimage as ndi
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
    # print(img.shape, img[0][0])

    if mode == 'grayscale' or mode == 'gs':
        if len(img.shape) == 3 and img.shape[2] == 2:
            img = img[..., 0]
        if len(img.shape) == 3 and img.shape[2] >= 3:
            img = np.dot(img[..., :3], rgb2gs)
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = np.squeeze(img, axis=2)
        if len(img.shape) != 2:
            raise RuntimeError('Could not convert image to grayscale')

    elif mode == 'grayscale1c' or mode == 'gs1c':
        if len(img.shape) == 3 and img.shape[2] == 2:
            img = img[..., 0]
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


def crop_image(im, x1, y1, x2, y2):
    """Crop the image according to specified pixel counts at each edge.

    Args:
        im (numpy.ndarray): Input image.
        x1 (int): Left edge crop.
        y1 (int): Top edge crop.
        x2 (int): Right edge crop.
        y2 (int): Bottom edge crop.

    Returns:
        numpy.ndarray: Cropped image.

    """
    if np.min([x1, y1, x2, y2, 0]) < 0:
        raise RuntimeError(
            'Negative pixel values: {0}'.format((x1, y1, x2, y2))
        )
    return im[y1:im.shape[0] - y2, x1:im.shape[1] - x2]


def slice_image(im, c, padmode='black'):
    """Return a slice of the image array according to the specified coords.

    Unlike using pure numpy slicing, this will attempt to retrieve the array
    slice even if the window lies partially outside of the array boundaries,
    instead of returning an indexing error. Missing data will be filled by
    padding. If the slice window lies completely outside of the image
    boundaries (i.e. there is no overlap), an error will be returned.

    Args:
        im (numpy.ndarray): Input image.
        c (tuple): A tuple of ints specifying the coordinates of the
            slice, of the format (x, y, width, height).
        padmode (str): If coords lie in part outside of the image coordinates,
            this specifies the padding to apply to the undefined areas.

    Returns:
        numpy.ndarray: Slice of the original image.

    """
    # Convert cartesian to rc coords
    c1 = int(c[0])
    r1 = int(c[1])
    c2 = int(c[0] + c[2])
    r2 = int(c[1] + c[3])

    # If slice is fully within bounds of the image, just return the slice
    if r1 >= 0 and c1 >= 0 and r2 <= im.shape[0] and c2 <= im.shape[1]:
        return im[r1:r2, c1:c2]

    # If slice is fully outside of bounds of the image, return error
    if r1 > im.shape[0] or c1 > im.shape[1] or r2 < 0 or c2 < 0:
        raise RuntimeError(
            'Slice fully outside of image boundaries: {0}'.format(c)
        )

    # If we are here, slice is partially overlapping image
    # Find needed padding size
    bpw = (np.max([r2, im.shape[0]]) - np.min([r1, 0])) - im.shape[0]
    bph = (np.max([c2, im.shape[1]]) - np.min([c1, 0])) - im.shape[1]
    p = np.max([bpw, bph])
    p_arg = [(p, p), (p, p)]

    # Account for extra dimension in multichannel case
    if len(im.shape) > 2:
        p_arg.append((0, 0))

    # Handle pad modes
    if padmode == 'black':
        im = np.pad(im, p_arg, 'constant')
    else:
        im = np.pad(im, p_arg, padmode)

    return im[r1 + p:r2 + p, c1 + p:c2 + p]


def sliding_window_2d(
    imsz, wsz, st, origin='lefttop', cutoff=False
):
    """Return a set of window coordinates according to passed parameters.

    The returned coordinates can be input directly into slice_image() to
    obtain an image slice.

    Args:
        imsz (tuple): Image size, or sliding window boundaries, in cartesian
            coordinates, i.e. (width, height).
        wsz (tuple): Size of the sliding window in cartesian coordinates,
            i.e. (width, height).
        st (tuple): Stride. How many pixels to advance on each iteration,
            in cartesian coordinates, i.e. (x, y).
        origin (str): Determines how the sliding window grid will be placed
            on the image. Accepts values:
                lefttop, middle.
        cutoff (bool): If image size is not a multiple of window size, some
            window coordinates may lie outside of the boundary of the image.
            If this parameter is set to True, only windows fully inside the
            image boundary will be returned, but parts of the image may not
            be iterated over. If set to False, some returned values may be
            negative or greater than the boundary specified in imsz.

    Returns:
        tuple: A tuple of ints specifying window coordinates, of the format
            (x, y, width, height).

    """
    # Create ranges for left top origin.
    rr = np.arange(0, imsz[1], st[1])
    cr = np.arange(0, imsz[0], st[0])

    # Transform ranges according to chosen origin.
    swbbox = (cr[-1] + wsz[0], rr[-1] + wsz[1])
    if origin == 'middle':
        rr += int((imsz[1] - swbbox[1]) / 2)
        cr += int((imsz[0] - swbbox[0]) / 2)

    elif origin != 'lefttop':
        raise RuntimeError(
            'Unexpected origin value: {0}'.format(origin)
        )

    # Iterate and return coordinates
    for r in rr:
        if cutoff and (r + wsz[1] > imsz[1] or r < 0):
            continue
        for c in cr:
            if cutoff and (c + wsz[0] > imsz[0] or c < 0):
                continue
            yield (c, r, wsz[0], wsz[1])


def rotate_image(im, rad, mode='constant'):
    """Rotate an image clockwise according to given radians.

    Note that rotation by non right angles will involve interpolation, and
    will enlarge the original image.

    Args:
        im (numpy.ndarray): Input image.
        rad (float): Rotate the image by this many radians.
        mode (str): This is passed to the rotate function.
            See scipy.ndimage.rotate()

    Returns:
        numpy.ndarray: Rotated image array.

    """
    rad = rad % (2 * np.pi)

    if rad == 0:
        return im
    if rad == 0.5 * np.pi:
        return np.rot90(im, 1)
    if rad == np.pi:
        return np.rot90(im, 2)
    if rad == 1.5 * np.pi:
        return np.rot90(im, 3)

    deg = rad * (180 / np.pi)
    return ndi.rotate(im, angle=deg, mode=mode)
