# -*- coding: utf-8 -*-
"""Dataset preprocessing filters.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import numpy as np
import skimage.draw
import skimage.feature

import lib.image as image


def lightness(im, values):
    """Change the lightness of the passed image.

    Only supports uint8 grayscale images.

    Args:
        im (numpy.ndarray): Image to modify.
        values (list): List of ints defining values to add to or
            subtract from image.

    Returns:
        list: List of modified images.

    """
    out = []
    for val in values:
        oim = np.copy(im)
        if val >= 0:
            lim = 255 - val
            oim[oim > lim] = 255
            oim[oim <= lim] += val
        else:
            nv = -val
            lim = 0 + nv
            oim[oim < lim] = 0
            oim[oim >= lim] -= nv
        out.append(('_l'+str(val), oim))

    return out


def gamma(im, values):
    """Gamma transform the passed image.

    Only supports uint8 grayscale images.

    Args:
        im (numpy.ndarray): Image to modify.
        values (list): List of gamma values.

    Returns:
        list: List of modified images.

    """
    out = []
    for val in values:
        oim = np.copy(im)

        invgamma = 1.0 / val
        lut = (((np.arange(0, 256) / 255.0) ** invgamma) * 255).astype('uint8')
        oim = lut[oim]

        out.append(('_g'+str(val), oim))

    return out


def disk_redraw(im, values):
    """Redraw blobs programmatically to ensure uniformity.

    Only supports uint8 grayscale images.

    Args:
        im (numpy.ndarray): Image to modify.
        values (dict): List of blob detection and drawing options.

    Returns:
        list: List of modified images.

    """
    def_values = {
        'min_r': 5,
        'max_r': 14,
        'min_sigma': 3,
        'max_sigma': 15,
        'num_sigma': 15,
        'threshold': .1,
        'type': 'disk',
        'radius': 5
    }
    values = {**def_values, **values}

    # blob_log performance on 3D arrays is abysmal, so this is necessary
    im = image.convmode(im, 'gs')

    blobs = skimage.feature.blob_log(im,
        min_sigma=values['min_sigma'], max_sigma=values['max_sigma'],
        num_sigma=values['num_sigma'], threshold=values['threshold']
    )
    blobs[:, 2] = np.clip(
        blobs[:, 2] * np.sqrt(2), values['min_r'], values['max_r']
    )

    ret = np.zeros(im.shape)
    for y, x, r in blobs:
        if values['type'] == 'disk':
            rr, cc = skimage.draw.disk(
                (y, x), values['radius'],
                shape=im.shape
            )
            ret[rr, cc] = 1
        elif type == 'square':
            rr, cc = skimage.draw.rectangle(
                (y - values['radius'], x - values['radius']),
                (y + values['radius'], x + values['radius']),
                shape=im.shape
            )
            ret[rr, cc] = 1

    ret = (ret * 255).astype('uint8')
    return [('_', ret)]
