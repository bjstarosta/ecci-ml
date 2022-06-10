# -*- coding: utf-8 -*-
"""Dataset preprocessing filters.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import numpy as np
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
