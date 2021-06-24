# -*- coding: utf-8 -*-
"""Sine cosine filter apply.

See: https://www.sciencedirect.com/science/article/pii/S0030401899001169

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import sys

import click
import numpy as np
import scipy as sp

if __name__ == '__main__':
    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    )

import lib.image as image
from datasets.tools import _load_images


def sine_cosine_filter(im, kernel_size):
    out = np.arctan2(
        sp.ndimage.filters.uniform_filter(np.sin(im), size=kernel_size),
        sp.ndimage.filters.uniform_filter(np.cos(im), size=kernel_size)
    )
    return out


@click.command()
@click.option(
    '-i',
    '--input',
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True)
)
@click.option(
    '-o',
    '--output',
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=True)
)
@click.option(
    '-ks',
    '--kernel-size',
    type=int,
    required=True,
    help="""Size of the filter kernel."""
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help="""Logs debug messages during script run."""
)
@click.pass_context
def main(ctx, **kwargs):
    """Pass an image through a sine-cosine filter of a desired kernel size.

    This can be applied to a series of images by pointing the --input and
    --output parameters to a directory containing valid images."""
    import logging
    import lib.logger

    logger = lib.logger.logger

    if kwargs['verbose'] is True:
        LOG_LEVEL = logging.DEBUG
    else:
        LOG_LEVEL = logging.INFO
    logger.setLevel(LOG_LEVEL)

    lib.logger.start_stream_log()

    ctx.obj['verbose'] = kwargs['verbose']
    ctx.obj['logger'] = logger

    ctx.obj['logger'].info("Operation begins: sine_cosine_filter")

    n = 0
    for im in _load_images(kwargs['input']):
        ctx.obj['logger'].debug(im[0])

        out = image.convtype(im[1], 'float32')
        out = sine_cosine_filter(out, kwargs['kernel_size'])
        out = image.convtype(out, 'uint8')

        path = os.path.join(kwargs['output'], os.path.basename(im[0]))
        image.save_image(path, out)
        n += 1

    ctx.obj['logger'].info("{0} new images created in '{1}'.".format(
        n, kwargs['output']))


if __name__ == '__main__':
    main(obj={})
