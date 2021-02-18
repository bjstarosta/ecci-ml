# -*- coding: utf-8 -*-
"""Dataset preprocessing tools.

Here go functions for preprocessing data prior to training, mainly for
preparing streaming pipelines for those large datasets.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import sys

import click
import numpy as np

if __name__ == '__main__':
    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import lib.utils as utils


def split_chunks(im, w, h, stride=1.):
    """Split an image into smaller chunks using a sliding window approach.

    Args:
        im (numpy.ndarray): Image to split.
        w (int): Desired chunk width.
        h (int): Desired chunk height.
        stride (float): Fraction of width and height to advance by per
            iteration.

    Returns:
        list: List of numpy.ndarray type images.

    """
    ret = []
    stride = (int(w * stride), int(h * stride))
    row, col = np.ogrid[0:h:1, 0:w:1]
    for r in np.arange(0, im.shape[0] - h, stride[1]):
        for c in np.arange(0, im.shape[1] - w, stride[0]):
            ret.append(im[row + r, col + c])
    return ret


def augment(im):
    """Augment passed image into 16 images.

    Args:
        im (numpy.ndarray): Image to augment.

    Returns:
        list: List of numpy.ndarray type images.

    """
    ret = [im]
    # rotate image by 90 degrees three times and save
    for i in range(0, 3):
        ret.append(np.rot90(ret[i]))
    # flip the saved images horizontally and save
    for i in range(0, 4):
        ret.append(np.fliplr(ret[i]))
    # flip the saved images vertically and save
    for i in range(0, 8):
        ret.append(np.flipud(ret[i]))
    return ret


def _load_images(path, type=None, mode=None):
    """Load an image or a directory of images into memory and return it.

    This is a generator function.

    Args:
        path (str): Path to load images from. Can be either a directory or
            an image file.

    Returns:
        str: Path to loaded image file.
        numpy.ndarray: Images in numpy.ndarray type.
    """
    if not os.path.isdir(path):
        if not utils.valid_image(path):
            raise IOError('"{0}": Not a valid image.'.format(path))
        yield path, utils.load_image(path, type, mode)
        return

    images = os.listdir(path)

    images_valid = []
    for im in images:
        im_path = os.path.join(path, im)
        if not utils.valid_image(im_path):
            continue
        images_valid.append(im_path)
    images_valid.sort()

    for im in images_valid:
        yield im, utils.load_image(im, type, mode)


@click.group()
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help="""Logs debug messages during script run."""
)
@click.option(
    '-f',
    '--file-log',
    is_flag=True,
    help="""Enables logging events to file. New log file will appear in the
        logs directory."""
)
@click.pass_context
def main(ctx, **kwargs):
    """Dataset preprocessing tools."""
    import logging
    import lib.logger

    logger = lib.logger.logger

    if kwargs['verbose'] is True:
        LOG_LEVEL = logging.DEBUG
    else:
        LOG_LEVEL = logging.INFO
    logger.setLevel(LOG_LEVEL)

    lib.logger.start_stream_log()

    if kwargs['file_log'] is True:
        lib.logger.start_file_log()

    ctx.obj['verbose'] = kwargs['verbose']
    ctx.obj['file_log'] = kwargs['file_log']
    ctx.obj['logger'] = logger


@main.command()
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
    '-w',
    '--width',
    type=int,
    required=True,
    help="""Width of the image chunk."""
)
@click.option(
    '-h',
    '--height',
    type=int,
    required=True,
    help="""Height of the image chunk."""
)
@click.option(
    '-s',
    '--stride',
    type=float,
    default=1.,
    show_default=True,
    help="""Proportion of width and height to advance the sliding window by."""
)
@click.pass_context
def split(ctx, **kwargs):
    """Split images into chunks of specified size."""
    ctx.obj['logger'].info("Operation begins: split")

    n = 0
    n2 = 0
    for im in _load_images(kwargs['input']):
        ctx.obj['logger'].debug(im[0])
        ext = os.path.splitext(os.path.basename(im[0]))

        out = split_chunks(im[1],
            kwargs['width'], kwargs['height'], kwargs['stride'])
        ctx.obj['logger'].debug("{0} chunks returned.".format(len(out)))

        for i, o in enumerate(out):
            path = os.path.join(kwargs['output'],
                ext[0] + '_' + str(i) + ext[1])
            utils.save_image(path, o)
            n2 += 1

        n += 1

    ctx.obj['logger'].info("{0} images processed.".format(n))
    ctx.obj['logger'].info("{0} new images created in '{1}'.".format(
        n2, kwargs['output']))


@main.command()
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
@click.pass_context
def augm(ctx, **kwargs):
    """Augment images for dataset expansion."""
    ctx.obj['logger'].info("Operation begins: augm")

    n = 0
    n2 = 0
    for im in _load_images(kwargs['input']):
        ctx.obj['logger'].debug(im[0])
        ext = os.path.splitext(os.path.basename(im[0]))

        out = augment(im[1])

        m = 0
        for o in out:
            path = os.path.join(kwargs['output'],
                ext[0] + '_' + str(m) + ext[1])
            utils.save_image(path, o)
            m += 1
            n2 += 1

        n += 1

    ctx.obj['logger'].info("{0} images processed.".format(n))
    ctx.obj['logger'].info("{0} new images created in '{1}'.".format(
        n2, kwargs['output']))


@main.command()
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
    '-m',
    '--mode',
    type=click.Choice(
        ['edge', 'reflect', 'symmetric', 'wrap'],
        case_sensitive=False
    ),
    default='symmetric',
    show_default=True,
    help="""Padding mode."""
)
@click.option(
    '-w',
    '--pad-width',
    type=int,
    nargs=2,
    required=True,
    help="""Width of the padding."""
)
@click.pass_context
def pad(ctx, **kwargs):
    """Pad images using chosen modes."""
    ctx.obj['logger'].info("Operation begins: pad")

    n = 0
    for im in _load_images(kwargs['input']):
        ctx.obj['logger'].debug(im[0])
        path = os.path.join(kwargs['output'], os.path.basename(im[0]))
        out = np.pad(im[1], kwargs['pad_width'], kwargs['mode'])
        utils.save_image(path, out)
        n += 1

    ctx.obj['logger'].info("{0} images processed.".format(n))


if __name__ == '__main__':
    main(obj={})
