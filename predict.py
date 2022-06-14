# -*- coding: utf-8 -*-
"""Prediction interface.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import logging

import click
import numpy as np
import matplotlib.pyplot as plt

import lib.logger
import lib.tf
import lib.utils
import lib.image
import models
import weights


logger = lib.logger.logger
lib.logger.start_stream_log()


_predict_click_options = [
    click.argument('model', nargs=1, type=str),
    click.argument('iteration', nargs=1, type=str),
    click.option(
        '-n',
        '--name',
        type=str,
        default=None,
        help="""Custom model name to add to the filename when saving trained
            weights."""
    ),
    click.option(
        '-s',
        '--stride',
        type=float,
        default=1,
        help="""Sliding window stride ratio when predicting images. Set to
            less than 1 for potentially better results at the cost of speed."""
    ),
    click.option(
        '-c',
        '--crop',
        type=int,
        default=0,
        help="""Number of pixels to crop from the edge of the image."""
    ),
    click.option(
        '--compare',
        is_flag=True,
        help="""Show image and prediction comparison in matplotlib."""
    )
]


def predict_click_options(func):
    for option in reversed(_predict_click_options):
        func = option(func)
    return func


def image_to_nparray(im, ishape, stride=1):
    imsz = (im.shape[1], im.shape[0])
    swsz = (ishape[1], ishape[0])
    s = (int(ishape[1] * stride), int(ishape[0] * stride))

    ret = []
    for sw in lib.image.sliding_window_2d(imsz, swsz, s, 'middle'):
        ret.append(lib.image.slice_image(im, sw, 'reflect'))

    return np.stack(ret, axis=0)


def nparray_to_image(arr, imshape, ishape, stride=1):
    imsz = (imshape[1], imshape[0])
    swsz = (ishape[1], ishape[0])
    s = (int(ishape[1] * stride), int(ishape[0] * stride))

    max_imsz = (imsz[0] + swsz[0], imsz[1] + swsz[1])
    retsz = (
        int(np.ceil((max_imsz[0] / s[0]) * s[0])),
        int(np.ceil((max_imsz[1] / s[1]) * s[0]))
    )

    ret = np.zeros((retsz[1], retsz[0]), dtype=arr.dtype)
    for i, sw in enumerate(
        lib.image.sliding_window_2d(retsz, swsz, s, 'middle', cutoff=True)
    ):
        ret[sw[1]:(sw[1] + sw[3]), sw[0]:(sw[0] + sw[2])] |= arr[i]

    if imsz != retsz:
        left = (retsz[0] / 2) - (imsz[0] / 2)
        top = (retsz[1] / 2) - (imsz[1] / 2)
        right = retsz[0] - (left + imsz[0])
        bottom = retsz[1] - (top + imsz[1])
        crop = (
            int(left) if left >= 0 else 0,
            int(top) if top >= 0 else 0,
            int(right) if right >= 0 else 0,
            int(bottom) if bottom >= 0 else 0
        )
        ret = lib.image.crop_image(ret, *crop)

    return ret


def mpl_comparison(im, pred, fname, mname):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 12))
    fig.suptitle('Prediction: {0} (model: {1})'.format(
        fname, mname
    ), fontsize=16)
    ax.imshow(im, cmap=plt.cm.gray)
    ax.imshow(pred, cmap=plt.cm.jet, alpha=0.5)
    plt.show()


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
    """Prediction supervisor."""
    if kwargs['verbose'] is True:
        LOG_LEVEL = logging.DEBUG
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
    else:
        LOG_LEVEL = logging.INFO
    logger.setLevel(LOG_LEVEL)

    if kwargs['file_log'] is True:
        lib.logger.start_file_log()

    ctx.obj['verbose'] = kwargs['verbose']
    ctx.obj['file_log'] = kwargs['file_log']


@main.command()
@predict_click_options
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
    type=click.Path(exists=True, file_okay=True, dir_okay=True, writable=True)
)
@click.pass_context
def image_dir(ctx, **kwargs):
    """Predict data using the selected model on a set of images."""
    if not models.model_exists(kwargs['model']):
        raise click.UsageError(
            "Model '{0}' does not exist.".format(kwargs['model']),
            ctx=ctx
        )

    if not weights.weights_exist(
        kwargs['model'],
        kwargs['iteration'],
        kwargs['name']
    ):
        raise click.UsageError(
            ("Model '{0}' (iteration: '{1}', name: '{2}') "
            "does not exist.").format(
                kwargs['model'], kwargs['iteration'], kwargs['name']),
            ctx=ctx
        )

    ctx.obj['model'] = kwargs['model']
    ctx.obj['weights'] = (kwargs['model'], kwargs['iteration'], kwargs['name'])

    logger.info('Scanning for images in `{0}`.'.format(kwargs['input']))
    lst = []
    for f in os.listdir(kwargs['input']):
        f_path = os.path.join(kwargs['input'], f)
        if not lib.image.valid_image(f_path):
            continue
        lst.append(f_path)
    lst.sort()
    logger.info('Found {0} images in directory.'.format(len(lst)))

    X = []
    with click.progressbar(
        label='Loading images...',
        length=len(lst),
        show_pos=True
    ) as pbar:
        for im in lst:
            im = lib.image.load_image(im)
            X.append(im)
            pbar.update(1)

    X = np.array(X)

    logger.info('Prediction starts.')
    logger.debug("Y.shape={0}".format(Y.shape))

    try:
        Y = lib.tf.predict(X, ctx.obj['model'], ctx.obj['weights'])
    except Exception:
        logger.error("Unrecoverable error.", exc_info=True)
        exit(1)

    logger.debug("min(Y)={0}, max(Y)={1}, avg(Y)={2}, var(Y)={3}".format(
        np.min(Y), np.max(Y), np.average(Y), np.var(Y)
    ))
    logger.debug("Y.shape={0}, Y.dtype={1}".format(Y.shape, Y.dtype))

    with click.progressbar(
        label='Saving images...',
        length=len(Y),
        show_pos=True
    ) as pbar:
        i = 0
        for im in Y:
            im = lib.image.save_image(
                os.path.join(kwargs['output'], os.path.basename(lst[i])), im
            )
            pbar.update(1)
            i += 1

    logger.info('Completed predictions on {0} images.'.format(len(Y)))


@main.command()
@predict_click_options
@click.option(
    '-i',
    '--input',
    required=True,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True
    )
)
@click.option(
    '-o',
    '--output',
    required=True,
    type=click.Path(
        exists=False, file_okay=True, dir_okay=False, writable=True
    )
)
@click.pass_context
def image(ctx, **kwargs):
    """Predict data using the selected model on an image."""
    if not models.model_exists(kwargs['model']):
        raise click.UsageError(
            "Model '{0}' does not exist.".format(kwargs['model']),
            ctx=ctx
        )

    if not weights.weights_exist(kwargs['model'], kwargs['iteration']):
        raise click.UsageError(
            "Model '{0}' (iteration: '{1}') does not exist.".format(
                kwargs['model'], kwargs['iteration']),
            ctx=ctx
        )

    ctx.obj['model'] = kwargs['model']
    ctx.obj['weights'] = (kwargs['model'], kwargs['iteration'])

    if not lib.image.valid_image(kwargs['input']):
        raise click.UsageError(
            "File '{0}' is not a valid image.".format(
                kwargs['input']),
            ctx=ctx
        )

    im = lib.image.load_image(kwargs['input'])

    if kwargs['crop'] > 0:
        im = lib.image.crop_image(im, *([kwargs['crop']] * 4))

    input_shape = models.model_input_shape(ctx.obj['model'])

    if im.shape != input_shape:
        logger.info(('Image shape different from input shape. '
            'Sliding window enabled.'))
        X = image_to_nparray(im, input_shape, stride=kwargs['stride'])
        swenabled = True
    else:
        X = np.array([im])

    logger.info('Prediction starts.')

    try:
        Y = lib.tf.predict(X, ctx.obj['model'], ctx.obj['weights'])
    except Exception:
        logger.error("Unrecoverable error.", exc_info=True)
        exit(1)

    logger.debug("min(Y)={0}, max(Y)={1}, avg(Y)={2}, var(Y)={3}".format(
        np.min(Y), np.max(Y), np.average(Y), np.var(Y)
    ))
    logger.debug("Y.shape={0}, Y.dtype={1}".format(Y.shape, Y.dtype))

    if swenabled:
        pred = nparray_to_image(
            Y, im.shape, input_shape, stride=kwargs['stride']
        )
    else:
        pred = Y[0]

    lib.image.save_image(kwargs['output'], pred)
    logger.info('Prediction saved to "{0}".'.format(kwargs['output']))

    mpl_comparison(im, pred,
        os.path.basename(kwargs['input']),
        '{0} {1}'.format(*ctx.obj['weights'])
    )


@main.command()
@predict_click_options
@click.option(
    '-i',
    '--input',
    required=True,
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, readable=True
    )
)
@click.option(
    '-o',
    '--output',
    required=True,
    type=click.Path(
        exists=False, file_okay=True, dir_okay=False, writable=True
    )
)
@click.pass_context
def fmapvis(ctx, **kwargs):
    """Visualise feature maps of a model."""
    if not models.model_exists(kwargs['model']):
        raise click.UsageError(
            "Model '{0}' does not exist.".format(kwargs['model']),
            ctx=ctx
        )

    if not weights.weights_exist(kwargs['model'], kwargs['iteration']):
        raise click.UsageError(
            "Model '{0}' (iteration: '{1}') does not exist.".format(
                kwargs['model'], kwargs['iteration']),
            ctx=ctx
        )

    ctx.obj['model'] = kwargs['model']
    ctx.obj['weights'] = (kwargs['model'], kwargs['iteration'])

    if not lib.image.valid_image(kwargs['input']):
        raise click.UsageError(
            "File '{0}' is not a valid image.".format(
                kwargs['input']),
            ctx=ctx
        )

    im = lib.image.load_image(kwargs['input'])


if __name__ == '__main__':
    main(obj={})
