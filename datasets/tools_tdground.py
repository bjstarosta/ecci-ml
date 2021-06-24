# -*- coding: utf-8 -*-
"""Dataset preprocessing tools - TD ground truth preparation.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import sys
import csv
import multiprocessing as mp
import functools

import click
import numpy as np
import skimage.feature

if __name__ == '__main__':
    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import lib.image as image
from datasets.tools import _load_images


def extract_blob_coords(im, min_r=5, max_r=14):
    """Detect blobs on passed image and return their coordinates.

    Args:
        im (numpy.ndarray): Image to process for blob detection.
            The passed image should be single channel greyscale
            (len(im.shape) == 2) to rule out out-of-memory errors.
        min_r (int): Minimum blob radius. Blob radii will be clipped to this
            value.
        max_r (int): Maximum blob radius.

    Returns:
        list: List of lists, where the coordinates are given in order of:
            [y, x, r]

    """
    blobs = skimage.feature.blob_log(im,
        min_sigma=3, max_sigma=15, num_sigma=15, threshold=.1)
    blobs[:, 2] = np.clip(blobs[:, 2] * np.sqrt(2), min_r, max_r)
    return blobs


def _blob_coords_mprun(input, kwargs):
    path, im = input

    blobs = extract_blob_coords(im, kwargs['min_r'], kwargs['max_r'])

    file, ext = os.path.splitext(os.path.basename(path))
    outpath = os.path.join(kwargs['output'], file + '.csv')
    with open(outpath, mode='w') as f:
        writer = csv.DictWriter(f, fieldnames=['x', 'y', 'r'])
        writer.writeheader()

        f.write("#image_width,{0:d}\n".format(im.shape[1]))
        f.write("#image_height,{0:d}\n".format(im.shape[0]))

        for row in blobs:
            writer.writerow({
                'x': int(row[1]),
                'y': int(row[0]),
                'r': float(row[2])
            })

    return outpath


def generate_chunks_csv(
    coords, im_shape, w, h, stride=1., type='disk', radius=5
):
    """Generate ground truth image chunks using a sliding window approach.

    Args:
        coords (list): List of marker coordinates. Should be a list of Dicts
            of the form {x: int, y: int, r: float}.
        im_shape (tuple): Two value tuple of the form
            (image height, image width). Needed for correct sliding window
            calculations.
        w (int): Desired chunk width.
        h (int): Desired chunk height.
        stride (float): Fraction of width and height to advance by per
            iteration.
        radius (int): Pixel radius of the generated markers.

    Returns:
        list: List of numpy.ndarray type images.

    """
    ret = []
    chunk_shape = (h, w)
    stride = (int(w * stride), int(h * stride))
    for wy in np.arange(0, im_shape[0] - h, stride[1]):
        for wx in np.arange(0, im_shape[1] - w, stride[0]):
            wx2 = wx + w
            wy2 = wy + h
            # print(wx, wy, wx2, wy2)
            im = np.zeros(chunk_shape)

            for row in coords:
                if ((wx + radius <= row['x'] <= wx2 - radius)
                and (wy + radius <= row['y'] <= wy2 - radius)):

                    mark_x = row['x'] - wx
                    mark_y = row['y'] - wy

                    if type == 'disk':
                        rr, cc = skimage.draw.disk(
                            (mark_y, mark_x),
                            radius,
                            shape=chunk_shape
                        )
                        im[rr, cc] = 1
                    elif type == 'square':
                        rr, cc = skimage.draw.rectangle(
                            (mark_y - radius, mark_x - radius),
                            (mark_y + radius, mark_x + radius),
                            shape=chunk_shape
                        )
                        im[rr, cc] = 1

            ret.append(im)
    return ret


def _read_csv(path):
    coords = []
    meta = {}

    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if row['x'][0] == '#':
                meta_name = row['x'][1:]
                if meta_name in ['image_width', 'image_height']:
                    meta[meta_name] = int(row['y'])
            else:
                coords.append({
                    'x': int(float(row['x'])),
                    'y': int(float(row['y'])),
                    'r': float(row['r'])
                })

    return coords, meta


def _load_csv(path):
    if not os.path.isdir(path):
        yield path, _read_csv(path)
        return

    files = os.listdir(path)

    files_valid = []
    for f in files:
        f_path = os.path.join(path, f)

        if not os.path.isfile(f_path):
            continue
        _, ext = os.path.splitext(f_path.lower())
        if ext != '.csv':
            continue

        files_valid.append(f_path)
    files_valid.sort()

    for f_path in files_valid:
        yield f_path, _read_csv(f_path)


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
    '-t',
    '--threads',
    type=int,
    default=1,
    help="""Number of threads to run in. Setting to 1 runs task in main
        process."""
)
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
    '--min_r',
    type=int,
    default=5,
    show_default=True,
    help="""Minimum blob radius in pixels."""
)
@click.option(
    '--max_r',
    type=int,
    default=14,
    show_default=True,
    help="""Maximum blob radius in pixels."""
)
@click.pass_context
def blob_coords(ctx, **kwargs):
    """Detect blobs and save their coordinates to a CSV file."""
    ctx.obj['logger'].info("Operation begins: blob_coords")

    n = 0
    if kwargs['threads'] > 1:
        mprun = functools.partial(_blob_coords_mprun, kwargs=kwargs)
        with mp.Pool(processes=kwargs['threads']) as pool:
            for p in pool.imap_unordered(
                func=mprun,
                iterable=_load_images(kwargs['input'], 'uint8', 'gs')
            ):
                ctx.obj['logger'].info("Processed: '{0}'.".format(p))
                n += 1
    else:
        for input in _load_images(kwargs['input'], 'uint8', 'gs'):
            p = _blob_coords_mprun(input, kwargs)
            ctx.obj['logger'].info("Processed: '{0}'.".format(p))
            n += 1

    ctx.obj['logger'].info("{0} images processed.".format(n))
    ctx.obj['logger'].info("CSV file with blob coords saved to '{0}'.".format(
        kwargs['output']))


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
@click.option(
    '-t',
    '--type',
    type=click.Choice(['disk', 'square'], case_sensitive=False),
    default='disk',
    help="""Type of generated markers."""
)
@click.option(
    '--diskradius',
    type=int,
    default=5,
    help="""Radius of generated marker disks."""
)
@click.pass_context
def make_circles(ctx, **kwargs):
    """Generate a sequence of ground truth images with circle markings.

    A CSV file with ground truth coordinates is loaded, then a sliding window
    algorithm passes over the coordinates and generates a series of images,
    marking the coordinates with filled white circles."""
    ctx.obj['logger'].info("Operation begins: make_circles")

    n_csv = 0
    n_img = 0
    for path, (coords, meta) in _load_csv(kwargs['input']):
        ctx.obj['logger'].debug("Processing: '{0}'.".format(path))
        ext = os.path.splitext(os.path.basename(path))

        out = generate_chunks_csv(
            coords, (meta['image_height'], meta['image_width']),
            kwargs['width'], kwargs['height'], kwargs['stride'],
            kwargs['type'], kwargs['diskradius']
        )
        ctx.obj['logger'].debug("{0} chunks returned.".format(len(out)))

        for i, o in enumerate(out):
            outpath = os.path.join(kwargs['output'],
                ext[0] + '_' + str(i) + '.tif')
            image.save_image(outpath, o, type='uint8')
            n_img += 1

        n_csv += 1

    ctx.obj['logger'].info("{0} CSV files processed.".format(n_csv))
    ctx.obj['logger'].info("{0} new images created in '{1}'.".format(
        n_img, kwargs['output']))


if __name__ == '__main__':
    main(obj={})
