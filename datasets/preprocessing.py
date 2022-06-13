# -*- coding: utf-8 -*-
"""Dataset preprocessing tools.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import gc
import sys
import multiprocessing as mp
import functools
import pathlib
import datetime

import click
import numpy as np
import skimage.transform
import cv2
import h5py

import lib.image as image
import datasets


class DatasetGenerator:
    """Dataset preparation and augmentation class.

    All paths are relative to the individual dataset directory (where the
    dataset's __init__.py file exists).

    Settable options:
    - crop: if set to True, will enable image cropping prior to transformations.
    - crop_px: a tuple of integers setting how many pixels are cropped from
        each edge. The order of direction is: left, top, right, bottom.
    - rotate: if set to True, will enable augmentation via image rotation.
    - rot_directions: a list of rotations in radians that will be used when
        augmenting the dataset.
    - rot_pad: decides how to fill in empty pixels on non right hand rotations,
        see rotate_image() in lib.image.
    - rot_crop: crops the image according to settings after the rotation.
        If set to False, no cropping will occur. Other settings are:
            - 'imsize': will attempt to crop the image to size of the original
                with the content centred.
    - flip_h: if set to True, will enable augmentation via horizontal flipping.
    - flip_v: if set to True, will enable augmentation via vertical flipping.
    - split_chunks: if set to True, will split each image into smaller chunks
        according to other settings.
    - chunk_size: a tuple containing (width, height) of the expected image
        chunks.
    - chunk_stride: a tuple containing (x, y) pixel values that determine
        the number of pixels advanced per iteration as the image is chunked.
        Setting these to less than the chunk_size means that adjacent chunks
        will overlap each other, which can be useful for augmentation.
    - chunk_cutoff: if set to True, it will skip chunks that only cover the
        image partially. If false, it will save partial chunks and attempt
        to fill in missing content.
    - chunk_padmode: accepts modes from numpy.pad() to decide how missing
        content in partial chunks is filled in.
    - chunk_origin: decides the origin of the bounding box of the whole chunk
        set, see sliding_window_2d() in lib.image.
    - hdf5_save_mode: mode used when writing images to the HDF5 file,
        see convmode() in lib.image.
    - hdf5_save_type: data type used when writing images to the HDF5 file,
        see convtype() in lib.image.
    - processes: number of processes to use in multicore operations.

    Attributes:
        dsid (str): ID string of the dataset.
        inputs (list): List of tuples containing respectively the measurement
            and the corresponding ground truth image.
            Example format:
                [('path/to/exp/img.tif', 'path/to/ground/truth.tif')]
        source_in (str): Path to source images defined in 'inputs'.
        example_out (str): Path to store processed measurement images in.
            Should be different from the source_in path as filenames may be
            reused.
        ground_out (str): Path to store processed ground truth images in.
            Should be different from the source_in path as filenames may be
            reused.
        hdf5_save (str): Path to hdf5 file for the processed dataset.
            If set to None, no hdf5 file will be created.
        options (dict): A dict of options for built in image processing.

    """

    def __init__(self,
        dsid, inputs, source_in, example_out, ground_out, hdf5_save=None,
        options=None
    ):
        if options is None:
            options = {}

        def_options = {
            'crop': False,
            'crop_px': (0, 0, 0, 0),
            'rotate': False,
            'rot_directions': [0, 0.5 * np.pi, np.pi, 1.5 * np.pi],
            'rot_pad': 'reflect',
            'rot_crop': False,
            'flip_h': False,
            'flip_v': False,
            'split_chunks': False,
            'chunk_size': (256, 256),
            'chunk_stride': (128, 128),
            'chunk_cutoff': True,
            'chunk_padmode': 'reflect',
            'chunk_origin': 'lefttop',
            'hdf5_save_mode': 'grayscale',
            'hdf5_save_type': 'uint8',
            'processes': 12
        }

        self.dsid = dsid
        self.inputs = inputs
        self.options = {**def_options, **options}

        self.basedir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            dsid
        )

        if hdf5_save is not None:
            self.hdf5 = True
            self.hdf5path = os.path.join(self.basedir, hdf5_save)
        else:
            self.hdf5 = False
            self.hdf5path = None

        self.source_in = os.path.join(self.basedir, source_in)
        self.example_out = os.path.join(self.basedir, example_out)
        self.ground_out = os.path.join(self.basedir, ground_out)

        self._filters = []

    def add_filter(self, f, fkwargs={}, fmode='both', delinputs=False):
        """Add filter function to the processing list.

        All passed functions should have one argument which will be used to
        pass in image data. If the passed function has other arguments, they
        should be pre-set using fkwargs when calling this method.

        The output of passed functions should be a list of tuples containing
        the suffix to be added to the image filename, and the transformed
        image data. The length of the list will determine how many new images
        are created in the working directory.

        Args:
            f (callable): Function to be called later.
            fkwargs (dict): Any function arguments to be set to constant.
            fmode (str): Which parts of the pair will the filter apply to.
                Options are: example, ground, both.
            delinputs (bool): If set to True, the images fed in as inputs
                will be deleted from the working directory when the
                filter operation is complete. Set to True when replacing
                images, False if merely augmenting the dataset.

        Returns:
            None

        """
        if not callable(f):
            raise RuntimeError(
                'Passed function is not callable. Function repr: {0} '.format(
                    repr(f)
                )
            )
        self._filters.append((f, fkwargs, fmode, delinputs))

    def run(self, ctx):
        """Run the dataset generator.

        Args:
            ctx (dict): Click context dict.

        Returns:
            None

        """

        self._ctx = ctx
        ctx.obj['logger'].info("Creating working directories...")
        pathlib.Path(self.example_out).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.ground_out).mkdir(parents=True, exist_ok=True)

        self.wpairs = []

        # Load and save images
        #
        # This step could be done with a simple copy, but this is also
        # an I/O test to see if images can be read by scripts.
        i = _progbar_wrapper(self, _load_image_pairs,
            'Loading image pairs...', len(self.inputs)
        )
        ctx.obj['logger'].info("Read and copied {0} image pairs.".format(i))

        # Crop images
        if self.options['crop']:
            ctx.obj['logger'].info("Operation: crop")
            ctx.obj['logger'].info("Crop options: {0}".format(
                self.options['crop_px']
            ))
            i = _progbar_wrapper(self, _crop_images,
                'Cropping image pairs...', len(self.wpairs)
            )
            ctx.obj['logger'].info("Cropped {0} image pairs.".format(i))

        # Create rotations
        if self.options['rotate']:
            ctx.obj['logger'].info("Operation: rotate")
            ctx.obj['logger'].info("Rotations: {0}".format(
                self.options['rot_directions']
            ))
            ctx.obj['logger'].info("Rotation padding: {0}".format(
                self.options['rot_pad']
            ))
            ctx.obj['logger'].info("Rotation cropping: {0}".format(
                self.options['rot_crop']
            ))
            i = _progbar_wrapper(self,
                functools.partial(
                    _mprun_wrapper, mpfn=_rotate_images_mprun, mpfnkwargs={
                        'rot_dir': self.options['rot_directions'],
                        'rot_pad': self.options['rot_pad'],
                        'crop': self.options['rot_crop'],
                        'path1': self.example_out,
                        'path2': self.ground_out
                    }
                ),
                'Creating image pair rotations...', len(self.wpairs)
            )
            ctx.obj['logger'].info("Created {0} pair rotations.".format(i))

        # Create flips
        if self.options['flip_h']:
            ctx.obj['logger'].info("Operation: flip_h")
            i = _progbar_wrapper(self, _hflip_images,
                'H-flipping image pairs...', len(self.wpairs)
            )
            ctx.obj['logger'].info("Flipped {0} image pairs.".format(i))

        if self.options['flip_v']:
            ctx.obj['logger'].info("Operation: flip_v")
            i = _progbar_wrapper(self, _vflip_images,
                'V-flipping image pairs...', len(self.wpairs)
            )
            ctx.obj['logger'].info("Flipped {0} image pairs.".format(i))

        # Apply custom filters
        for f, fkwargs, fmode, delinputs in self._filters:
            ctx.obj['logger'].info("Operation: filter-{0}".format(f.__name__))
            i = _progbar_wrapper(self,
                functools.partial(_mprun_wrapper,
                    mpfn=_filter_wrapper, mpfnkwargs={
                        'fn': f,
                        'fnkwargs': fkwargs,
                        'fmode': fmode,
                        'path1': self.example_out,
                        'path2': self.ground_out
                    }, unlink=delinputs
                ),
                'Processing...', len(self.wpairs)
            )
            ctx.obj['logger'].info("Processed {0} image pairs.".format(i))

        # Split into chunks
        if self.options['split_chunks']:
            ctx.obj['logger'].info("Operation: split_chunks")
            i = _progbar_wrapper(self,
                functools.partial(
                    _mprun_wrapper, mpfn=_split_chunks_mprun, mpfnkwargs={
                        'opts': {
                            'size': self.options['chunk_size'],
                            'stride': self.options['chunk_stride'],
                            'cutoff': self.options['chunk_cutoff'],
                            'padmode': self.options['chunk_padmode'],
                            'origin': self.options['chunk_origin']
                        },
                        'path1': self.example_out,
                        'path2': self.ground_out
                    }
                ),
                'Chunking image pairs...', len(self.wpairs)
            )
            ctx.obj['logger'].info("Chunked {0} image pairs.".format(i))

            ctx.obj['logger'].info("Checking dimensions...")
            errors = 0
            for pair in self.wpairs:
                im1path = os.path.join(self.example_out, pair[0])
                im2path = os.path.join(self.ground_out, pair[1])
                im1 = image.load_image(im1path)
                im2 = image.load_image(im2path)
                if im1 is None:
                    ctx.obj['logger'].warning(
                        "Exp. image '{0}' not found.".format(im1path)
                    )
                if im2 is None:
                    ctx.obj['logger'].warning(
                        "Ground image '{0}' not found.".format(im2path)
                    )
                if (im1.shape[1] != self.options['chunk_size'][0]
                    or im1.shape[0] != self.options['chunk_size'][1]
                ):
                    ctx.obj['logger'].warning(
                        ("Exp. image is of size {0}x{1},"
                        " expected size is {2}x{3}.").format(
                            im1.shape[1],
                            im1.shape[0],
                            self.options['chunk_size'][0],
                            self.options['chunk_size'][1]
                        )
                    )
                if (im2.shape[1] != self.options['chunk_size'][0]
                    or im2.shape[0] != self.options['chunk_size'][1]
                ):
                    ctx.obj['logger'].warning(
                        ("Ground image is of size {0}x{1},"
                        " expected size is {2}x{3}.").format(
                            im2.shape[1],
                            im2.shape[0],
                            self.options['chunk_size'][0],
                            self.options['chunk_size'][1]
                        )
                    )
            ctx.obj['logger'].info("{0} outliers found.".format(errors))

        # Stats
        pcount = len(self.wpairs)
        ctx.obj['logger'].info("Internal pairs count: {0}.".format(pcount))
        for d in [self.example_out, self.ground_out]:
            dcount = datasets.get_ds_size(self.example_out)
            ctx.obj['logger'].info("Image count in '{0}': {1}.".format(
                d, dcount
            ))

        # Save hdf5
        if self.hdf5:
            if not self.options['split_chunks']:
                ctx.obj['logger'].error(
                    "Unchunked HDF5 save currently unsupported. Exiting."
                )
                quit()

            ctx.obj['logger'].info(
                (
                    "Saving to HDF5 format"
                    " (target: '{0}')"
                ).format(self.hdf5path)
            )
            with h5py.File(self.hdf5path, 'w') as f:
                root = f.create_group('root')
                root.attrs['time_generated'] = datetime.datetime.timestamp(
                    datetime.datetime.now()
                )
                root.attrs['dataset'] = self.dsid
                if self.options['split_chunks']:
                    root.attrs['chunksize_x'] = self.options['chunk_size'][0]
                    root.attrs['chunksize_y'] = self.options['chunk_size'][1]

                dstype1 = datasets.get_ds_format(
                    self.example_out,
                    self.options['hdf5_save_type'],
                    self.options['hdf5_save_mode']
                )
                dslen1 = datasets.get_ds_size(self.example_out)
                dstype2 = datasets.get_ds_format(
                    self.ground_out,
                    self.options['hdf5_save_type'],
                    self.options['hdf5_save_mode']
                )
                dslen2 = datasets.get_ds_size(self.ground_out)
                if dslen1 != dslen2:
                    ctx.obj['logger'].warning(
                        ("Mismatch in dataset size: exp. set size: {0},"
                        " ground set size: {1}.").format(
                            dslen1, dslen2
                        )
                    )

                dsshape = tuple([dslen1] + list(dstype1[0]))

                try:
                    d1name = os.path.basename(self.example_out)
                    ctx.obj['logger'].info(
                        (
                            "Creating dataset: {0}"
                            " (shape: {1}, dtype: {2})."
                        ).format(d1name, dsshape, dstype1[1])
                    )
                    self._d1 = root.create_dataset(
                        d1name, dsshape, dtype=dstype1[1]
                    )

                    d2name = os.path.basename(self.ground_out)
                    ctx.obj['logger'].info(
                        (
                            "Creating dataset: {0}"
                            " (shape: {1}, dtype: {2})."
                        ).format(d2name, dsshape, dstype2[1])
                    )
                    self._d2 = root.create_dataset(
                        d2name, dsshape, dtype=dstype2[1]
                    )
                except OSError as error:
                    ctx.obj['logger'].error((
                            "Could not create dataset."
                            " Raised error: {0}"
                        ).format(repr(error))
                    )
                    quit()

                i = _progbar_wrapper(self, _save_images_hdf5,
                    'Writing dataset to HDF5...', len(self.wpairs)
                )

            ctx.obj['logger'].info("HDF5 file successfully written.")

        ctx.obj['logger'].info("Dataset generation ended.")


def _progbar_wrapper(dgen, fn, label, len, kwargs={}):
    with click.progressbar(fn(dgen, **kwargs), label=label, length=len) as h:
        i = 0
        for d in h:
            i += d
    gc.collect()
    return i


def _mprun_wrapper(dgen, mpfn, mpfnkwargs, unlink=True):
    mprun = functools.partial(mpfn, **mpfnkwargs)
    _wpairs = []

    with mp.Pool(processes=dgen.options['processes']) as pool:
        for result in pool.imap_unordered(
            func=mprun,
            iterable=_read_img_pair(
                (dgen.example_out, dgen.ground_out), dgen.wpairs
            )
        ):
            _wpairs.extend(result)
            yield 1

    if unlink:
        _unlink_img_pairs((dgen.example_out, dgen.ground_out), dgen.wpairs)
        dgen.wpairs = _wpairs
    else:
        dgen.wpairs.extend(_wpairs)


def _filter_wrapper(pair, fn, fnkwargs, fmode, path1, path2):
    _wpairs = []

    if fmode == 'both':
        exp = fn(pair[1], **fnkwargs)
        ground = fn(pair[3], **fnkwargs)

    elif fmode == 'example':
        exp = fn(pair[1], **fnkwargs)
        ground = []
        for i in exp:
            ground.append((i[0], pair[3]))

    elif fmode == 'ground':
        ground = fn(pair[3], **fnkwargs)
        exp = []
        for i in ground:
            exp.append((i[0], pair[1]))

    else:
        raise ValueError(
            ('Unknown value for filter mode: {0}. '
            'Check __init__.py for the dataset being processed.').format(
                fmode
            )
        )

    for i, im in enumerate(exp):
        expf = _suffix_filename(pair[0], exp[i][0])
        groundf = _suffix_filename(pair[2], ground[i][0])
        _write_img(path1, expf, exp[i][1])
        _write_img(path2, groundf, ground[i][1])

        _wpairs.append((expf, groundf))

    return _wpairs


def _load_image_pairs(dgen):
    dgen.wpairs = []
    for input in dgen.inputs:
        im1 = image.load_image(os.path.join(dgen.source_in, input[0]))
        im2 = image.load_image(os.path.join(dgen.source_in, input[1]))
        if im1 is None or im2 is None:
            yield 0
        _write_img(dgen.example_out, input[0], im1)
        _write_img(dgen.ground_out, input[1], im2)
        dgen.wpairs.append((input[0], input[1]))
        yield 1


def _crop_images(dgen):
    _wpairs = []
    for pair in dgen.wpairs:
        im1 = image.load_image(os.path.join(dgen.example_out, pair[0]))
        im2 = image.load_image(os.path.join(dgen.ground_out, pair[1]))
        _write_img(dgen.example_out, pair[0],
            image.crop_image(im1, *dgen.options['crop_px']))
        _write_img(dgen.ground_out, pair[1],
            image.crop_image(im2, *dgen.options['crop_px']))
        _wpairs.append((pair[0], pair[1]))
        yield 1
    dgen.wpairs = _wpairs


def _rotate_images_mprun(pair, rot_dir, rot_pad, crop, path1, path2):
    suffix = '_r{:03d}'
    _wpairs = []
    for a in rot_dir:
        exp = image.rotate_image(pair[1], a, rot_pad)
        ground = image.rotate_image(pair[3], a, rot_pad)

        ox, oy = pair[1].shape[1], pair[1].shape[0]
        rx, ry = exp.shape[1], exp.shape[0]
        if crop == 'imsize':
            left = (rx / 2) - (ox / 2)
            top = (ry / 2) - (oy / 2)
            right = rx - (left + ox)
            bottom = ry - (top + oy)
            crop_px = (
                int(left) if left >= 0 else 0,
                int(top) if top >= 0 else 0,
                int(right) if right >= 0 else 0,
                int(bottom) if bottom >= 0 else 0
            )

        if crop is not False and crop_px is not None:
            exp = image.crop_image(exp, *crop_px)
            ground = image.crop_image(ground, *crop_px)

        deg = int(np.round(a * (180 / np.pi)))
        expf = _suffix_filename(pair[0], suffix.format(deg))
        groundf = _suffix_filename(pair[2], suffix.format(deg))
        _write_img(path1, expf, exp)
        _write_img(path2, groundf, ground)

        _wpairs.append((expf, groundf))

    return _wpairs


def _hflip_images(dgen):
    suffix = 'h'
    _wpairs = []
    for pair in dgen.wpairs:
        im1 = image.load_image(os.path.join(dgen.example_out, pair[0]))
        im2 = image.load_image(os.path.join(dgen.ground_out, pair[1]))
        expf = _suffix_filename(pair[0], suffix)
        groundf = _suffix_filename(pair[1], suffix)
        _write_img(dgen.example_out, expf, np.fliplr(im1))
        _write_img(dgen.ground_out, groundf, np.fliplr(im2))
        _wpairs.append((pair[0], pair[1]))
        _wpairs.append((expf, groundf))
        yield 1
    dgen.wpairs = _wpairs


def _vflip_images(dgen):
    suffix = 'v'
    _wpairs = []
    for pair in dgen.wpairs:
        im1 = image.load_image(os.path.join(dgen.example_out, pair[0]))
        im2 = image.load_image(os.path.join(dgen.ground_out, pair[1]))
        expf = _suffix_filename(pair[0], suffix)
        groundf = _suffix_filename(pair[1], suffix)
        _write_img(dgen.example_out, expf, np.flipud(im1))
        _write_img(dgen.ground_out, groundf, np.flipud(im2))
        _wpairs.append((pair[0], pair[1]))
        _wpairs.append((expf, groundf))
        yield 1
    dgen.wpairs = _wpairs


def _split_chunks_mprun(pair, opts, path1, path2):
    suffix = '_{:d}'
    _wpairs = []

    imsz = (pair[1].shape[1], pair[1].shape[0])
    i = 0
    try:
        for sw in image.sliding_window_2d(imsz,
            opts['size'], opts['stride'], opts['origin'], opts['cutoff']
        ):
            exp = image.slice_image(pair[1], sw, opts['padmode'])
            ground = image.slice_image(pair[3], sw, opts['padmode'])

            expf = _suffix_filename(pair[0], suffix.format(i))
            groundf = _suffix_filename(pair[2], suffix.format(i))
            _write_img(path1, expf, exp)
            _write_img(path2, groundf, ground)

            _wpairs.append((expf, groundf))
            i += 1
    except RuntimeError as error:
        raise RuntimeError(
            'Error caught: {0} when processing images "{1}" and "{2}"'.format(
                repr(error), pair[0], pair[2]
            )
        )

    return _wpairs


def _save_images_hdf5(dgen):
    _wpairs = []
    i = 0
    for pair in dgen.wpairs:
        dgen._d1[i, :] = image.load_image(
            os.path.join(dgen.example_out, pair[0]),
            dgen.options['hdf5_save_type'],
            dgen.options['hdf5_save_mode']
        )
        dgen._d2[i, :] = image.load_image(
            os.path.join(dgen.ground_out, pair[1]),
            dgen.options['hdf5_save_type'],
            dgen.options['hdf5_save_mode']
        )
        _wpairs.append((pair[0], pair[1]))
        i += 1
        yield 1
    dgen.wpairs = _wpairs


def _read_img_pair(dirs, f, itype=None, imode=None):
    if type(f) is not list:
        f = [f]

    for f1, f2 in f:
        f1p = os.path.join(dirs[0], f1)
        f2p = os.path.join(dirs[1], f2)
        if not (image.valid_image(f1p) and image.valid_image(f2p)):
            continue

        yield (
            f1, image.load_image(f1p, itype, imode),
            f2, image.load_image(f2p, itype, imode)
        )


def _write_img(dir, f, content, itype=None, imode=None):
    if not os.path.isdir(dir):
        raise IOError('"{0}": Not a valid directory.'.format(dir))
        return None

    p = os.path.join(dir, f)
    return image.save_image(p, content, itype, imode)


def _unlink_img_pairs(dirs, f):
    if type(f) is not list:
        f = [f]

    for f1, f2 in f:
        f1p = os.path.join(dirs[0], f1)
        f2p = os.path.join(dirs[1], f2)
        pathlib.Path(f1p).unlink(missing_ok=True)
        pathlib.Path(f2p).unlink(missing_ok=True)

    return []


def _suffix_filename(p, suffix):
    ext = os.path.splitext(p)
    return ext[0] + str(suffix) + ext[1]
