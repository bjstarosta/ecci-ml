# -*- coding: utf-8 -*-
"""Datasets module.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import errno
import importlib
import inspect
import math

import numpy as np
import tensorflow.keras as K
import click

import lib.logger
import lib.utils as utils


logger = lib.logger.logger
PATH_DATASETS = os.path.dirname(os.path.abspath(__file__))


def path(ds_id, basename=False):
    """Return a path to the dataset folder.

    Args:
        ds_id (str): Dataset identifier.
        basename (bool): If False, an absolute path will be returned.
            If True, only the filename will be returned.

    Returns:
        str: Path to folder or full folder name.

    """
    bn = ds_id
    if basename is True:
        return bn
    else:
        return os.path.join(PATH_DATASETS, bn)


def load_dataset(ds_id, rs=None):
    """Load a dataset object.

    Args:
        ds_id (str): Dataset identifier.
        rs (numpy.random.RandomState): Optional random state definition.

    Returns:
        datasets.Dataset: Dataset object.

    """
    if dataset_exists(ds_id):
        logger.info('Loading dataset `{0}`.'.format(ds_id))
        mod = importlib.import_module('datasets.' + ds_id)
        cls = inspect.getmembers(mod, inspect.isclass)[-1][0]

        ds = getattr(mod, cls)()
        ds.id = ds_id
        ds.basepath = path(ds_id)
        ds.rs = rs
        ds.logger = logger

        return ds

    else:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), path(ds_id))


def list_datasets(with_desc=False):
    """List all valid datasets present in the directory.

    A valid dataset must have a __init__.py file in its directory that
    defines a subclass of datasets.Dataset with method overrides for
    any dataset specific data organisation.

    Args:
        with_desc (bool): If True, the returned list will be a list of tuples
            containing dataset ID, dataset __doc__.

    Returns:
        list: List of valid datasets.

    """
    lst = []
    for f in os.listdir(PATH_DATASETS):
        p = os.path.join(PATH_DATASETS, f)
        if not os.path.isdir(p):
            continue
        pi = os.path.join(p, '__init__.py')
        if not os.path.exists(pi) or not os.path.isfile(pi):
            continue
        if with_desc is True:
            mod = importlib.import_module('datasets.' + f)
            lst.append((f, mod.__doc__))
        else:
            lst.append(f)
    return sorted(lst)


def dataset_exists(ds_id):
    """Check if a given dataset exists.

    Args:
        ds_id (str): The identifier of the dataset.

    Returns:
        bool: True if dataset exists, False otherwise.

    """
    pi = os.path.join(path(ds_id), '__init__.py')
    return os.path.exists(path(ds_id)) and os.path.exists(path(pi))


class Dataset(K.utils.Sequence):
    """Abstract parent class for dataset subclasses.

    Attributes:
        basepath (str): Absolute path to dataset folder. Set by load_dataset().
        id (str): ID string of dataset. Set by load_dataset().
        desc (str): Description of dataset. Should be set by the inheriting
            class.
        generated (str): Date of generation in a strptime() compatible format.
            Should be set by the inheriting class.
        x (numpy.ndarray): Training sample data identifiers.
        y (numpy.ndarray): Ground truth/label data identifiers.
        rs (numpy.random.RandomState): Random state to be used during
            splitting.
        batch_size (int): Size of data batch to load per training iteration.
        shuffle (bool): If set to True, the data list will be shuffled every
            epoch.

    """

    def __init__(self):
        self.basepath = None
        self.id = None

        self.desc = ""
        self.generated = ""

        self.x = np.array([])
        self.y = np.array([])

        self.rs = np.random.default_rng(seed=None)
        self.batch_size = 32
        self.shuffle = False

        self.logger = None
        self._apply = lambda x: x

    def __len__(self):
        """Number of batches in the sequence.

        Returns:
            int

        """
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        """Return a complete batch at the specified offset.

        Args:
            idx (int): Offset index. Calculated by the batch size multiplied
                by iteration number.

        Returns:
            tuple: Tuple of two numpy.ndarray, first element being the training
                samples and the second being the corresponding
                ground truth/labels.

        """
        i = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = [self.x[k] for k in i]
        batch_y = [self.y[k] for k in i]

        return self.load_data(batch_x, batch_y)

    def on_epoch_end(self):
        """Update indices after each epoch.

        Method ran automatically by model.fit() at each epoch end.
        Should also be called at the end of setup() by inheriting classes.

        Returns:
            None

        """
        self.indices = np.arange(len(self.x))
        if self.shuffle is True:
            self.shuffle()

    def setup(self, limit=None):
        """Set up data sources.

        Returns:
            None

        """
        raise NotImplementedError()

    def load_data(self, batch_x, batch_y):
        """Return raw data loaded from the dataset using identifier lists.

        This method is supposed to be reimplemented according to dataset
        file structure.

        Args:
            batch_x (list): Identifiers of training data files to be loaded.
            batch_y (list): Identifiers of corresponding label data to be
                loaded.

        Returns:
            tuple: Tuple of two numpy.ndarray, first element being the training
                samples and the second being the corresponding
                ground truth/labels.

        """
        raise NotImplementedError()

    def shuffle(self):
        if self.indices is None:
            self.indices = np.arange(len(self.x))

        self.rs.shuffle(self.indices)

    def split(self, split=0.5):
        """Return a new object instance with the specified data split.

        The split variable defines the slicing point of the dataset data list.
        The object calling the method will retain the left portion of the
        data list, while the remainder will be transferred to the new object
        returned by this method.

        Args:
            split (float): The slicing point of the data, translated as
                split * len(data).

        Returns:
            Dataset: New object instance of the current class.

        """
        split = self.__class__()
        for attr in [
            'basepath', 'id', 'rs', 'desc', 'generated',
            'batch_size', 'shuffle',
            'logger'
        ]:
            setattr(split, attr, getattr(self, attr))

        split_offset = int(split * len(self.x))
        x_full = self.x
        y_full = self.y

        self.x = x_full[0:split_offset]
        self.y = y_full[0:split_offset]
        split.x = x_full[split_offset + 1:len(x_full)]
        split.y = y_full[split_offset + 1:len(y_full)]

        return split

    def apply(self, fn):
        """Set function to apply to each datapoint on load.

        Args:
            fn (callable): Function to pass loaded data through.

        Returns:
            None

        """
        if not callable(fn):
            raise DatasetException('Expected callable var in apply().')

        self._apply = fn

    def _list_images(self, dir, limit=None):
        """Prepare an image dataset file listing in memory and return it.

        For use from within the setup() dataset definition method.

        Args:
            dir (str): Directory name to read from. Specified relatively to
                the dataset directory.
            limit (int, optional): If set to a positive integer, will load only
                the first x amount of files in the directory.

        Returns:
            numpy.ndarray: List of image filenames present within the
                directory.

        """
        X = []
        path = os.path.join(self.basepath, dir)

        if not os.path.isdir(path):
            raise IOError('"{0}": Not a directory.'.format(path))

        logger.info("Reading images in '{0}'.".format(
            click.format_filename(path)
        ))

        images = os.listdir(path)

        for im in images:
            im_path = os.path.join(path, im)
            if not utils.valid_image(im_path):
                continue
            X.append(im_path)
        X.sort()

        if type(limit) is int:
            X = X[0:limit]

        logger.info("{0:d} images found in '{1}'.".format(
            len(X),
            click.format_filename(path)
        ))

        return np.array(X)

    def _load_images(self, dir, indices):
        """Load a list of images from the specified directory.

        For use from within the load_data() batch generation method.

        Args:
            dir (str): Directory name to load from. Specified relatively to
                the dataset directory.
            indices (list): List of image filenames.

        Returns:
            numpy.ndarray: Numpy array of images. Depending on the images
                loaded, will either be 3 or 4-dim with the top dimension
                grouping the images together.

        """
        X = []
        path = os.path.join(self.basepath, dir)

        for im in indices:
            im = utils.load_image(os.path.join(path, im))
            X.append(im)

        return self._apply(np.array(X))


class DatasetException(Exception):
    """Base class for other exceptions."""

    pass
