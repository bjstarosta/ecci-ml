# -*- coding: utf-8 -*-
"""Datasets module.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import errno
import importlib
import inspect

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
        ds.logger = logger
        if rs is not None:
            ds.rs = rs

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

    All subclasses of this class can be passed to tf.keras.model.fit() as a
    parameter directly.

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
        shuffle_on_epoch_end (bool): If set to True, the data list will be
            shuffled every epoch.

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
        self.shuffle_on_epoch_end = False

        self.logger = None
        self.indices = None
        self._apply = lambda x: x

    def __len__(self):
        """Number of batches in the sequence.

        Returns:
            int

        """
        return int(np.floor(len(self.x) / self.batch_size))

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
        i0 = idx * self.batch_size
        i1 = min((idx + 1) * self.batch_size, len(self.x))
        i = self.indices[i0:i1]

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
        self._generate_indices()
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
        """Shuffle the data indices using the internal numpy random state.

        Returns:
            None

        """
        if self.indices is None:
            self._generate_indices()

        self.rs.shuffle(self.indices)

    def split(self, split=0.5):
        """Return a new object instance with the specified data split.

        The split variable defines the slicing point of the dataset data list.
        The object calling the method will retain the left portion of the
        data list, while the remainder will be transferred to the new object
        returned by this method.

        The split happens according to the currently set batch size, rather
        than the absolute size of the dataset. As such, half split datasets may
        not be exactly equal in dataset length.

        Args:
            split (float): The slicing point of the data, translated as
                split * len(data).

        Returns:
            Dataset: New object instance of the current class.

        """
        ret = self.__class__()
        for attr in [
            'basepath', 'id', 'rs', 'desc', 'generated',
            'batch_size', 'shuffle_on_epoch_end',
            'logger', '_apply'
        ]:
            setattr(ret, attr, getattr(self, attr))

        x_full = self.x
        y_full = self.y

        split_offset = int(np.floor(split * len(self.x)))
        modulo = split_offset % self.batch_size
        if modulo > 0:
            modulo = self.batch_size - modulo
        split_offset = split_offset + modulo

        self.x = x_full[:split_offset]
        self.y = y_full[:split_offset]
        self._generate_indices()

        ret.x = x_full[split_offset:]
        ret.y = y_full[split_offset:]
        ret._generate_indices()

        return ret

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

    def _load_images(self, dir, indices, type=None, mode=None):
        """Load a list of images from the specified directory.

        For use from within the load_data() batch generation method.

        Args:
            dir (str): Directory name to load from. Specified relatively to
                the dataset directory.
            indices (list): List of image filenames.
            type (str): Value type to save the image with. See
                lib.utils.image_convtype() for documentation of accepted
                values.
            mode (str): Channel mode to save the image with. See
                lib.utils.image_convmode() for documentation of accepted
                values.

        Returns:
            numpy.ndarray: Numpy array of images. Depending on the images
                loaded, will either be 3 or 4-dim with the top dimension
                grouping the images together.

        """
        X = []
        path = os.path.join(self.basepath, dir)

        for im in indices:
            im = utils.load_image(os.path.join(path, im), type, mode)
            X.append(im)

        return self._apply(np.array(X))

    def _generate_indices(self):
        """Generate internal data indices.

        These are used for synchronised shuffling of both input and ground
        data.

        Returns:
            None

        """
        self.indices = np.arange(len(self.x))


class DatasetCollection(Dataset):
    """Class for grouping datasets together to functionally act as one dataset.

    Just like a Dataset object, this can be passed to tf.keras.model.fit() as a
    parameter.

    Attributes:
        datasets (list): List of datasets included in the collection.
            Use the add() method to add datasets to it to ensure correct
            indices.
        batch_size (int): Size of data batch to load per training iteration.
            Propagated to contained datasets.
        shuffle_on_epoch_end (bool): If set to True, the data list will be
            shuffled every epoch. Propagated to contained datasets.

    """

    def __init__(self):
        self.datasets = []
        self.indices = []

        self._rs = None
        self._batch_size = 32
        self._shuffle_on_epoch_end = False
        self._broadcast_attrs = ['batch_size', 'shuffle_on_epoch_end']

    def __len__(self):
        """Number of batches in the collection.

        Returns:
            int

        """
        return len(self.indices)

    def __getitem__(self, idx):
        ds, i = self.indices[idx]
        return ds[i]

    def on_epoch_end(self):
        """Update indices after each epoch.

        Method ran automatically by model.fit() at each epoch end.

        Returns:
            None

        """
        for ds in self.datasets:
            if ds.shuffle is True:
                ds.shuffle()

    def add(self, ds):
        if len(ds) == 0:
            raise RuntimeError('Passed dataset len = 0. Did you call setup()?')
        for attr in self._broadcast_attrs:
            setattr(ds, attr, getattr(self, attr))
        self.datasets.append(ds)
        self.indices = self.indices + [(ds, i) for i in range(len(ds))]
        self._propagate_rs()

    def _generate_indices(self):
        self.indices = []
        for ds in self.datasets:
            self.indices = self.indices + [(ds, i) for i in range(len(ds))]

    def _propagate_rs(self):
        if self._rs is None:
            self._rs = self.datasets[0].rs
        for ds in self.datasets:
            ds.rs = self._rs

    @property
    def rs(self):
        return self._rs

    @rs.setter
    def rs(self, value):
        self._rs = value
        self._propagate_rs()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        for ds in self.datasets:
            ds.batch_size = value
        self._generate_indices()

    @property
    def shuffle_on_epoch_end(self):
        return self._shuffle_on_epoch_end

    @shuffle_on_epoch_end.setter
    def shuffle_on_epoch_end(self, value):
        self._shuffle_on_epoch_end = value
        for ds in self.datasets:
            ds.shuffle_on_epoch_end = value

    def setup(self, limit=None):
        for ds in self.datasets:
            ds.setup(limit)

    def shuffle(self):
        for ds in self.datasets:
            ds.shuffle()

    def split(self, split=0.5):
        ret = self.__class__()
        for attr in self._broadcast_attrs:
            setattr(ret, attr, getattr(self, attr))
        for ds in self.datasets:
            ret.add(ds.split(split))
        self._generate_indices()
        return ret

    def apply(self, fn):
        for ds in self.datasets:
            ds.apply(fn)


class DatasetException(Exception):
    """Base class for Dataset specific exceptions."""

    pass
