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
import h5py
import click

import lib.logger
import lib.image as image


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


def load_dataset(ds_id, seed=None):
    """Load a dataset object.

    Args:
        ds_id (str): Dataset identifier.
        seed (int): Random number generator seed.

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
        ds.rs = np.random.default_rng(seed=seed)
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

    All subclasses of this class can be passed to tf.keras.model.fit() as a
    parameter directly.

    Dataset indexing is based on the specified batch size. Items can be called
    to using array notation, but what will be returned will always be a data
    batch, not a single item. This means that the indices change when the
    batch_size attribute is changed. The size of the dataset is always
    the number of data points / specified batch size.

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

        self.rs = None
        self.batch_size = 32
        self.shuffle_on_epoch_end = False

        self.logger = None
        self.indices = None

        self._apply = lambda x: x
        self._preprocess = lambda x: x
        self._broadcast_attrs = [
            'basepath', 'id', 'rs', 'desc', 'generated',
            'batch_size', 'shuffle_on_epoch_end',
            'logger', '_apply', '_preprocess'
        ]

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
        if isinstance(idx, int):
            i = self._get_indices(idx)
            batch_x = [self.x[k] for k in i]
            batch_y = [self.y[k] for k in i]
            return self.load_data(batch_x, batch_y)
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            ret = []
            for j in range(start, stop, step):
                i = self._get_indices(j)
                batch_x = [self.x[k] for k in i]
                batch_y = [self.y[k] for k in i]
                ret.append(self.load_data(batch_x, batch_y))
            return ret
        else:
            raise TypeError("index must be int or slice")

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

    def generate_dataset(self, ctx=None):
        """Generate the augmented dataset.

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
        """Return a new Dataset instance with the specified data split.

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
        for attr in self._broadcast_attrs:
            setattr(ret, attr, getattr(self, attr))

        x_full = self.x
        y_full = self.y

        if split <= 0:
            ret.x = []
            ret.y = []
            ret._generate_indices()
            return ret

        if split >= 1:
            self.x = []
            self.y = []
            self._generate_indices()
            ret.x = x_full
            ret.y = y_full
            ret._generate_indices()
            return ret

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

    def slice(self, indices):
        """Return a new Dataset instance containing only specified indices.

        The object from which this method is called will remain unchanged.

        Args:
            indices (list): A list of indices to be preserved in the new
                Dataset instance.

        Returns:
            Dataset: New object instance of the current class.

        """
        ret = self.__class__()
        for attr in self._broadcast_attrs:
            setattr(ret, attr, getattr(self, attr))

        i = []
        for idx in indices:
            i.extend(self._get_indices(idx))

        ret.x = [self.x[k] for k in i]
        ret.y = [self.y[k] for k in i]
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

    def preprocess(self, fn):
        """Set function to preprocess the training images on load.

        This function should contain only image filters and/or transformations
        to be applied to only the training images, and not the ground truth.

        Args:
            fn (callable): Function to pass loaded data through.

        Returns:
            None

        """
        if not callable(fn):
            raise DatasetException('Expected callable var in preprocess().')

        self._preprocess = fn

    def statistics(self):
        """Return statistical information about a dataset.

        Returns:
            tuple: Tuple containing the average, variance, min, max, count
                and NaN count of both the X and Y training datasets.

        """
        acc_x = np.array([[0.0, 0.0, 0, 0, 0, 0]], dtype=np.float64)
        acc_y = np.array([[0.0, 0.0, 0, 0, 0, 0]], dtype=np.float64)

        j = 1
        for i in self:
            acc_x = np.append(acc_x, np.array([[
                np.average(i[0]),
                np.var(i[0]),
                np.min(i[0]),
                np.max(i[0]),
                j,
                np.count_nonzero(np.isnan(i[0]))
            ]], dtype=np.float64), axis=0)
            acc_y = np.append(acc_y, np.array([[
                np.average(i[1]),
                np.var(i[1]),
                np.min(i[1]),
                np.max(i[1]),
                j,
                np.count_nonzero(np.isnan(i[1]))
            ]], dtype=np.float64), axis=0)
            j = j + 1

        var_x = np.sum(acc_x[:, 1])
        var_y = np.sum(acc_y[:, 1])

        return (
            (
                np.average(acc_x[:, 0]),
                var_x,
                np.sqrt(var_x) if var_x > 0 else 0,
                np.min(acc_x[:, 2]),
                np.max(acc_x[:, 3]),
                acc_x[len(acc_x) - 1, 4],
                np.sum(acc_x[:, 5])
            ), (
                np.average(acc_y[:, 0]),
                var_y,
                np.sqrt(var_y) if var_y > 0 else 0,
                np.min(acc_y[:, 2]),
                np.max(acc_y[:, 3]),
                acc_y[len(acc_y) - 1, 4],
                np.sum(acc_y[:, 5])
            )
        )

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
            if not image.valid_image(im_path):
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

    def _load_images(
        self, dir, indices, type=None, mode=None, preprocess=False
    ):
        """Load a list of images from the specified directory.

        For use from within the load_data() batch generation method.

        Args:
            dir (str): Directory name to load from. Specified relatively to
                the dataset directory.
            indices (list): List of image filenames.
            type (str): Value type to save the image with. See
                lib.image.convtype() for documentation of accepted
                values.
            mode (str): Channel mode to save the image with. See
                lib.image.convmode() for documentation of accepted
                values.
            preprocess (bool): Set to True if loaded images are to be passed
                through the preprocessing function.

        Returns:
            numpy.ndarray: Numpy array of images. Depending on the images
                loaded, will either be 3 or 4-dim with the top dimension
                grouping the images together.

        """
        X = []
        path = os.path.join(self.basepath, dir)

        for im in indices:
            im = image.load_image(os.path.join(path, im), type, mode)
            X.append(im)

        if preprocess is True:
            X = self._preprocess(X)

        return self._apply(np.array(X))

    def _load_images_hdf5(
        self, hdf5ds, indices, type=None, mode=None, preprocess=False
    ):
        """Load a list of images from the passed hdf5 dataset.

        For use from within the load_data() batch generation method.

        Args:
            hdf5ds (h5py.Dataset): An hdf5 dataset handle.
            indices (list): List of integers denoting image positions in the
                hdf5 dataset.
            type (str): Value type to save the image with. See
                lib.image.convtype() for documentation of accepted
                values.
            mode (str): Channel mode to save the image with. See
                lib.image.convmode() for documentation of accepted
                values.
            preprocess (bool): Set to True if loaded images are to be passed
                through the preprocessing function.

        Returns:
            numpy.ndarray: Numpy array of images. Depending on the images
                loaded, will either be 3 or 4-dim with the top dimension
                grouping the images together.

        """
        X = []

        for im in indices:
            im = hdf5ds[im] #np.copy()
            im = image.convtype(im, type)
            im = image.convmode(im, mode)
            X.append(im)

        if preprocess is True:
            X = self._preprocess(X)

        return self._apply(np.array(X))

    def _generate_indices(self):
        """Generate internal data indices.

        These are used for synchronised shuffling of both input and ground
        data.

        Returns:
            None

        """
        self.indices = np.arange(len(self.x))

    def _get_indices(self, idx):
        """Convert batch index to list of datapoint indices.

        Args:
            idx (int): Batch index.

        Returns:
            list: List of indices corresponding to values in self.x and self.y.

        """
        i0 = idx * self.batch_size
        i1 = min((idx + 1) * self.batch_size, len(self.x))
        return self.indices[i0:i1]

    def _get_hdf5(self, filename):
        if self._fh is None:
            fhpath = os.path.join(path(self.id), filename)
            self._fh = h5py.File(fhpath, 'r')
        return self._fh


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

    def __init__(self, seed=None):
        self.datasets = []
        self.indices = []

        self._rs = np.random.default_rng(seed=seed)
        self._batch_size = 32
        self._shuffle_on_epoch_end = False
        self._broadcast_attrs = ['rs', 'batch_size', 'shuffle_on_epoch_end']

    def __len__(self):
        """Number of batches in the collection.

        Returns:
            int

        """
        return len(self.indices)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            ds, i = self.indices[idx]
            return ds[i]
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            ret = []
            for j in range(start, stop, step):
                ds, i = self.indices[j]
                ret.append(ds[i])
            return ret
        else:
            raise TypeError("index must be int or slice")

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
        self.indices.extend([(ds, i) for i in range(len(ds))])
        self._propagate_rs()

    def _generate_indices(self):
        self.indices = []
        for ds in self.datasets:
            self.indices.extend([(ds, i) for i in range(len(ds))])

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

    def slice(self, indices):
        ret = self.__class__()
        for attr in self._broadcast_attrs:
            setattr(ret, attr, getattr(self, attr))

        sorted = {}
        for idx in indices:
            ds, i = self.indices[idx]
            if ds not in sorted:
                sorted[ds] = []
            sorted[ds].append(i)

        for ds, i in sorted.items():
            ret.add(ds.slice(i))

        return ret

    def apply(self, fn):
        for ds in self.datasets:
            ds.apply(fn)

    def preprocess(self, fn):
        for ds in self.datasets:
            ds.preprocess(fn)


class DatasetException(Exception):
    """Base class for Dataset specific exceptions."""
    pass


def get_ds_size(path):
    """Return number of files in a directory.

    Args:
        path (str): Path to directory.

    Returns:
        int: Non-negative number indicating number of files found.

    """
    plist = os.listdir(path)
    i = 0
    for f in plist:
        i += 1
    return i


def get_ds_format(path, itype=None, imode=None):
    """Return format of the first file in a directory.

    The assumption is that all data files in a directory are uniform.

    Args:
        path (str): Path to directory.

    Returns:
        tuple: Numpy array dimensions of the read file in rc format.
        numpy.dtype: Dtype of the read file.

    """
    file = os.path.join(path, os.listdir(path)[0])
    im = image.load_image(file, itype, imode)
    return im.shape, im.dtype
