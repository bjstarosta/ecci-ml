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
import sklearn.model_selection as ms
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
        ds.path = path(ds_id)
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


class Dataset(object):
    """Abstract parent class for dataset subclasses.

    Attributes:
        path (str): Absolute path to dataset folder. Set by load_dataset().
        id (str): ID string of dataset. Set by load_dataset().
        rs (numpy.random.RandomState): Random state to be used during
            splitting. Set by load_dataset().
        desc (str): Description of dataset. Should be set by the inheriting
            class.
        generated (str): Date of generation in a strptime() compatible format.
            Should be set by the inheriting class.
        X_train (numpy.ndarray): Training dataset inputs.
            Will be empty before self.load() is called.
        X_test (numpy.ndarray): Testing dataset inputs.
            Will be empty before self.split() is called.
        X_val (numpy.ndarray): Validation dataset inputs.
            Will be empty before self.split() is called.
        Y_train (numpy.ndarray): Training dataset labels/outputs.
            Will be empty before self.load() is called.
        Y_test (numpy.ndarray): Testing dataset labels/outputs.
            Will be empty before self.split() is called.
        Y_val (numpy.ndarray): Validation dataset labels/outputs.
            Will be empty before self.split() is called.

    """

    def __init__(self):
        self.path = None
        self.id = None
        self.rs = None

        self.desc = ""
        self.generated = ""

        self.X_train = np.array([])
        self.X_test = np.array([])
        self.X_val = np.array([])
        self.Y_train = np.array([])
        self.Y_test = np.array([])
        self.Y_val = np.array([])

        self.logger = None

    def is_loaded(self):
        """Check if dataset has been loaded into memory.

        Returns:
            bool: True if object contains dataset, False otherwise.

        """
        if self.X_train.shape[0] == 0 and self.Y_train.shape[0] == 0:
            return False
        else:
            return True

    def is_split(self):
        """Check if dataset has been split into testing and validation sets.

        Returns:
            bool: True if object contains split datasets, False otherwise.

        """
        if (self.X_test.shape[0] == 0 and self.Y_test.shape[0] == 0
        and self.X_val.shape[0] == 0 and self.Y_val.shape[0] == 0):
            return False
        else:
            return True

    def get_train(self):
        """Return the training dataset.

        Returns:
            numpy.ndarray, numpy.ndarray

        """
        return self.X_train, self.Y_train

    def get_test(self):
        """Return the testing dataset.

        Returns:
            numpy.ndarray, numpy.ndarray

        """
        return self.X_test, self.Y_test

    def get_val(self):
        """Return the validation dataset.

        Returns:
            numpy.ndarray, numpy.ndarray

        """
        return self.X_val, self.Y_val

    def log_dataset_info(self):
        """Append some short information about the dataset to the logger.

        Returns:
            None

        """
        raise NotImplementedError()

    def load(self, limit=None):
        """Load the whole dataset into memory.

        Args:
            limit (int): Limit to this amount of images.

        Returns:
            None

        """
        raise NotImplementedError()

    def apply(self, fn):
        """Apply a function onto the entire dataset.

        Args:
            fn (function): Function to be applied onto the dataset.

        Returns:
            None

        """
        # fn_ = np.vectorize(fn)
        self.X_train = fn(self.X_train)
        self.X_test = fn(self.X_test)
        self.X_val = fn(self.X_val)
        self.Y_train = fn(self.Y_train)
        self.Y_test = fn(self.Y_test)
        self.Y_val = fn(self.Y_val)

    def split(self, test_split=0.2, val_split=0.5):
        """Split the loaded dataset into training, test and validation sets.

        If self.rs (the random state) is set, it will be used for reproducible
        splitting.

        Args:
            test_split (float): Portion of the training dataset to set as
                testing.
            val_split (float): Portion of the testing dataset to set as
                validation.

        Returns:
            bool: True if split has occured, False if dataset was already
                split.

        """
        if self.is_split():
            return False

        X_len = len(self.X_train)
        Y_len = len(self.Y_train)

        final_val = test_split * val_split
        final_test = test_split - final_val
        final_train = 1 - final_test - final_val
        logger.info("Splitting dataset: "
            "train={0:.3g}, test={1:.3g}, val={2:.3g}.".format(
                final_train, final_test, final_val
            )
        )

        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            ms.train_test_split(self.X_train, self.Y_train,
            test_size=test_split, random_state=self.rs)
        self.X_test, self.X_val, self.Y_test, self.Y_val = \
            ms.train_test_split(self.X_test, self.Y_test,
            test_size=val_split, random_state=self.rs)

        logger.info("Final X array size: "
            "train={0:d}, test={1:d}, val={2:d} (total={3:d}).".format(
                len(self.X_train), len(self.X_test), len(self.X_val), X_len
            )
        )
        logger.info("Final Y array size: "
            "train={0:d}, test={1:d}, val={2:d} (total={3:d}).".format(
                len(self.Y_train), len(self.Y_test), len(self.Y_val), Y_len
            )
        )

        return True

    def load_images_from_dir(self, dir, limit=None):
        """Load an image dataset into memory and return it.

        Args:
            dir (str):
                Folder name to load from. Must be present inside the dataset
                directory.
            limit (int, optional):
                If set to a positive integer, will load only the first x amount
                of files in the directory.

        Returns:
            numpy.ndarray: Unsplit data for the model.
        """
        X = []
        path = os.path.join(self.path, dir)

        if not os.path.isdir(path):
            raise IOError('"{0}": Not a directory.'.format(path))

        logger.info("Loading images from '{0}'.".format(
            click.format_filename(path)
        ))

        images = os.listdir(path)

        images_valid = []
        for im in images:
            im_path = os.path.join(path, im)
            if not utils.valid_image(im_path):
                continue
            images_valid.append(im_path)
        images_valid.sort()

        if type(limit) is int:
            images_valid = images_valid[0:limit]

        with click.progressbar(
            label='Loading images...',
            length=len(images_valid),
            show_pos=True
        ) as pbar:
            for im in images_valid:
                im = utils.load_image(im)
                X.append(im)
                pbar.update(1)

        logger.info("{0:d} images loaded from '{1}'.".format(
            len(X),
            click.format_filename(path)
        ))

        return np.array(X)

    class DatasetException(Exception):
        """Base class for other exceptions."""

        pass
