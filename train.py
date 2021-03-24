# -*- coding: utf-8 -*-
"""Training interface.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import logging
import textwrap
import resource

import numpy as np
import click
import sklearn.model_selection

import lib.logger
import lib.tf
import lib.utils
import datasets
import models


logger = lib.logger.logger
lib.logger.start_stream_log()


_train_click_options = [
    click.argument('model', nargs=1, type=str),
    click.argument('dataset', nargs=1, type=str),
    click.option(
        '-lr',
        '--learning-rate',
        type=float,
        default=lib.tf.train_def_options['learning_rate'],
        show_default=True,
        help="""Learning rate."""
    ),
    click.option(
        '-bs',
        '--batch-size',
        type=int,
        default=lib.tf.train_def_options['batch_size'],
        show_default=True,
        help="""Batch size."""
    ),
    click.option(
        '-e',
        '--epochs',
        type=int,
        default=lib.tf.train_def_options['epochs'],
        show_default=True,
        help="""Number of epochs to train for."""
    ),
    click.option(
        '-s',
        '--seed',
        type=int,
        default=None,
        show_default='current UTC date in integer format, e.g. 1012020',
        help="""Random number generator seed."""
    ),
    click.option(
        '-r',
        '--revision',
        type=str,
        default=None,
        help="""Revision ID of model to load for updating weights.
            A new model revision will be created if this is not set."""
    ),
    click.option(
        '-ne',
        '--no-early-stopping',
        is_flag=True,
        help="""Flag for disabling early stopping functionality. Setting this
            means the model will always train for the set number of epochs,
            instead of stopping after reaching a threshold for loss
            improvement."""
    )
]


def train_click_options(func):
    for option in reversed(_train_click_options):
        func = option(func)
    return func


def load_dataset(ds_str, batch_size, flags, seed=None):
    dslist = ds_str.split(',')
    dsloaded = []
    for i, ds in enumerate(dslist):
        if not datasets.dataset_exists(ds):
            raise click.UsageError(
                "Dataset '{0}' does not exist.".format(ds)
            )

        logger.info('Loading dataset: {0} ({1}/{2})'.format(
            ds, i + 1, len(dslist)
        ))
        try:
            ds = datasets.load_dataset(ds, seed)
            ds.batch_size = batch_size
            if 'sanity-test' in flags:
                ds.setup(batch_size)
            else:
                ds.setup()

            dsloaded.append(ds)

        except Exception:
            logger.error("Unrecoverable error.", exc_info=True)
            exit(1)

    logger.info('{0} datasets loaded successfully.'.format(len(dslist)))

    if len(dslist) > 1:
        dataset = datasets.DatasetCollection(seed)
        dataset.batch_size = batch_size
        for ds in dsloaded:
            dataset.add(ds)
    else:
        dataset = dsloaded[0]

    logger.info('{0} training batches in total (batch size={1}).'.format(
        len(dataset), batch_size
    ))

    return dataset


def split_dataset(ds, test_split, val_split, flags):
    # Set up test mode if requested
    if 'sanity-test' in flags:
        logger.info('Test mode split enabled.')
        ds_test = ds
        ds_val = ds
        return ds, ds_test, ds_val

    logger.info(
        'Splitting dataset: train={0}, test={1}, val={2}.'.format(
            1 - test_split,
            test_split - (test_split * val_split),
            test_split * val_split
        )
    )

    ds_test = ds.split(test_split)
    if val_split > 0:
        ds_val = ds_test.split(val_split)
    else:
        ds_val = None

    return ds, ds_test, ds_val


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
    """Training supervisor."""
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
@train_click_options
@click.option(
    '-ts',
    '--test-split',
    type=float,
    default=0.2,
    show_default=True,
    help="""Fraction of loaded dataset to split off as test data."""
)
@click.option(
    '-vs',
    '--val-split',
    type=float,
    default=0.5,
    show_default=True,
    help="""Fraction of test dataset to split off as validation data."""
)
@click.option(
    '-t',
    '--test',
    is_flag=True,
    help="""Flag for sanity test mode (training using a dataset with a single
        image)."""
)
@click.pass_context
def run(ctx, **kwargs):
    """Train the selected model using the selected dataset.

    You can pass multiple datasets into the DATASET argument by separating
    the names using a comma (,). E.g.: dataset1,dataset2. This will cause
    multiple datasets to be loaded in as a collection. You must ensure that
    the dataset data structure is cross-compatible, otherwise the training
    will fail."""
    if not models.model_exists(kwargs['model']):
        raise click.UsageError(
            "Model '{0}' does not exist.".format(kwargs['model']),
            ctx=ctx
        )

    options = {
        'batch_size': kwargs['batch_size'],
        'epochs': kwargs['epochs'],
        'learning_rate': kwargs['learning_rate']
    }

    flags = []
    if kwargs['test'] is True:
        flags.append('sanity-test')
    if kwargs['no_early_stopping'] is True:
        flags.append('no-early-stopping')

    seed = lib.tf.set_seed(kwargs['seed'])
    dataset = load_dataset(
        kwargs['dataset'], kwargs['batch_size'], flags, seed
    )
    ds_train, ds_test, ds_val = split_dataset(
        dataset, kwargs['test_split'], kwargs['val_split'], flags
    )

    try:
        logger.info("Current process memory usage: {0:.3f} MB.".format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (10**3)
        ))
        lib.tf.train(
            kwargs['model'], kwargs['revision'],
            ds_train, ds_test, ds_val, seed, flags, options
        )
    except Exception:
        logger.error("Unrecoverable error.", exc_info=True)
        exit(1)


@main.command()
@train_click_options
@click.option(
    '-k',
    '--k-value',
    type=int,
    default=10,
    show_default=True,
    help="""Number of k-groups to split dataset into."""
)
@click.pass_context
def kfold(ctx, **kwargs):
    """Run K-Fold validation on the selected model."""
    if not models.model_exists(kwargs['model']):
        raise click.UsageError(
            "Model '{0}' does not exist.".format(kwargs['model']),
            ctx=ctx
        )

    model = models.load_model(kwargs['model'])

    options = {
        'batch_size': kwargs['batch_size'],
        'epochs': kwargs['epochs'],
        'learning_rate': kwargs['learning_rate']
    }

    flags = ['no-save']
    if kwargs['no_early_stopping'] is True:
        flags.append('no-early-stopping')

    seed = lib.tf.set_seed(kwargs['seed'])
    dataset = load_dataset(
        kwargs['dataset'], kwargs['batch_size'], flags, seed
    )

    kfold = sklearn.model_selection.KFold(
        n_splits=kwargs['k_value'], shuffle=True, random_state=seed
    )
    fold_no = 1
    metrics_all = []
    for idx_train, idx_test in kfold.split(dataset):
        try:
            logger.info("Current process memory usage: {0:.3f} MB.".format(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (10**3)
            ))
            logger.info("Training run {0}/{1}.".format(
                fold_no, kwargs['k_value']
            ))

            ds_train = dataset.slice(idx_train)
            ds_test = dataset.slice(idx_test)

            model_nn, metrics = lib.tf.train(
                kwargs['model'], ds_train, ds_test, None, seed, flags, options
            )
            metrics_all.append(metrics)

        except Exception:
            logger.error("Unrecoverable error.", exc_info=True)
            exit(1)

        fold_no += 1

    logger.info("Averaged metrics:")
    final_metrics = np.average(metrics_all, axis=0)
    model.metrics(final_metrics, logger)

    final_metrics_std = np.std(metrics_all, axis=0)
    for i, std in enumerate(final_metrics_std):
        logger.info('STD ({0}): {1:.6f}'.format(i, std))


@main.command()
@click.pass_context
def list_datasets(ctx, **kwargs):
    """List available datasets."""
    lst = datasets.list_datasets(True)
    for m in lst:
        id, doc = m
        doc = textwrap.indent(doc, "       ")
        print("  •  {0}:\n{1}".format(id, doc))


@main.command()
@click.pass_context
def list_models(ctx, **kwargs):
    """List available trainable models."""
    lst = models.list_models(True)
    for m in lst:
        id, doc = m
        doc = textwrap.indent(doc, "       ")
        print("  •  {0}:\n{1}".format(id, doc))


@main.command()
@click.argument('model', nargs=1, type=str)
@click.pass_context
def summarise(ctx, **kwargs):
    """Print a summary of the passed model."""
    if not models.model_exists(kwargs['model']):
        raise click.UsageError(
            "Model '{0}' does not exist.".format(kwargs['model']),
            ctx=ctx
        )

    model = models.load_model(kwargs['model'])
    model_nn = model.build(lib.tf.train_def_options['learning_rate'])
    model_nn.summary()


if __name__ == '__main__':
    main(obj={})
