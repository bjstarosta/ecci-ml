# -*- coding: utf-8 -*-
"""Training interface.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import logging
import textwrap
import resource

import click

import lib.logger
import lib.tf
import lib.utils
import datasets
import models


logger = lib.logger.logger
lib.logger.start_stream_log()


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
@click.argument('model', nargs=1, type=str)
@click.argument('dataset', nargs=1, type=str)
@click.option(
    '-ts',
    '--test-split',
    type=float,
    default=lib.tf.train_def_options['test_size'],
    show_default=True,
    help="""Fraction of loaded dataset to split off as test data."""
)
@click.option(
    '-vs',
    '--val-split',
    type=float,
    default=lib.tf.train_def_options['val_size'],
    show_default=True,
    help="""Fraction of test dataset to split off as validation data."""
)
@click.option(
    '-lr',
    '--learning-rate',
    type=float,
    default=lib.tf.train_def_options['learning_rate'],
    show_default=True,
    help="""Learning rate."""
)
@click.option(
    '-bs',
    '--batch-size',
    type=int,
    default=lib.tf.train_def_options['batch_size'],
    show_default=True,
    help="""Batch size."""
)
@click.option(
    '-e',
    '--epochs',
    type=int,
    default=lib.tf.train_def_options['epochs'],
    show_default=True,
    help="""Number of epochs to train for."""
)
@click.option(
    '-s',
    '--seed',
    type=int,
    default=None,
    show_default='current UTC date in integer format, e.g. 1012020',
    help="""Random number generator seed."""
)
@click.option(
    '-o',
    '--overwrite',
    is_flag=True,
    help="""Flag for overwriting previously trained model instead of adding
        to it."""
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

    flags = []
    if kwargs['overwrite'] is True:
        flags.append('overwrite-model')
    if kwargs['test'] is True:
        flags.append('sanity-test')

    dslist = kwargs['dataset'].split(',')
    dsloaded = []
    for ds in dslist:
        if not datasets.dataset_exists(ds):
            raise click.UsageError(
                "Dataset '{0}' does not exist.".format(ds),
                ctx=ctx
            )

        try:
            ds = datasets.load_dataset(ds)
            ds.batch_size = kwargs['batch_size']
            if 'sanity-test' in flags:
                ds.setup(kwargs['batch_size'])
            else:
                ds.setup()

            dsloaded.append(ds)

        except Exception:
            logger.error("Unrecoverable error.", exc_info=True)
            exit(1)

    if len(dslist) > 1:
        dataset = datasets.DatasetCollection()
        for ds in dsloaded:
            dataset.add(ds)
    else:
        dataset = dslist[0]

    options = {
        'batch_size': kwargs['batch_size'],
        'epochs': kwargs['epochs'],
        'learning_rate': kwargs['learning_rate'],
        'test_size': kwargs['test_split'],
        'val_size': kwargs['val_split']
    }

    try:
        logger.info("Current process memory usage: {0:.3f} MB.".format(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (10**3)
        ))

        lib.tf.train(
            dataset, kwargs['model'], kwargs['seed'], flags, options)
    except Exception:
        logger.error("Unrecoverable error.", exc_info=True)
        exit(1)


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
