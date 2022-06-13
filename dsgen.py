# -*- coding: utf-8 -*-
"""Dataset generation interface.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import logging

import click

import lib.logger
import datasets


logger = lib.logger.logger
lib.logger.start_stream_log()


@click.command()
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
@click.option(
    '-d',
    '--dataset',
    required=True,
    type=str,
    help="""Dataset name to regenerate."""
)
@click.pass_context
def main(ctx, **kwargs):
    """Dataset generation supervisor."""
    if kwargs['verbose'] is True:
        LOG_LEVEL = logging.DEBUG
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
    else:
        LOG_LEVEL = logging.INFO
    logger.setLevel(LOG_LEVEL)

    if kwargs['file_log'] is True:
        lib.logger.start_file_log()

    ctx.obj['verbose'] = kwargs['verbose']
    ctx.obj['logger'] = logger

    ds = kwargs['dataset']
    if not datasets.dataset_exists(ds):
        raise click.UsageError(
            "Dataset '{0}' does not exist.".format(ds)
        )

    try:
        ds = datasets.load_dataset(ds)
    except Exception:
        logger.error("Unrecoverable error.", exc_info=True)
        exit(1)

    ds.generate_dataset(ctx)


if __name__ == '__main__':
    main(obj={})
