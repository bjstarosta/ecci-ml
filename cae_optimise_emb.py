# -*- coding: utf-8 -*-
"""Convolutional autoencoder embedding size optimiser.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import ast
import logging

import click
import numpy as np
import matplotlib.pyplot as plt

import cae_train
import utils


class EvalOption(click.Option):
    """Extension of click's Option class to allow literal evals of params."""

    def type_cast_value(self, ctx, value):
        """Evaluate expression instead of type casting."""
        try:
            return ast.literal_eval(value)
        except Exception:
            raise click.BadParameter(value)


@click.command(name='Embedding size optimisation')
@click.option(
    '-s',
    '--sizes',
    cls=EvalOption,
    default="[10, 64, 128, 256, 512, 1024]",
    help="""Embedding sizes to test."""
)
@click.option(
    '--debug',
    is_flag=True,
    help="""Logs debug messages during script run."""
)
def main(**kwargs):
    """Train multiple autoencoders with specified embedding sizes."""
    LOG_FORMAT = '[%(levelname)s] %(message)s'
    if kwargs['debug'] is True:
        LOG_LEVEL = 'DEBUG'
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
    else:
        LOG_LEVEL = 'INFO'

    logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)

    try:
        fig_dir = 'comparisons/embedding/'
        fig_file = os.path.join(fig_dir, 'cae_embsize{0}.png')
        model_file = 'cae_emb{0}.h5'
        ckpt_dir = 'cae_model/checkpoints_emb{0}/'
        dataset_dir = cae_train._DEFAULTS['dataset_dir']

        utils.setup_path(fig_dir)

        # Loading noisy dataset
        X = np.array(
            utils.load_dataset(os.path.join(dataset_dir,
                'dipoles_hc_noise'))
            + utils.load_dataset(os.path.join(dataset_dir,
                'dipoles_lc_noise'))
            + utils.load_dataset(os.path.join(dataset_dir,
                'dipoles_vlc_noise'))
        )
        # Loading clean dataset
        Y = np.array(
            utils.load_dataset(os.path.join(dataset_dir,
                'dipoles_hc'))
            + utils.load_dataset(os.path.join(dataset_dir,
                'dipoles_lc'))
            + utils.load_dataset(os.path.join(dataset_dir,
                'dipoles_vlc'))
        )

        options = cae_train._DEFAULTS
        for emb_size in kwargs['sizes']:

            logging.info(
                'Training autoencoder with embedding size: {0} units.'.format(
                    emb_size))

            options['model_file'] = model_file.format(emb_size)
            options['checkpoint_dir'] = ckpt_dir.format(emb_size)
            options['emb_size'] = emb_size
            autoencoder, test = cae_train.train(
                X, Y,
                lr=options['learning_rate'],
                batch=options['batch_size'],
                epochs=options['epochs'],
                split=options['test_size'],
                seed=options['seed'],
                options=options
            )

            cae_train.visualise(autoencoder, test[0],
                'Image prediction (embedding size: {0})'.format(emb_size))
            plt.savefig(fig_file.format(emb_size), dpi=300)

    except KeyboardInterrupt:
        print('EXIT')


if __name__ == '__main__':
    main()
