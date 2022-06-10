# -*- coding: utf-8 -*-
"""nanodash_pos_202205 dataset.

Generated 01/12/2020.
Experimental images of GaN with marked ground truth, augmented.
Some images by Naresh Gunasekar
Ground truth discriminates positions only.

Author: SSD Group, Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import numpy as np
import h5py
import datasets
import datasets.preprocessing as pp
import datasets.filters as ppf


class nanodash_pos(datasets.Dataset):

    def __init__(self):
        super().__init__()

        self.desc = 'Experimental images of GaN w/ ground truth, 2nd rev.'
        self.generated = '2020-12-01'
        self.hdf5file = 'nanodash_pos_202205.hdf5'

        self._fh = None

    def setup(self, limit=None):
        if limit is not None and limit < 1:
            raise datasets.DatasetException('Limit must be at least 1.')

        if self._fh is None:
            fhpath = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self.hdf5file
            )
            self._fh = h5py.File(fhpath, 'r')

        # raw experimental images
        # self.x = self._list_images('exp', limit)
        self.x = range(0, len(self._fh['root']['exp']))
        # ground truth
        # self.y = self._list_images('ground', limit)
        self.y = range(0, len(self._fh['root']['ground']))

        self.on_epoch_end()

    def load_data(self, batch_x, batch_y):
        # return (
        #     self._load_images('exp', batch_x, 'uint8', 'gs', preprocess=True),
        #     self._load_images('ground', batch_y, 'uint8', 'gs')
        # )
        return (
            self._load_images_hdf5(self._fh['root']['exp'], batch_x,
                preprocess=True),
            self._load_images_hdf5(self._fh['root']['ground'], batch_y)
        )

    def generate_dataset(self, ctx=None):
        opts = {
            'crop': True,
            'crop_px': (2, 2, 2, 2),
            'rotate': True,
            'rot_directions': [0, np.pi / 3, 2 * np.pi / 3, np.pi],
            'rot_pad': 'reflect',
            'flip_h': True,
            'flip_v': True,
            'split_chunks': False,
            'chunk_size': (256, 256),
            'chunk_stride': (128, 128),
            'chunk_cutoff': True,
            'chunk_padmode': 'reflect',
            'chunk_origin': 'middle',
            'hdf5_save_mode': 'grayscale',
            'hdf5_save_type': 'uint8',
            'processes': 12
        }

        gen = pp.DatasetGenerator(self.id,
            [
                ('M2_5K01_original.tif', 'M2_5K01_ground.tif'),
                ('M6_5K_original.tif', 'M6_5K_ground.png'),
                ('naresh_slide3_original.tif', 'naresh_slide3_ground.tif'),
            ],
            'source', 'exp', 'ground', self.hdf5file, opts
        )
        gen.add_filter(ppf.gamma, {'values': [1.5, 0.75]}, False)
        gen.run(ctx)
