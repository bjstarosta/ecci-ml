# -*- coding: utf-8 -*-
"""Dipoles dataset.

Generated 21/07/2020 for the conv. autoencoder model (denoising dislocations).

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import numpy as np
import datasets


class Dipoles(datasets.Dataset):

    def __init__(self):
        super().__init__()

        self.desc = 'Noisy dipole dataset for convolutional autoencoder'
        self.generated = '2020-07-20'

    def load(self, limit=None):
        if limit is not None:
            if limit < 4:
                raise datasets.DatasetException('Limit must be at least 4.')
            limit = np.floor(limit / 4)
        # noisy dataset
        self.X_train = np.array(
            self.load_images_from_dir('dipoles_hc_noise', limit)
            + self.load_images_from_dir('dipoles_lc_noise', limit)
            + self.load_images_from_dir('dipoles_vlc_noise', limit)
            + self.load_images_from_dir('constants_noise', limit)
        )
        # clean dataset
        self.Y_train = np.array(
            self.load_images_from_dir('dipoles_hc', limit)
            + self.load_images_from_dir('dipoles_lc', limit)
            + self.load_images_from_dir('dipoles_vlc', limit)
            + self.load_images_from_dir('constants', limit)
        )
