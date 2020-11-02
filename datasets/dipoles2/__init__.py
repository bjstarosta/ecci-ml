# -*- coding: utf-8 -*-
"""Dipoles2 dataset.

Generated 18/09/2020 for the conv. autoencoder model.
Denoising multiple dislocations per image.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import datasets


class Dipoles2(datasets.Dataset):

    def __init__(self):
        super().__init__()

        self.desc = 'Noisy multi-dipole dataset for convolutional autoencoder'
        self.generated = '2020-09-18'

    def load(self, limit=None):
        if limit is not None and limit < 1:
            raise datasets.DatasetException('Limit must be at least 1.')
        # noisy dataset
        self.X_train = self.load_images_from_dir('dipoles2_noise', limit)
        # clean dataset
        self.Y_train = self.load_images_from_dir('dipoles2', limit)
