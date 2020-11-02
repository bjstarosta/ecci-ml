# -*- coding: utf-8 -*-
"""Dipoles_test dataset.

Generated 09/08/2020 for the conv. autoencoder model (denoising dislocations).
A test dataset containing only 12 images.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import numpy as np
import datasets


class Dipoles_test(datasets.Dataset):

    def __init__(self):
        super().__init__()

        self.desc = 'Test dipole dataset for convolutional autoencoder'
        self.generated = '2020-08-09'

    def load(self, limit=None):
        if limit is not None and limit < 1:
            raise datasets.DatasetException('Limit must be at least 1.')
        # noisy dataset
        self.X_train = self.load_images_from_dir('dipoles_test_noise', limit)
        # clean dataset
        self.Y_train = self.load_images_from_dir('dipoles_test', limit)
