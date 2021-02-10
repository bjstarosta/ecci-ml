# -*- coding: utf-8 -*-
"""Dipoles_test dataset.

Generated 09/08/2020 for the conv. autoencoder model (denoising dislocations).
A test dataset containing only 12 images.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import datasets


class Dipoles_test(datasets.Dataset):

    def __init__(self):
        super().__init__()

        self.desc = 'Test dipole dataset for convolutional autoencoder'
        self.generated = '2020-08-09'

    def setup(self, limit=None):
        if limit is not None and limit < 1:
            raise datasets.DatasetException('Limit must be at least 1.')

        # raw experimental images
        self.x = self._list_images('dipoles_test_noise', limit)
        # ground truth
        self.y = self._list_images('dipoles_test', limit)

        self.on_epoch_end()

    def load_data(self, batch_x, batch_y):
        return (
            self._load_images('dipoles_test_noise', batch_x),
            self._load_images('dipoles_test', batch_y)
        )
