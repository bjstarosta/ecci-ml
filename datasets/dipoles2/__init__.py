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

    def setup(self, limit=None):
        if limit is not None and limit < 1:
            raise datasets.DatasetException('Limit must be at least 1.')

        # raw experimental images
        self.x = self._list_images('dipoles2_noise', limit)
        # ground truth
        self.y = self._list_images('dipoles2', limit)

        self.on_epoch_end()

    def load_data(self, batch_x, batch_y):
        return (
            self._load_images('dipoles2_noise', batch_x),
            self._load_images('dipoles2', batch_y)
        )
