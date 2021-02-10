# -*- coding: utf-8 -*-
"""FNET_ECCI dataset.

Generated 18/11/2020.
Experimental images of GaN with marked ground truth, augmented.

Author: SSD Group, Bohdan Starosta
University of Strathclyde Physics Department
"""

import datasets


class FNET_ECCI(datasets.Dataset):

    def __init__(self):
        super().__init__()

        self.desc = 'Experimental images of GaN w/ ground truth, augmented'
        self.generated = '2020-11-18'

    def setup(self, limit=None):
        if limit is not None and limit < 1:
            raise datasets.DatasetException('Limit must be at least 1.')

        # raw experimental images
        self.x = self._list_images('exp', limit)
        # ground truth
        self.y = self._list_images('ground', limit)

        self.on_epoch_end()

    def load_data(self, batch_x, batch_y):
        return (
            self._load_images('exp', batch_x),
            self._load_images('ground', batch_y)
        )
