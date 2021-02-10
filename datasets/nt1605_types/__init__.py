# -*- coding: utf-8 -*-
"""nt1605_types dataset.

Generated 03/02/2021.
Experimental images of AlGaN with marked ground truth, augmented.
Ground truth discriminates types.

Author: SSD Group, Bohdan Starosta
University of Strathclyde Physics Department
"""

import datasets


class nt1605_types(datasets.Dataset):

    def __init__(self):
        super().__init__()

        self.desc = ('Experimental images of AlGaN (15% AlN, Tyndall) w/'
        + ' ground truth (incl. TD types)')
        self.generated = '2021-02-03'

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
