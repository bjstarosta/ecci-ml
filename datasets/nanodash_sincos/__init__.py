# -*- coding: utf-8 -*-
"""nanodash_sincos dataset.

Generated 01/12/2020.
Experimental images of GaN with marked ground truth, augmented.
Some images by Naresh Gunasekar
Ground truth discriminates positions only.
Sine cosine filter applied.

Author: SSD Group, Bohdan Starosta
University of Strathclyde Physics Department
"""

import datasets


class nanodash_sincos(datasets.Dataset):

    def __init__(self):
        super().__init__()

        self.desc = ('Experimental images of GaN w/ ground truth, 2nd rev.,'
            ' w/ sine cosine filter')
        self.generated = '2020-12-01'

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
            self._load_images('exp', batch_x, 'uint8', 'gs'),
            self._load_images('ground', batch_y, 'uint8', 'gs')
        )
