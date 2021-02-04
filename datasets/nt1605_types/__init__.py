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

    def load(self, limit=None):
        if limit is not None and limit < 1:
            raise datasets.DatasetException('Limit must be at least 1.')
        # experimental approximation dataset
        self.X_train = self.load_images_from_dir('exp', limit)
        # voronoi diagram dataset
        self.Y_train = self.load_images_from_dir('ground', limit)
