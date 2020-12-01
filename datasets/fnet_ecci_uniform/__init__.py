# -*- coding: utf-8 -*-
"""FNET_ECCI_Uniform dataset.

Generated 01/12/2020.
Experimental images of GaN with marked ground truth, augmented.
Ground truth regenerated in an attempt to produce smaller blobs.

Author: SSD Group, Bohdan Starosta
University of Strathclyde Physics Department
"""

import datasets


class FNET_ECCI_Uniform(datasets.Dataset):

    def __init__(self):
        super().__init__()

        self.desc = 'Experimental images of GaN w/ ground truth, 2nd rev.'
        self.generated = '2020-12-01'

    def load(self, limit=None):
        if limit is not None and limit < 1:
            raise datasets.DatasetException('Limit must be at least 1.')
        # experimental approximation dataset
        self.X_train = self.load_images_from_dir('exp', limit)
        # voronoi diagram dataset
        self.Y_train = self.load_images_from_dir('ground', limit)
