# -*- coding: utf-8 -*-
"""Dipoles3_Noise dataset.

Generated 16/10/2020.
Dataset mapping multiple dislocations per image onto a diagram with
dislocation centres marked in the labels. SEM noise is added on top.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import datasets


class Dipoles3_Noise(datasets.Dataset):

    def __init__(self):
        super().__init__()

        self.desc = 'Dipole to Marked TDs diagram dataset (SEM noise)'
        self.generated = '2020-11-17'

    def load(self, limit=None):
        if limit is not None and limit < 1:
            raise datasets.DatasetException('Limit must be at least 1.')
        # experimental approximation dataset
        self.X_train = self.load_images_from_dir('dipoles3', limit)
        # voronoi diagram dataset
        self.Y_train = self.load_images_from_dir('dipoles3_labels', limit)
