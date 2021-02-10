# -*- coding: utf-8 -*-
"""Dipoles3 dataset.

Generated 16/10/2020.
Dataset mapping multiple dislocations per image onto a voronoi diagram with
dislocation centres marked in the labels.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import datasets


class Dipoles3(datasets.Dataset):

    def __init__(self):
        super().__init__()

        self.desc = 'Dipole to Voronoi diagram dataset'
        self.generated = '2020-10-16'

    def setup(self, limit=None):
        if limit is not None and limit < 1:
            raise datasets.DatasetException('Limit must be at least 1.')

        # raw experimental images
        self.x = self._list_images('dipoles3', limit)
        # ground truth
        self.y = self._list_images('dipoles3_labels', limit)

        self.on_epoch_end()

    def load_data(self, batch_x, batch_y):
        return (
            self._load_images('dipoles3', batch_x),
            self._load_images('dipoles3_labels', batch_y)
        )
