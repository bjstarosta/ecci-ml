# -*- coding: utf-8 -*-
"""Show coords saved in csv file on the original image.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import sys
import csv

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.collections

if __name__ == '__main__':
    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

import lib.image as image


images = [
    ['M2_5K01.csv', 'M2_5K01_original.tif'],
    ['naresh_slide3.csv', 'naresh_slide3_original.tif']
]

for i, j in images:
    circles = []
    with open(i) as f:
        line = 0
        for row in csv.reader(f, delimiter=','):
            if line > 0:
                circles.append(row)
            line += 1

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    patches = []
    for x, y, r in circles:
        patches.append(matplotlib.patches.Circle((float(x), float(y)), float(r)))
    p = matplotlib.collections.PatchCollection(patches, alpha=0.5)
    p.set_color((1, 0, 0))
    ax.add_collection(p)
    ax.imshow(image.load_image(j), cmap=plt.cm.gray)

plt.show()
