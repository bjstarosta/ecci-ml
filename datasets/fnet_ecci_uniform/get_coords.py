# -*- coding: utf-8 -*-
"""Extract circle coordinates and radius into CSV file using Hough transform.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

import os
import sys
import re
import csv
from xml.dom import minidom

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.collections

if __name__ == '__main__':
    sys.path.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

import lib.image as image


images = [
    ['M2_5K01.svg', 'M2_5K01_original.tif'],
    ['naresh_slide3.svg', 'naresh_slide3_original.tif']
]
imdom = []
for i, j in images:
    imdom.append(minidom.parse(i))

for i, dom in enumerate(imdom):
    # get first path element's 'd' attribute which holds all the path data
    attr = dom.getElementsByTagName('path')[0].getAttribute('d')

    # split commands
    m = re.findall(r'(?P<cmd>[a-zA-Z]{1,2})([0-9-,.\s]+)\s', attr, flags=0)
    cmds = []
    for match in m:
        prm = []
        for p in match[1].split(' '):
            prm.append(p.split(','))
        cmds.append([match[0]] + prm)

    # parse the commands (only need to deal with cubic splines)
    circles = []
    tmp = []
    for c in cmds:
        # start/end+start path
        if c[0] == 'M' or c[0] == 'ZM':
            if len(tmp) > 0:
                # convert consecutive path points to a centroid and radius
                tmp = np.array(tmp, dtype=np.float32)
                x = np.average(tmp[:, 0])
                y = np.average(tmp[:, 1])
                r = np.linalg.norm(np.amax(tmp, axis=0) - np.array([x, y]))
                circles.append([x, y, r])
            tmp = [c[1]]
        # continue path
        elif c[0] == 'C':
            tmp.append(c[3])

    # normalise the radius
    # for j, c in enumerate(circles):
    #     if c[2] > 5:
    #         c[2] = 5
    #     circles[j] = c

    # save coords to csv
    csv_path = os.path.splitext(images[i][0])[0] + '.csv'
    with open(csv_path, mode='w') as f:
        w = csv.DictWriter(f,
            fieldnames=['x', 'y', 'r'], delimiter=',', quotechar='"')
        w.writeheader()
        for c in circles:
            w.writerow({'x': c[0], 'y': c[1], 'r': c[2]})

    # show figure with markings
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    patches = []
    for x, y, r in circles:
        patches.append(matplotlib.patches.Circle((x, y), r))
    p = matplotlib.collections.PatchCollection(patches, alpha=0.5)
    p.set_color((1, 0, 0))
    ax.add_collection(p)
    ax.imshow(image.load_image(images[i][1]), cmap=plt.cm.gray)

for i in imdom:
    i.unlink()

plt.show()
