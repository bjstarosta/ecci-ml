# -*- coding: utf-8 -*-
"""Extract circle coordinates and radius into CSV file using Hough transform.

Author: Bohdan Starosta
University of Strathclyde Physics Department
"""

# In [ ]

# import click
import numpy as np
import skimage.feature
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.collections

import lib.logger
import lib.utils as utils
import lib.tf


# In [ ]

logger = lib.logger.logger
lib.logger.start_stream_log()

# In [ ]

# images = [
#     '../fusionnet_20201203/predict_test_out/M2_5K_test1.TIF',
#     '../fusionnet_20201203/predict_test_out/M2_5K_test2.TIF',
#     '../fusionnet_20201203/predict_test_out/M5_5K_test1.TIF',
#     '../fusionnet_20201203/predict_test_out/M5_5K_test2.TIF',
#     '../fusionnet_20201203/predict_test_out/M55_5K_test2.TIF',
#     '../fusionnet_20201203/predict_test_out/M55_5K_test3.TIF',
# ]
# im = []
# for i in images:
#     im.append(utils.load_image(i))

image = utils.load_image('M2_5K.TIF')

# Get rid of extraneous dimension.
if len(image.shape) == 3:
    image = np.squeeze(image)

hw = (512, 512)  # height, width of sliding window
stride = (256, 256)  # row, column of sliding window stride

# Calculate proper padding so that the predictions can be stiched together
l_pad = stride[1]
t_pad = stride[0]
r_pad = stride[1] - (image.shape[1] % stride[1])
b_pad = stride[0] - (image.shape[0] % stride[0])
if r_pad % hw[1] > 0:
    r_pad += stride[1]
if b_pad % hw[0] > 0:
    b_pad += stride[0]

padding = (l_pad, t_pad, r_pad, b_pad)
print('l_pad, t_pad, r_pad, b_pad: ', padding)
print('image.shape (before pad): ', image.shape)

# Pad image to get correct stride coverage
image_padded = np.pad(image, ((t_pad, b_pad), (l_pad, r_pad)), 'constant',
    constant_values=((0, 0), (0, 0)))
print('image.shape (after pad): ', image.shape)

passes = np.zeros((
    int(image_padded.shape[0] / stride[0]),
    int(image_padded.shape[1] / stride[1])
))
print('passes.shape: ', passes.shape)

# In [ ]

# Calculate sliding window.
X = []
windows_mpl = []
for r in range(0, image_padded.shape[0] - stride[0], stride[0]):
    for c in range(0, image_padded.shape[1] - stride[1], stride[1]):

        r_ = int(r / stride[0])
        c_ = int(c / stride[1])
        passes[r_:r_ + 2, c_:c_ + 2] += 1

        X.append(image_padded[r:r + hw[0], c:c + hw[1]])
        windows_mpl.append(matplotlib.patches.Rectangle((c, r), hw[1], hw[0]))

X = np.array(X)
print('X.shape: ', X.shape)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16, 8))

i = 0
j = 0
for r in range(0, passes.shape[0]):
    for c in range(0, passes.shape[1]):
        r_ = int(r * stride[0])
        c_ = int(c * stride[1])
        ax.annotate("{0}".format(int(passes[r, c])), (c_, r_),
            xytext=(12, -12), textcoords='offset pixels',
            color='w', fontsize=10, ha='center', va='center')

p = matplotlib.collections.PatchCollection(windows_mpl, alpha=0.5)
p.set_edgecolor([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
p.set_facecolor('none')
ax.add_collection(p)
ax.imshow(image_padded, cmap=plt.cm.gray)

# In [ ]

# Perform predictions
try:
    Y = lib.tf.predict(X, 'fusionnet', ('fusionnet', '20201216_1'))
except Exception:
    logger.error("Unrecoverable error.", exc_info=True)
    exit(1)

print('Y.shape: ', Y.shape)

# In [ ]

# Stitch the predictions into 4 separate blob images.
# Find blobs as well while iterating over the predictions.
n_row = int(image_padded.shape[0] / stride[0]) - 1
n_col = int(image_padded.shape[1] / stride[1]) - 1

lo_row = np.floor(n_row / 2)
hi_row = np.ceil(n_row / 2)
lo_col = np.floor(n_col / 2)
hi_col = np.ceil(n_col / 2)

blob_l = np.zeros((int(hw[0] * hi_row), int(hw[1] * hi_col)))
blob_t = np.zeros((int(hw[0] * hi_row), int(hw[1] * lo_col)))
blob_r = np.zeros((int(hw[0] * lo_row), int(hw[1] * hi_col)))
blob_b = np.zeros((int(hw[0] * lo_row), int(hw[1] * lo_col)))
print('blob_l.shape, blob_t.shape, blob_r.shape, blob_b.shape: ')
print(blob_l.shape, blob_t.shape, blob_r.shape, blob_b.shape)
print()


min_r = 5  # minimum blob radius
max_r = 14  # maximum blob radius
tds = []

row = 0
col = 0
for i, Y_ in enumerate(Y):
    print('ROW, COL: ', row, col)
    Y_ = np.squeeze(Y_)

    if i % 2 == 0:
        x_i = col * stride[1]
    else:
        x_i = col * stride[1] - stride[1]
    # print('x_i:', x_i)

    if row % 2 == 0:
        y_i = row * stride[0]
        # print('y_i:', y_i)
        if i % 2 == 0:  # left
            blob_l[y_i:y_i + hw[0], x_i:x_i + hw[1]] = Y_
        else:  # top
            blob_t[y_i:y_i + hw[0], x_i:x_i + hw[1]] = Y_
    else:
        y_i = row * stride[0] - stride[0]
        # print('y_i:', y_i)
        if i % 2 == 0:  # bottom
            blob_b[y_i:y_i + hw[0], x_i:x_i + hw[1]] = Y_
        else:  # right
            blob_r[y_i:y_i + hw[0], x_i:x_i + hw[1]] = Y_

    # Blob detection begins here.
    # This line is what slows down this loop.
    blobs_log = skimage.feature.blob_log(Y_,
        min_sigma=3, max_sigma=15, num_sigma=15, threshold=.1)

    # Compute blob radius and clip its values.
    blobs_log[:, 2] = np.clip(blobs_log[:, 2] * np.sqrt(2), min_r, max_r)

    for c_y, c_x, r in blobs_log:
        # Extract all pixel values within the blob radius.
        c_y = int(c_y)
        c_x = int(c_x)
        r_i = int(r)
        sq = Y_[c_y - r_i:c_y + r_i, c_x - r_i:c_x + r_i]

        pred_n = 0
        pred = 0
        for sq_r, sq_r_ in enumerate(sq):
            for sq_c, sq_c_ in enumerate(sq_r_):
                if (sq_r - r_i)**2 + (sq_c - r_i)**2 > r_i**2:
                    continue
                pred += sq[sq_r, sq_c]
                pred_n += 1

        # Average of pixel values within each blob radius is set as the
        # prediction confidence of that marker. This is because the
        # autoencoder delivers fainter blobs the more "unsure" it is.
        if pred_n > 0:
            pred = (pred / pred_n) / 255
        else:
            pred = 0

        c_y_ = c_y + (row * stride[0]) - padding[1]
        c_x_ = c_x + (col * stride[1]) - padding[0]
        tds.append((c_y_, c_x_, r, pred))
    # Blob detection ends here.

    col += 1
    if col >= n_col:
        col = 0
        row += 1

print('TDs found initially: ', len(tds))

# In [ ]

td_border = 3  # get rid of all TDs within this many pixels of the border
tolerance = 2  # allow this many pixels of overlap

tds_pruned = []

# First pass for bad candidates
for i, td in enumerate(tds):
    (y, x, r, pred) = td

    # Prune border TDs as they are largely artifacts
    if (x <= td_border or x >= (image.shape[1] - td_border)
    or y < td_border or y >= (image.shape[0] - td_border)):
        continue

    # Prediction < 0.01 means pruned
    if pred < 0.01:
        continue

    tds_pruned.append(td)

# Second pass for prediction ranking

# All dislocations should intersect four times with a very close neighbour
# We find those four times and use the one with highest pred then average
# the pred.
tds_final = []
tds_visited = []

for i, td in enumerate(tds_pruned):
    if i in tds_visited:
        continue

    overlap = [td]

    # find all overlapping TDs
    for j, td_ in enumerate(tds_pruned):
        dx = td[1] - td_[1]
        dy = td[0] - td_[0]
        d = np.hypot(dx, dy)
        if d >= (td[2] + td_[2] - tolerance):
            continue

        overlap.append(td_)
        tds_visited.append(j)

    # sort by prediction confidence
    overlap = sorted(overlap, key=lambda x: x[3], reverse=True)

    # prediction confidence should always be an average of 4 overlapping
    overlap = overlap[:4]
    overlap += [(None, None, None, 0)] * (4 - len(overlap))
    pred = np.average([x[3] for x in overlap])

    tds_final.append((overlap[0][0], overlap[0][1], overlap[0][2], pred))

print('TDs after pruning: ', len(tds_final))


# In [ ]

# Crop the predictions.
offsets = [(t_pad, l_pad), (t_pad, 0), (0, 0), (0, l_pad)]
bc = 8  # crop this many pixels off the border as well

blobs = [blob_l, blob_t, blob_r, blob_b]
for i, blob in enumerate(blobs):
    top = offsets[i][0]
    left = offsets[i][1]
    bottom = top + image.shape[0]
    right = left + image.shape[1]
    blobs[i] = blob[top + bc:bottom - bc, left + bc:right - bc]


fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(20, 16), sharey=True)
ax[0, 0].imshow(blobs[0], cmap=plt.cm.gray)
ax[0, 0].set_title('Left')
ax[0, 1].imshow(blobs[1], cmap=plt.cm.gray)
ax[0, 1].set_title('Top')
ax[1, 1].imshow(blobs[2], cmap=plt.cm.gray)
ax[1, 1].set_title('Right')
ax[1, 0].imshow(blobs[3], cmap=plt.cm.gray)
ax[1, 0].set_title('Bottom')

# In [ ]

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 12))
patches_mpl = []
colours = []
td_n = 0
for td in tds_final:
    if td[3] < 0.33:
        continue

    ax.annotate("{:.2f}".format(td[3]), (td[1], td[0]),
        xytext=(0, 12), textcoords='offset pixels',
        color='w', fontsize=10, ha='center', va='center')

    colours.append(td[3])
    patches_mpl.append(
        matplotlib.patches.Circle((td[1], td[0]), td[2]))
    td_n += 1

p = matplotlib.collections.PatchCollection(patches_mpl,
    cmap=matplotlib.cm.jet_r, alpha=0.5)
p.set_array(np.array(colours))
p.set_clim([0, 1])
p.set_label('Confidence')
ax.add_collection(p)
fig.colorbar(p)

ax.imshow(image, cmap=plt.cm.gray)
fig.suptitle('TD location prediction',
    size=24, weight='bold')
ax.set_title('{0} TDs found. Numbers indicate prediction confidence.'.format(
    td_n))
fig.tight_layout()

plt.show()
