# In [ ]

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as backend_pdf

import lib.logger
import lib.tf
import lib.utils
import lib.image

logger = lib.logger.logger
lib.logger.start_stream_log()

# In [ ]

impath = '/home/bjs/python/disrecog/comy.png'
figpath = '/home/bjs/python/disrecog/diodefigs/'
predpath = '/home/bjs/python/disrecog/diodepredicts/'

newsize = [512, 819]
methods = [
    'bilinear',  # Bilinear interpolation. If antialias is true, becomes a
    # hat/tent filter function with radius 1 when downsampling.
    'lanczos3',  # Lanczos kernel with radius 3. High-quality practical filter
    # but may have some ringing, especially on synthetic images.
    'lanczos5',  # Lanczos kernel with radius 5. Very-high-quality filter but
    # may have stronger ringing.
    'bicubic',  # Cubic interpolant of Keys. Equivalent to Catmull-Rom kernel.
    # Reasonably good quality and faster than Lanczos3Kernel, particularly when
    # upsampling.
    'gaussian',  # Gaussian kernel with radius 3, sigma = 1.5 / 3.0.
    'nearest',  # Nearest neighbor interpolation. antialias has no effect when
    # used with nearest neighbor interpolation.
    'area',  # Anti-aliased resampling with area interpolation. antialias has
    # no effect when used with area interpolation; it always anti-aliases.
    'mitchellcubic'  # Mitchell-Netravali Cubic non-interpolating filter.
    # For synthetic images (especially those lacking proper prefiltering), less
    # ringing than Keys cubic kernel but less sharp.
]
model = 'fusionnet'
iteration = '20210218_all_pos'
weights_id = (model, iteration)

# In [ ]

im = lib.image.load_image(impath, 'uint8', 'gs1c')
imresized = []
imchunks = []

for i, method in enumerate(methods):
    im_ = tf.image.resize(
        im, newsize, method=method, preserve_aspect_ratio=True
    )
    print(method, im_.shape)
    imresized.append(im_)

    # split into two 512 pixel chunks
    X = [
        lib.image.convmode(im_[:512, :512], 'gs'),
        lib.image.convmode(im_[:512, -512:], 'gs')
    ]
    imchunks += X

X = np.array(imchunks)
print(X.shape)
try:
    Y = lib.tf.predict(X, model, weights_id)
except Exception:
    logger.error("Unrecoverable error.", exc_info=True)
    exit(1)
print(Y.shape)

# In [ ]

pdf = backend_pdf.PdfPages(os.path.join(figpath, 'all.pdf'))

for i, method in enumerate(methods):
    j = i * 2
    X_ = X[j:j + 2]
    Y_ = Y[j:j + 2]

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(16, 10),
        sharex='all', sharey='all')
    fig.suptitle('method: {0}'.format(method), fontsize=16)
    ax[0, 0].set_ylabel('Left chunk')
    ax[1, 0].set_ylabel('Right chunk')
    ax[1, 0].set_xlabel('Input image')
    ax[1, 1].set_xlabel('Prediction mask')
    ax[1, 2].set_xlabel('Mask overlaid on input (shown in red)')

    ax[0, 0].imshow(X_[0], cmap=plt.cm.gray)
    ax[0, 1].imshow(Y_[0], cmap=plt.cm.gray)
    ax[1, 0].imshow(X_[1], cmap=plt.cm.gray)
    ax[1, 1].imshow(Y_[1], cmap=plt.cm.gray)

    # make a new image turning the white blobs into red
    # and new image transparency
    opacity = 0.75
    Y_redblob = np.squeeze(Y_, axis=-1)
    Y_zeros = np.zeros(Y_redblob.shape)
    Y_ones = np.ones(Y_redblob.shape)
    Y_redblob = np.stack(
        (Y_redblob, Y_zeros, Y_zeros, Y_ones * opacity),
        axis=-1
    )

    # make black completely transparent
    black = np.all(Y_redblob == [0, 0, 0, opacity], axis=-1)
    Y_redblob[black] = np.array([0, 0, 0, 0])

    # convert original to rgba and combine with red blobs
    X_rgba = np.stack(
        (X_, X_, X_, np.ones(X_.shape) * 255),
        axis=-1
    )
    X_combo = X_rgba.astype(int) + Y_redblob.astype(int)

    ax[0, 2].imshow(X_combo[0], cmap=plt.cm.gray)
    ax[1, 2].imshow(X_combo[1], cmap=plt.cm.gray)

    p = os.path.join(figpath, '{0}.pdf'.format(method))
    plt.savefig(p, dpi=96)
    pdf.savefig(fig)

pdf.close()
plt.show()
