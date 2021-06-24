"""U-Net model constructor.

Ronneberger, O., Fischer, P. & Brox, T.
U-Net: Convolutional Networks for Biomedical Image Segmentation.
arXiv:1505.04597 [cs] (2015).
"""

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
import numpy as np

import lib.image as image


es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=1e-2,
    patience=2,
    verbose=1
)


def conv_block(feat_maps_out, act_fn, input):
    """Set up a convolution block followed by batch normalisation.

    Shape of input tensor will be preserved due to stride=1.

    Args:
        feat_maps_out (int): Number of filters.
        act_fn (str): Activation function.
        input (tf.keras.layers.Layer): Previous layer.

    Returns:
        tf.keras.layers.Layer

    """
    conv = L.Conv2D(feat_maps_out,
        kernel_size=3, strides=1, padding='same')(input)
    conv = act_fn(conv)
    conv = L.BatchNormalization()(conv)
    return conv


def conv_block_2(feat_maps_out, act_fn, input):
    """Set up a group of two consecutive convolution blocks.

    Shape of input tensor will be preserved.

    Args:
        feat_maps_out (int): Number of filters.
        act_fn (str): Activation function.
        input (tf.keras.layers.Layer): Previous layer.

    Returns:
        tf.keras.layers.Layer

    """
    conv = conv_block(feat_maps_out, act_fn, input)
    conv = conv_block(feat_maps_out, act_fn, conv)
    return conv


def maxpool(input):
    """Set up a max pool layer.

    Shape of input tensor will be halved, number of filters will be doubled.

    Args:
        input (tf.keras.layers.Layer): Previous layer.

    Returns:
        tf.keras.layers.Layer

    """
    pool = L.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(input)
    return pool


def conv_upsampling(input):
    """Set up an upsampling layer.

    Shape of input tensor will be doubled, number of filters will be halved.

    Args:
        feat_maps_out (int): Number of filters.
        act_fn (str): Activation function.
        input (tf.keras.layers.Layer): Previous layer.

    Returns:
        tf.keras.layers.Layer

    """
    pool = L.UpSampling2D(size=(2, 2))(input)
    return pool


def build(lr=0.001, input_shape=(640, 640, 1)):
    inputs = L.Input(input_shape)
    n_filters = 16

    def act_fn_encoder(input):
        return L.ReLU()(input)

    def act_fn_decoder(input):
        return L.ReLU()(input)

    # encoder
    down_1 = conv_block_2(n_filters * 1, act_fn_encoder, inputs)
    pool_1 = maxpool(down_1)
    down_2 = conv_block_2(n_filters * 2, act_fn_encoder, pool_1)
    pool_2 = maxpool(down_2)
    down_3 = conv_block_2(n_filters * 3, act_fn_encoder, pool_2)
    pool_3 = maxpool(down_3)
    down_4 = conv_block_2(n_filters * 4, act_fn_encoder, pool_3)
    pool_4 = maxpool(down_4)

    # bridge
    bridge = conv_block_2(n_filters * 4, act_fn_encoder, pool_4)

    # decoder
    deconv_1 = conv_upsampling(bridge)
    skip_1 = L.Concatenate()([deconv_1, down_4])
    up_1 = conv_block_2(n_filters * 4, act_fn_decoder, skip_1)
    deconv_2 = conv_upsampling(up_1)
    skip_2 = L.Concatenate()([deconv_2, down_3])
    up_2 = conv_block_2(n_filters * 3, act_fn_decoder, skip_2)
    deconv_3 = conv_upsampling(up_2)
    skip_3 = L.Concatenate()([deconv_3, down_2])
    up_3 = conv_block_2(n_filters * 2, act_fn_decoder, skip_3)
    deconv_4 = conv_upsampling(up_3)
    skip_4 = L.Concatenate()([deconv_4, down_1])
    up_4 = conv_block_2(n_filters * 1, act_fn_decoder, skip_4)

    # output
    out = L.Conv2D(filters=input_shape[2], kernel_size=(1, 1),
        padding='same')(up_4)
    out_2 = L.Activation('sigmoid')(out)

    model = K.Model(inputs=inputs, outputs=out_2)
    model.compile(
        optimizer=K.optimizers.Adam(lr=lr),
        loss=K.losses.Huber(),
        metrics=[K.metrics.MeanSquaredError()]
    )

    return model


def pack_data(X, train=False):
    """Convert array of images to machine trainable data.

    Args:
        X (numpy.ndarray): Image data represented as a single image
            or array of images.
        train (bool): Set to true if passed data is training data and
            transformations don't apply.

    Returns:
        numpy.ndarray: Transformed image data.

    """
    X_ = []
    for i in X:
        # scale image data to (0, 1)
        i = image.fscale(i, 0., 1., 0, 255)
        if train is False:
            # remove background
            i = image.bg_removal(i)
            # centre the background
            i = i + (0.5 - np.mean(i))
    X = np.array(X_)

    # pad image
    X = np.pad(X, ((0, 0), (64, 64), (64, 64)), 'reflect')
    # add channel dimension
    X = np.expand_dims(X, axis=-1)
    return X


def pack_training_data(X):
    return pack_data(X, True)


def unpack_data(X):
    """Convert neural network output data back to images.

    Args:
        X (numpy.ndarray): Transformed image data.

    Returns:
        numpy.ndarray: Image data represented as a single image
            or array of images.

    """
    X_ = []
    for i in X:
        # unpad image
        X_.append(i[64:-64, 64:-64])
    X = np.array(X_)

    # clip image data to avoid out of bounds values
    X = np.clip(X, 0., 1.)
    # convert float to greyscale int
    X = X * 255.0
    X = X.astype('uint8')
    return X


def metrics(m, log):
    """Output model evaluation metrics to the logger.

    Args:
        m (tuple): Result of tensorflow.keras.Model.evaluate()
        log (logging.Logger): Logger to log the metric data to.

    Returns:
        None

    """
    log.info('Loss: {:.6f}'.format(m[0]))
    log.info('MSE: {:.6f}'.format(m[1]))


if __name__ == '__main__':
    # Test model summary
    model = build()
    model.summary()
    K.utils.plot_model(model, to_file='fusionnet.png', show_shapes=True)
