"""ResNet-50 model constructor.

Ronneberger, O., Fischer, P. & Brox, T.
U-Net: Convolutional Networks for Biomedical Image Segmentation.
arXiv:1505.04597 [cs] (2015).
"""

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
import numpy as np


es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    min_delta=1e-2,
    patience=2,
    verbose=1
)


def identity_block(filters, act_fn, input):
    """Set up a ResNet identity block.

    Input size doesn't change.

    Args:
        feat_maps_out (int): Number of filters.
        act_fn (str): Activation function.
        input (tf.keras.layers.Layer): Previous layer.

    Returns:
        tf.keras.layers.Layer

    """
    regulariser = K.regularizers.l2(0.001)

    conv = L.Conv2D(filters[0],
        kernel_size=(1, 1), strides=(1, 1), padding='valid',
        kernel_regularizer=regulariser)(input)
    conv = act_fn(conv)
    conv = L.BatchNormalization()(conv)

    conv = L.Conv2D(filters[0],
        kernel_size=(3, 3), strides=(1, 1), padding='same',
        kernel_regularizer=regulariser)(conv)
    conv = act_fn(conv)
    conv = L.BatchNormalization()(conv)

    conv = L.Conv2D(filters[1],
        kernel_size=(1, 1), strides=(1, 1), padding='valid',
        kernel_regularizer=regulariser)(conv)
    conv = L.BatchNormalization()(conv)

    ret = L.Add()([conv, input])
    ret = act_fn(ret)
    return ret


def conv_block(filters, stride, act_fn, input):
    """Set up a ResNet convolutional block.

    Input size changes.

    Args:
        filters (tuple): Number of filters.
        act_fn (str): Activation function.
        input (tf.keras.layers.Layer): Previous layer.

    Returns:
        tf.keras.layers.Layer

    """
    regulariser = K.regularizers.l2(0.001)

    conv = L.Conv2D(filters[0],
        kernel_size=(1, 1), strides=(stride, stride), padding='valid',
        kernel_regularizer=regulariser)(input)
    conv = act_fn(conv)
    conv = L.BatchNormalization()(conv)

    conv = L.Conv2D(filters[0],
        kernel_size=(3, 3), strides=(1, 1), padding='same',
        kernel_regularizer=regulariser)(conv)
    conv = act_fn(conv)
    conv = L.BatchNormalization()(conv)

    conv = L.Conv2D(filters[1],
        kernel_size=(1, 1), strides=(1, 1), padding='valid',
        kernel_regularizer=regulariser)(conv)
    conv = L.BatchNormalization()(conv)

    conv_skip = L.Conv2D(filters[1],
        kernel_size=(1, 1), strides=(1, 1), padding='valid',
        kernel_regularizer=regulariser)(input)
    conv_skip = L.BatchNormalization()(conv_skip)

    ret = L.Add()([conv, conv_skip])
    ret = act_fn(ret)
    return ret


def build(lr=0.001, input_shape=(640, 640, 1)):

    stride = 1

    def act_fn(input):
        return L.ReLU()(input)

    inputs = L.Input(input_shape)
    pad = L.ZeroPadding2D(padding=(3, 3))(inputs)

    start = L.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2))(pad)
    start = L.BatchNormalization()(start)
    start = act_fn(start)
    start = L.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(start)

    layer2 = conv_block((64, 256), stride, act_fn, start)
    layer2 = identity_block((64, 256), act_fn, layer2)
    layer2 = identity_block((64, 256), act_fn, layer2)

    layer3 = conv_block((128, 512), stride, act_fn, layer2)
    layer3 = identity_block((128, 512), act_fn, layer3)
    layer3 = identity_block((128, 512), act_fn, layer3)
    layer3 = identity_block((128, 512), act_fn, layer3)

    layer4 = conv_block((256, 1024), stride, act_fn, layer3)
    layer4 = identity_block((256, 1024), act_fn, layer4)
    layer4 = identity_block((256, 1024), act_fn, layer4)
    layer4 = identity_block((256, 1024), act_fn, layer4)
    layer4 = identity_block((256, 1024), act_fn, layer4)
    layer4 = identity_block((256, 1024), act_fn, layer4)

    layer5 = conv_block((512, 2048), stride, act_fn, layer4)
    layer5 = identity_block((512, 2048), act_fn, layer5)
    layer5 = identity_block((512, 2048), act_fn, layer5)

    # output
    out = L.AveragePooling2D(pool_size=(2, 2), padding='same')(layer5)

    model = K.Model(inputs=inputs, outputs=out)
    model.compile(
        optimizer=K.optimizers.Adam(lr=lr),
        loss=K.losses.Huber(),
        metrics=[K.metrics.MeanSquaredError()]
    )

    return model


def pack_data(X):
    """Convert array of images to machine trainable data.

    Args:
        X (numpy.ndarray): Image data represented as a single image
            or array of images.

    Returns:
        numpy.ndarray: Transformed image data.

    """
    # scale image data to (0, 1)
    X = (X.astype('float32') / 255.0)
    # pad image
    X = np.pad(X, ((0, 0), (64, 64), (64, 64)), 'reflect')
    # add channel dimension
    X = np.expand_dims(X, axis=-1)
    return X


def unpack_data(X):
    """Convert neural network output data back to images.

    Args:
        X (numpy.ndarray): Transformed image data.

    Returns:
        numpy.ndarray: Image data represented as a single image
            or array of images.

    """
    # unpad image
    X_ = []
    for i in X:
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
