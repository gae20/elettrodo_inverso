# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, Model


def compute_cof(x, y):
    """
    x, y: two feature maps with the same shape, tensor
    """
    x = layers.Flatten()(x)
    y = layers.Flatten()(y)
    fz = tf.reduce_sum(x * y, axis=-1)
    fm = tf.norm(x, axis=-1) * tf.norm(y, axis=-1)
    return fz / (fm + 1e-8)


def Conv_1D_Block(x, kernel_num, kernel_size, strides):
    x = layers.Conv1D(
        filters=kernel_num,
        kernel_size=kernel_size,
        strides=strides,
        padding="same"
    )(x)
    x = layers.Activation("swish")(x)
    return x


def cof_layer(fms):
    """
    fms: list of feature maps
    """
    cof_list = []
    for i in range(len(fms) - 1):
        for j in range(i + 1, len(fms)):
            cof = layers.Lambda(lambda z: compute_cof(z[0], z[1]))([fms[i], fms[j]])
            cof = layers.Reshape((1,))(cof)
            cof_list.append(cof)
    return cof_list


def stem(inputs, num_filters=16, filter_len=7):
    """
    inputs: (batch, signal_length, channels)
    """
    conv = Conv_1D_Block(inputs, num_filters, filter_len, strides=2)

    if conv.shape[1] is not None and conv.shape[1] <= 2:
        pool = layers.MaxPooling1D(pool_size=1, strides=2, padding="same")(conv)
    else:
        pool = layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)

    return pool


def conv_block(x, num_filters, bottleneck=True):
    if bottleneck:
        x = Conv_1D_Block(x, num_filters * 4, kernel_size=1, strides=1)

    out = Conv_1D_Block(x, num_filters, kernel_size=7, strides=1)
    return out


def dense_block(x, num_filters, num_layers, bottleneck=True):
    cb_list = []
    for _ in range(num_layers):
        cb = conv_block(x, num_filters, bottleneck=bottleneck)
        cb_list.append(cb)
        x = layers.Concatenate(axis=-1)([x, cb])
    return x, cb_list


def branch_model(input_shape):
    """
    input_shape es. (SAMPLES_PER_WINDOW, 1)
    """
    inputs = layers.Input(shape=input_shape)

    stem_block = stem(inputs, num_filters=16, filter_len=7)
    dense_op, conv_op_list = dense_block(
        stem_block,
        num_filters=8,
        num_layers=3,
        bottleneck=True
    )

    conv_op_list.append(stem_block)
    conv_op_list.append(dense_op)

    return Model(inputs, conv_op_list, name="single_lead_branch")


def build_model(input_shape=(500, 6), output_dims=16):
    """
    input_shape: (length, channels), per 2s a 250 Hz => (500, n_canali)
    output_dims: numero classi
    """
    inputs = layers.Input(shape=input_shape)

    n_channels = input_shape[-1]

    conv1_ops, conv2_ops, conv3_ops = [], [], []
    dense_ops = []

    branch = branch_model((input_shape[0], 1))

    for i in range(n_channels):
        x_i = layers.Lambda(lambda z, idx=i: z[:, :, idx:idx+1])(inputs)
        conv1_op, conv2_op, conv3_op, stem_op, dense_op = branch(x_i)

        conv1_ops.append(conv1_op)
        conv2_ops.append(conv2_op)
        conv3_ops.append(conv3_op)
        dense_ops.append(dense_op)

    cof_list1 = cof_layer(conv1_ops)
    cof_list2 = cof_layer(conv2_ops)
    cof_list3 = cof_layer(conv3_ops)

    cof_alllist = cof_list1 + cof_list2 + cof_list3
    cofs = layers.Concatenate(axis=-1)(cof_alllist)

    fms = layers.Concatenate(axis=-1)(dense_ops)
    fms = layers.GlobalAveragePooling1D()(fms)

    all_feature = layers.Concatenate(axis=-1)([fms, cofs])
    all_feature = layers.Dense(64, activation="relu")(all_feature)
    outputs = layers.Dense(output_dims, activation="softmax")(all_feature)

    return Model(inputs, outputs, name="ILC_model")


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.config import SAMPLES_PER_WINDOW
    
    input_shape = (SAMPLES_PER_WINDOW, 6)   
    output_dims = 16 # 1 normale + 15 anomalie
    model = build_model(input_shape=input_shape, output_dims=output_dims)
    model.summary()