import tensorflow as tf

def Conv_1D_Block(x, model_width, kernel, strides):
    # NIENTE DROPOUT QUI
    x = tf.keras.layers.Conv1D(model_width, kernel, strides=strides, padding="same")(x)
    x = tf.keras.layers.Activation('swish')(x)
    return x

def stem(inputs, num_filters, filter_len):
    # STRIDE RIPRISTINATO A 2 come da Table 2 di Huang
    conv = Conv_1D_Block(inputs, num_filters, filter_len, strides=2) 
    pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)
    return pool

def conv_block(x, num_filters, kernel_lens, bottleneck=True):
    if bottleneck:
        num_filters_bottleneck = num_filters * 4
        x = Conv_1D_Block(x, num_filters_bottleneck, 1, strides=1)
    out = Conv_1D_Block(x, num_filters, kernel_lens, strides=1)
    return out

def dense_block(x, num_filters, num_layers, bottleneck=True):
    for i in range(num_layers):
        cb = conv_block(x, num_filters, 7, bottleneck=bottleneck)
        x = tf.keras.layers.concatenate([x, cb], axis=-1)
    return x

def build_model(input_shape, output_dims):
    inputs = tf.keras.Input(input_shape)
    stem_block = stem(inputs, num_filters=16, filter_len=11)
    Dense_Block_1 = dense_block(stem_block, num_filters=8, num_layers=3, bottleneck=True)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(Dense_Block_1)
    # DROPOUT SOLO NEL CLASSIFIER (come da paper: Rate 0.5)
    x = tf.keras.layers.Dropout(0.5)(x) 
    x = tf.keras.layers.Dense(output_dims, activation='softmax')(x)
    
    return tf.keras.Model(inputs, x)

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.config import SAMPLES_PER_WINDOW
    
    input_shape = (SAMPLES_PER_WINDOW, 6)
    output_dims = 6 # 1 normale + 5 anomalie
    model = build_model(input_shape, output_dims)
    model.summary()