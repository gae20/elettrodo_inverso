import tensorflow as tf

def Conv_1D_Block(x, model_width, kernel, strides, dropout_rate=0.2):
    '''
    AGGIUNTO: dropout_rate per Dropout dopo ogni Conv1D
    '''
    x = tf.keras.layers.Conv1D(model_width, kernel, strides=strides, padding="same")(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)  # ✅ DROPOUT DOPO ATTIVAZIONE
    return x

def stem(inputs, num_filters, filter_len):
    conv = Conv_1D_Block(inputs, num_filters, filter_len, 1)  #modifica 1: stride 2 -> 1
    if conv.shape[1] <= 2:
        pool = tf.keras.layers.MaxPooling1D(pool_size=1, strides=2, padding="same")(conv)
    else:
        pool = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)
    return pool

def conv_block(x, num_filters, kernel_lens, bottleneck=True, dropout_rate=0.2):
    if bottleneck:
        num_filters_bottleneck = num_filters * 4
        x = Conv_1D_Block(x, num_filters_bottleneck, 1, 1, dropout_rate)  # ✅ Dropout anche bottleneck

    out = Conv_1D_Block(x, num_filters, kernel_lens, 1, dropout_rate)  # ✅ Dropout principale
    return out

def dense_block(x, num_filters, num_layers, bottleneck=True, dropout_rate=0.2):
    for i in range(num_layers):
        cb = conv_block(x, num_filters, 7, bottleneck=bottleneck, dropout_rate=dropout_rate)
        x = tf.keras.layers.concatenate([x, cb], axis=-1)
    return x

# build LDenseNet CON DROPOUT OVUNQUE
def build_model(input_shape, output_dims, dropout_rate=0.3):  # ✅ Parametro globale dropout
    '''
    AGGIUNTO: dropout_rate=0.3 per tutto il modello (riduci se troppo aggressivo)
    '''
    inputs = tf.keras.Input(input_shape)
    
    # STEM con dropout
    stem_block = stem(inputs, num_filters=16, filter_len=11)
    
    # DENSE BLOCK con dropout in ogni conv
    Dense_Block_1 = dense_block(stem_block, num_filters=8, num_layers=3, bottleneck=True, dropout_rate=dropout_rate)
    
    # Global Pool + Dropout + Output
    x = tf.keras.layers.GlobalAveragePooling1D()(Dense_Block_1)
    x = tf.keras.layers.Dropout(dropout_rate)(x)  # ✅ Dropout finale (era già qui, ora configurabile)
    x = tf.keras.layers.Dense(output_dims, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, x)
    return model

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils.config import SAMPLES_PER_WINDOW
    
    input_shape = (SAMPLES_PER_WINDOW, 6)
    output_dims = 6 # 1 normale + 5 anomalie
    model = build_model(input_shape, output_dims, dropout_rate=0.3)
    model.summary()