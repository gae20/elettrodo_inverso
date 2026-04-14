# -*- coding: utf-8 -*-
"""
Training script per ILC model su ECG di 2 secondi @ 125 Hz
usando solo le derivazioni precordiali V1-V6 lette da dataset HDF5.

Assunzioni:
- I file H5 contengono:
    X -> shape (N, 12, 250)
    Y -> shape (N,)
- Ordine canali:
    [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py

from collections import Counter
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

from ILC_model import build_model


font = {'family': 'DejaVu Sans'}
import matplotlib as mpl
mpl.rcParams['font.family'] = 'DejaVu Sans'


# -----------------------------
# Metriche
# -----------------------------
def cal_metrics(conf_mat):
    """
    conf_mat: numpy array, shape (num_classes, num_classes)
    """
    n_classes = conf_mat.shape[0]
    metrics_result = []

    for i in range(n_classes):
        ALL = np.sum(conf_mat)
        TP = conf_mat[i, i]
        FP = np.sum(conf_mat[:, i]) - TP
        FN = np.sum(conf_mat[i, :]) - TP
        TN = ALL - TP - FP - FN

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        specificity = TN / (TN + FP + 1e-8)

        metrics_result.append([precision, recall, specificity, f1])

    return np.array(metrics_result)


# -----------------------------
# Valutazione multiclass
# -----------------------------
def evaluater(x_test, y_test, model, path):
    """
    x_test: shape (N, 250, 6)
    y_test: shape (N,)
    """
    y_pred = model.predict(x_test, verbose=0)
    num = y_pred.shape[-1]
    y_pred = np.argmax(y_pred, axis=1)

    acc = np.mean(y_pred == y_test)
    C = confusion_matrix(y_test, y_pred, labels=range(num))

    plt.figure(figsize=(3.3, 3.3), dpi=600)
    plt.rc('font', **font)
    plt.matshow(C, cmap=plt.cm.Reds)

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(
                C[j, i],
                xy=(i, j),
                horizontalalignment='center',
                verticalalignment='center'
            )

    name = os.path.basename(path).split('.')[0]
    if name.upper() == 'PTBXL':
        name = '(a) ' + name
    else:
        name = '(b) ' + name

    ticks = list(range(num))
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label' + '\n\n' + name)
    plt.savefig(path, bbox_inches='tight', pad_inches=0.0, dpi=600)
    plt.close()

    metrics_result = cal_metrics(C)
    return metrics_result, acc


# -----------------------------
# Valutazione binaria opzionale
# classe 0 vs tutte le altre
# -----------------------------
def merge_label(labels, to_cat=True):
    """
    labels: numpy array, shape (N,)
    """
    new_labels = np.where(labels != 0, 1, 0)
    if to_cat:
        new_labels = to_categorical(new_labels, num_classes=2)
    return new_labels


def evaluater_binary(x_test, y_test, model):
    """
    x_test: shape (N, 250, 6)
    y_test: shape (N,)
    """
    y_pred = model.predict(x_test, verbose=0)
    num = 2

    y_pred = np.argmax(y_pred, axis=1)
    y_pred = merge_label(y_pred, to_cat=False)
    y_test = merge_label(y_test, to_cat=False)
    acc = np.mean(y_pred == y_test)

    C = confusion_matrix(y_test, y_pred, labels=range(num))

    plt.figure(figsize=(3.3, 3.3), dpi=600)
    plt.rc('font', **font)
    plt.matshow(C, cmap=plt.cm.Reds)

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(
                C[j, i],
                xy=(i, j),
                horizontalalignment='center',
                verticalalignment='center'
            )

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    metrics_result = cal_metrics(C)
    return metrics_result, acc


# -----------------------------
# Plot history
# -----------------------------
def print_history(result_array):
    """
    result_array: np.array([
        train_loss, val_loss, train_f1, val_f1, train_acc, val_acc
    ])
    """
    fig = plt.figure()

    ax = plt.subplot(3, 1, 1)
    ax.plot(result_array[0])
    ax.plot(result_array[1])
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    plt.legend(['Train_loss', 'Val_loss'], loc='upper right')

    ax = plt.subplot(3, 1, 2)
    ax.plot(result_array[2])
    ax.plot(result_array[3])
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Macro F1', fontsize=12)
    plt.legend(['Train_F1', 'Val_F1'], loc='upper right')

    ax = plt.subplot(3, 1, 3)
    ax.plot(result_array[4])
    ax.plot(result_array[5])
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    plt.legend(['Train_acc', 'Val_acc'], loc='upper right')

    fig.subplots_adjust(hspace=0.5)


# -----------------------------
# Lettura H5
# -----------------------------
def load_h5_data(file_path):
    """
    Legge:
    X -> segnali ECG
    Y -> etichette
    """
    with h5py.File(file_path, 'r') as f:
        x = np.array(f['X'])
        y = np.array(f['Y'])
    return x, y


# -----------------------------
# Preprocessing:
# seleziona precordiali V1-V6
# -----------------------------
def preprocess_set(x):
    """
    Input atteso:
        x shape = (N, 12, 250)

    Output:
        x shape = (N, 250, 6)

    Selezione:
        V1..V6 = canali 6..11
    """
    if x.ndim != 3:
        raise ValueError(f"Atteso array 3D, trovato shape {x.shape}")

    if x.shape[1] != 12:
        raise ValueError(
            f"Atteso input con 12 canali in asse 1, trovato shape {x.shape}. "
            f"Questo script assume formato (N, 12, 250)."
        )

    if x.shape[2] != 250:
        raise ValueError(
            f"Attesi 250 sample temporali, trovato shape {x.shape}. "
            f"Questo script è per ECG di 2 s a 125 Hz."
        )

    # Prende solo V1-V6
    x = x[:, 6:12, :]           # (N, 6, 250)

    # Trasponi a (N, 250, 6)
    x = np.transpose(x, (0, 2, 1))

    return x.astype(np.float32)


# -----------------------------
# Training
# -----------------------------
def train_model(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    save_path,
    EP,
    LR,
    BS,
    input_shape,
    output_dims,
    pic_path_ptb,
    pic_path_ptbxl,
    test=True
):
    opt = Adam(learning_rate=LR)

    me = tf.keras.metrics.F1Score(
        average='macro',
        name='f1_score'
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy', me]
    )

    checkpoint = ModelCheckpoint(
        save_path,
        monitor='val_f1_score',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='max'
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=EP,
        batch_size=BS,
        validation_data=(x_val, y_val),
        shuffle=True,
        callbacks=[checkpoint]
    )

    history_dict = history.history
    result = np.array([
        history_dict['loss'],
        history_dict['val_loss'],
        history_dict['f1_score'],
        history_dict['val_f1_score'],
        history_dict['accuracy'],
        history_dict['val_accuracy']
    ], dtype=object)
    np.save(save_path + '_history.npy', result)

    if test:
        print("\n--- Valutazione finale ---")
        best_model = build_model(input_shape, output_dims)
        best_model.load_weights(save_path)

        metrics_result_ptb, ptb_acc = evaluater(
            x_test[0], y_test[0], best_model, pic_path_ptb
        )
        metrics_result_ptbxl, ptbxl_acc = evaluater(
            x_test[1], y_test[1], best_model, pic_path_ptbxl
        )

        return history, metrics_result_ptb, ptb_acc, metrics_result_ptbxl, ptbxl_acc

    return history


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    # Configurazione
    input_shape = (250, 6)        # 250 sample, 6 precordiali
    output_dims = 10               # cambia se hai un numero diverso di classi
    EP = 60
    LR = 1e-3 / 2
    BS = 1024

    save_path = 'best_model_precordial_6.weights.h5'

    # Path dataset
    train_h5_path = 'train_dataset_clean.h5'
    val_h5_path = 'val_dataset_clean.h5'
    test_h5_path = 'test_dataset_clean.h5'

    print("Caricamento dataset in corso...")

    try:
        x_train_raw, y_train_raw = load_h5_data(train_h5_path)
        x_val_raw, y_val_raw = load_h5_data(val_h5_path)
        x_test_raw, y_test_raw = load_h5_data(test_h5_path)
    except KeyError as e:
        raise KeyError(
            f"Errore nelle chiavi del file H5: {e}. "
            f"Questo script si aspetta dataset 'X' e 'Y'."
        )

    # Preprocessing
    x_train = preprocess_set(x_train_raw)
    x_val = preprocess_set(x_val_raw)
    x_test = preprocess_set(x_test_raw)

    # Label
    # Qui assumo che le etichette siano già intere e consecutive: 0..output_dims-1
    y_train = to_categorical(y_train_raw, num_classes=output_dims)
    y_val = to_categorical(y_val_raw, num_classes=output_dims)

    print("Dataset caricato con successo.")
    print(f"Shape X_train: {x_train.shape}")
    print(f"Shape X_val:   {x_val.shape}")
    print(f"Shape X_test:  {x_test.shape}")
    print(f"Distribuzione train: {Counter(y_train_raw)}")
    print(f"Distribuzione val:   {Counter(y_val_raw)}")
    print(f"Distribuzione test:  {Counter(y_test_raw)}")

    