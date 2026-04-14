# -*- coding: utf-8 -*-
"""
Created on Sat May 13 14:39:56 2023

@author: 10671
"""

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from LDenseNet import build_model
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import f1_score, confusion_matrix 
import matplotlib.pyplot as plt
import h5py
from collections import Counter

font = {'family': 'DejaVu Sans'}
import matplotlib as mpl
mpl.rcParams['font.family'] = 'DejaVu Sans'

#calculate metrics
def cal_metrics(confusion_matrix):
    '''
    confusion_matrix: numpy array, (class_number, class_number)
    '''
    n_classes = confusion_matrix.shape[0]
    metrics_result = []
    for i in range(n_classes):
        ALL = np.sum(confusion_matrix)
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP
        TN = ALL - TP - FP - FN
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1 = 2*precision*recall/(precision+recall)
        specificity = TN/(TN+FP)
        metrics_result.append([precision, recall, specificity, f1])
    return metrics_result


# test the model
def evaluater(x_test, y_test, model, path):
    '''
    x_test: numpy array, (sample number, 1200, 3)
    y_test: numpy array, (sample number,)
    model: trained LDenseNet model
    path: path for result saving
    '''
    y_pred = model.predict(x_test)
    num = y_pred.shape[-1]
    y_pred = np.argmax(y_pred, axis=1)
    acc = len(np.where(y_pred==y_test)[0])/y_pred.shape[0]
    C = confusion_matrix(y_test, y_pred, labels=range(num))
    #np.save("C:/Users/nyapass/Desktop/papers/comfuse/ptbxl_limb.npy", C)
    plt.figure(figsize=(3.3,3.3), dpi=600)
    plt.rc('font', **font)
    plt.matshow(C, cmap=plt.cm.Reds) 

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    name = path.split('/')[-1].split('.')[0]
    if name == 'PTBXL':
        name = '(a) ' + name
    else:
        name = '(b) ' + name
    plt.ylabel('True label')
    plt.xlabel('Predicted label' + '\n' + '\n' + name)
    #plt.savefig(path, dpi=600, format = 'tiff')
    plt.savefig(path, bbox_inches='tight', pad_inches=0.0, dpi=600, format = 'tiff')

    metrics_result = cal_metrics(C)
    metrics_result = np.array(metrics_result)
    return metrics_result, acc

# used when evaluate the binary classification performance
def merge_label(labels, to_cat=True):
    '''
    labels: numpy array, (test sample number, )
    to_cat: whether transfer labels to one-hot labels or not
    '''
    new_labels = []
    for i in range(labels.shape[0]):
        if labels[i]!=0:
            new_labels.append(1)
        else:
            new_labels.append(0)
    new_labels = np.array(new_labels)
    if to_cat:
        new_labels = to_categorical(new_labels, num_classes = 2)
    return new_labels

# evaluate the binary classification performance
def evaluater_binary(x_test, y_test, model):
    '''
    x_test: numpy array, (sample number, 1200, 3)
    y_test: numpy array, (sample number,)
    model: trained LDenseNet model
    '''
    y_pred = model.predict(x_test)
    num = 2
    
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = merge_label(y_pred, to_cat=False)
    y_test = merge_label(y_test, to_cat=False)
    acc = len(np.where(y_pred==y_test)[0])/y_pred.shape[0]
    
    C = confusion_matrix(y_test, y_pred, labels=range(num))
    plt.figure(figsize=(3.3,3.3), dpi=600)
    plt.rc('font', **font)
    plt.matshow(C, cmap=plt.cm.Reds)
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    metrics_result = cal_metrics(C)
    metrics_result = np.array(metrics_result)
    return metrics_result, acc

# print the train and validation history
def print_history(result_array):
    '''
    result_array: numpy array
    '''
    fig = plt.figure()
    ax = plt.subplot(2,1,1)
    ax.plot(result_array[0])
    ax.plot(result_array[1])
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    plt.legend(['Train_loss', 'Val_loss'], loc = 'upper right')
    ax = plt.subplot(2,1,2)
    ax.plot(result_array[2])
    ax.plot(result_array[3])
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Weighted APUC', fontsize=12)
    fig.subplots_adjust(hspace=0.4)
    plt.legend(['Train_APUC', 'Val_APUC', 'Train_loss', 'Val_loss'], loc = 'upper right')

#fix 1: upgrade di questa parte alla sintassi moderna di tensor flow
# Custom metrics for validation set monitoring 
#def py_auprc(y_true, y_pred):
    #y_pred = np.argmax(y_pred, axis=-1)
    #y_true = np.argmax(y_true, axis=-1)
    #score = f1_score(y_true, y_pred, average='macro')
    #score = score.astype(np.float32)
    #return score
#def tf_auprc(y_true, y_pred):
    #return tf.numpy_function(py_auprc, (y_true, y_pred), tf.float32)

#class F1S(tf.keras.metrics.Metric):
    #def __init__(self, name="F1", **kwargs):
        #super(F1S, self).__init__(name=name, **kwargs)
        #self.score = self.add_weight(name="f1s", initializer="zeros")
    #def update_state(self, y_true, y_pred, sample_weight=None):
        #self.score.assign_add(tf_auprc(y_true, y_pred))
    #def result(self):
        #return self.score
    #def reset_states(self):
        #self.score.assign(0.0)

#fix 2 ora train model è scritta nel moderno tensor flow, ma e la stessa di Huang
def train_model(model, x_train, y_train, x_val, y_val, x_test, y_test, save_path, EP, LR, BS, input_shape, output_dims, pic_path_ptb, pic_path_ptbxl, test = True):
    # Usiamo il learning_rate corretto per le nuove versioni
    opt = Adam(learning_rate=LR)
    
    # Definiamo la metrica F1 ufficiale di Keras
    # 'macro' è quello che usava Huang (media semplice tra le classi)
    ME = tf.keras.metrics.F1Score(average='macro', name='f1_score')
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', ME])
    
    # IMPORTANTE: Il monitor ora deve puntare a 'val_f1_score'
    checkpoint = ModelCheckpoint(
        save_path, 
        monitor='val_f1_score', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=True, 
        mode='max'
    )
    
    callbacks_list = [checkpoint]
    
    # Avvio del training
    history = model.fit(
        x_train, y_train, 
        epochs=EP, 
        batch_size=BS, 
        validation_data=(x_val, y_val), 
        shuffle=True, 
        callbacks=callbacks_list
    )
    
    # Salvataggio storia
    history_dict = history.history
    # Nota: salviamo i nomi corretti delle metriche nel file history.npy
    result = np.array([history_dict['loss'], history_dict['val_loss'], history_dict['f1_score'], history_dict['val_f1_score']])
    np.save(save_path + '_history.npy', result)
    
    if test:
        print("\n--- Valutazione Finale ---")
        best_model = build_model(input_shape, output_dims)
        best_model.load_weights(save_path)
        metrics_result_ptb, ptb_acc = evaluater(x_test[0], y_test[0], best_model, pic_path_ptb)
        metrics_result_ptbxl, ptbxl_acc = evaluater(x_test[1], y_test[1], best_model, pic_path_ptbxl)
        return history, metrics_result_ptb, ptb_acc, metrics_result_ptbxl, ptbxl_acc

if __name__ == '__main__':
    # --- 1. CONFIGURAZIONE PARAMETRI ---
    input_shape = (250, 6)  
    output_dims = 8         
    EP = 20 #modifica 4: EP 100 -> 10, 10 -> 20
    LR = 1e-3 #modifica 2: LR 1e-3 -> 1e-4
    BS = 32 #modifica 3: BS 512 -> 32
    save_path = 'best_model_limb_6.weights.h5'

    # --- 2. MAPPING ETICHETTE (Fondamentale!) ---
    # Trasformiamo le stringhe del file H5 in numeri per il modello
    LABEL_MAP = {
        'normale': 0,
        'LA-RA': 1,
        'RA-LL': 2,
        'LA-LL': 3,
        'ROT_ORARIA': 4,
        'ROT_ANTIORARIA': 5,
        'RL-RA': 6,
        'RL-LA': 7
    }

    def load_h5_data(file_path):
        with h5py.File(file_path, 'r') as f:
            x = np.array(f['X'])
            y = np.array(f['Y'])   # già 0..7
        return x, y

    print("Caricamento dataset in corso...")
    try:
        x_train_raw, y_train_raw = load_h5_data("train_dataset.h5")
        x_val_raw, y_val_raw = load_h5_data("val_dataset.h5")
        x_test_raw, y_test_raw = load_h5_data("test_dataset.h5")
    except KeyError as e:
        print(f"Errore: Controlla le chiavi del file H5. L'errore è {e}")
        # Suggerimento: controlla se nel file H5 hai usato 'X' o 'x'

    # --- 3. SELEZIONE DERIVAZIONI E TRASPOSIZIONE ---
    def preprocess_set(x):
        # Prendiamo solo le prime 6 derivazioni periferiche
        x = x[:, :6, :] 
        # Trasponiamo da (N, 6, 250) -> (N, 250, 6)
        x = np.transpose(x, (0, 2, 1))
        return x

    x_train = preprocess_set(x_train_raw)
    x_val = preprocess_set(x_val_raw)
    x_test = preprocess_set(x_test_raw)

    # --- 4. PREPARAZIONE ETICHETTE ---
    y_train = to_categorical(y_train_raw, num_classes=output_dims)
    y_val = to_categorical(y_val_raw, num_classes=output_dims)
    # y_test_raw lo teniamo "flat" (non categorico) perché evaluater usa argmax

    print(f"Dataset caricato con successo!")
    print(f"Shape X_train: {x_train.shape} | Esempi per classe: {Counter(y_train_raw)}")

    # --- 5. COSTRUZIONE E TRAINING ---
    model = build_model(input_shape, output_dims)

    pic_path_ptb = "confusion_matrix_test.tiff"
    pic_path_ptbxl = "confusion_matrix_val.tiff"
    
    # Avvio del training usando la funzione di Huang
    history, result_ptb, acc_ptb, result_ptbxl, acc_ptbxl = train_model(
        model, 
        x_train, y_train, 
        x_val, y_val, 
        [x_test, x_val], [y_test_raw, y_val_raw], # Testiamo su test set e val set
        save_path, EP, LR, BS, 
        input_shape, output_dims, 
        pic_path_ptb, pic_path_ptbxl,
        test=True
    )

    print(f"Training completato. Accuratezza finale Test: {acc_ptb:.4f}")