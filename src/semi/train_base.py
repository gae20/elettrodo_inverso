"""
train_base.py

Addestra il modello base (LDenseNet) sui soli dati reali iniziali:
1. Carica train_small_init.h5 e val_small_init.h5.
2. Inizializza e compila il modello.
3. Esegue il fit con EarlyStopping su val_f1_score.
4. Salva i pesi finali in results/model_base.weights.h5.
"""

import os
import sys
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- Setup path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.join(SCRIPT_DIR, '..')
THESIS_DIR = os.path.join(SRC_DIR, '..')

sys.path.append(SRC_DIR)
from models.ldensenet import build_model

# --- Percorsi ---
H5_DIR         = os.path.join(SCRIPT_DIR, 'results', 'semi_h5')
RESULTS_DIR    = os.path.join(SCRIPT_DIR, 'results')
TRAIN_INIT_H5  = os.path.join(H5_DIR, 'train_small_init.h5')
VAL_INIT_H5    = os.path.join(H5_DIR, 'val_small_init.h5')

# --- Parametri ---
EP = 40
LR = 1e-3
BS = 256

def load_h5_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
    X_transposed = np.transpose(X, (0, 2, 1)) # Da (N, 6, 500) a (N, 500, 6)
    return X_transposed, Y

if __name__ == '__main__':
    # Imposta seed globale per determinismo e riproducibilità totale
    tf.keras.utils.set_random_seed(42)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponibili: {len(gpus)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("ADDESTRAMENTO MODELLO BASE (SOLO DATI REALI)")
    print("=" * 60)

    # 1. Carica i dati iniziali
    print("\n[Caricamento dati di partenza...]")
    if not (os.path.exists(TRAIN_INIT_H5) and os.path.exists(VAL_INIT_H5)):
        print("ERRORE: I file H5 iniziali non sono pronti. Esegui prima prepare_datasets.py.")
        sys.exit(1)

    X_train, Y_train = load_h5_data(TRAIN_INIT_H5)
    X_val, Y_val     = load_h5_data(VAL_INIT_H5)

    print(f"  Train : {X_train.shape[0]} finestre")
    print(f"  Val   : {X_val.shape[0]} finestre")

    # 2. Addestra il modello
    model = build_model((500, 6), 6)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=LR),
        metrics=['accuracy', tf.keras.metrics.F1Score(average='macro', name='f1_score')]
    )

    base_weights_path = os.path.join(RESULTS_DIR, 'model_base.weights.h5')
    callbacks = [
        EarlyStopping(monitor='val_f1_score', patience=8, restore_best_weights=True, mode='max', verbose=1)
    ]

    Y_train_oh = to_categorical(Y_train, 6)
    Y_val_oh   = to_categorical(Y_val, 6)

    print("\n[Inizio Addestramento...]")
    model.fit(
        X_train, Y_train_oh,
        batch_size=BS,
        epochs=EP,
        validation_data=(X_val, Y_val_oh),
        callbacks=callbacks,
        verbose=1
    )

    model.save_weights(base_weights_path)
    print(f"\nPesi salvati correttamente in: {base_weights_path}")
    print("=" * 60)
