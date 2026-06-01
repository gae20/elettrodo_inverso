"""
step4_train_ssl.py

Riaddestra LDenseNet da zero sul dataset SSL:
  - Training: unlabelled_z_median_limbs_train_ssl.h5  (sintetico + pseudo-labeled)
  - Validation: unlabelled_z_median_limbs_val.h5       (sintetico originale)
  - Test: labelled_z_median_limbs_test_validation.h5   (dati reali, invariato)

Pesi salvati in: results/ssl_weights/best_model_ssl_limbs.weights.h5
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.join(SCRIPT_DIR, '..')
THESIS_DIR = os.path.join(SRC_DIR, '..')

sys.path.append(SRC_DIR)
from models.ldensenet import build_model
from utils.config import SAMPLES_PER_WINDOW

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# --- Percorsi ---
DATASETS_FINAL = os.path.join(THESIS_DIR, 'datasets', 'unlabelled_simulated_final')
DATASETS_LABEL = os.path.join(THESIS_DIR, 'datasets')

DATASET_TRAIN = os.path.join(DATASETS_FINAL, 'unlabelled_z_median_limbs_train_ssl.h5')
DATASET_VAL   = os.path.join(DATASETS_FINAL, 'unlabelled_z_median_limbs_val.h5')
DATASET_TEST  = os.path.join(DATASETS_LABEL, 'labelled_z_median_limbs_test_validation.h5')

OUT_DIR    = os.path.join(SCRIPT_DIR, 'results', 'ssl_weights')
SAVE_PATH  = os.path.join(OUT_DIR, 'best_model_ssl_limbs.weights.h5')
CM_TEST    = os.path.join(OUT_DIR, 'ssl_cm_test.png')
CM_VAL     = os.path.join(OUT_DIR, 'ssl_cm_val.png')

CLASS_NAMES = ['normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']


# ---------------------------------------------------------------------------
# Utilities (identiche a train_limbs.py)
# ---------------------------------------------------------------------------

def cal_metrics(cm):
    n_classes = cm.shape[0]
    results = []
    for i in range(n_classes):
        ALL = np.sum(cm)
        TP  = cm[i, i]
        FP  = np.sum(cm[:, i]) - TP
        FN  = np.sum(cm[i, :]) - TP
        TN  = ALL - TP - FP - FN
        precision   = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall      = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        results.append([precision, recall, specificity, f1])
    return results


def evaluater(x_test, y_test, model, path):
    print(f"\n> Confusion matrix: {os.path.basename(path)}")
    y_pred     = model.predict(x_test, batch_size=64, verbose=0)
    y_pred_idx = np.argmax(y_pred, axis=1)
    acc        = np.mean(y_pred_idx == y_test)
    C          = confusion_matrix(y_test, y_pred_idx, labels=range(6))

    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    ax.matshow(C, cmap=plt.cm.Reds)
    for i in range(len(C)):
        for j in range(len(C)):
            ax.annotate(str(C[j, i]), xy=(i, j),
                        horizontalalignment='center', verticalalignment='center')
    ax.set_xticks(range(6)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='left')
    ax.set_yticks(range(6)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close()
    return np.array(cal_metrics(C)), acc


def evaluater_pro(x_test, y_test_idx, model):
    y_probs    = model.predict(x_test, batch_size=64, verbose=0)
    y_test_oh  = to_categorical(y_test_idx, num_classes=6)
    auroc      = roc_auc_score(y_test_oh, y_probs, multi_class='ovr', average='macro')
    auprc      = average_precision_score(y_test_oh, y_probs, average='macro')
    return auroc, auprc


# ---------------------------------------------------------------------------
# Data Generator (identico a train_limbs.py)
# ---------------------------------------------------------------------------

class H5DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_path, batch_size=256, num_classes=6, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.file_path   = file_path
        self.batch_size  = batch_size
        self.num_classes = num_classes
        self.shuffle     = shuffle
        with h5py.File(self.file_path, 'r') as f:
            y_all = f['Y'][:]
            self.indices      = np.where(y_all < self.num_classes)[0]
            self.total_samples = len(self.indices)
            n_filtered = f['Y'].shape[0] - self.total_samples
            if n_filtered > 0:
                print(f"  [Generator] Ignorate {n_filtered} finestre con label >= {self.num_classes}")
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.total_samples / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        idx_batch  = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        idx_sorted = sorted(idx_batch)
        with h5py.File(self.file_path, 'r') as f:
            x = f['X'][idx_sorted]
            y = f['Y'][idx_sorted]
        x_limbs = x[:, :6, :]
        x_out   = np.transpose(x_limbs, (0, 2, 1))
        y_out   = to_categorical(y, num_classes=self.num_classes)
        return x_out, y_out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, train_gen, val_gen, x_test, y_test, x_val, y_val,
                save_path, ep, lr, bs, cm_test_path, cm_val_path):
    opt = Adam(learning_rate=lr)
    ME  = tf.keras.metrics.F1Score(average='macro', name='f1_score')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', ME])

    callbacks = [
        ModelCheckpoint(save_path, monitor='val_f1_score', verbose=1,
                        save_best_only=True, save_weights_only=True, mode='max'),
        EarlyStopping(monitor='val_f1_score', patience=4, verbose=1,
                      mode='max', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_f1_score', factor=0.5, patience=5,
                          min_lr=1e-5, mode='max', verbose=1),
    ]

    print(f"\nInizio addestramento SSL (BS={bs}, EP={ep}, LR={lr})...")
    model.fit(train_gen, epochs=ep, validation_data=val_gen, callbacks=callbacks, verbose=1)

    print("\n" + "=" * 55)
    print("VALUTAZIONE FINALE (migliori pesi)")
    print("=" * 55)
    model.load_weights(save_path)

    metrics, acc = evaluater(x_test, y_test, model, cm_test_path)
    auroc, auprc = evaluater_pro(x_test, y_test, model)

    print(f"\n[TEST SET REALE]")
    print(f"  Accuratezza: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUROC:       {auroc:.4f}")
    print(f"  AuPRC:       {auprc:.4f}")
    print("-" * 55)
    print(f"  {'Classe':<12} | {'Prec.':>6} | {'Recall':>6} | {'Spec.':>6} | {'F1':>6}")
    print("-" * 55)
    for i, row in enumerate(metrics):
        p, r, s, f1 = row
        print(f"  {CLASS_NAMES[i]:<12} | {p:>6.4f} | {r:>6.4f} | {s:>6.4f} | {f1:>6.4f}")
    print("-" * 55)

    evaluater(x_val, y_val, model, cm_val_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponibili: {len(gpus)}")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 55)
    print("STEP 4 - Riaddestramento SSL")
    print("=" * 55)
    print(f"  Train:  {DATASET_TRAIN}")
    print(f"  Val:    {DATASET_VAL}")
    print(f"  Test:   {DATASET_TEST}")
    print(f"  Pesi:   {SAVE_PATH}")

    # Carica test set reale
    print(f"\nCaricamento test set reale...")
    with h5py.File(DATASET_TEST, 'r') as f:
        y_all     = f['Y'][:]
        valid_idx = np.where(y_all < 6)[0]
        x_test    = np.transpose(f['X'][valid_idx, :6, :], (0, 2, 1))
        y_test    = y_all[valid_idx]
    print(f"  Campioni test: {len(y_test)}")

    # Carica val set simulato
    print(f"Caricamento validation set...")
    with h5py.File(DATASET_VAL, 'r') as f:
        x_val = np.transpose(f['X'][:, :6, :], (0, 2, 1))
        y_val = f['Y'][:]
    print(f"  Campioni val: {len(y_val)}")

    # Verifica training set SSL
    print(f"Verifica training set SSL...")
    with h5py.File(DATASET_TRAIN, 'r') as f:
        y_train = f['Y'][:]
    print(f"  Totale finestre: {len(y_train):,}")
    for c, name in enumerate(CLASS_NAMES):
        n = int(np.sum(y_train == c))
        print(f"    {name:<12}: {n:>7,}  ({100*n/len(y_train):.1f}%)")

    input_shape = (SAMPLES_PER_WINDOW, 6)
    output_dims = 6
    EP = 50
    LR = 1e-3
    BS = 256

    print(f"\n--- CONFIGURAZIONE ---")
    print(f"  Input shape: {input_shape}")
    print(f"  Batch size:  {BS}")
    print(f"  Epochs max:  {EP}")
    print(f"  LR iniziale: {LR}")

    train_gen = H5DataGenerator(DATASET_TRAIN, batch_size=BS, num_classes=output_dims, shuffle=True)
    val_gen   = H5DataGenerator(DATASET_VAL,   batch_size=BS, num_classes=output_dims, shuffle=False)

    model = build_model(input_shape, output_dims)

    train_model(model, train_gen, val_gen,
                x_test, y_test, x_val, y_val,
                SAVE_PATH, EP, LR, BS, CM_TEST, CM_VAL)
