"""
test_base.py

Valuta il modello base (LDenseNet) sul test set fisso:
1. Carica test_small.h5.
2. Carica i pesi da results/model_base.weights.h5.
3. Esegue la valutazione stampando le metriche sul test set.
4. Salva la matrice di confusione in results/cm_base.png.
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

# --- Setup path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.join(SCRIPT_DIR, '..')
THESIS_DIR = os.path.join(SRC_DIR, '..')

sys.path.append(SRC_DIR)
from models.ldensenet import build_model

# --- Percorsi ---
H5_DIR         = os.path.join(SCRIPT_DIR, 'results', 'semi_h5')
RESULTS_DIR    = os.path.join(SCRIPT_DIR, 'results')
TEST_H5        = os.path.join(H5_DIR, 'test_small.h5')

# --- Costanti ---
CLASS_NAMES = ['normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']

def load_h5_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
    X_transposed = np.transpose(X, (0, 2, 1)) # Da (N, 6, 500) a (N, 500, 6)
    return X_transposed, Y

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

def evaluate_on_test(x_test, y_test, model, save_cm_path):
    y_probs    = model.predict(x_test, batch_size=256, verbose=0)
    y_pred_idx = np.argmax(y_probs, axis=1)
    acc        = np.mean(y_pred_idx == y_test)
    cm         = confusion_matrix(y_test, y_pred_idx, labels=range(6))

    # Salva matrice di confusione
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.matshow(cm, cmap=plt.cm.Blues)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.annotate(str(cm[j, i]), xy=(i, j),
                        horizontalalignment='center', verticalalignment='center')
    ax.set_xticks(range(6)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='left')
    ax.set_yticks(range(6)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_ylabel('Classe Reale')
    ax.set_xlabel('Classe Predetta')
    plt.tight_layout()
    plt.savefig(save_cm_path, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close()

    metrics = cal_metrics(cm)
    y_test_oh = to_categorical(y_test, num_classes=6)
    auroc = roc_auc_score(y_test_oh, y_probs, multi_class='ovr', average='macro')
    auprc = average_precision_score(y_test_oh, y_probs, average='macro')

    return acc, auroc, auprc, metrics, cm

if __name__ == '__main__':
    # Imposta seed globale per coerenza totale
    tf.keras.utils.set_random_seed(42)

    print("=" * 60)
    print("TEST MODELLO BASE (SOLO DATI REALI)")
    print("=" * 60)

    # 1. Carica i dati
    if not os.path.exists(TEST_H5):
        print("ERRORE: Il file test_small.h5 non esiste.")
        sys.exit(1)

    X_test, Y_test = load_h5_data(TEST_H5)
    print(f"  Test set : {X_test.shape[0]} finestre")

    # 2. Carica modello e pesi
    model = build_model((500, 6), 6)
    base_weights_path = os.path.join(RESULTS_DIR, 'model_base.weights.h5')

    if not os.path.exists(base_weights_path):
        print(f"ERRORE: I pesi {base_weights_path} non esistono. Esegui prima train_base.py.")
        sys.exit(1)

    model.load_weights(base_weights_path)
    print("  Pesi modello base caricati correttamente.")

    # 3. Valuta
    cm_path = os.path.join(RESULTS_DIR, 'cm_base.png')
    acc, auroc, auprc, metrics, cm = evaluate_on_test(X_test, Y_test, model, cm_path)

    print(f"\n[RISULTATI MODELLO BASE]")
    print(f"  Accuratezza: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  AUROC:       {auroc:.4f}")
    print(f"  AUPRC:       {auprc:.4f}")

    print("\n  Metriche per classe:")
    print(f"  {'Classe':<12} | {'Precision':<10} | {'Recall':<10} | {'Specif.':<10} | {'F1-score':<10}")
    print("-" * 60)
    for c in range(6):
        m = metrics[c]
        print(f"  {CLASS_NAMES[c]:<12} | {m[0]:<10.4f} | {m[1]:<10.4f} | {m[2]:<10.4f} | {m[3]:<10.4f}")

    print(f"\nMatrice di confusione salvata in: {cm_path}")
    print("=" * 60)
