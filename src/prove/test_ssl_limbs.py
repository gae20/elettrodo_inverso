"""
test_ssl_limbs.py

Valuta il modello SSL sui dati reali e stampa un confronto diretto
con il modello originale (unlabelled_z_median_limbs).

Output:
  - Classification report (Precision / Recall / F1 per classe)
  - Accuracy, AUROC, AuPRC
  - Confusion matrix salvata in: self_supervised_weights_and_cm/ssl_cm_realtest.png
  - Confronto numerico tra SSL e modello originale
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from keras.utils import to_categorical

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.ldensenet import build_model
from utils.config import SAMPLES_PER_WINDOW

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

CLASS_NAMES = ['normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']

# --- Percorsi ---
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR     = os.path.join(BASE_DIR, '..', 'training')
DATASET_TEST     = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', 'datasets', 'labelled_z_median_limbs_test_validation.h5'))

# Pesi SSL — salvati da step4_train_ssl.py
SSL_DIR      = os.path.join(BASE_DIR, 'self_supervised', 'results', 'ssl_weights')
WEIGHTS_SSL  = os.path.join(SSL_DIR, 'best_model_ssl_limbs.weights.h5')
# Pesi originale (per confronto)
WEIGHTS_ORIG = os.path.join(BASE_DIR, 'models', 'best_model_final_noise_limbs.weights.h5')

OUT_DIR = SSL_DIR
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------

def load_test_data(path):
    with h5py.File(path, 'r') as f:
        y_all = f['Y'][:]
        valid_idx = np.where(y_all < 6)[0]
        x_raw = f['X'][valid_idx, :6, :]
        x = np.transpose(x_raw, (0, 2, 1))
        y = y_all[valid_idx]
    print(f"  Campioni test: {len(y)}  |  Classi: {np.unique(y)}")
    return x, y


def evaluate(model, x, y, cm_path=None, model_name=""):
    y_probs = model.predict(x, batch_size=64, verbose=0)
    y_pred  = np.argmax(y_probs, axis=1)
    acc     = np.mean(y_pred == y)

    C = confusion_matrix(y, y_pred, labels=range(6))

    y_oh   = to_categorical(y, num_classes=6)
    auroc  = roc_auc_score(y_oh, y_probs, multi_class='ovr', average='macro')
    auprc  = average_precision_score(y_oh, y_probs, average='macro')

    if cm_path:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        im = ax.matshow(C, cmap=plt.cm.Reds)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, str(C[i, j]),
                        ha='center', va='center', fontsize=10)
        ax.set_xticks(range(6)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='left')
        ax.set_yticks(range(6)); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix — {model_name}', pad=20)
        plt.tight_layout()
        plt.savefig(cm_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Matrice di confusione salvata in: {cm_path}")

    return acc, auroc, auprc, C, y_probs, y_pred


def print_report(y, y_pred, acc, auroc, auprc, C, model_name):
    print(f"\n{'='*55}")
    print(f"  MODELLO: {model_name}")
    print(f"{'='*55}")
    print(classification_report(y, y_pred, target_names=CLASS_NAMES, digits=3))
    print(f"  Accuratezza Totale : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUROC (Macro)      : {auroc:.4f}")
    print(f"  AuPRC (Macro)      : {auprc:.4f}")
    print(f"\n  [ANALISI ERRORI]")
    for i, name in enumerate(CLASS_NAMES):
        tp     = C[i, i]
        total  = C[i].sum()
        errors = [(CLASS_NAMES[j], C[i, j]) for j in range(6) if j != i and C[i, j] > 0]
        errors.sort(key=lambda x: -x[1])
        if errors:
            err_str = ', '.join([f"{n}={v}" for n, v in errors])
            print(f"    {name:<12} (n={total:>4}): TP={tp:>4} | confuso con: {err_str}")
        else:
            print(f"    {name:<12} (n={total:>4}): TP={tp:>4} | nessun errore")


def print_comparison(metrics_orig, metrics_ssl):
    acc_o, auroc_o, auprc_o = metrics_orig
    acc_s, auroc_s, auprc_s = metrics_ssl

    print(f"\n{'='*55}")
    print("  CONFRONTO ORIGINALE vs SSL")
    print(f"{'='*55}")
    print(f"  {'Metrica':<20} {'Originale':>10} {'SSL':>10} {'Delta':>10}")
    print(f"  {'-'*50}")

    def delta_str(new, old):
        d = new - old
        sign = '+' if d >= 0 else ''
        return f"{sign}{d*100:.2f}%"

    print(f"  {'Accuracy':<20} {acc_o*100:>9.2f}% {acc_s*100:>9.2f}% {delta_str(acc_s, acc_o):>10}")
    print(f"  {'AUROC (macro)':<20} {auroc_o:>10.4f} {auroc_s:>10.4f} {delta_str(auroc_s, auroc_o):>10}")
    print(f"  {'AuPRC (macro)':<20} {auprc_o:>10.4f} {auprc_s:>10.4f} {delta_str(auprc_s, auprc_o):>10}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    input_shape = (SAMPLES_PER_WINDOW, 6)
    output_dims = 6

    print("=" * 55)
    print("VALUTAZIONE SSL — Test Set Reale")
    print("=" * 55)

    print(f"\nCaricamento test set da: {DATASET_TEST}")
    x_test, y_test = load_test_data(DATASET_TEST)

    # --- Modello originale ---
    print(f"\n[1/2] Modello ORIGINALE: {WEIGHTS_ORIG}")
    model = build_model(input_shape, output_dims)
    model.load_weights(WEIGHTS_ORIG)
    cm_orig = os.path.join(OUT_DIR, 'comparison_cm_original.png')
    acc_o, auroc_o, auprc_o, C_o, _, y_pred_o = evaluate(
        model, x_test, y_test, cm_path=cm_orig, model_name="Originale"
    )
    print_report(y_test, y_pred_o, acc_o, auroc_o, auprc_o, C_o, "Originale")

    # --- Modello SSL ---
    print(f"\n[2/2] Modello SSL: {WEIGHTS_SSL}")
    model_ssl = build_model(input_shape, output_dims)
    model_ssl.load_weights(WEIGHTS_SSL)
    cm_ssl = os.path.join(OUT_DIR, 'ssl_cm_realtest.png')
    acc_s, auroc_s, auprc_s, C_s, _, y_pred_s = evaluate(
        model_ssl, x_test, y_test, cm_path=cm_ssl, model_name="SSL"
    )
    print_report(y_test, y_pred_s, acc_s, auroc_s, auprc_s, C_s, "SSL")

    # --- Confronto ---
    print_comparison(
        (acc_o, auroc_o, auprc_o),
        (acc_s, auroc_s, auprc_s)
    )
