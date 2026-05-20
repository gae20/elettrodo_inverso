import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Aggiungi src al path per importare i moduli interni
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.ldensenet import build_model
from data.data_pipeline import compute_window_features
from utils.config import SAMPLES_PER_WINDOW, QUALITY_CFG

def profile_errors():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, "..", "best_model_unlabelled_z_median_limbs.weights.h5")
    dataset_path = os.path.join(base_dir, "..", "unlabelled_z_median_limbs_test_validation.h5")
    
    if not os.path.exists(weights_path) or not os.path.exists(dataset_path):
        print("Errore: Pesi o dataset non trovati.")
        return

    # 1. Caricamento Dati
    with h5py.File(dataset_path, 'r') as f:
        x_real_raw = f['X'][:, :6, :]
        x_real = np.transpose(x_real_raw, (0, 2, 1))
        y_real = f['Y'][:]
    
    # 2. Caricamento Modello
    input_shape = (SAMPLES_PER_WINDOW, 6)
    model = build_model(input_shape, 6)
    model.load_weights(weights_path)
    
    # 3. Predizione
    y_probs = model.predict(x_real, batch_size=32)
    y_pred = np.argmax(y_probs, axis=1)
    
    # 4. Matrice di Confusione
    cm = confusion_matrix(y_real, y_pred, labels=range(6))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix on REAL Data')
    plt.savefig(os.path.join(base_dir, "error_confusion_matrix.png"))
    plt.close()
    
    # 5. Analisi Qualità degli Errori (False Positives)
    # Casi dove era normale (0) ma ha predetto anomalia (!=0)
    fp_indices = np.where((y_real == 0) & (y_pred != 0))[0]
    # Casi dove era normale (0) ed è stato predetto correttamente (0)
    tp_indices = np.where((y_real == 0) & (y_pred == 0))[0]
    
    print(f"\n--- Analisi Qualità dei {len(fp_indices)} Falsi Positivi vs True Positives ---")
    
    # Breakdown dei Falsi Positivi
    fp_preds = y_pred[fp_indices]
    unique_preds, counts = np.unique(fp_preds, return_counts=True)
    print("\nI Normali (0) sono stati scambiati per:")
    for cls, count in zip(unique_preds, counts):
        print(f"Classe {cls}: {count} casi")
        
    def get_avg_mad(indices):
        mads = []
        for idx in indices:
            win = x_real[idx, :, 0]
            feats = compute_window_features(win, cfg=QUALITY_CFG, lead_idx=0)
            mads.append(feats['mad_diff'])
        return np.mean(mads) if mads else 0

    mad_fp = get_avg_mad(fp_indices)
    mad_tp = get_avg_mad(tp_indices)
    
    print(f"MAD Medio nei False Positives: {mad_fp:.4f}")
    print(f"MAD Medio nei True Positives:  {mad_tp:.4f}")
    
    if mad_fp > mad_tp * 1.5:
        print("\nSospetto CONFERMATO: I Falsi Positivi sono significativamente più rumorosi dei segnali corretti.")
    else:
        print("\nSospetto INCERTO: Il rumore (MAD) non sembra essere l'unica causa dei Falsi Positivi.")
    
    # 6. Salvataggio Galleria Errori
    errors = np.where(y_pred != y_real)[0]
    confidences = np.max(y_probs, axis=1)
    high_conf_errors = errors[np.argsort(confidences[errors])[::-1]]
    
    plt.figure(figsize=(15, 12))
    n_show = min(12, len(high_conf_errors))
    for i in range(n_show):
        idx = high_conf_errors[i]
        plt.subplot(4, 3, i+1)
        plt.plot(x_real[idx, :, 0], label='Lead I', alpha=0.8)
        plt.plot(x_real[idx, :, 1], label='Lead II', alpha=0.8)
        # Mostra anche il MAD del sample
        sample_mad = compute_window_features(x_real[idx, :, 0], cfg=QUALITY_CFG, lead_idx=0)['mad_diff']
        plt.title(f"T:{y_real[idx]} P:{y_pred[idx]} Conf:{confidences[idx]:.2f}\nMAD:{sample_mad:.2f}")
        plt.legend(prop={'size': 6})
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "top_errors_gallery.png"))
    plt.close()
    
    # 7. Riepilogo Finale
    fn_norm = np.where((y_real != 0) & (y_pred == 0))[0]
    print(f"\nFalse Positives (Normal as Anomaly): {len(fp_indices)}")
    print(f"False Negatives (Anomaly as Normal): {len(fn_norm)}")

if __name__ == "__main__":
    profile_errors()
