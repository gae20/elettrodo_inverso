import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import signal

# Aggiungi src al path per importare i moduli interni
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data.data_pipeline import get_ecg, all_leads_preprocessing
from models.ldensenet import build_model
from utils.config import SAMPLES_PER_WINDOW, ALL_LEADS, FS_OLD, FS_NEW

def add_baseline_wander(sig, fs, amplitude=500, freq=0.15):
    """Aggiunge un drift sinusoidale lento."""
    t = np.arange(len(sig)) / fs
    wander = amplitude * np.sin(2 * np.pi * freq * t)
    return sig + wander

def add_emg_noise(sig, amplitude=50):
    """Aggiunge rumore ad alta frequenza (EMG)."""
    noise = np.random.normal(0, amplitude, len(sig))
    return sig + noise

def zscore_ecg(sigs_array, eps=1e-8):
    x = sigs_array.astype(np.float32)
    mean_global = x.mean()
    std_global = x.std()
    std_global = 1.0 if std_global < eps else std_global
    x_norm = (x - mean_global) / std_global
    return x_norm

def test_noise_impact():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, "..", "best_model_unlabelled_z_limbs.weights.h5")
    
    # ID di un ECG normale noto (es. il primo dai test_norm_ids di prima)
    # Per semplicità ne usiamo uno che sappiamo esistere (visto nei log precedenti)
    test_id = 1157430 
    
    print(f"Caricamento ECG normale {test_id}...")
    ecg_data = get_ecg(test_id)
    if not ecg_data:
        print("ECG non trovato.")
        return
    
    raw_signals = ecg_data["signals"]
    
    # 1. Pipeline Originale (Clean)
    sigs_clean = all_leads_preprocessing(raw_signals)
    x_clean = np.array([sigs_clean[l] for l in ALL_LEADS[:6]], dtype=np.float32)
    x_clean_norm = np.transpose(zscore_ecg(x_clean)[:, :SAMPLES_PER_WINDOW], (1, 0))
    
    # 2. Pipeline con Rumore Aggiunto PRIMA del filtraggio
    raw_noisy = {}
    for lead, sig in raw_signals.items():
        # Aggiungiamo Baseline Wander e EMG più aggressivi
        s_n = add_baseline_wander(sig, FS_OLD, amplitude=4000, freq=0.3) # Più ampiezza e freq
        s_n = add_emg_noise(s_n, amplitude=300) # Più rumore EMG
        raw_noisy[lead] = s_n
        
    sigs_noisy = all_leads_preprocessing(raw_noisy) # Notch + Bandpass 0.5-120Hz
    x_noisy = np.array([sigs_noisy[l] for l in ALL_LEADS[:6]], dtype=np.float32)
    x_noisy_norm = np.transpose(zscore_ecg(x_noisy)[:, :SAMPLES_PER_WINDOW], (1, 0))
    
    # 3. Caricamento Modello e Inferenza
    model = build_model((SAMPLES_PER_WINDOW, 6), 6)
    model.load_weights(weights_path)
    
    p_clean = model.predict(np.expand_dims(x_clean_norm, 0))[0]
    p_noisy = model.predict(np.expand_dims(x_noisy_norm, 0))[0]
    
    print("\n--- Risultati Inferenza ---")
    print(f"Originale: Pred={np.argmax(p_clean)}, Conf={np.max(p_clean):.4f}")
    print(f"Probabilità: {p_clean}")
    print(f"\nCon Rumore Filtrato: Pred={np.argmax(p_noisy)}, Conf={np.max(p_noisy):.4f}")
    print(f"Probabilità: {p_noisy}")
    
    # 4. Visualizzazione
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(x_clean_norm[:, 0], label='Clean (Lead I)')
    plt.plot(x_noisy_norm[:, 0], label='Noisy + Preproc (Lead I)', alpha=0.7)
    plt.title(f"Effetto del rumore filtrato (0.5-120Hz)\nPred Originale: {np.argmax(p_clean)} | Pred Noisy: {np.argmax(p_noisy)}")
    plt.legend()
    
    # Confronto Spettrale localizzato
    plt.subplot(2, 1, 2)
    f_c, p_c = signal.welch(x_clean_norm[:, 0], fs=FS_NEW)
    f_n, p_n = signal.welch(x_noisy_norm[:, 0], fs=FS_NEW)
    plt.semilogy(f_c, p_c, label='Clean')
    plt.semilogy(f_n, p_n, label='Noisy + Preproc')
    plt.title("Confronto PSD")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "noise_simulation_result.png"))
    print(f"\nGrafico salvato in: noise_simulation_result.png")

if __name__ == "__main__":
    test_noise_impact()
