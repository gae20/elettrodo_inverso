import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import os

def load_data(path):
    with h5py.File(path, 'r') as f:
        # Consideriamo solo i canali periferici (primi 6)
        x = f['X'][:, :6, :]
        y = f['Y'][:]
    return x, y

def compute_psd(data, fs=250):
    # data shape: (N, 6, 500)
    freqs, psd = welch(data, fs=fs, axis=-1, nperseg=250)
    # Media sulle finestre e sui canali
    mean_psd = np.mean(psd, axis=(0, 1))
    return freqs, mean_psd

def analyze_domains():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_synth = os.path.join(base_dir, "limbs_synthetic_pure.h5")
    path_real = os.path.join(base_dir, "..", "unlabelled_z_median_limbs_test_validation.h5")
    
    if not os.path.exists(path_synth) or not os.path.exists(path_real):
        print("Errore: Dataset non trovati.")
        return

    x_synth, y_synth = load_data(path_synth)
    x_real, y_real = load_data(path_real)
    
    print(f"Dataset Synth: {x_synth.shape}")
    print(f"Dataset Real:  {x_real.shape}")
    
    # 1. Statistiche Temporali
    stats = {
        "Synth": {
            "mean": np.mean(x_synth),
            "std": np.std(x_synth),
            "max": np.max(x_synth),
            "min": np.min(x_synth)
        },
        "Real": {
            "mean": np.mean(x_real),
            "std": np.std(x_real),
            "max": np.max(x_real),
            "min": np.min(x_real)
        }
    }
    
    print("\n--- Statistiche Temporali ---")
    for domain, s in stats.items():
        print(f"[{domain}] Mean: {s['mean']:.4f}, Std: {s['std']:.4f}, Range: [{s['min']:.4f}, {s['max']:.4f}]")
        
    # 2. Analisi Spettrale
    f_s, psd_s = compute_psd(x_synth)
    f_r, psd_r = compute_psd(x_real)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(f_s, psd_s, label='Synthetic', alpha=0.8, linewidth=2)
    plt.semilogy(f_r, psd_r, label='Real', alpha=0.8, linewidth=2)
    plt.title("Confronto Power Spectral Density (PSD)")
    plt.xlabel("Frequenza (Hz)")
    plt.ylabel("Potenza/Frequenza (dB/Hz)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(base_dir, "psd_comparison.png"))
    print(f"\nGrafico PSD salvato in: psd_comparison.png")
    
    # 3. Distribuzione Ampiezze (Istogramma)
    plt.figure(figsize=(10, 6))
    plt.hist(x_synth.flatten(), bins=100, density=True, alpha=0.5, label='Synthetic', color='blue')
    plt.hist(x_real.flatten(), bins=100, density=True, alpha=0.5, label='Real', color='red')
    plt.title("Distribuzione delle Ampiezze (Normalizzate Z-Score)")
    plt.xlabel("Ampiezza")
    plt.ylabel("Densità")
    plt.legend()
    plt.savefig(os.path.join(base_dir, "amplitude_distribution.png"))
    print(f"Istogramma ampiezze salvato in: amplitude_distribution.png")
    
    # 4. Analisi per Classe (Opzionale, ma utile)
    plt.figure(figsize=(15, 10))
    for c in range(6):
        plt.subplot(2, 3, c+1)
        # Prendi un esempio a caso per classe
        idx_s = np.where(y_synth == c)[0]
        idx_r = np.where(y_real == c)[0]
        
        if len(idx_s) > 0 and len(idx_r) > 0:
            plt.plot(x_synth[idx_s[0], 0, :], label='Synth', alpha=0.7)
            plt.plot(x_real[idx_r[0], 0, :], label='Real', alpha=0.7)
            plt.title(f"Classe {c} (Lead I)")
            plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "class_comparison_examples.png"))
    print(f"Esempi morfologici salvati in: class_comparison_examples.png")

if __name__ == "__main__":
    analyze_domains()
