import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def load_class_data(path, target_class):
    with h5py.File(path, 'r') as f:
        y = f['Y'][:]
        idx = np.where(y == target_class)[0]
        # x shape in h5 is (N, 6, 500)
        x = f['X'][idx, :6, :]
    return x

def compute_correlations(x):
    # x shape: (N, 6, 500)
    # Lead I is index 0, Lead II is index 1
    corrs = []
    for i in range(x.shape[0]):
        lead_I = x[i, 0, :]
        lead_II = x[i, 1, :]
        # Check if arrays are not constant to avoid pearsonr warning/NaN
        if np.std(lead_I) > 1e-6 and np.std(lead_II) > 1e-6:
            r, _ = pearsonr(lead_I, lead_II)
            corrs.append(r)
    return corrs

def test_correlation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_synth = os.path.join(base_dir, "limbs_synthetic_pure.h5")
    path_real = os.path.join(base_dir, "..", "unlabelled_z_median_limbs_test_validation.h5")
    
    print("Calcolo della correlazione Lead I vs Lead II...")
    
    # 1. Normale (Classe 0)
    x_synth_0 = load_class_data(path_synth, 0)
    x_real_0 = load_class_data(path_real, 0)
    corr_s_0 = compute_correlations(x_synth_0)
    corr_r_0 = compute_correlations(x_real_0)
    
    # 2. LA-LL (Classe 3)
    x_synth_3 = load_class_data(path_synth, 3)
    x_real_3 = load_class_data(path_real, 3)
    corr_s_3 = compute_correlations(x_synth_3)
    corr_r_3 = compute_correlations(x_real_3)
    
    print("\n--- Risultati Mediani (Correlazione di Pearson I vs II) ---")
    print(f"Normale (Classe 0) - Sintetico: {np.nanmedian(corr_s_0):.4f}")
    print(f"Normale (Classe 0) - Reale:     {np.nanmedian(corr_r_0):.4f}")
    print(f"LA-LL (Classe 3)   - Sintetico: {np.nanmedian(corr_s_3):.4f}")
    print(f"LA-LL (Classe 3)   - Reale:     {np.nanmedian(corr_r_3):.4f}")
    
    # Plot Histograms
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    axs[0].hist(corr_s_0, bins=50, alpha=0.5, label='Synth Normal', density=True)
    axs[0].hist(corr_r_0, bins=50, alpha=0.5, label='Real Normal', density=True)
    axs[0].set_title("Distribuzione Correlazione (I vs II) - NORMALE")
    axs[0].legend()
    
    axs[1].hist(corr_s_3, bins=50, alpha=0.5, label='Synth LA-LL (Class 3)', density=True)
    axs[1].hist(corr_r_3, bins=50, alpha=0.5, label='Real LA-LL (Class 3)', density=True)
    axs[1].set_title("Distribuzione Correlazione (I vs II) - LA-LL")
    axs[1].legend()
    
    plt.tight_layout()
    out_path = os.path.join(base_dir, "correlation_I_II_test.png")
    plt.savefig(out_path)
    print(f"\nGrafico salvato in: {out_path}")

if __name__ == "__main__":
    test_correlation()
