import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Percorsi
TRAIN_H5 = "../../datasets/unlabelled_simulated_gain/unlabelled_targeted_noise_limbs_train.h5"
TEST_H5  = "../../datasets/labelled_z_median_limbs_test_validation.h5"
OUT_PLOT = "noise_distribution_comparison_h5.png"

def calculate_noise_metric(win):
    """
    Calcola il rumore come MAD della derivata prima (mad_diff).
    win shape: (6, 500) o (12, 500)
    Prendiamo solo le prime 6 derivazioni (limbs).
    """
    x = win[:6, :]
    dx = np.diff(x, axis=-1)
    # MAD della derivata per ogni lead
    mads = np.median(np.abs(dx - np.median(dx, axis=-1, keepdims=True)), axis=-1)
    # Media sulle lead I, II, III (più indicative del rumore periferico)
    return np.mean(mads[:3])

def collect_mads(h5_path, name, max_samples=1000):
    print(f"Analisi {name}...")
    with h5py.File(h5_path, 'r') as f:
        X = f['X']
        Y = f['Y'][:]
        
        # Analizziamo tutte le classi da 0 a 5
        mads_per_class = {i: [] for i in range(6)}
        
        for cls in range(6):
            idx = np.where(Y == cls)[0]
            if len(idx) == 0: continue
            if len(idx) > max_samples:
                idx = np.random.choice(idx, max_samples, replace=False)
            
            for i in tqdm(idx, desc=f"Classe {cls}", leave=False):
                mads_per_class[cls].append(calculate_noise_metric(X[i]))
                
    return mads_per_class

def main():
    if not os.path.exists(TEST_H5):
        print(f"Errore: Il file {TEST_H5} non esiste.")
        return

    real_mads = collect_mads(TEST_H5, "REALE (Testset)")

    plt.figure(figsize=(14, 8))
    
    colors = sns.color_palette("husl", 6)
    class_names = ['Normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']

    for cls in range(6):
        if len(real_mads[cls]) > 0:
            sns.kdeplot(real_mads[cls], label=f'REAL {class_names[cls]} (Cls {cls})', color=colors[cls], fill=True, alpha=0.1)

    plt.title("Distribuzione Rumore per TUTTE le Classi REALI\n(MAD della derivata)")
    plt.xlabel("MAD Diff")
    plt.ylabel("Densità")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 0.8)
    
    plt.savefig("real_noise_all_classes.png", dpi=150)
    print("\nGrafico salvato in: real_noise_all_classes.png")
    plt.show()

    plt.title("Confronto Distribuzione Rumore (MAD della derivata)\nSimulato vs Reale")
    plt.xlabel("MAD Diff (indicatore di rumore)")
    plt.ylabel("Densità")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(OUT_PLOT, dpi=150)
    print(f"\nGrafico salvato in: {OUT_PLOT}")
    plt.show()

if __name__ == "__main__":
    main()
