import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import zipfile
import sqlite3
from tqdm import tqdm

# Aggiungi src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.data_pipeline import (
    read_edf_data, all_leads_preprocessing, apply_electrode_gain, 
    add_extra_noise, check_ecg_quality
)
from utils.config import ALL_LEADS, FS_OLD, QUALITY_CFG

# Percorsi
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DB_PATH = os.path.join(BASE_DIR, "datasets", "dataset", "records.db")
ZIP_DIR = os.path.join(BASE_DIR, "datasets", "dataset", "DATASET")
REAL_H5 = os.path.join(BASE_DIR, "datasets", "labelled_z_median_limbs_test_validation.h5")

def calculate_noise_metric(win_array):
    """Calcola MAD diff sulle prime 3 lead (I, II, III)."""
    # Se win_array ha 12 canali, prendiamo solo i primi 6 (limbs)
    x = win_array[:6, :]
    dx = np.diff(x, axis=-1)
    mads = np.median(np.abs(dx - np.median(dx, axis=-1, keepdims=True)), axis=-1)
    return np.mean(mads[:3])

def get_real_mads_from_h5(max_samples=1000):
    print(f"Estrazione MAD dai dati REALI ({os.path.basename(REAL_H5)})...")
    with h5py.File(REAL_H5, 'r') as f:
        X = f['X']
        Y = f['Y'][:]
        mads = {i: [] for i in range(6)}
        for cls in range(6):
            idx = np.where(Y == cls)[0]
            if len(idx) == 0: continue
            if len(idx) > max_samples: idx = np.random.choice(idx, max_samples, replace=False)
            for i in tqdm(idx, desc=f"Classe {cls}", leave=False):
                mads[cls].append(calculate_noise_metric(X[i]))
    return mads

def robust_scale_ecg(sigs_array, eps=1e-8):
    x = sigs_array.astype(np.float32)
    medians = np.median(x, axis=1, keepdims=True)
    q75, q25 = np.percentile(x, [75, 25])
    iqr_global = q75 - q25
    scale_global = iqr_global / 1.34896
    scale_global = max(scale_global, eps)
    return (x - medians) / scale_global

def simulate_with_params(base_multiplier, targeted_range, n_samples=200):
    """
    Simula on-the-fly con i nuovi parametri.
    targeted_range: tuple (min, max) per il moltiplicatore extra.
    """
    print(f"Simulazione con Base={base_multiplier}, Targeted={targeted_range}...")
    
    # Prendiamo record base puliti
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB non trovato: {DB_PATH}")
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM records WHERE status='reported' LIMIT 50")
    ids = [str(r[0]) for r in cursor.fetchall()]
    conn.close()
    
    if not os.path.exists(ZIP_DIR):
        raise FileNotFoundError(f"ZIP_DIR non trovata: {ZIP_DIR}")
        
    zip_files = [f for f in os.listdir(ZIP_DIR) if f.endswith('.zip')]
    base_signals = []
    
    for eid in ids:
        found = False
        for zf in zip_files:
            try:
                z_path = os.path.join(ZIP_DIR, zf)
                with zipfile.ZipFile(z_path, 'r') as z:
                    if f"{eid}.edf" in z.namelist():
                        base_signals.append(read_edf_data(z.read(f"{eid}.edf"))["signals"])
                        found = True
                        break
            except: continue
        if found and len(base_signals) >= 10: break

    if not base_signals:
        raise ValueError("Nessun segnale base caricato. Controlla i percorsi dei file ZIP.")

    sim_mads = {1: [], 5: []}
    
    for i in tqdm(range(n_samples), desc="Simulazione", leave=False):
        raw = base_signals[np.random.randint(len(base_signals))]
        
        # Classe 1 (LA-RA) - Solo base
        s1 = apply_electrode_gain(raw, fs=FS_OLD, noise_multiplier=base_multiplier)
        p1 = all_leads_preprocessing(s1)
        
        # Verifica lunghezza minima
        lead_len = len(next(iter(p1.values())))
        if lead_len < 1000: continue
        
        # Estrazione finestra e NORMALIZZAZIONE
        win1_raw = np.array([p1[l][500:1000] for l in ALL_LEADS])
        win1 = robust_scale_ecg(win1_raw)
        m1 = calculate_noise_metric(win1)
        sim_mads[1].append(m1)
        
        # Classe 5 (ROT_ANT) - Base + Targeted Random
        mult = np.random.uniform(*targeted_range)
        s5 = apply_electrode_gain(raw, fs=FS_OLD, noise_multiplier=base_multiplier)
        s5_extra = add_extra_noise(s5, multiplier=mult, fs=FS_OLD)
        p5 = all_leads_preprocessing(s5_extra)
        
        # Estrazione finestra e NORMALIZZAZIONE
        win5_raw = np.array([p5[l][500:1000] for l in ALL_LEADS])
        win5 = robust_scale_ecg(win5_raw)
        m5 = calculate_noise_metric(win5)
        sim_mads[5].append(m5)
        
        if i == 0:
            print(f"\n[DEBUG] Sample 0 - MAD LA-RA: {m1:.4f}, MAD ROT_ANT: {m5:.4f} (mult: {mult:.2f})")

    print(f"Fine simulazione. Campioni validi: LA-RA={len(sim_mads[1])}, ROT_ANT={len(sim_mads[5])}")
    return sim_mads

def simulate_all_classes(base_multiplier, n_samples=200):
    print(f"Simulazione Multi-Classe (Base={base_multiplier})...")
    
    # Prendiamo record base puliti
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM records WHERE status='reported' LIMIT 50")
    ids = [str(r[0]) for r in cursor.fetchall()]
    conn.close()
    
    zip_files = [f for f in os.listdir(ZIP_DIR) if f.endswith('.zip')]
    base_signals = []
    for eid in ids:
        found = False
        for zf in zip_files:
            try:
                with zipfile.ZipFile(os.path.join(ZIP_DIR, zf), 'r') as z:
                    if f"{eid}.edf" in z.namelist():
                        base_signals.append(read_edf_data(z.read(f"{eid}.edf"))["signals"])
                        found = True
                        break
            except: continue
        if found and len(base_signals) >= 10: break

    sim_mads = {i: [] for i in range(6)}
    
    # Definizione range rumore extra per classe
    # 0: Norm, 1: LA-RA, 2: RA-LL, 3: LA-LL, 4: ROT_ORA, 5: ROT_ANT
    extra_ranges = {
        0: None,
        1: None,
        2: (1.2, 4.0),
        3: (1.2, 4.0),
        4: (1.2, 4.0),
        5: (1.5, 8.0)
    }

    for cls in range(6):
        for _ in tqdm(range(n_samples), desc=f"Sim Classe {cls}", leave=False):
            raw = base_signals[np.random.randint(len(base_signals))]
            
            # Applica base augmentation
            s = apply_electrode_gain(raw, fs=FS_OLD, noise_multiplier=base_multiplier)
            
            # Applica targeted noise se previsto
            erange = extra_ranges[cls]
            if erange:
                mult = np.random.uniform(*erange)
                s = add_extra_noise(s, multiplier=mult, fs=FS_OLD)
            
            p = all_leads_preprocessing(s)
            
            # Verifica lunghezza minima
            lead_len = len(next(iter(p.values())))
            if lead_len < 1000: continue
            
            win_raw = np.array([p[l][500:1000] for l in ALL_LEADS])
            win = robust_scale_ecg(win_raw)
            m = calculate_noise_metric(win)
            sim_mads[cls].append(m)
            
    return sim_mads

def main():
    real_mads = get_real_mads_from_h5()
    sim_mads = simulate_all_classes(base_multiplier=1.1)

    plt.figure(figsize=(15, 10))
    colors = sns.color_palette("husl", 6)
    class_names = ['Normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']

    for cls in range(6):
        # Plot Reale (Dashed)
        if len(real_mads[cls]) > 0:
            sns.kdeplot(real_mads[cls], color=colors[cls], linestyle='--', alpha=0.5)
        # Plot Simulato (Solid)
        if len(sim_mads[cls]) > 0:
            sns.kdeplot(sim_mads[cls], label=f'SIM {class_names[cls]}', color=colors[cls], fill=True, alpha=0.2)

    plt.title("CONFRONTO FINALE: Simulazione Multi-Livello vs Reale\n(Solid=Sim, Dashed=Real)")
    plt.xlabel("MAD Diff")
    plt.ylabel("Densità")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 0.8)
    
    out_img = "final_multi_class_noise_comparison.png"
    plt.savefig(out_img, dpi=150)
    print(f"\nGrafico finale salvato in: {out_img}")
    plt.show()

if __name__ == "__main__":
    main()
