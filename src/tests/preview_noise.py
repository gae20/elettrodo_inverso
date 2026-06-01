import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import zipfile
from tqdm import tqdm

# Aggiungi src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.data_pipeline import (
    read_edf_data, all_leads_preprocessing, apply_electrode_gain, 
    add_extra_noise, check_ecg_quality, check_window_quality
)
from utils.config import ALL_LEADS, FS_OLD, QUALITY_CFG

DB_PATH = "../../datasets/dataset/records.db"
ZIP_DIR = "C:/Users/carme/THESIS/datasets/dataset/DATASET/"
OUT_PLOT = "noise_distribution_preview_targeted.png"

def calculate_mad(sig):
    return np.median(np.abs(sig - np.median(sig))) / 0.6745

def get_real_mads(n_samples=50):
    print(f"Estrazione MAD da {n_samples} record REALI...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM records WHERE status='reported' LIMIT ?", (n_samples,))
    ids = [r[0] for r in cursor.fetchall()]
    conn.close()
    
    mads = []
    zip_files = [f for f in os.listdir(ZIP_DIR) if f.endswith('.zip')]
    
    # Per semplicità cerchiamo nei primi batch
    for ecg_id in tqdm(ids, desc="Reali"):
        found = False
        for zf in zip_files:
            try:
                with zipfile.ZipFile(os.path.join(ZIP_DIR, zf), 'r') as z:
                    if f"{ecg_id}.edf" in z.namelist():
                        edf_bytes = z.read(f"{ecg_id}.edf")
                        ecg_data = read_edf_data(edf_bytes)
                        sigs = all_leads_preprocessing(ecg_data["signals"])
                        mads.append(np.mean([calculate_mad(sigs[l]) for l in ('I', 'II', 'III')]))
                        found = True; break
            except: continue
        if not found: continue
    return mads

def get_simulated_mads(n_samples=100):
    print(f"Generazione MAD per {n_samples} record SIMULATI...")
    # Usiamo un record pulito come base per la simulazione
    # (In realtà ne usiamo diversi per avere variabilità)
    
    mads_std = []
    mads_rot = []
    
    # Prendiamo qualche record base
    zip_files = [f for f in os.listdir(ZIP_DIR) if f.endswith('.zip')]
    with zipfile.ZipFile(os.path.join(ZIP_DIR, zip_files[0]), 'r') as z:
        edf_names = [n for n in z.namelist() if n.endswith('.edf')][:10]
        base_ecgs = [read_edf_data(z.read(n))["signals"] for n in edf_names]

    for _ in tqdm(range(n_samples), desc="Simulati"):
        raw = base_ecgs[np.random.randint(len(base_ecgs))]
        
        # Simulazione STANDARD (0.6x)
        s_std = apply_electrode_gain(raw, fs=FS_OLD, noise_multiplier=0.6)
        p_std = all_leads_preprocessing(s_std)
        mads_std.append(np.mean([calculate_mad(p_std[l]) for l in ('I', 'II', 'III')]))
        
        # Simulazione ROT_ANT (0.6x + 1.4x extra)
        s_rot = apply_electrode_gain(raw, fs=FS_OLD, noise_multiplier=0.6)
        p_rot = all_leads_preprocessing(s_rot)
        p_extra = add_extra_noise(p_rot, multiplier=1.4, fs=FS_OLD)
        mads_rot.append(np.mean([calculate_mad(p_extra[l]) for l in ('I', 'II', 'III')]))
        
    return mads_std, mads_rot

def main():
    real_mads = get_real_mads(40)
    sim_std, sim_rot = get_simulated_mads(100)
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(real_mads, label=f'Reale (N={len(real_mads)})', fill=True, color='green', bw_adjust=0.8)
    sns.kdeplot(sim_std, label=f'Simulato STANDARD (N={len(sim_std)})', fill=True, color='blue', bw_adjust=0.8)
    sns.kdeplot(sim_rot, label=f'Simulato ROT_ANT (N={len(sim_rot)})', fill=True, color='red', bw_adjust=0.8)
    
    plt.title("Distribuzione del Rumore (MAD) - Validazione Calibrazione")
    plt.xlabel("MAD (microvolt)")
    plt.ylabel("Densità")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 60)
    
    plt.savefig(OUT_PLOT, dpi=150)
    print(f"\nConfronto distribuzioni salvato in: {OUT_PLOT}")
    plt.show()

if __name__ == "__main__":
    main()
