import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import io

# Aggiungi src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.data_pipeline import (
    read_edf_data, all_leads_preprocessing, apply_electrode_gain, 
    add_extra_noise, limb_interchange_simulation
)
from utils.config import ALL_LEADS, FS_OLD

# Percorso corretto dallo screenshot
ZIP_DIR = "C:/Users/carme\THESIS/datasets/dataset/DATASET/"

def calculate_mad(sig):
    return np.median(np.abs(sig - np.median(sig))) / 0.6745

def test_targeted_noise():
    # 1. Trova il primo zip batch
    zip_files = [f for f in os.listdir(ZIP_DIR) if f.endswith('.zip')]
    zip_path = os.path.join(ZIP_DIR, zip_files[0])
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        target_edf = [n for n in z.namelist() if n.endswith('.edf')][0]
        edf_bytes = z.read(target_edf)
        ecg_data = read_edf_data(edf_bytes)
    
    raw_signals = ecg_data["signals"]
    
    # 0. BASELINE (Segnale originale preprocessato)
    sigs_base = all_leads_preprocessing(raw_signals)
    mad_base = np.mean([calculate_mad(sigs_base[l]) for l in ('I', 'II', 'III')])
    
    # --- CASO A: STANDARD (Classi 0-4) ---
    signals_std = apply_electrode_gain(raw_signals, fs=FS_OLD, noise_multiplier=0.6)
    sigs_std_proc = all_leads_preprocessing(signals_std)
    
    # --- CASO B: TARGETED (Classe 5 - ROT_ANT) ---
    signals_rot = apply_electrode_gain(raw_signals, fs=FS_OLD, noise_multiplier=0.6)
    sigs_rot_proc = all_leads_preprocessing(signals_rot)
    sigs_rot_extra = add_extra_noise(sigs_rot_proc, multiplier=1.4, fs=FS_OLD)
    
    # --- CALCOLO MAD ---
    mad_std = np.mean([calculate_mad(sigs_std_proc[l]) for l in ('I', 'II', 'III')])
    mad_rot = np.mean([calculate_mad(sigs_rot_extra[l]) for l in ('I', 'II', 'III')])
    
    print("\n" + "="*45)
    print("RISULTATI TEST RUMORE MIRATO (CALIBRATI)")
    print("="*45)
    print(f"MAD Base (Originale):      {mad_base:.2f} uV")
    print(f"MAD Standard (+0.6x Gain): {mad_std:.2f} uV (Inc: {mad_std - mad_base:+.2f})")
    print(f"MAD ROT_ANT (+1.4x Extra): {mad_rot:.2f} uV (Inc: {mad_rot - mad_base:+.2f})")
    print("-" * 45)
    print(f"GAP RUMORE OTTENUTO:       {mad_rot - mad_std:.2f} uV")
    print("="*45)
    
    # --- PLOT ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    t = np.arange(len(sigs_std_proc['I'])) / FS_OLD
    
    axes[0].plot(t, sigs_std_proc['I'], label=f'Standard (MAD: {mad_std:.1f}uV)', color='blue', alpha=0.7)
    axes[0].set_title("Simulazione STANDARD (Classi 0-4)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t, sigs_rot_extra['I'], label=f'ROT_ANT Extra Noise (MAD: {mad_rot:.1f}uV)', color='red', alpha=0.7)
    axes[1].set_title("Simulazione ROT_ANT (Classe 5)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    plt.xlabel("Tempo (s)"); plt.xlim(2, 5)
    plt.tight_layout()
    plt.savefig("test_targeted_noise_result.png", dpi=150)
    print(f"\nGrafico salvato: test_targeted_noise_result.png")

if __name__ == "__main__":
    test_targeted_noise()
