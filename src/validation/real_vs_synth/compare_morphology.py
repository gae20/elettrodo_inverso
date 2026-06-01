import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sp_signal

# Aggiungi src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data.data_pipeline import get_ecg, all_leads_preprocessing, limb_interchange_simulation, bandpass_filter
from utils.config import ALL_LEADS, LIMB_LEADS, MAPPING_INV, LABEL_MAP_CLEAN, FS_OLD

# Configura i percorsi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "..", "datasets", "dataset", "thesis-sample.csv"))

# --- HELPER PER IL RUMORE ---

def add_realistic_noise(sigs_dict, fs=FS_OLD):
    """Aggiunge deriva e EMG ai segnali raw."""
    noisy_sigs = {}
    t = np.arange(next(iter(sigs_dict.values())).size) / fs
    
    for lead, sig in sigs_dict.items():
        # 1. Deriva casuale (0.1 - 0.4 Hz)
        freq = np.random.uniform(0.1, 0.4)
        amp = np.random.uniform(50, 300) # Ampiezza in uV
        wander = amp * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi))
        
        # 2. EMG (Rumore bianco)
        emg_amp = np.random.uniform(5, 25)
        emg = np.random.normal(0, emg_amp, sig.size)
        
        noisy_sigs[lead] = sig + wander + emg
    return noisy_sigs

# --- NORMALIZZAZIONI ---

def zscore_per_channel(sigs_array):
    """Z-score classico per ogni lead (rimuove ampiezze relative)."""
    x = sigs_array.astype(np.float32)
    means = x.mean(axis=1, keepdims=True)
    stds = x.std(axis=1, keepdims=True)
    stds[stds < 1e-8] = 1.0
    return (x - means) / stds

def robust_global_scale(sigs_array):
    """Z-median globale (preserva ampiezze relative)."""
    x = sigs_array.astype(np.float32)
    medians = np.median(x, axis=1, keepdims=True)
    q75, q25 = np.percentile(x, [75, 25])
    scale = (q75 - q25) / 1.34896
    scale = max(scale, 1e-8)
    return (x - medians) / scale

# --- PIPELINE ---

def legacy_pipeline(raw_sigs, mode):
    """Pipeline Attuale: Filtro -> Swap -> Z-score per-canale."""
    # 1. Filtro
    sigs_filtered = all_leads_preprocessing(raw_sigs)
    # 2. Swap
    sigs_swapped = limb_interchange_simulation(mode, sigs_filtered)
    # 3. Array + Z-score
    x = np.array([sigs_swapped[l] for l in LIMB_LEADS], dtype=np.float32)
    return zscore_per_channel(x)

def proposed_pipeline(raw_sigs, mode):
    """Pipeline Proposta: Rumore -> Swap -> Filtro -> Z-median globale."""
    # 1. Aggiunta Rumore al RAW
    sigs_noisy = add_realistic_noise(raw_sigs)
    # 2. Swap al RAW
    sigs_swapped = limb_interchange_simulation(mode, sigs_noisy)
    # 3. Filtro (Preprocessing)
    sigs_filtered = all_leads_preprocessing(sigs_swapped)
    # 4. Array + Robust Global Scale
    x = np.array([sigs_filtered[l] for l in LIMB_LEADS], dtype=np.float32)
    return robust_global_scale(x)

def run_pipeline_test(class_name='LA-RA'):
    print(f"\n=== TEST PIPELINE COMPARATIVO: {class_name} ===")
    
    df = pd.read_csv(CSV_PATH)
    df_valido = df[df["Inversione"] != "?"].copy()
    df_valido["Inversione"] = df_valido["Inversione"].apply(lambda x: LABEL_MAP_CLEAN.get(x, x))
    
    # Prendi esempi
    real_id = df_valido[df_valido["Inversione"] == class_name].iloc[0]["Num"]
    norm_id = df_valido[df_valido["Inversione"] == "normale"].iloc[0]["Num"]
    
    data_real = get_ecg(real_id)
    data_norm = get_ecg(norm_id)
    
    mode = MAPPING_INV[class_name]
    
    # 1. Segnale REALE (Reference)
    sigs_real_proc = all_leads_preprocessing(data_real["signals"])
    x_real = np.array([sigs_real_proc[l] for l in LIMB_LEADS], dtype=np.float32)
    x_real_ref = robust_global_scale(x_real)
    
    # 2. Sintetico LEGACY
    x_legacy = legacy_pipeline(data_norm["signals"], mode)
    
    # 3. Sintetico PROPOSTO
    x_proposed = proposed_pipeline(data_norm["signals"], mode)
    
    # Allineamento e Plot
    min_len = 1000 # 4 secondi a 250Hz
    t = np.arange(min_len) / 250.0
    
    fig, axes = plt.subplots(6, 1, figsize=(15, 20), sharex=True)
    fig.suptitle(f"Test di Omologazione: {class_name}\nReale vs Legacy vs Proposto", fontsize=18)
    
    print(f"\n{'Lead':<6} | {'Real Std':<10} | {'Legacy Err':<12} | {'Proposed Err':<12}")
    print("-" * 50)
    
    for i, lead in enumerate(LIMB_LEADS):
        r = x_real_ref[i, :min_len]
        l = x_legacy[i, :min_len]
        p = x_proposed[i, :min_len]
        
        std_r = np.std(r)
        std_l = np.std(l)
        std_p = np.std(p)
        
        err_l = np.abs(std_r - std_l)
        err_p = np.abs(std_r - std_p)
        print(f"{lead:<6} | {std_r:10.4f} | {err_l:12.4f} | {err_p:12.4f}")
        
        axes[i].plot(t, r, label='REALE (Target)', color='black', linewidth=1.5, alpha=0.9)
        axes[i].plot(t, l, label='LEGACY (Per-Channel)', color='red', linestyle='--', alpha=0.6)
        axes[i].plot(t, p, label='PROPOSTO (Global + Noise)', color='green', alpha=0.8)
        
        axes[i].set_title(f"Lead {lead} (Real Std: {std_r:.2f}, Legacy Err: {err_l:.2f}, Prop Err: {err_p:.2f})")
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.2)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(BASE_DIR, f"turing_test_{class_name}.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nGrafico salvato: {out_path}")

if __name__ == "__main__":
    run_pipeline_test('LA-RA')
