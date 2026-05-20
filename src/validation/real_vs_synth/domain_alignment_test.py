import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Aggiungi src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data.data_pipeline import get_ecg, all_leads_preprocessing, limb_interchange_simulation
from utils.config import ALL_LEADS, LIMB_LEADS, MAPPING_INV, LABEL_MAP_CLEAN, FS_OLD

# Configura i percorsi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "..", "datasets", "dataset", "thesis-sample.csv"))

def add_realistic_noise(sigs_dict, fs=FS_OLD):
    noisy_sigs = {}
    for lead, sig in sigs_dict.items():
        # Deriva e EMG
        t = np.arange(sig.size) / fs
        freq = np.random.uniform(0.1, 0.4)
        amp = np.random.uniform(100, 400)
        wander = amp * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi))
        emg = np.random.normal(0, np.random.uniform(10, 30), sig.size)
        noisy_sigs[lead] = sig + wander + emg
    return noisy_sigs

def robust_global_scale(sigs_array):
    x = sigs_array.astype(np.float32)
    medians = np.median(x, axis=1, keepdims=True)
    q75, q25 = np.percentile(x, [75, 25])
    scale = (q75 - q25) / 1.34896
    scale = max(scale, 1e-8)
    return (x - medians) / scale

def get_rms_vector(sigs_array):
    """Calcola il vettore RMS delle 6 lead limb."""
    return np.sqrt(np.mean(sigs_array**2, axis=1))

def calculate_domain_metrics(class_name='LA-RA', n_samples=30):
    print(f"\n--- Analisi Allineamento di Dominio: {class_name} ---")
    
    df = pd.read_csv(CSV_PATH)
    df_valido = df[df["Inversione"] != "?"].copy()
    df_valido["Inversione"] = df_valido["Inversione"].apply(lambda x: LABEL_MAP_CLEAN.get(x, x))
    
    real_subset = df_valido[df_valido["Inversione"] == class_name]
    norm_subset = df_valido[df_valido["Inversione"] == "normale"]
    
    n_real = min(len(real_subset), n_samples)
    n_norm = min(len(norm_subset), n_samples)
    
    real_vectors = []
    legacy_z_vectors = []
    proposed_vectors = []
    
    mode = MAPPING_INV[class_name]
    
    # 1. Carica Real ECGs
    print(f"Caricamento {n_real} ECG reali...")
    for ecg_id in tqdm(real_subset["Num"].iloc[:n_real]):
        data = get_ecg(ecg_id)
        if not data: continue
        sigs = all_leads_preprocessing(data["signals"])
        x = np.array([sigs[l] for l in LIMB_LEADS], dtype=np.float32)
        x_norm = robust_global_scale(x)
        real_vectors.append(get_rms_vector(x_norm))
        
    # 2. Genera Synthetic ECGs
    print(f"Generazione {n_norm} ECG sintetici...")
    for ecg_id in tqdm(norm_subset["Num"].iloc[:n_norm]):
        data = get_ecg(ecg_id)
        if not data: continue
        
        # PROPOSTA: Noise -> Swap -> Preproc -> Global Scale
        sigs_noise = add_realistic_noise(data["signals"])
        sigs_swap_p = limb_interchange_simulation(mode, sigs_noise)
        sigs_proc_p = all_leads_preprocessing(sigs_swap_p)
        x_p = np.array([sigs_proc_p[l] for l in LIMB_LEADS], dtype=np.float32)
        proposed_vectors.append(get_rms_vector(robust_global_scale(x_p)))
        
        # LEGACY + Z-MEDIAN: Preproc -> Swap -> Global Scale
        sigs_proc_l = all_leads_preprocessing(data["signals"])
        sigs_swap_l = limb_interchange_simulation(mode, sigs_proc_l)
        x_l = np.array([sigs_swap_l[l] for l in LIMB_LEADS], dtype=np.float32)
        legacy_z_vectors.append(get_rms_vector(robust_global_scale(x_l)))
        
    # Medie
    V_real = np.mean(real_vectors, axis=0)
    V_legacy_z = np.mean(legacy_z_vectors, axis=0)
    V_proposed = np.mean(proposed_vectors, axis=0)
    
    # Distanze Euclidee
    dist_legacy_z = np.linalg.norm(V_real - V_legacy_z)
    dist_proposed = np.linalg.norm(V_real - V_proposed)
    
    print(f"\nRISULTATI (Distanza dal Dominio Reale):")
    print(f"  - Pipeline Legacy + Z-Median: {dist_legacy_z:.4f}")
    print(f"  - Pipeline Proposta Full:    {dist_proposed:.4f}")
    
    diff = dist_legacy_z - dist_proposed
    if diff > 0:
        print(f"\nIl Rumore e l'Ordine aggiungono un miglioramento extra del {diff/dist_legacy_z*100:.1f}% rispetto al solo Z-Median.")
    else:
        print(f"\nLo Z-Median domina l'allineamento. Rumore/Ordine hanno impatto trascurabile sulla statistica RMS.")
    
    print(f"\nProfilo RMS Reale (Target):")
    for i, lead in enumerate(LIMB_LEADS):
        print(f"  {lead:<5}: {V_real[i]:.3f}")

if __name__ == "__main__":
    calculate_domain_metrics('LA-RA', n_samples=30)
    calculate_domain_metrics('ROT_ANTIORARIA', n_samples=20)
