import os
import sys
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Aggiungi src al path per importare i moduli interni
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from data.data_pipeline import (
    get_ecg, all_leads_preprocessing, check_ecg_quality, 
    check_window_quality, limb_interchange_simulation
)
from utils.config import (
    SAMPLES_PER_WINDOW, STRIDE_SAMPLES,
    ALL_LEADS, MAPPING_INV, ACTIVE_SYNTH_CLASSES, 
    QUALITY_CFG, LABEL_MAP_CLEAN
)

LIMB_INDICES = list(range(6))
LOCAL_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'datasets', 'dataset', 'thesis-sample.csv')

def robust_scale_ecg(sigs_array, eps=1e-8):
    x = sigs_array.astype(np.float32)
    medians = np.median(x, axis=1, keepdims=True)
    q75, q25 = np.percentile(x, [75, 25])
    iqr_global = q75 - q25
    scale_global = iqr_global / 1.34896
    scale_global = max(scale_global, eps)
    x_norm = (x - medians) / scale_global
    return x_norm

def create_windows(signals_dict, lead_order=ALL_LEADS, win_size=SAMPLES_PER_WINDOW, stride=STRIDE_SAMPLES):
    full_signal = np.array([signals_dict[l] for l in lead_order], dtype=np.float32)
    if full_signal.ndim != 2 or full_signal.shape[0] != 12:
        return np.empty((0, 12, win_size), dtype=np.float32)
    if full_signal.shape[1] < win_size:
        return np.empty((0, 12, win_size), dtype=np.float32)
    windows = []
    for start in range(0, full_signal.shape[1] - win_size + 1, stride):
        windows.append(full_signal[:, start:start + win_size])
    return np.array(windows, dtype=np.float32)

def compute_good_window_mask_from_raw(sigs_array, cfg, min_valid_leads_per_window=5, lead_indices=None):
    fs = cfg["fs"]
    win_size = int(cfg["win_sec"] * fs)
    stride = int(cfg["stride_sec"] * fs)
    n_leads, n_samples = sigs_array.shape
    if n_samples < win_size:
        return np.zeros((0,), dtype=bool)
    indices = lead_indices if lead_indices is not None else list(range(n_leads))
    win_starts = list(range(0, n_samples - win_size + 1, stride))
    mask_win = np.zeros(len(win_starts), dtype=bool)
    for w_idx, start in enumerate(win_starts):
        lead_valid_flags = []
        for lead_idx in indices:
            seg = sigs_array[lead_idx, start:start + win_size]
            res = check_window_quality(seg, cfg=cfg, lead_idx=lead_idx)
            lead_valid_flags.append(res["valid"])
        mask_win[w_idx] = (sum(lead_valid_flags) >= min_valid_leads_per_window)
    return mask_win

def build_synthetic_benchmark():
    if not os.path.exists(LOCAL_CSV_PATH):
        print(f"Errore: CSV non trovato in {LOCAL_CSV_PATH}")
        return

    df = pd.read_csv(LOCAL_CSV_PATH)
    df_valido = df[df["Inversione"] == "normale"].copy()
    normal_ids = df_valido["Num"].unique()
    
    # Usiamo lo stesso split di testset_validation.py per coerenza
    _, vt_norm_ids = train_test_split(normal_ids, test_size=0.20, random_state=42)
    _, test_norm_ids = train_test_split(vt_norm_ids, test_size=0.50, random_state=42)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    h5_name = os.path.join(base_dir, "limbs_synthetic_pure.h5")
    
    class_windows = {i: [] for i in range(6)}
    class_name_map = {lab: idx + 1 for idx, lab in enumerate(ACTIVE_SYNTH_CLASSES)}
    class_name_map['normale'] = 0

    print(f"Generazione benchmark 100% SINTETICO da {len(test_norm_ids)} ECG normali...")
    
    for ecg_id in tqdm(test_norm_ids, desc="Processing Normals"):
        try:
            ecg_data = get_ecg(ecg_id)
            if not ecg_data or not ecg_data["signals"]: continue
            
            # Preprocessing originale (senza inversioni)
            sigs_orig = all_leads_preprocessing(ecg_data["signals"])
            
            # Per ogni possibile inversione (comprese 'normale')
            for label_name, class_idx in class_name_map.items():
                if label_name == 'normale':
                    sigs_curr = sigs_orig
                else:
                    # Simula inversione
                    sigs_curr = limb_interchange_simulation(MAPPING_INV[label_name], sigs_orig)
                
                sigs_array = np.array([sigs_curr[l] for l in ALL_LEADS], dtype=np.float32)
                
                # SQI (Fondamentale: usiamo gli stessi criteri del reale)
                quality_result = check_ecg_quality(sigs_array, cfg=QUALITY_CFG, lead_indices=LIMB_INDICES)
                if not quality_result['global_valid']: continue
                
                win_mask = compute_good_window_mask_from_raw(sigs_array, cfg=QUALITY_CFG, min_valid_leads_per_window=5, lead_indices=LIMB_INDICES)
                if win_mask.size == 0 or not win_mask.any(): continue
                
                # Normalization
                sigs_norm = robust_scale_ecg(sigs_array)
                sigs_norm_dict = {lead: sigs_norm[i] for i, lead in enumerate(ALL_LEADS)}
                
                # Windows
                wins = create_windows(sigs_norm_dict)
                n_win_final = min(wins.shape[0], win_mask.size)
                wins_good = wins[:n_win_final][win_mask[:n_win_final]]
                
                if wins_good.shape[0] > 0:
                    class_windows[class_idx].extend(list(wins_good))
                    
        except Exception as e:
            continue

    # Bilanciamento (facoltativo per analisi statistica, ma utile per parità di campionamento)
    counts = {i: len(class_windows[i]) for i in range(6)}
    print(f"\nConteggi sintetici estratti: {counts}")
    
    # Salvataggio
    x_final = []
    y_final = []
    for i in range(6):
        if len(class_windows[i]) > 0:
            # Opzionale: cap a 500 finestre per classe per non avere dataset enormi
            windows = class_windows[i]
            if len(windows) > 500:
                np.random.shuffle(windows)
                windows = windows[:500]
            
            x_final.extend(windows)
            y_final.extend([i] * len(windows))
    
    x_final = np.array(x_final, dtype='float32')
    y_final = np.array(y_final, dtype='int8')
    
    if os.path.exists(h5_name): os.remove(h5_name)
    with h5py.File(h5_name, 'w') as f:
        f.create_dataset('X', data=x_final, compression='lzf')
        f.create_dataset('Y', data=y_final)
    
    print(f"\nDataset SINTETICO creato: {h5_name} ({len(x_final)} finestre totali)")

if __name__ == "__main__":
    build_synthetic_benchmark()
