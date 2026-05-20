import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Aggiungi src al path per importare i moduli interni
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data.data_pipeline import get_ecg, all_leads_preprocessing, check_ecg_quality, check_window_quality
from models.ldensenet import build_model
from utils.config import SAMPLES_PER_WINDOW, STRIDE_SAMPLES, ALL_LEADS, QUALITY_CFG
from validation.testset_validation import load_real_test_ids

LIMB_INDICES = list(range(6))

def create_windows(signals_dict, lead_order=ALL_LEADS, win_size=SAMPLES_PER_WINDOW, stride=STRIDE_SAMPLES):
    full_signal = np.array([signals_dict[l] for l in lead_order], dtype=np.float32)
    if full_signal.ndim != 2 or full_signal.shape[0] != len(lead_order):
        return np.empty((0, len(lead_order), win_size), dtype=np.float32)
    if full_signal.shape[1] < win_size:
        return np.empty((0, len(lead_order), win_size), dtype=np.float32)
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

def zscore_independent(sigs_array, eps=1e-8):
    """Normalizzazione attuale (sbagliata per le ampiezze relative)"""
    x = sigs_array.astype(np.float32)
    means = x.mean(axis=1, keepdims=True)
    stds = x.std(axis=1, keepdims=True)
    stds = np.where(stds < eps, 1.0, stds)
    x_norm = (x - means) / stds
    return x_norm

def zscore_global(sigs_array, eps=1e-8):
    """Normalizzazione globale (mantiene le ampiezze relative)"""
    x = sigs_array.astype(np.float32)
    # Calcoliamo media e std su tutti i canali contemporaneamente
    mean_global = x.mean()
    std_global = x.std()
    std_global = 1.0 if std_global < eps else std_global
    x_norm = (x - mean_global) / std_global
    return x_norm

def test_normalization_impact():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, "..", "best_model_unlabelled_limbs.weights.h5")
    
    # Carica ID reali
    ids_per_class = load_real_test_ids()
    if not ids_per_class: return
    normal_ids = ids_per_class[0]
    
    print(f"Estratti {len(normal_ids)} ECG normali per il test.")
    
    # Carica modello
    model = build_model((SAMPLES_PER_WINDOW, 6), 6)
    model.load_weights(weights_path)
    
    fp_indep = 0
    fp_global = 0
    total_windows = 0
    
    # Prepara finestre per entrambi i metodi
    windows_indep_list = []
    windows_global_list = []
    
    for ecg_id in tqdm(normal_ids, desc="Processing Normals"):
        try:
            ecg_data = get_ecg(ecg_id)
            if not ecg_data or not ecg_data["signals"]: continue
            
            sigs = all_leads_preprocessing(ecg_data["signals"])
            sigs_array = np.array([sigs[l] for l in ALL_LEADS[:6]], dtype=np.float32)
            
            # Filtro qualità per prendere solo finestre "buone"
            win_mask = compute_good_window_mask_from_raw(sigs_array, cfg=QUALITY_CFG, min_valid_leads_per_window=5, lead_indices=LIMB_INDICES)
            if win_mask.size == 0 or not win_mask.any(): continue
            
            # Metodo 1: Independent
            sigs_indep = zscore_independent(sigs_array)
            dict_indep = {ALL_LEADS[i]: sigs_indep[i] for i in range(6)}
            wins_indep = create_windows(dict_indep, lead_order=ALL_LEADS[:6])
            
            # Metodo 2: Global
            sigs_global = zscore_global(sigs_array)
            dict_global = {ALL_LEADS[i]: sigs_global[i] for i in range(6)}
            wins_global = create_windows(dict_global, lead_order=ALL_LEADS[:6])
            
            n_win_final = min(wins_indep.shape[0], win_mask.size)
            windows_indep_list.append(wins_indep[:n_win_final][win_mask[:n_win_final]])
            windows_global_list.append(wins_global[:n_win_final][win_mask[:n_win_final]])
            
        except Exception as e:
            continue
            
    if not windows_indep_list: return
    
    x_indep = np.concatenate(windows_indep_list, axis=0)
    x_global = np.concatenate(windows_global_list, axis=0)
    
    # Trasponi per il modello (Samples, SAMPLES_PER_WINDOW, Channels)
    x_indep = np.transpose(x_indep, (0, 2, 1))
    x_global = np.transpose(x_global, (0, 2, 1))
    
    total_windows = x_indep.shape[0]
    
    # Inferenza
    preds_indep = np.argmax(model.predict(x_indep, batch_size=32, verbose=0), axis=1)
    preds_global = np.argmax(model.predict(x_global, batch_size=32, verbose=0), axis=1)
    
    fp_indep = np.sum(preds_indep != 0)
    fp_global = np.sum(preds_global != 0)
    
    print("\n" + "="*50)
    print("RISULTATI TEST NORMALIZZAZIONE SU SEGNI NORMALI")
    print("="*50)
    print(f"Finestre Totali Testate: {total_windows}")
    print(f"\n1. Metodo Attuale (Independent Z-Score):")
    print(f"   Falsi Positivi: {fp_indep} ({(fp_indep/total_windows)*100:.2f}%)")
    
    print(f"\n2. Metodo Proposto (Global Z-Score):")
    print(f"   Falsi Positivi: {fp_global} ({(fp_global/total_windows)*100:.2f}%)")
    print("="*50)
    
    if fp_global < fp_indep:
        print("\nCONCLUSIONE: La normalizzazione globale RIDUCE i falsi positivi.")
        print("ATTENZIONE: Essendo il modello addestrato con l'Independent Z-score, potrebbe aver perso accuratezza sulle vere anomalie.")
    else:
        print("\nCONCLUSIONE: Alimentare il modello attuale con dati Global Z-score lo disorienta.")
        print("Ciò è normale: il modello si aspetta input con std=1.0 per ogni canale. Serve riaddestrarlo.")

if __name__ == "__main__":
    test_normalization_impact()
