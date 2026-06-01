import os
import sys
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyedflib
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(r"c:\Users\cancr\OneDrive\Desktop\tesi\elettrodo_inverso\src")

from data.data_pipeline import (
    _parse_edf_file, leads_preprocessing, check_ecg_quality,
    check_window_quality, limb_interchange_simulation
)
from utils.config import (
    SAMPLES_PER_WINDOW, STRIDE_SAMPLES,
    LIMB_LEADS, ALL_LEADS, robust_scale_ecg, LABEL_MAP_CLEAN
)
from utils.sqa_real_config import QUALITY_CFG_REAL
from models.ldensenet import build_model

# Constants from testset_validation.py
LOCAL_CSV_PATH = r"c:\Users\cancr\OneDrive\Desktop\tesi\datasets\dataset_reals\thesis-sample.csv"
REAL_EDF_DIR   = r"c:\Users\cancr\OneDrive\Desktop\tesi\datasets\dataset_reals"
CLEAN_H5_PATH  = r"c:\Users\cancr\OneDrive\Desktop\tesi\datasets\labelled_z_median_limbs_test_validation_clean.h5"
WEIGHTS_PATH   = r"c:\Users\cancr\OneDrive\Desktop\tesi\elettrodo_inverso\src\prove\models\best_model_final_noise_limbs.weights.h5"

def create_windows(signals_dict, lead_order, win_size=SAMPLES_PER_WINDOW, stride=STRIDE_SAMPLES):
    full_signal = np.array([signals_dict[l] for l in lead_order], dtype=np.float32)
    n_leads = len(lead_order)
    if full_signal.ndim != 2 or full_signal.shape[0] != n_leads:
        return np.empty((0, n_leads, win_size), dtype=np.float32)
    if full_signal.shape[1] < win_size:
        return np.empty((0, n_leads, win_size), dtype=np.float32)
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

def get_edf_metadata(edf_id):
    path = os.path.join(REAL_EDF_DIR, f"record{edf_id}.edf")
    if not os.path.exists(path):
        return None
    try:
        f = pyedflib.EdfReader(path)
        bdate = f.getBirthdate()
        sex = f.getSex()
        f.close()
        return bdate, sex
    except Exception:
        return None

def load_signals(edf_id):
    path = os.path.join(REAL_EDF_DIR, f"record{edf_id}.edf")
    ecg_data = _parse_edf_file(path)
    if not ecg_data or "signals" not in ecg_data:
        return None
    signals = ecg_data["signals"]
    sigs_dict = {}
    for l in LIMB_LEADS:
        sigs_dict[l] = leads_preprocessing(signals[l])
    return sigs_dict

def main():
    print("=== rebuilding clinically and physically verified real test set ===")
    
    # 1. Carica gli ID di test reali coerenti con testset_validation.py
    from prove.testset_validation import load_real_test_ids
    test_ids_per_class = load_real_test_ids()
    if not test_ids_per_class:
        print("Errore: impossibile caricare gli ID di test.")
        return
        
    solved_csv_path = r"c:\Users\cancr\OneDrive\Desktop\tesi\datasets\dataset_reals\thesis-sample-solved.csv"
    df_solved = pd.read_csv(solved_csv_path).set_index("Num")
    
    class_name_map = {
        'normale': 0, 'LA-RA': 1, 'RA-LL': 2,
        'LA-LL': 3, 'ROT_ORARIA': 4, 'ROT_ANTIORARIA': 5
    }
    
    # 2. Processa e carica i record reali con le etichette fisiche corrette
    class_windows = {i: [] for i in range(6)}
    correction_count = 0
    direct_count = 0
    
    for original_class in range(6):
        ids = test_ids_per_class[original_class]
        desc = f"Processamento Classe {original_class}"
        for ecg_id in tqdm(ids, desc=desc):
            try:
                # Carica i segnali periferici
                sigs_dict = load_signals(ecg_id)
                if not sigs_dict:
                    continue
                    
                sigs_array = np.array([sigs_dict[l] for l in LIMB_LEADS], dtype=np.float32)
                
                # Controllo Qualità
                qual = check_ecg_quality(sigs_array, cfg=QUALITY_CFG_REAL, lead_indices=list(range(6)))
                if not qual['global_valid']:
                    continue
                    
                win_mask = compute_good_window_mask_from_raw(sigs_array, cfg=QUALITY_CFG_REAL, min_valid_leads_per_window=5, lead_indices=list(range(6)))
                if win_mask.size == 0 or not win_mask.any():
                    continue
                    
                # Ottiene l'etichetta già risolta fisicamente
                assigned_class_name = df_solved.loc[ecg_id, "Solved_Label"]
                assigned_class = class_name_map[assigned_class_name]
                
                if df_solved.loc[ecg_id, "Method"] == "Physical (Einthoven)":
                    if assigned_class != original_class:
                        correction_count += 1
                else:
                    direct_count += 1
                    
                # Normalizzazione ROBUSTA
                sigs_norm = robust_scale_ecg(sigs_array)
                sigs_norm_dict = {LIMB_LEADS[i]: sigs_norm[i] for i in range(6)}
                
                # Finestre
                wins = create_windows(sigs_norm_dict, lead_order=LIMB_LEADS)
                n_win_final = min(wins.shape[0], win_mask.size)
                wins_good = wins[:n_win_final][win_mask[:n_win_final]]
                
                if len(wins_good) > 0:
                    class_windows[assigned_class].extend(list(wins_good))
                    
            except Exception as e:
                print(f"  [ERRORE] Record {ecg_id}: {e}")
                
    print(f"\nStatistiche accoppiamento e correzione:")
    print(f"  Record corretti fisicamente:  {correction_count}")
    print(f"  Record diretti (no normal):   {direct_count}")
    
    # 4. Bilanciamento delle classi
    counts = {i: len(class_windows[i]) for i in range(6)}
    print(f"\nConteggi finestre per classe fisica reale: {counts}")
    
    available_anomalies = [counts[i] for i in range(1, 6) if counts[i] > 0]
    if not available_anomalies:
        print("Errore: nessuna anomalia reale disponibile!")
        return
        
    n_per_anomaly = min(available_anomalies)
    max_per_anomaly = counts[0] // 5
    n_per_anomaly = min(n_per_anomaly, max_per_anomaly)
    
    print(f"Bilanciamento test set a {n_per_anomaly} finestre per classe di anomalia.")
    
    for i in range(1, 6):
        np.random.seed(42)
        np.random.shuffle(class_windows[i])
        class_windows[i] = class_windows[i][:n_per_anomaly]
        
    target_norm = 5 * n_per_anomaly
    np.random.seed(42)
    np.random.shuffle(class_windows[0])
    class_windows[0] = class_windows[0][:target_norm]
    
    # Assemblaggio
    x_final = []
    y_final = []
    for i in range(6):
        x_final.extend(class_windows[i])
        y_final.extend([i] * len(class_windows[i]))
        
    x_final = np.array(x_final, dtype=np.float32)
    y_final = np.array(y_final, dtype=np.int8)
    
    print(f"Dataset finale H5: X shape = {x_final.shape}, Y shape = {y_final.shape}")
    
    # Salva su file H5
    if os.path.exists(CLEAN_H5_PATH):
        os.remove(CLEAN_H5_PATH)
    with h5py.File(CLEAN_H5_PATH, 'w') as f:
        f.create_dataset('X', data=x_final, compression='lzf')
        f.create_dataset('Y', data=y_final)
    print(f"Dataset salvato con successo in: {CLEAN_H5_PATH}")
    
    # 5. Valutazione finale del modello
    print("\n=== VALUTAZIONE DEL MODELLO SUL TEST SET REALE PULITO ===")
    x_t = np.transpose(x_final, (0, 2, 1))
    
    model = build_model((SAMPLES_PER_WINDOW, 6), 6)
    model.load_weights(WEIGHTS_PATH)
    
    probs = model.predict(x_t, batch_size=64, verbose=0)
    preds = np.argmax(probs, axis=1)
    
    print("\n=== CLASSIFICATION REPORT FOR CLEAN CLINICAL TEST SET ===")
    class_names = ['normale (0)', 'LA-RA (1)', 'RA-LL (2)', 'LA-LL (3)', 'ROT_ORARIA (4)', 'ROT_ANTIORARIA (5)']
    print(classification_report(y_final, preds, target_names=class_names, digits=4))
    
    print("\n=== CONFUSION MATRIX ===")
    print("Columns: Predicted, Rows: True")
    print(f"      {' '.join([f'P{i}' for i in range(6)])}")
    cm = confusion_matrix(y_final, preds)
    for i in range(6):
        print(f"T{i:2d}:  " + "  ".join([f"{val:3d}" for val in cm[i]]))

if __name__ == "__main__":
    main()
