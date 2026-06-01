import os
import sys
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Aggiungi src al path per importare i moduli interni
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.data_pipeline import (
    _parse_edf_file, all_leads_preprocessing, check_ecg_quality, 
    check_window_quality
)
from utils.config import (
    SAMPLES_PER_WINDOW, STRIDE_SAMPLES,
    ALL_LEADS, MAPPING_INV, ACTIVE_SYNTH_CLASSES, 
    QUALITY_CFG, LABEL_MAP_CLEAN
)

LIMB_INDICES = list(range(6))
# Basato sullo screenshot:
LOCAL_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'dataset', 'dataset_small', 'thesis-sample-corrected.csv')
REAL_EDF_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'dataset', 'dataset_small')

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

def load_real_test_ids():
    if not os.path.exists(LOCAL_CSV_PATH):
        print(f"Errore: CSV non trovato in {LOCAL_CSV_PATH}")
        return None
    df = pd.read_csv(LOCAL_CSV_PATH)
    df_cand = df[df["Inversione"] != "?"].copy()
    df_cand["Inversione"] = df_cand["Inversione"].apply(lambda x: LABEL_MAP_CLEAN.get(x, x))
    df_valido = df_cand.set_index("Num")
    
    # Label mapping invertito per classi progetto (1-5)
    class_name_map = {
        'normale': 0,
        'LA-RA': 1,
        'RA-LL': 2,
        'LA-LL': 3,
        'ROT_ORARIA': 4,
        'ROT_ANTIORARIA': 5
    }
    
    normal_ids = df_valido[df_valido['Inversione'] == 'normale'].index.unique()
    anomaly_ids = df_valido[df_valido['Inversione'] != 'normale'].index.unique()
    
    # Split coerente con build_dataset.py
    _, vt_norm_ids = train_test_split(normal_ids, test_size=0.20, random_state=42)
    _, test_norm_ids = train_test_split(vt_norm_ids, test_size=0.50, random_state=42)
    _, test_anom_ids = train_test_split(anomaly_ids, test_size=0.50, random_state=42)
    
    test_ids_per_class = {i: [] for i in range(6)}
    
    for idx in test_norm_ids:
        test_ids_per_class[0].append(idx)
    
    for idx in test_anom_ids:
        label = df_valido.loc[idx, 'Inversione']
        if label in class_name_map:
            c = class_name_map[label]
            if c <= 5:
                test_ids_per_class[c].append(idx)
                
    return test_ids_per_class

def build_testset_validation_real():
    ids_per_class = load_real_test_ids()
    if ids_per_class is None: return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    h5_name = os.path.join(base_dir, "../../datasets/labelled_z_median_limbs_test_validation_corrected.h5")
    
    class_windows = {i: [] for i in range(6)}
    
    print(f"Creazione test set validation usando SOLO etichette reali...")
    
    for c in range(6):
        ids = ids_per_class[c]
        desc = f"Classe {c}"
        for ecg_id in tqdm(ids, desc=desc):
            try:
                # Lettura DIRETTA dallo screenshot: recordXXXX.edf
                edf_path = os.path.join(REAL_EDF_DIR, f"record{ecg_id}.edf")
                if not os.path.exists(edf_path):
                    print(f"  [SALTO] Record {ecg_id}: File non trovato in {edf_path}")
                    continue
                
                ecg_data = _parse_edf_file(edf_path)
                if ecg_data is None or "signals" not in ecg_data:
                    continue
                
                # Preprocessing manuale dei soli Limbs
                from data.data_pipeline import leads_preprocessing
                signals = ecg_data["signals"]
                
                LIMB_LEADS = ALL_LEADS[:6]
                sigs_limbs_dict = {}
                try:
                    for l in LIMB_LEADS:
                        # Preprocessiamo ogni derivazione periferica singolarmente
                        sigs_limbs_dict[l] = leads_preprocessing(signals[l])
                except KeyError as e:
                    print(f"  [SALTO] Record {ecg_id}: Manca derivazione periferica {e}")
                    continue
                    
                sigs_array_limbs = np.array([sigs_limbs_dict[l] for l in LIMB_LEADS], dtype=np.float32)
                
                # SQI (Qualità) calcolata sui 6 canali periferici
                quality_result = check_ecg_quality(sigs_array_limbs, cfg=QUALITY_CFG, lead_indices=list(range(6)))
                if not quality_result['global_valid']:
                    print(f"  [SALTO] Record {ecg_id}: Qualità globale insufficiente.")
                    continue
                
                win_mask = compute_good_window_mask_from_raw(sigs_array_limbs, cfg=QUALITY_CFG, min_valid_leads_per_window=5, lead_indices=list(range(6)))
                if win_mask.size == 0 or not win_mask.any():
                    print(f"  [SALTO] Record {ecg_id}: Nessuna finestra buona.")
                    continue
                
                # Normalizzazione ROBUSTA (Sincronizzata con Training: calcolata SOLO sui Limbs)
                sigs_norm_limbs = robust_scale_ecg(sigs_array_limbs)
                
                # Ricostruiamo il dizionario per create_windows (il modello vuole 6 canali)
                sigs_norm_dict = {LIMB_LEADS[i]: sigs_norm_limbs[i] for i in range(6)}
                
                # Windows (Specifichiamo LIMB_LEADS perché abbiamo solo 6 canali)
                wins = create_windows(sigs_norm_dict, lead_order=LIMB_LEADS)
                n_win_final = min(wins.shape[0], win_mask.size)
                wins_good = wins[:n_win_final][win_mask[:n_win_final]]
                
                if len(wins_good) == 0: continue
                
                # Aggiungiamo le finestre
                class_windows[c].extend(list(wins_good))
                
            except Exception as e:
                print(f"  [ERRORE] Record {ecg_id}: {e}")
                continue

    # Bilanciamento
    counts = {i: len(class_windows[i]) for i in range(6)}
    print(f"\nConteggi reali estratti: {counts}")
    
    min_anomalie = min(counts[i] for i in range(1, 6))
    if min_anomalie == 0:
        print("Attenzione: Almeno una classe reale ha 0 finestre nel set di test!")
        # Troviamo il minimo tra quelle non zero? No, il bilanciamento richiede tutte.
        # Ma se Class 4 ha 0 finestre nel TEST set, non possiamo averla.
    
    # Prendiamo il minimo tra le anomalie (escludendo eventuali 0 se proprio necessario, 
    # ma seguiamo la richiesta del bilanciamento stretto)
    n_per_anomaly = min_anomalie
    
    # Se vogliamo essere flessibili con classi mancanti nel test set:
    available_anomalies = [counts[i] for i in range(1, 6) if counts[i] > 0]
    if available_anomalies:
        n_per_anomaly = min(available_anomalies)
    
    max_per_anomaly = counts[0] // 5
    n_per_anomaly = min(n_per_anomaly, max_per_anomaly)
    
    print(f"Bilanciamento a {n_per_anomaly} finestre per classe di anomalia.")
    
    # Bilancia classi 1-5
    for i in range(1, 6):
        np.random.shuffle(class_windows[i])
        class_windows[i] = class_windows[i][:n_per_anomaly]
    
    # Classe 0 deve essere il 50% (5 * n_per_anomaly)
    target_norm = 5 * n_per_anomaly
    np.random.shuffle(class_windows[0])
    class_windows[0] = class_windows[0][:target_norm]
    
    # Assemblaggio
    x_final = []
    y_final = []
    for i in range(6):
        if len(class_windows[i]) > 0:
            x_final.extend(class_windows[i])
            y_final.extend([i] * len(class_windows[i]))
    
    if not x_final:
        print("Errore: Nessuna finestra valida estratta!")
        return

    x_final = np.array(x_final, dtype='float32')
    y_final = np.array(y_final, dtype='int8')
    
    # Shuffle
    idx = np.arange(len(x_final))
    np.random.shuffle(idx)
    x_final = x_final[idx]
    y_final = y_final[idx]
    
    # Salvataggio
    if os.path.exists(h5_name): os.remove(h5_name)
    with h5py.File(h5_name, 'w') as f:
        f.create_dataset('X', data=x_final, compression='lzf')
        f.create_dataset('Y', data=y_final)
    
    print(f"\nDataset creato con etichette REALI: {h5_name}")
    final_counts = {i: np.sum(y_final == i) for i in range(6)}
    print(f"Conteggi finali: {final_counts}")

if __name__ == "__main__":
    build_testset_validation_real()
