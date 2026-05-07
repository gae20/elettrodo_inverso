import os
import sys
import json
import sqlite3
import h5py
import numpy as np
import copy
import zipfile
import io
from tqdm import tqdm

def train_test_split(data, test_size=0.25, random_state=42):
    np.random.seed(random_state)
    shuffled = np.random.permutation(data)
    split_idx = int(len(data) * (1 - test_size))
    return list(shuffled[:split_idx]), list(shuffled[split_idx:])

# Import locali
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.data_pipeline import (
    read_edf_data, all_leads_preprocessing, check_ecg_quality, 
    check_window_quality, precordial_interchange_simulation
)
from utils.config import (
    SAMPLES_PER_WINDOW, FS_NEW,
    ALL_LEADS, PRECORDIAL_MAPPING, 
    QUALITY_CFG
)

PRECORDIAL_INDICES = list(range(6, 12))

# Override stride: NO SOVRAPPOSIZIONE
STRIDE_SAMPLES = int(FS_NEW * 2.0)

# QUALITY_CFG_HOLTER è quella originale (più tollerante)
QUALITY_CFG_HOLTER = copy.deepcopy(QUALITY_CFG)

# QUALITY_CFG_STANDARD è più rigorosa per gli ECG da 10s
QUALITY_CFG_STANDARD = copy.deepcopy(QUALITY_CFG)
QUALITY_CFG_STANDARD["baseline_max_uv"] = 500.0
QUALITY_CFG_STANDARD["mad_noise_limb"] = 20.0
QUALITY_CFG_STANDARD["mad_noise_prec"] = 25.0
QUALITY_CFG_STANDARD["min_valid_ratio"] = 0.70

# Dizionario per memorizzare se un ID è Holter
IS_HOLTER_DICT = {}

def compute_good_window_mask_from_raw(sigs_array, cfg, min_valid_leads_per_window=5, lead_indices=None):
    fs = cfg["fs"]
    win_size = int(cfg["win_sec"] * fs)
    stride = int(cfg["stride_sec"] * fs)
    n_leads, n_samples = sigs_array.shape
    if n_samples < win_size:
        return np.zeros((0,), dtype=bool)

    indices = lead_indices if lead_indices is not None else list(range(n_leads))
    win_starts = list(range(0, n_samples - win_size + 1, stride))
    n_win = len(win_starts)
    mask_win = np.zeros(n_win, dtype=bool)

    for w_idx, start in enumerate(win_starts):
        lead_valid_flags = []
        for lead_idx in indices:
            seg = sigs_array[lead_idx, start:start + win_size]
            res = check_window_quality(seg, cfg=cfg, lead_idx=lead_idx)
            lead_valid_flags.append(res["valid"])
        lead_valid_flags = np.array(lead_valid_flags, dtype=bool)
        mask_win[w_idx] = (lead_valid_flags.sum() >= min_valid_leads_per_window)
    return mask_win

def zscore_ecg(sigs_array, eps=1e-8):
    x = sigs_array.astype(np.float32)
    means = x.mean(axis=1, keepdims=True)
    stds = x.std(axis=1, keepdims=True)
    stds = np.where(stds < eps, 1.0, stds)
    x_norm = (x - means) / stds
    return x_norm, means.squeeze(), stds.squeeze()

def create_windows(signals_dict, lead_order=ALL_LEADS, win_size=SAMPLES_PER_WINDOW, stride=STRIDE_SAMPLES):
    full_signal = np.array([signals_dict[l] for l in lead_order], dtype=np.float32)
    if full_signal.ndim != 2 or full_signal.shape[0] != 12:
        raise ValueError(f"Shape non valida: {full_signal.shape}")

    if full_signal.shape[1] < win_size:
        return np.empty((0, 12, win_size), dtype=np.float32)

    windows = []
    for start in range(0, full_signal.shape[1] - win_size + 1, stride):
        windows.append(full_signal[:, start:start + win_size])
    return np.array(windows, dtype=np.float32)

def _append_to_h5(dset_x, dset_y, windows, labels):
    n = windows.shape[0]
    curr = dset_x.shape[0]
    dset_x.resize(curr + n, axis=0)
    dset_y.resize(curr + n, axis=0)
    dset_x[curr:curr + n] = windows
    dset_y[curr:curr + n] = labels

def build_unlabelled_precordials_dataset(ids_list, h5_name, id_to_zip, max_windows_per_class=None):
    if os.path.exists(h5_name): os.remove(h5_name)
    h5_tmp = h5_name + ".tmp"
    if os.path.exists(h5_tmp): os.remove(h5_tmp)

    all_labels = ['normale'] + list(PRECORDIAL_MAPPING.keys())
    label_to_int = {lab: idx for idx, lab in enumerate(all_labels)}
    name = os.path.basename(h5_name)

    with h5py.File(h5_tmp, 'w') as f:
        dset_x = f.create_dataset('X', shape=(0, 12, SAMPLES_PER_WINDOW), maxshape=(None, 12, SAMPLES_PER_WINDOW), dtype='float32', chunks=(64, 12, SAMPLES_PER_WINDOW), compression='lzf')
        dset_y = f.create_dataset('Y', shape=(0,), maxshape=(None,), dtype='int8')

        skipped = 0
        class_counts = {lab: 0 for lab in all_labels}
        
        bar = tqdm(ids_list, desc=f"{name} | extraction", unit="ecg")
        for ecg_id in bar:
            if max_windows_per_class is not None:
                if all(class_counts[lab] >= max_windows_per_class for lab in all_labels):
                    break

            try:
                zip_path = id_to_zip.get(str(ecg_id))
                if not zip_path:
                    skipped += 1; continue
                
                with zipfile.ZipFile(zip_path, 'r') as z_in:
                    edf_bytes = z_in.read(f"{ecg_id}.edf")
                    ecg_data = read_edf_data(edf_bytes)

                if not ecg_data or not ecg_data["signals"]:
                    skipped += 1; continue

                sigs = all_leads_preprocessing(ecg_data["signals"])
                sigs_array = np.array([sigs[l] for l in ALL_LEADS], dtype=np.float32)

                # QUALITY CHECK DINAMICO SULLE PRECORDIALI
                is_holter = IS_HOLTER_DICT.get(ecg_id, False)
                cfg = QUALITY_CFG_HOLTER if is_holter else QUALITY_CFG_STANDARD
                cfg["stride_sec"] = 2.0 

                quality_result = check_ecg_quality(sigs_array, cfg=cfg, lead_indices=PRECORDIAL_INDICES)
                if not quality_result['global_valid']:
                    skipped += 1; continue

                win_mask = compute_good_window_mask_from_raw(sigs_array, cfg=cfg, min_valid_leads_per_window=5, lead_indices=PRECORDIAL_INDICES)
                if win_mask.size == 0 or not win_mask.any():
                    skipped += 1; continue

                # Z-SCORE sui segnali originali (Normale)
                sigs_norm, _, _ = zscore_ecg(sigs_array)
                sigs_norm_dict = {lead: sigs_norm[i] for i, lead in enumerate(ALL_LEADS)}
                wins_all = create_windows(sigs_norm_dict, stride=STRIDE_SAMPLES)
                
                n_win = min(wins_all.shape[0], win_mask.size)
                wins_r_good = wins_all[:n_win][win_mask[:n_win]]
                
                if wins_r_good.shape[0] == 0:
                    skipped += 1; continue

                n_to_add = wins_r_good.shape[0]
                if max_windows_per_class is not None:
                    remaining = max_windows_per_class - class_counts['normale']
                    n_to_add = min(n_to_add, remaining)

                if n_to_add > 0:
                    labels_r = np.full(n_to_add, label_to_int['normale'], dtype='int8')
                    _append_to_h5(dset_x, dset_y, wins_r_good[:n_to_add], labels_r)
                    class_counts['normale'] += n_to_add

                # SIMULAZIONE INVERSIONI PRECORDIALI
                for inv_name in PRECORDIAL_MAPPING.keys():
                    n_to_add_inv = wins_r_good.shape[0]
                    if max_windows_per_class is not None:
                        rem_inv = max_windows_per_class - class_counts[inv_name]
                        n_to_add_inv = min(n_to_add_inv, rem_inv)
                        
                    if n_to_add_inv > 0:
                        sim_sigs = precordial_interchange_simulation(PRECORDIAL_MAPPING[inv_name], sigs)
                        sim_sigs_array = np.array([sim_sigs[l] for l in ALL_LEADS], dtype=np.float32)
                        
                        sim_sigs_norm, _, _ = zscore_ecg(sim_sigs_array)
                        sim_sigs_norm_dict = {lead: sim_sigs_norm[i] for i, lead in enumerate(ALL_LEADS)}
                        
                        wins_s = create_windows(sim_sigs_norm_dict, stride=STRIDE_SAMPLES)
                        wins_s_good = wins_s[:n_win][win_mask[:n_win]]
                        
                        if wins_s_good.shape[0] > 0:
                            actual_add = min(n_to_add_inv, wins_s_good.shape[0])
                            labels_s = np.full(actual_add, label_to_int[inv_name], dtype='int8')
                            _append_to_h5(dset_x, dset_y, wins_s_good[:actual_add], labels_s)
                            class_counts[inv_name] += actual_add

                bar.set_postfix(windows=dset_x.shape[0], skip=skipped)
            except Exception as e:
                skipped += 1; continue

        total = dset_x.shape[0]
        if total > 0:
            with h5py.File(h5_name, 'w') as dst:
                dst_x = dst.create_dataset('X', shape=(total, 12, SAMPLES_PER_WINDOW), dtype='float32', chunks=(64, 12, SAMPLES_PER_WINDOW), compression='lzf')
                dst_y = dst.create_dataset('Y', shape=(total,), dtype='int8')
                shuffled_idx = np.random.permutation(total)
                for start in tqdm(range(0, total, 4096), desc=f"{name} | fase 3/3 shuffle", unit="batch"):
                    end = min(start + 4096, total)
                    idx = shuffled_idx[start:end]
                    x_blk = dset_x[np.sort(idx)]
                    y_blk = dset_y[np.sort(idx)]
                    inv_sort = np.argsort(np.argsort(idx))
                    dst_x[start:end] = x_blk[inv_sort]
                    dst_y[start:end] = y_blk[inv_sort]

    if os.path.exists(h5_tmp): os.remove(h5_tmp)
    print(f"  [{name}] Completato: {total:,} finestre, {skipped} ECG scartati")
    for lab, cnt in class_counts.items():
        print(f"    - {lab}: {cnt} finestre")

def get_clean_ecg_ids(db_path, max_ecgs=None):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, report, text FROM records WHERE status='reported'")
    rows = cursor.fetchall()
    conn.close()

    text_bad_keywords = ['inversion', 'scambio', 'errato', 'periferic', 'elettrod', 'sbagliat', 'artefatt', 'posizionament', 'braccia']
    rejection_codes = {'BTWG01', 'BTWG02', 'BTWG03', 'BTWG04', 'BTWG05', 'BTWC1109', 'BTWC1110'}
    
    clean_ids = []

    for r in rows:
        id_ = r[0]
        report_str = r[1]
        text_str = (r[2] or "").lower()
        
        if any(kw in text_str for kw in text_bad_keywords):
            continue
            
        try:
            data = json.loads(report_str)
            codified = data.get('codified', [])
            codes = [c['value'] for c in codified if c.get('type') == 'code']
            
            if any(c in rejection_codes for c in codes):
                continue
                
            is_holter = 'BTWSCQQ43' in codes
            IS_HOLTER_DICT[id_] = is_holter
            
            clean_ids.append(id_)
        except Exception:
            continue

    np.random.seed(42)
    np.random.shuffle(clean_ids)
    if max_ecgs and len(clean_ids) > max_ecgs:
        clean_ids = clean_ids[:max_ecgs]
    return clean_ids

def build_zip_index(dataset_dir):
    id_to_zip = {}
    print("Indicizzazione degli EDF nei file ZIP...")
    zips = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.zip')]
    for zip_path in tqdm(zips, desc="Lettura ZIP"):
        with zipfile.ZipFile(zip_path, 'r') as z:
            for edf_name in z.namelist():
                if edf_name.endswith('.edf'):
                    ecg_id = edf_name.replace('.edf', '')
                    id_to_zip[ecg_id] = zip_path
    print(f"Indicizzati {len(id_to_zip)} ECG nei file ZIP.")
    return id_to_zip

if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'dataset', 'records.db')
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'dataset', 'DATASET')
    
    id_to_zip = build_zip_index(dataset_dir)
    
    print("Estrazione ID puliti (no inversioni note, no rejected)...")
    all_clean_ids_db = get_clean_ecg_ids(db_path, max_ecgs=20000)
    all_clean_ids = [cid for cid in all_clean_ids_db if str(cid) in id_to_zip]
    print(f"Trovati {len(all_clean_ids)} ECG utilizzabili e presenti nei file ZIP.")

    train_ids, vt_ids = train_test_split(all_clean_ids, test_size=0.20, random_state=42)
    val_ids, test_ids = train_test_split(vt_ids, test_size=0.50, random_state=42)

    print(f"Dataset Split:")
    print(f"  Train: {len(train_ids)} ECG")
    print(f"  Val:   {len(val_ids)} ECG")
    print(f"  Test:  {len(test_ids)} ECG")

    os.makedirs(out_dir, exist_ok=True)
    
    print("\n=== PRECORDIALS UNLABELLED ===")
    build_unlabelled_precordials_dataset(test_ids,  os.path.join(out_dir, "unlabelled_precordials_test.h5"),  id_to_zip, max_windows_per_class=None)
    build_unlabelled_precordials_dataset(val_ids,   os.path.join(out_dir, "unlabelled_precordials_val.h5"),   id_to_zip, max_windows_per_class=None)
    build_unlabelled_precordials_dataset(train_ids, os.path.join(out_dir, "unlabelled_precordials_train.h5"), id_to_zip, max_windows_per_class=None)
    print("\nFatto.")
