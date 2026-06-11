import os
import sys
import json
import sqlite3
import h5py
import numpy as np
import copy
import zipfile
from tqdm import tqdm

# Import locali
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.data_pipeline import (
    read_edf_data, all_leads_preprocessing, check_ecg_quality, 
    check_window_quality, precordial_interchange_simulation, apply_electrode_gain,
    add_extra_noise
)
from utils.config import (
    SAMPLES_PER_WINDOW, FS_NEW, FS_OLD,
    ALL_LEADS, PRECORDIAL_MAPPING, ACTIVE_SYNTH_CLASSES_PREC, 
    QUALITY_CFG
)

# MODIFICA 1: Indici per le precordiali (V1-V6)
PRECORDIAL_INDICES = list(range(6, 12))
STRIDE_SAMPLES = int(FS_NEW * 2.0)

# Quality configs
QUALITY_CFG_HOLTER = copy.deepcopy(QUALITY_CFG)
QUALITY_CFG_STANDARD = copy.deepcopy(QUALITY_CFG)
QUALITY_CFG_STANDARD["baseline_max_uv"] = 500.0
QUALITY_CFG_STANDARD["mad_noise_limb"] = 15.0
QUALITY_CFG_STANDARD["mad_noise_prec"] = 20.0
QUALITY_CFG_STANDARD["min_valid_ratio"] = 0.70

IS_HOLTER_DICT = {}

def train_test_split(data, test_size=0.25, random_state=42):
    np.random.seed(random_state)
    shuffled = np.random.permutation(data)
    split_idx = int(len(data) * (1 - test_size))
    return list(shuffled[:split_idx]), list(shuffled[split_idx:])

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

def robust_scale_ecg(sigs_array, eps=1e-8, reference_leads=None):
    x = sigs_array.astype(np.float32)
    medians = np.median(x, axis=1, keepdims=True)
    ref = x[reference_leads, :] if reference_leads is not None else x
    q75, q25 = np.percentile(ref, [75, 25])
    iqr_global = q75 - q25
    scale_global = iqr_global / 1.34896
    scale_global = max(scale_global, eps)
    x_norm = (x - medians) / scale_global
    return x_norm, medians.squeeze(), scale_global

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

def _append_to_h5(dset_x, dset_y, dset_p, windows, labels, patient_id):
    n = windows.shape[0]
    curr = dset_x.shape[0]
    dset_x.resize(curr + n, axis=0)
    dset_y.resize(curr + n, axis=0)
    dset_p.resize(curr + n, axis=0)
    dset_x[curr:curr + n] = windows
    dset_y[curr:curr + n] = labels
    dset_p[curr:curr + n] = np.full(n, patient_id, dtype='int64')

def build_unlabelled_precordials_dataset_with_patients(ids_list, h5_name, id_to_zip, add_gain=True):
    if os.path.exists(h5_name):
        print(f"File {h5_name} esiste già. Verrà sovrascritto.")
        os.remove(h5_name)
    h5_tmp = h5_name + ".tmp"
    if os.path.exists(h5_tmp):
        os.remove(h5_tmp)

    all_labels = ['normale'] + list(ACTIVE_SYNTH_CLASSES_PREC)
    label_to_int = {lab: idx for idx, lab in enumerate(['normale'] + list(PRECORDIAL_MAPPING.keys()))}
    name = os.path.basename(h5_name)

    with h5py.File(h5_tmp, 'w') as f:
        dset_x = f.create_dataset('X', shape=(0, 12, SAMPLES_PER_WINDOW), maxshape=(None, 12, SAMPLES_PER_WINDOW), dtype='float32', chunks=(64, 12, SAMPLES_PER_WINDOW), compression='lzf')
        dset_y = f.create_dataset('Y', shape=(0,), maxshape=(None,), dtype='int8')
        dset_p = f.create_dataset('patient_ids', shape=(0,), maxshape=(None,), dtype='int64')

        skipped = 0
        class_counts = {lab: 0 for lab in all_labels}
        
        bar = tqdm(ids_list, desc=f"{name} | estrazione", unit="ecg")
        for ecg_id in bar:
            try:
                zip_path = id_to_zip.get(str(ecg_id))
                if not zip_path:
                    skipped += 1
                    continue
                
                with zipfile.ZipFile(zip_path, 'r') as z_in:
                    edf_bytes = z_in.read(f"{ecg_id}.edf")
                    ecg_data = read_edf_data(edf_bytes)

                if not ecg_data or not ecg_data["signals"]:
                    skipped += 1
                    continue

                if add_gain:
                    ecg_data["signals"] = apply_electrode_gain(ecg_data["signals"], fs=FS_OLD, noise_multiplier=1.1)

                sigs = all_leads_preprocessing(ecg_data["signals"])
                sigs_array = np.array([sigs[l] for l in ALL_LEADS], dtype=np.float32)

                is_holter = IS_HOLTER_DICT.get(ecg_id, False)
                cfg = QUALITY_CFG_HOLTER if is_holter else QUALITY_CFG_STANDARD
                cfg["stride_sec"] = 2.0 

                # MODIFICA 2: Controllo qualità basato sulle precordiali
                quality_result = check_ecg_quality(sigs_array, cfg=cfg, lead_indices=PRECORDIAL_INDICES)
                if not quality_result['global_valid']:
                    skipped += 1
                    continue

                win_mask = compute_good_window_mask_from_raw(sigs_array, cfg=cfg, min_valid_leads_per_window=5, lead_indices=PRECORDIAL_INDICES)
                if win_mask.size == 0 or not win_mask.any():
                    skipped += 1
                    continue

                # MODIFICA 3: Calcolo mediane e scaling usando le precordiali come reference
                sigs_norm, _, _ = robust_scale_ecg(sigs_array, reference_leads=PRECORDIAL_INDICES)
                sigs_norm_dict = {lead: sigs_norm[i] for i, lead in enumerate(ALL_LEADS)}
                wins_all = create_windows(sigs_norm_dict, stride=STRIDE_SAMPLES)
                
                n_win = min(wins_all.shape[0], win_mask.size)
                wins_r_good = wins_all[:n_win][win_mask[:n_win]]
                
                if wins_r_good.shape[0] == 0:
                    skipped += 1
                    continue

                # Aggiunta classe normale (0)
                n_to_add = wins_r_good.shape[0]
                labels_r = np.full(n_to_add, label_to_int['normale'], dtype='int8')
                _append_to_h5(dset_x, dset_y, dset_p, wins_r_good, labels_r, ecg_id)
                class_counts['normale'] += n_to_add

                # Inversioni (Precordiali)
                for inv_name in ACTIVE_SYNTH_CLASSES_PREC:
                    raw_inv = precordial_interchange_simulation(PRECORDIAL_MAPPING[inv_name], ecg_data["signals"])
                    
                    # Rumore extra generico per dare varianza ai dati simulati
                    if np.random.random() < 0.3:
                        extra_mult = np.random.uniform(1.2, 3.0)
                        raw_inv_noisy = add_extra_noise(raw_inv, multiplier=extra_mult, fs=FS_OLD)
                        sim_sigs = all_leads_preprocessing(raw_inv_noisy)
                    else:
                        sim_sigs = all_leads_preprocessing(raw_inv)

                    sim_sigs_array = np.array([sim_sigs[l] for l in ALL_LEADS], dtype=np.float32)
                    sim_sigs_norm, _, _ = robust_scale_ecg(sim_sigs_array, reference_leads=PRECORDIAL_INDICES)
                    sim_sigs_norm_dict = {lead: sim_sigs_norm[i] for i, lead in enumerate(ALL_LEADS)}
                    
                    wins_s = create_windows(sim_sigs_norm_dict, stride=STRIDE_SAMPLES)
                    wins_s_good = wins_s[:n_win][win_mask[:n_win]]
                    
                    if wins_s_good.shape[0] > 0:
                        labels_s = np.full(wins_s_good.shape[0], label_to_int[inv_name], dtype='int8')
                        _append_to_h5(dset_x, dset_y, dset_p, wins_s_good, labels_s, ecg_id)
                        class_counts[inv_name] += wins_s_good.shape[0]

                bar.set_postfix(windows=dset_x.shape[0], skip=skipped)
            except Exception as e:
                skipped += 1
                continue

        total = dset_x.shape[0]
        if total > 0:
            with h5py.File(h5_name, 'w') as dst:
                dst_x = dst.create_dataset('X', shape=(total, 12, SAMPLES_PER_WINDOW), dtype='float32', chunks=(64, 12, SAMPLES_PER_WINDOW), compression='lzf')
                dst_y = dst.create_dataset('Y', shape=(total,), dtype='int8')
                dst_p = dst.create_dataset('patient_ids', shape=(total,), dtype='int64')
                
                shuffled_idx = np.random.permutation(total)
                for start in tqdm(range(0, total, 4096), desc=f"{name} | fase 3/3 shuffle", unit="batch"):
                    end = min(start + 4096, total)
                    idx = shuffled_idx[start:end]
                    sorted_idx = np.sort(idx)
                    
                    x_blk = dset_x[sorted_idx]
                    y_blk = dset_y[sorted_idx]
                    p_blk = dset_p[sorted_idx]
                    
                    inv_sort = np.argsort(np.argsort(idx))
                    dst_x[start:end] = x_blk[inv_sort]
                    dst_y[start:end] = y_blk[inv_sort]
                    dst_p[start:end] = p_blk[inv_sort]

    if os.path.exists(h5_tmp):
        os.remove(h5_tmp)
    print(f"\n[{name}] Completato: {total:,} finestre, {skipped} ECG scartati")
    for lab, cnt in class_counts.items():
        print(f"  - {lab}: {cnt} finestre")

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
        if any(kw in text_str for kw in text_bad_keywords): continue
        try:
            data = json.loads(report_str)
            codified = data.get('codified', [])
            codes = [c['value'] for c in codified if c.get('type') == 'code']
            if any(c in rejection_codes for c in codes): continue
            is_holter = 'BTWSCQQ43' in codes
            IS_HOLTER_DICT[id_] = is_holter
            clean_ids.append(id_)
        except Exception: continue

    np.random.seed(42)
    np.random.shuffle(clean_ids)
    if max_ecgs and len(clean_ids) > max_ecgs: clean_ids = clean_ids[:max_ecgs]
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
    return id_to_zip

if __name__ == "__main__":
    db_path = r"C:\Users\carme\THESIS\datasets\dataset\records.db"
    out_dir = r"C:\Users\carme\THESIS\datasets\unlabelled_simulated_final"
    dataset_dir = r"C:\Users\carme\THESIS\datasets\dataset\DATASET_complete"
    
    id_to_zip = build_zip_index(dataset_dir)
    all_clean_ids_db = get_clean_ecg_ids(db_path, max_ecgs=20000)
    all_clean_ids = [cid for cid in all_clean_ids_db if str(cid) in id_to_zip]
    
    train_ids, vt_ids = train_test_split(all_clean_ids, test_size=0.20, random_state=42)
    val_ids, test_ids = train_test_split(vt_ids, test_size=0.50, random_state=42)

    os.makedirs(out_dir, exist_ok=True)
    
    # MODIFICA 4: Nome file output corretto
    target_h5 = os.path.join(out_dir, "unlabelled_z_median_precordials_test_with_patients.h5")

    print(f"\n=== GENERAZIONE TEST SET PRECORDIALI CON ID PAZIENTI ===")
    print(f"Target: {target_h5}")
    
    build_unlabelled_precordials_dataset_with_patients(test_ids, target_h5, id_to_zip, add_gain=True)
    print("\nFatto.")