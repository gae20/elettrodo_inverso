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
import multiprocessing
import concurrent.futures


def train_test_split(data, test_size=0.25, random_state=42):
    np.random.seed(random_state)
    shuffled = np.random.permutation(data)
    split_idx = int(len(data) * (1 - test_size))
    return list(shuffled[:split_idx]), list(shuffled[split_idx:])

# Import locali
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.generate_ids import get_clean_ecg_ids, IS_HOLTER_DICT
from data.data_pipeline import (
    read_edf_data, all_leads_preprocessing, check_ecg_quality, 
    check_window_quality, limb_interchange_simulation, apply_electrode_gain, add_extra_noise,
    apply_random_scaling, add_baseline_wander
)
from utils.config import (
    SAMPLES_PER_WINDOW, FS_NEW, FS_OLD,
    ALL_LEADS, MAPPING_INV, ACTIVE_SYNTH_CLASSES, 
    QUALITY_CFG, robust_scale_ecg
)
from utils.sqa_real_config import QUALITY_CFG_SYNTH_RELAXED

LIMB_INDICES = list(range(6))
STRIDE_SAMPLES = int(FS_NEW * 2.0)



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

def process_single_ecg(args):
    ecg_id, zip_path, add_gain, is_holter = args
    np.random.seed()  # seed randomico distinto in ogni worker
    results = {}

    try:
        if not zip_path:
            return None

        with zipfile.ZipFile(zip_path, 'r') as z_in:
            edf_bytes = z_in.read(f"{ecg_id}.edf")
            ecg_data = read_edf_data(edf_bytes)

        if not ecg_data or not ecg_data["signals"]:
            return None

        # ── Segnale pulito originale (NO augmentation qui) ───────────────
        raw_clean = ecg_data["signals"]

        # SQA sull'originale pulito per verificare che l'ECG di base sia OK
        sigs_clean = all_leads_preprocessing(raw_clean)
        sigs_array_clean = np.array([sigs_clean[l] for l in ALL_LEADS], dtype=np.float32)

        cfg = QUALITY_CFG_SYNTH_RELAXED.copy()  # .copy() per evitare side-effects
        cfg["stride_sec"] = 2.0

        quality_result = check_ecg_quality(sigs_array_clean, cfg=cfg, lead_indices=LIMB_INDICES)
        if not quality_result['global_valid']:
            return None

        # ── Parametri augmentation (generati una volta, applicati per-classe) ──
        global_extra_mult = np.random.uniform(0.5, 1.5) if add_gain else 0.0
        global_scale = np.random.uniform(0.6, 1.4) if add_gain else 1.0

        def augment_raw(raw_signals):
            # Nessuna augmentation sintetica applicata.
            # I segnali clinici reali contengono già la varianza naturale (baseline wander,
            # artefatti muscolari, rumore elettrodico) e la normalizzazione annullerebbe
            # comunque qualsiasi random scaling dell'ampiezza.
            return raw_signals

        # ── 1. Classe Normale ────────────────────────────────────────────
        raw_norm_aug = augment_raw(raw_clean)
        sigs_norm = all_leads_preprocessing(raw_norm_aug)
        sigs_array_norm = np.array([sigs_norm[l] for l in ALL_LEADS], dtype=np.float32)

        # FIX 1: win_mask per la classe normale (dopo preprocessing)
        win_mask_norm = compute_good_window_mask_from_raw(
            sigs_array_norm, cfg=cfg, min_valid_leads_per_window=5,
            lead_indices=LIMB_INDICES)

        if win_mask_norm.size > 0 and win_mask_norm.any():
            # Ogni classe ha la sua normalizzazione (come nei dati reali)
            sigs_scaled_norm, _, _ = robust_scale_ecg(sigs_array_norm, reference_leads=LIMB_INDICES)
            sigs_dict_norm = {lead: sigs_scaled_norm[i] for i, lead in enumerate(ALL_LEADS)}
            wins_norm = create_windows(sigs_dict_norm, stride=STRIDE_SAMPLES)
            n_win = min(wins_norm.shape[0], win_mask_norm.size)
            wins_good = wins_norm[:n_win][win_mask_norm[:n_win]]
            if wins_good.shape[0] > 0:
                results['normale'] = wins_good

        # ── 2. Classi Invertite ──────────────────────────────────────────
        for inv_name in ACTIVE_SYNTH_CLASSES:
            # FIX 2: Inversione sul segnale PULITO, augmentation DOPO
            raw_inv = limb_interchange_simulation(MAPPING_INV[inv_name], raw_clean)
            raw_inv_aug = augment_raw(raw_inv)

            sim_sigs = all_leads_preprocessing(raw_inv_aug)
            sim_sigs_array = np.array([sim_sigs[l] for l in ALL_LEADS], dtype=np.float32)

            # FIX 1: win_mask per-classe (dopo inversione + augmentation)
            win_mask_inv = compute_good_window_mask_from_raw(
                sim_sigs_array, cfg=cfg, min_valid_leads_per_window=5,
                lead_indices=LIMB_INDICES)

            if win_mask_inv.size == 0 or not win_mask_inv.any():
                continue

            # Normalizzazione indipendente per-classe (come nei dati reali)
            sim_sigs_norm, _, _ = robust_scale_ecg(sim_sigs_array, reference_leads=LIMB_INDICES)
            sim_sigs_dict = {lead: sim_sigs_norm[i] for i, lead in enumerate(ALL_LEADS)}

            wins_s = create_windows(sim_sigs_dict, stride=STRIDE_SAMPLES)
            n_win = min(wins_s.shape[0], win_mask_inv.size)
            wins_good = wins_s[:n_win][win_mask_inv[:n_win]]

            if wins_good.shape[0] > 0:
                results[inv_name] = wins_good

        return results if results else None
    except Exception as e:
        return None

def build_unlabelled_limbs_dataset(ids_list, h5_name, id_to_zip, max_windows_per_class=None, add_gain=True):
    if os.path.exists(h5_name): os.remove(h5_name)
    h5_tmp = h5_name + ".tmp"
    try:
        if os.path.exists(h5_tmp): os.remove(h5_tmp)
    except PermissionError:
        import time; time.sleep(1)
        if os.path.exists(h5_tmp): os.remove(h5_tmp)

    all_labels = ['normale'] + list(ACTIVE_SYNTH_CLASSES)
    all_mapping = ['normale'] + list(MAPPING_INV.keys())
    label_to_int = {lab: idx for idx, lab in enumerate(all_mapping)}
    name = os.path.basename(h5_name)

    # Preparazione dei task per il pool di processi
    tasks = []
    for ecg_id in ids_list:
        zip_path = id_to_zip.get(str(ecg_id))
        is_holter = IS_HOLTER_DICT.get(ecg_id, False)
        tasks.append((ecg_id, zip_path, add_gain, is_holter))

    with h5py.File(h5_tmp, 'w') as f:
        dset_x = f.create_dataset('X', shape=(0, 12, SAMPLES_PER_WINDOW), maxshape=(None, 12, SAMPLES_PER_WINDOW), dtype='float32', chunks=(64, 12, SAMPLES_PER_WINDOW), compression='lzf')
        dset_y = f.create_dataset('Y', shape=(0,), maxshape=(None,), dtype='int8')

        skipped = 0
        class_counts = {lab: 0 for lab in all_labels}
        
        # Usa quasi tutti i core disponibili
        n_workers = max(1, multiprocessing.cpu_count() - 1)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_single_ecg, t) for t in tasks]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"{name} | extraction", unit="ecg"):
                if max_windows_per_class is not None:
                    if all(class_counts[lab] >= max_windows_per_class for lab in all_labels):
                        # Cancella task rimanenti se limite raggiunto
                        for f_cancel in futures: f_cancel.cancel()
                        break

                try:
                    res = future.result()
                except Exception:
                    res = None

                if not res:
                    skipped += 1
                    continue

                for lab in all_labels:
                    if lab in res:
                        wins = res[lab]
                        n_to_add = wins.shape[0]
                        if max_windows_per_class is not None:
                            remaining = max_windows_per_class - class_counts[lab]
                            n_to_add = min(n_to_add, remaining)

                        if n_to_add > 0:
                            labels = np.full(n_to_add, label_to_int[lab], dtype='int8')
                            _append_to_h5(dset_x, dset_y, wins[:n_to_add], labels)
                            class_counts[lab] += n_to_add

        total = dset_x.shape[0]
        if total > 0:
            with h5py.File(h5_name, 'w') as dst:
                dst_x = dst.create_dataset('X', shape=(total, 12, SAMPLES_PER_WINDOW), dtype='float32', chunks=(64, 12, SAMPLES_PER_WINDOW), compression='lzf')
                dst_y = dst.create_dataset('Y', shape=(total,), dtype='int8')
                # BLOCK-LEVEL SHUFFLE IN RAM: Velocissimo (lettura/scrittura sequenziale)
                # Invece di fare seek casuali su disco, leggiamo enormi blocchi,
                # li mischiamo in RAM e li riscriviamo. Per il training è equivalente!
                block_size = 50000 
                for start in tqdm(range(0, total, block_size), desc=f"{name} | fase 3/3 shuffle", unit="block"):
                    end = min(start + block_size, total)
                    
                    # 1. Lettura sequenziale (I/O velocissimo)
                    x_blk = dset_x[start:end]
                    y_blk = dset_y[start:end]
                    
                    # 2. Shuffle in RAM (Istanteo)
                    local_idx = np.random.permutation(end - start)
                    
                    # 3. Scrittura sequenziale
                    dst_x[start:end] = x_blk[local_idx]
                    dst_y[start:end] = y_blk[local_idx]

    if os.path.exists(h5_tmp): os.remove(h5_tmp)
    print(f"  [{name}] Completato: {total:,} finestre, {skipped} ECG scartati")
    for lab, cnt in class_counts.items():
        print(f"    - {lab}: {cnt} finestre")

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
    # Necessario su Windows per ProcessPoolExecutor
    multiprocessing.freeze_support()
    
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'records_complete.db')
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'datasets')
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'datasets', 'dataset_normals')
    
    id_to_zip = build_zip_index(dataset_dir)
    all_clean_ids_db = get_clean_ecg_ids(db_path, max_ecgs=20000)
    all_clean_ids = [cid for cid in all_clean_ids_db if str(cid) in id_to_zip]
    
    train_ids, vt_ids = train_test_split(all_clean_ids, test_size=0.20, random_state=42)
    val_ids, test_ids = train_test_split(vt_ids, test_size=0.50, random_state=42)

    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n=== LIMBS UNLABELLED GAIN PARALLEL (SQA 15uV) ===")
    print(f"Output directory: {out_dir}")
    print(f"CPU Threads totali in uso: {max(1, multiprocessing.cpu_count() - 1)}")
    
    build_unlabelled_limbs_dataset(test_ids,  os.path.join(out_dir, "unlabelled_final_noise_limbs_test.h5"),  id_to_zip, max_windows_per_class=None, add_gain=False)
    build_unlabelled_limbs_dataset(val_ids,   os.path.join(out_dir, "unlabelled_final_noise_limbs_val.h5"),   id_to_zip, max_windows_per_class=None, add_gain=False)
    build_unlabelled_limbs_dataset(train_ids, os.path.join(out_dir, "unlabelled_final_noise_limbs_train.h5"), id_to_zip, max_windows_per_class=None, add_gain=True)
    print("\nFatto.")
