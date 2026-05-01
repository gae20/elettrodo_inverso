import os
import sys
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Import locali
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.data_pipeline import (
    get_ecg, all_leads_preprocessing, check_ecg_quality, 
    check_window_quality, limb_interchange_simulation,
    BUCKET, BASE_PATH, s3_client
)
from utils.config import (
    SAMPLES_PER_WINDOW, STRIDE_SAMPLES,
    ALL_LEADS, MAPPING_INV, ACTIVE_SYNTH_CLASSES, 
    QUALITY_CFG, LABEL_MAP_CLEAN
)

LIMB_INDICES = list(range(6))

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

def _append_to_h5(dset_x, dset_y, windows, labels):
    n = windows.shape[0]
    curr = dset_x.shape[0]
    dset_x.resize(curr + n, axis=0)
    dset_y.resize(curr + n, axis=0)
    dset_x[curr:curr + n] = windows
    dset_y[curr:curr + n] = labels

def build_limbs_dataset(ids_list, df_valido, h5_name, augment=False):
    if os.path.exists(h5_name): os.remove(h5_name)
    h5_tmp = h5_name + ".tmp"
    if os.path.exists(h5_tmp): os.remove(h5_tmp)

    all_labels = ['normale'] + list(MAPPING_INV.keys())
    label_to_int = {lab: idx for idx, lab in enumerate(all_labels)}
    name = os.path.basename(h5_name)

    with h5py.File(h5_tmp, 'w') as f:
        dset_x = f.create_dataset('X', shape=(0, 12, SAMPLES_PER_WINDOW), maxshape=(None, 12, SAMPLES_PER_WINDOW), dtype='float32', chunks=(64, 12, SAMPLES_PER_WINDOW), compression='lzf')
        dset_y = f.create_dataset('Y', shape=(0,), maxshape=(None,), dtype='int8')

        skipped = 0
        # ORDINE: Processiamo prima i normali per stabilire il target (CAPPING)
        sorted_ids = sorted(ids_list, key=lambda x: 0 if str(df_valido.loc[x, 'Inversione']) == 'normale' else 1)
        
        class_counts = {lab: 0 for lab in all_labels}
        target_windows = 0
        
        bar = tqdm(sorted_ids, desc=f"{name} | fase 1/3 reali", unit="ecg")
        for ecg_id in bar:
            try:
                label_reale = str(df_valido.loc[ecg_id, 'Inversione'])
                if label_reale != 'normale' and target_windows > 0 and class_counts.get(label_reale, 0) >= target_windows:
                    continue

                ecg_data = get_ecg(ecg_id)
                if not ecg_data or not ecg_data["signals"] or ecg_id not in df_valido.index:
                    skipped += 1; continue

                sigs = all_leads_preprocessing(ecg_data["signals"])
                sigs_array = np.array([sigs[l] for l in ALL_LEADS], dtype=np.float32)

                quality_result = check_ecg_quality(sigs_array, cfg=QUALITY_CFG, lead_indices=LIMB_INDICES)
                if not quality_result['global_valid']:
                    skipped += 1; continue

                win_mask = compute_good_window_mask_from_raw(sigs_array, cfg=QUALITY_CFG, min_valid_leads_per_window=5, lead_indices=LIMB_INDICES)
                if win_mask.size == 0 or not win_mask.any():
                    skipped += 1; continue

                sigs_norm, _, _ = zscore_ecg(sigs_array)
                sigs_norm_dict = {lead: sigs_norm[i] for i, lead in enumerate(ALL_LEADS)}
                wins_all = create_windows(sigs_norm_dict)
                n_win = min(wins_all.shape[0], win_mask.size)
                wins_r_good = wins_all[:n_win][win_mask[:n_win]]
                
                if wins_r_good.shape[0] == 0:
                    skipped += 1; continue

                n_to_add = wins_r_good.shape[0]
                if label_reale != 'normale' and target_windows > 0:
                    remaining = target_windows - class_counts.get(label_reale, 0)
                    n_to_add = min(n_to_add, remaining)
                
                if n_to_add <= 0: continue

                labels_r = np.full(n_to_add, label_to_int[label_reale], dtype='int8')
                _append_to_h5(dset_x, dset_y, wins_r_good[:n_to_add], labels_r)
                
                class_counts[label_reale] += n_to_add
                if label_reale == 'normale':
                    target_windows = class_counts['normale']
                
                bar.set_postfix(windows=dset_x.shape[0], skip=skipped)
            except:
                skipped += 1; continue

        if augment and target_windows > 0:
            needed_per_class = {lab: max(0, target_windows - class_counts.get(lab, 0)) for lab in ACTIVE_SYNTH_CLASSES}
            bar2 = tqdm(ids_list, desc=f"{name} | fase 2/3 augment", unit="ecg")
            for ecg_id in bar2:
                if not any(v > 0 for v in needed_per_class.values()): break
                try:
                    if str(df_valido.loc[ecg_id, 'Inversione']) != 'normale': continue
                    ecg_data = get_ecg(ecg_id)
                    sigs = all_leads_preprocessing(ecg_data["signals"])
                    for inv_name in ACTIVE_SYNTH_CLASSES:
                        needed = needed_per_class[inv_name]
                        if needed <= 0: continue
                        sim_sigs = limb_interchange_simulation(MAPPING_INV[inv_name], sigs)
                        wins_s = create_windows(sim_sigs)
                        n_add = min(needed, wins_s.shape[0])
                        if n_add > 0:
                            _append_to_h5(dset_x, dset_y, wins_s[:n_add], np.full(n_add, label_to_int[inv_name], dtype='int8'))
                            needed_per_class[inv_name] -= n_add
                    bar2.set_postfix(still_needed=sum(needed_per_class.values()))
                except: continue

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

LOCAL_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'dataset', 'thesis-sample.csv')

def load_local_df():
    if os.path.exists(LOCAL_CSV_PATH):
        df = pd.read_csv(LOCAL_CSV_PATH)
        df_cand = df[df["Inversione"] != "?"].copy()
        df_cand["Inversione"] = df_cand["Inversione"].apply(lambda x: LABEL_MAP_CLEAN.get(x, x))
        return df_cand.set_index("Num")
    return None

if __name__ == "__main__":
    df_valido = load_local_df()
    if df_valido is not None:
        normal_ids = df_valido[df_valido['Inversione'] == 'normale'].index.unique()
        anomaly_ids = df_valido[df_valido['Inversione'] != 'normale'].index.unique()
        
        train_norm_ids, vt_norm_ids = train_test_split(normal_ids, test_size=0.20, random_state=42)
        val_norm_ids, test_norm_ids = train_test_split(vt_norm_ids, test_size=0.50, random_state=42)
        val_anom_ids, test_anom_ids = train_test_split(anomaly_ids, test_size=0.50, random_state=42)
        
        train_ids = list(train_norm_ids)
        val_ids   = list(val_norm_ids) + list(val_anom_ids)
        test_ids  = list(test_norm_ids) + list(test_anom_ids)

        print(f"Dataset Split (Huang Strategy):")
        print(f"  Train: {len(train_ids)} ECG (solo Normali, verranno aumentati)")
        print(f"  Val:   {len(val_ids)} ECG (mix Normali + Anomalie Reali)")
        print(f"  Test:  {len(test_ids)} ECG (mix Normali + Anomalie Reali)")

        os.makedirs("../datasets", exist_ok=True)
        print("\n=== LIMBS (periferiche) ===")
        build_limbs_dataset(test_ids,  df_valido, "../datasets/limbs_test.h5",  augment=True)
        build_limbs_dataset(val_ids,   df_valido, "../datasets/limbs_val.h5",   augment=True)
        build_limbs_dataset(train_ids, df_valido, "../datasets/limbs_train.h5", augment=True)
        print("\nDone.")
