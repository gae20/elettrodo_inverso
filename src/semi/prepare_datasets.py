"""
prepare_datasets.py

Carica thesis-sample.csv, esegue lo split stratificato dei pazienti in Train (80%), Val (10%), Test (10%)
e compila i relativi file H5 iniziali con le finestre estratte dagli EDF reali di dataset_small/.

Bilanciamento:
  - Train e Val: upsampling per avere lo STESSO numero di finestre per ciascuna delle 6 classi.
  - Test: 50% campioni normali + 50% campioni di inversione divisi equamente tra le 5 classi.
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Setup path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.join(SCRIPT_DIR, '..')
THESIS_DIR = os.path.join(SRC_DIR, '..')

sys.path.append(SRC_DIR)
from data.data_pipeline import (
    read_edf_data, all_leads_preprocessing,
    check_ecg_quality, check_window_quality,
)
from utils.config import (
    SAMPLES_PER_WINDOW, FS_NEW, ALL_LEADS, QUALITY_CFG, STRIDE_SAMPLES,
)

# --- Percorsi ---
SMALL_DIR  = os.path.join(THESIS_DIR, 'datasets', 'dataset', 'dataset_small')
CSV_PATH   = os.path.join(SMALL_DIR, 'thesis-sample.csv')
OUT_DIR    = os.path.join(SCRIPT_DIR, 'results', 'semi_h5')

# --- Costanti ---
LIMB_INDICES = list(range(6))
LIMB_LEADS   = ALL_LEADS[:6]

CLASS_NAMES = ['normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']

LABEL_MAPPING = {
    'normale':    0,
    'RL':         1,  # LA-RA
    'RF':         2,  # RA-LL
    'LF':         3,  # LA-LL
    'orario':     4,  # ROT_ORARIA (ROT_ORA)
    'antiorario': 5   # ROT_ANTIORARIA (ROT_ANT)
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def robust_scale_ecg(sigs_array, eps=1e-8):
    x = sigs_array.astype(np.float32)
    medians = np.median(x, axis=1, keepdims=True)
    q75, q25 = np.percentile(x, [75, 25])
    scale = max((q75 - q25) / 1.34896, eps)
    return (x - medians) / scale


def compute_good_window_mask(sigs_array, cfg, min_valid_leads=5):
    fs       = cfg["fs"]
    win_size = int(cfg["win_sec"] * fs)
    _, n_samples = sigs_array.shape
    if n_samples < win_size:
        return np.zeros(0, dtype=bool)
    starts  = list(range(0, n_samples - win_size + 1, STRIDE_SAMPLES))
    mask    = np.zeros(len(starts), dtype=bool)
    for w_idx, start in enumerate(starts):
        valid_count = sum(
            check_window_quality(sigs_array[li, start:start + win_size],
                                 cfg=cfg, lead_idx=li)["valid"]
            for li in LIMB_INDICES
        )
        mask[w_idx] = (valid_count >= min_valid_leads)
    return mask


def create_windows(sigs_array):
    _, n_samples = sigs_array.shape
    if n_samples < SAMPLES_PER_WINDOW:
        return np.empty((0, 6, SAMPLES_PER_WINDOW), dtype=np.float32)
    starts = list(range(0, n_samples - SAMPLES_PER_WINDOW + 1, STRIDE_SAMPLES))
    windows = []
    for start in starts:
        windows.append(sigs_array[:, start:start + SAMPLES_PER_WINDOW])
    return np.array(windows, dtype=np.float32)


def extract_windows_from_edf(patient_id):
    edf_path = os.path.join(SMALL_DIR, f"record{patient_id}.edf")
    if not os.path.exists(edf_path):
        return None
    try:
        with open(edf_path, 'rb') as f:
            edf_bytes = f.read()
        ecg_data = read_edf_data(edf_bytes)
    except Exception:
        return None

    if not ecg_data or not ecg_data.get("signals"):
        return None
    if not all(l in ecg_data["signals"] for l in LIMB_LEADS):
        return None

    sigs     = all_leads_preprocessing(ecg_data["signals"])
    sigs_raw = np.array([sigs[l] for l in LIMB_LEADS], dtype=np.float32)

    quality  = check_ecg_quality(sigs_raw, cfg=QUALITY_CFG, lead_indices=LIMB_INDICES)
    if not quality['global_valid']:
        return None

    win_mask  = compute_good_window_mask(sigs_raw, cfg=QUALITY_CFG, min_valid_leads=5)
    sigs_norm = robust_scale_ecg(sigs_raw)
    windows   = create_windows(sigs_norm)

    n    = min(windows.shape[0], win_mask.size)
    good = windows[:n][win_mask[:n]]
    return good if good.shape[0] > 0 else None


# ---------------------------------------------------------------------------
# Split stratificato
# ---------------------------------------------------------------------------

def stratified_split(df, seed=42):
    np.random.seed(seed)
    train_ids, val_ids, test_ids = [], [], []

    for c in range(6):
        class_df = df[df['class_idx'] == c]
        ids = class_df.index.unique().tolist()
        np.random.shuffle(ids)
        n = len(ids)

        if n >= 3:
            n_train = int(round(0.8 * n))
            n_val   = int(round(0.1 * n))
            if n_val == 0: n_val = 1
            n_test  = n - n_train - n_val
            if n_test <= 0:
                n_test  = 1
                n_train = n - n_val - n_test

            train_ids.extend(ids[:n_train])
            val_ids.extend(ids[n_train:n_train + n_val])
            test_ids.extend(ids[n_train + n_val:])
        else:
            if n == 2:
                train_ids.append(ids[0])
                test_ids.append(ids[1])
            elif n == 1:
                train_ids.append(ids[0])

    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Bilanciamento
# ---------------------------------------------------------------------------

def upsample_to_max(X_by_class, seed=42):
    """
    Porta tutte le classi allo stesso numero di finestre della classe più numerosa
    tramite oversampling (replicazione + campionamento casuale del residuo).
    """
    counts = [X_by_class[c].shape[0] for c in range(6)]
    target = max(counts)
    print(f"  Target finestre per classe (upsample): {target:,}")

    X_bal, Y_bal = [], []
    np.random.seed(seed)
    for c in range(6):
        X_c = X_by_class[c]
        n_c = X_c.shape[0]
        if n_c == 0:
            continue
        factor    = target // n_c
        remainder = target % n_c
        X_rep = np.tile(X_c, (factor, 1, 1))
        if remainder > 0:
            chosen = np.random.choice(n_c, size=remainder, replace=(remainder > n_c))
            X_rep  = np.concatenate([X_rep, X_c[chosen]], axis=0)
        X_bal.append(X_rep)
        Y_bal.append(np.full(target, c, dtype=np.int8))
        print(f"    {CLASS_NAMES[c]:<12}: {n_c:>5,} → {target:>5,}  (×{factor} + {remainder})")

    X_out = np.concatenate(X_bal, axis=0)
    Y_out = np.concatenate(Y_bal, axis=0)
    perm  = np.random.permutation(len(Y_out))
    return X_out[perm], Y_out[perm]


def build_test_balanced(X_by_class, seed=42):
    """
    Costruisce il test set con:
      - 50% campioni normali (classe 0)
      - 50% campioni di inversione, divisi equamente tra le 5 classi (1-5)
    
    La dimensione di ogni classe di inversione è limitata al minimo disponibile
    tra le classi 1-5. I normali vengono sottocampionati/sovracampionati allo
    stesso totale.
    """
    np.random.seed(seed)

    # Trova il minimo tra le classi di inversione (1-5)
    inversion_counts = [len(X_by_class[c]) for c in range(1, 6)]
    min_inv = min(inversion_counts)
    print(f"  Minimo finestre tra le classi di inversione: {min_inv}")
    print(f"  Target per classe di inversione: {min_inv}")
    print(f"  Target normali (= 5 × {min_inv}): {5 * min_inv}")

    X_bal, Y_bal = [], []

    # Classi di inversione: campionamento/upsampling a min_inv
    for c in range(1, 6):
        X_c = X_by_class[c]
        n_c = len(X_c)
        if n_c >= min_inv:
            chosen = np.random.choice(n_c, size=min_inv, replace=False)
        else:
            # Upsample se necessario (caso raro nel test)
            chosen = np.random.choice(n_c, size=min_inv, replace=True)
        X_bal.append(X_c[chosen])
        Y_bal.append(np.full(min_inv, c, dtype=np.int8))
        print(f"    {CLASS_NAMES[c]:<12}: {n_c:>5,} → {min_inv}")

    # Normali: campionamento a 5 × min_inv
    n_normal = 5 * min_inv
    X_norm   = X_by_class[0]
    n_c_norm = len(X_norm)
    if n_c_norm >= n_normal:
        chosen_norm = np.random.choice(n_c_norm, size=n_normal, replace=False)
    else:
        chosen_norm = np.random.choice(n_c_norm, size=n_normal, replace=True)
    X_bal.append(X_norm[chosen_norm])
    Y_bal.append(np.full(n_normal, 0, dtype=np.int8))
    print(f"    {CLASS_NAMES[0]:<12}: {n_c_norm:>5,} → {n_normal}")

    X_out = np.concatenate(X_bal, axis=0)
    Y_out = np.concatenate(Y_bal, axis=0)
    perm  = np.random.permutation(len(Y_out))
    return X_out[perm], Y_out[perm]


# ---------------------------------------------------------------------------
# Costruzione H5
# ---------------------------------------------------------------------------

def collect_windows_by_class(patient_ids, labels_dict):
    """Estrae tutte le finestre e le raccoglie per classe."""
    X_by_class = {c: [] for c in range(6)}
    for pid in tqdm(patient_ids, desc="Processamento EDF"):
        label = labels_dict[pid]
        wins  = extract_windows_from_edf(pid)
        if wins is not None:
            X_by_class[label].append(wins)

    # Concatena per classe
    result = {}
    for c in range(6):
        if X_by_class[c]:
            result[c] = np.concatenate(X_by_class[c], axis=0)
        else:
            result[c] = np.empty((0, 6, SAMPLES_PER_WINDOW), dtype=np.float32)
        print(f"  {CLASS_NAMES[c]:<12}: {result[c].shape[0]:,} finestre raw")
    return result


def save_h5(X, Y, out_path):
    if os.path.exists(out_path):
        os.remove(out_path)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('X', data=X, compression='lzf')
        f.create_dataset('Y', data=Y)
    print(f"  Salvato: {out_path}  ({len(Y):,} finestre totali)")
    for c in range(6):
        n = int(np.sum(Y == c))
        print(f"    {CLASS_NAMES[c]:<12}: {n:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("PREPARAZIONE DATASET INIZIALI (Dati Reali da dataset_small)")
    print("=" * 60)

    if not os.path.exists(CSV_PATH):
        print(f"ERRORE: CSV non trovato in {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    df_valid = df[df['Inversione'].isin(LABEL_MAPPING.keys())].copy()
    df_valid['class_idx'] = df_valid['Inversione'].map(LABEL_MAPPING)
    df_valid = df_valid.set_index('Num')
    labels_dict = df_valid['class_idx'].to_dict()

    # Split stratificato dei pazienti
    train_ids, val_ids, test_ids = stratified_split(df_valid, seed=42)
    print(f"\nSplit Pazienti: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # --- TRAIN ---
    print("\n--- TRAIN ---")
    X_by_class_train = collect_windows_by_class(train_ids, labels_dict)
    print("\n  Bilanciamento Train (upsample a parità di classe):")
    X_train, Y_train = upsample_to_max(X_by_class_train)
    save_h5(X_train, Y_train, os.path.join(OUT_DIR, 'train_small_init.h5'))

    # --- VAL ---
    print("\n--- VAL ---")
    X_by_class_val = collect_windows_by_class(val_ids, labels_dict)
    print("\n  Bilanciamento Val (upsample a parità di classe):")
    X_val, Y_val = upsample_to_max(X_by_class_val)
    save_h5(X_val, Y_val, os.path.join(OUT_DIR, 'val_small_init.h5'))

    # --- TEST (50% normali, 50% inversioni bilanciate) ---
    print("\n--- TEST ---")
    X_by_class_test = collect_windows_by_class(test_ids, labels_dict)
    print("\n  Bilanciamento Test (50% normale | 50% inversioni):")
    X_test, Y_test = build_test_balanced(X_by_class_test)
    save_h5(X_test, Y_test, os.path.join(OUT_DIR, 'test_small.h5'))

    print("\nOK! Tutti i dataset iniziali reali sono pronti.")
