"""
step3_build_pseudolabelled_h5.py

Costruisce il nuovo training set SSL (Opzione C del piano):
  - Subsample stratificato del training originale a 200.000 finestre (33.333/classe)
  - Pseudo-labeled replicati fino a ~10% del totale (~22.000 finestre)
  - Output: unlabelled_z_median_limbs_train_ssl.h5

L'originale (unlabelled_z_median_limbs_train.h5) NON viene mai modificato.
"""

import os
import sys
import json
import copy
import zipfile
import h5py
import numpy as np
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
    SAMPLES_PER_WINDOW, FS_NEW, ALL_LEADS, QUALITY_CFG,
)

# --- Percorsi ---
RESULTS_DIR  = os.path.join(SCRIPT_DIR, 'results')
DATASETS_DIR = os.path.join(THESIS_DIR, 'datasets', 'unlabelled_simulated_final')
ORIG_TRAIN   = os.path.join(DATASETS_DIR, 'unlabelled_z_median_limbs_train.h5')
SSL_TRAIN    = os.path.join(DATASETS_DIR, 'unlabelled_z_median_limbs_train_ssl.h5')

# --- Costanti ---
LIMB_INDICES   = list(range(6))
STRIDE_SAMPLES = int(FS_NEW * 2.0)

N_ORIG_SUBSET    = 200_000
TARGET_RATIO     = 0.10
NUM_ORIG_CLASSES = 6
N_PER_CLASS_ORIG = N_ORIG_SUBSET // NUM_ORIG_CLASSES  # 33.333
CHUNK_SIZE       = 4096   # finestre lette per volta dall'H5

# Quality config per ECG reali (10s), basata su QUALITY_CFG di config.py
QUALITY_CFG_STD = copy.deepcopy(QUALITY_CFG)
QUALITY_CFG_STD["baseline_max_uv"] = 500.0
QUALITY_CFG_STD["mad_noise_limb"]  = 15.0
QUALITY_CFG_STD["mad_noise_prec"]  = 20.0
QUALITY_CFG_STD["min_valid_ratio"] = 0.70
QUALITY_CFG_STD["stride_sec"]      = 2.0

CLASS_NAMES = ['normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def robust_scale_ecg(sigs_array, eps=1e-8, reference_leads=None):
    x = sigs_array.astype(np.float32)
    medians = np.median(x, axis=1, keepdims=True)
    ref = x[reference_leads, :] if reference_leads is not None else x
    q75, q25 = np.percentile(ref, [75, 25])
    scale = max((q75 - q25) / 1.34896, eps)
    return (x - medians) / scale


def compute_good_window_mask(sigs_array, cfg, min_valid_leads=5, lead_indices=None):
    fs       = cfg["fs"]
    win_size = int(cfg["win_sec"] * fs)
    stride   = int(cfg["stride_sec"] * fs)
    _, n_samples = sigs_array.shape
    if n_samples < win_size:
        return np.zeros(0, dtype=bool)
    indices = lead_indices if lead_indices is not None else list(range(sigs_array.shape[0]))
    starts  = list(range(0, n_samples - win_size + 1, stride))
    mask    = np.zeros(len(starts), dtype=bool)
    for w_idx, start in enumerate(starts):
        valid_count = sum(
            check_window_quality(sigs_array[li, start:start + win_size],
                                 cfg=cfg, lead_idx=li)["valid"]
            for li in indices
        )
        mask[w_idx] = (valid_count >= min_valid_leads)
    return mask


def create_windows(sigs_norm, win_size=SAMPLES_PER_WINDOW, stride=STRIDE_SAMPLES):
    n_leads, n_samples = sigs_norm.shape
    if n_samples < win_size:
        return np.empty((0, n_leads, win_size), dtype=np.float32)
    starts = range(0, n_samples - win_size + 1, stride)
    return np.array([sigs_norm[:, s:s + win_size] for s in starts], dtype=np.float32)


def find_real_dataset_paths():
    possible_csv_paths = [
        os.path.join(THESIS_DIR, 'datasets', 'dataset', 'dataset_small', 'thesis-sample.csv'),
        os.path.join(THESIS_DIR, 'datasets', 'datasets', 'thesis-sample.csv'),
        os.path.join(THESIS_DIR, 'datasets', 'dataset', 'thesis-sample.csv'),
    ]
    for p in possible_csv_paths:
        if os.path.exists(p):
            return os.path.dirname(p)
    return None


def extract_windows_from_edf(ecg_id, zip_path):
    try:
        if zip_path and zip_path.endswith('.zip'):
            with zipfile.ZipFile(zip_path, 'r') as z:
                edf_bytes = z.read(f"{ecg_id}.edf")
        else:
            edf_dir = find_real_dataset_paths()
            if not edf_dir:
                return None
            edf_path = os.path.join(edf_dir, f"record{ecg_id}.edf")
            if not os.path.exists(edf_path):
                edf_path_alt = os.path.join(edf_dir, f"{ecg_id}.edf")
                if os.path.exists(edf_path_alt):
                    edf_path = edf_path_alt
                else:
                    return None
            with open(edf_path, 'rb') as f:
                edf_bytes = f.read()
    except Exception:
        return None


    ecg_data = read_edf_data(edf_bytes)
    if not ecg_data or not ecg_data.get("signals"):
        return None
    if not all(l in ecg_data["signals"] for l in ALL_LEADS):
        return None

    sigs     = all_leads_preprocessing(ecg_data["signals"])
    sigs_raw = np.array([sigs[l] for l in ALL_LEADS], dtype=np.float32)

    quality = check_ecg_quality(sigs_raw, cfg=QUALITY_CFG_STD, lead_indices=LIMB_INDICES)
    if not quality['global_valid']:
        return None

    # Mask sul RAW, finestre sul normalizzato
    win_mask  = compute_good_window_mask(sigs_raw, cfg=QUALITY_CFG_STD,
                                         min_valid_leads=5, lead_indices=LIMB_INDICES)
    sigs_norm = robust_scale_ecg(sigs_raw, reference_leads=LIMB_INDICES)
    windows   = create_windows(sigs_norm)

    n    = min(windows.shape[0], win_mask.size)
    good = windows[:n][win_mask[:n]]
    return good if good.shape[0] > 0 else None


# ---------------------------------------------------------------------------
# Step 1 — Subsampling stratificato con lettura a chunk
# ---------------------------------------------------------------------------

def subsample_original_train(h5_orig_path, n_per_class=N_PER_CLASS_ORIG,
                              n_classes=NUM_ORIG_CLASSES):
    print(f"\n[1/3] Subsampling del training originale -> {n_per_class:,} finestre/classe")
    print(f"      ({n_classes * n_per_class:,} finestre totali su 585.132)\n")

    with h5py.File(h5_orig_path, 'r') as f:
        y_all = f['Y'][:]

    # Selezione indici per classe
    np.random.seed(42)
    selected_idx = []
    for c in range(n_classes):
        cls_idx = np.where(y_all == c)[0]
        chosen  = np.random.choice(cls_idx, size=min(n_per_class, len(cls_idx)), replace=False)
        selected_idx.append(chosen)
        print(f"  Classe {c} ({CLASS_NAMES[c]:<10}): {len(chosen):,} / {len(cls_idx):,}")

    selected_idx_sorted = np.sort(np.concatenate(selected_idx))
    total = len(selected_idx_sorted)
    print(f"\n  Lettura di {total:,} finestre dall'H5 (a chunk da {CHUNK_SIZE})...")

    # Lettura a chunk con barra di progresso
    X_sub = np.empty((total, 12, SAMPLES_PER_WINDOW), dtype=np.float32)
    with h5py.File(h5_orig_path, 'r') as f:
        x_dset = f['X']
        n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
        bar = tqdm(range(n_chunks), desc="  Lettura H5", unit="chunk",
                   bar_format="{l_bar}{bar:30}{r_bar}")
        for ci in bar:
            lo = ci * CHUNK_SIZE
            hi = min(lo + CHUNK_SIZE, total)
            idx_chunk = selected_idx_sorted[lo:hi]
            X_sub[lo:hi] = x_dset[idx_chunk]
            bar.set_postfix(finestre=f"{hi:,}/{total:,}")

    Y_sub = y_all[selected_idx_sorted]

    perm = np.random.permutation(total)
    print(f"  Shuffle completato.")
    return X_sub[perm], Y_sub[perm]


# ---------------------------------------------------------------------------
# Step 2 — Raccolta finestre pseudo-labeled
# ---------------------------------------------------------------------------

def collect_pseudo_windows(pseudolabels: list):
    print(f"\n[2/3] Estrazione finestre dagli ECG pseudo-labeled ({len(pseudolabels)} ECG)...")
    all_windows = []
    all_labels  = []
    skipped     = 0
    class_wins  = {name: 0 for name in CLASS_NAMES}

    bar = tqdm(pseudolabels, desc="  Estrazione EDF", unit="ecg",
               bar_format="{l_bar}{bar:30}{r_bar}")
    for entry in bar:
        label    = entry['predicted_class']
        # if label == 0:  # Salta la classe normale
        #     continue
        ecg_id   = entry['id']
        zip_path = entry['zip_path']

        windows = extract_windows_from_edf(ecg_id, zip_path)
        if windows is None:
            skipped += 1
            bar.set_postfix(ok=len(pseudolabels)-skipped-len([e for e in pseudolabels if e==entry]),
                            skip=skipped)
            continue

        all_windows.append(windows)
        all_labels.extend([label] * windows.shape[0])
        class_wins[CLASS_NAMES[label]] += windows.shape[0]
        bar.set_postfix(finestre=sum(class_wins.values()), skip=skipped)

    print(f"\n  ECG processati: {len(pseudolabels) - skipped}  |  scartati (SQA): {skipped}")
    total_wins = sum(class_wins.values())
    print(f"  Finestre raw estratte: {total_wins:,}")
    for name, cnt in class_wins.items():
        if cnt > 0:
            print(f"    {name:<15}: {cnt:,}")

    if not all_windows:
        return np.empty((0, 12, SAMPLES_PER_WINDOW), dtype=np.float32), np.array([], dtype=np.int8)

    X_pseudo = np.concatenate(all_windows, axis=0)
    Y_pseudo = np.array(all_labels, dtype=np.int8)
    return X_pseudo, Y_pseudo

def augment_ecg_windows(X_batch, noise_level=0.03, scale_range=(0.90, 1.10)):
    """
    Applica Data Augmentation a un batch di finestre ECG: (N, leads, samples).
    Aggiunge rumore gaussiano e variazione di gain casuale.
    """
    X_aug = X_batch.copy()
    
    # 1. Rumore Gaussiano (simula interferenze)
    noise = np.random.normal(0, noise_level, X_aug.shape)
    X_aug += noise
    
    # 2. Random Scaling (simula sbalzi di ampiezza)
    scales = np.random.uniform(scale_range[0], scale_range[1], 
                               size=(X_aug.shape[0], X_aug.shape[1], 1))
    X_aug *= scales
    
    return X_aug.astype(np.float32)
# ---------------------------------------------------------------------------
# Upsampling
# ---------------------------------------------------------------------------

def upsample_pseudo(X_pseudo, Y_pseudo, n_orig_subset, target_ratio=TARGET_RATIO, num_classes=6, max_replication=15):
    n_pseudo_raw = len(Y_pseudo)
    if n_pseudo_raw == 0:
        print("  ATTENZIONE: nessuna finestra pseudo-labeled disponibile.")
        return X_pseudo, Y_pseudo

    target_pseudo = int(n_orig_subset * target_ratio / (1 - target_ratio))
    target_per_class = target_pseudo // num_classes

    print(f"\n  Bilanciamento e Upsampling per singola classe (con limite replication max {max_replication}x):")
    print(f"    Target teorico totale:               {target_pseudo:,}")
    print(f"    Target teorico per singola classe:   {target_per_class:,}")

    X_balanced_list = []
    Y_balanced_list = []

    for c in range(num_classes):
        class_indices = np.where(Y_pseudo == c)[0]
        n_class = len(class_indices)
        class_name = CLASS_NAMES[c]

        if n_class == 0:
            print(f"    Classe {c} ({class_name:<10}): 0 finestre. Saltata.")
            continue

        class_target = min(target_per_class, n_class * max_replication)
        upsample_factor = class_target // n_class
        remainder = class_target % n_class

        if upsample_factor > 0:
            # ---> 1. La prima copia resta pulita e originale <---
            X_class_list = [X_pseudo[class_indices]]
            Y_class_list = [Y_pseudo[class_indices]]
            
            # ---> 2. Le successive (upsample_factor - 1) sono aumentate <---
            for _ in range(upsample_factor - 1):
                X_aug = augment_ecg_windows(X_pseudo[class_indices])
                X_class_list.append(X_aug)
                Y_class_list.append(Y_pseudo[class_indices])
                
            X_class = np.concatenate(X_class_list, axis=0)
            Y_class = np.concatenate(Y_class_list, axis=0)
        else:
            X_class = np.empty((0, 12, SAMPLES_PER_WINDOW), dtype=np.float32)
            Y_class = np.empty((0,), dtype=np.int8)

        if remainder > 0:
            np.random.seed(42 + c)
            chosen_remainder_idx = np.random.choice(class_indices, size=remainder, replace=(remainder > n_class))
            X_rem_base = X_pseudo[chosen_remainder_idx]
            
            # ---> 3. Applichiamo augmentation anche ai resti <---
            X_rem = augment_ecg_windows(X_rem_base)
            Y_rem = Y_pseudo[chosen_remainder_idx]
            
            X_class = np.concatenate([X_class, X_rem], axis=0) if len(X_class) > 0 else X_rem
            Y_class = np.concatenate([Y_class, Y_rem], axis=0) if len(Y_class) > 0 else Y_rem

        print(f"    Classe {c} ({class_name:<10}): da {n_class:>5} a {len(Y_class):>5} finestre (moltiplicatore ~{len(Y_class)/n_class:.1f}x)")
        X_balanced_list.append(X_class)
        Y_balanced_list.append(Y_class)

    X_up = np.concatenate(X_balanced_list, axis=0)
    Y_up = np.concatenate(Y_balanced_list, axis=0)

    perm = np.random.permutation(len(Y_up))
    X_up = X_up[perm]
    Y_up = Y_up[perm]

    actual_pct = 100 * len(Y_up) / (n_orig_subset + len(Y_up))
    print(f"  Finestre pseudo-labeled finali:    {len(Y_up):,}  ({actual_pct:.1f}% del totale)")
    return X_up, Y_up


# ---------------------------------------------------------------------------
# Scrittura H5 finale con progress bar
# ---------------------------------------------------------------------------

def write_ssl_h5(X_orig, Y_orig, X_pseudo, Y_pseudo, out_path):
    print(f"\n[3/3] Scrittura del dataset SSL...")
    if os.path.exists(out_path):
        os.remove(out_path)

    print("  Concatenazione e shuffle...")
    X_all = np.concatenate([X_orig, X_pseudo], axis=0)
    Y_all = np.concatenate([Y_orig, Y_pseudo], axis=0).astype(np.int8)
    total = len(Y_all)

    np.random.seed(42)
    perm  = np.random.permutation(total)
    X_all = X_all[perm]
    Y_all = Y_all[perm]

    print(f"  Scrittura di {total:,} finestre in: {out_path}")
    with h5py.File(out_path, 'w') as f:
        dset_x = f.create_dataset(
            'X', shape=(total, 12, SAMPLES_PER_WINDOW),
            dtype='float32', chunks=(64, 12, SAMPLES_PER_WINDOW), compression='lzf'
        )
        dset_y = f.create_dataset('Y', shape=(total,), dtype='int8')

        n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
        bar = tqdm(range(n_chunks), desc="  Scrittura H5", unit="chunk",
                   bar_format="{l_bar}{bar:30}{r_bar}")
        for ci in bar:
            lo = ci * CHUNK_SIZE
            hi = min(lo + CHUNK_SIZE, total)
            dset_x[lo:hi] = X_all[lo:hi]
            dset_y[lo:hi] = Y_all[lo:hi]
            bar.set_postfix(finestre=f"{hi:,}/{total:,}")

    print(f"\n  Totale finestre scritte: {total:,}")
    print("  Distribuzione finale per classe:")
    for c, name in enumerate(CLASS_NAMES):
        n = int(np.sum(Y_all == c))
        pct = 100 * n / total
        print(f"    {name:<15}: {n:>7,}  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("STEP 3 - Costruzione Training Set SSL (Opzione C)")
    print("=" * 60)
    print(f"  Training originale:  {ORIG_TRAIN}")
    print(f"  Output SSL:          {SSL_TRAIN}")
    print(f"  Sintetico subset:    {N_ORIG_SUBSET:,} finestre ({N_PER_CLASS_ORIG:,}/classe)")
    print(f"  Target ratio reale:  ~{TARGET_RATIO:.0%}")

    pseudolabels_path = os.path.join(RESULTS_DIR, 'pseudolabels.json')
    if not os.path.exists(pseudolabels_path):
        print(f"\nERRORE: {pseudolabels_path} non trovato. Esegui prima step2.")
        sys.exit(1)

    with open(pseudolabels_path, 'r', encoding='utf-8') as f:
        pseudolabels = json.load(f)
    print(f"\nPseudo-labels caricati: {len(pseudolabels)} ECG")

    # 1. Subsample sintetico
    X_orig, Y_orig = subsample_original_train(ORIG_TRAIN)

    # 2. Finestre pseudo-labeled
    X_pseudo_raw, Y_pseudo_raw = collect_pseudo_windows(pseudolabels)

    # 3. Upsampling
    X_pseudo, Y_pseudo = upsample_pseudo(X_pseudo_raw, Y_pseudo_raw, n_orig_subset=N_ORIG_SUBSET)

    # 4. Scrittura H5
    write_ssl_h5(X_orig, Y_orig, X_pseudo, Y_pseudo, SSL_TRAIN)

    # Report
    report_path = os.path.join(RESULTS_DIR, 'ssl_stats_report.txt')
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(f"\n\n{'=' * 60}\n")
        f.write("STEP 3 - Dataset SSL costruito\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Training originale subsampled: {len(Y_orig):,} finestre\n")
        f.write(f"Pseudo-labeled upsampled:      {len(Y_pseudo):,} finestre\n")
        f.write(f"Totale SSL:                    {len(Y_orig) + len(Y_pseudo):,} finestre\n")
        f.write(f"Output: {SSL_TRAIN}\n")

    print(f"\nOK Training set SSL completato.")
    print(f"   Riaddestra con train_limbs.py usando:")
    print(f"   {SSL_TRAIN}")
