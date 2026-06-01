"""
train_semi.py

Esegue l'addestramento semi-supervisionato (Iterative Self-Training) sui dati reali:
1. Carica train_small_init.h5 e val_small_init.h5.
2. Inizializza il modello caricando i pesi del modello base (model_base.weights.h5).
3. Indicizza i tracciati non etichettati con status 'rejected' da rejected_dataset.
4. Per ogni iterazione di addestramento:
   - Seleziona un blocco di 1000 ECG validi non ancora usati (SQA superata).
   - Estrae le finestre e predice le classi usando il modello corrente.
   - Filtra per soglia di confidenza (specifica per classe) e applica capping per bilanciare (max 500 per classe).
   - Aggiunge le finestre pseudo-etichettate al training set.
   - Esegue il bilanciamento in memoria ed il ri-addestramento da zero (Retrain).
   - Salva i nuovi pesi come results/model_iter_{k}.weights.h5.
"""

import os
import sys
import h5py
import json
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm

# --- Setup path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.join(SCRIPT_DIR, '..')
THESIS_DIR = os.path.join(SRC_DIR, '..')

sys.path.append(SRC_DIR)
from models.ldensenet import build_model
from data.data_pipeline import (
    read_edf_data, all_leads_preprocessing,
    check_ecg_quality, check_window_quality,
)
from utils.config import (
    SAMPLES_PER_WINDOW, FS_NEW, ALL_LEADS, QUALITY_CFG, STRIDE_SAMPLES,
)

# --- Percorsi ---
DATASET_DIR    = os.path.join(THESIS_DIR, 'datasets', 'dataset')
H5_DIR         = os.path.join(SCRIPT_DIR, 'results', 'semi_h5')
RESULTS_DIR    = os.path.join(SCRIPT_DIR, 'results')

# Tutti i ZIP con ECG unlabelled da indicizzare
ALL_ZIPS = [
    os.path.join(DATASET_DIR, 'DATASET_complete', f'dataset_batch_{i}.zip')
    for i in range(1, 6)
] + [
    os.path.join(DATASET_DIR, 'dataset_inverted_real_unlabelled.zip'),
    os.path.join(DATASET_DIR, 'dataset_self.zip'),
]

TRAIN_INIT_H5  = os.path.join(H5_DIR, 'train_small_init.h5')
VAL_INIT_H5    = os.path.join(H5_DIR, 'val_small_init.h5')

# File con gli ID degli ECG rejected
REJECTED_IDS_FILE = os.path.join(SCRIPT_DIR, 'rejected_ids.txt')
REJECTED_DATASET_DIR = os.path.join(SCRIPT_DIR, 'rejected_dataset')

# --- Parametri ---
CLASS_NAMES = ['normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']
LIMB_INDICES = list(range(6))
LIMB_LEADS   = ALL_LEADS[:6]

# Parametri esperimento
NUM_ITERATIONS = 1
BATCH_ECG_SIZE = 200
CONFIDENCE_THRS = {
    0: 0.95,  # normale
    1: 0.95,  # LA-RA
    2: 0.98,  # RA-LL
    3: 0.99,  # LA-LL
    4: 0.98,  # ROT_ORA
    5: 0.95   # ROT_ANT
}
MAX_PSEUDO_PATIENTS_PER_CLASS = 4

# Parametri training keras
EP = 40
LR = 1e-3
BS = 256

# ---------------------------------------------------------------------------
# Preprocessing unlabelled
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
    stride   = int(cfg["stride_sec"] * fs)
    _, n_samples = sigs_array.shape
    if n_samples < win_size:
        return np.zeros(0, dtype=bool)
    starts  = list(range(0, n_samples - win_size + 1, stride))
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

def process_unlabelled_bytes(edf_bytes):
    try:
        ecg_data = read_edf_data(edf_bytes)
    except Exception:
        return None

    if not ecg_data or not ecg_data.get("signals"):
        return None
    if not all(l in ecg_data["signals"] for l in LIMB_LEADS):
        return None

    sigs = all_leads_preprocessing(ecg_data["signals"])
    sigs_raw = np.array([sigs[l] for l in LIMB_LEADS], dtype=np.float32)

    quality = check_ecg_quality(sigs_raw, cfg=QUALITY_CFG, lead_indices=LIMB_INDICES)
    if not quality['global_valid']:
        return None

    win_mask  = compute_good_window_mask(sigs_raw, cfg=QUALITY_CFG, min_valid_leads=5)
    sigs_norm = robust_scale_ecg(sigs_raw)
    windows   = create_windows(sigs_norm)

    n    = min(windows.shape[0], win_mask.size)
    good = windows[:n][win_mask[:n]]
    return good if good.shape[0] > 0 else None

# ---------------------------------------------------------------------------
# Caricamento e Bilanciamento
# ---------------------------------------------------------------------------

def load_h5_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
    X_transposed = np.transpose(X, (0, 2, 1))
    return X_transposed, Y

def balance_dataset(X, Y):
    class_counts = [np.sum(Y == c) for c in range(6)]
    max_count = max(class_counts)
    
    print(f"\n  [Balancing] Bilanciamento training set in corso (target: {max_count:,} finestre per classe)...")
    
    X_balanced = []
    Y_balanced = []
    
    for c in range(6):
        idx = np.where(Y == c)[0]
        n_class = len(idx)
        if n_class == 0:
            continue
            
        factor = max_count // n_class
        remainder = max_count % n_class
        
        X_c = np.tile(X[idx], (factor, 1, 1))
        Y_c = np.tile(Y[idx], factor)
        
        if remainder > 0:
            np.random.seed(42 + c)
            chosen = np.random.choice(idx, size=remainder, replace=(remainder > n_class))
            X_c = np.concatenate([X_c, X[chosen]], axis=0)
            Y_c = np.concatenate([Y_c, Y[chosen]], axis=0)
            
        X_balanced.append(X_c)
        Y_balanced.append(Y_c)
        
    X_bal = np.concatenate(X_balanced, axis=0)
    Y_bal = np.concatenate(Y_balanced, axis=0)
    
    perm = np.random.permutation(len(Y_bal))
    return X_bal[perm], Y_bal[perm]

if __name__ == '__main__':
    # Imposta seed globale per coerenza e determinismo
    tf.keras.utils.set_random_seed(42)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU disponibili: {len(gpus)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("INIZIO ADDESTRAMENTO SEMI-SUPERVISIONATO (SELF-TRAINING)")
    print("=" * 60)

    # 1. Carica i dati iniziali
    print("\n[Caricamento dati di partenza...]")
    if not (os.path.exists(TRAIN_INIT_H5) and os.path.exists(VAL_INIT_H5)):
        print("ERRORE: I file H5 iniziali non sono pronti. Esegui prima prepare_datasets.py.")
        sys.exit(1)

    X_train_curr, Y_train_curr = load_h5_data(TRAIN_INIT_H5)
    X_val, Y_val               = load_h5_data(VAL_INIT_H5)
    Y_val_oh                   = to_categorical(Y_val, 6)

    print(f"  Train iniziale : {X_train_curr.shape[0]} finestre")
    print(f"  Val iniziale   : {X_val.shape[0]} finestre")

    # 2. Carica modello base pre-esistente
    base_weights_path = os.path.join(RESULTS_DIR, 'model_base.weights.h5')
    if not os.path.exists(base_weights_path):
        print(f"ERRORE: I pesi base {base_weights_path} non esistono. Esegui prima train_base.py.")
        sys.exit(1)

    print("\n[Inizializzazione modello base...]")
    model = build_model((500, 6), 6)
    model.load_weights(base_weights_path)
    print("  Modello base caricato correttamente.")

    # Registro storico delle iterazioni
    history_report = []

    # 3. Indicizzazione pool unlabelled 'rejected'
    print("\n[Costruzione pool unlabelled da rejected_ids.txt...]")
    if not os.path.exists(REJECTED_IDS_FILE):
        print(f"ERRORE: {REJECTED_IDS_FILE} non trovato.")
        sys.exit(1)
    with open(REJECTED_IDS_FILE, 'r') as f:
        rejected_ids = set(int(line.strip()) for line in f if line.strip())
    print(f"  ID rejected caricati da file: {len(rejected_ids):,}")

    edf_index = {}

    # Scansione rejected_dataset
    if os.path.exists(REJECTED_DATASET_DIR):
        print(f"  Scansione cartella rejected_dataset: {REJECTED_DATASET_DIR}")
        for root, dirs, files in os.walk(REJECTED_DATASET_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.zip'):
                    count = 0
                    try:
                        with zipfile.ZipFile(file_path, 'r') as z:
                            for name in z.namelist():
                                if not name.endswith('.edf'):
                                    continue
                                try:
                                    rid = int(os.path.splitext(os.path.basename(name))[0])
                                except ValueError:
                                    continue
                                if rid not in edf_index:
                                    edf_index[rid] = {
                                        'type': 'zip',
                                        'zip_path': file_path,
                                        'edf_name': name
                                    }
                                    count += 1
                        print(f"    [zip] {file}: {count:,} EDF indicizzati")
                    except Exception as e:
                        print(f"    [errore zip] {file}: {e}")
                elif file.endswith('.edf'):
                    try:
                        rid = int(os.path.splitext(file)[0])
                        if rid not in edf_index:
                            edf_index[rid] = {
                                'type': 'file',
                                'file_path': file_path
                            }
                    except ValueError:
                        continue

    # Scansione ZIP tradizionali
    for zp in ALL_ZIPS:
        if not os.path.exists(zp):
            continue
        count = 0
        try:
            with zipfile.ZipFile(zp, 'r') as z:
                for name in z.namelist():
                    if not name.endswith('.edf'):
                        continue
                    try:
                        rid = int(os.path.splitext(os.path.basename(name))[0])
                    except ValueError:
                        continue
                    if rid not in edf_index:
                        edf_index[rid] = {
                            'type': 'zip',
                            'zip_path': zp,
                            'edf_name': name
                        }
                        count += 1
            print(f"  {os.path.basename(zp)}: {count:,} EDF indicizzati")
        except Exception as e:
            print(f"  [ERRORE] {os.path.basename(zp)}: {e}")
            continue
    print(f"  Totale EDF unici indicizzati: {len(edf_index):,}")

    matching_ids = sorted(rejected_ids & set(edf_index.keys()))
    print(f"  Pool finale (rejected + EDF disponibile): {len(matching_ids):,} tracciati")

    if len(matching_ids) == 0:
        print("ERRORE: Nessun tracciato trovato nel pool.")
        sys.exit(1)

    # 4. Loop di Self-Training
    used_ids = set()
    current_pool_idx = 0

    for k in range(1, NUM_ITERATIONS + 1):
        print("\n" + "=" * 60)
        print(f"ITERAZIONE {k} / {NUM_ITERATIONS} — Pseudo-Labeling ed Espansione")
        print("=" * 60)

        X_cand_list = []
        sampled_ids = []
        sampled_records_info = []
        current_idx = 0

        pbar = tqdm(total=BATCH_ECG_SIZE, desc=f"Ricerca {BATCH_ECG_SIZE} ECG validi (SQA ok)")
        while len(sampled_ids) < BATCH_ECG_SIZE and current_pool_idx < len(matching_ids):
            rid = matching_ids[current_pool_idx]
            current_pool_idx += 1

            if rid in used_ids:
                continue

            item = edf_index[rid]
            try:
                if item['type'] == 'zip':
                    with zipfile.ZipFile(item['zip_path'], 'r') as z:
                        edf_bytes = z.read(item['edf_name'])
                else:
                    with open(item['file_path'], 'rb') as f:
                        edf_bytes = f.read()
                wins = process_unlabelled_bytes(edf_bytes)
                if wins is not None:
                    wins_t = np.transpose(wins, (0, 2, 1))
                    num_wins = wins_t.shape[0]
                    X_cand_list.append(wins_t)
                    sampled_ids.append(rid)
                    used_ids.add(rid)
                    
                    sampled_records_info.append({
                        'rid': rid,
                        'start_idx': current_idx,
                        'end_idx': current_idx + num_wins
                    })
                    current_idx += num_wins
                    pbar.update(1)
            except Exception:
                continue
        pbar.close()

        print(f"  Scansionati tracciati nel pool fino all'indice {current_pool_idx}/{len(matching_ids)}")
        print(f"  Trovati ed elaborati con successo {len(sampled_ids)} ECG validi")

        if not X_cand_list:
            print("  Nessuna finestra valida estratta. Salto iterazione.")
            continue

        X_cand = np.concatenate(X_cand_list, axis=0)
        print(f"  Totale finestre candidate estratte: {X_cand.shape[0]:,}")

        # Inferenza
        print("  Inferenza con modello corrente...")
        y_probs = model.predict(X_cand, batch_size=256, verbose=1)

        # Consenso a livello di paziente
        candidates_by_class = {c: [] for c in range(6)}
        for rec in sampled_records_info:
            start_idx = rec['start_idx']
            end_idx = rec['end_idx']
            probs_rec = y_probs[start_idx:end_idx]
            
            # Media delle probabilità per questo paziente/tracciato
            mean_prob = np.mean(probs_rec, axis=0)
            pred_class = np.argmax(mean_prob)
            conf = mean_prob[pred_class]
            
            # Se supera la soglia della classe, viene aggiunto ai candidati
            if conf >= CONFIDENCE_THRS[pred_class]:
                X_rec = X_cand[start_idx:end_idx]
                Y_rec = np.full((X_rec.shape[0],), pred_class, dtype=np.int8)
                candidates_by_class[pred_class].append({
                    'rid': rec['rid'],
                    'X': X_rec,
                    'Y': Y_rec,
                    'confidence': conf
                })

        # Distribuzione candidati
        print("\n  Distribuzione candidati a livello di paziente (consenso ad alta confidenza):")
        for c in range(6):
            n_pats = len(candidates_by_class[c])
            print(f"    - Class {c} ({CLASS_NAMES[c]:<12}): {n_pats} pazienti")

        # Bilanciamento (Capping a livello di paziente)
        X_pseudo_balanced = []
        Y_pseudo_balanced = []
        print(f"\n  Bilanciamento pseudo-labels (cap max {MAX_PSEUDO_PATIENTS_PER_CLASS} pazienti per classe):")
        
        for c in range(6):
            pats = candidates_by_class[c]
            n_pats = len(pats)
            if n_pats == 0:
                continue
                
            # Ordina i pazienti per confidenza decrescente per selezionare i migliori
            pats_sorted = sorted(pats, key=lambda x: x['confidence'], reverse=True)
            
            if n_pats > MAX_PSEUDO_PATIENTS_PER_CLASS:
                selected_pats = pats_sorted[:MAX_PSEUDO_PATIENTS_PER_CLASS]
                print(f"    - Class {c} ({CLASS_NAMES[c]:<12}): {n_pats} pazienti -> ridotti a {MAX_PSEUDO_PATIENTS_PER_CLASS} (top confidenza)")
            else:
                selected_pats = pats_sorted
                print(f"    - Class {c} ({CLASS_NAMES[c]:<12}): {n_pats} pazienti -> tenuti tutti")
                
            for p in selected_pats:
                X_pseudo_balanced.append(p['X'])
                Y_pseudo_balanced.extend(p['Y'])

        if not X_pseudo_balanced:
            print("  Nessun paziente ha superato la soglia con consenso. Salto iterazione.")
            continue

        X_pseudo_balanced = np.concatenate(X_pseudo_balanced, axis=0)
        Y_pseudo_balanced = np.array(Y_pseudo_balanced, dtype=np.int8)
        print(f"  -> Finestre pseudo-labeled finali aggiunte: {len(Y_pseudo_balanced):,}")

        # Unione ed espansione
        X_train_curr = np.concatenate([X_train_curr, X_pseudo_balanced], axis=0)
        Y_train_curr = np.concatenate([Y_train_curr, Y_pseudo_balanced], axis=0)

        # Shuffle
        perm = np.random.permutation(len(Y_train_curr))
        X_train_curr = X_train_curr[perm]
        Y_train_curr = Y_train_curr[perm]

        print(f"  Nuova dimensione del Training Set: {X_train_curr.shape[0]:,} finestre")

        # Ri-addestramento da zero
        print(f"\n  Ri-addestramento del modello da zero (Iterazione {k})...")
        model = build_model((500, 6), 6)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=LR),
            metrics=['accuracy', tf.keras.metrics.F1Score(average='macro', name='f1_score')]
        )

        iter_weights_path = os.path.join(RESULTS_DIR, f'model_iter_{k}.weights.h5')
        callbacks = [
            EarlyStopping(monitor='val_f1_score', patience=8, restore_best_weights=True, mode='max', verbose=1)
        ]

        X_train_bal, Y_train_bal = balance_dataset(X_train_curr, Y_train_curr)
        Y_train_oh = to_categorical(Y_train_bal, 6)

        model.fit(
            X_train_bal, Y_train_oh,
            batch_size=BS,
            epochs=EP,
            validation_data=(X_val, Y_val_oh),
            callbacks=callbacks,
            verbose=1
        )
        model.save_weights(iter_weights_path)
        print(f"  Pesi salvati correttamente in: {iter_weights_path}")

        history_report.append({
            'iteration': k,
            'added_windows': len(Y_pseudo_balanced),
            'total_train_windows': X_train_curr.shape[0]
        })

    # Salva report JSON
    report_path = os.path.join(RESULTS_DIR, 'semi_training_info.json')
    with open(report_path, 'w') as f:
        json.dump(history_report, f, indent=4)
    print(f"\nReport di addestramento semi-supervisionato salvato in: {report_path}")
    print("=" * 60)
