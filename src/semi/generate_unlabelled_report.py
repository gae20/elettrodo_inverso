"""
generate_unlabelled_report.py

Genera un report CSV dettagliato per i tracciati NON ETICHETTATI con
status = 'rejected' nel database clinico:
1. Interroga records.db per trovare TUTTI gli ID con status='rejected'.
2. Cerca i corrispondenti file EDF in TUTTI i ZIP disponibili.
3. Per ciascun tracciato valido:
   - Mostra il referto clinico originale.
   - Applica SQA ed estrae le finestre.
   - Esegue l'inferenza con il modello base.
   - Calcola classe predetta media e confidenza.
4. Salva tutto in un report CSV (leggibile su Excel).
"""

import os
import sys
import zipfile
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
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
    SAMPLES_PER_WINDOW, ALL_LEADS, QUALITY_CFG, STRIDE_SAMPLES,
)

# --- Percorsi ---
DATASET_DIR  = os.path.join(THESIS_DIR, 'datasets', 'dataset')
DB_PATH      = os.path.join(DATASET_DIR, 'records.db')
RESULTS_DIR  = os.path.join(SCRIPT_DIR, 'results')
REPORT_CSV   = os.path.join(RESULTS_DIR, 'inversioni_report.csv')

# ZIP da analizzare (in ordine di priorità)
ALL_ZIPS = [
    os.path.join(DATASET_DIR, 'DATASET_complete', f'dataset_batch_{i}.zip')
    for i in range(1, 6)
] + [
    os.path.join(DATASET_DIR, 'dataset_inverted_real_unlabelled.zip'),
    os.path.join(DATASET_DIR, 'dataset_self.zip'),
]

# --- Costanti ---
CLASS_NAMES  = ['normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']
LIMB_INDICES = list(range(6))
LIMB_LEADS   = ALL_LEADS[:6]
MAX_RECORDS  = 1000
# File con gli ID degli ECG rejected (generato da export_rejected_ids.py)
REJECTED_IDS_FILE = os.path.join(SCRIPT_DIR, 'rejected_ids.txt')
REJECTED_DATASET_DIR = os.path.join(SCRIPT_DIR, 'rejected_dataset')


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
    win_size = int(cfg["win_sec"] * cfg["fs"])
    _, n_samples = sigs_array.shape
    if n_samples < win_size:
        return np.zeros(0, dtype=bool)
    starts = list(range(0, n_samples - win_size + 1, STRIDE_SAMPLES))
    mask   = np.zeros(len(starts), dtype=bool)
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
    starts  = list(range(0, n_samples - SAMPLES_PER_WINDOW + 1, STRIDE_SAMPLES))
    windows = [sigs_array[:, s:s + SAMPLES_PER_WINDOW] for s in starts]
    return np.array(windows, dtype=np.float32)


def extract_valid_windows(edf_bytes):
    try:
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
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 65)
    print("REPORT PREDIZIONI — ECG CON status='rejected'")
    print("=" * 65)

    # 1. Leggi gli ID rejected dal file txt
    print(f"\n[1] Lettura ID da {os.path.basename(REJECTED_IDS_FILE)}...")
    if not os.path.exists(REJECTED_IDS_FILE):
        print(f"ERRORE: {REJECTED_IDS_FILE} non trovato.")
        sys.exit(1)
    with open(REJECTED_IDS_FILE, 'r') as f:
        rejected_ids = set(int(line.strip()) for line in f if line.strip())
    print(f"  ID rejected caricati: {len(rejected_ids):,}")

    # Recupera i referti dal DB per tutti questi ID
    print(f"\n[1b] Recupero referti da records.db...")
    if not os.path.exists(DB_PATH):
        print(f"ERRORE: records.db non trovato in {DB_PATH}")
        sys.exit(1)
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Query a blocchi per i soli ID che ci interessano
    rejected_list = list(rejected_ids)
    referto_map   = {}
    for i in range(0, len(rejected_list), 500):
        chunk = rejected_list[i:i+500]
        placeholders = ','.join('?' for _ in chunk)
        cursor.execute(f"SELECT id, text FROM records WHERE id IN ({placeholders})", chunk)
        for rid, text in cursor.fetchall():
            referto_map[rid] = (text or '').strip().replace('\n', ' | ').replace('\r', '')
    conn.close()
    print(f"  Referti recuperati: {len(referto_map):,}")

    # 2. Indicizzazione dei file EDF in tutti i ZIP disponibili e nella cartella rejected_dataset
    print(f"\n[2] Indicizzazione dei file EDF in tutti i ZIP disponibili e nella cartella rejected_dataset...")
    edf_index = {}  # id -> {'type': 'zip'/'file', ...}

    # 1. Scansiona prima la cartella rejected_dataset se esiste (ha priorità)
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

    # 2. Scansiona gli altri ZIP tradizionali (se non già indicizzati)
    for zp in ALL_ZIPS:
        if not os.path.exists(zp):
            print(f"  [skip] {os.path.basename(zp)} — non trovato")
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

    # 3. Intersezione: ID con 'inversione' nel referto e EDF disponibile
    matching_ids = sorted(rejected_ids & set(edf_index.keys()))
    print(f"\n[3] Tracciati rejected con EDF disponibile: {len(matching_ids):,}")

    if len(matching_ids) == 0:
        print("Nessun record trovato. Impossibile generare il report.")
        sys.exit(0)

    # 4. Carica il modello
    base_weights = os.path.join(RESULTS_DIR, 'model_base.weights.h5')
    if not os.path.exists(base_weights):
        print(f"\nERRORE: Modello base non trovato ({base_weights}).")
        sys.exit(1)

    print(f"\n[4] Caricamento modello base...")
    model = build_model((500, 6), 6)
    model.load_weights(base_weights)

    # 5. Inferenza e costruzione report
    print(f"\n[5] Inferenza su {MAX_RECORDS} tracciati validi rejected...")
    records_data = []

    current_idx = 0
    successful_count = 0

    pbar = tqdm(total=MAX_RECORDS, desc="Inferenza")
    while successful_count < MAX_RECORDS and current_idx < len(matching_ids):
        rid = matching_ids[current_idx]
        current_idx += 1

        referto = referto_map[rid]
        info    = edf_index[rid]

        valid_windows = None
        status = 'Elaborato con successo'

        try:
            if info['type'] == 'zip':
                with zipfile.ZipFile(info['zip_path'], 'r') as z:
                    edf_bytes = z.read(info['edf_name'])
            else:
                with open(info['file_path'], 'rb') as f:
                    edf_bytes = f.read()
            valid_windows = extract_valid_windows(edf_bytes)
        except Exception as e:
            status = f'Errore lettura EDF: {e}'

        if valid_windows is None and status == 'Elaborato con successo':
            status = 'Scartato: qualità SQA insufficiente'

        if status != 'Elaborato con successo':
            records_data.append({
                'Record_ID':             rid,
                'Referto_Clinico':       referto,
                'Stato':                 status,
                'Classe_Predetta':       'N/A',
                'Confidenza':            None,
                'Prob_normale':          None,
                'Prob_LA-RA':            None,
                'Prob_RA-LL':            None,
                'Prob_LA-LL':            None,
                'Prob_ROT_ORA':          None,
                'Prob_ROT_ANT':          None,
                'Finestre_Valide':       0,
            })
            continue

        # Inferenza
        X_input  = np.transpose(valid_windows, (0, 2, 1))   # (N, 500, 6)
        y_probs  = model.predict(X_input, batch_size=64, verbose=0)
        mean_p   = np.mean(y_probs, axis=0)
        pred_idx = int(np.argmax(mean_p))
        conf     = float(mean_p[pred_idx])

        records_data.append({
            'Record_ID':       rid,
            'Referto_Clinico': referto,
            'Stato':           status,
            'Classe_Predetta': CLASS_NAMES[pred_idx],
            'Confidenza':      round(conf, 4),
            'Prob_normale':    round(float(mean_p[0]), 4),
            'Prob_LA-RA':      round(float(mean_p[1]), 4),
            'Prob_RA-LL':      round(float(mean_p[2]), 4),
            'Prob_LA-LL':      round(float(mean_p[3]), 4),
            'Prob_ROT_ORA':    round(float(mean_p[4]), 4),
            'Prob_ROT_ANT':    round(float(mean_p[5]), 4),
            'Finestre_Valide': X_input.shape[0],
        })
        successful_count += 1
        pbar.update(1)

    pbar.close()

    # 6. Salva CSV
    df = pd.DataFrame(records_data)
    df.to_csv(REPORT_CSV, index=False, sep=';', encoding='utf-8-sig')

    ok_mask = df['Stato'] == 'Elaborato con successo'
    print(f"\n{'=' * 65}")
    print(f"Report salvato in: {REPORT_CSV}")
    print(f"  Tracciati totali nel report : {len(df)}")
    print(f"  Elaborati con successo      : {ok_mask.sum()}")
    print(f"  Scartati da SQA/errori      : {(~ok_mask).sum()}")

    if ok_mask.sum() > 0:
        print(f"\n  Distribuzione classi predette:")
        for cls, cnt in df[ok_mask]['Classe_Predetta'].value_counts().items():
            pct = cnt / ok_mask.sum() * 100
            print(f"    - {cls:<12}: {cnt:>4}  ({pct:.1f}%)")
