"""
step2_predict_and_pseudolabel.py

Per ogni ECG candidato (da candidate_ids.json):
1. Legge l'EDF dallo ZIP
2. Applica lo stesso preprocessing del test set reale (NO gain, NO noise)
3. Fa inferenza con il modello LDenseNet addestrato sui limb leads
4. Assegna una pseudo-label se la confidenza media (majority voting) >= 95%

Output: results/pseudolabels.json
"""

import os
import sys
import json
import copy
import zipfile
import numpy as np
from tqdm import tqdm

# --- Setup path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
THESIS_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))

sys.path.append(SRC_DIR)
from data.data_pipeline import (
    read_edf_data, all_leads_preprocessing,
    check_ecg_quality, check_window_quality,
)
from utils.config import (
    SAMPLES_PER_WINDOW, FS_NEW, ALL_LEADS,
    ACTIVE_SYNTH_CLASSES, MAPPING_INV, STRIDE_SAMPLES,
)
from utils.sqa_real_config import QUALITY_CFG_REAL

# --- Percorsi ---
RESULTS_DIR  = os.path.join(SCRIPT_DIR, 'results')
WEIGHTS_PATH = os.path.join(
    THESIS_DIR, 'src', 'prove', 'models',
    'best_model_final_noise_limbs.weights.h5'
)

# --- Costanti ---
LIMB_INDICES  = list(range(6))
CONFIDENCE_THRESHOLD = 0.95

# Label mapping: indice → nome classe
CLASS_NAMES = ['normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']

# Usa la configurazione dei reali (più tollerante, stride 0.5s)
QUALITY_CFG_SSL = QUALITY_CFG_REAL


def robust_scale_ecg(sigs_array, eps=1e-8, reference_leads=None):
    """
    Normalizzazione Robust Scaler.
    L'IQR viene calcolato SOLO sulle lead periferiche (reference_leads=LIMB_INDICES)
    per coerenza con il dataset di training e il test set reale.
    """
    x = sigs_array.astype(np.float32)
    medians = np.median(x, axis=1, keepdims=True)
    ref = x[reference_leads, :] if reference_leads is not None else x
    q75, q25 = np.percentile(ref, [75, 25])
    iqr_global = q75 - q25
    scale_global = max(iqr_global / 1.34896, eps)
    return (x - medians) / scale_global


def compute_good_window_mask(sigs_array, cfg, min_valid_leads=5, lead_indices=None):
    """Maschera booleana sulle finestre: True = finestra valida."""
    fs       = cfg["fs"]
    win_size = int(cfg["win_sec"] * fs)
    stride   = int(cfg["stride_sec"] * fs)
    _, n_samples = sigs_array.shape
    if n_samples < win_size:
        return np.zeros(0, dtype=bool)

    indices   = lead_indices if lead_indices is not None else list(range(sigs_array.shape[0]))
    starts    = list(range(0, n_samples - win_size + 1, stride))
    mask      = np.zeros(len(starts), dtype=bool)
    for w_idx, start in enumerate(starts):
        valid_count = sum(
            check_window_quality(sigs_array[li, start:start + win_size],
                                 cfg=cfg, lead_idx=li)["valid"]
            for li in indices
        )
        mask[w_idx] = (valid_count >= min_valid_leads)
    return mask


def create_windows(sigs_norm, win_size=SAMPLES_PER_WINDOW, stride=STRIDE_SAMPLES):
    """Crea finestre da un array (n_leads, n_samples). Restituisce (N, n_leads, win_size)."""
    n_leads, n_samples = sigs_norm.shape
    if n_samples < win_size:
        return np.empty((0, n_leads, win_size), dtype=np.float32)
    starts = range(0, n_samples - win_size + 1, stride)
    return np.array([sigs_norm[:, s:s + win_size] for s in starts], dtype=np.float32)


def preprocess_edf(edf_bytes):
    """
    Preprocessing identico al training data builder:
      read_edf → all_leads_preprocessing → SQA (su RAW) → robust_scale_ecg
    
    NON applica gain né noise.
    
    Restituisce (sigs_raw, sigs_norm) entrambi con shape (12, n_samples),
    oppure (None, None) se l'ECG non supera la SQA globale.
    La mask di qualità per-finestra va calcolata su sigs_raw.
    """
    ecg_data = read_edf_data(edf_bytes)
    if not ecg_data or not ecg_data.get("signals"):
        return None, None

    sigs = all_leads_preprocessing(ecg_data["signals"])

    if not all(l in sigs for l in ALL_LEADS):
        return None, None

    sigs_raw = np.array([sigs[l] for l in ALL_LEADS], dtype=np.float32)

    # SQA globale sulle lead periferiche (su segnale RAW, come nel training builder)
    quality = check_ecg_quality(sigs_raw, cfg=QUALITY_CFG_SSL, lead_indices=LIMB_INDICES)
    if not quality['global_valid']:
        return None, None

    # Normalizzazione: IQR calcolato sulle sole lead periferiche
    sigs_norm = robust_scale_ecg(sigs_raw, reference_leads=LIMB_INDICES)
    return sigs_raw, sigs_norm


def predict_ecg(model, sigs_raw: np.ndarray, sigs_norm: np.ndarray):
    """
    Majority voting a livello ECG.
    - win_mask calcolata su sigs_RAW (soglie SQA in uV, come nel training builder)
    - Finestre estratte da sigs_NORM (normalizzato)
    """
    # Window mask sul segnale RAW (soglie in uV)
    win_mask = compute_good_window_mask(
        sigs_raw, cfg=QUALITY_CFG_SSL,
        min_valid_leads=5, lead_indices=LIMB_INDICES
    )

    # Finestre estratte dal segnale normalizzato
    windows_all = create_windows(sigs_norm, stride=STRIDE_SAMPLES)  # (N, 12, win)
    n_win = min(windows_all.shape[0], win_mask.size)
    if n_win == 0:
        return None, 0.0, 0

    windows_good = windows_all[:n_win][win_mask[:n_win]]
    if windows_good.shape[0] == 0:
        return None, 0.0, 0

    # Il modello usa solo le prime 6 lead (limb), con shape (N, win_size, 6)
    x_limbs = windows_good[:, :6, :]                     # (N, 6, win_size)
    x_input = np.transpose(x_limbs, (0, 2, 1))           # (N, win_size, 6)

    y_probs = model.predict(x_input, batch_size=64, verbose=0)  # (N, 6)
    avg_probs = y_probs.mean(axis=0)                            # (6,)
    predicted_class = int(np.argmax(avg_probs))
    confidence = float(avg_probs[predicted_class])

    return predicted_class, confidence, int(windows_good.shape[0])


def load_model():
    """Costruisce e carica i pesi del modello LDenseNet per i limb leads."""
    import tensorflow as tf
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        print(f"  GPU disponibili: {len(gpus)}")

    from models.ldensenet import build_model
    model = build_model(input_shape=(SAMPLES_PER_WINDOW, 6), output_dims=6)
    model.load_weights(WEIGHTS_PATH)
    print(f"  Pesi caricati da: {WEIGHTS_PATH}")
    return model


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 2 — Inferenza e pseudo-labelling")
    print("=" * 60)

    # --- Carica candidati ---
    candidates_path = os.path.join(RESULTS_DIR, 'candidate_ids.json')
    if not os.path.exists(candidates_path):
        print(f"ERRORE: {candidates_path} non trovato. Esegui prima step1.")
        sys.exit(1)

    with open(candidates_path, 'r', encoding='utf-8') as f:
        candidates = json.load(f)
    print(f"\nCandidati caricati: {len(candidates)}")

    # --- Carica modello ---
    print("\n[Caricamento modello...]")
    model = load_model()

    # --- Inferenza ---
    pseudolabels = []
    stats = {
        'total': len(candidates),
        'sqa_failed': 0,
        'no_windows': 0,
        'low_confidence': 0,
        'accepted': 0,
        'class_counts': {name: 0 for name in CLASS_NAMES},
    }

    print(f"\n[Inferenza su {len(candidates)} candidati...]")
    bar = tqdm(candidates, desc="Pseudo-labelling", unit="ecg")

    for cand in bar:
        ecg_id   = cand['id']
        zip_path = cand['zip_path']
        text     = cand.get('text', '')

        try:
            # Leggi EDF dallo ZIP
            with zipfile.ZipFile(zip_path, 'r') as z:
                edf_bytes = z.read(f"{ecg_id}.edf")

            # Preprocessing reale (no gain, no noise)
            sigs_raw, sigs_norm = preprocess_edf(edf_bytes)
            if sigs_raw is None:
                stats['sqa_failed'] += 1
                continue

            # Predizione con majority voting (mask su RAW, finestre su NORM)
            predicted_class, confidence, n_windows = predict_ecg(model, sigs_raw, sigs_norm)
            if predicted_class is None:
                stats['no_windows'] += 1
                continue

            if confidence < CONFIDENCE_THRESHOLD:
                stats['low_confidence'] += 1
                continue

            # Accettato
            class_name = CLASS_NAMES[predicted_class]
            pseudolabels.append({
                'id':              ecg_id,
                'predicted_class': predicted_class,
                'class_name':      class_name,
                'confidence':      round(confidence, 4),
                'n_windows':       n_windows,
                'text':            text,
                'zip_path':        zip_path,
            })
            stats['accepted'] += 1
            stats['class_counts'][class_name] += 1

        except KeyError:
            # EDF non trovato nel ZIP (nome inatteso)
            stats['sqa_failed'] += 1
            continue
        except Exception as e:
            stats['sqa_failed'] += 1
            continue

        bar.set_postfix(accepted=stats['accepted'], conf_thr=f"{CONFIDENCE_THRESHOLD:.0%}")

    # --- Salva risultati ---
    out_path = os.path.join(RESULTS_DIR, 'pseudolabels.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(pseudolabels, f, ensure_ascii=False, indent=2)

    # --- Report ---
    report_lines = [
        "=" * 60,
        "REPORT PSEUDO-LABELLING",
        "=" * 60,
        f"Candidati totali:          {stats['total']}",
        f"Scartati (SQA / errore):   {stats['sqa_failed']}",
        f"Scartati (no finestre):    {stats['no_windows']}",
        f"Scartati (conf < {CONFIDENCE_THRESHOLD:.0%}):   {stats['low_confidence']}",
        f"Accettati (conf >= {CONFIDENCE_THRESHOLD:.0%}): {stats['accepted']}",
        "",
        "Distribuzione classi pseudo-labeled:",
    ]
    for name, count in stats['class_counts'].items():
        pct = 100 * count / max(stats['accepted'], 1)
        report_lines.append(f"  {name:<15}: {count:>4}  ({pct:.1f}%)")

    report_lines += [
        "",
        f"Output: {out_path}",
        "=" * 60,
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = os.path.join(RESULTS_DIR, 'ssl_stats_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text + "\n")
    print(f"Report salvato in: {report_path}")
