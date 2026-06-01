"""
test_ssl_limbs.py

Valuta il modello SSL e il modello originale sul test set.
Supporta due modalità di valutazione via riga di comando:
  1. Window-level (default): valuta sulle finestre del test set simulato H5.
  2. Patient-level: valuta sui pazienti reali di dataset_small con upsampling per bilanciamento.

Output:
  - Classification report (Precision / Recall / F1 per classe)
  - Accuracy, AUROC, AuPRC
  - Confusion matrix salvate in results/ssl_weights/
  - Confronto numerico tra SSL e modello originale
"""

import os
import sys
import h5py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from keras.utils import to_categorical

# Configurazione path
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SRC_DIR      = os.path.join(BASE_DIR, '..')
THESIS_DIR   = os.path.join(SRC_DIR, '..')
TRAINING_DIR = os.path.join(SRC_DIR, 'training')

sys.path.append(SRC_DIR)
from models.ldensenet import build_model
from data.data_pipeline import (
    read_edf_data, all_leads_preprocessing,
    check_ecg_quality, check_window_quality,
)
from utils.config import (
    SAMPLES_PER_WINDOW, STRIDE_SAMPLES, ALL_LEADS, QUALITY_CFG, LABEL_MAP_CLEAN
)

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

CLASS_NAMES = ['normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']

# --- Percorsi ---
DATASET_TEST_WINDOW = os.path.join(BASE_DIR, '..', '..', 'datasets', 'unlabelled_simulated_final', 'unlabelled_z_median_limbs_test_backup.h5')
DATASET_TEST_WITH_PATIENTS = os.path.join(BASE_DIR, '..', '..', 'datasets', 'unlabelled_simulated_final', 'unlabelled_z_median_limbs_test_with_patients.h5')

# Pesi SSL — salvati da step4_train_ssl.py
SSL_DIR      = os.path.join(SRC_DIR, 'self_supervised', 'results', 'ssl_weights')
WEIGHTS_SSL  = os.path.join(SSL_DIR, 'best_model_ssl_limbs.weights.h5')
# Pesi originale (per confronto)
WEIGHTS_ORIG = os.path.join(TRAINING_DIR, 'unlabelled_z_median_weights_and_cm',
                             'PROVA_best_model_targeted_noise_limbs.weights.h5')

OUT_DIR = SSL_DIR
os.makedirs(OUT_DIR, exist_ok=True)


# ===========================================================================
# MODALITÀ 1: Valutazione a Finestra (Originale)
# ===========================================================================

def load_test_data_window(path):
    with h5py.File(path, 'r') as f:
        y_all = f['Y'][:]
        valid_idx = np.where(y_all < 6)[0]
        x_raw = f['X'][valid_idx, :6, :]
        x = np.transpose(x_raw, (0, 2, 1))
        y = y_all[valid_idx]
    print(f"  Campioni test: {len(y)}  |  Classi: {np.unique(y)}")
    return x, y


def evaluate_window(model, x, y, cm_path=None, model_name=""):
    y_probs = model.predict(x, batch_size=64, verbose=0)
    y_pred  = np.argmax(y_probs, axis=1)
    acc     = np.mean(y_pred == y)

    C = confusion_matrix(y, y_pred, labels=range(6))

    y_oh   = to_categorical(y, num_classes=6)
    auroc  = roc_auc_score(y_oh, y_probs, multi_class='ovr', average='macro')
    auprc  = average_precision_score(y_oh, y_probs, average='macro')

    if cm_path:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        im = ax.matshow(C, cmap=plt.cm.Reds)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, str(C[i, j]),
                        ha='center', va='center', fontsize=10)
        ax.set_xticks(range(6)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='left')
        ax.set_yticks(range(6)); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix — {model_name}', pad=20)
        plt.tight_layout()
        plt.savefig(cm_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Matrice di confusione salvata in: {cm_path}")

    return acc, auroc, auprc, C, y_probs, y_pred


# ===========================================================================
# MODALITÀ 2: Valutazione a Paziente (Nuova)
# ===========================================================================

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
            for li in range(6)
        )
        mask[w_idx] = (valid_count >= min_valid_leads)
    return mask

def create_windows_patient(sigs_array):
    _, n_samples = sigs_array.shape
    if n_samples < SAMPLES_PER_WINDOW:
        return np.empty((0, 6, SAMPLES_PER_WINDOW), dtype=np.float32)
    starts = list(range(0, n_samples - SAMPLES_PER_WINDOW + 1, STRIDE_SAMPLES))
    windows = []
    for start in starts:
        windows.append(sigs_array[:, start:start + SAMPLES_PER_WINDOW])
    return np.array(windows, dtype=np.float32)

def extract_patient_windows(patient_id, edf_dir, cfg=QUALITY_CFG):
    edf_path = os.path.join(edf_dir, f"record{patient_id}.edf")
    if not os.path.exists(edf_path):
        edf_path = os.path.join(edf_dir, f"{patient_id}.edf")
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
    if not all(l in ecg_data["signals"] for l in ALL_LEADS[:6]):
        return None

    sigs = all_leads_preprocessing(ecg_data["signals"])
    sigs_raw = np.array([sigs[l] for l in ALL_LEADS[:6]], dtype=np.float32)

    quality = check_ecg_quality(sigs_raw, cfg=cfg, lead_indices=list(range(6)))
    if not quality['global_valid']:
        return None

    win_mask = compute_good_window_mask(sigs_raw, cfg=cfg, min_valid_leads=5)
    if win_mask.size == 0 or not win_mask.any():
        return None

    sigs_norm = robust_scale_ecg(sigs_raw)
    windows = create_windows_patient(sigs_norm)

    n = min(windows.shape[0], win_mask.size)
    good_windows = windows[:n][win_mask[:n]]
    if good_windows.shape[0] == 0:
        return None

    return np.transpose(good_windows, (0, 2, 1))

def predict_model_patients(model, valid_patients, patient_windows_dict):
    patients_by_class = {c: [] for c in range(6)}
    for pat in valid_patients:
        pid = pat['id']
        c = pat['true_class']
        wins = patient_windows_dict[pid]
        y_probs = model.predict(wins, batch_size=64, verbose=0)
        avg_prob = np.mean(y_probs, axis=0)
        patients_by_class[c].append(avg_prob)
    return patients_by_class

def balance_and_prepare(patients_by_class, seed=42):
    np.random.seed(seed)
    max_patients = max(len(patients_by_class[c]) for c in range(6))
    print(f"  Target di pazienti per classe (upsample): {max_patients}")
    
    balanced_probs = []
    balanced_trues = []
    
    for c in range(6):
        pats = patients_by_class[c]
        n_pats = len(pats)
        if n_pats == 0:
            continue
        chosen_indices = np.random.choice(n_pats, size=max_patients, replace=True)
        for idx in chosen_indices:
            balanced_probs.append(pats[idx])
            balanced_trues.append(c)
            
    return np.array(balanced_trues), np.array(balanced_probs)

def evaluate_balanced_patient(y_true, y_probs, cm_path=None, model_name=""):
    y_pred = np.argmax(y_probs, axis=1)
    acc = np.mean(y_pred == y_true)
    C = confusion_matrix(y_true, y_pred, labels=range(6))
    
    if cm_path:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        im = ax.matshow(C, cmap=plt.cm.Reds)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, str(C[i, j]),
                        ha='center', va='center', fontsize=10)
        ax.set_xticks(range(6)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='left')
        ax.set_yticks(range(6)); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix (Paziente Bilanciato) — {model_name}', pad=20)
        plt.tight_layout()
        plt.savefig(cm_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Matrice di confusione salvata in: {cm_path}")
        
    y_oh = to_categorical(y_true, num_classes=6)
    auroc = roc_auc_score(y_oh, y_probs, multi_class='ovr', average='macro')
    auprc = average_precision_score(y_oh, y_probs, average='macro')
    
    return acc, auroc, auprc, C, y_probs, y_pred


# ===========================================================================
# Visualizzazione Condivisa
# ===========================================================================

def print_report(y, y_pred, acc, auroc, auprc, C, model_name):
    print(f"\n{'='*55}")
    print(f"  MODELLO: {model_name}")
    print(f"{'='*55}")
    print(classification_report(y, y_pred, target_names=CLASS_NAMES, digits=3))
    print(f"  Accuratezza Totale : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUROC (Macro)      : {auroc:.4f}")
    print(f"  AuPRC (Macro)      : {auprc:.4f}")
    print(f"\n  [ANALISI ERRORI]")
    for i, name in enumerate(CLASS_NAMES):
        tp     = C[i, i]
        total  = C[i].sum()
        errors = [(CLASS_NAMES[j], C[i, j]) for j in range(6) if j != i and C[i, j] > 0]
        errors.sort(key=lambda x: -x[1])
        if errors:
            err_str = ', '.join([f"{n}={v}" for n, v in errors])
            print(f"    {name:<12} (n={total:>4}): TP={tp:>4} | confuso con: {err_str}")
        else:
            print(f"    {name:<12} (n={total:>4}): TP={tp:>4} | nessun errore")


def print_comparison(metrics_orig, metrics_ssl, mode="window"):
    acc_o, auroc_o, auprc_o = metrics_orig
    acc_s, auroc_s, auprc_s = metrics_ssl

    print(f"\n{'='*55}")
    if mode == "window":
        mode_str = "A livello di Finestra (Simulato)"
    elif mode == "patient_simulated":
        mode_str = "A livello di Paziente Simulato"
    else:
        mode_str = "A livello di Paziente Bilanciato (Reale)"
    print(f"  CONFRONTO ORIGINALE vs SSL ({mode_str})")
    print(f"{'='*55}")
    print(f"  {'Metrica':<20} {'Originale':>10} {'SSL':>10} {'Delta':>10}")
    print(f"  {'-'*50}")

    def delta_str(new, old):
        d = new - old
        sign = '+' if d >= 0 else ''
        return f"{sign}{d*100:.2f}%"

    print(f"  {'Accuracy':<20} {acc_o*100:>9.2f}% {acc_s*100:>9.2f}% {delta_str(acc_s, acc_o):>10}")
    print(f"  {'AUROC (macro)':<20} {auroc_o:>10.4f} {auroc_s:>10.4f} {delta_str(auroc_s, auroc_o):>10}")
    print(f"  {'AuPRC (macro)':<20} {auprc_o:>10.4f} {auprc_s:>10.4f} {delta_str(auprc_s, auprc_o):>10}")
    print(f"{'='*55}")


# ===========================================================================
# MODALITÀ 3: Valutazione a Paziente Simulato (Nuova)
# ===========================================================================

def load_test_data_patient_simulated(path):
    with h5py.File(path, 'r') as f:
        y_all = f['Y'][:]
        valid_idx = np.where(y_all < 6)[0]
        x_raw = f['X'][valid_idx, :6, :]
        x = np.transpose(x_raw, (0, 2, 1))
        y = y_all[valid_idx]
        patient_ids = f['patient_ids'][valid_idx]
    print(f"  Campioni test simulati: {len(y)}  |  Pazienti unici: {len(np.unique(patient_ids))}")
    return x, y, patient_ids


def evaluate_patient_simulated(model, x, y, patient_ids, cm_path=None, model_name=""):
    y_probs = model.predict(x, batch_size=64, verbose=0)
    
    unique_cases = {}
    for i in range(len(y)):
        pid = patient_ids[i]
        c_true = y[i]
        case_key = (pid, c_true)
        if case_key not in unique_cases:
            unique_cases[case_key] = []
        unique_cases[case_key].append(y_probs[i])
        
    y_true_cases = []
    y_probs_cases = []
    for (pid, c_true), probs_list in unique_cases.items():
        avg_prob = np.mean(probs_list, axis=0)
        y_true_cases.append(c_true)
        y_probs_cases.append(avg_prob)
        
    y_true_cases = np.array(y_true_cases)
    y_probs_cases = np.array(y_probs_cases)
    y_pred_cases = np.argmax(y_probs_cases, axis=1)
    
    acc = np.mean(y_pred_cases == y_true_cases)
    C = confusion_matrix(y_true_cases, y_pred_cases, labels=range(6))
    
    y_oh = to_categorical(y_true_cases, num_classes=6)
    auroc = roc_auc_score(y_oh, y_probs_cases, multi_class='ovr', average='macro')
    auprc = average_precision_score(y_oh, y_probs_cases, average='macro')
    
    if cm_path:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        im = ax.matshow(C, cmap=plt.cm.Reds)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, str(C[i, j]),
                        ha='center', va='center', fontsize=10)
        ax.set_xticks(range(6)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='left')
        ax.set_yticks(range(6)); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix (Paziente Simulato) — {model_name}', pad=20)
        plt.tight_layout()
        plt.savefig(cm_path, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Matrice di confusione salvata in: {cm_path}")
        
    return acc, auroc, auprc, C, y_probs_cases, y_pred_cases, y_true_cases


# ===========================================================================
# Main Orchestrator
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Valuta il modello SSL ed Originale.")
    parser.add_argument("--mode", type=str, default="window", choices=["window", "patient", "patient_simulated"],
                        help="Modalità di valutazione: 'window', 'patient' (su pazienti reali) o 'patient_simulated' (su test set con ID)")
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    input_shape = (SAMPLES_PER_WINDOW, 6)
    output_dims = 6

    if args.mode == "window":
        print("=" * 55)
        print("VALUTAZIONE SSL — Test Set Simulato (Finestre)")
        print("=" * 55)

        print(f"\nCaricamento test set da: {DATASET_TEST_WINDOW}")
        x_test, y_test = load_test_data_window(DATASET_TEST_WINDOW)

        # --- Modello originale ---
        print(f"\n[1/2] Modello ORIGINALE: {WEIGHTS_ORIG}")
        model = build_model(input_shape, output_dims)
        model.load_weights(WEIGHTS_ORIG)
        cm_orig = os.path.join(OUT_DIR, 'comparison_cm_original.png')
        acc_o, auroc_o, auprc_o, C_o, _, y_pred_o = evaluate_window(
            model, x_test, y_test, cm_path=cm_orig, model_name="Originale"
        )
        print_report(y_test, y_pred_o, acc_o, auroc_o, auprc_o, C_o, "Originale")

        # --- Modello SSL ---
        print(f"\n[2/2] Modello SSL: {WEIGHTS_SSL}")
        model_ssl = build_model(input_shape, output_dims)
        model_ssl.load_weights(WEIGHTS_SSL)
        cm_ssl = os.path.join(OUT_DIR, 'ssl_cm_realtest.png')
        acc_s, auroc_s, auprc_s, C_s, _, y_pred_s = evaluate_window(
            model_ssl, x_test, y_test, cm_path=cm_ssl, model_name="SSL"
        )
        print_report(y_test, y_pred_s, acc_s, auroc_s, auprc_s, C_s, "SSL")

        # --- Confronto ---
        print_comparison(
            (acc_o, auroc_o, auprc_o),
            (acc_s, auroc_s, auprc_s),
            mode="window"
        )

    elif args.mode == "patient_simulated":
        print("=" * 65)
        print("VALUTAZIONE COMPARATIVA A LIVELLO DI PAZIENTE SIMULATO")
        print("=" * 65)

        print(f"\nCaricamento test set da: {DATASET_TEST_WITH_PATIENTS}")
        if not os.path.exists(DATASET_TEST_WITH_PATIENTS):
            print(f"Errore: File non trovato. Esegui prima build_test_dataset_with_patients.py per crearlo.")
            sys.exit(1)

        x_test, y_test, patient_ids = load_test_data_patient_simulated(DATASET_TEST_WITH_PATIENTS)

        # --- Modello originale ---
        print(f"\n[1/2] Modello ORIGINALE: {WEIGHTS_ORIG}")
        model = build_model(input_shape, output_dims)
        model.load_weights(WEIGHTS_ORIG)
        cm_orig = os.path.join(OUT_DIR, 'ssl_cm_original_patient_simulated.png')
        acc_o, auroc_o, auprc_o, C_o, _, y_pred_o, y_true_cases_o = evaluate_patient_simulated(
            model, x_test, y_test, patient_ids, cm_path=cm_orig, model_name="Originale (Paziente Simulato)"
        )
        print_report(y_true_cases_o, y_pred_o, acc_o, auroc_o, auprc_o, C_o, "Originale")

        # --- Modello SSL ---
        print(f"\n[2/2] Modello SSL: {WEIGHTS_SSL}")
        model_ssl = build_model(input_shape, output_dims)
        model_ssl.load_weights(WEIGHTS_SSL)
        cm_ssl = os.path.join(OUT_DIR, 'ssl_cm_realtest_patient_simulated.png')
        acc_s, auroc_s, auprc_s, C_s, _, y_pred_s, y_true_cases_s = evaluate_patient_simulated(
            model_ssl, x_test, y_test, patient_ids, cm_path=cm_ssl, model_name="SSL (Paziente Simulato)"
        )
        print_report(y_true_cases_s, y_pred_s, acc_s, auroc_s, auprc_s, C_s, "SSL")

        # --- Confronto ---
        print_comparison(
            (acc_o, auroc_o, auprc_o),
            (acc_s, auroc_s, auprc_s),
            mode="patient_simulated"
        )

    else:
        # args.mode == "patient"
        print("=" * 65)
        print("VALUTAZIONE COMPARATIVA PAZIENTI REALI BILANCIATI (dataset_small)")
        print("=" * 65)

        # 1. Ricerca CSV e cartella EDF
        csv_paths = [
            os.path.join(THESIS_DIR, 'datasets', 'dataset', 'dataset_small', 'thesis-sample-corrected.csv'),
            os.path.join(THESIS_DIR, 'datasets', 'dataset', 'dataset_small', 'thesis-sample.csv'),
            os.path.join(THESIS_DIR, 'datasets', 'dataset_small', 'thesis-sample-corrected.csv'),
            os.path.join(THESIS_DIR, 'datasets', 'dataset_small', 'thesis-sample.csv'),
        ]
        csv_path = None
        for p in csv_paths:
            if os.path.exists(p):
                csv_path = p
                break

        if not csv_path:
            print("Errore: Impossibile trovare il file thesis-sample.csv o thesis-sample-corrected.csv")
            sys.exit(1)

        edf_dir = os.path.dirname(csv_path)
        print(f"Trovato CSV in: {csv_path}")
        print(f"Cartella EDF:   {edf_dir}")

        # 2. Caricamento pazienti
        df = pd.read_csv(csv_path)
        df_cand = df[df["Inversione"] != "?"].copy()
        df_cand["Inversione"] = df_cand["Inversione"].apply(lambda x: LABEL_MAP_CLEAN.get(x, x))

        class_name_map = {
            'normale': 0,
            'LA-RA': 1,
            'RA-LL': 2,
            'LA-LL': 3,
            'ROT_ORARIA': 4,
            'ROT_ANTIORARIA': 5
        }

        patients_data = []
        for _, row in df_cand.iterrows():
            pid = row['Num']
            inv_str = row['Inversione']
            if inv_str in class_name_map:
                c = class_name_map[inv_str]
                ref_text = row.get('Referto', '')
                if pd.isna(ref_text):
                    ref_text = ''
                patients_data.append({
                    'id': pid,
                    'true_class': c,
                    'text': str(ref_text)
                })

        print(f"Candidati caricati dal CSV: {len(patients_data)}")

        # 3. Estrazione finestre in memoria
        patient_windows_dict = {}
        valid_patients = []
        
        print("\n[Estrazione finestre per ciascun paziente (SQA globale + window SQA)...]")
        for pat in tqdm(patients_data, desc="Caricamento EDF"):
            pid = pat['id']
            wins = extract_patient_windows(pid, edf_dir, cfg=QUALITY_CFG)
            if wins is not None:
                patient_windows_dict[pid] = wins
                valid_patients.append(pat)

        print(f"Pazienti validi (superato SQA e con almeno una finestra buona): {len(valid_patients)}")
        counts_valid = {c: sum(1 for p in valid_patients if p['true_class'] == c) for c in range(6)}
        print(f"Distribuzione pazienti per classe: {counts_valid}")

        # 4. Modello originale
        print(f"\n[1/2] Predizione con modello ORIGINALE: {WEIGHTS_ORIG}")
        if not os.path.exists(WEIGHTS_ORIG):
            print(f"Errore: Pesi originali non trovati in {WEIGHTS_ORIG}")
            sys.exit(1)
        
        model_orig = build_model(input_shape, output_dims)
        model_orig.load_weights(WEIGHTS_ORIG)
        
        pats_by_class_orig = predict_model_patients(model_orig, valid_patients, patient_windows_dict)
        y_true_bal_o, y_probs_bal_o = balance_and_prepare(pats_by_class_orig, seed=42)
        
        cm_path_orig = os.path.join(OUT_DIR, 'ssl_cm_original_patient.png')
        acc_o, auroc_o, auprc_o, C_o, _, y_pred_o = evaluate_balanced_patient(
            y_true_bal_o, y_probs_bal_o, cm_path=cm_path_orig, model_name="Originale"
        )
        print_report(y_true_bal_o, y_pred_o, acc_o, auroc_o, auprc_o, C_o, "Originale")

        # 5. Modello SSL
        print(f"\n[2/2] Predizione con modello SSL: {WEIGHTS_SSL}")
        if not os.path.exists(WEIGHTS_SSL):
            print(f"Errore: Pesi SSL non trovati in {WEIGHTS_SSL}")
            sys.exit(1)
            
        model_ssl = build_model(input_shape, output_dims)
        model_ssl.load_weights(WEIGHTS_SSL)
        
        pats_by_class_ssl = predict_model_patients(model_ssl, valid_patients, patient_windows_dict)
        y_true_bal_s, y_probs_bal_s = balance_and_prepare(pats_by_class_ssl, seed=42)
        
        cm_path_ssl = os.path.join(OUT_DIR, 'ssl_cm_realtest_patient.png')
        acc_s, auroc_s, auprc_s, C_s, _, y_pred_s = evaluate_balanced_patient(
            y_true_bal_s, y_probs_bal_s, cm_path=cm_path_ssl, model_name="SSL"
        )
        print_report(y_true_bal_s, y_pred_s, acc_s, auroc_s, auprc_s, C_s, "SSL")

        # 6. Confronto
        print_comparison(
            (acc_o, auroc_o, auprc_o),
            (acc_s, auroc_s, auprc_s),
            mode="patient"
        )
