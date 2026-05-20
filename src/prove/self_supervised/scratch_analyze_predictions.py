"""
Analisi approfondita della classe 5 (ROT_ANT):
- Distribuzione confidenza per classe (reali vs simulati)
- Caratteristiche del segnale (rumore, ampiezza, MAD) per classe 5 reale vs simulata
- Dove finiscono i ROT_ANT reali quando il modello sbaglia
"""
import os, sys, h5py, numpy as np
import tensorflow as tf

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
THESIS_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))
sys.path.append(SRC_DIR)

from models.ldensenet import build_model
from utils.config import SAMPLES_PER_WINDOW, ALL_LEADS, LIMB_LEADS

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

CLASS_NAMES = ['normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']
DATASETS_DIR = os.path.abspath(os.path.join(THESIS_DIR, '..', 'datasets'))

# --- Carica modello ---
model = build_model(input_shape=(SAMPLES_PER_WINDOW, 6), output_dims=6)
weights_path = os.path.join(SRC_DIR, 'prove', 'models', 'best_model_final_noise_limbs.weights.h5')
model.load_weights(weights_path)
print("Modello caricato.\n")

# ═══════════════════════════════════════════════════════════════
# 1. Carica test set REALE (clean)
# ═══════════════════════════════════════════════════════════════
real_h5 = os.path.join(DATASETS_DIR, 'labelled_z_median_limbs_test_validation_clean.h5')
with h5py.File(real_h5, 'r') as f:
    y_real_all = f['Y'][:]
    valid = np.where(y_real_all < 6)[0]
    X_real = f['X'][valid, :6, :]       # (N, 6, win_size)
    y_real = y_real_all[valid].astype(int)

x_real_input = np.transpose(X_real, (0, 2, 1))  # (N, win_size, 6)
y_real_probs = model.predict(x_real_input, batch_size=64, verbose=0)
y_real_pred  = np.argmax(y_real_probs, axis=1)

# ═══════════════════════════════════════════════════════════════
# 2. Carica test set SIMULATO
# ═══════════════════════════════════════════════════════════════
synth_h5 = os.path.join(DATASETS_DIR, 'unlabelled_final_noise_limbs_test.h5')
with h5py.File(synth_h5, 'r') as f:
    y_synth_all = f['Y'][:]
    valid_s = np.where(y_synth_all < 6)[0]
    # Bilanciamento: max 500 per classe
    np.random.seed(42)
    balanced = []
    for c in range(6):
        c_idx = valid_s[y_synth_all[valid_s] == c]
        chosen = np.random.choice(c_idx, min(500, len(c_idx)), replace=False)
        balanced.extend(chosen)
    balanced = np.array(sorted(balanced))
    X_synth = f['X'][balanced, :6, :]
    y_synth = y_synth_all[balanced].astype(int)

x_synth_input = np.transpose(X_synth, (0, 2, 1))
y_synth_probs = model.predict(x_synth_input, batch_size=64, verbose=0)
y_synth_pred  = np.argmax(y_synth_probs, axis=1)


# ═══════════════════════════════════════════════════════════════
# 3. ANALISI CONFIDENZA PER CLASSE
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("ANALISI CONFIDENZA PER CLASSE (Reali vs Simulati)")
print("=" * 70)
print(f"{'Classe':<12} | {'N_real':>6} {'Conf_real':>10} {'Acc_real':>9} | {'N_sim':>6} {'Conf_sim':>10} {'Acc_sim':>9}")
print("-" * 70)

for c in range(6):
    # Reali
    mask_r = (y_real == c)
    n_r = mask_r.sum()
    if n_r > 0:
        confs_r = y_real_probs[mask_r, y_real_pred[mask_r]]
        conf_correct_r = y_real_probs[mask_r & (y_real_pred == c), c]
        acc_r = (y_real_pred[mask_r] == c).mean()
        conf_mean_r = confs_r.mean()
    else:
        acc_r = conf_mean_r = 0.0

    # Simulati
    mask_s = (y_synth == c)
    n_s = mask_s.sum()
    if n_s > 0:
        confs_s = y_synth_probs[mask_s, y_synth_pred[mask_s]]
        acc_s = (y_synth_pred[mask_s] == c).mean()
        conf_mean_s = confs_s.mean()
    else:
        acc_s = conf_mean_s = 0.0

    print(f"{CLASS_NAMES[c]:<12} | {n_r:>6} {conf_mean_r:>10.4f} {acc_r:>8.1%} | {n_s:>6} {conf_mean_s:>10.4f} {acc_s:>8.1%}")

# ═══════════════════════════════════════════════════════════════
# 4. DETTAGLIO ROT_ANT (classe 5) — dove finiscono quando sbagliano?
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DETTAGLIO ROT_ANT (classe 5) — REALI")
print("=" * 70)

mask_5_real = (y_real == 5)
n_5_real = mask_5_real.sum()
print(f"Totale finestre ROT_ANT reali: {n_5_real}")

if n_5_real > 0:
    preds_5_real = y_real_pred[mask_5_real]
    probs_5_real = y_real_probs[mask_5_real]

    # Distribuzione predizioni
    print("\nDove vengono classificate le finestre ROT_ANT reali:")
    for c in range(6):
        n = (preds_5_real == c).sum()
        pct = 100 * n / n_5_real
        avg_conf = probs_5_real[preds_5_real == c, c].mean() if n > 0 else 0
        marker = " [OK]" if c == 5 else ""
        print(f"  -> {CLASS_NAMES[c]:<12}: {n:>4} ({pct:>5.1f}%)  conf media={avg_conf:.4f}{marker}")

    # Distribuzione confidenza per la classe corretta (prob classe 5)
    prob_class5 = probs_5_real[:, 5]
    print(f"\nProbabilità assegnata alla classe ROT_ANT (veri ROT_ANT reali):")
    print(f"  Media:   {prob_class5.mean():.4f}")
    print(f"  Mediana: {np.median(prob_class5):.4f}")
    print(f"  Min:     {prob_class5.min():.4f}")
    print(f"  Max:     {prob_class5.max():.4f}")
    print(f"  Std:     {prob_class5.std():.4f}")

    # Percentili
    for pct in [10, 25, 50, 75, 90]:
        print(f"  P{pct}: {np.percentile(prob_class5, pct):.4f}")

print("\n" + "=" * 70)
print("DETTAGLIO ROT_ANT (classe 5) — SIMULATI")
print("=" * 70)

mask_5_synth = (y_synth == 5)
n_5_synth = mask_5_synth.sum()
print(f"Totale finestre ROT_ANT simulati: {n_5_synth}")

if n_5_synth > 0:
    preds_5_synth = y_synth_pred[mask_5_synth]
    probs_5_synth = y_synth_probs[mask_5_synth]

    print("\nDove vengono classificate le finestre ROT_ANT simulati:")
    for c in range(6):
        n = (preds_5_synth == c).sum()
        pct = 100 * n / n_5_synth
        avg_conf = probs_5_synth[preds_5_synth == c, c].mean() if n > 0 else 0
        marker = " [OK]" if c == 5 else ""
        print(f"  -> {CLASS_NAMES[c]:<12}: {n:>4} ({pct:>5.1f}%)  conf media={avg_conf:.4f}{marker}")

    prob_class5_s = probs_5_synth[:, 5]
    print(f"\nProbabilità assegnata alla classe ROT_ANT (veri ROT_ANT simulati):")
    print(f"  Media:   {prob_class5_s.mean():.4f}")
    print(f"  Mediana: {np.median(prob_class5_s):.4f}")
    print(f"  Min:     {prob_class5_s.min():.4f}")
    print(f"  Max:     {prob_class5_s.max():.4f}")
    print(f"  Std:     {prob_class5_s.std():.4f}")


# ═══════════════════════════════════════════════════════════════
# 5. CARATTERISTICHE DEL SEGNALE — ROT_ANT reali vs simulati
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CARATTERISTICHE SEGNALE — ROT_ANT reali vs simulati")
print("=" * 70)

def signal_stats(X_subset):
    """Calcola statistiche per un subset di finestre (N, 6, win_size)."""
    stds   = np.std(X_subset, axis=2)        # (N, 6)
    ptps   = np.ptp(X_subset, axis=2)        # (N, 6)
    mads   = np.median(np.abs(np.diff(X_subset, axis=2) -
                               np.median(np.diff(X_subset, axis=2), axis=2, keepdims=True)), axis=2)
    med_abs = np.median(np.abs(X_subset), axis=2)
    return {
        'std_mean': stds.mean(axis=0),
        'ptp_mean': ptps.mean(axis=0),
        'mad_mean': mads.mean(axis=0),
        'med_abs_mean': med_abs.mean(axis=0),
        'std_global': stds.mean(),
        'ptp_global': ptps.mean(),
        'mad_global': mads.mean(),
        'med_abs_global': med_abs.mean(),
    }

if n_5_real > 0:
    stats_real = signal_stats(X_real[mask_5_real])
    print(f"\nROT_ANT REALI ({n_5_real} finestre):")
    print(f"  STD media (tutte le lead):       {stats_real['std_global']:.4f}")
    print(f"  PTP media (tutte le lead):       {stats_real['ptp_global']:.4f}")
    print(f"  MAD diff media (tutte le lead):  {stats_real['mad_global']:.4f}")
    print(f"  Median |x| media:                {stats_real['med_abs_global']:.4f}")
    print(f"  STD per lead:  {stats_real['std_mean']}")
    print(f"  PTP per lead:  {stats_real['ptp_mean']}")
    print(f"  MAD per lead:  {stats_real['mad_mean']}")

if n_5_synth > 0:
    stats_synth = signal_stats(X_synth[mask_5_synth])
    print(f"\nROT_ANT SIMULATI ({n_5_synth} finestre):")
    print(f"  STD media (tutte le lead):       {stats_synth['std_global']:.4f}")
    print(f"  PTP media (tutte le lead):       {stats_synth['ptp_global']:.4f}")
    print(f"  MAD diff media (tutte le lead):  {stats_synth['mad_global']:.4f}")
    print(f"  Median |x| media:                {stats_synth['med_abs_global']:.4f}")
    print(f"  STD per lead:  {stats_synth['std_mean']}")
    print(f"  PTP per lead:  {stats_synth['ptp_mean']}")
    print(f"  MAD per lead:  {stats_synth['mad_mean']}")

# Confronto anche TUTTE le classi per vedere se è un problema specifico di ROT_ANT
print("\n" + "=" * 70)
print("CONFRONTO SEGNALE REALI vs SIMULATI — TUTTE LE CLASSI")
print("=" * 70)
print(f"{'Classe':<12} | {'STD_real':>10} {'STD_sim':>10} {'Ratio':>7} | {'MAD_real':>10} {'MAD_sim':>10} {'Ratio':>7}")
print("-" * 75)
for c in range(6):
    mr = (y_real == c)
    ms = (y_synth == c)
    if mr.sum() > 0 and ms.sum() > 0:
        sr = signal_stats(X_real[mr])
        ss = signal_stats(X_synth[ms])
        std_ratio = sr['std_global'] / ss['std_global'] if ss['std_global'] > 0 else 0
        mad_ratio = sr['mad_global'] / ss['mad_global'] if ss['mad_global'] > 0 else 0
        print(f"{CLASS_NAMES[c]:<12} | {sr['std_global']:>10.2f} {ss['std_global']:>10.2f} {std_ratio:>6.2f}x | {sr['mad_global']:>10.2f} {ss['mad_global']:>10.2f} {mad_ratio:>6.2f}x")

# ═══════════════════════════════════════════════════════════════
# 6. ENTROPIA delle predizioni (incertezza del modello)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ENTROPIA PREDIZIONI (incertezza) — PER CLASSE")
print("=" * 70)

def entropy(probs):
    p = np.clip(probs, 1e-10, 1.0)
    return -np.sum(p * np.log2(p), axis=1)

ent_real = entropy(y_real_probs)
ent_synth = entropy(y_synth_probs)

print(f"{'Classe':<12} | {'Ent_real':>10} {'Ent_sim':>10} | {'Gap':>10}")
print("-" * 50)
for c in range(6):
    mr = (y_real == c)
    ms = (y_synth == c)
    er = ent_real[mr].mean() if mr.sum() > 0 else 0
    es = ent_synth[ms].mean() if ms.sum() > 0 else 0
    print(f"{CLASS_NAMES[c]:<12} | {er:>10.4f} {es:>10.4f} | {er - es:>+10.4f}")

print("\nAnalisi completata.")
