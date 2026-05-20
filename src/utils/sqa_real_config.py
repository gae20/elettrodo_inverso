"""
sqa_real_config.py
==================
Config SQA separate per scenari diversi.

- QUALITY_CFG_REAL          → ECG reali etichettati (cardiologo ha già validato)
- QUALITY_CFG_SYNTH_RELAXED → Sintetici generati da ECG refertati con augmentation

Il QUALITY_CFG originale in config.py rimane invariato (usato come default ovunque
che non sia esplicitamente sovrascritto).

Principio chiave
----------------
Un ECG refertato da un cardiologo è clinicamente valido per definizione,
anche se è rumoroso o ha baseline wander. La SQA sui reali serve SOLO a
rilevare anomalie strutturali (flatline, RL-RA/RL-LA, ADC fault) che
rendono fisicamente impossibile leggere l'ECG.

Il per-window filtering rimane attivo in entrambi i config: singole finestre
di 2s con artefatti locali vengono comunque escluse, ma un ECG non viene
scartato interamente solo perché alcune finestre sono rumorose.
"""

import copy
from utils.config import QUALITY_CFG

# =============================================================================
# CONFIG PER ECG REALI ETICHETTATI
# =============================================================================
# Disabilita i check che eliminano variabilità clinica legittima.
# Mantiene solo i check strutturali che rilevano classi 6/7 (RL-RA / RL-LA).
# =============================================================================

QUALITY_CFG_REAL = copy.deepcopy(QUALITY_CFG)

# ── DISABILITATI ──────────────────────────────────────────────────────────────
# Rumore (mad_diff): ECG clinici con artefatti di movimento / EMG sono validi
QUALITY_CFG_REAL["mad_noise_limb"]    = 9999.0  # era 25.0  → disabilitato
QUALITY_CFG_REAL["mad_noise_prec"]    = 9999.0  # era 35.0  → disabilitato

# Baseline wander: comune in anziani, BPCO, registrazioni lunghe
QUALITY_CFG_REAL["baseline_max_uv"]   = 9999.0  # era 600.0 → disabilitato

# Low energy (std): inversioni LA-LL / RA-LL riducono l'ampiezza di alcune derivazioni
QUALITY_CFG_REAL["std_low_limb"]      = 0.1     # era 15.0  → quasi disabilitato
QUALITY_CFG_REAL["std_low_prec"]      = 0.1     # era 25.0  → quasi disabilitato

# ── AMMORBIDITI ───────────────────────────────────────────────────────────────
# no_morphology (min_mad_diff): QRS piccoli in anziani, inversioni con ampiezza ridotta
# Il check globale (check_lead_quality_global) usa questa soglia direttamente.
# Con ECG a bassa ampiezza (500µV) il mad_diff globale è ~1.07.
QUALITY_CFG_REAL["min_mad_diff_limb"] = 0.3     # era 1.5
QUALITY_CFG_REAL["min_mad_diff_prec"] = 0.3     # era 1.5

# min_valid_ratio: un ECG è accettato se almeno il 30% delle finestre è valida.
# Le finestre non valide nel restante 70% vengono comunque scartate singolarmente.
QUALITY_CFG_REAL["min_valid_ratio"]   = 0.30    # era 0.60

# ── INVARIATI (strutturali — rilevano classi 6/7) ─────────────────────────────
# flatline_std_thr    = 25.0    → rimane
# flatline_ptp_thr    = 40.0    → rimane
# near_zero_median_thr = 14.0   → rimane  (RL-RA / RL-LA detection)
# amplitude_max       = 6000.0  → rimane  (ADC saturation)
# clip_ratio_thr      = 0.02    → rimane
# adc_step_limit      = 2000.0  → rimane
# structural RL-RA/LA check     → rimane  (lead II/III piatta)


# =============================================================================
# CONFIG PER SINTETICI (versione alleggerita)
# =============================================================================
# I sintetici partono da ECG reali già refertati, poi vengono augmentati con
# rumore e baseline wander. La SQA originale scartava le versioni più rumorose
# riducendo artificialmente la varianza del training set (ratio varianza 3×).
# Con questa config accettiamo più istanze rumorose, avvicinando la distribuzione
# sintetica a quella reale.
# =============================================================================

QUALITY_CFG_SYNTH_RELAXED = copy.deepcopy(QUALITY_CFG)

QUALITY_CFG_SYNTH_RELAXED["mad_noise_limb"]  = 100.0   # era 25.0 (global usa ×1.1 → 110)
QUALITY_CFG_SYNTH_RELAXED["mad_noise_prec"]  = 120.0   # era 35.0 (global usa ×1.1 → 132)
QUALITY_CFG_SYNTH_RELAXED["baseline_max_uv"] = 1500.0  # era 600.0
QUALITY_CFG_SYNTH_RELAXED["std_low_limb"]    = 5.0     # era 15.0
QUALITY_CFG_SYNTH_RELAXED["std_low_prec"]    = 8.0     # era 25.0
QUALITY_CFG_SYNTH_RELAXED["min_valid_ratio"] = 0.40    # era 0.60

# Invariati: flatline, low_amplitude, near_zero_median, clipping, adc_step, structural_RL
