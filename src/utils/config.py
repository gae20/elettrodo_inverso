import numpy as np

# --- FREQUENCY & WINDOW SETTINGS ---
# The sampling rate should be at least 240Hz to allow a 120Hz Nyquist frequency.
# FS_NEW is set to 250 Hz.
FS_OLD = 1000
FS_NEW = 250
WIN_SEC = 2.0
STRIDE_SEC = 0.5

SAMPLES_PER_WINDOW = int(FS_NEW * WIN_SEC)
STRIDE_SAMPLES = int(FS_NEW * STRIDE_SEC)

# --- LEAD DEFINITIONS ---
LIMB_LEADS = ['I', 'II', 'III', 'aVr', 'aVl', 'aVf']
PRECORDIAL_LEADS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
ALL_LEADS = LIMB_LEADS + PRECORDIAL_LEADS

# --- CLASSES MAP (Limb Leads) ---
MAPPING_INV = {
    'LA-RA': 1, 
    'RA-LL': 2, 
    'LA-LL': 3, 
    'ROT_ORARIA': 4, 
    'ROT_ANTIORARIA': 5,
    'RL-RA': 6, 
    'RL-LA': 7
}

# --- MAPPING 1: PULIZIA ETICHETTE REALI (Dal Medico alla Tesi) ---
# Converte le sigle usate dal medico nel CSV in classi interne del progetto.
# RF = Rosso-Verde (filo rosso RA nel connettore verde LL) → RA-LL (Mode 2)
# LF = Giallo-Verde (filo giallo LA nel connettore verde LL) → LA-LL (Mode 3)
# RN = Rosso-Nero (filo rosso RA nel connettore nero RL) → RL-RA (Mode 6) ← flatline
# LN = Giallo-Nero (filo giallo LA nel connettore nero RL) → RL-LA (Mode 7) ← flatline
# RL = scambio LA-RA → Lead I invertita, ECG valido ma specchiato
LABEL_MAP_CLEAN = {
    'normale':    'normale',
    'RL':         'LA-RA',
    'antiorario': 'ROT_ANTIORARIA',
    'orario':     'ROT_ORARIA',
    'RF':         'RA-LL',
    'LF':         'LA-LL',
    'RN':         'RL-RA',
    'LN':         'RL-LA',
}
# We use only the first 5 for Limb Lead model.
ACTIVE_SYNTH_CLASSES = list(MAPPING_INV.keys())[:5]

# Classes that the SQA MUST reject (RL used as active electrode → near-zero signal).
# These are excluded from the limb-lead training dataset entirely.
FLATLINE_CLASSES = {'RL-RA', 'RL-LA'}

# --- CLASSES MAP (Precordial Leads) ---
PRECORDIAL_MAPPING = {
    'V1-V2': 1, 'V1-V3': 2, 'V1-V4': 3, 'V1-V5': 4, 'V1-V6': 5,
    'V2-V3': 6, 'V2-V4': 7, 'V2-V5': 8, 'V2-V6': 9,
    'V3-V4': 10, 'V3-V5': 11, 'V3-V6': 12,
    'V4-V5': 13, 'V4-V6': 14,
    'V5-V6': 15
}

# --- SIGNAL QUALITY CONFIGURATION ---
# Adjusted for Holter monitors which can be noisier.
QUALITY_CFG = {
    "fs": FS_NEW,
    "win_sec": WIN_SEC,
    "stride_sec": STRIDE_SEC,
    
    # Baseline wandering (DC offset) 
    # Increased slightly for Holters from 500uV
    "baseline_max_uv": 600.0, 
    
    # ADC Saturation / Clipping
    "amplitude_max": 6000.0, 
    "clip_ratio_thr": 0.02,
    
    # ADC Reset Step
    "adc_step_limit": 2000.0, 
    
    # Flatline Thresholds (RL-RA / RL-LA: RL è GND, porta la lead a ~0 o molto bassa ampiezza)
    # Alzato da 15/20 a 25/40 per catturare più casi reali
    "flatline_std_thr": 25.0,
    "flatline_ptp_thr": 40.0,
    
    # Low Energy Thresholds (QRS presence)
    # Holter ha segnali piu' deboli: abbassato da 35/50 a 15/25
    # Alcune inversioni (LA-LL, RA-LL) riducono l'ampiezza di alcune lead
    "std_low_limb": 15.0,
    "std_low_prec": 25.0,
    
    # Noise (MAD threshold)
    # Holter e' piu' rumoroso: alzato da 15/20 a 25/35
    "mad_noise_limb": 25.0,
    "mad_noise_prec": 35.0,

    # Smoothness (QRS activity check)
    # Se il segnale e' TROPPO liscio significa che non c'e' attivita' QRS.
    # Rileva il cavo RL usato come RA/LA (RL-RA, RL-LA): il cavo di riferimento
    # non registra un ECG cardiaco proprio e ha derivata con variabilita' molto bassa.
    # Valore calibrato sui casi RL-RA/RL-LA osservati (mad_diff tipico < 3.0).
    "min_mad_diff_limb": 1.5,   # lead periferico con mad < 1.5 -> nessun QRS
    "min_mad_diff_prec": 1.5,   # lead precordiale
    
    # Low-amplitude check (is_low_amplitude)
    # Catches RL-RA / RL-LA cases where the signal is not strictly flat but has
    # systematically near-zero absolute amplitude (no real cardiac contribution).
    # Calibrate via test_sqa: raise if too many false-rejections on classes 0-5,
    # lower if classes 6/7 still sneak through.
    "near_zero_median_thr": 14.0,

    # Aggregation rules
    "min_valid_ratio": 0.60,   # abbassato da 0.70 per Holter (più finestre noisy accettabili)
    "flatline_ratio_thr": 0.80,
    "noisy_ratio_thr": 0.70,
    "min_valid_leads": 10      # usato solo come base per il riscalamento in check_ecg_quality
}
