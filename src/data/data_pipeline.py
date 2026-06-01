import os
import sys
import numpy as np
import pyedflib
import tempfile
import gc
from scipy import signal
from tqdm.auto import tqdm
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import (
    FS_OLD, FS_NEW, WIN_SEC, STRIDE_SEC, SAMPLES_PER_WINDOW, STRIDE_SAMPLES,
    LIMB_LEADS, PRECORDIAL_LEADS, ALL_LEADS, MAPPING_INV, PRECORDIAL_MAPPING,
    ACTIVE_SYNTH_CLASSES, QUALITY_CFG
)

<<<<<<< Updated upstream
# Cartella locale dove sono i file EDF estratti
LOCAL_DATASETS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'dataset')
=======
# --- S3 CONFIGURATION ---
from dotenv import load_dotenv

# Carica le credenziali dal file .env nella directory principale
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=env_path)

BUCKET = "btw-ml-data"
BASE_PATH = "polito-thesis/"
ECG_PATH = BASE_PATH + "record%s.edf"

# Cartella locale dove sono i file EDF scaricati dalla console S3
LOCAL_DATASETS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'datasets', 'dataset')
>>>>>>> Stashed changes

def _parse_edf_file(filepath):
    """
    Legge un file EDF da disco e lo converte in dict con le derivazioni.
    Applica una pulizia dei nomi delle label per matchare ALL_LEADS.
    """
    f = pyedflib.EdfReader(filepath)
    try:
        n_sig = f.signals_in_file
        mapping = {
            'AVR': 'aVr', 'AVL': 'aVl', 'AVF': 'aVf',
            'DI': 'I', 'DII': 'II', 'DIII': 'III'
        }
        labels_clean = []
        for raw in f.getSignalLabels():
            clean = raw.strip().upper().replace("ECG", "").replace("LEAD", "").replace("-", "").strip()
            labels_clean.append(mapping.get(clean, clean))

        sigs = [f.readSignal(i) for i in range(n_sig)]
        duration = f.getFileDuration()
    finally:
        f.close()

    signals_raw = {labels_clean[i]: sigs[i] for i in range(len(labels_clean))}
    signals_ordered = {lead: signals_raw[lead] for lead in ALL_LEADS if lead in signals_raw}

    return {
        "signals": signals_ordered,
        "duration_sec": duration
    }

def read_edf_data(bytes_data):
    """
    Legge un file EDF in bytes e lo converte in dict con le derivazioni.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
        tmp_file.write(bytes_data)
        tmp_path = tmp_file.name
    try:
        return _parse_edf_file(tmp_path)
    finally:
        os.remove(tmp_path)

def get_ecg(ecgId):
    """
    Carica un ECG per ID. Cerca in locale (datasets/dataset/record{id}.edf)
    """
    local_path = os.path.join(LOCAL_DATASETS_DIR, f"record{ecgId}.edf")
    if os.path.exists(local_path):
        # print(f"[LOCAL] Lettura da: {local_path}")
        return _parse_edf_file(local_path)

    return None

# --- PREPROCESSING PIPELINE ---

def bandpass_filter(signal_data, fs=FS_OLD, lowcut=0.5, highcut=120.0, order=4):
    """
    Filtro passabanda. 
    Highcut settato a 120Hz per mantenere più informazione clinica (Holter).
    Utilizziamo sosfiltfilt (zero-phase) per evitare distorsioni.
    """
    sos = signal.butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, signal_data)

def leads_preprocessing(raw_signal):
    """
    Preprocessa un singolo segnale (1 derivazione).
    Ordina: Notch -> Bandpass -> Downsampling
    """
    x = np.asarray(raw_signal, dtype=np.float64).squeeze()
    if x.ndim != 1 or x.size == 0:
        return np.array([], dtype=np.float32)
        
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # 1. Notch Filter (50Hz Powerline Interference)
    # iirnotch con Q=30, poi applicato con filtfilt (zero-phase, nessuna distorsione di fase)
    b_notch, a_notch = signal.iirnotch(50.0, 30.0, fs=FS_OLD)
    x = signal.filtfilt(b_notch, a_notch, x)

    # 2. Bandpass Filter (0.5Hz - 120Hz)
    x = bandpass_filter(x, fs=FS_OLD, lowcut=0.5, highcut=120.0)

    # 3. Downsampling a FS_NEW (es. 250Hz)
    x = signal.resample_poly(x, up=FS_NEW, down=FS_OLD)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)

def all_leads_preprocessing(signals_dict):
    return {lead: leads_preprocessing(sig) for lead, sig in signals_dict.items()}

# --- AUGMENTATION (MISPLACEMENTS) ---

def apply_electrode_gain(
    signals_dict,
    fs=FS_OLD,
    gain_range=(0.8, 1.2),      # Moltiplicatore random per elettrodo
    emg_sigma_range=(2.0, 15.0), # Rumore leggero
    wander_amp_range=(10.0, 80.0),
    noise_multiplier=1.0        # AGGIUNTO: moltiplicatore per rumore specifico
):
    """
    Applica guadagno variabile e rumore leggero agli elettrodi fisici (RA, LA, LL).
    
    Questa augmentation simula:
    1. Differenze di contatto/impedenza (Gain) -> risolve confusioni geometriche.
    2. Rumore ambientale leggero -> aggiunge robustezza.
    """
    limb_ref = next((signals_dict[l] for l in ('I', 'II', 'III') if l in signals_dict), None)
    if limb_ref is None:
        return signals_dict

    n_samples = len(limb_ref)
    t = np.arange(n_samples) / fs

    # Generazione parametri indipendenti per RA, LA, LL
    elec_mods = {}
    for elec in ('RA', 'LA', 'LL'):
        gain   = np.random.uniform(*gain_range)
        sigma  = np.random.uniform(*emg_sigma_range) * noise_multiplier
        emg    = np.random.normal(0.0, sigma, n_samples)
        amp    = np.random.uniform(*wander_amp_range) * noise_multiplier
        freq   = np.random.uniform(0.1, 0.4)
        phase  = np.random.uniform(0.0, 2.0 * np.pi)
        wander = amp * np.sin(2.0 * np.pi * freq * t + phase)
        elec_mods[elec] = {'gain': gain, 'noise': emg + wander}

    g_ra, n_ra = elec_mods['RA']['gain'], elec_mods['RA']['noise']
    g_la, n_la = elec_mods['LA']['gain'], elec_mods['LA']['noise']
    g_ll, n_ll = elec_mods['LL']['gain'], elec_mods['LL']['noise']

    noisy = {}
    for lead, sig in signals_dict.items():
        s = np.asarray(sig, dtype=np.float64)
        if lead == 'I':   # LA - RA
            noisy[lead] = (s * ((g_la + g_ra)/2)) + (n_la - n_ra)
        elif lead == 'II':  # LL - RA
            noisy[lead] = (s * ((g_ll + g_ra)/2)) + (n_ll - n_ra)
        elif lead == 'III': # LL - LA
            noisy[lead] = (s * ((g_ll + g_la)/2)) + (n_ll - n_la)
        elif lead == 'aVr':
            noisy[lead] = (s * ((g_la + g_ll + 2*g_ra)/4)) - (n_la + n_ll - 2*n_ra)/2
        elif lead == 'aVl':
            noisy[lead] = (s * ((2*g_la + g_ra + g_ll)/4)) + (n_la - n_ra/2 - n_ll/2)
        elif lead == 'aVf':
            noisy[lead] = (s * ((2*g_ll + g_ra + g_la)/4)) + (n_ll - n_ra/2 - n_la/2)
        else:
            noisy[lead] = sig
    return noisy


def add_extra_noise(signals_dict, multiplier=1.5, fs=FS_OLD):
    """
    Aggiunge un surplus di rumore EMG e Baseline Wander a un segnale già esistente.
    Utile per differenziare le classi più rumorose (es. ROT_ANT).
    """
    noisy = {}
    n_samples = len(next(iter(signals_dict.values())))
    t = np.arange(n_samples) / fs
    
    for lead, sig in signals_dict.items():
        # Calibrato per alzare il MAD di circa 5-7 uV
        sigma = np.random.uniform(5.0, 15.0) * (multiplier - 1.0)
        emg = np.random.normal(0.0, sigma, n_samples)
        
        amp = np.random.uniform(20.0, 50.0) * (multiplier - 1.0)
        freq = np.random.uniform(0.1, 0.4)
        phase = np.random.uniform(0.0, 2.0 * np.pi)
        wander = amp * np.sin(2.0 * np.pi * freq * t + phase)
        
        noisy[lead] = sig + emg + wander
        
    return noisy


def limb_interchange_simulation(mode, signals_dict):
    """
    Simula misplacements periferici (Classi 1-7).
    """
    leads = np.array([signals_dict[l] for l in LIMB_LEADS])
    transformed = np.copy(leads)

    if mode == 1: # LA-RA
        transformed[0], transformed[1], transformed[2] = -leads[0], leads[2], leads[1]
        transformed[3], transformed[4] = leads[4], leads[3]
    elif mode == 2: # RA-LL
        transformed[0], transformed[1], transformed[2] = -leads[2], -leads[1], -leads[0]
        transformed[3], transformed[5] = leads[5], leads[3]
    elif mode == 3: # LA-LL
        transformed[0], transformed[1], transformed[2] = leads[1], leads[0], -leads[2]
        transformed[4], transformed[5] = leads[5], leads[4]
    elif mode == 4: # Rotazione Oraria
        transformed[0], transformed[1], transformed[2] = leads[2], -leads[0], -leads[1]
        transformed[3], transformed[4], transformed[5] = leads[4], leads[5], leads[3]
    elif mode == 5: # Rotazione Antioraria
        transformed[0], transformed[1], transformed[2] = -leads[1], -leads[2], leads[0]
        transformed[3], transformed[4], transformed[5] = leads[5], leads[3], leads[4]
    elif mode == 6:  # RL-RA
        transformed[0], transformed[1], transformed[2] = leads[1], leads[2], leads[2] - leads[1]
        transformed[3] = -0.5 * (transformed[0] + transformed[1])
        transformed[4] = transformed[0] - 0.5 * transformed[1]
        transformed[5] = transformed[1] - 0.5 * transformed[0]
    elif mode == 7:  # RL-LA
        transformed[1] = leads[1]
        transformed[0] = leads[2] - leads[1]
        transformed[2] = transformed[1] - transformed[0]
        transformed[3] = -0.5 * (transformed[0] + transformed[1])
        transformed[4] = transformed[0] - 0.5 * transformed[1]
        transformed[5] = transformed[1] - 0.5 * transformed[0]
    
    new_sigs = {name: transformed[i] for i, name in enumerate(LIMB_LEADS)}
    for l in signals_dict:
        if l not in LIMB_LEADS: new_sigs[l] = signals_dict[l]
    return new_sigs

def precordial_interchange_simulation(mode, signals_dict):
    """
    Simula misplacements precordiali (Classi 1-15).
    """
    all_leads = np.stack([signals_dict[name] for name in signals_dict.keys()])
    transformed = np.copy(all_leads)
    
    if 1 <= mode <= 15:
        # Mappatura statica per scambi
        swaps = {
            1: (6,7), 2: (6,8), 3: (6,9), 4: (6,10), 5: (6,11),
            6: (7,8), 7: (7,9), 8: (7,10), 9: (7,11),
            10: (8,9), 11: (8,10), 12: (8,11),
            13: (9,10), 14: (9,11), 15: (10,11)
        }
        idx1, idx2 = swaps[mode]
        transformed[idx1], transformed[idx2] = all_leads[idx2], all_leads[idx1]
        
    lead_names = list(signals_dict.keys())
    return {name: transformed[i] for i, name in enumerate(lead_names)}

# --- NOISE AUGMENTATION ---

def _apply_electrode_noise(signals_dict, noise_func, **kwargs):
    """
    Applica una funzione di generazione rumore a livello di elettrodo fisico
    per garantire la consistenza fisica (legge di Einthoven).
    """
    n_samples = len(next(iter(signals_dict.values())))
    
    # Genera rumore indipendente per gli elettrodi fisici
    # RL è il ground, lo consideriamo 0 (o il riferimento comune che si cancella).
    n_RA = noise_func(n_samples, **kwargs)
    n_LA = noise_func(n_samples, **kwargs)
    n_LL = noise_func(n_samples, **kwargs)
    n_V1 = noise_func(n_samples, **kwargs)
    n_V2 = noise_func(n_samples, **kwargs)
    n_V3 = noise_func(n_samples, **kwargs)
    n_V4 = noise_func(n_samples, **kwargs)
    n_V5 = noise_func(n_samples, **kwargs)
    n_V6 = noise_func(n_samples, **kwargs)
    
    # Deriva il rumore per ciascuna derivazione
    # WCT (Wilson Central Terminal)
    n_WCT = (n_RA + n_LA + n_LL) / 3.0
    
    noise_leads = {
        'I': n_LA - n_RA,
        'II': n_LL - n_RA,
        'III': n_LL - n_LA,
        'aVr': n_RA - (n_LA + n_LL) / 2.0,
        'aVl': n_LA - (n_RA + n_LL) / 2.0,
        'aVf': n_LL - (n_RA + n_LA) / 2.0,
        'V1': n_V1 - n_WCT,
        'V2': n_V2 - n_WCT,
        'V3': n_V3 - n_WCT,
        'V4': n_V4 - n_WCT,
        'V5': n_V5 - n_WCT,
        'V6': n_V6 - n_WCT,
    }
    
    noisy_signals = {}
    for lead, sig in signals_dict.items():
        if lead in noise_leads:
            noisy_signals[lead] = (np.array(sig, dtype=np.float32) + noise_leads[lead]).astype(np.float32)
        else:
            noisy_signals[lead] = np.array(sig, dtype=np.float32)
            
    return noisy_signals

def apply_electrode_gain(signals_dict, fs=FS_OLD, noise_multiplier=1.1):
    """
    Applica un gain (rumore fisico simulato) ai segnali crudi, rispettando
    la consistenza tra elettrodi (Einthoven).
    """
    global_std = np.mean([np.std(sig) for sig in signals_dict.values()])
    def _white_noise(n):
        # Usiamo il factor diviso per sqrt(2) approssimativamente,
        # perche' noise_I = n_LA - n_RA (varianza si somma)
        # std(Noise_I) = sqrt(2) * std(n_LA)
        return np.random.normal(0, (global_std * 0.05 * noise_multiplier) / 1.414, size=n)
        
    return _apply_electrode_noise(signals_dict, _white_noise)

def apply_random_scaling(signals_dict, min_scale=0.5, max_scale=1.5):
    """
    Moltiplica l'ampiezza dell'intero ECG per un fattore casuale.
    Aiuta la rete a diventare invariante all'ampiezza assoluta (risolvendo il drop di varianza).
    """
    scale = np.random.uniform(min_scale, max_scale)
    return {lead: (np.array(sig, dtype=np.float32) * scale) for lead, sig in signals_dict.items()}

def add_baseline_wander(signals_dict, fs=FS_OLD, intensity=300.0):
    """
    Aggiunge una lenta fluttuazione fisiologica (es. respiro).
    Utilizza rumore bianco filtrato passa-basso per renderlo stocastico
    e rispetta la geometria di Einthoven tra le derivazioni.
    """
    def _stochastic_wander(n):
        # Compensazione di ~4.0 empirica per mantenere std~200 come in precedenza
        white = np.random.normal(0, intensity * 4.0 / 1.414, size=n)
        # Lowpass a 0.5Hz per variazioni lente (respiro)
        b, a = signal.butter(2, 0.5, btype='low', fs=fs)
        return signal.filtfilt(b, a, white)
        
    return _apply_electrode_noise(signals_dict, _stochastic_wander)

def add_extra_noise(signals_dict, multiplier=1.5, fs=FS_OLD):
    """
    Aggiunge rumore extra differenziato: EMG (alta frequenza) e artefatti
    da movimento (salti a gradino smussati) con consistenza di Einthoven.
    """
    global_std = np.mean([np.std(sig) for sig in signals_dict.values()])
    
    def _complex_noise(n):
        # 1. Rumore EMG (burst ad alta frequenza)
        emg_white = np.random.normal(0, (global_std * 0.1 * multiplier) / 1.414, size=n)
        b2, a2 = signal.butter(2, [20, min(120, fs/2.5)], btype='bandpass', fs=fs)
        emg_noise = signal.filtfilt(b2, a2, emg_white)
        
        # 2. Artefatti da movimento (salti a gradino) - 20% di probabilita'
        motion = np.zeros(n)
        if np.random.rand() < 0.20:
            jump_pos = np.random.randint(int(fs*0.5), int(n - fs*0.5))
            jump_amp = np.random.choice([-1, 1]) * np.random.uniform(global_std*0.5, global_std*2.0)
            motion[jump_pos:] = jump_amp
            # Smussamento del gradino
            b3, a3 = signal.butter(1, 2.0, btype='low', fs=fs)
            motion = signal.filtfilt(b3, a3, motion)
            
        return emg_noise + motion

    return _apply_electrode_noise(signals_dict, _complex_noise)

# --- SIGNAL QUALITY ASSESSMENT (SQA) ---

def iter_windows_1d(x, win_size=SAMPLES_PER_WINDOW, stride=STRIDE_SAMPLES):
    for start in range(0, len(x) - win_size + 1, stride):
        yield x[start:start + win_size]

def is_limb_lead(index_or_name):
    if isinstance(index_or_name, str):
        return index_or_name in LIMB_LEADS
    return ALL_LEADS[index_or_name] in LIMB_LEADS

def check_physiological_ecg(window, fs=FS_NEW, min_qrs_amplitude=50.0):
    """
    Heuristic check to detect at least one heartbeat.
    Detects QRS-like peaks with physiologically plausible inter-peak distance.

    For RL-RA / RL-LA inversions the RL reference cable has no true cardiac
    morphology: even if not perfectly flat, it will show no separable QRS peaks
    above the adaptive threshold.
    """
    x = np.asarray(window, dtype=np.float64).squeeze()
    if len(x) < 3 or np.any(~np.isfinite(x)):
        return False, 0

    diff_x = np.abs(np.diff(x))
    mean_diff = np.mean(diff_x)
    if mean_diff == 0:
        return False, 0

    # Adaptive threshold: sensitive enough to pick up small but real QRS events.
    # min_qrs_amplitude / 10 (lowered from /8) to be more morphology-sensitive.
    threshold = max(mean_diff * 4, min_qrs_amplitude / 10)
    peaks = np.where(diff_x > threshold)[0]
    if len(peaks) == 0:
        return False, 0

    # Count peaks separated by at least 0.3 s (200 bpm upper limit)
    actual_peaks = 1
    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i - 1] > (fs * 0.3):
            actual_peaks += 1

    # At least 1 beat, at most 6 in 2 s window (300 bpm), and ptp above threshold
    if 1 <= actual_peaks <= 6 and np.ptp(x) > min_qrs_amplitude:
        return True, actual_peaks

    return False, 0

def compute_window_features(window, cfg=QUALITY_CFG, lead_idx=None):
    x = np.asarray(window, dtype=np.float64).squeeze()
    if len(x) == 0 or np.any(~np.isfinite(x)): return None

    std_val = float(np.std(x))
    ptp_val = float(np.ptp(x))
    median_abs = float(np.median(np.abs(x)))   # robust amplitude estimator
    clip_ratio = float(np.mean(np.abs(x) >= cfg["amplitude_max"]))
    max_step = float(np.max(np.abs(np.diff(x)))) if len(x) > 1 else 0.0
    baseline_drift = float(np.abs(np.mean(x)))

    dx = np.diff(x)
    mad_diff = float(np.median(np.abs(dx - np.median(dx)))) if len(dx) > 0 else 0.0
    has_heartbeat, n_beats = check_physiological_ecg(x, fs=cfg["fs"])

    return {
        "std": std_val, "ptp": ptp_val, "median_abs": median_abs,
        "clip_ratio": clip_ratio, "max_step": max_step,
        "baseline_drift": baseline_drift,
        "mad_diff": mad_diff, "has_heartbeat": bool(has_heartbeat), "n_beats": int(n_beats)
    }

def check_window_quality(window, cfg=QUALITY_CFG, lead_idx=None):
    feats = compute_window_features(window, cfg=cfg, lead_idx=lead_idx)
    if feats is None:
        return {"valid": False, "reason": "feature_fail", "checks": {}, "n_beats": 0}

    is_saturated      = feats["clip_ratio"] > cfg["clip_ratio_thr"]
    is_step_reset     = feats["max_step"]   > cfg["adc_step_limit"]
    is_baseline_wander = feats["baseline_drift"] > cfg["baseline_max_uv"]
    is_flatline       = (feats["std"] < cfg["flatline_std_thr"]) and \
                        (feats["ptp"] < cfg["flatline_ptp_thr"])
    is_no_heartbeat   = not feats["has_heartbeat"]

    std_low_thr = cfg["std_low_limb"] if is_limb_lead(lead_idx) else cfg["std_low_prec"]
    is_low_energy = feats["std"] < std_low_thr

    mad_noise_thr = cfg["mad_noise_limb"] if is_limb_lead(lead_idx) else cfg["mad_noise_prec"]
    is_noise = feats["mad_diff"] > mad_noise_thr

    # --- NEW: Low-amplitude check (RL-RA / RL-LA detection) ---
    # The RL reference cable produces a signal with near-zero absolute amplitude
    # (the median of |x| is systematically low even when std/ptp barely exceed
    # the flatline thresholds due to noise).
    is_low_amplitude = feats["median_abs"] < cfg["near_zero_median_thr"]

    # --- NEW: No-morphology check ---
    # min_mad_diff guards against signals that are too smooth to contain a QRS.
    # RL-RA / RL-LA signals have very little high-frequency variability because
    # the reference cable acts as a near-DC offset, not a cardiac electrode.
    min_mad = cfg.get("min_mad_diff_limb", 3.5) if is_limb_lead(lead_idx) \
              else cfg.get("min_mad_diff_prec", 2.0)
    is_no_morphology = feats["mad_diff"] < min_mad

    checks = {
        "flatline":       is_flatline,
        "low_energy":     is_low_energy,
        "low_amplitude":  is_low_amplitude,
        "no_morphology":  is_no_morphology,
        "noise":          is_noise,
        "clipping":       is_saturated,
        "adc_step":       is_step_reset,
        "baseline_drift": is_baseline_wander,
        "no_heartbeat":   is_no_heartbeat,
    }

    is_bad = any(checks.values())
    fail_reasons = [k for k, v in checks.items() if v]

    return {
        "valid": not is_bad,
        "reason": ", ".join(fail_reasons) if fail_reasons else "OK",
        "checks": checks,
        "n_beats": feats["n_beats"],
        "feats": feats,   # expose raw features for verbose test output
    }

def check_lead_quality_global(lead_signal, cfg=QUALITY_CFG, lead_idx=None):
    x = np.asarray(lead_signal, dtype=np.float64).squeeze()
    if len(x) == 0 or np.any(~np.isfinite(x)):
        return {"valid": False, "reason": "invalid_lead"}

    std_val    = float(np.std(x))
    ptp_val    = float(np.ptp(x))
    median_abs = float(np.median(np.abs(x)))
    clip_ratio = float(np.mean(np.abs(x) >= cfg["amplitude_max"]))
    dx = np.diff(x)
    mad_diff = float(np.median(np.abs(dx - np.median(dx)))) if len(dx) > 0 else 0.0

    is_flatline  = (std_val < cfg["flatline_std_thr"]) and (ptp_val < cfg["flatline_ptp_thr"])
    is_clipping  = clip_ratio > cfg["clip_ratio_thr"]

    std_low_thr   = cfg["std_low_limb"]    if is_limb_lead(lead_idx) else cfg["std_low_prec"]
    mad_noise_thr = cfg["mad_noise_limb"] * 1.1 if is_limb_lead(lead_idx) \
                    else cfg["mad_noise_prec"] * 1.1
    min_mad = cfg.get("min_mad_diff_limb", 3.5) if is_limb_lead(lead_idx) \
              else cfg.get("min_mad_diff_prec", 2.0)

    is_low_energy    = std_val    < std_low_thr
    is_noise         = mad_diff   > mad_noise_thr
    # Low-amplitude: robust global version (same concept as per-window)
    is_low_amplitude = median_abs < cfg["near_zero_median_thr"]
    # No-morphology: globally too smooth → no QRS activity across whole lead
    is_no_morphology = mad_diff   < min_mad
    # Noisy-flatline: segnale con zero contenuto cardiaco reale ma enormi picchi di rumore
    is_noisy_flatline = (median_abs < 26.0) and (ptp_val > 1000 or mad_diff > 18.0)

    checks = {
        "flatline":      is_flatline,
        "low_energy":    is_low_energy,
        "low_amplitude": is_low_amplitude,
        "no_morphology": is_no_morphology,
        "noisy_flatline":is_noisy_flatline,
        "noise":         is_noise,
        "clipping":      is_clipping,
    }

    is_bad = any(checks.values())
    fail_reasons = [k for k, v in checks.items() if v]

    return {
        "valid": not is_bad,
        "reason": ", ".join(fail_reasons) if fail_reasons else "OK",
        "checks": checks,
        "std": std_val, "ptp": ptp_val, "median_abs": median_abs,
        "clip_ratio": clip_ratio, "mad_diff": mad_diff
    }

def check_lead_quality(lead_signal, cfg=QUALITY_CFG, lead_idx=None):
    x = np.asarray(lead_signal, dtype=np.float64).squeeze()
    global_result = check_lead_quality_global(x, cfg=cfg, lead_idx=lead_idx)

    summary = {
        "n_windows": 0, "n_valid": 0,
        "n_flatline": 0, "n_low_energy": 0, "n_low_amplitude": 0, "n_no_morphology": 0,
        "n_noise": 0, "n_clipping": 0, "n_adc_step": 0, "n_baseline_drift": 0, "n_no_heartbeat": 0,
        "flatline_ratio": 0.0, "low_amplitude_ratio": 0.0, "no_morphology_ratio": 0.0,
        "noise_ratio": 0.0, "clipping_ratio": 0.0, "no_heartbeat_ratio": 0.0,
    }

    if not global_result["valid"]:
        return {
            "valid": False, 
            "reason": f"global_fail: {global_result['reason']}", 
            "global_result": global_result, 
            "summary": summary
        }

    for window in iter_windows_1d(x):
        res = check_window_quality(window, cfg=cfg, lead_idx=lead_idx)
        summary["n_windows"] += 1
        if res["valid"]: summary["n_valid"] += 1
        for key in res["checks"]:
            if res["checks"][key]:
                summary_key = f"n_{key}"
                if summary_key in summary: summary[summary_key] += 1

    n_win = summary["n_windows"]
    if n_win > 0:
        summary["flatline_ratio"]     = summary["n_flatline"]     / n_win
        summary["low_amplitude_ratio"] = summary["n_low_amplitude"] / n_win
        summary["no_morphology_ratio"] = summary["n_no_morphology"] / n_win
        summary["noise_ratio"]        = summary["n_noise"]        / n_win
        summary["clipping_ratio"]     = summary["n_clipping"]     / n_win
        summary["no_heartbeat_ratio"] = summary["n_no_heartbeat"] / n_win

        valid = (summary["n_valid"] / n_win >= cfg["min_valid_ratio"])
        reason = "OK" if valid else "low_valid_ratio"
    else:
        valid = False
        reason = "no_windows"

    return {"valid": valid, "reason": reason, "global_result": global_result, "summary": summary}

def check_ecg_quality(ecg, cfg=QUALITY_CFG, lead_indices=None):
    """
    Valuta la qualità dell'ECG sulle derivazioni specificate.
    lead_indices: lista di indici (0-11) da controllare.
                  None = tutte e 12 le lead.
                  [0..5] = solo periferiche (LIMB).
                  [6..11] = solo precordiali.
    min_valid_leads viene riscalato proporzionalmente al subset.
    """
    ecg = np.asarray(ecg)
    if len(ecg.shape) == 2 and ecg.shape[0] > ecg.shape[1]: 
        ecg = ecg.T

    indices = lead_indices if lead_indices is not None else list(range(12))
    # Scala il requisito di lead valide: mantieni la stessa proporzione (10/12 ≈ 83%)
    base_min = cfg["min_valid_leads"]       # default 10 su 12
    min_valid = max(1, round(base_min * len(indices) / 12))

    lead_results = []
    for i in indices:
        res = check_lead_quality(ecg[i], cfg=cfg, lead_idx=i)
        res["lead_idx"] = i
        lead_results.append(res)

    valid_leads = sum(r["valid"] for r in lead_results)
    invalid_lead_indices = [r["lead_idx"] for r in lead_results if not r["valid"]]

    valid_ecg = (valid_leads >= min_valid)
    reason = "OK" if valid_ecg else "too_many_bad_leads"

    return {
        "global_valid": valid_ecg,
        "valid_leads": valid_leads,
        "total_leads": len(indices),
        "invalid_lead_indices": invalid_lead_indices,
        "reason": reason,
        "lead_results": lead_results
    }
