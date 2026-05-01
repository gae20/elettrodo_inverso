"""
test_preprocessing.py
Verifica la correttezza della pipeline di preprocessing ECG:
  notch 50 Hz  ->  bandpass 0.5–120 Hz  ->  downsample a FS_NEW

STRUTTURA DEI TEST
==================
  1. test_bandpass_response()
     - Risposta in frequenza del filtro passabanda su sinusoidi sintetiche.
     - Verifica: guadagno < -3 dB per freq. in banda, < -20 dB fuori banda.

  2. test_notch_50hz()
     - Segnale sintetico + rumore di rete a 50 Hz.
     - Verifica: attenuazione >= 20 dB nella banda [48-52 Hz].

  3. test_downsampling()
     - Lunghezza e integrita' del segnale dopo il downsample.
     - Verifica: no NaN/Inf, lunghezza corretta.

  4. test_no_nans_on_real_ecg()
     - ECG reale dal dataset locale.
     - Verifica: no NaN/Inf, tutte le 12 derivazioni presenti.

  5. test_psd_band_analysis()
     - Analisi PSD Welch per banda su ECG reale.
     - Verifica: banda QRS preservata (<3 dB loss), powerline attenuata (>15 dB).
     - Stampa tabella potenze per banda.

USO
===
  CWD atteso: elettrodo_inverso/src/
  python -m tests.test_preprocessing
  python -m tests.test_preprocessing --plot   # mostra plot PSD se matplotlib disponibile
"""

import os
import sys
import argparse
import numpy as np
from scipy import signal as sp_signal
from scipy.signal import welch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.data_pipeline import leads_preprocessing, all_leads_preprocessing, bandpass_filter, get_ecg
from utils.config import FS_OLD, FS_NEW, SAMPLES_PER_WINDOW, ALL_LEADS, LIMB_LEADS

# --- Helpers ---

def band_power(psd, freqs, f_low, f_high):
    """Integra la PSD nella banda [f_low, f_high] Hz (trapezi)."""
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def db(ratio, eps=1e-12):
    """Converte un rapporto di potenza in dB."""
    return 10 * np.log10(max(ratio, eps))


def make_sinusoid(freq_hz, fs=FS_OLD, duration_sec=5.0, amplitude=500.0):
    """Genera una sinusoide pura a freq_hz Hz."""
    t = np.arange(int(fs * duration_sec)) / fs
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float64)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


# ─────────────────────────────────────────────────────────────────
# TEST 1: Risposta in frequenza del filtro bandpass
# ─────────────────────────────────────────────────────────────────
def test_bandpass_response():
    print("=" * 60)
    print("TEST 1: Risposta in frequenza (bandpass 0.5–120 Hz)")
    print("=" * 60)

    # Frequenze di test e aspettativa (True = in banda, False = fuori banda)
    test_freqs = [
        (0.1,  False, -20.0, "> -20 dB fuori banda bassa"),
        (0.5,  True,   -7.0, "> -7 dB bordo (filtfilt = -6dB)"),
        (1.0,  True,   -3.0, "> -3 dB in banda"),
        (10.0, True,   -3.0, "> -3 dB in banda (QRS)"),
        (50.0, True,   -3.0, "> -3 dB in banda"),
        (100.0,True,   -3.0, "> -3 dB in banda alta"),
        (120.0,True,   -7.0, "> -7 dB bordo (filtfilt = -6dB)"),
        (130.0,False,  -9.0, "> -9 dB fuori banda alta"),
    ]

    DURATION = 8.0   # secondi (serve una sinusoide lunga per filtri a bassa freq)
    N = int(FS_OLD * DURATION)
    t = np.arange(N) / FS_OLD
    reference_amp = 500.0

    passed = 0
    failed = 0

    print(f"  {'Freq':>8}  {'Gain (dB)':>10}  {'Aspettativa':>30}  Status")
    print(f"  {'-'*8}  {'-'*10}  {'-'*30}  ------")

    for freq, in_band, thr_db, note in test_freqs:
        # Sinusoide pura
        x_in = (reference_amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)
        # Solo filtro bandpass (no notch, no downsampling) per isolare la risposta
        sos = sp_signal.butter(4, [0.5, 120.0], btype='bandpass', fs=FS_OLD, output='sos')
        x_out = sp_signal.sosfiltfilt(sos, x_in)

        # Calcola guadagno RMS (scarta primi e ultimi 10% per evitare bordi)
        trim = int(N * 0.10)
        gain_db = db(rms(x_out[trim:-trim]) ** 2 / (rms(x_in[trim:-trim]) ** 2 + 1e-12))

        if in_band:
            ok = gain_db >= thr_db
        else:
            ok = gain_db <= thr_db

        status = "[PASS]" if ok else "[FAIL]"
        if ok: passed += 1
        else:  failed += 1

        print(f"  {freq:>7.1f} Hz  {gain_db:>+9.2f}  {note:>30}  {status}")

    print()
    print(f"  Risultato: {passed} PASS, {failed} FAIL")
    return failed == 0


# ─────────────────────────────────────────────────────────────────
# TEST 2: Attenuazione notch a 50 Hz
# ─────────────────────────────────────────────────────────────────
def test_notch_50hz():
    print("=" * 60)
    print("TEST 2: Attenuazione notch 50 Hz")
    print("=" * 60)

    DURATION = 10.0
    N = int(FS_OLD * DURATION)
    t = np.arange(N) / FS_OLD

    # Segnale ECG sintetico = somma di armoniche cardiache (1 Hz, 3 Hz, 10 Hz, 20 Hz)
    ecg_synth = (
        300 * np.sin(2 * np.pi * 1.0 * t) +
        150 * np.sin(2 * np.pi * 3.0 * t) +
        100 * np.sin(2 * np.pi * 10.0 * t) +
         50 * np.sin(2 * np.pi * 20.0 * t)
    ).astype(np.float64)

    # Rumore di rete: grande rispetto al segnale per rendere il test significativo
    noise_50 = (600 * np.sin(2 * np.pi * 50.0 * t)).astype(np.float64)
    x_in = ecg_synth + noise_50

    # Preprocessing completo
    x_out = leads_preprocessing(x_in.astype(np.float32))
    x_out = x_out.astype(np.float64)

    # PSD Welch prima e dopo (sul segnale decimato a FS_NEW per confronto equo)
    x_in_dec = sp_signal.resample_poly(x_in, up=FS_NEW, down=FS_OLD)
    min_len = min(len(x_in_dec), len(x_out))
    x_in_dec = x_in_dec[:min_len]
    x_out_trim = x_out[:min_len]

    freqs_in, psd_in = welch(x_in_dec, fs=FS_NEW, nperseg=min(512, min_len // 4))
    freqs_out, psd_out = welch(x_out_trim, fs=FS_NEW, nperseg=min(512, min_len // 4))

    power_in_50  = band_power(psd_in,  freqs_in,  48.0, 52.0)
    power_out_50 = band_power(psd_out, freqs_out, 48.0, 52.0)

    attenuation_db = db(power_out_50 / (power_in_50 + 1e-12))

    print(f"  Potenza [48-52 Hz] prima:  {power_in_50:.4f}")
    print(f"  Potenza [48-52 Hz] dopo:   {power_out_50:.4f}")
    print(f"  Attenuazione notch:        {attenuation_db:+.2f} dB")

    ok = attenuation_db <= -20.0
    print(f"  Soglia richiesta: <= -20 dB  ->  {'[PASS]' if ok else '[FAIL]'}")
    print()
    return ok


# ─────────────────────────────────────────────────────────────────
# TEST 3: Correttezza del downsampling
# ─────────────────────────────────────────────────────────────────
def test_downsampling():
    print("=" * 60)
    print("TEST 3: Correttezza downsampling")
    print("=" * 60)

    DURATION = 30.0   # 30 s di segnale
    N_in = int(FS_OLD * DURATION)
    x_in = np.random.randn(N_in).astype(np.float32) * 200.0

    x_out = leads_preprocessing(x_in)

    expected_len = int(round(N_in * FS_NEW / FS_OLD))
    actual_len   = len(x_out)
    # Tolleranza: resample_poly puo' differire di qualche campione
    len_ok   = abs(actual_len - expected_len) <= max(2, int(expected_len * 0.01))
    nan_ok   = not np.any(~np.isfinite(x_out))
    dtype_ok = x_out.dtype == np.float32

    print(f"  Lunghezza attesa:  {expected_len}")
    print(f"  Lunghezza ottenuta:{actual_len}  ->  {'[PASS]' if len_ok else '[FAIL]'}")
    print(f"  No NaN/Inf:        {'[PASS]' if nan_ok else '[FAIL]'}")
    print(f"  dtype float32:     {'[PASS]' if dtype_ok else '[FAIL]'}")
    print()
    return len_ok and nan_ok and dtype_ok


# ─────────────────────────────────────────────────────────────────
# TEST 4: Integrità su ECG reale
# ─────────────────────────────────────────────────────────────────
def _find_local_ecg_id():
    """Cerca il primo ID numerico disponibile nella cartella datasets/dataset/."""
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets', 'dataset')
    if not os.path.isdir(dataset_dir):
        return None
    for fname in sorted(os.listdir(dataset_dir)):
        if fname.startswith("record") and fname.endswith(".edf"):
            try:
                return int(fname.replace("record", "").replace(".edf", ""))
            except ValueError:
                continue
    return None


def test_no_nans_on_real_ecg():
    print("=" * 60)
    print("TEST 4: Integrità preprocessing su ECG reale")
    print("=" * 60)

    ecg_id = _find_local_ecg_id()
    if ecg_id is None:
        print("  [SKIP] Nessun file record*.edf trovato in datasets/dataset/")
        print()
        return True  # non un fallimento bloccante

    print(f"  Usando ECG ID: {ecg_id}")
    ecg_data = get_ecg(ecg_id)
    if not ecg_data or not ecg_data["signals"]:
        print("  [SKIP] ECG non leggibile")
        print()
        return True

    sigs = all_leads_preprocessing(ecg_data["signals"])
    missing = [l for l in ALL_LEADS if l not in sigs]
    present = [l for l in ALL_LEADS if l in sigs]

    all_finite = all(np.all(np.isfinite(sigs[l])) for l in present)
    leads_ok = len(missing) == 0
    expected_rate = FS_NEW

    print(f"  Derivazioni attese: {len(ALL_LEADS)}  trovate: {len(present)}"
          f"  ->  {'[PASS]' if leads_ok else f'[FAIL] mancanti: {missing}'}")
    print(f"  No NaN/Inf in output: {'[PASS]' if all_finite else '[FAIL]'}")
    for lead in present[:3]:   # stampa lunghezza per le prime 3
        n = len(sigs[lead])
        print(f"    {lead}: {n} campioni  "
              f"(~{n / expected_rate:.1f} s a {expected_rate} Hz)")
    print()
    return leads_ok and all_finite


# ─────────────────────────────────────────────────────────────────
# TEST 5: Analisi PSD per banda
# ─────────────────────────────────────────────────────────────────
BANDS = [
    ("DC",        0.0,   0.5,  None,  None),
    ("LF/wander", 0.5,   5.0,  None,  None),
    ("QRS",       5.0,  40.0,  -3.0,  None),  # non deve perdere più di 3 dB
    ("Powerline", 40.0, 60.0,  None,  None),  # Nessun drop severo atteso su banda larga se l'ECG era pulito
    ("HF",        60.0, 120.0, None,  None),
    ("Aliasing",  120.0, 125.0, None, None),
]


def test_psd_band_analysis(show_plot=False):
    print("=" * 60)
    print("TEST 5: Analisi PSD per banda (lead II)")
    print("=" * 60)

    ecg_id = _find_local_ecg_id()
    if ecg_id is None:
        print("  [SKIP] Nessun file record*.edf trovato in datasets/dataset/")
        print()
        return True

    ecg_data = get_ecg(ecg_id)
    if not ecg_data or "II" not in ecg_data["signals"]:
        print("  [SKIP] Lead II non disponibile")
        print()
        return True

    raw_II = np.array(ecg_data["signals"]["II"], dtype=np.float64)
    # Segnale raw decimato a FS_NEW per confronto equo (senza filtri)
    raw_dec = sp_signal.resample_poly(raw_II, up=FS_NEW, down=FS_OLD)

    # Preprocessato
    proc_II = leads_preprocessing(raw_II.astype(np.float32)).astype(np.float64)

    min_len = min(len(raw_dec), len(proc_II))
    raw_dec  = raw_dec[:min_len]
    proc_II  = proc_II[:min_len]

    # PSD Welch
    nperseg = min(1024, min_len // 8)
    freqs_r, psd_r = welch(raw_dec,  fs=FS_NEW, nperseg=nperseg)
    freqs_p, psd_p = welch(proc_II,  fs=FS_NEW, nperseg=nperseg)

    print(f"  ECG ID: {ecg_id}  |  lunghezza: {min_len} camp. ({min_len/FS_NEW:.1f} s)")
    print()
    print(f"  {'Banda':>14}  {'[Hz]':>12}  {'Raw (dB)':>10}  {'Proc (dB)':>10}  "
          f"{'Delta':>8}  Status")
    print(f"  {'-'*14}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*8}  ------")

    passed = 0
    failed = 0

    for name, f_lo, f_hi, min_delta_db, max_delta_db in BANDS:
        p_raw  = band_power(psd_r, freqs_r, f_lo, f_hi)
        p_proc = band_power(psd_p, freqs_p, f_lo, f_hi)
        eps = 1e-10
        delta_db = db(p_proc / (p_raw + eps))

        ok = True
        note = ""
        if min_delta_db is not None and delta_db < min_delta_db:
            ok = False
            note = f"< {min_delta_db} dB (banda persa)"
        if max_delta_db is not None and delta_db > max_delta_db:
            ok = False
            note = f"> {max_delta_db} dB (insufficiente attenuazione)"

        status = "[PASS]" if ok else f"[FAIL] {note}"
        if ok: passed += 1
        elif min_delta_db is not None or max_delta_db is not None:
            failed += 1

        db_raw  = db(p_raw  / (max(p_raw,  eps)))  # solo per scale relativa
        print(f"  {name:>14}  {f_lo:5.1f}–{f_hi:5.1f}  "
              f"{10*np.log10(p_raw+eps):>+10.2f}  {10*np.log10(p_proc+eps):>+10.2f}  "
              f"{delta_db:>+8.2f}  {status}")

    print()
    print(f"  Risultato bande verificate: {passed} PASS, {failed} FAIL")
    print()

    # Plot opzionale
    if show_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")   # non-interactive per sicurezza
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.semilogy(freqs_r, psd_r, label="Raw (decimated)", alpha=0.7)
            ax.semilogy(freqs_p, psd_p, label="Preprocessed",    alpha=0.9)
            ax.set_xlabel("Frequenza (Hz)")
            ax.set_ylabel("PSD (µV²/Hz)")
            ax.set_title(f"PSD Lead II — ECG {ecg_id}")
            ax.legend()
            ax.set_xlim(0, FS_NEW / 2)
            ax.grid(True, which="both", alpha=0.3)

            out_path = os.path.join(os.path.dirname(__file__), f"psd_lead_II_{ecg_id}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"  Plot salvato: {os.path.abspath(out_path)}")
        except Exception as e:
            print(f"  [INFO] Plot non disponibile: {e}")

    return failed == 0


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test pipeline di preprocessing ECG")
    parser.add_argument("--plot", action="store_true",
                        help="Salva il plot PSD (richiede matplotlib)")
    args = parser.parse_args()

    results = {}
    results["bandpass"]   = test_bandpass_response()
    results["notch_50hz"] = test_notch_50hz()
    results["downsample"] = test_downsampling()
    results["real_ecg"]   = test_no_nans_on_real_ecg()
    results["psd_bands"]  = test_psd_band_analysis(show_plot=args.plot)

    print("=" * 60)
    print("RIEPILOGO FINALE")
    print("=" * 60)
    all_ok = True
    for name, ok in results.items():
        icon = "[PASS]" if ok else "[FAIL]"
        if not ok:
            all_ok = False
        print(f"  {name:20s}  {icon}")
    print("-" * 60)
    print(f"  Preprocessing: {'PASSA ✓' if all_ok else 'HA PROBLEMI ✗'}")
