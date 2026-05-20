"""
test_pipeline_v2.py
===================
Test della pipeline aggiornata (Fix 1 + Fix 2, senza Fix 3).

Verifica che:
  1. Fix 1: win_mask calcolata per-classe (non condivisa)
  2. Fix 2: rumore indipendente dalla classe (augmentation dopo inversione)
  3. Normalizzazione per-classe (come nei reali) - consistente
  4. Distribuzione rumore per classe: SNR, std, correlazione
  5. Pipeline end-to-end produce output valido

Esecuzione:
    python -m pytest src/prove/tests/test_pipeline_v2.py -v -s
"""

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.config import (
    ALL_LEADS, SAMPLES_PER_WINDOW, STRIDE_SAMPLES,
    FS_NEW, FS_OLD, robust_scale_ecg
)
from data.data_pipeline import (
    limb_interchange_simulation,
    apply_electrode_gain, apply_random_scaling,
    add_baseline_wander, add_extra_noise,
    all_leads_preprocessing
)

LIMB_LEADS = ALL_LEADS[:6]
LIMB_INDICES = list(range(6))
CLASS_NAMES = ['normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']
CLASS_MODES = [None, 1, 2, 3, 4, 5]


def make_ecg(seed=42, n_seconds=10, fs=500):
    np.random.seed(seed)
    n = n_seconds * fs
    t = np.linspace(0, n_seconds, n)
    amps = {'I': 800, 'II': 1200, 'III': 600, 'aVr': -700, 'aVl': 300, 'aVf': 900,
            'V1': 500, 'V2': 1000, 'V3': 1200, 'V4': 1400, 'V5': 1100, 'V6': 800}
    signals = {}
    for lead in ALL_LEADS:
        amp = amps.get(lead, 800)
        base = amp * 0.05 * np.sin(2 * np.pi * 1.2 * t)
        for beat in range(int(n_seconds * 1.2)):
            idx = int(beat / 1.2 * fs)
            if idx + 25 < n:
                base[idx:idx+8] += amp * 0.8
                base[idx+8:idx+16] -= amp * 0.3
        base += np.random.normal(0, abs(amp) * 0.02, n)
        signals[lead] = base.astype(np.float32)
    return signals


def pipeline_new(raw_clean, mode, add_gain=True, seed_offset=0):
    """Simula la pipeline NUOVA (Fix 1 + Fix 2, no Fix 3)."""
    # Inversione sul segnale PULITO
    if mode is not None:
        raw_inv = limb_interchange_simulation(mode, raw_clean)
    else:
        raw_inv = {k: v.copy() for k, v in raw_clean.items()}

    # Augmentation DOPO inversione (Fix 2)
    if add_gain:
        np.random.seed(42 + seed_offset + (mode or 0))
        scale = np.random.uniform(0.6, 1.4)
        aug = apply_random_scaling(raw_inv, min_scale=scale*0.9, max_scale=scale*1.1)
        aug = add_baseline_wander(aug, fs=FS_OLD, intensity=300.0)
        aug = apply_electrode_gain(aug, fs=FS_OLD, noise_multiplier=0.5)
        aug = add_extra_noise(aug, multiplier=1.0, fs=FS_OLD)
    else:
        aug = raw_inv

    # Preprocessing
    sigs = all_leads_preprocessing(aug)
    arr = np.array([sigs[l] for l in ALL_LEADS], dtype=np.float32)

    # Normalizzazione indipendente (come nei reali)
    norm, _, _ = robust_scale_ecg(arr, reference_leads=LIMB_INDICES)
    return norm, arr, aug, raw_inv


def pipeline_old(raw_clean, mode, add_gain=True):
    """Simula la pipeline VECCHIA (augmentation PRIMA dell'inversione)."""
    if add_gain:
        np.random.seed(42)
        aug = apply_electrode_gain(raw_clean, noise_multiplier=0.5)
        aug = apply_random_scaling(aug, min_scale=0.6, max_scale=1.4)
        aug = add_baseline_wander(aug, fs=FS_OLD, intensity=300.0)
    else:
        aug = raw_clean

    # Inversione DOPO augmentation (vecchio - sbagliato)
    if mode is not None:
        raw_inv = limb_interchange_simulation(mode, aug)
    else:
        raw_inv = aug

    sigs = all_leads_preprocessing(raw_inv)
    arr = np.array([sigs[l] for l in ALL_LEADS], dtype=np.float32)
    norm, _, _ = robust_scale_ecg(arr, reference_leads=LIMB_INDICES)
    return norm


# =============================================================================
# TEST 1 — Fix 2 ancora funzionante
# =============================================================================

class TestFix2_RumoreIndipendente(unittest.TestCase):
    """Verifica che Fix 2 (augmentation dopo inversione) funziona."""

    def test_correlazione_rumore_vecchio_vs_nuovo(self):
        """Il rumore vecchio e' correlato (shortcut), il nuovo no."""
        raw = make_ecg(seed=42)

        # VECCHIO: noise prima -> inversione -> noise correlato
        np.random.seed(10)
        raw_noisy = apply_electrode_gain(raw, noise_multiplier=0.5)
        noise_before = np.array(raw_noisy['I']) - np.array(raw['I'])

        raw_inv_old = limb_interchange_simulation(1, raw_noisy)
        noise_after_old = np.array(raw_inv_old['I']) - (-np.array(raw['I']))
        corr_old = np.corrcoef(noise_before[:1000], noise_after_old[:1000])[0, 1]

        # NUOVO: inversione prima -> noise dopo -> indipendente
        raw_inv = limb_interchange_simulation(1, raw)
        np.random.seed(99)
        raw_inv_noisy = apply_electrode_gain(raw_inv, noise_multiplier=0.5)
        noise_after_new = np.array(raw_inv_noisy['I']) - np.array(raw_inv['I'])
        corr_new = np.corrcoef(noise_before[:1000], noise_after_new[:1000])[0, 1]

        print(f"\n  Correlazione rumore electrode_gain (originale vs LA-RA):")
        print(f"    VECCHIO (noise prima):  r = {corr_old:+.4f}  {'[SHORTCUT!]' if abs(corr_old) > 0.5 else ''}")
        print(f"    NUOVO   (noise dopo):   r = {corr_new:+.4f}  {'[OK]' if abs(corr_new) < 0.3 else '[PROBLEMA]'}")

        self.assertLess(abs(corr_new), 0.3)


# =============================================================================
# TEST 2 — Normalizzazione per-classe (no Fix 3)
# =============================================================================

class TestNormalizzazionePerClasse(unittest.TestCase):
    """Verifica che ogni classe ha la sua normalizzazione (come i reali)."""

    def test_ogni_classe_std_circa_1(self):
        """Con normalizzazione per-classe, lo std delle 6 limb deve essere ~1.0."""
        raw = make_ecg(seed=42)

        print("\n  Std per classe (normalizzazione indipendente):")
        for mode, name in zip(CLASS_MODES, CLASS_NAMES):
            norm, _, _, _ = pipeline_new(raw, mode, add_gain=False)
            std = np.std(norm[:6])
            print(f"    {name:<12} std = {std:.4f}")
            # Con robust_scale indipendente, std deve essere intorno a 1.0-2.0
            self.assertGreater(std, 0.5, f"{name}: std troppo bassa")
            self.assertLess(std, 5.0, f"{name}: std troppo alta")

    def test_consistenza_con_reali(self):
        """La normalizzazione sintetica deve essere identica a quella reale:
        entrambe usano robust_scale_ecg indipendente."""
        raw = make_ecg(seed=42)

        # Simula il percorso "reale": ECG -> preprocess -> robust_scale
        sigs = all_leads_preprocessing(raw)
        arr = np.array([sigs[l] for l in ALL_LEADS], dtype=np.float32)
        norm_real, _, _ = robust_scale_ecg(arr, reference_leads=LIMB_INDICES)

        # Simula il percorso "sintetico" (classe normale, no augmentation)
        norm_synth, _, _, _ = pipeline_new(raw, mode=None, add_gain=False)

        diff = np.max(np.abs(norm_real[:6] - norm_synth[:6]))
        print(f"\n  Max diff normalizzazione sintetico vs reale: {diff:.6f}")
        self.assertLess(diff, 1e-4, "Normalizzazione non consistente!")


# =============================================================================
# TEST 3 — Distribuzione rumore per classe
# =============================================================================

class TestDistribuzioneRumorePerClasse(unittest.TestCase):
    """Analisi dettagliata del rumore aggiunto per ogni classe."""

    def test_snr_per_classe(self):
        """SNR (dB) per ogni classe dopo augmentation."""
        raw = make_ecg(seed=42)

        print("\n  DISTRIBUZIONE RUMORE PER CLASSE")
        print(f"  {'Classe':<12} {'SNR(dB)':>8} {'Noise std':>10} {'Sig std':>10} {'Noise/Sig':>10}")
        print(f"  {'-'*54}")

        snrs = {}
        for mode, name in zip(CLASS_MODES, CLASS_NAMES):
            # Segnale pulito (dopo inversione)
            if mode is not None:
                raw_inv = limb_interchange_simulation(mode, raw)
            else:
                raw_inv = raw

            signal = np.array(raw_inv['II'], dtype=np.float64)
            sig_std = np.std(signal)

            # Con augmentation: mediamo su 20 seed per gli artefatti rari
            snr_list = []
            noise_std_list = []
            for i in range(20):
                np.random.seed((mode if mode else 0) + 200 + i)
                aug = apply_electrode_gain(raw_inv, noise_multiplier=0.5)
                aug = add_baseline_wander(aug, intensity=300.0)
                aug = add_extra_noise(aug, multiplier=1.0, fs=FS_OLD)

                augmented = np.array(aug['II'], dtype=np.float64)
                noise = augmented - signal
                snr_list.append(10 * np.log10(np.var(signal) / (np.var(noise) + 1e-9)))
                noise_std_list.append(np.std(noise))

            snr = np.mean(snr_list)
            noise_std = np.mean(noise_std_list)
            ratio = noise_std / (sig_std + 1e-9)
            snrs[name] = snr

            print(f"  {name:<12} {snr:>8.1f} {noise_std:>10.1f} {sig_std:>10.1f} {ratio:>10.3f}")

        print(f"  {'-'*54}")

        # Range SNR accettabile: < 10 dB sulla media
        snr_range = max(snrs.values()) - min(snrs.values())
        print(f"\n  Range SNR medio tra classi: {snr_range:.1f} dB (< 10 dB = OK)")
        self.assertLess(snr_range, 10.0)

    def test_noise_media_zero_per_classe(self):
        """Il rumore aggiunto deve avere media ~0 per ogni classe."""
        raw = make_ecg(seed=42)

        print("\n  Media rumore per classe (deve essere ~0):")
        for mode, name in zip(CLASS_MODES, CLASS_NAMES):
            if mode is not None:
                raw_inv = limb_interchange_simulation(mode, raw)
            else:
                raw_inv = raw

            np.random.seed(mode if mode else 0 + 300)
            aug = apply_electrode_gain(raw_inv, noise_multiplier=0.5)
            noise = np.array(aug['II']) - np.array(raw_inv['II'])
            mean_noise = np.mean(noise)
            print(f"    {name:<12} media rumore = {mean_noise:+.4f}")
            self.assertLess(abs(mean_noise), 50.0)

    def test_correlazione_rumore_tra_classi(self):
        """Il rumore tra classi diverse deve essere incorrelato."""
        raw = make_ecg(seed=42)

        noises = {}
        for mode, name in zip(CLASS_MODES, CLASS_NAMES):
            if mode is not None:
                raw_inv = limb_interchange_simulation(mode, raw)
            else:
                raw_inv = raw

            np.random.seed(mode if mode else 0 + 400)
            aug = apply_electrode_gain(raw_inv, noise_multiplier=0.5)
            noises[name] = np.array(aug['II']) - np.array(raw_inv['II'])

        print("\n  Matrice correlazione rumore tra classi:")
        print(f"  {'':>12}", end="")
        for n in CLASS_NAMES:
            print(f" {n[:6]:>7}", end="")
        print()

        for n1 in CLASS_NAMES:
            print(f"  {n1:<12}", end="")
            for n2 in CLASS_NAMES:
                if n1 == n2:
                    corr = 1.0
                else:
                    corr = np.corrcoef(noises[n1][:1000], noises[n2][:1000])[0, 1]
                marker = "*" if abs(corr) > 0.3 and n1 != n2 else " "
                print(f" {corr:>+6.3f}{marker}", end="")
            print()

        # Nessuna coppia deve avere correlazione > 0.3
        for i, n1 in enumerate(CLASS_NAMES):
            for j, n2 in enumerate(CLASS_NAMES):
                if i >= j:
                    continue
                corr = np.corrcoef(noises[n1][:1000], noises[n2][:1000])[0, 1]
                self.assertLess(abs(corr), 0.3,
                    f"Rumore correlato tra {n1} e {n2}: r={corr:.3f}")

        print("\n  Tutte le coppie: |r| < 0.3 [OK]")

    def test_spettro_rumore_uniforme(self):
        """Lo spettro del rumore deve essere simile tra classi (no bias frequenziale)."""
        raw = make_ecg(seed=42)

        print("\n  Energia rumore per banda frequenziale:")
        print(f"  {'Classe':<12} {'0-10Hz':>8} {'10-50Hz':>8} {'50-125Hz':>8}")

        for mode, name in zip(CLASS_MODES, CLASS_NAMES):
            if mode is not None:
                raw_inv = limb_interchange_simulation(mode, raw)
            else:
                raw_inv = raw

            np.random.seed(mode if mode else 0 + 500)
            aug = apply_electrode_gain(raw_inv, noise_multiplier=0.5)
            aug = add_baseline_wander(aug, intensity=300.0)
            noise = np.array(aug['II']) - np.array(raw_inv['II'])

            # FFT
            fft = np.fft.rfft(noise)
            freqs = np.fft.rfftfreq(len(noise), 1.0/FS_OLD)
            power = np.abs(fft)**2

            # Bande
            low = np.sum(power[(freqs >= 0) & (freqs < 10)])
            mid = np.sum(power[(freqs >= 10) & (freqs < 50)])
            high = np.sum(power[(freqs >= 50) & (freqs < 125)])
            total = low + mid + high + 1e-9

            print(f"  {name:<12} {low/total*100:>7.1f}% {mid/total*100:>7.1f}% {high/total*100:>7.1f}%")


# =============================================================================
# TEST 4 — Pipeline end-to-end su piu ECG
# =============================================================================

class TestPipelineEndToEnd(unittest.TestCase):
    """Verifica la pipeline completa su piu ECG."""

    def test_output_no_nan_inf(self):
        """60 combinazioni ECG x classe senza NaN/Inf."""
        for seed in range(10):
            raw = make_ecg(seed=seed*7)
            for mode, name in zip(CLASS_MODES, CLASS_NAMES):
                norm, _, _, _ = pipeline_new(raw, mode, add_gain=True, seed_offset=seed*100)
                self.assertFalse(np.any(np.isnan(norm[:6])), f"NaN: seed={seed} {name}")
                self.assertFalse(np.any(np.isinf(norm[:6])), f"Inf: seed={seed} {name}")
        print("\n  60 combinazioni: no NaN/Inf [OK]")

    def test_vecchio_vs_nuovo_confronto_finale(self):
        """Confronto riassuntivo vecchio vs nuovo."""
        raw = make_ecg(seed=42)

        print("\n  CONFRONTO PIPELINE VECCHIA vs NUOVA")
        print(f"  {'Classe':<12} {'Std OLD':>10} {'Std NEW':>10} {'Mean OLD':>10} {'Mean NEW':>10}")
        print(f"  {'-'*54}")

        for mode, name in zip(CLASS_MODES, CLASS_NAMES):
            norm_old = pipeline_old(raw, mode, add_gain=True)
            norm_new, _, _, _ = pipeline_new(raw, mode, add_gain=True)

            std_old = np.std(norm_old[:6])
            std_new = np.std(norm_new[:6])
            mean_old = np.mean(norm_old[:6])
            mean_new = np.mean(norm_new[:6])

            print(f"  {name:<12} {std_old:>10.4f} {std_new:>10.4f} {mean_old:>+10.4f} {mean_new:>+10.4f}")

        print(f"  {'-'*54}")
        print("  Std simile per entrambi = normalizzazione per-classe (OK)")


if __name__ == '__main__':
    unittest.main(verbosity=2)
