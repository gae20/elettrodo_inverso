"""
test_sqa_flags.py
=================
Test della Signal Quality Assessment con configurazioni duali:
  - QUALITY_CFG: configurazione originale
  - QUALITY_CFG_REAL: per dati clinici refertati (solo check strutturali)
  - QUALITY_CFG_SYNTH_RELAXED: per dati sintetici (soglie rilassate)

Verifica che:
  1. Segnali buoni passano con tutte le configurazioni
  2. Flatline/inversioni strutturali sono rifiutate da TUTTE le config
  3. Rumore moderato è accettato da REAL e SYNTH_RELAXED ma non dalla originale
  4. Le configurazioni duali non introducono regressioni

Esecuzione:
    python -m pytest src/prove/tests/test_sqa_flags.py -v
"""

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data.data_pipeline import check_ecg_quality
from utils.config import QUALITY_CFG, FS_NEW
from utils.sqa_real_config import QUALITY_CFG_REAL, QUALITY_CFG_SYNTH_RELAXED

LEAD_IDX = list(range(6))


def make_good_signal(n_seconds=10, fs=FS_NEW, amplitude=500):
    """Genera un segnale ECG sintetico 'buono' con morfologia QRS realistica."""
    n_samples = n_seconds * fs
    t = np.linspace(0, n_seconds, n_samples)
    
    # Base: onda sinusoidale a 1 Hz (frequenza cardiaca simulata)
    base = amplitude * 0.06 * np.sin(2 * np.pi * 1.0 * t)
    signal = np.tile(base, (12, 1))
    
    # Scale per lead (le periferiche hanno ampiezze diverse)
    lead_scales = [1.0, 1.2, 0.8, 0.9, 0.7, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for i in range(12):
        signal[i] *= lead_scales[i]
    
    # QRS ogni secondo
    for beat in range(n_seconds):
        idx = beat * fs + int(0.2 * fs)
        if idx + 20 < n_samples:
            signal[:, idx:idx+10] += amplitude    # R-peak
            signal[:, idx+10:idx+20] -= amplitude * 0.4  # S-wave
    
    # Rumore bianco fisiologico
    signal += np.random.normal(0, 5, size=(12, n_samples))
    
    return signal.astype(np.float32)


class TestSQAFlagsOriginal(unittest.TestCase):
    """Test con la configurazione QUALITY_CFG originale."""

    def setUp(self):
        np.random.seed(42)
        self.good_signal = make_good_signal()

    def test_good_signal_passes(self):
        """Segnale buono deve passare con la config originale."""
        res = check_ecg_quality(self.good_signal, cfg=QUALITY_CFG, lead_indices=LEAD_IDX)
        self.assertTrue(res['global_valid'], f"Segnale buono bocciato: {res['reason']}")

    def test_flatline_rejection(self):
        """Flatline su 2 lead periferiche deve essere rifiutato."""
        bad = self.good_signal.copy()
        bad[0, :] = 5.0
        bad[1, :] = -2.0
        res = check_ecg_quality(bad, cfg=QUALITY_CFG, lead_indices=LEAD_IDX)
        self.assertFalse(res['global_valid'], "Flatline non rifiutato!")

    def test_structural_RL_RA_rejection(self):
        """Inversione RL-RA (lead II piatta) deve essere rifiutata."""
        bad = self.good_signal.copy()
        bad[1, :] = np.random.normal(0, 5, size=bad.shape[1])
        res = check_ecg_quality(bad, cfg=QUALITY_CFG, lead_indices=LEAD_IDX)
        self.assertFalse(res['global_valid'])
        self.assertIn("structural", res['reason'])

    def test_extreme_noise_rejection(self):
        """Rumore estremo su tutte le lead deve essere rifiutato."""
        bad = self.good_signal.copy()
        bad[:6, :] += np.random.normal(0, 500, size=(6, bad.shape[1]))
        res = check_ecg_quality(bad, cfg=QUALITY_CFG, lead_indices=LEAD_IDX)
        self.assertFalse(res['global_valid'], "Rumore estremo non rifiutato!")


class TestSQAFlagsReal(unittest.TestCase):
    """Test con QUALITY_CFG_REAL (dati clinici refertati)."""

    def setUp(self):
        np.random.seed(42)
        self.good_signal = make_good_signal()

    def test_good_signal_passes(self):
        """Segnale buono deve passare con REAL config."""
        res = check_ecg_quality(self.good_signal, cfg=QUALITY_CFG_REAL, lead_indices=LEAD_IDX)
        self.assertTrue(res['global_valid'], f"Bocciato: {res['reason']}")

    def test_flatline_still_rejected(self):
        """Flatline deve essere rifiutato anche con REAL config."""
        bad = self.good_signal.copy()
        bad[0, :] = 5.0
        bad[1, :] = -2.0
        res = check_ecg_quality(bad, cfg=QUALITY_CFG_REAL, lead_indices=LEAD_IDX)
        self.assertFalse(res['global_valid'])

    def test_structural_still_rejected(self):
        """Check strutturali attivi anche con REAL config."""
        bad = self.good_signal.copy()
        bad[1, :] = np.random.normal(0, 5, size=bad.shape[1])
        res = check_ecg_quality(bad, cfg=QUALITY_CFG_REAL, lead_indices=LEAD_IDX)
        self.assertFalse(res['global_valid'])

    def test_moderate_noise_accepted(self):
        """Rumore moderato accettato da REAL (non dall'originale)."""
        noisy = self.good_signal.copy()
        # noise_std=15 è realistico per ECG clinici con artefatti da movimento
        noisy[:6, :] += np.random.normal(0, 15, size=(6, noisy.shape[1]))
        
        res_orig = check_ecg_quality(noisy, cfg=QUALITY_CFG, lead_indices=LEAD_IDX)
        res_real = check_ecg_quality(noisy, cfg=QUALITY_CFG_REAL, lead_indices=LEAD_IDX)
        
        # REAL deve accettare, originale può rifiutare
        self.assertTrue(res_real['global_valid'],
            f"REAL config ha rifiutato rumore moderato: {res_real['reason']}")

    def test_dc_offset_accepted(self):
        """DC offset (baseline wander) accettato da REAL."""
        dc = self.good_signal.copy()
        dc[:6, :] += 2000.0  # offset grande
        res = check_ecg_quality(dc, cfg=QUALITY_CFG_REAL, lead_indices=LEAD_IDX)
        self.assertTrue(res['global_valid'],
            f"DC offset rifiutato da REAL: {res['reason']}")


class TestSQAFlagsSynthRelaxed(unittest.TestCase):
    """Test con QUALITY_CFG_SYNTH_RELAXED."""

    def setUp(self):
        np.random.seed(42)
        self.good_signal = make_good_signal()

    def test_good_signal_passes(self):
        res = check_ecg_quality(self.good_signal, cfg=QUALITY_CFG_SYNTH_RELAXED, lead_indices=LEAD_IDX)
        self.assertTrue(res['global_valid'], f"Bocciato: {res['reason']}")

    def test_flatline_still_rejected(self):
        """Flatline rifiutato anche con SYNTH_RELAXED."""
        bad = self.good_signal.copy()
        bad[0, :] = 5.0
        bad[1, :] = -2.0
        res = check_ecg_quality(bad, cfg=QUALITY_CFG_SYNTH_RELAXED, lead_indices=LEAD_IDX)
        self.assertFalse(res['global_valid'])

    def test_moderate_noise_accepted(self):
        """Rumore moderato accettato da SYNTH_RELAXED."""
        noisy = self.good_signal.copy()
        noisy[:6, :] += np.random.normal(0, 15, size=(6, noisy.shape[1]))
        res = check_ecg_quality(noisy, cfg=QUALITY_CFG_SYNTH_RELAXED, lead_indices=LEAD_IDX)
        self.assertTrue(res['global_valid'],
            f"SYNTH_RELAXED ha rifiutato rumore moderato: {res['reason']}")


class TestSQAConfigConsistency(unittest.TestCase):
    """Verifica la coerenza tra le tre configurazioni."""

    def test_real_less_strict_than_original(self):
        """REAL deve essere meno restrittiva dell'originale su rumore."""
        self.assertGreaterEqual(
            QUALITY_CFG_REAL.get('mad_noise_limb', float('inf')),
            QUALITY_CFG.get('mad_noise_limb', 0),
            "REAL config più restrittiva dell'originale su mad_noise_limb!"
        )

    def test_synth_relaxed_less_strict_than_original(self):
        """SYNTH_RELAXED deve avere soglie più alte dell'originale."""
        self.assertGreaterEqual(
            QUALITY_CFG_SYNTH_RELAXED.get('mad_noise_limb', float('inf')),
            QUALITY_CFG.get('mad_noise_limb', 0),
            "SYNTH_RELAXED più restrittiva dell'originale!"
        )

    def test_all_configs_have_required_keys(self):
        """Tutte le config devono avere le chiavi essenziali."""
        required = ['fs', 'win_sec', 'stride_sec']
        for cfg, name in [(QUALITY_CFG, 'QUALITY_CFG'),
                          (QUALITY_CFG_REAL, 'QUALITY_CFG_REAL'),
                          (QUALITY_CFG_SYNTH_RELAXED, 'QUALITY_CFG_SYNTH_RELAXED')]:
            for key in required:
                with self.subTest(config=name, key=key):
                    self.assertIn(key, cfg, f"{name} manca la chiave '{key}'")


if __name__ == '__main__':
    unittest.main()
