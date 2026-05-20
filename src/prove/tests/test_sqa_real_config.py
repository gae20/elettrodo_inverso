"""
test_sqa_real_config.py
=======================
Verifica che QUALITY_CFG_REAL e QUALITY_CFG_SYNTH_RELAXED si comportino
correttamente rispetto ai casi limite clinici.

Eseguire con:
    python -m pytest src/prove/tests/test_sqa_real_config.py -v
"""

import os
import sys
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, '..', '..')
sys.path.insert(0, SRC_DIR)

from utils.config import QUALITY_CFG, FS_NEW, SAMPLES_PER_WINDOW
from utils.sqa_real_config import QUALITY_CFG_REAL, QUALITY_CFG_SYNTH_RELAXED
from data.data_pipeline import check_ecg_quality, check_window_quality

N_LEADS = 6
WIN     = SAMPLES_PER_WINDOW  # 500
FS      = FS_NEW              # 250 Hz
LEAD_IDX = list(range(N_LEADS))

# Ampiezze in µV — calibrate sull'output reale di compute_window_features:
#   amp=500  → median_abs≈8,  std≈97,  mad_diff≈1.07
#   amp=1000 → median_abs≈17, std≈194, mad_diff≈2.15
#   amp=2000 → median_abs≈34, std≈389, mad_diff≈4.30


def make_realistic_ecg(n_samples=5000, amplitude=1000.0, dc_offset=0.0, noise_std=0.0):
    """ECG sintetico con P-QRS-T continuo."""
    t = np.arange(n_samples) / FS
    x = np.zeros(n_samples, dtype=np.float64)
    for bc in np.arange(0, n_samples / FS, 60.0 / 72):
        tc = t - bc
        x += amplitude * 1.0  * np.exp(-0.5 * (tc / 0.015) ** 2)
        x += amplitude * 0.08 * np.exp(-0.5 * (tc / 0.04) ** 2)
        x -= amplitude * 0.3  * np.exp(-0.5 * ((tc - 0.03) / 0.012) ** 2)
        x += amplitude * 0.25 * np.exp(-0.5 * ((tc - 0.15) / 0.06) ** 2)
    x += dc_offset
    if noise_std > 0:
        x += np.random.default_rng(42).normal(0, noise_std, n_samples)
    return x.astype(np.float32)


def make_ecg_array(n_samples=5000, **kwargs):
    """Array (N_LEADS, n_samples) con scaling inter-lead."""
    scales = [1.0, 1.2, 0.8, 0.9, 0.7, 1.1]
    return np.stack([make_realistic_ecg(n_samples=n_samples, **kwargs) * s
                     for s in scales], axis=0)


class TestQualityCfgReal(unittest.TestCase):

    def test_flatline_ancora_rifiutato(self):
        """Segnale piatto (classe 6/7) → scartato da QUALITY_CFG_REAL."""
        ecg = np.zeros((N_LEADS, 5000), dtype=np.float32)
        result = check_ecg_quality(ecg, cfg=QUALITY_CFG_REAL, lead_indices=LEAD_IDX)
        self.assertFalse(result['global_valid'])
        print(f"\n  [OK] Flatline rifiutato. Motivo: {result['reason']}")

    def test_ecg_rumoroso_accettato_da_real(self):
        """ECG con rumore alto → accettato da QUALITY_CFG_REAL."""
        ecg = make_ecg_array(amplitude=2000.0, noise_std=200.0)
        result = check_ecg_quality(ecg, cfg=QUALITY_CFG_REAL, lead_indices=LEAD_IDX)
        print(f"\n  QUALITY_CFG_REAL: {result['global_valid']} ({result['reason']})")
        self.assertTrue(result['global_valid'],
            "ECG rumoroso ma refertato deve passare QUALITY_CFG_REAL")

    def test_ecg_con_dc_offset_accettato_da_real(self):
        """
        ECG con DC offset 1000µV:
        - Originale rifiuta (baseline_drift > 600 e/o no_morphology con mad_diff < 1.5)
        - REAL accetta (baseline disabilitato, min_mad_diff=0.3)
        """
        ecg = make_ecg_array(amplitude=1000.0, dc_offset=1000.0)
        result_orig = check_ecg_quality(ecg, cfg=QUALITY_CFG, lead_indices=LEAD_IDX)
        result_real = check_ecg_quality(ecg, cfg=QUALITY_CFG_REAL, lead_indices=LEAD_IDX)
        print(f"\n  QUALITY_CFG:      {result_orig['global_valid']} ({result_orig['reason']})")
        print(f"  QUALITY_CFG_REAL: {result_real['global_valid']} ({result_real['reason']})")
        self.assertFalse(result_orig['global_valid'],
            "Config originale dovrebbe rifiutare ECG con DC offset alto")
        self.assertTrue(result_real['global_valid'],
            "QUALITY_CFG_REAL dovrebbe accettare ECG con baseline drift")

    def test_finestra_saturata_scartata(self):
        """Finestra con ADC saturo → scartata anche da QUALITY_CFG_REAL."""
        window = np.ones(WIN, dtype=np.float32) * 7000.0
        result = check_window_quality(window, cfg=QUALITY_CFG_REAL, lead_idx=0)
        self.assertFalse(result['valid'])
        print(f"\n  [OK] Finestra saturata scartata. Motivo: {result['reason']}")

    def test_finestra_ecg_normale_accettata(self):
        """Finestra ECG con ampiezza realistica (1000µV) → accettata."""
        window = make_realistic_ecg(n_samples=WIN, amplitude=1000.0)
        result = check_window_quality(window, cfg=QUALITY_CFG_REAL, lead_idx=0)
        self.assertTrue(result['valid'],
            f"Finestra ECG dovrebbe essere accettata. Motivo: {result['reason']}")

    def test_rl_ra_ancora_rifiutato(self):
        """ECG con lead II piatta (RL-RA) → scartato da entrambi i config."""
        good = make_realistic_ecg(n_samples=5000, amplitude=1000.0)
        flat = np.random.default_rng(0).normal(0, 3.0, 5000).astype(np.float32)
        ecg = np.stack([good, flat, good,
                        -0.5*(good+flat), good-0.5*flat, flat-0.5*good], axis=0)
        result_orig = check_ecg_quality(ecg, cfg=QUALITY_CFG, lead_indices=LEAD_IDX)
        result_real = check_ecg_quality(ecg, cfg=QUALITY_CFG_REAL, lead_indices=LEAD_IDX)
        print(f"\n  QUALITY_CFG:      {result_orig['global_valid']} ({result_orig['reason']})")
        print(f"  QUALITY_CFG_REAL: {result_real['global_valid']} ({result_real['reason']})")
        self.assertFalse(result_orig['global_valid'])
        self.assertFalse(result_real['global_valid'])


class TestQualityCfgSynthRelaxed(unittest.TestCase):

    def test_ecg_rumoroso_moderato_accettato(self):
        """ECG con rumore moderato → accettato da SYNTH_RELAXED (non dall'originale)."""
        # noise_std=80 → mad_diff ~ 54-77 per le diverse lead (scale 0.7-1.2)
        # Originale: soglia globale = 25 * 1.1 = 27.5 → rifiuta
        # SYNTH_RELAXED: soglia globale = 100 * 1.1 = 110 → accetta
        ecg = make_ecg_array(amplitude=2000.0, noise_std=80.0)
        result = check_ecg_quality(ecg, cfg=QUALITY_CFG_SYNTH_RELAXED, lead_indices=LEAD_IDX)
        print(f"\n  SYNTH_RELAXED: {result['global_valid']} ({result['reason']})")
        for lr in result['lead_results']:
            if not lr['valid']:
                print(f"    Lead {lr['lead_idx']}: {lr['reason']}")
        self.assertTrue(result['global_valid'],
            "SYNTH_RELAXED deve accettare sintetico con rumore moderato")

    def test_flatline_rifiutato(self):
        """Flatline → scartato anche da SYNTH_RELAXED."""
        ecg = np.zeros((N_LEADS, 5000), dtype=np.float32)
        result = check_ecg_quality(ecg, cfg=QUALITY_CFG_SYNTH_RELAXED, lead_indices=LEAD_IDX)
        self.assertFalse(result['global_valid'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
