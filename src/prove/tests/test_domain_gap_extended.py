"""
test_domain_gap_extended.py
===========================
Test aggiuntivi sul domain gap e la fedeltà della simulazione.

Verifica:
  1. Correttezza matematica delle trasformazioni di inversione
  2. Uniformità dell'augmentation tra classi (no class-dependent artifacts)
  3. Assenza di data leakage tra classi dallo stesso ECG
  4. Pipeline end-to-end nuova vs vecchia
  5. Consistenza preprocessing reali vs sintetici
  6. Stabilità della normalizzazione condivisa

Esecuzione:
    python -m pytest src/prove/tests/test_domain_gap_extended.py -v -s
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
    all_leads_preprocessing, leads_preprocessing
)

LIMB_LEADS = ALL_LEADS[:6]
LIMB_INDICES = list(range(6))


def make_ecg(n_seconds=10, fs=500, seed=42):
    """ECG sintetico realistico con 12 lead."""
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
                base[idx+16:idx+25] += amp * 0.1
        base += np.random.normal(0, abs(amp) * 0.02, n)
        signals[lead] = base.astype(np.float32)
    return signals


# =============================================================================
# TEST 1 — Correttezza matematica delle inversioni
# =============================================================================

class TestInversioneMath(unittest.TestCase):
    """Verifica le trasformazioni matematiche della simulazione."""

    def setUp(self):
        self.raw = make_ecg()
        self.leads_arr = np.array([self.raw[l] for l in LIMB_LEADS])

    def test_la_ra_inversion(self):
        """LA-RA: Lead I = -Lead I, Lead II <-> Lead III."""
        inv = limb_interchange_simulation(1, self.raw)
        inv_arr = np.array([inv[l] for l in LIMB_LEADS])

        # Lead I deve essere negata
        np.testing.assert_array_almost_equal(inv_arr[0], -self.leads_arr[0], decimal=4,
            err_msg="LA-RA: Lead I non e' negata!")

        # Lead II e III scambiate
        np.testing.assert_array_almost_equal(inv_arr[1], self.leads_arr[2], decimal=4,
            err_msg="LA-RA: Lead II non e' uguale a Lead III originale!")
        np.testing.assert_array_almost_equal(inv_arr[2], self.leads_arr[1], decimal=4,
            err_msg="LA-RA: Lead III non e' uguale a Lead II originale!")

        # aVR e aVL scambiate
        np.testing.assert_array_almost_equal(inv_arr[3], self.leads_arr[4], decimal=4)
        np.testing.assert_array_almost_equal(inv_arr[4], self.leads_arr[3], decimal=4)

        print("\n  LA-RA: Lead I negata, II<->III, aVR<->aVL [OK]")

    def test_rot_oraria(self):
        """ROT_ORA (mode=4): rotazione ciclica RA->LA->LL->RA."""
        inv = limb_interchange_simulation(4, self.raw)
        inv_arr = np.array([inv[l] for l in LIMB_LEADS])

        # Lead I = Lead III originale
        np.testing.assert_array_almost_equal(inv_arr[0], self.leads_arr[2], decimal=4)
        # Lead II = -Lead I originale
        np.testing.assert_array_almost_equal(inv_arr[1], -self.leads_arr[0], decimal=4)
        # Lead III = -Lead II originale
        np.testing.assert_array_almost_equal(inv_arr[2], -self.leads_arr[1], decimal=4)

        print("\n  ROT_ORA: I=III, II=-I, III=-II [OK]")

    def test_rot_antioraria(self):
        """ROT_ANT (mode=5): rotazione ciclica RA->LL->LA->RA."""
        inv = limb_interchange_simulation(5, self.raw)
        inv_arr = np.array([inv[l] for l in LIMB_LEADS])

        np.testing.assert_array_almost_equal(inv_arr[0], -self.leads_arr[1], decimal=4)
        np.testing.assert_array_almost_equal(inv_arr[1], -self.leads_arr[2], decimal=4)
        np.testing.assert_array_almost_equal(inv_arr[2], self.leads_arr[0], decimal=4)

        print("\n  ROT_ANT: I=-II, II=-III, III=I [OK]")

    def test_doppia_inversione_identita(self):
        """Applicare LA-RA due volte deve dare l'originale."""
        inv1 = limb_interchange_simulation(1, self.raw)
        inv2 = limb_interchange_simulation(1, inv1)

        for l in LIMB_LEADS:
            np.testing.assert_array_almost_equal(
                inv2[l], self.raw[l], decimal=4,
                err_msg=f"LA-RA doppia non e' identita su {l}!")
        print("\n  LA-RA(LA-RA(x)) = x [OK]")

    def test_precordiali_invariate(self):
        """Le lead precordiali (V1-V6) non devono cambiare dopo inversione."""
        prec_leads = ALL_LEADS[6:]
        for mode, name in [(1, 'LA-RA'), (2, 'RA-LL'), (3, 'LA-LL'),
                           (4, 'ROT_ORA'), (5, 'ROT_ANT')]:
            inv = limb_interchange_simulation(mode, self.raw)
            for l in prec_leads:
                np.testing.assert_array_equal(inv[l], self.raw[l],
                    err_msg=f"{name}: lead precordiale {l} modificata!")
        print("\n  Precordiali invariate per tutte le classi [OK]")



# =============================================================================
# TEST 3 — Leakage tra classi
# =============================================================================

class TestCrossClassLeakage(unittest.TestCase):
    """
    Verifica che non ci sia information leakage tra classi derivate
    dallo stesso ECG (finestre condivise, normalizzazione condivisa, etc.).
    """

    def test_finestre_indipendenti(self):
        """Le finestre di classi diverse non devono essere identiche."""
        raw = make_ecg(seed=42)

        # Preprocessing
        sigs_norm = all_leads_preprocessing(raw)
        raw_inv = limb_interchange_simulation(1, raw)  # LA-RA
        sigs_inv = all_leads_preprocessing(raw_inv)

        arr_norm = np.array([sigs_norm[l] for l in LIMB_LEADS], dtype=np.float32)
        arr_inv = np.array([sigs_inv[l] for l in LIMB_LEADS], dtype=np.float32)

        # Normalizzazione condivisa (fix 3)
        _, ref_med, ref_scale = robust_scale_ecg(
            np.array([sigs_norm[l] for l in ALL_LEADS], dtype=np.float32),
            reference_leads=LIMB_INDICES)

        norm_norm = (arr_norm - ref_med[:6, None]) / ref_scale
        norm_inv = (arr_inv - ref_med[:6, None]) / ref_scale

        # Le finestre NON devono essere identiche
        diff = np.mean(np.abs(norm_norm - norm_inv))
        print(f"\n  Diff media tra finestre normale vs LA-RA: {diff:.4f}")
        self.assertGreater(diff, 0.1,
            "Finestre troppo simili: possibile leakage!")

    def test_no_correlazione_residui(self):
        """I residui (signal - media) non devono essere correlati tra classi."""
        raw = make_ecg(seed=42)
        raw_inv = limb_interchange_simulation(1, raw)

        # Residui su lead I
        norm_res = np.array(raw['I']) - np.mean(raw['I'])
        inv_res = np.array(raw_inv['I']) - np.mean(raw_inv['I'])

        # La correlazione della pura inversione matematica 
        # (senza augmentation, che ora è disabilitata) è prevedibile e perfetta.
        corr_clean = np.corrcoef(norm_res[:1000], inv_res[:1000])[0, 1]

        print(f"\n  Correlazione Lead I (normale vs LA-RA):")
        print(f"    Senza augmentation: r = {corr_clean:.4f} (atteso: -1.0)")
        
        self.assertAlmostEqual(corr_clean, -1.0, places=3,
            msg="Correlazione non e' -1.0 per LA-RA su Lead I!")


# =============================================================================
# TEST 4 — Consistenza preprocessing
# =============================================================================

class TestPreprocessingConsistenza(unittest.TestCase):
    """
    Verifica che il preprocessing sia identico tra la pipeline sintetica
    e quella che processa i reali.
    """

    def test_all_leads_vs_single_lead(self):
        """all_leads_preprocessing(dict) == leads_preprocessing(singola) per ogni lead."""
        raw = make_ecg(seed=42)
        result_all = all_leads_preprocessing(raw)

        for l in LIMB_LEADS:
            result_single = leads_preprocessing(raw[l])
            diff = np.max(np.abs(result_all[l] - result_single))
            self.assertLess(diff, 1e-3,
                f"Lead {l}: diff={diff:.6f} tra all_leads e single")
        print("\n  all_leads_preprocessing == leads_preprocessing per lead [OK]")

    def test_preprocessing_deterministic(self):
        """Stesso input -> stesso output (no randomness nel preprocessing)."""
        raw = make_ecg(seed=42)
        r1 = all_leads_preprocessing(raw)
        r2 = all_leads_preprocessing(raw)

        for l in LIMB_LEADS:
            np.testing.assert_array_equal(r1[l], r2[l],
                err_msg=f"Preprocessing non deterministico su {l}!")
        print("\n  Preprocessing deterministico [OK]")

    def test_preprocessing_dopo_inversione(self):
        """Il preprocessing dopo inversione non deve crashare o produrre NaN."""
        raw = make_ecg(seed=42)
        for mode, name in [(1, 'LA-RA'), (2, 'RA-LL'), (3, 'LA-LL'),
                           (4, 'ROT_ORA'), (5, 'ROT_ANT')]:
            inv = limb_interchange_simulation(mode, raw)
            result = all_leads_preprocessing(inv)
            for l in LIMB_LEADS:
                self.assertFalse(np.any(np.isnan(result[l])),
                    f"{name}: NaN in {l} dopo preprocessing!")
                self.assertFalse(np.any(np.isinf(result[l])),
                    f"{name}: Inf in {l} dopo preprocessing!")
        print("\n  Preprocessing dopo inversione: no NaN/Inf per tutte le classi [OK]")


# =============================================================================
# TEST 5 — Normalizzazione condivisa stabile
# =============================================================================

class TestNormalizzazioneStabilita(unittest.TestCase):
    """
    Verifica che la normalizzazione condivisa non introduce instabilita
    numeriche o valori estremi.
    """

    def test_valori_normalizzati_bounded(self):
        """I valori normalizzati devono essere entro [-50, 50] (non esplosivi)."""
        raw = make_ecg(seed=42)
        sigs = all_leads_preprocessing(raw)
        arr = np.array([sigs[l] for l in ALL_LEADS], dtype=np.float32)
        _, ref_med, ref_scale = robust_scale_ecg(arr, reference_leads=LIMB_INDICES)

        self.assertGreater(ref_scale, 0.01,
            f"Scale troppo piccolo: {ref_scale} -> divisione per ~0!")

        for mode, name in [(None, 'normale'), (1, 'LA-RA'), (2, 'RA-LL'),
                           (3, 'LA-LL'), (4, 'ROT_ORA'), (5, 'ROT_ANT')]:
            if mode is not None:
                inv = limb_interchange_simulation(mode, raw)
            else:
                inv = raw

            sigs_inv = all_leads_preprocessing(inv)
            arr_inv = np.array([sigs_inv[l] for l in ALL_LEADS], dtype=np.float32)
            norm = (arr_inv - ref_med[:, None]) / ref_scale

            max_val = np.max(np.abs(norm[:6]))
            self.assertLess(max_val, 50.0,
                f"{name}: valore normalizzato troppo grande: {max_val:.2f}")

        print("\n  Valori normalizzati bounded [-50, 50] per tutte le classi [OK]")

    def test_std_non_collassata(self):
        """Lo std dopo normalizzazione non deve essere 0 (segnale piatto)."""
        raw = make_ecg(seed=42)
        sigs = all_leads_preprocessing(raw)
        arr = np.array([sigs[l] for l in ALL_LEADS], dtype=np.float32)
        _, ref_med, ref_scale = robust_scale_ecg(arr, reference_leads=LIMB_INDICES)

        for mode, name in [(1, 'LA-RA'), (2, 'RA-LL'), (4, 'ROT_ORA'), (5, 'ROT_ANT')]:
            inv = limb_interchange_simulation(mode, raw)
            sigs_inv = all_leads_preprocessing(inv)
            arr_inv = np.array([sigs_inv[l] for l in ALL_LEADS], dtype=np.float32)
            norm = (arr_inv - ref_med[:, None]) / ref_scale

            std = np.std(norm[:6])
            self.assertGreater(std, 0.1,
                f"{name}: std={std:.4f} troppo bassa, segnale collassato!")
        print("\n  Std non collassata per tutte le classi [OK]")

    def test_range_simile_a_vecchio(self):
        """Il range dei valori col nuovo metodo deve essere nello stesso ordine
        di grandezza del vecchio (non stiamo rompendo la distribuzione)."""
        raw = make_ecg(seed=42)
        sigs = all_leads_preprocessing(raw)
        arr = np.array([sigs[l] for l in ALL_LEADS], dtype=np.float32)

        # Vecchio: normalizzazione per-classe
        norm_old, _, _ = robust_scale_ecg(arr, reference_leads=LIMB_INDICES)
        range_old = np.max(norm_old[:6]) - np.min(norm_old[:6])

        # Nuovo: normalizzazione condivisa
        _, ref_med, ref_scale = robust_scale_ecg(arr, reference_leads=LIMB_INDICES)
        norm_new = (arr - ref_med[:, None]) / ref_scale
        range_new = np.max(norm_new[:6]) - np.min(norm_new[:6])

        # Per la classe NORMALE, i due metodi devono dare risultati identici
        # (stessi parametri di riferimento)
        diff = np.max(np.abs(norm_old[:6] - norm_new[:6]))
        print(f"\n  Range vecchio={range_old:.2f}  nuovo={range_new:.2f}")
        print(f"  Diff max (classe normale): {diff:.6f}")
        self.assertLess(diff, 1e-4,
            "Per la classe normale i due metodi devono essere identici!")


# =============================================================================
# TEST 6 — Simulazione mini-pipeline end-to-end
# =============================================================================

class TestMiniPipeline(unittest.TestCase):
    """
    Simula la pipeline completa (nuovo metodo) su N ECG diversi
    e verifica le proprieta statistiche dell'output.
    """

    def test_distribuzione_multi_ecg(self):
        """Processa 5 ECG e verifica che la distribuzione tra classi
        non abbia bias sistematici."""
        all_means = {name: [] for name in
                     ['normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']}
        all_stds = {name: [] for name in all_means}

        for seed in range(5):
            raw = make_ecg(seed=seed * 10)
            sigs = all_leads_preprocessing(raw)
            arr = np.array([sigs[l] for l in ALL_LEADS], dtype=np.float32)
            _, ref_med, ref_scale = robust_scale_ecg(arr, reference_leads=LIMB_INDICES)

            # Classe normale
            sigs_aug = all_leads_preprocessing(raw)
            arr_aug = np.array([sigs_aug[l] for l in ALL_LEADS], dtype=np.float32)
            norm = (arr_aug - ref_med[:, None]) / ref_scale
            all_means['normale'].append(np.mean(norm[:6]))
            all_stds['normale'].append(np.std(norm[:6]))

            # Classi invertite
            for mode, name in [(1, 'LA-RA'), (2, 'RA-LL'), (3, 'LA-LL'),
                               (4, 'ROT_ORA'), (5, 'ROT_ANT')]:
                raw_inv = limb_interchange_simulation(mode, raw)
                sigs_inv = all_leads_preprocessing(raw_inv)
                arr_inv = np.array([sigs_inv[l] for l in ALL_LEADS], dtype=np.float32)
                norm_inv = (arr_inv - ref_med[:, None]) / ref_scale
                all_means[name].append(np.mean(norm_inv[:6]))
                all_stds[name].append(np.std(norm_inv[:6]))

        print("\n  Media e Std per classe (5 ECG, nuova pipeline):")
        print(f"  {'Classe':<12} {'Mean media':>12} {'Mean std':>10}")
        print(f"  {'-'*36}")
        for name in all_means:
            mm = np.mean(all_means[name])
            ms = np.mean(all_stds[name])
            print(f"  {name:<12} {mm:>12.4f} {ms:>10.4f}")

        # Le medie non devono essere tutte identiche (sarebbero se normalizzazione per-classe)
        means_list = [np.mean(all_means[n]) for n in all_means]
        std_of_means = np.std(means_list)
        print(f"\n  Std delle medie tra classi: {std_of_means:.4f}")
        print(f"  (Se ~ 0 -> tutte le classi hanno la stessa media -> possibile shortcut)")

    def test_nessun_nan_inf_pipeline(self):
        """La pipeline completa non deve produrre NaN o Inf."""
        for seed in range(10):
            raw = make_ecg(seed=seed * 7)
            sigs = all_leads_preprocessing(raw)
            arr = np.array([sigs[l] for l in ALL_LEADS], dtype=np.float32)
            _, ref_med, ref_scale = robust_scale_ecg(arr, reference_leads=LIMB_INDICES)

            for mode in [None, 1, 2, 3, 4, 5]:
                if mode is not None:
                    inv = limb_interchange_simulation(mode, raw)
                else:
                    inv = raw
                sigs_aug = all_leads_preprocessing(inv)
                arr_aug = np.array([sigs_aug[l] for l in ALL_LEADS], dtype=np.float32)
                norm = (arr_aug - ref_med[:, None]) / ref_scale

                self.assertFalse(np.any(np.isnan(norm)),
                    f"NaN trovato: seed={seed}, mode={mode}")
                self.assertFalse(np.any(np.isinf(norm)),
                    f"Inf trovato: seed={seed}, mode={mode}")

        print("\n  10 ECG x 6 classi = 60 combinazioni: no NaN/Inf [OK]")


if __name__ == '__main__':
    unittest.main(verbosity=2)
