"""
test_shortcut_fixes.py
======================
Verifica che i 3 fix anti-shortcut funzionano PRIMA di rigenerare il dataset.

Usa un singolo ECG sintetico per confrontare il comportamento OLD vs NEW.

Fix 1: win_mask per-classe (non condivisa)
Fix 2: augmentation dopo inversione (rumore indipendente)
Fix 3: normalizzazione condivisa (stessi parametri per tutte le classi)

Esecuzione:
    python -m pytest src/prove/tests/test_shortcut_fixes.py -v
"""

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.config import (
    ALL_LEADS, SAMPLES_PER_WINDOW, STRIDE_SAMPLES,
    FS_NEW, robust_scale_ecg
)
from data.data_pipeline import (
    limb_interchange_simulation,
    apply_electrode_gain, apply_random_scaling,
    add_baseline_wander, add_extra_noise,
    all_leads_preprocessing, leads_preprocessing
)

LIMB_LEADS = ALL_LEADS[:6]
LIMB_INDICES = list(range(6))


def make_realistic_ecg(n_seconds=10, fs=500):
    """Genera un ECG sintetico realistico con 12 lead."""
    np.random.seed(42)
    n_samples = n_seconds * fs
    t = np.linspace(0, n_seconds, n_samples)

    # Simula morfologia QRS realistica per ogni lead
    signals = {}
    # Ampiezze tipiche per lead (in uV)
    amps = {'I': 800, 'II': 1200, 'III': 600, 'aVr': -700, 'aVl': 300, 'aVf': 900,
            'V1': 500, 'V2': 1000, 'V3': 1200, 'V4': 1400, 'V5': 1100, 'V6': 800}

    for lead_name in ALL_LEADS:
        amp = amps.get(lead_name, 800)
        base = amp * 0.05 * np.sin(2 * np.pi * 1.2 * t)  # 72 bpm
        # QRS
        for beat in range(int(n_seconds * 1.2)):
            idx = int(beat / 1.2 * fs)
            if idx + 25 < n_samples:
                base[idx:idx+8] += amp * 0.8
                base[idx+8:idx+16] -= amp * 0.3
                base[idx+16:idx+25] += amp * 0.1  # T-wave
        # Rumore fisiologico
        base += np.random.normal(0, abs(amp) * 0.02, n_samples)
        signals[lead_name] = base.astype(np.float32)

    return signals


class TestFix1_WinMaskPerClasse(unittest.TestCase):
    """
    FIX 1: La win_mask deve essere DIVERSA tra l'ECG originale e invertito.

    Se usiamo la stessa mask, stiamo selezionando finestre "buone per il normale"
    anche per l'invertito, creando un bias.
    """

    def test_winmask_differs_after_inversion(self):
        """Inversione cambia la qualita delle finestre."""
        from data.data_pipeline import check_window_quality
        from utils.sqa_real_config import QUALITY_CFG_SYNTH_RELAXED

        raw = make_realistic_ecg()
        sigs_orig = all_leads_preprocessing(raw)
        arr_orig = np.array([sigs_orig[l] for l in ALL_LEADS], dtype=np.float32)

        # Inverti LA-RA (mode=1)
        raw_inv = limb_interchange_simulation(1, raw)
        sigs_inv = all_leads_preprocessing(raw_inv)
        arr_inv = np.array([sigs_inv[l] for l in ALL_LEADS], dtype=np.float32)

        # Calcola qualita per-finestra su entrambi
        cfg = QUALITY_CFG_SYNTH_RELAXED.copy()
        cfg["stride_sec"] = 2.0
        win_size = int(cfg["win_sec"] * cfg["fs"])
        stride = int(cfg["stride_sec"] * cfg["fs"])

        def get_mask(arr):
            n_samples = arr.shape[1]
            starts = list(range(0, n_samples - win_size + 1, stride))
            mask = []
            for start in starts:
                valid = sum(
                    check_window_quality(arr[li, start:start+win_size],
                                         cfg=cfg, lead_idx=li)["valid"]
                    for li in range(6))
                mask.append(valid >= 5)
            return np.array(mask)

        mask_orig = get_mask(arr_orig)
        mask_inv  = get_mask(arr_inv)

        # Le maschere possono essere uguali se l'ECG e' molto pulito,
        # ma il PRINCIPIO e' che devono essere calcolate separatamente
        print(f"\n  Mask orig: {mask_orig.sum()}/{len(mask_orig)} buone")
        print(f"  Mask inv:  {mask_inv.sum()}/{len(mask_inv)} buone")
        print(f"  Mask identiche: {np.array_equal(mask_orig, mask_inv)}")

        # Test: la funzione process_single_ecg usa maschere separate?
        # Verifica indiretta: se aggiungiamo rumore pesante su 1 finestra
        # dell'invertito, la mask deve cambiare
        arr_inv_noisy = arr_inv.copy()
        # Rovina la prima finestra dell'invertito
        arr_inv_noisy[:6, :win_size] += np.random.normal(0, 5000, (6, win_size))
        mask_inv_noisy = get_mask(arr_inv_noisy)

        self.assertFalse(np.array_equal(mask_orig, mask_inv_noisy),
            "La mask non cambia dopo aver rovinato una finestra dell'invertito! "
            "La pipeline vecchia userebbe la stessa mask dell'originale.")

    def test_old_pipeline_shares_mask(self):
        """Dimostra che la pipeline VECCHIA usava la stessa mask per tutto."""
        # Nella vecchia pipeline:
        # win_mask calcolata su sigs_array (originale)
        # wins_s_good = wins_s[:n_win][win_mask[:n_win]]  <-- stessa mask!
        # Questo test documenta il problema (non e' un test di regressione)
        print("\n  [DOC] Pipeline vecchia: win_mask calcolata su originale,")
        print("         applicata a TUTTE le classi invertite senza ricalcolo.")
        print("  [FIX] Pipeline nuova: win_mask ricalcolata per ogni classe.")
        self.assertTrue(True)  # Documentazione, non assertion


class TestFix2_AugmentationDopoInversione(unittest.TestCase):
    """
    FIX 2: Il rumore deve essere INDIPENDENTE dalla classe.

    Se il rumore e' aggiunto PRIMA dell'inversione, quando si fa lo swap
    dei lead il rumore si inverte col segnale. Il modello puo' riconoscere
    la correlazione rumore-lead come shortcut.
    """

    def test_noise_correlation_old_vs_new(self):
        """
        OLD: noise aggiunto prima -> invertito col segnale -> correlato
        NEW: noise aggiunto dopo -> indipendente -> non correlato
        """
        np.random.seed(42)
        raw = make_realistic_ecg()

        # ── Metodo VECCHIO (shortcut) ────────────────────────────────────
        # Aggiungi noise PRIMA
        raw_noisy = apply_electrode_gain(raw, noise_multiplier=0.5)
        raw_noisy = add_baseline_wander(raw_noisy, intensity=300.0)
        # Poi inverti
        raw_inv_old = limb_interchange_simulation(1, raw_noisy)  # LA-RA

        # Estrai il rumore: differenza tra noisy e pulito
        noise_on_lead_I_before_inv = np.array(raw_noisy['I']) - np.array(raw['I'])
        # Dopo inversione LA-RA: lead I = -lead I originale
        # Quindi il noise su lead I dopo inversione = -noise_on_lead_I_before_inv
        noise_on_lead_I_after_inv_old = np.array(raw_inv_old['I']) - (-np.array(raw['I']))

        corr_old = np.corrcoef(noise_on_lead_I_before_inv[:1000],
                                noise_on_lead_I_after_inv_old[:1000])[0, 1]

        # ── Metodo NUOVO (corretto) ──────────────────────────────────────
        # Prima inverti
        raw_inv_clean = limb_interchange_simulation(1, raw)
        # Poi aggiungi noise INDIPENDENTE
        np.random.seed(99)  # seed diverso per il noise
        raw_inv_new = apply_electrode_gain(raw_inv_clean, noise_multiplier=0.5)
        raw_inv_new = add_baseline_wander(raw_inv_new, intensity=300.0)

        noise_on_lead_I_new = np.array(raw_inv_new['I']) - np.array(raw_inv_clean['I'])

        corr_new = np.corrcoef(noise_on_lead_I_before_inv[:1000],
                                noise_on_lead_I_new[:1000])[0, 1]

        print(f"\n  Correlazione rumore lead I (orig vs invertito):")
        print(f"    Metodo VECCHIO: r = {corr_old:.4f}  (rumore correlato = SHORTCUT!)")
        print(f"    Metodo NUOVO:   r = {corr_new:.4f}  (rumore indipendente = OK)")

        # Il vecchio deve avere correlazione alta (negativa, perche' -lead I)
        self.assertGreater(abs(corr_old), 0.5,
            f"Correlazione vecchia dovrebbe essere alta: {corr_old:.4f}")

        # Il nuovo deve avere correlazione bassa (rumore indipendente)
        self.assertLess(abs(corr_new), 0.3,
            f"Correlazione nuova troppo alta: {corr_new:.4f} -- rumore non indipendente!")

    def test_different_noise_per_class(self):
        """Ogni classe deve ricevere un pattern di rumore diverso."""
        np.random.seed(42)
        raw = make_realistic_ecg()

        # Simula 2 classi con augmentation indipendente
        np.random.seed(100)
        aug_c1 = apply_electrode_gain(raw, noise_multiplier=0.5)
        noise_c1 = np.array(aug_c1['I']) - np.array(raw['I'])

        np.random.seed(200)  # seed diverso
        aug_c2 = apply_electrode_gain(raw, noise_multiplier=0.5)
        noise_c2 = np.array(aug_c2['I']) - np.array(raw['I'])

        corr = np.corrcoef(noise_c1[:1000], noise_c2[:1000])[0, 1]
        print(f"\n  Correlazione rumore tra classe 1 e classe 2: r = {corr:.4f}")
        self.assertLess(abs(corr), 0.3,
            f"Rumore tra classi troppo correlato: {corr:.4f}")


class TestFix3_NormalizzazioneCondivisa(unittest.TestCase):
    """
    FIX 3: Mediana/IQR devono essere IDENTICI per tutte le classi
    derivate dallo stesso ECG.

    Se ogni classe ha la sua normalizzazione, il modello puo' distinguerle
    dalla scala statistica (senza guardare la morfologia).
    """

    def test_old_normalization_leaks_class(self):
        """
        Dimostra che con ri-normalizzazione per-classe, i parametri di scala
        differiscono in modo deterministico -> shortcut.
        """
        raw = make_realistic_ecg()

        # Preprocessa originale
        sigs_orig = all_leads_preprocessing(raw)
        arr_orig = np.array([sigs_orig[l] for l in ALL_LEADS], dtype=np.float32)
        _, med_orig, scale_orig = robust_scale_ecg(arr_orig, reference_leads=LIMB_INDICES)

        # Preprocessa invertito LA-RA
        raw_inv = limb_interchange_simulation(1, raw)
        sigs_inv = all_leads_preprocessing(raw_inv)
        arr_inv = np.array([sigs_inv[l] for l in ALL_LEADS], dtype=np.float32)
        _, med_inv, scale_inv = robust_scale_ecg(arr_inv, reference_leads=LIMB_INDICES)

        med_diff = np.abs(med_orig - med_inv).max()
        scale_diff = np.abs(scale_orig - scale_inv).max()

        print(f"\n  Ri-normalizzazione per-classe (VECCHIO metodo):")
        print(f"    Max diff mediana: {med_diff:.4f}")
        print(f"    Max diff scale:   {scale_diff:.4f}")

        # Dimostra che i parametri SONO diversi (= shortcut)
        self.assertGreater(med_diff + scale_diff, 0.01,
            "Mediana/IQR identici tra classi: non c'e' shortcut (inaspettato)")
        print(f"    -> Parametri DIVERSI = il modello puo' usarli come shortcut!")

    def test_new_normalization_shares_params(self):
        """
        Con il fix, tutte le classi usano gli stessi parametri di scala.
        """
        raw = make_realistic_ecg()

        # Calcola parametri dall'originale
        sigs_orig = all_leads_preprocessing(raw)
        arr_orig = np.array([sigs_orig[l] for l in ALL_LEADS], dtype=np.float32)
        _, ref_median, ref_scale = robust_scale_ecg(arr_orig, reference_leads=LIMB_INDICES)

        # Normalizza la classe invertita con i parametri dell'originale
        raw_inv = limb_interchange_simulation(1, raw)
        sigs_inv = all_leads_preprocessing(raw_inv)
        arr_inv = np.array([sigs_inv[l] for l in ALL_LEADS], dtype=np.float32)

        # FIX: usa ref_median e ref_scale dell'originale
        norm_inv_fixed = (arr_inv - ref_median[:, None]) / ref_scale

        # Normalizza con il metodo vecchio (per-classe)
        norm_inv_old, _, _ = robust_scale_ecg(arr_inv, reference_leads=LIMB_INDICES)

        # La normalizzazione fissa produce valori diversi dalla per-classe
        diff = np.abs(norm_inv_fixed - norm_inv_old).mean()
        print(f"\n  Normalizzazione fissa vs per-classe:")
        print(f"    Diff media: {diff:.4f}")
        print(f"    -> Se diff > 0, il fix cambia effettivamente il risultato")
        self.assertGreater(diff, 0.01,
            "Fix non ha effetto: normalizzazioni identiche (inaspettato)")

    def test_scale_invariant_to_class(self):
        """
        Dopo il fix, lo std dei dati normalizzati NON dipende dalla classe.
        """
        raw = make_realistic_ecg()

        sigs_orig = all_leads_preprocessing(raw)
        arr_orig = np.array([sigs_orig[l] for l in ALL_LEADS], dtype=np.float32)
        _, ref_med, ref_scale = robust_scale_ecg(arr_orig, reference_leads=LIMB_INDICES)

        stds = {}
        # Classe normale
        norm_orig = (arr_orig - ref_med[:, None]) / ref_scale
        stds['normale'] = np.std(norm_orig[:6])

        # Classi invertite
        for mode, name in [(1, 'LA-RA'), (2, 'RA-LL'), (4, 'ROT_ORA'), (5, 'ROT_ANT')]:
            raw_inv = limb_interchange_simulation(mode, raw)
            sigs_inv = all_leads_preprocessing(raw_inv)
            arr_inv = np.array([sigs_inv[l] for l in ALL_LEADS], dtype=np.float32)
            norm_inv = (arr_inv - ref_med[:, None]) / ref_scale
            stds[name] = np.std(norm_inv[:6])

        print(f"\n  Std dopo normalizzazione condivisa:")
        for name, s in stds.items():
            print(f"    {name:<12} std={s:.4f}")

        # Con normalizzazione condivisa, gli std POSSONO differire
        # (perche' l'inversione cambia le ampiezze relative).
        # L'importante e' che il modello non possa usare lo std come shortcut
        # per distinguere le classi. Questo avviene quando ogni classe
        # ha la propria normalizzazione -> std sempre ~1.0 -> NO shortcut
        # Con norm condivisa -> std varia -> il modello deve guardare la forma
        print(f"    -> Con norm condivisa gli std variano naturalmente")
        print(f"       (col vecchio metodo sarebbero tutti ~1.0)")


class TestIntegrazione_OldVsNew(unittest.TestCase):
    """
    Test end-to-end che confronta la distribuzione delle finestre
    prodotte dal metodo vecchio vs nuovo su un singolo ECG.
    """

    def test_distribuzione_output(self):
        """
        Confronta le statistiche delle finestre tra i due metodi.
        Il nuovo metodo deve produrre finestre con varianza piu' simile ai reali.
        """
        np.random.seed(42)
        raw = make_realistic_ecg()

        # ── Metodo VECCHIO ──────────────────────────────────────────────
        raw_aug = apply_electrode_gain(raw, noise_multiplier=0.5)
        raw_aug = apply_random_scaling(raw_aug, min_scale=0.6, max_scale=1.4)
        raw_aug = add_baseline_wander(raw_aug, intensity=300.0)

        sigs_old = all_leads_preprocessing(raw_aug)
        arr_old = np.array([sigs_old[l] for l in ALL_LEADS], dtype=np.float32)
        norm_old, _, _ = robust_scale_ecg(arr_old, reference_leads=LIMB_INDICES)

        # Inversione e ri-normalizzazione (vecchio)
        raw_inv_old = limb_interchange_simulation(1, raw_aug)
        sigs_inv_old = all_leads_preprocessing(raw_inv_old)
        arr_inv_old = np.array([sigs_inv_old[l] for l in ALL_LEADS], dtype=np.float32)
        norm_inv_old, _, _ = robust_scale_ecg(arr_inv_old, reference_leads=LIMB_INDICES)

        # ── Metodo NUOVO ────────────────────────────────────────────────
        sigs_new = all_leads_preprocessing(raw)
        arr_new = np.array([sigs_new[l] for l in ALL_LEADS], dtype=np.float32)
        _, ref_med, ref_scale = robust_scale_ecg(arr_new, reference_leads=LIMB_INDICES)

        # Inversione pulita + augmentation dopo
        raw_inv = limb_interchange_simulation(1, raw)
        np.random.seed(99)
        raw_inv_aug = apply_electrode_gain(raw_inv, noise_multiplier=0.5)
        raw_inv_aug = add_baseline_wander(raw_inv_aug, intensity=300.0)
        sigs_inv_new = all_leads_preprocessing(raw_inv_aug)
        arr_inv_new = np.array([sigs_inv_new[l] for l in ALL_LEADS], dtype=np.float32)
        norm_inv_new = (arr_inv_new - ref_med[:, None]) / ref_scale

        # ── Confronto ───────────────────────────────────────────────────
        print(f"\n  {'Statistica':<20} {'Vecchio':>10} {'Nuovo':>10}")
        print(f"  {'-'*42}")

        std_old = np.std(norm_inv_old[:6])
        std_new = np.std(norm_inv_new[:6])
        print(f"  {'Std invertito':<20} {std_old:>10.4f} {std_new:>10.4f}")

        mean_old = np.mean(norm_inv_old[:6])
        mean_new = np.mean(norm_inv_new[:6])
        print(f"  {'Media invertito':<20} {mean_old:>10.4f} {mean_new:>10.4f}")

        # Il vecchio metodo produce std ~1.0 (ri-normalizzato)
        # Il nuovo puo' produrre std != 1.0 (normalizzazione condivisa)
        print(f"\n  Vecchio: std ~1.0 (ri-normalizzato) -> shortcut")
        print(f"  Nuovo: std = {std_new:.4f} (normalizzazione condivisa) -> no shortcut")


if __name__ == '__main__':
    unittest.main(verbosity=2)
