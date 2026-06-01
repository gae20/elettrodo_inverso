"""
test_domain_gap_diagnostic.py
=============================
Test diagnostici approfonditi per capire DOVE si genera il domain gap.

Ipotesi da verificare:
  H1. Preprocessing diverso (sintetici: all_leads_preprocessing vs reali: leads_preprocessing)
  H2. Normalizzazione diversa (sintetici: 12 lead con reference vs reali: solo 6 lead)
  H3. Augmentation del gain (noise, scaling, baseline wander) altera la distribuzione
  H4. La win_mask è diversa (sintetici: stride 2s, reali: stride diverso)
  H5. Il modello ha imparato artefatti della simulazione (shortcut learning)
  H6. La distribuzione delle feature latenti è diversa (embedding shift)

Esecuzione:
    python src/prove/tests/test_domain_gap_diagnostic.py
"""

import os
import sys
import unittest
import h5py
import numpy as np

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DIR   = os.path.join(BASE_DIR, '..', '..')
PROVE_DIR = os.path.join(BASE_DIR, '..')
DATA_DIR  = os.path.join(BASE_DIR, '..', '..', '..', '..', 'datasets')

sys.path.insert(0, SRC_DIR)

from utils.config import (
    SAMPLES_PER_WINDOW, ALL_LEADS, robust_scale_ecg
)
from models.ldensenet import build_model

SYNTH_PATH  = os.path.join(DATA_DIR, "unlabelled_final_noise_limbs_val.h5")
REAL_PATH   = os.path.join(DATA_DIR, "labelled_z_median_limbs_test_validation.h5")
WEIGHTS     = os.path.join(PROVE_DIR, "models", "best_model_final_noise_limbs.weights.h5")

CLASS_NAMES = ['normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']
N_CLASSES   = 6
LIMB_LEADS  = ALL_LEADS[:6]


def _load_raw(path, n_per_class=100):
    """Carica dati raw (6, T) per analisi distribuzione."""
    with h5py.File(path, 'r') as f:
        y = f['Y'][:]
        x = f['X'][:]
    valid = np.where(y < N_CLASSES)[0]
    x, y = x[valid], y[valid]
    idxs = []
    for c in range(N_CLASSES):
        ci = np.where(y == c)[0]
        np.random.seed(42 + c)
        idxs.extend(np.random.choice(ci, min(n_per_class, len(ci)), replace=False))
    idxs = np.array(sorted(idxs))
    return x[idxs, :6, :], y[idxs]


# =============================================================================
# H1. Test preprocessing consistency
# =============================================================================

class TestH1_Preprocessing(unittest.TestCase):
    """
    H1: I sintetici usano all_leads_preprocessing (12 lead),
    i reali usano leads_preprocessing (singola lead).
    Se la funzione è diversa → distribuzione diversa.
    """

    def test_preprocessing_functions_equivalent(self):
        """Verifica che leads_preprocessing e all_leads_preprocessing
        producano lo stesso risultato sulle 6 lead periferiche."""
        from data.data_pipeline import leads_preprocessing, all_leads_preprocessing

        # Genera un segnale fake con 12 lead
        np.random.seed(42)
        n_samples = 2500  # 10s a 250Hz
        raw = {}
        for i, l in enumerate(ALL_LEADS):
            raw[l] = (np.random.randn(n_samples) * 500).astype(np.float32)

        # Approccio sintetico: all_leads_preprocessing
        synth_result = all_leads_preprocessing(raw)

        # Approccio reale: leads_preprocessing per ogni lead
        real_result = {}
        for l in LIMB_LEADS:
            real_result[l] = leads_preprocessing(raw[l])

        # Confronta le 6 lead periferiche
        max_diff = 0
        for l in LIMB_LEADS:
            diff = np.max(np.abs(synth_result[l] - real_result[l]))
            max_diff = max(max_diff, diff)
            print(f"  Lead {l}: max diff = {diff:.6f}")

        self.assertLess(max_diff, 1e-3,
            f"Le due funzioni di preprocessing differiscono di {max_diff:.6f}! "
            "Questo potrebbe causare domain gap.")


# =============================================================================
# H2. Test normalizzazione
# =============================================================================

class TestH2_Normalizzazione(unittest.TestCase):
    """
    H2: I sintetici normalizzano con reference_leads=LIMB_INDICES su 12 lead,
    i reali normalizzano solo le 6 lead.
    Se mediana/IQR è diversa → scala diversa.
    """

    def test_robust_scale_equivalence(self):
        """robust_scale_ecg(12 lead, ref=0-5) deve dare lo stesso risultato
        di robust_scale_ecg(6 lead) sulle prime 6."""
        np.random.seed(42)
        sig_12 = np.random.randn(12, 2500).astype(np.float32) * 500
        sig_6  = sig_12[:6, :].copy()

        norm_12, _, _ = robust_scale_ecg(sig_12, reference_leads=list(range(6)))
        norm_6 = robust_scale_ecg(sig_6)
        # norm_6 potrebbe essere una tupla o un array
        if isinstance(norm_6, tuple):
            norm_6 = norm_6[0]

        max_diff = np.max(np.abs(norm_12[:6, :] - norm_6))
        print(f"  Max diff normalizzazione 12-lead vs 6-lead: {max_diff:.8f}")
        self.assertLess(max_diff, 1e-5,
            f"Normalizzazione diversa! Diff={max_diff}")


# =============================================================================
# H3. Test distribuzione feature per lead
# =============================================================================

class TestH3_DistribuzionePerLead(unittest.TestCase):
    """
    Confronto dettagliato delle distribuzioni statistiche per ogni lead e classe.
    """

    @classmethod
    def setUpClass(cls):
        ok = os.path.exists(SYNTH_PATH) and os.path.exists(REAL_PATH)
        cls._skip = not ok
        if cls._skip:
            return
        cls.synth_x, cls.synth_y = _load_raw(SYNTH_PATH, n_per_class=200)
        cls.real_x,  cls.real_y  = _load_raw(REAL_PATH, n_per_class=200)

    def setUp(self):
        if self.__class__._skip:
            self.skipTest("Dataset mancanti")

    def test_media_per_lead_classe_normale(self):
        """La media per lead della classe normale deve essere simile."""
        s_norm = self.synth_x[self.synth_y == 0]
        r_norm = self.real_x[self.real_y == 0]
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
        print("\n  Classe NORMALE — media per lead:")
        for li, name in enumerate(lead_names):
            mean_s = np.mean(s_norm[:, li, :])
            mean_r = np.mean(r_norm[:, li, :])
            diff = abs(mean_s - mean_r)
            print(f"    {name}: synth={mean_s:.4f}  real={mean_r:.4f}  diff={diff:.4f}")

    def test_std_per_lead_classe_normale(self):
        """La std per lead della classe normale deve essere simile."""
        s_norm = self.synth_x[self.synth_y == 0]
        r_norm = self.real_x[self.real_y == 0]
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
        print("\n  Classe NORMALE — std per lead:")
        max_ratio = 0
        for li, name in enumerate(lead_names):
            std_s = np.std(s_norm[:, li, :])
            std_r = np.std(r_norm[:, li, :])
            ratio = max(std_s, std_r) / (min(std_s, std_r) + 1e-9)
            max_ratio = max(max_ratio, ratio)
            print(f"    {name}: synth={std_s:.4f}  real={std_r:.4f}  ratio={ratio:.2f}")
        # Tolleranza 5x (i reali hanno più variabilità)
        self.assertLess(max_ratio, 5.0,
            f"Std ratio troppo alto: {max_ratio:.2f}")

    def test_percentili_distribuzione(self):
        """Confronta percentili P5, P25, P50, P75, P95."""
        s_flat = self.synth_x[self.synth_y == 0].flatten()
        r_flat = self.real_x[self.real_y == 0].flatten()
        percs = [5, 25, 50, 75, 95]
        print("\n  Percentili classe NORMALE:")
        print(f"    {'P':>5}  {'Synth':>10}  {'Real':>10}  {'Ratio':>8}")
        for p in percs:
            ps = np.percentile(s_flat, p)
            pr = np.percentile(r_flat, p)
            ratio = abs(ps) / (abs(pr) + 1e-9) if abs(pr) > 0.01 else 0
            print(f"    {p:>5}  {ps:>10.4f}  {pr:>10.4f}  {ratio:>8.2f}")


# =============================================================================
# H4. Test confidence del modello per dominio
# =============================================================================

class TestH4_ConfidencePerDominio(unittest.TestCase):
    """
    Se il modello è molto più confident sui sintetici che sui reali,
    sta usando shortcut che non esistono nei reali.
    """

    @classmethod
    def setUpClass(cls):
        ok = all(os.path.exists(p) for p in [SYNTH_PATH, REAL_PATH, WEIGHTS])
        cls._skip = not ok
        if cls._skip:
            return
        cls.model = build_model((SAMPLES_PER_WINDOW, 6), N_CLASSES)
        cls.model.load_weights(WEIGHTS)

        synth_x, cls.synth_y = _load_raw(SYNTH_PATH, n_per_class=200)
        real_x,  cls.real_y  = _load_raw(REAL_PATH, n_per_class=200)

        synth_t = np.transpose(synth_x, (0, 2, 1))
        real_t  = np.transpose(real_x,  (0, 2, 1))

        cls.synth_probs = cls.model.predict(synth_t, batch_size=64, verbose=0)
        cls.real_probs  = cls.model.predict(real_t,  batch_size=64, verbose=0)

    def setUp(self):
        if self.__class__._skip:
            self.skipTest("File mancanti")

    def test_confidence_gap_globale(self):
        """Gap confidence medio synth-real."""
        conf_s = np.mean(np.max(self.synth_probs, axis=1))
        conf_r = np.mean(np.max(self.real_probs, axis=1))
        gap = conf_s - conf_r
        print(f"\n  Confidence media: synth={conf_s:.4f}  real={conf_r:.4f}  gap={gap:.4f}")
        # Se il gap > 0.15, il modello ha probabilmente imparato artefatti
        self.assertLess(gap, 0.25,
            f"Confidence gap={gap:.4f} > 0.25: possibile shortcut learning!")

    def test_confidence_per_classe(self):
        """Confidence per classe: dove il gap è maggiore?"""
        print("\n  Confidence per classe:")
        print(f"  {'Classe':<12} {'Synth':>8} {'Real':>8} {'Gap':>8}")
        print(f"  {'-'*40}")
        for c in range(N_CLASSES):
            ms = self.synth_y == c
            mr = self.real_y == c
            if ms.sum() == 0 or mr.sum() == 0:
                continue
            cs = np.mean(np.max(self.synth_probs[ms], axis=1))
            cr = np.mean(np.max(self.real_probs[mr], axis=1))
            gap = cs - cr
            flag = " [!]" if gap > 0.15 else ""
            print(f"  {CLASS_NAMES[c]:<12} {cs:>8.4f} {cr:>8.4f} {gap:>+8.4f}{flag}")

    def test_entropia_per_dominio(self):
        """Entropia delle predizioni per dominio."""
        eps = 1e-9
        h_s = np.mean(-np.sum(self.synth_probs * np.log(self.synth_probs + eps), axis=1))
        h_r = np.mean(-np.sum(self.real_probs * np.log(self.real_probs + eps), axis=1))
        print(f"\n  Entropia: synth={h_s:.4f}  real={h_r:.4f}  ratio={h_r/h_s:.2f}x")
        # Se entropia reale >> sintetica, il modello è molto più incerto sui reali

    def test_predizioni_incorrette_confidence(self):
        """Analizza la confidence media degli errori: se il modello sbaglia
        con alta confidence → shortcut learning."""
        synth_t = np.transpose(_load_raw(SYNTH_PATH, 200)[0], (0, 2, 1))
        real_t  = np.transpose(_load_raw(REAL_PATH, 200)[0], (0, 2, 1))

        # Errori sui reali
        y_pred_r = np.argmax(self.real_probs, axis=1)
        wrong_r  = y_pred_r != self.real_y
        if wrong_r.sum() > 0:
            conf_wrong_r = np.mean(np.max(self.real_probs[wrong_r], axis=1))
            print(f"\n  Errori reali: {wrong_r.sum()}/{len(self.real_y)}, "
                  f"confidence media errori={conf_wrong_r:.4f}")
            if conf_wrong_r > 0.80:
                print("  ⚠️ Il modello sbaglia con ALTA confidence sui reali!")
                print("  → Possibile shortcut learning dalla simulazione")


# =============================================================================
# H5. Test embedding shift (layer penultimo)
# =============================================================================

class TestH5_EmbeddingShift(unittest.TestCase):
    """
    Confronta le rappresentazioni latenti (embedding) del penultimo layer
    tra sintetici e reali. Se sono molto diversi → il modello vede due
    distribuzioni distinte.
    """

    @classmethod
    def setUpClass(cls):
        ok = all(os.path.exists(p) for p in [SYNTH_PATH, REAL_PATH, WEIGHTS])
        cls._skip = not ok
        if cls._skip:
            return

        import tensorflow as tf
        model = build_model((SAMPLES_PER_WINDOW, 6), N_CLASSES)
        model.load_weights(WEIGHTS)

        # Prendi il penultimo dense layer (embedding)
        embed_layer = None
        for layer in reversed(model.layers):
            if 'dense' in layer.name.lower() and layer != model.layers[-1]:
                embed_layer = layer
                break
        if embed_layer is None:
            # Fallback: prendi il GlobalAveragePooling
            for layer in reversed(model.layers):
                if 'pool' in layer.name.lower() or 'flatten' in layer.name.lower():
                    embed_layer = layer
                    break

        if embed_layer is None:
            cls._skip = True
            return

        cls.embed_model = tf.keras.Model(inputs=model.input, outputs=embed_layer.output)

        synth_x, cls.synth_y = _load_raw(SYNTH_PATH, n_per_class=200)
        real_x,  cls.real_y  = _load_raw(REAL_PATH, n_per_class=200)

        cls.synth_emb = cls.embed_model.predict(
            np.transpose(synth_x, (0, 2, 1)), batch_size=64, verbose=0)
        cls.real_emb = cls.embed_model.predict(
            np.transpose(real_x, (0, 2, 1)), batch_size=64, verbose=0)

        cls.embed_dim = cls.synth_emb.shape[-1]
        print(f"\n  Embedding dim: {cls.embed_dim}")

    def setUp(self):
        if self.__class__._skip:
            self.skipTest("File mancanti o layer non trovato")

    def test_cosine_similarity_embeddings(self):
        """Similarità coseno media tra centroidi synth e real per classe."""
        print("\n  Cosine similarity centroidi per classe:")
        for c in range(N_CLASSES):
            ms = self.synth_y == c
            mr = self.real_y == c
            if ms.sum() == 0 or mr.sum() == 0:
                continue
            centroid_s = np.mean(self.synth_emb[ms], axis=0)
            centroid_r = np.mean(self.real_emb[mr], axis=0)
            cos_sim = np.dot(centroid_s, centroid_r) / (
                np.linalg.norm(centroid_s) * np.linalg.norm(centroid_r) + 1e-9)
            flag = " [!]" if cos_sim < 0.80 else " [OK]"
            print(f"    {CLASS_NAMES[c]:<12} cos_sim={cos_sim:.4f}{flag}")

    def test_embedding_variance_ratio(self):
        """La varianza degli embedding reali deve essere comparabile ai sintetici."""
        var_s = np.mean(np.var(self.synth_emb, axis=0))
        var_r = np.mean(np.var(self.real_emb, axis=0))
        ratio = max(var_s, var_r) / (min(var_s, var_r) + 1e-9)
        print(f"\n  Embedding variance: synth={var_s:.4f}  real={var_r:.4f}  ratio={ratio:.2f}")

    def test_mean_embedding_distance(self):
        """Distanza euclidea media tra embedding synth e real per la stessa classe."""
        print("\n  Distanza L2 centroidi embedding:")
        for c in range(N_CLASSES):
            ms = self.synth_y == c
            mr = self.real_y == c
            if ms.sum() == 0 or mr.sum() == 0:
                continue
            c_s = np.mean(self.synth_emb[ms], axis=0)
            c_r = np.mean(self.real_emb[mr], axis=0)
            dist = np.linalg.norm(c_s - c_r)
            print(f"    {CLASS_NAMES[c]:<12} L2={dist:.4f}")


# =============================================================================
# H6. Test confusioni specifiche
# =============================================================================

class TestH6_ConfusioniSpecifiche(unittest.TestCase):
    """
    Analizza QUALI confusioni avvengono SOLO sui reali e non sui sintetici.
    Se una confusione è specifica del dominio reale → il gap è lì.
    """

    @classmethod
    def setUpClass(cls):
        ok = all(os.path.exists(p) for p in [SYNTH_PATH, REAL_PATH, WEIGHTS])
        cls._skip = not ok
        if cls._skip:
            return
        model = build_model((SAMPLES_PER_WINDOW, 6), N_CLASSES)
        model.load_weights(WEIGHTS)

        synth_x, cls.synth_y = _load_raw(SYNTH_PATH, n_per_class=200)
        real_x,  cls.real_y  = _load_raw(REAL_PATH, n_per_class=200)

        cls.synth_pred = np.argmax(model.predict(
            np.transpose(synth_x, (0, 2, 1)), batch_size=64, verbose=0), axis=1)
        cls.real_pred = np.argmax(model.predict(
            np.transpose(real_x, (0, 2, 1)), batch_size=64, verbose=0), axis=1)

    def setUp(self):
        if self.__class__._skip:
            self.skipTest("File mancanti")

    def test_confusioni_reali_vs_sintetici(self):
        """Stampa le top confusioni che avvengono SOLO sui reali."""
        from sklearn.metrics import confusion_matrix

        cm_s = confusion_matrix(self.synth_y, self.synth_pred, labels=range(N_CLASSES))
        cm_r = confusion_matrix(self.real_y,  self.real_pred,  labels=range(N_CLASSES))

        # Normalizza per riga (recall-based)
        cm_s_n = cm_s / (cm_s.sum(axis=1, keepdims=True) + 1e-9)
        cm_r_n = cm_r / (cm_r.sum(axis=1, keepdims=True) + 1e-9)

        # Differenza: dove i reali confondono di più
        diff = cm_r_n - cm_s_n

        print("\n  Confusioni specifiche dei REALI (diff > 0.05):")
        print(f"  {'Vero':<12} {'Predetto':<12} {'Synth%':>8} {'Real%':>8} {'Diff':>8}")
        print(f"  {'-'*52}")
        
        pairs = []
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                if i != j and diff[i, j] > 0.05:
                    pairs.append((diff[i, j], i, j))

        pairs.sort(reverse=True)
        for d, i, j in pairs:
            print(f"  {CLASS_NAMES[i]:<12} {CLASS_NAMES[j]:<12} "
                  f"{cm_s_n[i,j]*100:>7.1f}% {cm_r_n[i,j]*100:>7.1f}% {d*100:>+7.1f}%")
            if d > 0.15:
                print(f"  [!!!] GROSSO gap: {CLASS_NAMES[i]}->{CLASS_NAMES[j]}")


# =============================================================================
# Esecuzione
# =============================================================================

if __name__ == '__main__':
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestH1_Preprocessing, TestH2_Normalizzazione,
                TestH3_DistribuzionePerLead, TestH4_ConfidencePerDominio,
                TestH5_EmbeddingShift, TestH6_ConfusioniSpecifiche]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
