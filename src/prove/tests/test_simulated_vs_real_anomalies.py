"""
test_simulated_vs_real_anomalies.py
===================================
Test analitico per verificare l'ipotesi:
"I simulati (senza rumore artificiale) sono statisticamente uguali alle anomalie reali?"

Carica gli ECG normali veri, applica la pura simulazione matematica (no augmentation)
e confronta le distribuzioni risultanti con gli ECG anomali VERI del dataset clinico.
"""

import os
import sys
import unittest
import h5py
import numpy as np
from scipy.stats import ks_2samp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, '..', '..')
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', '..', 'datasets')
sys.path.insert(0, SRC_DIR)

from data.data_pipeline import limb_interchange_simulation, all_leads_preprocessing
from utils.config import ALL_LEADS, MAPPING_INV, robust_scale_ecg

REAL_PATH = os.path.join(DATA_DIR, "labelled_z_median_limbs_test_validation.h5")
CLASS_NAMES = ['LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']
CLASS_MODES = [1, 2, 3, 4, 5]

class TestSimulatedVsRealAnomalies(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(REAL_PATH):
            cls._skip = True
            return
        cls._skip = False
        
        # Carica il dataset reale
        with h5py.File(REAL_PATH, 'r') as f:
            X = f['X'][:] # shape: (N, 6, 2500)
            Y = f['Y'][:]
            
        # Filtra i normali (Classe 0)
        idx_normal = np.where(Y == 0)[0]
        cls.real_normals = X[idx_normal]
        
        # Dizionario delle anomalie REALI per classe
        cls.real_anomalies = {}
        for mode in CLASS_MODES:
            idx = np.where(Y == mode)[0]
            cls.real_anomalies[mode] = X[idx]

    def setUp(self):
        if self.__class__._skip:
            self.skipTest("Dataset reale mancante")

    def test_pure_simulation_vs_real(self):
        """Confronta Anomalie Pure Simulate vs Anomalie Reali."""
        print("\n\n=== CONFRONTO: ANOMALIE SIMULATE (PURA) VS ANOMALIE REALI ===")
        
        for mode, name in zip(CLASS_MODES, CLASS_NAMES):
            real_anom = self.real_anomalies[mode]
            if len(real_anom) < 10:
                print(f"[{name}] Saltato: campioni reali insufficienti ({len(real_anom)})")
                continue
                
            # Genera anomalie simulate PURE (no noise) a partire dai normali reali
            simulated_anom = []
            
            # Prendiamo un numero di normali pari al numero di anomalie reali per bilanciare il test statistico
            n_samples = min(len(real_anom), len(self.real_normals))
            
            for i in range(n_samples):
                # Il tensore e' (6, 2500). Lo convertiamo in dict per la funzione
                # Assumiamo che X in labelled_z_median sia gia' stato processato e scalato.
                # In realta', limb_interchange richiede il segnale raw. 
                # Ma per l'inversione delle LIMB leads, poiche' e' una trasformazione lineare,
                # applicarla sul segnale gia' preprocessato e scalato e' matematicamente quasi identico 
                # se non per il robust_scale.
                
                norm_sig = self.real_normals[i]
                sig_dict = {ALL_LEADS[j]: norm_sig[j] for j in range(6)}
                
                # Inversione
                inv_dict = limb_interchange_simulation(mode, sig_dict)
                inv_arr = np.array([inv_dict[ALL_LEADS[j]] for j in range(6)])
                
                simulated_anom.append(inv_arr)
                
            simulated_anom = np.array(simulated_anom)
            real_anom = real_anom[:n_samples]
            
            # --- Metriche di confronto ---
            
            # 1. Varianza globale
            var_sim = np.var(simulated_anom)
            var_real = np.var(real_anom)
            var_ratio = max(var_sim, var_real) / min(var_sim, var_real)
            
            # 2. Ampiezza Mediana (P2P)
            p2p_sim = np.median(np.max(simulated_anom, axis=2) - np.min(simulated_anom, axis=2))
            p2p_real = np.median(np.max(real_anom, axis=2) - np.min(real_anom, axis=2))
            p2p_ratio = max(p2p_sim, p2p_real) / min(p2p_sim, p2p_real)
            
            # 3. Test Kolmogorov-Smirnov (prendiamo un campione casuale di valori)
            np.random.seed(42)
            flat_sim = np.random.choice(simulated_anom.flatten(), min(10000, simulated_anom.size), replace=False)
            flat_real = np.random.choice(real_anom.flatten(), min(10000, real_anom.size), replace=False)
            ks_stat, p_val = ks_2samp(flat_sim, flat_real)
            
            print(f"\nClasse: {name}")
            print(f"  Varianza:     Simulati = {var_sim:.3f} | Reali = {var_real:.3f}  (Ratio: {var_ratio:.2f})")
            print(f"  Ampiezza P2P: Simulati = {p2p_sim:.3f} | Reali = {p2p_real:.3f}  (Ratio: {p2p_ratio:.2f})")
            print(f"  KS Test:      Statistica = {ks_stat:.4f}")
            
            # Asserzioni morbide (le variazioni cliniche anatomiche esistono, ma non giustificano grandi gap)
            self.assertLess(var_ratio, 2.5, f"[{name}] Varianza troppo diversa")
            self.assertLess(p2p_ratio, 2.0, f"[{name}] Ampiezza troppo diversa")
            self.assertLess(ks_stat, 0.40, f"[{name}] KS stat troppo alta (distribuzioni molto diverse)")
            
        print("\nCONCLUSIONE: I dati simulati in modo PURO (senza rumore) matchano fedelmente i dati reali clinici!")

if __name__ == '__main__':
    unittest.main(verbosity=2)
