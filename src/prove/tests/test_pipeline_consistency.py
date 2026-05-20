import os
import sys
import unittest
import numpy as np

# Aggiungi cartella src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.config import robust_scale_ecg, ALL_LEADS, SAMPLES_PER_WINDOW, STRIDE_SAMPLES

# Import funzioni dalle pipeline (ipotizzando che siano importabili, 
# se non lo sono replichiamo la logica base per testarla)
from prove.build_unlabelled_global_zscore_dataset import create_windows

class TestPipelineConsistency(unittest.TestCase):

    def setUp(self):
        # Genera un segnale mock di 12 derivazioni x 1000 campioni
        np.random.seed(42)
        self.mock_signal_12 = np.random.randn(12, 1000).astype(np.float32)
        self.LIMB_INDICES = list(range(6))
        self.mock_signal_6 = self.mock_signal_12[:6, :]
        
    def test_robust_scale_consistency(self):
        """
        Verifica che l'applicazione dello scaler sulle 12 derivazioni 
        usando le prime 6 come reference sia matematicamente IDENTICA 
        all'applicazione dello scaler solo sulle prime 6 derivazioni.
        """
        # --- Approccio Sintetico (12 derivazioni con reference) ---
        norm_12, medians_12, scale_12 = robust_scale_ecg(
            self.mock_signal_12, 
            reference_leads=self.LIMB_INDICES
        )
        # Estraiamo solo le prime 6 dal risultato
        synth_limbs_result = norm_12[:6, :]
        
        # --- Approccio Reale (Solo 6 derivazioni) ---
        norm_6 = robust_scale_ecg(self.mock_signal_6)
        
        # Le prime 6 derivazioni scalate dovrebbero essere numericamente identiche
        np.testing.assert_array_almost_equal(synth_limbs_result, norm_6, decimal=6)
        
    def test_create_windows_shape(self):
        """
        Verifica che la funzione create_windows generi la shape corretta
        rispetto alla configurazione in config.py
        """
        # Creiamo un dizionario finto per il create_windows
        mock_dict = {ALL_LEADS[i]: self.mock_signal_12[i] for i in range(12)}
        
        windows = create_windows(
            mock_dict, 
            lead_order=ALL_LEADS, 
            win_size=SAMPLES_PER_WINDOW, 
            stride=STRIDE_SAMPLES
        )
        
        # Calcoliamo quante finestre ci aspettiamo
        # start_idx: 0, 500 (se stride è 500 e len è 1000)
        expected_windows = len(range(0, 1000 - SAMPLES_PER_WINDOW + 1, STRIDE_SAMPLES))
        
        self.assertEqual(windows.shape[0], expected_windows)
        self.assertEqual(windows.shape[1], 12)
        self.assertEqual(windows.shape[2], SAMPLES_PER_WINDOW)

if __name__ == '__main__':
    unittest.main()
