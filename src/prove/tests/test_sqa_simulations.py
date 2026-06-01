import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.config import ALL_LEADS
from data.data_pipeline import limb_interchange_simulation
from prove.build_unlabelled_global_zscore_dataset import compute_good_window_mask_from_raw
from utils.sqa_real_config import QUALITY_CFG_SYNTH_RELAXED

class TestSQAOnSimulations(unittest.TestCase):
    def test_sqa_rejection_rates(self):
        # Generiamo un segnale realistico
        np.random.seed(42)
        fs = QUALITY_CFG_SYNTH_RELAXED["fs"]
        n_samples = 10 * fs
        t = np.linspace(0, 10, n_samples)
        
        # Creiamo un ECG finto con un po' di baseline wander
        raw_signals = {}
        for l in ALL_LEADS:
            noise = np.random.normal(0, 50, n_samples)
            wander = np.sin(2 * np.pi * 0.5 * t) * 200
            # Aggiungiamo dei picchi QRS finti ogni secondo
            qrs = np.zeros(n_samples)
            for i in range(fs, n_samples, fs):
                qrs[i:i+10] = 1000
                qrs[i+10:i+20] = -500
            raw_signals[l] = noise + wander + qrs
            
        raw_arr = np.array([raw_signals[l] for l in ALL_LEADS], dtype=np.float32)
        
        # Controllo SQA su normale
        mask_norm = compute_good_window_mask_from_raw(
            raw_arr, cfg=QUALITY_CFG_SYNTH_RELAXED, lead_indices=list(range(6))
        )
        norm_valid = mask_norm.sum()
        print(f"\nNormale: {norm_valid}/{len(mask_norm)} finestre valide")
        
        # Controllo SQA sulle classi simulate
        modes = [(1, 'LA-RA'), (2, 'RA-LL'), (3, 'LA-LL'), (4, 'ROT_ORA'), (5, 'ROT_ANT')]
        for mode, name in modes:
            sim_dict = limb_interchange_simulation(mode, raw_signals)
            sim_arr = np.array([sim_dict[l] for l in ALL_LEADS], dtype=np.float32)
            mask_sim = compute_good_window_mask_from_raw(
                sim_arr, cfg=QUALITY_CFG_SYNTH_RELAXED, lead_indices=list(range(6))
            )
            sim_valid = mask_sim.sum()
            print(f"{name}: {sim_valid}/{len(mask_sim)} finestre valide")
            
            # Devono scartare esattamente le stesse finestre!
            np.testing.assert_array_equal(mask_norm, mask_sim, 
                err_msg=f"L'SQA scarta finestre diverse per {name} rispetto al normale!")

if __name__ == '__main__':
    unittest.main(verbosity=2)
