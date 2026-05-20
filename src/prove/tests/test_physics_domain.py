"""
test_physics_domain.py
======================
Test per verificare che le augmentation di rumore e artefatti rispettino
le leggi della fisica (Triangolo di Einthoven) e contengano le giuste
caratteristiche spettrali (EMG, wander stocastico).
"""

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from data.data_pipeline import (
    apply_electrode_gain, add_baseline_wander, add_extra_noise
)
from utils.config import FS_OLD, ALL_LEADS

def make_clean_ecg(n_seconds=10, fs=FS_OLD):
    """Crea un finto ECG piatto (zero) per testare solo il rumore."""
    n = int(n_seconds * fs)
    return {lead: np.zeros(n, dtype=np.float32) for lead in ALL_LEADS}

def check_einthoven(signals, tol=1e-3):
    """Verifica II = I + III per ogni campione."""
    # Nelle augmentation, Lead II dovrebbe essere uguale a I + III a meno di errori numerici
    diff = np.abs(signals['II'] - (signals['I'] + signals['III']))
    return np.max(diff) < tol

def check_wct_precordials(signals, tol=1e-3):
    """Controlla se WCT e' usato, ma poiche' modifichiamo i fisici V1-V6, 
    questo dipendera' da come WCT e' calcolato. E' opzionale."""
    pass

class TestPhysicsDomain(unittest.TestCase):

    def test_electrode_gain_einthoven(self):
        """Il rumore termico/gain deve rispettare Einthoven."""
        np.random.seed(42)
        # Segnale con deviazione standard per far generare rumore
        clean = {lead: np.random.normal(0, 100, int(10*FS_OLD)).astype(np.float32) for lead in ALL_LEADS}
        noisy = apply_electrode_gain(clean, noise_multiplier=2.0)
        
        # Estrai il solo rumore
        noise = {l: noisy[l] - clean[l] for l in ALL_LEADS}
        self.assertTrue(check_einthoven(noise), "apply_electrode_gain viola Einthoven!")

    def test_baseline_wander_einthoven_and_stochastic(self):
        """Il wander deve rispettare Einthoven ed essere stocastico."""
        np.random.seed(42)
        clean = make_clean_ecg()
        noisy = add_baseline_wander(clean, intensity=300.0)
        noise = noisy
        
        self.assertTrue(check_einthoven(noise), "add_baseline_wander viola Einthoven!")
        
        # Verifica che non sia puramente periodico. 
        # La stocasticita' implica che il segnale contiene varie frequenze a bassa frequenza.
        fft_I = np.abs(np.fft.rfft(noise['I']))
        self.assertGreater(np.sum(fft_I > np.max(fft_I)*0.1), 3, "Il wander sembra essere puramente sinusoidale.")

    def test_extra_noise_einthoven_and_emg(self):
        """Il rumore extra deve rispettare Einthoven e avere energia EMG (20-120Hz)."""
        np.random.seed(42)
        clean = {lead: np.random.normal(0, 100, int(10*FS_OLD)).astype(np.float32) for lead in ALL_LEADS}
        noisy = add_extra_noise(clean, multiplier=2.0)
        noise = {l: noisy[l] - clean[l] for l in ALL_LEADS}
        
        self.assertTrue(check_einthoven(noise), "add_extra_noise viola Einthoven!")
        
        # Verifica spettrale per EMG (deve esserci energia nel range 20-120Hz)
        fft_val = np.abs(np.fft.rfft(noise['I']))
        freqs = np.fft.rfftfreq(len(noise['I']), 1.0/FS_OLD)
        
        low_energy = np.sum(fft_val[(freqs >= 0) & (freqs < 10)]**2)
        emg_energy = np.sum(fft_val[(freqs >= 20) & (freqs < 120)]**2)
        
        # C'e' sia motion (bassa freq) che EMG (alta freq)
        self.assertGreater(emg_energy, 0, "Nessun rumore EMG trovato.")
        
    def test_extra_noise_motion_artifacts(self):
        """Verifica la presenza di motion artifacts a gradino."""
        np.random.seed(1)
        clean = {lead: np.random.normal(0, 100, int(10*FS_OLD)).astype(np.float32) for lead in ALL_LEADS}
        
        found_jump = False
        for i in range(10): # 20% prob, in 10 iterazioni dovrebbe apparirne uno
            noisy = add_extra_noise(clean, multiplier=0.0) # azzeriamo EMG
            noise = noisy['I'] - clean['I']
            diffs = np.diff(noise)
            if np.max(np.abs(noise)) > 10.0: # Se c'è un artefatto, l'ampiezza supererà zero (essendo l'EMG nullo qui)
                found_jump = True
                break
                
        self.assertTrue(found_jump, "Nessun motion artifact rilevato su 10 tentativi.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
