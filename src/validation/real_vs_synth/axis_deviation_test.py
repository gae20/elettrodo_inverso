import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Aggiungi src al path per importare i moduli interni
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data.data_pipeline import get_ecg, all_leads_preprocessing
from models.ldensenet import build_model
from utils.config import SAMPLES_PER_WINDOW, ALL_LEADS

def robust_scale_ecg(sigs_array, eps=1e-8):
    x = sigs_array.astype(np.float32)
    medians = np.median(x, axis=1, keepdims=True)
    q75, q25 = np.percentile(x, [75, 25])
    iqr_global = q75 - q25
    scale_global = iqr_global / 1.34896
    scale_global = max(scale_global, eps)
    x_norm = (x - medians) / scale_global
    return x_norm

def test_axis_deviation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, "..", "best_model_unlabelled_z_median_limbs.weights.h5")
    
    # ID di un ECG normale noto che prima era predetto correttamente
    test_id = 1157430 
    
    print(f"Caricamento ECG normale {test_id} per test Varianza Fisiologica...")
    ecg_data = get_ecg(test_id)
    if not ecg_data:
        print("ECG non trovato.")
        return
    
    raw_signals = ecg_data["signals"]
    sigs_clean = all_leads_preprocessing(raw_signals)
    
    # Caricamento Modello
    model = build_model((SAMPLES_PER_WINDOW, 6), 6)
    model.load_weights(weights_path)
    
    # Test scalatura Lead III da +1.0 a -1.0
    scales = np.linspace(1.0, -1.0, 10)
    
    print(f"\nAlterazione Progressiva della derivazione III (simulazione deviazione assiale/variante normale):")
    print(f"{'Scale Lead III':<15} | {'Predizione':<12} | {'Confidenza Classe 0':<20} | {'Confidenza Classe 3':<20}")
    print("-" * 75)
    
    for scale in scales:
        # Copia i segnali originali puliti
        modified_sigs = {l: np.copy(sigs_clean[l]) for l in ALL_LEADS[:6]}
        
        # Alteriamo SOLO la Lead III (indice 2)
        modified_sigs['III'] = modified_sigs['III'] * scale
        
        # Prepara tensore
        x_mod = np.array([modified_sigs[l] for l in ALL_LEADS[:6]], dtype=np.float32)
        x_mod_norm = np.transpose(robust_scale_ecg(x_mod)[:, :SAMPLES_PER_WINDOW], (1, 0))
        
        # Inferenza
        p_probs = model.predict(np.expand_dims(x_mod_norm, 0), verbose=0)[0]
        pred_class = np.argmax(p_probs)
        conf_0 = p_probs[0]
        conf_3 = p_probs[3]
        
        pred_str = f"Classe {pred_class}"
        if pred_class != 0:
            pred_str = f"-> CLASSE {pred_class} <-"
            
        print(f"{scale:<15.2f} | {pred_str:<12} | {conf_0:<20.4f} | {conf_3:<20.4f}")

if __name__ == "__main__":
    test_axis_deviation()
