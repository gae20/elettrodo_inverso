import os
import sys
import numpy as np

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

def test_flat_lead():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(base_dir, "..", "best_model_unlabelled_z_median_limbs.weights.h5")
    
    test_id = 1157430 
    ecg_data = get_ecg(test_id)
    raw_signals = ecg_data["signals"]
    sigs_clean = all_leads_preprocessing(raw_signals)
    
    model = build_model((SAMPLES_PER_WINDOW, 6), 6)
    model.load_weights(weights_path)
    
    # 1. Baseline
    x_base = np.array([sigs_clean[l] for l in ALL_LEADS[:6]], dtype=np.float32)
    x_base_norm = np.transpose(robust_scale_ecg(x_base)[:, :SAMPLES_PER_WINDOW], (1, 0))
    p_base = model.predict(np.expand_dims(x_base_norm, 0), verbose=0)[0]
    
    print(f"Base: Pred={np.argmax(p_base)} Conf0={p_base[0]:.4f} Conf2={p_base[2]:.4f} Conf3={p_base[3]:.4f}")
    
    # 2. Flat Lead III
    sigs_flat3 = {l: np.copy(sigs_clean[l]) for l in ALL_LEADS[:6]}
    sigs_flat3['III'] = np.zeros_like(sigs_flat3['III'])
    x_flat3 = np.array([sigs_flat3[l] for l in ALL_LEADS[:6]], dtype=np.float32)
    x_flat3_norm = np.transpose(robust_scale_ecg(x_flat3)[:, :SAMPLES_PER_WINDOW], (1, 0))
    p_flat3 = model.predict(np.expand_dims(x_flat3_norm, 0), verbose=0)[0]
    
    print(f"Flat Lead III: Pred={np.argmax(p_flat3)} Conf0={p_flat3[0]:.4f} Conf2={p_flat3[2]:.4f} Conf3={p_flat3[3]:.4f}")
    
    # 3. Flat Lead LL (which affects II and III in Einthoven) - wait, we just flat lead II
    sigs_flat2 = {l: np.copy(sigs_clean[l]) for l in ALL_LEADS[:6]}
    sigs_flat2['II'] = np.zeros_like(sigs_flat2['II'])
    x_flat2 = np.array([sigs_flat2[l] for l in ALL_LEADS[:6]], dtype=np.float32)
    x_flat2_norm = np.transpose(robust_scale_ecg(x_flat2)[:, :SAMPLES_PER_WINDOW], (1, 0))
    p_flat2 = model.predict(np.expand_dims(x_flat2_norm, 0), verbose=0)[0]
    
    print(f"Flat Lead II: Pred={np.argmax(p_flat2)} Conf0={p_flat2[0]:.4f} Conf2={p_flat2[2]:.4f} Conf3={p_flat2[3]:.4f}")

if __name__ == "__main__":
    test_flat_lead()
