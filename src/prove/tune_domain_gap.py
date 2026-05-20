import os
import sys
import copy
import h5py
import numpy as np
import zipfile
from scipy.stats import ks_2samp

# Import locali
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.data_pipeline import (
    read_edf_data, all_leads_preprocessing, check_ecg_quality, 
    check_window_quality, limb_interchange_simulation, apply_electrode_gain, add_extra_noise,
    apply_random_scaling, add_baseline_wander
)
from utils.config import (
    SAMPLES_PER_WINDOW, FS_NEW, FS_OLD,
    ALL_LEADS, MAPPING_INV, ACTIVE_SYNTH_CLASSES, 
    QUALITY_CFG, robust_scale_ecg
)
from data.generate_ids import get_clean_ecg_ids, IS_HOLTER_DICT
from prove.build_unlabelled_global_zscore_dataset import build_zip_index, create_windows, compute_good_window_mask_from_raw

LIMB_INDICES = list(range(6))
STRIDE_SAMPLES = int(FS_NEW * 2.0)

# QUALITY_CFG_HOLTER è quella originale (più tollerante)
QUALITY_CFG_HOLTER = copy.deepcopy(QUALITY_CFG)

# QUALITY_CFG_STANDARD è più rigorosa per gli ECG da 10s
QUALITY_CFG_STANDARD = copy.deepcopy(QUALITY_CFG)
QUALITY_CFG_STANDARD["baseline_max_uv"] = 500.0
QUALITY_CFG_STANDARD["mad_noise_limb"] = 15.0  # PIÙ SEVERA (era 20.0)
QUALITY_CFG_STANDARD["mad_noise_prec"] = 20.0  # PIÙ SEVERA (era 25.0)
QUALITY_CFG_STANDARD["min_valid_ratio"] = 0.70

def tune_domain_gap():
    # 1. Carica le distribuzioni reali
    base_dir = os.path.dirname(os.path.abspath(__file__))
    real_h5 = os.path.join(base_dir, "..", "..", "..", "datasets", "labelled_z_median_limbs_test_validation.h5")
    
    if not os.path.exists(real_h5):
        print(f"[Errore] Il dataset reale non esiste in {real_h5}")
        print("Devi prima eseguire testset_validation.py per ottenere la baseline clinica!")
        return

    print("Caricamento dataset reale per la baseline...")
    real_data = {}
    with h5py.File(real_h5, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
        for i in range(6):
            real_data[i] = X[Y == i].flatten()
            if len(real_data[i]) == 0:
                print(f"[Avviso] Nessun dato reale per la classe {i}")

    # 2. Genera un mini-dataset sintetico
    print("\nGenerazione di un mini-dataset sintetico per il tuning (50 ECG)...")
    db_path = os.path.join(base_dir, '..', 'data', 'records_complete.db')
    dataset_dir = os.path.join(base_dir, '..', '..', '..', 'datasets', 'dataset_normals')
    
    id_to_zip = build_zip_index(dataset_dir)
    clean_ids = get_clean_ecg_ids(db_path, max_ecgs=100) # Estraiamo fino a 100
    clean_ids = [cid for cid in clean_ids if str(cid) in id_to_zip][:50] # Ne usiamo 50
    
    synth_data = {i: [] for i in range(6)}
    all_mapping = ['normale'] + list(ACTIVE_SYNTH_CLASSES)
    
    for ecg_id in clean_ids:
        zip_path = id_to_zip.get(str(ecg_id))
        try:
            with zipfile.ZipFile(zip_path, 'r') as z_in:
                edf_bytes = z_in.read(f"{ecg_id}.edf")
                ecg_data = read_edf_data(edf_bytes)
                
            if not ecg_data or not ecg_data["signals"]: continue
            
            # --- 1. APPLICAZIONE AUGMENTATION GLOBALI AL RAW ---
            # Gain base (molto basso per non affogare nel rumore bianco)
            raw_base = apply_electrode_gain(ecg_data["signals"], fs=FS_OLD, noise_multiplier=0.5)
            # Scaling randomico
            raw_base = apply_random_scaling(raw_base, min_scale=0.6, max_scale=1.4)
            # Baseline wander (simula patologia/anzianità e alza IQR)
            raw_base = add_baseline_wander(raw_base, fs=FS_OLD, intensity=300.0)
            
            # --- 2. SQA SU DATI PULITI (Per non scartare ECG validi) ---
            sigs_sqa = all_leads_preprocessing(raw_base)
            sigs_array_sqa = np.array([sigs_sqa[l] for l in ALL_LEADS], dtype=np.float32)
            
            is_holter = IS_HOLTER_DICT.get(ecg_id, False)
            cfg = QUALITY_CFG_HOLTER if is_holter else QUALITY_CFG_STANDARD
            cfg["stride_sec"] = 2.0 

            quality_result = check_ecg_quality(sigs_array_sqa, cfg=cfg, lead_indices=LIMB_INDICES)
            if not quality_result['global_valid']: continue
            
            win_mask = compute_good_window_mask_from_raw(sigs_array_sqa, cfg=cfg, min_valid_leads_per_window=5, lead_indices=LIMB_INDICES)
            if win_mask.size == 0 or not win_mask.any(): continue
            
            # Estraggo il numero di finestre valide
            sigs_array_normale, _, _ = robust_scale_ecg(sigs_array_sqa, reference_leads=LIMB_INDICES)
            sigs_norm_dict = {lead: sigs_array_normale[i] for i, lead in enumerate(ALL_LEADS)}
            wins_all_sqa = create_windows(sigs_norm_dict, stride=STRIDE_SAMPLES)
            n_win = min(wins_all_sqa.shape[0], win_mask.size)
            
            # --- 3. CREAZIONE CLASSI CON EXTRA NOISE UNIFORME ---
            # Creiamo un "extra_mult" che vale PER TUTTE LE CLASSI DI QUESTO RECORD (compresa la Normale!)
            # Questo evita il shortcut learning perché il rumore non è più legato alla classe
            global_extra_mult = np.random.uniform(0.5, 1.5)
            
            # ---> CLASSE NORMALE <---
            raw_norm_noisy = add_extra_noise(raw_base, multiplier=global_extra_mult, fs=FS_OLD)
            sigs_norm = all_leads_preprocessing(raw_norm_noisy)
            sigs_array_norm = np.array([sigs_norm[l] for l in ALL_LEADS], dtype=np.float32)
            sigs_norm_scaled, _, _ = robust_scale_ecg(sigs_array_norm, reference_leads=LIMB_INDICES)
            sigs_norm_scaled_dict = {lead: sigs_norm_scaled[i] for i, lead in enumerate(ALL_LEADS)}
            
            wins_norm = create_windows(sigs_norm_scaled_dict, stride=STRIDE_SAMPLES)
            wins_norm_good = wins_norm[:n_win][win_mask[:n_win]]
            if wins_norm_good.shape[0] > 0:
                synth_data[0].append(wins_norm_good.flatten())
                
            # ---> CLASSI DI INVERSIONE <---
            for inv_idx, inv_name in enumerate(ACTIVE_SYNTH_CLASSES):
                class_int = inv_idx + 1
                # Applico scambio sul raw_base
                raw_inv = limb_interchange_simulation(MAPPING_INV[inv_name], raw_base)
                
                # Aggiungo lo STESSO extra noise della classe normale
                raw_inv_noisy = add_extra_noise(raw_inv, multiplier=global_extra_mult, fs=FS_OLD)
                sim_sigs = all_leads_preprocessing(raw_inv_noisy)
                    
                sim_sigs_array = np.array([sim_sigs[l] for l in ALL_LEADS], dtype=np.float32)
                sim_sigs_norm, _, _ = robust_scale_ecg(sim_sigs_array, reference_leads=LIMB_INDICES)
                sim_sigs_norm_dict = {lead: sim_sigs_norm[i] for i, lead in enumerate(ALL_LEADS)}
                
                wins_s = create_windows(sim_sigs_norm_dict, stride=STRIDE_SAMPLES)
                wins_s_good = wins_s[:n_win][win_mask[:n_win]]
                if wins_s_good.shape[0] > 0:
                    synth_data[class_int].append(wins_s_good.flatten())
                    
        except Exception as e:
            pass

    # 3. Calcolo metriche
    print("\n" + "="*70)
    print("= RISULTATI DOMAIN GAP TUNE (Reale vs Sintetico) =")
    print("="*70)
    print(f"{'Classe':<15} | {'Var Real':<10} | {'Var Synth':<10} | {'Ratio (S/R)':<12} | {'KS Stat':<8}")
    print("-" * 70)
    
    for c_idx in range(6):
        c_name = all_mapping[c_idx]
        if len(real_data.get(c_idx, [])) == 0:
            print(f"{c_name:<15} | Manca Dati Reali")
            continue
        if len(synth_data.get(c_idx, [])) == 0:
            print(f"{c_name:<15} | Manca Dati Sintetici (troppo rumore e bocciati da SQA?)")
            continue
            
        r_arr = real_data[c_idx]
        s_arr = np.concatenate(synth_data[c_idx])
        
        # Sottocampionamento per stabilità statistica
        np.random.seed(42)
        n_samples = min(50000, len(r_arr), len(s_arr))
        r_sub = np.random.choice(r_arr, n_samples, replace=False)
        s_sub = np.random.choice(s_arr, n_samples, replace=False)
        
        var_r = float(np.var(r_sub))
        var_s = float(np.var(s_sub))
        ratio = var_s / var_r if var_r > 0 else 0
        
        stat, p_val = ks_2samp(s_sub, r_sub)
        
        print(f"{c_name:<15} | {var_r:<10.4f} | {var_s:<10.4f} | {ratio:<12.4f} | {stat:<8.4f}")
        
    print("="*70)
    print("[SUGGERIMENTI]")
    print("- Ratio (S/R) < 0.8 : Sintetico TROPPO PULITO. Aumenta 'extra_mult' o 'noise_multiplier' (gain).")
    print("- Ratio (S/R) > 1.2 : Sintetico TROPPO RUMOROSO. Riduci 'extra_mult'.")
    print("- Ideale           : Ratio vicino a 1.0 (tra 0.9 e 1.1).")
    print("\nSe modifichi i parametri qui dentro e sei soddisfatto, ricordati di")
    print("copiarli anche in 'build_unlabelled_global_zscore_dataset.py'!")

if __name__ == '__main__':
    tune_domain_gap()
