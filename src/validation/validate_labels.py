import os
import sys
import numpy as np
import pandas as pd
from scipy import signal

# Setup paths relative to src/validation
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.join(SCRIPT_DIR, '..')
THESIS_DIR = os.path.join(SRC_DIR, '..')

sys.path.append(SRC_DIR)
from data.data_pipeline import _parse_edf_file, leads_preprocessing

# Mappings
LABEL_MAPPING = {
    'normale':    0,
    'RL':         1,  # LA-RA
    'RF':         2,  # RA-LL
    'LF':         3,  # LA-LL
    'orario':     4,  # ROT_ORA
    'antiorario': 5   # ROT_ANT
}
CLASS_NAMES = ['normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']

SMALL_DIR  = os.path.join(THESIS_DIR, 'datasets', 'dataset', 'dataset_small')
CSV_PATH   = os.path.join(SMALL_DIR, 'thesis-sample.csv')

def get_qrs_polarity_vector(signals_dict):
    """
    Extracts QRS net amplitude (R-peak + S-trough) for each lead.
    We use V6 as a stable reference for heartbeat detection.
    """
    fs = 250  # We preprocess to 250Hz
    
    # Preprocess leads
    leads_list = ['I', 'II', 'III', 'aVr', 'aVl', 'aVf', 'V6']
    prep_sigs = {}
    for l in leads_list:
        if l not in signals_dict:
            return None
        prep_sigs[l] = leads_preprocessing(signals_dict[l])
        
    v6 = prep_sigs['V6']
    if len(v6) < 500:
        return None
        
    # High-pass filter V6 to emphasize QRS (2 to 30 Hz)
    sos = signal.butter(4, [2.0, 30.0], btype='bandpass', fs=fs, output='sos')
    v6_filt = signal.sosfiltfilt(sos, v6)
    
    # Find peaks on squared signal (local energy)
    v6_sq = v6_filt ** 2
    thr = np.percentile(v6_sq, 90) * 0.4
    thr = max(thr, 100.0) # minimal amplitude threshold
    peaks, _ = signal.find_peaks(v6_sq, distance=int(0.4 * fs), height=thr)
    
    if len(peaks) < 3:
        return None
        
    # Extract QRS net amplitudes around each peak
    # Window: -40ms to +80ms (around peak)
    pre_samples = int(0.04 * fs)
    post_samples = int(0.08 * fs)
    
    beat_amplitudes = {l: [] for l in leads_list}
    
    for p in peaks:
        if p - pre_samples < 0 or p + post_samples >= len(v6):
            continue
            
        for l in leads_list:
            seg = prep_sigs[l][p - pre_samples : p + post_samples]
            if len(seg) == 0:
                continue
            # Net amplitude = max(x) + min(x)
            seg_centered = seg - np.median(seg)
            max_val = np.max(seg_centered)
            min_val = np.min(seg_centered)
            net_amp = max_val + min_val
            beat_amplitudes[l].append(net_amp)
            
    # Calculate mean polarity vector
    mean_vector = {}
    for l in leads_list:
        amps = beat_amplitudes[l]
        if len(amps) == 0:
            return None
        mean_vector[l] = np.mean(amps)
        
    return mean_vector

def restore_leads(vec, c):
    """
    Applies the inverse misplacement transformation for class c.
    Returns the restored vector: [I, II, III, aVR, aVL, aVF]
    """
    I_m, II_m, III_m = vec['I'], vec['II'], vec['III']
    aVR_m, aVL_m, aVF_m = vec['aVr'], vec['aVl'], vec['aVf']
    
    if c == 0: # Normale
        return I_m, II_m, III_m, aVR_m, aVL_m, aVF_m
    elif c == 1: # LA-RA swap (RL in label)
        return -I_m, III_m, II_m, aVL_m, aVR_m, aVF_m
    elif c == 2: # RA-LL swap (RF in label)
        return -III_m, -II_m, -I_m, aVF_m, aVL_m, aVR_m
    elif c == 3: # LA-LL swap (LF in label)
        return II_m, I_m, -III_m, aVR_m, aVF_m, aVL_m
    elif c == 4: # ROT_ORA (orario in label)
        return -II_m, -III_m, I_m, aVF_m, aVR_m, aVL_m
    elif c == 5: # ROT_ANT (antiorario in label)
        return III_m, -I_m, -II_m, aVL_m, aVF_m, aVR_m
    else:
        return 0, 0, 0, 0, 0, 0

def compute_plausibility_score(I_r, II_r, III_r, aVR_r, aVL_r, aVF_r, V6_val):
    """
    Computes a physiological score for the restored leads.
    Higher score means the combination is highly consistent with a real healthy or pathologic ECG.
    """
    score = 0.0
    
    # 1. aVR Negativity Check (Crucial: aVR must be negative)
    if aVR_r < 0:
        score += 5.0
        max_abs = max(abs(I_r), abs(II_r), abs(III_r), 1e-5)
        score += min(2.0, -aVR_r / max_abs)
    else:
        score -= 10.0
        
    # 2. Heart Axis Check
    theta = np.arctan2(aVF_r, I_r) * 180.0 / np.pi
    
    if -30.0 <= theta <= 90.0:
        score += 4.0 # Normoaxis
    elif -90.0 <= theta < -30.0:
        score += 2.0 # Left axis deviation (common)
    elif 90.0 < theta <= 120.0:
        score += 2.0 # Right axis deviation (common)
    elif 120.0 < theta <= 180.0:
        score += 0.0 # Extreme right axis deviation (unusual)
    else:
        score -= 6.0 # Extreme axis (no man's land)
        
    # 3. Coherence with V6 (Lead I and V6 should have same polarity)
    sign_I = np.sign(I_r)
    sign_V6 = np.sign(V6_val)
    if sign_I == sign_V6:
        score += 3.0
    else:
        score -= 4.0
        
    return score, theta

def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV file not found at {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    # Clean records: remove rows with empty or unknown labels
    df_valid = df[df['Inversione'].isin(LABEL_MAPPING.keys())].copy()
    
    print(f"Validating labels for {len(df_valid)} records in thesis-sample.csv...")
    print("-" * 80)
    
    discrepancies = []
    validated_count = 0
    skipped_count = 0
    
    for idx, row in df_valid.iterrows():
        patient_id = row['Num']
        true_label_str = row['Inversione']
        true_class_idx = LABEL_MAPPING[true_label_str]
        
        edf_path = os.path.join(SMALL_DIR, f"record{patient_id}.edf")
        if not os.path.exists(edf_path):
            skipped_count += 1
            continue
            
        try:
            ecg_data = _parse_edf_file(edf_path)
            if not ecg_data or "signals" not in ecg_data:
                skipped_count += 1
                continue
                
            # Compute average QRS polarity vector
            qrs_vec = get_qrs_polarity_vector(ecg_data["signals"])
            if qrs_vec is None:
                skipped_count += 1
                continue
                
            # Score all 6 possible restorations
            scores = []
            axes = []
            for c in range(6):
                I_r, II_r, III_r, aVR_r, aVL_r, aVF_r = restore_leads(qrs_vec, c)
                score, theta = compute_plausibility_score(I_r, II_r, III_r, aVR_r, aVL_r, aVF_r, qrs_vec['V6'])
                scores.append(score)
                axes.append(theta)
                
            best_class = np.argmax(scores)
            best_score = scores[best_class]
            
            # Check if best class matches the CSV label
            if best_class == true_class_idx:
                validated_count += 1
            else:
                margin = best_score - scores[true_class_idx]
                if margin > 2.0: # Significant difference
                    discrepancies.append({
                        'patient_id': patient_id,
                        'csv_label': CLASS_NAMES[true_class_idx],
                        'csv_score': scores[true_class_idx],
                        'csv_axis': axes[true_class_idx],
                        'calc_label': CLASS_NAMES[best_class],
                        'calc_score': best_score,
                        'calc_axis': axes[best_class],
                        'margin': margin
                    })
                    
        except Exception as e:
            skipped_count += 1
            
    print(f"Validation complete:")
    print(f"  - Validated & Correct labels: {validated_count}")
    print(f"  - Discrepancies flagged:      {len(discrepancies)}")
    print(f"  - Skipped (missing/noisy):    {skipped_count}")
    print("-" * 80)
    
    if len(discrepancies) > 0:
        print("Flagged Discrepancies Details:")
        print(f"{'Patient ID':<12} | {'CSV Label (Axis)':<22} | {'Calc Label (Axis)':<22} | {'Score Diff':<10}")
        print("-" * 80)
        for d in discrepancies:
            csv_info = f"{d['csv_label']} ({d['csv_axis']:.0f}°)"
            calc_info = f"{d['calc_label']} ({d['calc_axis']:.0f}°)"
            print(f"{d['patient_id']:<12} | {csv_info:<22} | {calc_info:<22} | {d['margin']:.2f}")
            
        # Save to csv file in project root for easy inspection
        out_csv = os.path.join(THESIS_DIR, "flagged_discrepancies.csv")
        pd.DataFrame(discrepancies).to_csv(out_csv, index=False)
        print(f"\nSaved discrepancies details to: flagged_discrepancies.csv in the project root.")
    else:
        print("No significant discrepancies found. All labels are consistent with ECG math and physiology!")

if __name__ == '__main__':
    main()
