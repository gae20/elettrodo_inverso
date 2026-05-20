"""
Diagnosi: perché 274 ECG finiscono in 'no finestre'?
"""
import os, sys, json, zipfile, copy, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
THESIS_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))
DATASETS_DIR = os.path.abspath(os.path.join(THESIS_DIR, '..', 'datasets'))
sys.path.insert(0, SRC_DIR)

from data.data_pipeline import read_edf_data, all_leads_preprocessing, check_ecg_quality, check_window_quality
from utils.config import SAMPLES_PER_WINDOW, FS_NEW, ALL_LEADS, QUALITY_CFG, STRIDE_SAMPLES
from utils.sqa_real_config import QUALITY_CFG_REAL

LIMB_INDICES   = list(range(6))

# Ora usiamo QUALITY_CFG_REAL (quello dei reali) per il self-supervised (stride 0.5s)
QUALITY_CFG_STD = QUALITY_CFG_REAL

# Usa la config originale (stride 0.5s)
QUALITY_CFG_ORIG = QUALITY_CFG

SSL_ZIP = os.path.join(DATASETS_DIR, 'dataset_ssl.zip')
CANDIDATES = os.path.join(SCRIPT_DIR, 'results', 'candidate_ids.json')

with open(CANDIDATES) as f:
    candidates = json.load(f)

def robust_scale_ecg(x, reference_leads=None, eps=1e-8):
    x = x.astype(np.float32)
    medians = np.median(x, axis=1, keepdims=True)
    ref = x[reference_leads, :] if reference_leads is not None else x
    q75, q25 = np.percentile(ref, [75, 25])
    scale = max((q75 - q25) / 1.34896, eps)
    return (x - medians) / scale

results = {'sqa_fail':0, 'no_windows_std':0, 'no_windows_orig':0, 'ok':0, 'ok_orig_only':0}

for cand in candidates[:50]:  # Test su primi 50
    ecg_id = cand['id']
    zip_path = cand['zip_path']
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            edf_bytes = z.read(f"{ecg_id}.edf")
        ecg_data = read_edf_data(edf_bytes)
        sigs = all_leads_preprocessing(ecg_data["signals"])
        sigs_arr = np.array([sigs[l] for l in ALL_LEADS], dtype=np.float32)

        # Test con config STANDARD
        q_std = check_ecg_quality(sigs_arr, cfg=QUALITY_CFG_STD, lead_indices=LIMB_INDICES)
        if not q_std['global_valid']:
            results['sqa_fail'] += 1
            continue

        # Finestre con config STANDARD
        sigs_norm = robust_scale_ecg(sigs_arr, reference_leads=LIMB_INDICES)
        fs = QUALITY_CFG_STD["fs"]
        win_size = int(QUALITY_CFG_STD["win_sec"] * fs)
        stride = int(QUALITY_CFG_STD["stride_sec"] * fs)
        n_samples = sigs_norm.shape[1]
        starts = list(range(0, n_samples - win_size + 1, stride))
        
        good_windows_std = 0
        good_windows_orig = 0
        for s in starts:
            for li in LIMB_INDICES:
                seg_raw = sigs_arr[li, s:s+win_size]
                r_std  = check_window_quality(seg_raw, cfg=QUALITY_CFG_STD, lead_idx=li)
                r_orig = check_window_quality(seg_raw, cfg=QUALITY_CFG_ORIG, lead_idx=li)
            # Count per window (simplified: check all 6 leads)
            valid_std  = sum(check_window_quality(sigs_arr[li, s:s+win_size], cfg=QUALITY_CFG_STD, lead_idx=li)["valid"] for li in LIMB_INDICES)
            valid_orig = sum(check_window_quality(sigs_arr[li, s:s+win_size], cfg=QUALITY_CFG_ORIG, lead_idx=li)["valid"] for li in LIMB_INDICES)
            if valid_std >= 5: good_windows_std += 1
            if valid_orig >= 5: good_windows_orig += 1

        if good_windows_std == 0:
            results['no_windows_std'] += 1
            if good_windows_orig > 0:
                results['ok_orig_only'] += 1
                # Debug: first failure
                if results['ok_orig_only'] == 1:
                    print(f"\n--- ECG {ecg_id} fails STD but passes ORIG ---")
                    for s in starts[:3]:
                        for li in LIMB_INDICES:
                            seg_raw = sigs_arr[li, s:s+win_size]
                            r = check_window_quality(seg_raw, cfg=QUALITY_CFG_STD, lead_idx=li)
                            if not r['valid']:
                                print(f"  Window {s}, Lead {li}: {r['reason']}")
        else:
            results['ok'] += 1
            
        if good_windows_orig == 0:
            results['no_windows_orig'] += 1
    except Exception as e:
        results['sqa_fail'] += 1

print("\n=== DIAGNOSI (50 ECG) ===")
for k,v in results.items():
    print(f"  {k}: {v}")
