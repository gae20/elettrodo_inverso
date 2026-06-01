import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data.data_pipeline import check_ecg_quality
from utils.config import QUALITY_CFG, FS_NEW

np.random.seed(42)
n_samples = 10 * FS_NEW
base_signal = np.random.normal(0, 5, size=(12, n_samples))
for i in range(10):
    idx = i * FS_NEW + int(0.2 * FS_NEW)
    base_signal[:, idx:idx+10] += 500
    base_signal[:, idx+10:idx+20] -= 200

res = check_ecg_quality(base_signal, cfg=QUALITY_CFG, lead_indices=list(range(6)))
for lead in res['lead_results']:
    print(f"Lead {lead['lead_idx']}: {lead['reason']}")
    print(f"  global: {lead.get('global_result', {}).get('reason', '')}")
    if 'summary' in lead:
        print(f"  summary: {lead['summary']}")
