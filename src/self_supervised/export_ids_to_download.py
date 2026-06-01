"""
export_ids_to_download.py

Interroga il DB e restituisce tutti gli ID degli ECG candidati
(status='rejected' + keyword inversione periferica), indipendentemente
dal fatto che siano già presenti nei file ZIP scaricati.

Output: results/ids_to_download.txt  (un ID per riga)
"""

import os
import re
import sys
import sqlite3

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR  = os.path.join(SCRIPT_DIR, '..', '..')
DB_PATH     = os.path.join(THESIS_DIR, 'datasets', 'dataset', 'records.db')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
OUTPUT_TXT  = os.path.join(RESULTS_DIR, 'ids_to_download.txt')

# Stessa logica di step1
LIMB_INVERSION_KEYWORDS = [
    'inversione periferiche', 'inversione periferica',
    'inversione periferic', 'elettrodi periferici',
    'elettrodi periferic', 'scambio periferici',
    'inversione arti', 'elettrodi degli arti', 'elettrodi arti',
    'derivazioni periferiche', 'periferic',
    'elettrodi invertiti', 'inversione elettrodi',
    'malposizionamento elettrodi', 'mal posizionamento elettrodi',
    'errato posizionamento elettrodi', 'errata posizione elettrodi',
    'verosimile inversione', 'possibile inversione',
    'possibile mal', 'probabile mal',
    'scambio di elettrodi', 'scambio elettrodi',
]
PRECORDIAL_ONLY_PATTERN = re.compile(r'\bv[1-6]\b', re.IGNORECASE)

def decode_text(raw):
    if raw is None: return ''
    if isinstance(raw, bytes): return raw.decode('utf-8', errors='replace')
    return str(raw)

def is_limb_inversion_candidate(text_str):
    t = text_str.lower().strip()
    if not t: return False
    has_limb_kw = any(kw in t for kw in LIMB_INVERSION_KEYWORDS)
    if not has_limb_kw: return False
    has_precordial = bool(PRECORDIAL_ONLY_PATTERN.search(t))
    if has_precordial:
        explicit_limb = any(kw in t for kw in ['periferic', 'arti', 'periferiche', 'periferica'])
        if not explicit_limb:
            return False
    return True

os.makedirs(RESULTS_DIR, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT id, text FROM records WHERE status='rejected'")
rows = cursor.fetchall()
conn.close()

candidate_ids = []
for row in rows:
    ecg_id   = row[0]
    text_str = decode_text(row[1])
    if is_limb_inversion_candidate(text_str):
        candidate_ids.append(str(ecg_id))

with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(candidate_ids) + '\n')

print(f"Totale ECG candidati da scaricare: {len(candidate_ids)}")
print(f"File salvato in: {OUTPUT_TXT}")
print("\nPrimi 10 ID:")
for i in candidate_ids[:10]:
    print(f"  {i}")
