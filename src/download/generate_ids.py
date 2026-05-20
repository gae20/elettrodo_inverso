import sqlite3
import json
import numpy as np

# Dizionario per memorizzare se un ID è Holter
IS_HOLTER_DICT = {}

def get_clean_ecg_ids(db_path, max_ecgs=None):
    """
    Legge dal database locale i record ECG puliti, escludendo quelli con artefatti
    o con testo che suggerisce un'inversione fisiologica già presente.
    Popola anche IS_HOLTER_DICT.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, report, text FROM records WHERE status='reported'")
    rows = cursor.fetchall()
    conn.close()

    text_bad_keywords = ['inversion', 'scambio', 'errato', 'periferic', 'elettrod', 'sbagliat', 'artefatt', 'posizionament', 'braccia']
    rejection_codes = {'BTWG01', 'BTWG02', 'BTWG03', 'BTWG04', 'BTWG05', 'BTWC1109', 'BTWC1110'}
    
    clean_ids = []
    for r in rows:
        id_ = r[0]
        report_str = r[1]
        text_str = (r[2] or "").lower()
        if any(kw in text_str for kw in text_bad_keywords): continue
        try:
            data = json.loads(report_str)
            codified = data.get('codified', [])
            codes = [c['value'] for c in codified if c.get('type') == 'code']
            if any(c in rejection_codes for c in codes): continue
            is_holter = 'BTWSCQQ43' in codes
            IS_HOLTER_DICT[id_] = is_holter
            clean_ids.append(id_)
        except Exception: continue

    np.random.seed(42)
    np.random.shuffle(clean_ids)
    if max_ecgs and len(clean_ids) > max_ecgs: clean_ids = clean_ids[:max_ecgs]
    return clean_ids
