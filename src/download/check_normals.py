import sqlite3
import json
import re

def check_electrode_placement():
    conn = sqlite3.connect('datasets/dataset/records.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, report, text FROM records')
    rows = cursor.fetchall()
    
    total = len(rows)
    suspected_inversions = 0
    pathological = 0
    clean = 0
    
    # keywords for electrode misplacement in Italian
    keywords = ['inversion', 'scambio', 'errat', 'inverit', 'invertit', 'periferic', 'braccia', 'elettrod']
    
    for r in rows:
        id_ = r[0]
        report_str = r[1]
        text_str = (r[2] or "").lower()
        
        # Check text for suspected inversions
        if any(kw in text_str for kw in keywords):
            suspected_inversions += 1
            # let's look closer, maybe just a sample
            continue
            
        try:
            data = json.loads(report_str)
        except Exception:
            continue
            
        codified = data.get('codified', [])
        codes = [c['value'] for c in codified if c.get('type') == 'code']
        
        # We consider it cleanly placed if no text suggests otherwise.
        clean += 1
        
    print(f"Total ECGs: {total}")
    print(f"Suspected misplacements (by text): {suspected_inversions}")
    print(f"Assumed properly placed: {clean}")

check_electrode_placement()
