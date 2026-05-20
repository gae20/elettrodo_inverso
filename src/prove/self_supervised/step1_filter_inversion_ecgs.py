"""
step1_filter_inversion_ecgs.py

Filtra il database clinico per identificare ECG candidati al pseudo-labelling:
- status = 'rejected'
- campo 'text' contiene keyword di inversione degli elettrodi periferici (limb leads)
- ID presente in almeno uno dei 5 ZIP del DATASET_complete

Output: results/candidate_ids.json
"""

import os
import sys
import re
import json
import sqlite3
import zipfile
from tqdm import tqdm

# --- Percorsi ---
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SRC_DIR      = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
THESIS_DIR   = os.path.abspath(os.path.join(SRC_DIR, '..'))
DATASETS_DIR = os.path.abspath(os.path.join(THESIS_DIR, '..', 'datasets'))

DB_PATH      = os.path.join(SRC_DIR, 'data', 'records_complete.db')
# ZIP scaricato per il self-supervised (contiene i ECG candidati)
SSL_ZIP_PATH = os.path.join(DATASETS_DIR, 'dataset_ssl.zip')
RESULTS_DIR  = os.path.join(SCRIPT_DIR, 'results')

# --- Keyword per inversioni periferiche (limb leads) ---
# Questi pattern identificano i testi in cui il medico segnala un problema
# agli elettrodi degli ARTI (I, II, III, aVR, aVL, aVF), NON alle precordiali.
LIMB_INVERSION_KEYWORDS = [
    'inversione periferiche',
    'inversione periferica',
    'inversione periferic',
    'elettrodi periferici',
    'elettrodi periferic',
    'scambio periferici',
    'inversione arti',
    'elettrodi degli arti',
    'elettrodi arti',
    'derivazioni periferiche',
    'periferic',
    'elettrodi invertiti',
    'inversione elettrodi',
    'malposizionamento elettrodi',
    'mal posizionamento elettrodi',
    'errato posizionamento elettrodi',
    'errata posizione elettrodi',
    'verosimile inversione',
    'possibile inversione',
    'possibile mal',
    'probabile mal',
    'scambio di elettrodi',
    'scambio elettrodi',
]

# Pattern regex per lead precordiali (V1-V6) — usato per escludere i record
# che menzionano SOLO precordiali senza alcun riferimento alle periferiche.
PRECORDIAL_ONLY_PATTERN = re.compile(r'\bv[1-6]\b', re.IGNORECASE)


def decode_text(raw):
    """Decodifica il campo text che può essere bytes o str."""
    if raw is None:
        return ''
    if isinstance(raw, bytes):
        return raw.decode('utf-8', errors='replace')
    return str(raw)


def is_limb_inversion_candidate(text_str: str) -> bool:
    """
    Restituisce True se il testo suggerisce un'inversione degli elettrodi periferici.

    Logica:
    1. Il testo deve contenere almeno una keyword di inversione periferica.
    2. Se il testo menziona lead precordiali (v1-v6) MA NON contiene keyword
       esplicitamente periferiche, viene escluso (es. "posizione V1-V3 da verificare").
    """
    t = text_str.lower().strip()
    if not t:
        return False

    has_limb_kw = any(kw in t for kw in LIMB_INVERSION_KEYWORDS)
    if not has_limb_kw:
        return False

    # Se ha keyword limb ma menziona anche precordiali, controlliamo
    # se la keyword limb è sufficientemente specifica da non essere ambigua.
    has_precordial = bool(PRECORDIAL_ONLY_PATTERN.search(t))
    if has_precordial:
        # Escludi solo se il testo non ha keyword esplicitamente periferiche
        # (es. "periferic", "arti", "periferiche") — in quel caso è ambiguo ma lo teniamo.
        explicit_limb = any(kw in t for kw in [
            'periferic', 'arti', 'periferiche', 'periferica',
        ])
        if not explicit_limb:
            return False  # Solo precordiali menzionate → escludi

    return True


def build_zip_index(zip_path: str) -> dict:
    """Indicizza tutti gli EDF presenti nel file ZIP specificato."""
    id_to_zip = {}
    print(f"Indicizzazione ZIP: {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for name in z.namelist():
                if name.endswith('.edf'):
                    ecg_id = name.replace('.edf', '').strip()
                    id_to_zip[ecg_id] = zip_path
    except Exception as e:
        print(f"  [ERRORE] Impossibile aprire {zip_path}: {e}")
    print(f"  -> {len(id_to_zip):,} EDF indicizzati.")
    return id_to_zip


def filter_candidates(db_path: str) -> list:
    """
    Estrae dal DB tutti i record rejected con testo di inversione periferica.
    Restituisce lista di dict {id, text}.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, text FROM records WHERE status='rejected'")
    rows = cursor.fetchall()
    conn.close()

    candidates = []
    n_rejected_total = len(rows)
    for row in rows:
        ecg_id = row[0]
        text_str = decode_text(row[1])
        if is_limb_inversion_candidate(text_str):
            candidates.append({
                'id': str(ecg_id),
                'text': text_str.strip(),
            })

    print(f"\nRecord rejected totali nel DB: {n_rejected_total:,}")
    print(f"Candidati con keyword limb inversion: {len(candidates)}")
    return candidates


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("STEP 1 — Filtro DB per ECG con inversione periferica")
    print("=" * 60)

    # 1. Filtra il DB
    print("\n[1/3] Interrogazione del database...")
    candidates = filter_candidates(DB_PATH)

    # 2. Costruisci il ZIP index
    print("\n[2/3] Indicizzazione del file ZIP...")
    id_to_zip = build_zip_index(SSL_ZIP_PATH)

    # 3. Filtra solo i candidati effettivamente presenti negli ZIP
    print("\n[3/3] Verifica presenza degli EDF negli ZIP...")
    final_candidates = []
    missing = 0
    for c in tqdm(candidates, desc="Verifica EDF"):
        if str(c['id']) in id_to_zip:
            c['zip_path'] = id_to_zip[str(c['id'])]
            final_candidates.append(c)
        else:
            missing += 1

    # 4. Salva il risultato
    out_path = os.path.join(RESULTS_DIR, 'candidate_ids.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_candidates, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Candidati trovati nel DB:           {len(candidates)}")
    print(f"Candidati presenti negli ZIP:       {len(final_candidates)}")
    print(f"Candidati assenti negli ZIP:        {missing}")
    print(f"Output salvato in: {out_path}")
    print("=" * 60)
