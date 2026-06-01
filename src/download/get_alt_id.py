import os
import sys
import sqlite3
import json
import numpy as np

# Aggiungiamo src al path per poter importare
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data.build_unlabelled_dataset import get_clean_ecg_ids

db_path = 'datasets/dataset/records.db'
out_path = 'datasets/dataset/required_edfs.txt'

with open(out_path, 'r') as f:
    existing = set(f.read().splitlines())

clean_ids = get_clean_ecg_ids(db_path, max_ecgs=11000)

new_ids = [cid for cid in clean_ids if str(cid) not in existing]

print(f"ID alternativo: {new_ids[0] if new_ids else 'Nessuno'}")
