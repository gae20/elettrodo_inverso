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

print("Estrazione ID puliti in corso...")
clean_ids = get_clean_ecg_ids(db_path, max_ecgs=10000)

with open(out_path, 'w') as f:
    for cid in clean_ids:
        f.write(f"{cid}\n")

print(f"File salvato in: {os.path.abspath(out_path)}")
