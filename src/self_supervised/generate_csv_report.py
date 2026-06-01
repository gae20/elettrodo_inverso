import os
import json
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
JSON_PATH = os.path.join(RESULTS_DIR, 'pseudolabels.json')
CSV_PATH = os.path.join(RESULTS_DIR, 'pseudolabels_report.csv')

if not os.path.exists(JSON_PATH):
    print(f"Errore: {JSON_PATH} non trovato. Esegui prima lo step 2.")
    exit(1)

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Scrittura del file CSV separato da punto e virgola per Excel
with open(CSV_PATH, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(['ID', 'Classe Predetta', 'Confidenza', 'Finestre Valide', 'Referto Clinico'])
    for entry in data:
        writer.writerow([
            entry['id'],
            entry['class_name'],
            f"{entry['confidence'] * 100:.2f}%",
            entry['n_windows'],
            entry['text']
        ])

print(f"OK! Report CSV generato con successo in:\n  {CSV_PATH}")
