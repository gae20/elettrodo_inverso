import os
import glob
import h5py
import numpy as np

# Mapping delle classi dalle configurazioni (copia semplice per script standalone)
LIMB_MAP = {
    0: 'normale',
    1: 'LA-RA', 
    2: 'RA-LL', 
    3: 'LA-LL', 
    4: 'ROT_ORARIA', 
    5: 'ROT_ANTIORARIA',
    6: 'RL-RA', 
    7: 'RL-LA'
}

PRECORDIAL_MAP = {
    0: 'normale',
    1: 'V1-V2', 2: 'V1-V3', 3: 'V1-V4', 4: 'V1-V5', 5: 'V1-V6',
    6: 'V2-V3', 7: 'V2-V4', 8: 'V2-V5', 9: 'V2-V6',
    10: 'V3-V4', 11: 'V3-V5', 12: 'V3-V6',
    13: 'V4-V5', 14: 'V4-V6',
    15: 'V5-V6'
}

def analyze_h5_datasets(dataset_dir):
    h5_files = glob.glob(os.path.join(dataset_dir, '*.h5'))
    if not h5_files:
        print(f"Nessun file .h5 trovato nella cartella {dataset_dir}")
        return

    print(f"=== ANALISI DATASET IN {os.path.abspath(dataset_dir)} ===\n")
    
    for h5_path in sorted(h5_files):
        filename = os.path.basename(h5_path)
        print(f"--- Dataset: {filename} ---")
        
        # Scegli la mappa corretta in base al nome
        is_limb = 'limbs' in filename.lower()
        is_prec = 'precordials' in filename.lower()
        
        label_map = LIMB_MAP if is_limb else (PRECORDIAL_MAP if is_prec else {})
        
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'Y' not in f:
                    print("  Dataset 'Y' non trovato.")
                    continue
                
                y_data = f['Y'][:]
                total_windows = len(y_data)
                
                # Conta occorrenze
                unique, counts = np.unique(y_data, return_counts=True)
                
                print(f"  Totale finestre: {total_windows:,}")
                print(f"  Distribuzione Classi:")
                
                for u, c in zip(unique, counts):
                    class_name = label_map.get(u, f"Classe_{u}")
                    pct = (c / total_windows) * 100
                    print(f"    - {class_name:<15}: {c:>8,} ({pct:>5.1f}%)")
                    
        except Exception as e:
            print(f"  Errore nella lettura di {filename}: {e}")
        
        print("\n")

if __name__ == "__main__":
    # Percorso cartella datasets relativa allo script (src/validation/..)
    dataset_dir = os.path.join(os.path.dirname(__file__), '../..', 'datasets')
    analyze_h5_datasets(dataset_dir)
