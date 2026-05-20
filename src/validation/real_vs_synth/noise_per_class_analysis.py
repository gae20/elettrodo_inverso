import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Aggiungi src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data.data_pipeline import get_ecg, all_leads_preprocessing
from utils.config import ALL_LEADS, LIMB_LEADS, LABEL_MAP_CLEAN

# Configura i percorsi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "..", "datasets", "dataset", "thesis-sample.csv"))

def estimate_noise(sigs_array):
    """
    Stima il rumore come la Median Absolute Deviation (MAD) 
    della derivata prima (che enfatizza le alte frequenze del rumore).
    """
    # Derivata prima
    diff = np.diff(sigs_array, axis=1)
    # MAD (Median Absolute Deviation)
    mad = np.median(np.abs(diff - np.median(diff, axis=1, keepdims=True)), axis=1)
    return np.mean(mad) # Media tra le 6 lead periferiche

def analyze_noise_distribution():
    print(f"\n--- Analisi Rumore per Classe (Dataset Reale) ---")
    
    df = pd.read_csv(CSV_PATH)
    df_valido = df[df["Inversione"] != "?"].copy()
    df_valido["Inversione"] = df_valido["Inversione"].apply(lambda x: LABEL_MAP_CLEAN.get(x, x))
    
    results = []
    
    classes = df_valido["Inversione"].unique()
    
    for cls in classes:
        subset = df_valido[df_valido["Inversione"] == cls]
        print(f"Analisi classe {cls} ({len(subset)} record)...")
        
        noise_levels = []
        # Limitiamo a 50 campioni per classe per velocità
        for ecg_id in tqdm(subset["Num"].iloc[:50]):
            data = get_ecg(ecg_id)
            if not data: continue
            
            # Preprocessing (Filtro 0.5-120Hz)
            sigs = all_leads_preprocessing(data["signals"])
            x = np.array([sigs[l] for l in LIMB_LEADS], dtype=np.float32)
            
            # Stima rumore residuo dopo filtraggio
            noise = estimate_noise(x)
            noise_levels.append(noise)
            
        if noise_levels:
            results.append({
                "Classe": cls,
                "Rumore Medio (MAD)": np.mean(noise_levels),
                "Std Rumore": np.std(noise_levels),
                "Max Rumore": np.max(noise_levels)
            })
            
    # Crea tabella riassuntiva
    res_df = pd.DataFrame(results).sort_values("Rumore Medio (MAD)", ascending=False)
    print("\n" + "="*50)
    print("DISTRIBUZIONE RUMORE CLINICO")
    print("="*50)
    print(res_df.to_string(index=False))
    print("="*50)
    
    # Conclusioni
    top_class = res_df.iloc[0]["Classe"]
    bottom_class = res_df.iloc[-1]["Classe"]
    ratio = res_df.iloc[0]["Rumore Medio (MAD)"] / res_df.iloc[-1]["Rumore Medio (MAD)"]
    print(f"\nLa classe più rumorosa è: {top_class}")
    print(f"Rapporto di rumore {top_class} vs {bottom_class}: {ratio:.2f}x")

if __name__ == "__main__":
    analyze_noise_distribution()
