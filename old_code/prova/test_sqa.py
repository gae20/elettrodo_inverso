import os
import sys

# Import functions from dataset_extension_2.py
from dataset_extension_2 import (
    df_valido, get_ecg, all_leads_preprocessing, check_ecg_quality, 
    QUALITY_CFG, print_quality_summary, ALL_LEADS
)

def test_sqa_on_flatline_classes():
    # Trova alcuni ID con etichetta RL-RA (6) o RL-LA (7)
    df_cand = df_valido[df_valido['Inversione'].isin(['RL-RA', 'RL-LA'])]
    
    if len(df_cand) == 0:
        print("Nessun ECG trovato con etichetta RL-RA o RL-LA nel dataframe filtrato.")
        return
    
    # Prendi i primi 5 ECG
    sample_ids = df_cand.index[:5]
    print(f"Sto testando SQA su {len(sample_ids)} ECG di classi 6/7 che dovrebbero produrre flatline.")
    
    for ecg_id in sample_ids:
        label = df_valido.loc[ecg_id, 'Inversione']
        print(f"\n--- Test ID: {ecg_id} | Label: {label} ---")
        
        ecg_data = get_ecg(ecg_id)
        if ecg_data is None:
            print("Errore nel caricamento del segnale.")
            continue
            
        raw_data = ecg_data["signals"]
        sigs = all_leads_preprocessing(raw_data)
        
        # Converte in array
        import numpy as np
        sigs_array = np.array([sigs[l] for l in ALL_LEADS], dtype=np.float32)
        
        # Esegui SQA
        quality_result = check_ecg_quality(sigs_array, min_valid_leads=QUALITY_CFG["min_valid_leads"])
        
        print_quality_summary(f"ECG ID: {ecg_id} ({label})", quality_result, ALL_LEADS)
        
        if not quality_result["global_valid"]:
            print(f"✅ SUCCESSO: L'ECG {ecg_id} è stato scartato correttamente dalla SQA.")
            # Controlla se il motivo è flatline
            flatline_leads = [res["lead_idx"] for res in quality_result["lead_results"] if res["global_result"]["checks"].get("flatline", False)]
            if flatline_leads:
                lead_names = [ALL_LEADS[i] for i in flatline_leads]
                print(f"   Flatline rilevata nelle derivazioni: {lead_names}")
        else:
            print(f"❌ FALLIMENTO: L'ECG {ecg_id} è stato considerato VALIDO dalla SQA nonostante la label {label}!")

if __name__ == "__main__":
    test_sqa_on_flatline_classes()
