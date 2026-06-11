import os
import sys
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, classification_report

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.ldensenet import build_model
from utils.config import SAMPLES_PER_WINDOW

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def cal_metrics(confusion_matrix):
    n_classes = confusion_matrix.shape[0]
    metrics_result = []
    for i in range(n_classes):
        ALL = np.sum(confusion_matrix)
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP
        TN = ALL - TP - FP - FN
        
        # Aggiunto calcolo dell'accuratezza per singola classe
        accuracy = (TP + TN) / ALL if ALL > 0 else 0
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0
        recall = TP/(TP+FN) if (TP+FN) > 0 else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
        specificity = TN/(TN+FP) if (TN+FP) > 0 else 0
        metrics_result.append([accuracy, precision, recall, specificity, f1])
    return metrics_result

def cal_binary_metrics(y_true, y_pred):
    """
    Mappa le 6 classi in 2 macro-classi:
    0 = Normale
    1 = Anormale (classi da 1 a 5, ovvero errori di posizionamento)
    """
    y_true_bin = np.where(y_true == 0, 0, 1)
    y_pred_bin = np.where(y_pred == 0, 0, 1)
    
    C_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    TN, FP, FN, TP = C_bin.ravel()
    
    acc = (TP + TN) / (TN + FP + FN + TP) if (TN + FP + FN + TP) > 0 else 0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0 # Sensibilità
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0 # Specificità
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    return C_bin, acc, prec, rec, spec, f1

def print_binary_report(y_true, y_pred, title=""):
    C_bin, acc, prec, rec, spec, f1 = cal_binary_metrics(y_true, y_pred)
    print(f"\n[ANALISI BINARIA: NORMALE vs ANORMALE - {title}]")
    print(f"Matrice di Confusione Binaria:\n {C_bin}")
    print(f" -> Veri Normali (TN): {C_bin[0,0]:>6} | Falsi Anormali (FP): {C_bin[0,1]:>6}")
    print(f" -> Falsi Normali (FN): {C_bin[1,0]:>6} | Veri Anormali (TP): {C_bin[1,1]:>6}")
    print("-" * 55)
    print(f"Accuratezza Binaria:     {acc:.4f} (Capacità di distinguere la macro-categoria)")
    print(f"Specificità (Sani):      {spec:.4f} (Capacità di riconoscere i VERI NORMALI)")
    print(f"Sensibilità/Rec (Malati):{rec:.4f} (Capacità di intercettare le ANOMALIE)")
    print(f"Precisione Binaria:      {prec:.4f}")
    print(f"F1-Score Binario:        {f1:.4f}")
    print("-" * 55)

def evaluater(x_test, y_test, model, path):
    print(f"\n> Generazione report e matrice per: {os.path.basename(path)}")
    y_pred = model.predict(x_test, batch_size=32)
    num_classes = y_pred.shape[-1]
    
    y_pred_idx = np.argmax(y_pred, axis=1)
    acc = len(np.where(y_pred_idx==y_test)[0])/y_pred_idx.shape[0]
    
    C = confusion_matrix(y_test, y_pred_idx, labels=range(num_classes))
    
    plt.figure(figsize=(7,7), dpi=100)
    plt.matshow(C, cmap=plt.cm.Reds, fignum=1) 
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(str(C[j, i]), xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close()
    return np.array(cal_metrics(C)), acc

def evaluater_pro(x_test, y_test_idx, model):
    y_probs = model.predict(x_test, batch_size=32)
    num_classes = y_probs.shape[-1]
    y_test_one_hot = to_categorical(y_test_idx, num_classes=num_classes)
    auroc = roc_auc_score(y_test_one_hot, y_probs, multi_class='ovr', average='macro')
    auprc = average_precision_score(y_test_one_hot, y_probs, average='macro')
    return auroc, auprc

if __name__ == '__main__':
    # --- CONFIGURAZIONE ARGPARSE ---
    parser = argparse.ArgumentParser(description="Valutazione del modello su finestre o pazienti.")
    parser.add_argument("--mode", type=str, default="window", choices=["window", "patient_simulated"],
                        help="Modalità: 'window' (originale) o 'patient_simulated' (majority voting).")
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU disponibili: {len(gpus)}")
        except RuntimeError as e:
            print(f"⚠️ Errore configurazione GPU: {e}")

    CLASS_NAMES = ['normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']
    input_shape = (SAMPLES_PER_WINDOW, 6)
    output_dims = 6 
    
    save_path = '../training/unlabelled_z_median_weights_and_cm/PROVA_best_model_targeted_noise_limbs.weights.h5'
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # =========================================================================
    # MODALITÀ 1: WINDOW
    # =========================================================================
    if args.mode == "window":
        dataset_path_test = "../../datasets/unlabelled_simulated_final/unlabelled_z_median_limbs_test_backup.h5"
        
        print(f"Caricamento dataset di test REALI da: {dataset_path_test}")
        with h5py.File(dataset_path_test, 'r') as f:
            y_all = f['Y'][:]
            valid_idx = np.where(y_all < 6)[0]
            x_test_raw = f['X'][valid_idx, :6, :]
            x_test = np.transpose(x_test_raw, (0, 2, 1))
            y_test = y_all[valid_idx]

        print("Costruzione del modello...")
        model = build_model(input_shape, output_dims)
        
        print(f"Caricamento dei pesi da {save_path}...")
        model.load_weights(os.path.join(base_dir, save_path))

        print("\n" + "="*50)
        print("VALUTAZIONE TEST SET (LIVELLO FINESTRA)")
        print("="*50)
        
        pic_path_test = os.path.join(base_dir, "../training/unlabelled_z_median_weights_and_cm/PROVA_test_cm.png")
        
        y_probs = model.predict(x_test, batch_size=32)
        y_pred = np.argmax(y_probs, axis=1)
        
        C = confusion_matrix(y_test, y_pred)
        
        os.makedirs(os.path.dirname(pic_path_test), exist_ok=True)
        plt.figure(figsize=(8,8), dpi=100)
        plt.matshow(C, cmap=plt.cm.Reds, fignum=1) 
        for i in range(len(C)):
            for j in range(len(C)):
                plt.annotate(str(C[j, i]), xy=(i, j), horizontalalignment='center', verticalalignment='center')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
        plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        plt.savefig(pic_path_test, bbox_inches='tight', pad_inches=0.1, dpi=200)
        plt.close()
        print(f"✅ Matrice di confusione salvata in: {pic_path_test}")

        acc = np.sum(np.diag(C)) / np.sum(C)
        auroc, auprc = evaluater_pro(x_test, y_test, model)
        
        # Nuova stampa dei risultati in formato tabellare personalizzato
        metrics = cal_metrics(C)
        print(f"\n[RISULTATI TEST SET - FINESTRE]")
        print(f"Accuratezza Totale Multiclasse: {acc:.4f}")
        print(f"AUROC (Macro):       {auroc:.4f}")
        print(f"AuPRC (Macro):       {auprc:.4f}")
        print("-" * 75)
        print(f"{'Classe':<12} | {'Acc.':<8} | {'Prec.':<8} | {'Rec.':<8} | {'Spec.':<8} | {'F1':<8}")
        print("-" * 75)
        for i in range(len(metrics)):
            a, p, r, s, f1 = metrics[i]
            print(f"{CLASS_NAMES[i]:<12} | {a:<8.4f} | {p:<8.4f} | {r:<8.4f} | {s:<8.4f} | {f1:<8.4f}")
        print("-" * 75)

        # Analisi binaria (Normale vs Anormale)
        print_binary_report(y_test, y_pred, title="LIVELLO FINESTRA")

        print("\n[ANALISI ERRORI]")
        row_sums = C.sum(axis=1)
        for i in range(len(CLASS_NAMES)):
            true_name = CLASS_NAMES[i]
            tp = C[i, i]
            errors = [(CLASS_NAMES[j], C[i, j]) for j in range(len(CLASS_NAMES)) if j != i and C[i, j] > 0]
            errors.sort(key=lambda x: -x[1])
            if errors:
                err_str = ", ".join([f"{name}={val}" for name, val in errors])
                print(f"  {true_name:<12} (n={row_sums[i]:>4}): TP={tp:>4} | confuso con → {err_str}")
            else:
                print(f"  {true_name:<12} (n={row_sums[i]:>4}): TP={tp:>4} | nessun errore ✅")


    # =========================================================================
    # MODALITÀ 2: PATIENT_SIMULATED (MAJORITY VOTING)
    # =========================================================================
    elif args.mode == "patient_simulated":
        dataset_path_test = "../../datasets/unlabelled_simulated_final/unlabelled_z_median_limbs_test_with_patients.h5"
        
        print(f"Caricamento dataset di test SIMULATI da: {dataset_path_test}")
        with h5py.File(dataset_path_test, 'r') as f:
            y_all = f['Y'][:]
            valid_idx = np.where(y_all < 6)[0]
            x_test_raw = f['X'][valid_idx, :6, :]
            x_test = np.transpose(x_test_raw, (0, 2, 1))
            y_test = y_all[valid_idx]
            patient_ids = f['patient_ids'][valid_idx]

        print("Costruzione del modello...")
        model = build_model(input_shape, output_dims)
        
        print(f"Caricamento dei pesi da {save_path}...")
        model.load_weights(os.path.join(base_dir, save_path))

        print("\n" + "="*50)
        print("VALUTAZIONE TEST SET (A LIVELLO PAZIENTE - VOTING)")
        print("="*50)

        # 1. Predice su tutte le finestre
        y_probs_windows = model.predict(x_test, batch_size=32)

        # 2. Raggruppa per paziente
        unique_cases = {}
        for i in range(len(y_test)):
            pid = patient_ids[i]
            c_true = y_test[i]
            case_key = (pid, c_true)
            if case_key not in unique_cases:
                unique_cases[case_key] = []
            unique_cases[case_key].append(y_probs_windows[i])

        # 3. Applica il Majority Voting (media delle probabilità)
        y_true_cases = []
        y_probs_cases = []
        for (pid, c_true), probs_list in unique_cases.items():
            avg_prob = np.mean(probs_list, axis=0) 
            y_true_cases.append(c_true)
            y_probs_cases.append(avg_prob)

        y_true_cases = np.array(y_true_cases)
        y_probs_cases = np.array(y_probs_cases)
        y_pred_cases = np.argmax(y_probs_cases, axis=1)
        
        C = confusion_matrix(y_true_cases, y_pred_cases)
        
        # 4. Salva la matrice di confusione a livello paziente
        pic_path_test = os.path.join(base_dir, "../training/unlabelled_z_median_weights_and_cm/PROVA_test_cm_patient_simulated.png")
        os.makedirs(os.path.dirname(pic_path_test), exist_ok=True)
        
        plt.figure(figsize=(8,8), dpi=100)
        plt.matshow(C, cmap=plt.cm.Reds, fignum=1) 
        for i in range(len(C)):
            for j in range(len(C)):
                plt.annotate(str(C[j, i]), xy=(i, j), horizontalalignment='center', verticalalignment='center')
        plt.ylabel('True label (Patient)')
        plt.xlabel('Predicted label (Patient)')
        plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
        plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        plt.savefig(pic_path_test, bbox_inches='tight', pad_inches=0.1, dpi=200)
        plt.close()
        print(f"✅ Matrice di confusione (Patient) salvata in: {pic_path_test}")

        # 5. Calcolo Metriche Tabellari
        acc = np.sum(np.diag(C)) / np.sum(C)
        y_test_one_hot = to_categorical(y_true_cases, num_classes=6)
        auroc = roc_auc_score(y_test_one_hot, y_probs_cases, multi_class='ovr', average='macro')
        auprc = average_precision_score(y_test_one_hot, y_probs_cases, average='macro')
        
        metrics = cal_metrics(C)
        print(f"\n[RISULTATI TEST SET - PAZIENTE]")
        print(f"Accuratezza Totale Multiclasse: {acc:.4f}")
        print(f"AUROC (Macro):       {auroc:.4f}")
        print(f"AuPRC (Macro):       {auprc:.4f}")
        print("-" * 75)
        print(f"{'Classe':<12} | {'Acc.':<8} | {'Prec.':<8} | {'Rec.':<8} | {'Spec.':<8} | {'F1':<8}")
        print("-" * 75)
        for i in range(len(metrics)):
            a, p, r, s, f1 = metrics[i]
            print(f"{CLASS_NAMES[i]:<12} | {a:<8.4f} | {p:<8.4f} | {r:<8.4f} | {s:<8.4f} | {f1:<8.4f}")
        print("-" * 75)

        # Analisi binaria (Normale vs Anormale)
        print_binary_report(y_true_cases, y_pred_cases, title="LIVELLO PAZIENTE (VOTING)")

        print("\n[ANALISI ERRORI PAZIENTI]")
        row_sums = C.sum(axis=1)
        for i in range(len(CLASS_NAMES)):
            true_name = CLASS_NAMES[i]
            tp = C[i, i]
            errors = [(CLASS_NAMES[j], C[i, j]) for j in range(len(CLASS_NAMES)) if j != i and C[i, j] > 0]
            errors.sort(key=lambda x: -x[1])
            if errors:
                err_str = ", ".join([f"{name}={val}" for name, val in errors])
                print(f"  {true_name:<12} (n={row_sums[i]:>4}): TP={tp:>4} | confuso con → {err_str}")
            else:
                print(f"  {true_name:<12} (n={row_sums[i]:>4}): TP={tp:>4} | nessun errore ✅")