import os
import sys
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.ilc import build_model  # Rete ILC invece di LDenseNet
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
        
        # Calcolo dell'accuratezza della singola classe
        accuracy = (TP + TN) / ALL if ALL > 0 else 0
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0
        recall = TP/(TP+FN) if (TP+FN) > 0 else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
        specificity = TN/(TN+FP) if (TN+FP) > 0 else 0
        
        # Aggiunta dell'accuratezza all'output
        metrics_result.append([accuracy, precision, recall, specificity, f1])
    return metrics_result

def cal_binary_metrics(y_true, y_pred):
    """
    Mappa le 16 classi in 2 macro-classi:
    0 = Normale
    1 = Anormale (classi da 1 a 15)
    """
    y_true_bin = np.where(y_true == 0, 0, 1)
    y_pred_bin = np.where(y_pred == 0, 0, 1)
    
    C_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    TN, FP, FN, TP = C_bin.ravel()
    
    acc = (TP + TN) / (TN + FP + FN + TP) if (TN + FP + FN + TP) > 0 else 0
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0 # Sensibilità sul malato
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0 # Specificità sul sano
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    return C_bin, acc, prec, rec, spec, f1

def print_binary_report(y_true, y_pred, title=""):
    C_bin, acc, prec, rec, spec, f1 = cal_binary_metrics(y_true, y_pred)
    print(f"\n[ANALISI BINARIA: NORMALE vs ANORMALE - {title}]")
    print(f"Matrice di Confusione Binaria:\n {C_bin}")
    print(f" -> Veri Normali (TN): {C_bin[0,0]} | Falsi Anormali (FP): {C_bin[0,1]}")
    print(f" -> Falsi Normali (FN): {C_bin[1,0]} | Veri Anormali (TP): {C_bin[1,1]}")
    print("-" * 50)
    print(f"Accuratezza Binaria:     {acc:.4f} (Capacità complessiva di azzeccare la macro-categoria)")
    print(f"Specificità (Sani):      {spec:.4f} (Capacità di riconoscere i VERI NORMALI)")
    print(f"Sensibilità/Rec (Malati):{rec:.4f} (Capacità di intercettare le ANOMALIE)")
    print(f"Precisione Binaria:      {prec:.4f}")
    print(f"F1-Score Binario:        {f1:.4f}")
    print("-" * 50)

def evaluater(x_test, y_test, model, path):
    print(f"\n> Generazione report e matrice per: {os.path.basename(path)}")
    y_pred = model.predict(x_test, batch_size=32)
    num_classes = y_pred.shape[-1]
    
    y_pred_idx = np.argmax(y_pred, axis=1)
    acc = len(np.where(y_pred_idx==y_test)[0])/y_pred_idx.shape[0]
    C = confusion_matrix(y_test, y_pred_idx, labels=range(num_classes))
    
    plt.figure(figsize=(10,10), dpi=100) 
    plt.matshow(C, cmap=plt.cm.Reds, fignum=1) 
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(str(C[j, i]), xy=(i, j), horizontalalignment='center', verticalalignment='center', fontsize=8)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1, dpi=200)
    plt.close()
    
    return np.array(cal_metrics(C)), acc, y_pred_idx

def evaluater_pro(x_test, y_test_idx, model):
    y_probs = model.predict(x_test, batch_size=32)
    num_classes = y_probs.shape[-1]
    y_test_one_hot = to_categorical(y_test_idx, num_classes=num_classes)
    auroc = roc_auc_score(y_test_one_hot, y_probs, multi_class='ovr', average='macro')
    auprc = average_precision_score(y_test_one_hot, y_probs, average='macro')
    return auroc, auprc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Valutazione del modello ILC su finestre o pazienti.")
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

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_shape = (SAMPLES_PER_WINDOW, 6)
    output_dims = 16 
    
    print("Costruzione del modello ILC...")
    model = build_model(input_shape, output_dims)
    
    weights_path = os.path.join(base_dir, '..', 'training', 'unlabelled_simulated_weights_and_cm', 'best_model_unlabelled_z_median_precordials.weights.h5')
    if os.path.exists(weights_path):
        print(f"Caricamento dei pesi da {weights_path}...")
        model.load_weights(weights_path)
    else:
        print(f"⚠️ ATTENZIONE: {weights_path} non trovato. Verrà valutato un modello non addestrato.")

    # =========================================================================
    # MODALITÀ 1: WINDOW
    # =========================================================================
    if args.mode == "window":
        dataset_path_test = os.path.join(base_dir, '..', '..', 'datasets', 'unlabelled_simulated_final', 'unlabelled_z_median_precordials_test.h5')
        dataset_path_val  = os.path.join(base_dir, '..', '..', 'datasets', 'unlabelled_simulated_final', 'unlabelled_z_median_precordials_val.h5')
        
        print("Caricamento dataset di test e validation PRECORDIALI...")
        with h5py.File(dataset_path_test, 'r') as f:
            x_test_raw = f['X'][:, 6:, :]
            x_test = np.transpose(x_test_raw, (0, 2, 1))
            y_test = f['Y'][:]
            
        with h5py.File(dataset_path_val, 'r') as f:
            x_val_raw = f['X'][:, 6:, :]
            x_val_eval = np.transpose(x_val_raw, (0, 2, 1))
            y_val_eval = f['Y'][:]

        print("\n" + "="*50)
        print("VALUTAZIONE TEST SET (FINESTRA)")
        print("="*50)
        cm_test_path = os.path.join(base_dir, '..', 'training', 'unlabelled_simulated_weights_and_cm', 'prova.png')
        metrics, acc, y_pred_test = evaluater(x_test, y_test, model, cm_test_path)
        auroc, auprc = evaluater_pro(x_test, y_test, model)
        
        print(f"\n[RISULTATI TEST SET]")
        print(f"Accuratezza Totale Multiclasse: {acc:.4f}")
        print(f"AUROC (Macro):       {auroc:.4f}")
        print(f"AuPRC (Macro):       {auprc:.4f}")
        print("-" * 72)
        print(f"{'Classe':<10} | {'Acc.':<8} | {'Prec.':<8} | {'Rec.':<8} | {'Spec.':<8} | {'F1':<8}")
        print("-" * 72)
        for i in range(len(metrics)):
            a, p, r, s, f1 = metrics[i]
            print(f"Classe {i:<3} | {a:<8.4f} | {p:<8.4f} | {r:<8.4f} | {s:<8.4f} | {f1:<8.4f}")
        print("-" * 72)

        # Report di distinzione Normale vs Anormale
        print_binary_report(y_test, y_pred_test, title="TEST SET FINESTRE")

        print("\n" + "="*50)
        print("VALUTAZIONE VALIDATION SET")
        print("="*50)
        cm_val_path = os.path.join(base_dir, '..', 'training', 'unlabelled_simulated_weights_and_cm', 'unlabelled_z_median_precordials_cm_val.png')
        metrics_val, acc_val, y_pred_val = evaluater(x_val_eval, y_val_eval, model, cm_val_path)
        auroc_val, auprc_val = evaluater_pro(x_val_eval, y_val_eval, model)
        
        print(f"\n[RISULTATI VALIDATION SET]")
        print(f"Accuratezza Totale Multiclasse: {acc_val:.4f}")
        print(f"AUROC (Macro):       {auroc_val:.4f}")
        print(f"AuPRC (Macro):       {auprc_val:.4f}")
        print("-" * 72)
        print(f"{'Classe':<10} | {'Acc.':<8} | {'Prec.':<8} | {'Rec.':<8} | {'Spec.':<8} | {'F1':<8}")
        print("-" * 72)
        for i in range(len(metrics_val)):
            a, p, r, s, f1 = metrics_val[i]
            print(f"Classe {i:<3} | {a:<8.4f} | {p:<8.4f} | {r:<8.4f} | {s:<8.4f} | {f1:<8.4f}")
        print("-" * 72)

        # Report di distinzione Normale vs Anormale
        print_binary_report(y_val_eval, y_pred_val, title="VALIDATION SET FINESTRE")

    # =========================================================================
    # MODALITÀ 2: PATIENT_SIMULATED (MAJORITY VOTING)
    # =========================================================================
    elif args.mode == "patient_simulated":
        dataset_path_test = os.path.join(base_dir, '..', '..', 'datasets', 'unlabelled_simulated_final', 'unlabelled_z_median_precordials_test_with_patients.h5')
        
        print(f"Caricamento dataset di test SIMULATI da: {dataset_path_test}")
        if not os.path.exists(dataset_path_test):
            print(f"Errore: File non trovato: {dataset_path_test}")
            sys.exit(1)

        with h5py.File(dataset_path_test, 'r') as f:
            x_test_raw = f['X'][:, 6:, :]
            x_test = np.transpose(x_test_raw, (0, 2, 1))
            y_test = f['Y'][:]
            patient_ids = f['patient_ids'][:]

        print("\n" + "="*50)
        print("VALUTAZIONE TEST SET (A LIVELLO PAZIENTE - VOTING)")
        print("="*50)

        y_probs_windows = model.predict(x_test, batch_size=32)

        unique_cases = {}
        for i in range(len(y_test)):
            pid = patient_ids[i]
            c_true = y_test[i]
            case_key = (pid, c_true)
            if case_key not in unique_cases:
                unique_cases[case_key] = []
            unique_cases[case_key].append(y_probs_windows[i])

        y_true_cases = []
        y_probs_cases = []
        for (pid, c_true), probs_list in unique_cases.items():
            avg_prob = np.mean(probs_list, axis=0) 
            y_true_cases.append(c_true)
            y_probs_cases.append(avg_prob)

        y_true_cases = np.array(y_true_cases)
        y_probs_cases = np.array(y_probs_cases)
        y_pred_cases = np.argmax(y_probs_cases, axis=1)

        C = confusion_matrix(y_true_cases, y_pred_cases, labels=range(output_dims))
        metrics = cal_metrics(C)
        acc = len(np.where(y_pred_cases==y_true_cases)[0])/len(y_true_cases)

        y_test_one_hot = to_categorical(y_true_cases, num_classes=output_dims)
        auroc = roc_auc_score(y_test_one_hot, y_probs_cases, multi_class='ovr', average='macro')
        auprc = average_precision_score(y_test_one_hot, y_probs_cases, average='macro')

        pic_path_test = os.path.join(base_dir, '..', 'training', 'unlabelled_simulated_weights_and_cm', 'prova_patient_simulated.png')
        plt.figure(figsize=(10,10), dpi=100)
        plt.matshow(C, cmap=plt.cm.Reds, fignum=1) 
        for i in range(len(C)):
            for j in range(len(C)):
                plt.annotate(str(C[j, i]), xy=(i, j), horizontalalignment='center', verticalalignment='center', fontsize=8)
        plt.ylabel('True label (Patient)')
        plt.xlabel('Predicted label (Patient)')
        plt.savefig(pic_path_test, bbox_inches='tight', pad_inches=0.1, dpi=200)
        plt.close()
        print(f"✅ Matrice di confusione (Patient) salvata in: {pic_path_test}")

        print(f"\n[RISULTATI TEST SET - PAZIENTE]")
        print(f"Accuratezza Totale Multiclasse: {acc:.4f}")
        print(f"AUROC (Macro):       {auroc:.4f}")
        print(f"AuPRC (Macro):       {auprc:.4f}")
        print("-" * 72)
        print(f"{'Classe':<10} | {'Acc.':<8} | {'Prec.':<8} | {'Rec.':<8} | {'Spec.':<8} | {'F1':<8}")
        print("-" * 72)
        for i in range(len(metrics)):
            a, p, r, s, f1 = metrics[i]
            print(f"Classe {i:<3} | {a:<8.4f} | {p:<8.4f} | {r:<8.4f} | {s:<8.4f} | {f1:<8.4f}")
        print("-" * 72)

        # Report di distinzione Normale vs Anormale a livello paziente
        print_binary_report(y_true_cases, y_pred_cases, title="LIVELLO PAZIENTE (VOTING)")