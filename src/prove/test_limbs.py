import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

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
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0
        recall = TP/(TP+FN) if (TP+FN) > 0 else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
        specificity = TN/(TN+FP) if (TN+FP) > 0 else 0
        metrics_result.append([precision, recall, specificity, f1])
    return metrics_result

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
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU disponibili: {len(gpus)}")
        except RuntimeError as e:
            print(f"⚠️ Errore configurazione GPU: {e}")

    CLASS_NAMES = ['normale', 'LA-RA', 'RA-LL', 'LA-LL', 'ROT_ORA', 'ROT_ANT']

    # ----------------------------------------------------
    # TOGGLE TEST SET:
    # Imposta a True per valutare sul test set reale purificato/corretto fisicamente (Gold Standard)
    # Imposta a False per valutare sul test set reale clinico originale (con rumore di etichette)
    USE_CLEAN_TEST_SET = False
    # ----------------------------------------------------

    base_dir = os.path.dirname(os.path.abspath(__file__))
    h5_filename = "labelled_z_median_limbs_test_validation_clean.h5" if USE_CLEAN_TEST_SET else "labelled_z_median_limbs_test_validation.h5"
    dataset_path_test = os.path.join(base_dir, "..", "..", "..", "datasets", h5_filename)
    
    print(f"Caricamento dataset di test REALI da: {dataset_path_test}")
    with h5py.File(dataset_path_test, 'r') as f:
        y_all = f['Y'][:]
        valid_idx = np.where(y_all < 6)[0]
        x_test_raw = f['X'][valid_idx, :6, :]
        x_test = np.transpose(x_test_raw, (0, 2, 1))
        y_test = y_all[valid_idx]

    input_shape = (SAMPLES_PER_WINDOW, 6)
    output_dims = 6 
    
    # Percorso pesi prodotti da train_limbs.py
    save_path = os.path.join(base_dir, "models", "best_model_final_noise_limbs.weights.h5")

    print("Costruzione del modello...")
    model = build_model(input_shape, output_dims)
    
    print(f"Caricamento dei pesi da {save_path}...")
    model.load_weights(save_path)

    print("\n" + "="*50)
    print("VALUTAZIONE TEST SET (DATI REALI)")
    print("="*50)
    
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    pic_path_test = os.path.join(base_dir, "logs", "best_model_cm_test.png")
    
    # Valutazione
    y_probs = model.predict(x_test, batch_size=32)
    y_pred = np.argmax(y_probs, axis=1)
    
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n[REPORT DI CLASSIFICAZIONE]")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    C = confusion_matrix(y_test, y_pred, labels=range(len(CLASS_NAMES)))
    acc = np.sum(np.diag(C)) / np.sum(C)
    auroc, auprc = evaluater_pro(x_test, y_test, model)
    
    # Salva matrice di confusione
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    im = ax.matshow(C, cmap=plt.cm.Reds)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(C)):
        for j in range(len(C)):
            ax.annotate(str(C[i, j]), xy=(j, i), horizontalalignment='center', verticalalignment='center', fontsize=12)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='left', fontsize=10)
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    plt.title(f'Confusion Matrix — Acc: {acc:.3f}', fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(pic_path_test, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"\n✅ Matrice di confusione salvata in: {pic_path_test}")
    
    print(f"\nAccuratezza Totale: {acc:.4f}")
    print(f"AUROC (Macro):     {auroc:.4f}")
    print(f"AuPRC (Macro):     {auprc:.4f}")

    # Analisi errori
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

    # ══════════════════════════════════════════════════════════════════════
    # VALUTAZIONE TEST SET SIMULATI
    # ══════════════════════════════════════════════════════════════════════
    dataset_path_synth = os.path.join(base_dir, "..", "..", "..", "datasets", "unlabelled_final_noise_limbs_test.h5")
    
    if os.path.exists(dataset_path_synth):
        print("\n\n" + "="*50)
        print("VALUTAZIONE TEST SET (DATI SIMULATI)")
        print("="*50)
        
        with h5py.File(dataset_path_synth, 'r') as f:
            y_synth_all = f['Y'][:]
            valid_synth = np.where(y_synth_all < 6)[0]
            # Bilanciamento: prendi al massimo N per classe
            n_per_class = min(np.bincount(y_synth_all[valid_synth].astype(int)).min(), 2000)
            balanced_idx = []
            for c in range(6):
                c_idx = valid_synth[y_synth_all[valid_synth] == c]
                np.random.seed(42)
                chosen = np.random.choice(c_idx, min(n_per_class, len(c_idx)), replace=False)
                balanced_idx.extend(chosen)
            balanced_idx = np.array(sorted(balanced_idx))
            x_synth_raw = f['X'][balanced_idx, :6, :]
            x_synth = np.transpose(x_synth_raw, (0, 2, 1))
            y_synth = y_synth_all[balanced_idx]
        
        print(f"Campioni per classe: {n_per_class}")
        
        y_synth_probs = model.predict(x_synth, batch_size=32)
        y_synth_pred = np.argmax(y_synth_probs, axis=1)
        
        print("\n[REPORT DI CLASSIFICAZIONE — SIMULATI]")
        print(classification_report(y_synth, y_synth_pred, target_names=CLASS_NAMES))
        
        C_synth = confusion_matrix(y_synth, y_synth_pred, labels=range(len(CLASS_NAMES)))
        acc_synth = np.sum(np.diag(C_synth)) / np.sum(C_synth)
        auroc_synth, auprc_synth = evaluater_pro(x_synth, y_synth, model)
        
        # Salva matrice di confusione simulati
        pic_path_synth = os.path.join(base_dir, "logs", "best_model_cm_synth_test.png")
        fig2, ax2 = plt.subplots(figsize=(8, 8), dpi=100)
        im2 = ax2.matshow(C_synth, cmap=plt.cm.Blues)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        for i in range(len(C_synth)):
            for j in range(len(C_synth)):
                ax2.annotate(str(C_synth[i, j]), xy=(j, i), horizontalalignment='center', verticalalignment='center', fontsize=12)
        ax2.set_xticks(range(len(CLASS_NAMES)))
        ax2.set_yticks(range(len(CLASS_NAMES)))
        ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha='left', fontsize=10)
        ax2.set_yticklabels(CLASS_NAMES, fontsize=10)
        ax2.set_ylabel('True label', fontsize=12)
        ax2.set_xlabel('Predicted label', fontsize=12)
        ax2.xaxis.set_label_position('bottom')
        ax2.xaxis.tick_bottom()
        plt.title(f'Confusion Matrix (Simulati) — Acc: {acc_synth:.3f}', fontsize=13, pad=15)
        plt.tight_layout()
        plt.savefig(pic_path_synth, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"\n✅ Matrice di confusione simulati salvata in: {pic_path_synth}")
        
        print(f"\nAccuratezza Simulati: {acc_synth:.4f}")
        print(f"AUROC (Macro):       {auroc_synth:.4f}")
        print(f"AuPRC (Macro):       {auprc_synth:.4f}")
        
        # Analisi errori simulati
        print("\n[ANALISI ERRORI — SIMULATI]")
        row_sums_s = C_synth.sum(axis=1)
        for i in range(len(CLASS_NAMES)):
            true_name = CLASS_NAMES[i]
            tp = C_synth[i, i]
            errors = [(CLASS_NAMES[j], C_synth[i, j]) for j in range(len(CLASS_NAMES)) if j != i and C_synth[i, j] > 0]
            errors.sort(key=lambda x: -x[1])
            if errors:
                err_str = ", ".join([f"{name}={val}" for name, val in errors])
                print(f"  {true_name:<12} (n={row_sums_s[i]:>4}): TP={tp:>4} | confuso con → {err_str}")
            else:
                print(f"  {true_name:<12} (n={row_sums_s[i]:>4}): TP={tp:>4} | nessun errore ✅")
        
        # Confronto diretto reali vs simulati
        print("\n" + "="*50)
        print("CONFRONTO REALI vs SIMULATI")
        print("="*50)
        print(f"  {'Metrica':<20} {'Reali':>10} {'Simulati':>10} {'Gap':>10}")
        print(f"  {'-'*50}")
        print(f"  {'Accuratezza':<20} {acc:>10.4f} {acc_synth:>10.4f} {acc_synth - acc:>+10.4f}")
        print(f"  {'AUROC':<20} {auroc:>10.4f} {auroc_synth:>10.4f} {auroc_synth - auroc:>+10.4f}")
        print(f"  {'AuPRC':<20} {auprc:>10.4f} {auprc_synth:>10.4f} {auprc_synth - auprc:>+10.4f}")
    else:
        print(f"\n⚠️ Dataset simulato non trovato: {dataset_path_synth}")