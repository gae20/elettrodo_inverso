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

    # Dataset di test: dati REALI etichettati
    dataset_path_test = "../../datasets/unlabelled_simulated_final/unlabelled_z_median_limbs_test_backup.h5"
    
    print(f"Caricamento dataset di test REALI da: {dataset_path_test}")
    with h5py.File(dataset_path_test, 'r') as f:
        y_all = f['Y'][:]
        valid_idx = np.where(y_all < 6)[0]
        x_test_raw = f['X'][valid_idx, :6, :]
        x_test = np.transpose(x_test_raw, (0, 2, 1))
        y_test = y_all[valid_idx]

    input_shape = (SAMPLES_PER_WINDOW, 6)
    output_dims = 6 
    
    # Percorso pesi Final Multi-Level Noise
    save_path = '../training/unlabelled_z_median_weights_and_cm/PROVA_best_model_targeted_noise_limbs.weights.h5'
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("Costruzione del modello...")
    model = build_model(input_shape, output_dims)
    
    print(f"Caricamento dei pesi da {save_path}...")
    model.load_weights(os.path.join(base_dir, save_path))

    print("\n" + "="*50)
    print("VALUTAZIONE TEST SET (DATI REALI)")
    print("="*50)
    
    pic_path_test = os.path.join(base_dir, "../training/unlabelled_z_median_weights_and_cm/prova3.png")
    
    # Valutazione
    y_probs = model.predict(x_test, batch_size=32)
    y_pred = np.argmax(y_probs, axis=1)
    
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n[REPORT DI CLASSIFICAZIONE]")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
    
    C = confusion_matrix(y_test, y_pred)
    
    # Salvataggio dell'immagine della matrice di confusione
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