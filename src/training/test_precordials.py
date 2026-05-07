import os
import sys
import h5py
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
    
    plt.figure(figsize=(10,10), dpi=100) # Ingrandito per 16 classi
    plt.matshow(C, cmap=plt.cm.Reds, fignum=1) 
    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(str(C[j, i]), xy=(i, j), horizontalalignment='center', verticalalignment='center', fontsize=8)
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

    # Aggiornato ai dataset delle precordiali
    dataset_path_test  = "../../datasets/unlabelled_precordials_test.h5"
    dataset_path_val   = "../../datasets/unlabelled_precordials_val.h5"
    
    print("Caricamento dataset di test e validation PRECORDIALI...")
    with h5py.File(dataset_path_test, 'r') as f:
        # Prende SOLO i canali da 6 in poi (precordiali)
        x_test_raw = f['X'][:, 6:, :]
        x_test = np.transpose(x_test_raw, (0, 2, 1))
        y_test = f['Y'][:]
        
    with h5py.File(dataset_path_val, 'r') as f:
        x_val_raw = f['X'][:, 6:, :]
        x_val_eval = np.transpose(x_val_raw, (0, 2, 1))
        y_val_eval = f['Y'][:]

    input_shape = (SAMPLES_PER_WINDOW, 6)
    output_dims = 16 # 1 normale + 15 anomalie precordiali

    print("Costruzione del modello ILC...")
    model = build_model(input_shape, output_dims)
    
    weights_path = 'best_model_precordials.weights.h5'
    if os.path.exists(weights_path):
        print(f"Caricamento dei pesi da {weights_path}...")
        model.load_weights(weights_path)
    else:
        print(f"⚠️ ATTENZIONE: {weights_path} non trovato. Verrà valutato un modello non addestrato.")

    print("\n" + "="*50)
    print("VALUTAZIONE TEST SET")
    print("="*50)
    metrics, acc = evaluater(x_test, y_test, model, 'precordials_cm_test.png')
    auroc, auprc = evaluater_pro(x_test, y_test, model)
    
    print(f"\n[RISULTATI TEST SET]")
    print(f"Accuratezza Totale: {acc:.4f}")
    print(f"AUROC (Macro):     {auroc:.4f}")
    print(f"AuPRC (Macro):     {auprc:.4f}")
    print("-" * 60)
    print(f"{'Classe':<10} | {'Prec.':<8} | {'Rec.':<8} | {'Spec.':<8} | {'F1':<8}")
    print("-" * 60)
    for i in range(len(metrics)):
        p, r, s, f1 = metrics[i]
        print(f"Classe {i:<3} | {p:<8.4f} | {r:<8.4f} | {s:<8.4f} | {f1:<8.4f}")
    print("-" * 60)

    print("\n" + "="*50)
    print("VALUTAZIONE VALIDATION SET")
    print("="*50)
    metrics_val, acc_val = evaluater(x_val_eval, y_val_eval, model, 'precordials_cm_val.png')
    auroc_val, auprc_val = evaluater_pro(x_val_eval, y_val_eval, model)
    
    print(f"\n[RISULTATI VALIDATION SET]")
    print(f"Accuratezza Totale: {acc_val:.4f}")
    print(f"AUROC (Macro):     {auroc_val:.4f}")
    print(f"AuPRC (Macro):     {auprc_val:.4f}")
    print("-" * 60)
    print(f"{'Classe':<10} | {'Prec.':<8} | {'Rec.':<8} | {'Spec.':<8} | {'F1':<8}")
    print("-" * 60)
    for i in range(len(metrics_val)):
        p, r, s, f1 = metrics_val[i]
        print(f"Classe {i:<3} | {p:<8.4f} | {r:<8.4f} | {s:<8.4f} | {f1:<8.4f}")
    print("-" * 60)
