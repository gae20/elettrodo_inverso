import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

# Aggiungi src al path per importare i moduli interni
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.ldensenet import build_model
from utils.config import SAMPLES_PER_WINDOW

def cal_metrics(confusion_matrix):
    """Calcola Precision, Recall, Specificity e F1-score dalla matrice di confusione."""
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
    """Esegue la predizione e genera la matrice di confusione salvandola come immagine."""
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
    """Calcola AUROC e AuPRC macro."""
    y_probs = model.predict(x_test, batch_size=32)
    num_classes = y_probs.shape[-1]
    y_test_one_hot = to_categorical(y_test_idx, num_classes=num_classes)
    auroc = roc_auc_score(y_test_one_hot, y_probs, multi_class='ovr', average='macro')
    auprc = average_precision_score(y_test_one_hot, y_probs, average='macro')
    return auroc, auprc

if __name__ == "__main__":
    # Configurazione percorsi relativa alla cartella 'src/validation'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_dir = os.path.join(base_dir, '..', 'training')
    weights_path = os.path.join(training_dir, 'best_model_limbs.weights.h5')
    dataset_path_test = os.path.join(base_dir, '..', 'datasets', 'limbs_test.h5')
    dataset_path_val = os.path.join(base_dir, '..', 'datasets', 'limbs_val.h5')
    
    # Verifica esistenza file pesi
    if not os.path.exists(weights_path):
        print(f"❌ Errore: Pesi non trovati in {os.path.abspath(weights_path)}")
        sys.exit(1)

    print(f"--- MODULO VALUTAZIONE MODELLO LIMBS ---")
    print(f"Percorso pesi: {os.path.abspath(weights_path)}")
    
    # Caricamento dati (solo canali periferici)
    print("Caricamento dataset...")
    with h5py.File(dataset_path_test, 'r') as f:
        x_test_raw = f['X'][:, :6, :]
        x_test = np.transpose(x_test_raw, (0, 2, 1))
        y_test = f['Y'][:]
        
    with h5py.File(dataset_path_val, 'r') as f:
        x_val_raw = f['X'][:, :6, :]
        x_val = np.transpose(x_val_raw, (0, 2, 1))
        y_val = f['Y'][:]

    # Ricostruzione architettura e caricamento pesi
    input_shape = (SAMPLES_PER_WINDOW, 6)
    output_dims = 6 # 1 normale + 5 anomalie
    model = build_model(input_shape, output_dims, dropout_rate=0.3)
    model.load_weights(weights_path)
    print("✅ Modello e pesi caricati correttamente.")

    # 1. Valutazione su TEST SET
    metrics, acc = evaluater(x_test, y_test, model, os.path.join(base_dir, "evaluation_limbs_test.png"))
    auroc, auprc = evaluater_pro(x_test, y_test, model)
    
    print(f"\n" + "="*40)
    print(f"[REPORT FINALE: TEST SET]")
    print(f"="*40)
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

    # 2. Valutazione su VALIDATION SET
    evaluater(x_val, y_val, model, os.path.join(base_dir, "evaluation_limbs_val.png"))
    
    print(f"\n✅ Valutazione terminata. Matrici salvate in: {base_dir}")
