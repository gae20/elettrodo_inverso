# -*- coding: utf-8 -*-
import os
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
from LDenseNet import build_model
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import h5py



# --- CONFIGURAZIONE GRAFICA ---
font = {'family': 'DejaVu Sans'}
import matplotlib as mpl
mpl.rcParams['font.family'] = 'DejaVu Sans'

# --- METRICHE E VALUTAZIONE ---
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

# --- DATA GENERATOR ---
class H5DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_path, batch_size=32, num_classes=8, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.file_path = file_path
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        with h5py.File(self.file_path, 'r') as f:
            self.total_samples = f['X'].shape[0]
            self.indices = np.arange(self.total_samples)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.total_samples / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        idx_in_batch = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        idx_sorted = sorted(idx_in_batch)
        with h5py.File(self.file_path, 'r') as f:
            x = f['X'][idx_sorted]
            y = f['Y'][idx_sorted]
        x = np.transpose(x, (0, 2, 1))
        y = to_categorical(y, num_classes=self.num_classes)
        return x, y

# --- LOGICA DI TRAINING ---
def train_model(model, train_data, val_data, x_test_list, y_test_list, save_path, EP, LR, BS, pic_path_test, pic_path_val):
    opt = Adam(learning_rate=LR)
    ME = tf.keras.metrics.F1Score(average='macro', name='f1_score')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', ME])
    
    callbacks = [
        ModelCheckpoint(save_path, monitor='val_f1_score', verbose=1, save_best_only=True, save_weights_only=True, mode='max'),
        EarlyStopping(monitor='val_f1_score', patience=12, verbose=1, mode='max', restore_best_weights=True)
    ]
    
    print(f"\nInizio addestramento con Batch Size: {BS}...")
    history = model.fit(train_data, epochs=EP, validation_data=val_data, callbacks=callbacks, verbose=1)
    
    print("\n" + "="*50)
    print("VALUTAZIONE FINALE (MIGLIORI PESI CARICATI)")
    print("="*50)
    
    model.load_weights(save_path)
    
    metrics, acc = evaluater(x_test_list[0], y_test_list[0], model, pic_path_test)
    auroc, auprc = evaluater_pro(x_test_list[0], y_test_list[0], model)
    
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

    evaluater(x_test_list[1], y_test_list[1], model, pic_path_val)
    return history

# --- MAIN ---
if __name__ == '__main__':
    # --- CONFIGURAZIONE GPU ---
    # Verifica disponibilità GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Permetti crescita dinamica della memoria GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU disponibili: {len(gpus)}")
            print(f"   Dispositivi: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"⚠️ Errore configurazione GPU: {e}")
    else:
        print("⚠️ Nessuna GPU rilevata, uso CPU")

    with h5py.File("test_dataset_new.h5", 'r') as f:
        num_leads = f['X'].shape[1] 
        time_steps = f['X'].shape[2]
        x_test = np.transpose(f['X'][:], (0, 2, 1))
        y_test = f['Y'][:]
        
    with h5py.File("val_dataset_new.h5", 'r') as f:
        x_val_eval = np.transpose(f['X'][:], (0, 2, 1))
        y_val_eval = f['Y'][:]

    # Parametri
    input_shape = (time_steps, num_leads)
    output_dims = 8 
    EP = 30        
    LR = 1e-3 
    BS = 256 # <--- BATCH SIZE IMPOSTATA A 1024
    save_path = 'best_model_ecg_thesis.weights.h5'
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"--- CONFIGURAZIONE RILEVATA ---")
    print(f"Batch Size: {BS}")
    print(f"Input Shape: {input_shape}")
    print(f"--------------------------------")

    train_gen = H5DataGenerator("train_dataset_new.h5", batch_size=BS, num_classes=output_dims, shuffle=True)
    val_gen = H5DataGenerator("val_dataset_new.h5", batch_size=BS, num_classes=output_dims, shuffle=False)

    model = build_model(input_shape, output_dims)
    model.summary()

    train_model(
        model, 
        train_gen, 
        val_gen, 
        [x_test, x_val_eval], 
        [y_test, y_val_eval], 
        save_path, 
        EP, LR, BS, 
        os.path.join(base_dir, "confusion_matrix_test.png"), 
        os.path.join(base_dir, "confusion_matrix_val_final.png")
    )

    print("\n✅ Processo completato.")