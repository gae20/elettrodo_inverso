import os
import sys
import h5py
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from train_limbs import build_model, H5DataGenerator, evaluater

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import SAMPLES_PER_WINDOW

# ============================================================
# FINE-TUNING CON CLASS WEIGHTS E CHECKPOINT PER EPOCA
# Obiettivo: migliorare recall di ROT_ANT (cl.5) e ROT_ORA (cl.4)
# ============================================================

def resume_training():
    # --- 1. CONFIGURAZIONE EPOCHE E LEARNING RATE ---
    EP = 10         # Solo 10 epoche per il fine-tuning
    LR = 1e-5       # LR molto basso per non alterare drasticamente i pesi pre-addestrati
    BS = 256
    input_shape = (SAMPLES_PER_WINDOW, 6)
    output_dims = 6

    # --- 2. CONFIGURAZIONE CLASS WEIGHTS ---
    # Forziamo il modello a prestare più attenzione alle rotazioni
    CLASS_WEIGHT = {
        0: 1.0,   # normale
        1: 1.0,   # LA-RA
        2: 1.5,   # RA-LL
        3: 1.0,   # LA-LL
        4: 1.5,   # ROT_ORA 
        5: 1.8,   # ROT_ANT 
    }

    # --- PATH DATASET ---
    dataset_dir = "../../datasets/unlabelled_simulated_gain"
    dataset_path_train = os.path.join(dataset_dir, "unlabelled_targeted_noise_limbs_train.h5")
    dataset_path_val   = os.path.join(dataset_dir, "unlabelled_targeted_noise_limbs_val.h5")
    dataset_path_test  = "../../datasets/labelled_z_median_limbs_test_validation.h5"

    # --- 3. PATH DEI PESI DA CARICARE E SALVARE ---
    load_weights_from = 'unlabelled_targeted_noise_weights_and_cm/best_model_targeted_noise_limbs.weights.h5'

    out_dir = 'class_weights_finetuning_more_layers'
    os.makedirs(out_dir, exist_ok=True)
    
    # Salvataggio con tag dell'epoca nel nome del file
    save_path = os.path.join(out_dir, 'cw_finetuned_more_layers_limbs_epoch_{epoch:02d}.weights.h5')

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # --- CARICAMENTO DATI ---
    print(f"Caricamento test set REALE da: {dataset_path_test}")
    with h5py.File(dataset_path_test, 'r') as f:
        y_all = f['Y'][:]
        valid_idx = np.where(y_all < 6)[0]
        x_test_raw = f['X'][valid_idx, :6, :]
        x_test = np.transpose(x_test_raw, (0, 2, 1))
        y_test = y_all[valid_idx]

    train_gen = H5DataGenerator(dataset_path_train, batch_size=BS, num_classes=output_dims, shuffle=True)
    val_gen   = H5DataGenerator(dataset_path_val,   batch_size=BS, num_classes=output_dims, shuffle=False)

    # --- COSTRUZIONE E CARICAMENTO MODELLO ---
    model = build_model(input_shape, output_dims)

    weights_path = os.path.join(base_dir, load_weights_from)
    if os.path.exists(weights_path):
        print(f"\n✅ Caricamento pesi pre-addestrati da: {weights_path}")
        model.load_weights(weights_path)
    else:
        print(f"\n❌ ERRORE: Pesi non trovati in {weights_path}")
        return

    # --- CONGELAMENTO PARZIALE DEI LAYER ---
    UNFREEZE_LAYERS = 3  # Sblocca gli ultimi 3 layer per dare più flessibilità
    print(f"\n❄️ Congelamento dei layer base, lasciando addestrabili gli ultimi {UNFREEZE_LAYERS} layer...")
    
    # Congeliamo tutti i layer tranne gli ultimi N
    for layer in model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False
        
    # Ci assicuriamo che gli ultimi N layer siano sbloccati
    for layer in model.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True
        
    print("\nStato dei layer:")
    for i, layer in enumerate(model.layers):
        print(f"Layer {i} ({layer.name}): Trainable = {layer.trainable}")
    # --------------------------------------------------------

    # --- VALUTAZIONE BASELINE ---
    print("\nCalcolo baseline sul test set reale (pre-finetuning)...")
    y_probs_base = model.predict(x_test, batch_size=32)
    y_pred_base = np.argmax(y_probs_base, axis=1)
    acc_base = np.mean(y_pred_base == y_test)
    print(f"Accuratezza baseline: {acc_base:.4f}\n")

    # --- FINE-TUNING ---
    opt = Adam(learning_rate=LR)
    ME = tf.keras.metrics.F1Score(average='macro', name='f1_score')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', ME])

    # Callback per salvare OGNI epoca
    callbacks = [
        ModelCheckpoint(save_path, monitor='val_f1_score', verbose=1,
                        save_best_only=False, save_weights_only=True, mode='max'),
        EarlyStopping(monitor='val_f1_score', patience=4, verbose=1, mode='max')
    ]

    print(f"\n🚀 Inizio fine-tuning con class weights: {CLASS_WEIGHT}")
    model.fit(
        train_gen,
        epochs=EP,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=CLASS_WEIGHT,
        verbose=1
    )

    # --- FINE ADDESTRAMENTO ---
    print(f"\n✅ Fine-tuning completato.")
    print(f"I pesi di ogni epoca sono stati salvati in: {out_dir}")
    print("Ora esegui lo script di test per valutare le singole epoche sui dati reali.")

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Configurata memoria GPU: {gpu}")
    resume_training()