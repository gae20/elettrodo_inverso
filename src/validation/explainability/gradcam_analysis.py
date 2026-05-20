import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Aggiungi src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.ldensenet import build_model
from data.data_pipeline import get_ecg, all_leads_preprocessing, limb_interchange_simulation
from utils.config import ALL_LEADS, LIMB_LEADS, MAPPING_INV, LABEL_MAP_CLEAN, SAMPLES_PER_WINDOW

# Configura i percorsi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", "..", "datasets", "dataset", "thesis-sample.csv"))
WEIGHTS_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "best_model_unlabelled_z_median_limbs.weights.h5"))

def robust_global_scale(sigs_array):
    x = sigs_array.astype(np.float32)
    medians = np.median(x, axis=1, keepdims=True)
    q75, q25 = np.percentile(x, [75, 25])
    scale = (q75 - q25) / 1.34896
    scale = max(scale, 1e-8)
    return (x - medians) / scale

def make_gradcam_heatmap(input_signal, model, last_conv_layer_name, pred_index=None):
    # Model to extract features and predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(input_signal)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the class score wrt feature maps
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Mean intensity of gradients per feature map (importance)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # Weighting the feature maps
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def analyze_gradcam(class_name='LA-RA'):
    print(f"\n--- Grad-CAM Analysis: {class_name} ---")
    
    # Load Model
    input_shape = (SAMPLES_PER_WINDOW, 6)
    model = build_model(input_shape, 6)
    if os.path.exists(WEIGHTS_PATH):
        model.load_weights(WEIGHTS_PATH)
        print("Pesi caricati.")
    else:
        print(f"Errore: Pesi non trovati in {WEIGHTS_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    df_valido = df[df["Inversione"] != "?"].copy()
    df_valido["Inversione"] = df_valido["Inversione"].apply(lambda x: LABEL_MAP_CLEAN.get(x, x))
    
    # Prendi esempi
    real_id = df_valido[df_valido["Inversione"] == class_name].iloc[0]["Num"]
    norm_id = df_valido[df_valido["Inversione"] == "normale"].iloc[0]["Num"]
    
    data_real = get_ecg(real_id)
    data_norm = get_ecg(norm_id)
    mode = MAPPING_INV[class_name]
    
    # 1. Real Signal
    sigs_real = all_leads_preprocessing(data_real["signals"])
    x_real = np.array([sigs_real[l] for l in LIMB_LEADS], dtype=np.float32)
    x_real_norm = robust_global_scale(x_real)[:, :SAMPLES_PER_WINDOW]
    x_real_input = np.transpose(x_real_norm, (1, 0))[np.newaxis, ...]
    
    # 2. Synthetic Signal (Proposed logic)
    sigs_norm = all_leads_preprocessing(data_norm["signals"])
    sigs_sim = limb_interchange_simulation(mode, sigs_norm)
    x_sim = np.array([sigs_sim[l] for l in LIMB_LEADS], dtype=np.float32)
    x_sim_norm = robust_global_scale(x_sim)[:, :SAMPLES_PER_WINDOW]
    x_sim_input = np.transpose(x_sim_norm, (1, 0))[np.newaxis, ...]
    
    # Compute Heatmaps
    heatmap_real = make_gradcam_heatmap(x_real_input, model, "concatenate_2")
    heatmap_sim = make_gradcam_heatmap(x_sim_input, model, "concatenate_2")
    
    # Rescale heatmaps to signal length
    heatmap_real_res = np.interp(np.linspace(0, len(heatmap_real), SAMPLES_PER_WINDOW), np.arange(len(heatmap_real)), heatmap_real)
    heatmap_sim_res = np.interp(np.linspace(0, len(heatmap_sim), SAMPLES_PER_WINDOW), np.arange(len(heatmap_sim)), heatmap_sim)
    
    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # REAL
    ax = axes[0]
    for i, lead in enumerate(LIMB_LEADS):
        ax.plot(x_real_norm[i] + i*5, color='gray', alpha=0.5)
    ax.set_title(f"Real ECG: {class_name} (ID {real_id})")
    # Overlay Heatmap
    heatmap_img = np.tile(heatmap_real_res, (10, 1))
    ax.imshow(heatmap_img, extent=[0, SAMPLES_PER_WINDOW, -2, 30], aspect='auto', cmap='Reds', alpha=0.4)
    
    # SYNTHETIC
    ax = axes[1]
    for i, lead in enumerate(LIMB_LEADS):
        ax.plot(x_sim_norm[i] + i*5, color='gray', alpha=0.5)
    ax.set_title(f"Synthetic ECG: {class_name}")
    # Overlay Heatmap
    heatmap_img_s = np.tile(heatmap_sim_res, (10, 1))
    ax.imshow(heatmap_img_s, extent=[0, SAMPLES_PER_WINDOW, -2, 30], aspect='auto', cmap='Reds', alpha=0.4)
    
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, f"gradcam_{class_name}.png")
    plt.savefig(out_path, dpi=150)
    print(f"Analisi completata. Grafico: {out_path}")

if __name__ == "__main__":
    analyze_gradcam('ROT_ANTIORARIA')
