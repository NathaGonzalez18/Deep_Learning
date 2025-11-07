st.set_page_config(
    page_title="Clasificador de Género",
    page_icon="🟣",
    layout="wide"
)

st.markdown("""
<style>

h1, h2, h3, h4 {
    font-weight: 600;
    text-align: center;
    color: #C084FC; /* Púrpura brillante para títulos */
}

[class^="stMetric"] {
    background-color: #1a1b1e !important;
    border-radius: 10px;
    padding: 15px;
}

.uploadedFile {
    background-color: #1a1b1e !important;
    padding: 12px;
    border-radius: 8px;
}

img {
    border-radius: 12px;
}

/* Botones estilo neón */
.stButton>button {
    background: linear-gradient(90deg, #8b5cf6, #c084fc);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}
.stButton>button:hover {
    opacity: 0.88;
    cursor: pointer;
}

</style>
""", unsafe_allow_html=True)


import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Cargar modelo
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("modelo_genero.h5")
    return model

model = load_model()

IMG_SIZE = 224  # cambia si tu modelo usa otro tamaño

st.title("Clasificación de Género con Interpretabilidad (Grad-CAM & Saliency Map)")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

# ---------------------- Función Grad-CAM ----------------------
def grad_cam(model, img_array, last_conv_layer_name="conv1_block3"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return heatmap

# ---------------------- Función Saliency Map ----------------------
def saliency_map(model, img_array):
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        prediction = model(img_tensor)
    gradient = tape.gradient(prediction, img_tensor)[0]
    gradient = tf.reduce_max(tf.abs(gradient), axis=-1)
    gradient = gradient.numpy()
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
    return gradient

# ---------------------- PROCESAMIENTO PRINCIPAL ----------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", width=250)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0][0]

        # --- Probabilidad Hombre/Mujer ---
    prob_mujer = prob
    prob_hombre = 1 - prob

    # --- Generar Grad-CAM y Saliency ---
    heatmap = grad_cam(model, img_array)
    heatmap_overlay = cv2.addWeighted(np.array(img.resize((IMG_SIZE, IMG_SIZE))), 0.6, heatmap, 0.4, 0)

    sal_map = saliency_map(model, img_array)

    # --- PESTAÑAS ---
    tab1, tab2, tab3 = st.tabs(["🔍 Clasificación", "🔥 Grad-CAM", "🌈 Saliency Map"])

    with tab1:
        st.subheader("Resultado de la Clasificación")
        st.image(image, caption="Imagen original", use_column_width=True)
        st.metric("Probabilidad Mujer", f"{prob_mujer:.3f}")
        st.metric("Probabilidad Hombre", f"{prob_hombre:.3f}")

        if prob_mujer >= 0.5:
            st.success("Clasificación: **Mujer** 👩")
        else:
            st.success("Clasificación: **Hombre** 👨")

    with tab2:
        st.subheader("Grad-CAM: Regiones que más influyeron en la decisión")
        st.image(heatmap_overlay, channels="BGR", use_column_width=True)

    with tab3:
        st.subheader("Saliency Map: Sensibilidad por píxel")
        st.image(sal_map, clamp=True, use_column_width=True)

