import streamlit as st
import zipfile
import time
import os
import base64

import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO

from model_manager import ModelManager
from confidence_slider import ConfidenceSlider
from image_loader import load_image_from_url
from performance_monitor import monitor_memory_usage
from preprocessing import preprocess_image
from predict import classify_image
from visualization import generate_saliency_map, overlay_saliency
from evaluation import evaluate_predictions
from utils import export_results_to_csv

# ─── Helpers ────────────────────────────────────────────────────────────

def load_pdf(path: str) -> str:
    """Return HTML link that embeds & allows downloading a PDF."""
    with open(path, "rb") as f:
        pdf_bytes = f.read()
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    filename = os.path.basename(path)
    href = (
        f'<a href="data:application/pdf;base64,{b64}" '
        f'download="{filename}">Download User Guide</a>'
    )
    return href

# ─── Sidebar: About & Settings ────────────────────────────────────────────

GUIDE_PATH = "WBC_Classifier_User_Guide_App.pdf"
st.sidebar.title("About")
st.sidebar.markdown(load_pdf(GUIDE_PATH), unsafe_allow_html=True)

st.sidebar.header("Settings & Model")
model_manager = ModelManager({
    "Enhanced CNN (v2)":          "models/enhanced_cnnv2.keras",
    "MobileNetV2 Head-only":     "models/mobilenet_v2_head_manual.keras",
    "MobileNetV2 Fine-tuned":    "models/mobilenet_v2_finetuned_manual.keras",
})
model_choice   = st.sidebar.selectbox("Choose Model", list(model_manager.available_models.keys()))
uploaded_model = st.sidebar.file_uploader("Or upload your custom model", type=["keras"])
model = (
    model_manager.upload_custom_model(uploaded_model)
    if uploaded_model
    else model_manager.load_model_by_name(model_choice)
)

confidence_slider   = ConfidenceSlider()
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
confidence_slider.adjust_threshold(confidence_threshold)

normalization = st.sidebar.selectbox("Normalization Method", ["0-1", "mean-std"])
image_url     = st.sidebar.text_input("Image URL")

labels_file = st.sidebar.file_uploader("Upload true labels CSV (optional)", type=["csv"])
labels_df   = pd.read_csv(labels_file) if labels_file and labels_file.type=="text/csv" else None

CLASS_LABELS = ['Neutrophil', 'Eosinophil', 'Basophil', 'Lymphocyte', 'Monocyte']

# ─── Main ─────────────────────────────────────────────────────────────────

st.title("WBC Classification Application")
uploaded_file = st.file_uploader("Upload Image or ZIP", type=["jpg","jpeg","png","zip"])

def process_image(image: Image.Image, filename: str):
    arr, orig = preprocess_image(image, normalization=normalization)
    tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
    idx, conf, _ = classify_image(model, tensor)
    return idx, conf, orig, generate_saliency_map(model, tensor, idx)

results = []

# -- URL‐based image
if image_url:
    try:
        img = load_image_from_url(image_url)
        idx, conf, orig, sal = process_image(img, image_url)
        if conf >= confidence_slider.threshold:
            results.append({
                "Filename":   image_url,
                "Prediction": CLASS_LABELS[idx],
                "Confidence": conf,
                "Original":   orig,
                "Saliency":   sal
            })
        else:
            st.warning(f"Classification confidence ({conf:.2f}) below threshold "
                       f"({confidence_slider.threshold:.2f}) for URL image.")
    except Exception:
        st.error("Failed to load or classify image from URL.")

# -- Local upload
if uploaded_file:
    # handle ZIPs of any mime type
    if zipfile.is_zipfile(uploaded_file):
        uploaded_file.seek(0)
        with zipfile.ZipFile(uploaded_file, "r") as z:
            for fname in z.namelist():
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    with z.open(fname) as f:
                        image = Image.open(f).convert("RGB")
                        idx, conf, orig, sal = process_image(image, fname)
                        if conf >= confidence_slider.threshold:
                            results.append({
                                "Filename":   os.path.basename(fname),
                                "Prediction": CLASS_LABELS[idx],
                                "Confidence": conf,
                                "Original":   orig,
                                "Saliency":   sal
                            })
                        else:
                            st.warning(f"Classification confidence ({conf:.2f}) below threshold "
                                       f"({confidence_slider.threshold:.2f}) for {fname}.")
        uploaded_file.seek(0)
    else:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            idx, conf, orig, sal = process_image(image, uploaded_file.name)
            if conf >= confidence_slider.threshold:
                results.append({
                    "Filename":   uploaded_file.name,
                    "Prediction": CLASS_LABELS[idx],
                    "Confidence": conf,
                    "Original":   orig,
                    "Saliency":   sal
                })
            else:
                st.warning(f"Classification confidence ({conf:.2f}) below threshold "
                           f"({confidence_slider.threshold:.2f}) for {uploaded_file.name}.")
        except Exception:
            st.error("Uploaded file is not a valid image or ZIP archive.")

# -- Display results
if results:
    df = pd.DataFrame(results)
    st.subheader("Classification Results")
    st.dataframe(df[['Filename','Prediction','Confidence']])
    export_results_to_csv(df[['Filename','Prediction','Confidence']])

    if len(df) > 1:
        summary = (
            df['Prediction']
            .value_counts()
            .rename_axis('Class')
            .reset_index(name='Count')
        )
        summary['Percent'] = (summary['Count']/summary['Count'].sum()*100).round(2)
        st.subheader("Batch Summary")
        st.table(summary)

    if labels_df is not None:
        evaluate_predictions(df, labels_df, CLASS_LABELS, st)

    idx = st.number_input("Select Image Index", 0, len(results)-1, 0)
    sel = results[idx]

    # Show original
    buf = BytesIO()
    sel["Original"].save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(f'<img src="data:image/png;base64,{b64}" width="300">', unsafe_allow_html=True)

    # Show overlay
    overlay = overlay_saliency(np.array(sel["Original"]), sel["Saliency"])
    st.image(overlay, caption="Saliency Map", width=300)

st.sidebar.write(f"Memory Usage: {monitor_memory_usage():.2f} MB")
