import streamlit as st
import zipfile, time, os
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO
import base64

from model_manager import ModelManager
from confidence_slider import ConfidenceSlider
from image_loader import load_image_from_url
from performance_monitor import monitor_memory_usage
from preprocessing import preprocess_image
from predict import classify_image
from visualization import generate_saliency_map, overlay_saliency
from evaluation import evaluate_predictions
from utils import export_results_to_csv

# Sidebar: Model and Settings
st.sidebar.header("Settings & Model")
model_manager = ModelManager({
    "Enhanced CNN (v2)": "models/enhanced_cnnv2.keras",
    "MobileNetV2 Head-only": "models/mobilenet_v2_head_manual.keras",
    "MobileNetV2 Fine-tuned": "models/mobilenet_v2_finetuned_manual.keras"
})

model_choice = st.sidebar.selectbox("Choose Model", list(model_manager.available_models.keys()))
uploaded_model = st.sidebar.file_uploader("Or upload your custom model", type=["keras"])
if uploaded_model:
    model = model_manager.upload_custom_model(uploaded_model)
else:
    model = model_manager.load_model_by_name(model_choice)

confidence_slider = ConfidenceSlider()
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
confidence_slider.adjust_threshold(confidence_threshold)

normalization = st.sidebar.selectbox("Normalization Method", ["0-1", "mean-std"])
image_url = st.sidebar.text_input("Image URL")

labels_file = st.sidebar.file_uploader("Upload true labels CSV (optional)", type=["csv"])
labels_df = pd.read_csv(labels_file) if labels_file and labels_file.type == "text/csv" else None

# Define the class labels
class_labels = ['Neutrophil', 'Eosinophil', 'Basophil', 'Lymphocyte', 'Monocyte']

# Main application logic
st.title("WBC Classification Application")
uploaded_file = st.file_uploader("Upload Image or ZIP", type=["jpg", "jpeg", "png", "zip"])

def process_image(image, filename):
    arr, orig = preprocess_image(image, normalization=normalization)
    tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
    idx, conf, _ = classify_image(model, tensor)
    if conf < confidence_slider.threshold:
        return None
    label = class_labels[idx]
    saliency = generate_saliency_map(model, tensor, idx)
    return {"Filename": filename, "Prediction": label, "Confidence": conf, "Original": orig, "Saliency": saliency}

results = []
if image_url:
    image = load_image_from_url(image_url)
    result = process_image(image, image_url)
    if result:
        results.append(result)

if uploaded_file:
    if uploaded_file.type == "application/zip":
        with zipfile.ZipFile(uploaded_file, "r") as z:
            for fname in z.namelist():
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    with z.open(fname) as f:
                        image = Image.open(f).convert("RGB")
                        base_name = os.path.basename(fname)
                        result = process_image(image, base_name)
                        if result:
                            results.append(result)
    else:
        image = Image.open(uploaded_file).convert("RGB")
        result = process_image(image, uploaded_file.name)
        if result:
            results.append(result)

if results:
    df = pd.DataFrame(results)
    st.subheader("Classification Results")
    st.dataframe(df[['Filename', 'Prediction', 'Confidence']])
    export_results_to_csv(df[['Filename', 'Prediction', 'Confidence']])

    if len(df) > 1:
        summary = df['Prediction'].value_counts().rename_axis('Class').reset_index(name='Count')
        summary['Percent'] = (summary['Count'] / summary['Count'].sum() * 100).round(2)
        st.subheader("Batch Summary")
        st.table(summary)

    if labels_df is not None:
        evaluate_predictions(df, labels_df, class_labels, st)

    idx = st.number_input("Select Image Index", 0, len(results)-1, 0)
    selected = results[idx]
    orig = selected["Original"]
    heatmap = selected["Saliency"]

    buf = BytesIO()
    orig.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(f'<img src="data:image/png;base64,{b64}" alt="Original WBC" width="300">', unsafe_allow_html=True)
    overlay_img = overlay_saliency(np.array(orig), heatmap)
    st.image(overlay_img, caption="Saliency Map", width=300)

st.sidebar.write(f"Memory Usage: {monitor_memory_usage():.2f} MB")
