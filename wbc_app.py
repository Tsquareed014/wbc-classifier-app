import streamlit as st
import zipfile, os
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

# New imports
from styles import apply_text_size
from keyboard_nav import nav_index

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
GUIDE_PATH = "WBC_Classifier_User_Guide_App.pdf"
CLASS_LABELS = ['Neutrophil', 'Eosinophil', 'Basophil', 'Lymphocyte', 'Monocyte']

# -------------------------------------------------------------------
# Sidebar: Settings & Model
# -------------------------------------------------------------------
st.sidebar.header("Settings & Model")

model_manager = ModelManager({
    "Enhanced CNN (v2)": "models/enhanced_cnnv2.keras",
    "MobileNetV2 Head-only": "models/mobilenet_v2_head_manual.keras",
    "MobileNetV2 Fine-tuned": "models/mobilenet_v2_finetuned_manual.keras"
})

model_choice   = st.sidebar.selectbox("Choose Model", list(model_manager.available_models.keys()))
uploaded_model = st.sidebar.file_uploader("Or upload your custom model", type=["keras"])
if uploaded_model:
    model = model_manager.upload_custom_model(uploaded_model)
else:
    model = model_manager.load_model_by_name(model_choice)

# Confidence threshold
confidence_slider     = ConfidenceSlider()
confidence_threshold  = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
confidence_slider.adjust_threshold(confidence_threshold)

# Normalization choice
normalization = st.sidebar.selectbox("Normalization Method", ["0-1", "mean-std"])

# Adjustable text size via CSS
text_size = st.sidebar.slider("Text size (%)", 80, 150, 100, 5)
apply_text_size(text_size)

# Accessibility: keyboard navigation toggle
enable_kb_nav = st.sidebar.checkbox("Enable keyboard navigation", value=True)

# Labels CSV (for evaluation)
labels_file = st.sidebar.file_uploader("Upload true labels CSV (optional)", type=["csv"])
labels_df   = pd.read_csv(labels_file) if labels_file and labels_file.type == "text/csv" else None

# PDF download button
if os.path.exists(GUIDE_PATH):
    with open(GUIDE_PATH, "rb") as f:
        guide_bytes = f.read()
    st.sidebar.download_button(
        label="📄 Download User Guide",
        data=guide_bytes,
        file_name="WBC_Classifier_User_Guide.pdf",
        mime="application/pdf"
    )
else:
    st.sidebar.warning("User guide not found. Push the PDF to your repo.")

# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
st.title("WBC Classification Application")

uploaded_file = st.file_uploader("Upload Image or ZIP", type=["jpg","jpeg","png","zip"])
image_url     = st.sidebar.text_input("Image URL")

def process_image(image, filename):
    arr, orig = preprocess_image(image, normalization=normalization)
    tensor    = tf.convert_to_tensor(arr, dtype=tf.float32)
    idx, conf, _ = classify_image(model, tensor)
    if conf < confidence_slider.threshold:
        return None
    label   = CLASS_LABELS[idx]
    saliency = generate_saliency_map(model, tensor, idx)
    return {
        "Filename": filename,
        "Prediction": label,
        "Confidence": conf,
        "Original": orig,
        "Saliency": saliency
    }

results = []

# URL workflow
if image_url:
    img = load_image_from_url(image_url)
    res = process_image(img, image_url)
    if res:
        results.append(res)

# File upload workflow
if uploaded_file:
    if uploaded_file.type == "application/zip":
        with zipfile.ZipFile(uploaded_file, "r") as z:
            for fname in z.namelist():
                if fname.lower().endswith((".jpg",".jpeg",".png")):
                    with z.open(fname) as f:
                        img = Image.open(f).convert("RGB")
                        res = process_image(img, os.path.basename(fname))
                        if res:
                            results.append(res)
    else:
        img = Image.open(uploaded_file).convert("RGB")
        res = process_image(img, uploaded_file.name)
        if res:
            results.append(res)

# Display results
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
        summary['Percent'] = (summary['Count'] / summary['Count'].sum() * 100).round(2)
        st.subheader("Batch Summary")
        st.table(summary)

    if labels_df is not None:
        evaluate_predictions(df, labels_df, CLASS_LABELS, st)

    # Image navigation
    max_idx = len(results) - 1
    # initialize session state
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0

    if enable_kb_nav:
        # arrow key driven
        new_idx = nav_index(st.session_state.current_idx, max_idx)
        st.session_state.current_idx = new_idx
    else:
        # fallback number input
        new_idx = st.number_input(
            "Select Image Index",
            min_value=0,
            max_value=max_idx,
            value=st.session_state.current_idx
        )
        st.session_state.current_idx = new_idx

    sel = results[st.session_state.current_idx]
    orig = sel["Original"]
    heat = sel["Saliency"]

    # Show original
    buf = BytesIO()
    orig.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(f'<img src="data:image/png;base64,{b64}" width="300">', unsafe_allow_html=True)

    # Show saliency overlay
    overlay_img = overlay_saliency(np.array(orig), heat)
    st.image(overlay_img, caption="Saliency Map", width=300)

st.sidebar.write(f"Memory Usage: {monitor_memory_usage():.2f} MB")

