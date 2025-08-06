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

# ─── Helper to embed & download your PDF guide ────────────────────────────────
def load_pdf(path: str) -> str:
    """Return an <iframe> snippet with your PDF base64-inlined."""
    raw = open(path, "rb").read()
    b64  = base64.b64encode(raw).decode("utf-8")
    return (
        f'<iframe src="data:application/pdf;base64,{b64}" '
        'width="100%" height="300px" style="border: none;"></iframe>'
    )

GUIDE_PATH = "WBC_Classifier_User_Guide_App.pdf"

# ─── Sidebar: About & PDF ─────────────────────────────────────────────────────
st.sidebar.title("About")
st.sidebar.markdown(load_pdf(GUIDE_PATH), unsafe_allow_html=True)
with open(GUIDE_PATH, "rb") as f:
    pdf_bytes = f.read()
st.sidebar.download_button(
    label="📄 Download User Guide",
    data=pdf_bytes,
    file_name=os.path.basename(GUIDE_PATH),
    mime="application/pdf"
)

# ─── Sidebar: Model & Settings ───────────────────────────────────────────────
st.sidebar.header("Settings & Model")
model_manager = ModelManager({
    "Enhanced CNN (v2)":      "models/enhanced_cnnv2.keras",
    "MobileNetV2 Head-only":  "models/mobilenet_v2_head_manual.keras",
    "MobileNetV2 Fine-tuned": "models/mobilenet_v2_finetuned_manual.keras",
})
model_choice   = st.sidebar.selectbox("Choose Model", list(model_manager.available_models.keys()))
uploaded_model = st.sidebar.file_uploader("Or upload custom model", type=["keras"])
if uploaded_model:
    model = model_manager.upload_custom_model(uploaded_model)
else:
    model = model_manager.load_model_by_name(model_choice)

confidence_slider    = ConfidenceSlider()
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
confidence_slider.adjust_threshold(confidence_threshold)

normalization = st.sidebar.selectbox("Normalization Method", ["0-1", "mean-std"])
image_url     = st.sidebar.text_input("Image URL")

labels_file = st.sidebar.file_uploader("Upload true labels CSV (optional)", type=["csv"])
labels_df   = pd.read_csv(labels_file) if labels_file and labels_file.type=="text/csv" else None

# ─── Main App ────────────────────────────────────────────────────────────────
st.title("WBC Classification Application")
uploaded_file = st.file_uploader("Upload Image or ZIP", type=["jpg","jpeg","png","zip"])

CLASS_LABELS = ['Neutrophil','Eosinophil','Basophil','Lymphocyte','Monocyte']

def process_image(img: Image.Image, name: str):
    arr, orig = preprocess_image(img, normalization=normalization)
    tensor    = tf.convert_to_tensor(arr, dtype=tf.float32)
    idx, conf, _ = classify_image(model, tensor)
    if conf < confidence_slider.threshold:
        return None
    sal = generate_saliency_map(model, tensor, idx)
    return {
        "Filename":   name,
        "Prediction": CLASS_LABELS[idx],
        "Confidence": conf,
        "Original":   orig,
        "Saliency":   sal
    }

results = []

# via URL
if image_url:
    try:
        img = load_image_from_url(image_url)
        r   = process_image(img, image_url)
        if r: results.append(r)
    except Exception:
        st.warning("Couldn't load image from URL.")

# via upload
if uploaded_file:
    if uploaded_file.type == "application/zip":
        with zipfile.ZipFile(uploaded_file, "r") as z:
            for fname in z.namelist():
                if fname.lower().endswith((".jpg","jpeg","png")):
                    with z.open(fname) as f:
                        img = Image.open(f).convert("RGB")
                        r   = process_image(img, os.path.basename(fname))
                        if r: results.append(r)
    else:
        img = Image.open(uploaded_file).convert("RGB")
        r   = process_image(img, uploaded_file.name)
        if r: results.append(r)

# ─── Display Results ─────────────────────────────────────────────────────────
if results:
    df = pd.DataFrame(results)
    st.subheader("Classification Results")
    st.dataframe(df[['Filename','Prediction','Confidence']])
    export_results_to_csv(df[['Filename','Prediction','Confidence']])

    if len(df) > 1:
        cnt = df['Prediction'].value_counts().rename_axis('Class').reset_index(name='Count')
        cnt['Percent'] = (cnt['Count']/cnt['Count'].sum()*100).round(2)
        st.subheader("Batch Summary")
        st.table(cnt)

    if labels_df is not None:
        evaluate_predictions(df, labels_df, CLASS_LABELS, st)

    # ─── Stateful index + Prev/Next buttons ─────────────────────────────────
    if 'idx' not in st.session_state:
        st.session_state.idx = 0

    c1, c2, c3 = st.columns([1,2,1])
    with c1:
        if st.button("← Previous"):
            st.session_state.idx = max(0, st.session_state.idx - 1)
    with c3:
        if st.button("Next →"):
            st.session_state.idx = min(len(results)-1, st.session_state.idx + 1)
    with c2:
        st.write(f"Image {st.session_state.idx+1} of {len(results)}")

    sel = results[st.session_state.idx]

    # show original
    buf = BytesIO()
    sel["Original"].save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(f'<img src="data:image/png;base64,{b64}" width="300">', unsafe_allow_html=True)

    # show saliency
    overlay_img = overlay_saliency(np.array(sel["Original"]), sel["Saliency"])
    st.image(overlay_img, caption="Saliency Map", width=300)

    # ─── Inject JS for Arrow‐Key → click Prev/Next ─────────────────────────────
    st.markdown(
        """
        <script>
        document.addEventListener('keydown', function(e) {
          if (e.key === 'ArrowLeft') {
            // click first button on page (our "← Previous")
            const btn1 = document.querySelector('button');
            if (btn1) btn1.click();
          }
          if (e.key === 'ArrowRight') {
            // click the second button on page (our "Next →")
            const btns = document.querySelectorAll('button');
            if (btns.length > 1) btns[1].click();
          }
        });
        </script>
        """,
        unsafe_allow_html=True
    )

# ─── Footer: Memory Usage ─────────────────────────────────────────────────────
st.sidebar.write(f"Memory Usage: {monitor_memory_usage():.2f} MB")


