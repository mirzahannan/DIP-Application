from __future__ import annotations

import os
import tempfile
from typing import List

import streamlit as st

from detect import run_yolo_detection, ensure_dir_exists

st.set_page_config(page_title="YOLO Object Detection", page_icon="🤖", layout="centered")

# Directories for results
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(APP_ROOT, "static", "results")
ensure_dir_exists(RESULTS_DIR)

# Model weights from env or default
MODEL_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")

st.title("YOLO Object Detection")
st.write("Upload an image. The app will detect objects and list their names.")

weights_input = st.text_input(
    "Model Weights (optional)",
    value=MODEL_WEIGHTS,
    help="Path to .pt file (leave as yolov8n.pt for COCO pre-trained)",
)

conf_threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.image(tmp_path, caption="Original", use_container_width=True)

    if st.button("Run Detection"):
        with st.spinner("Running YOLO detection..."):
            out_path, labels = run_yolo_detection(
                image_path=tmp_path,
                output_dir=RESULTS_DIR,
                model_weights_path=weights_input or MODEL_WEIGHTS,
                confidence_threshold=float(conf_threshold),
            )
        st.success("Done!")
        st.image(out_path, caption="Detected", use_container_width=True)

        unique_labels: List[str] = sorted(set(labels))
        if unique_labels:
            st.subheader("Detected Labels")
            st.write(", ".join(unique_labels))
        else:
            st.info("No labels detected.")

        st.download_button(
            label="Download annotated image",
            data=open(out_path, "rb").read(),
            file_name=os.path.basename(out_path),
            mime="image/jpeg",
        )
