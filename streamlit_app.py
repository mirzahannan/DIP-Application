from __future__ import annotations

import os
import tempfile
from typing import List

import streamlit as st
from PIL import Image

from detect_torchvision import run_torchvision_detection, ensure_dir_exists

st.set_page_config(page_title="Object Detection", page_icon="🤖", layout="centered")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(APP_ROOT, "static", "results")
ensure_dir_exists(RESULTS_DIR)

st.title("Object Detection (TorchVision)")
st.write("Upload an image. The app will detect objects and list their names.")

conf_threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.6, 0.05)
min_area_ratio = st.slider("Min Box Area (% of image)", 0.0, 5.0, 1.0, 0.1) / 100.0
max_detections = st.slider("Max Detections", 1, 50, 12, 1)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        uploaded_file.seek(0)
        preview_img = Image.open(uploaded_file).convert("RGB")
        st.image(preview_img, caption="Original", use_column_width=True)
    except Exception as e:
        st.warning("Could not preview image; proceeding to detection.")

    try:
        with st.spinner("Running detection..."):
            uploaded_file.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            out_path, labels = run_torchvision_detection(
                image_path=tmp_path,
                output_dir=RESULTS_DIR,
                confidence_threshold=float(conf_threshold),
                min_area_ratio=float(min_area_ratio),
                max_detections=int(max_detections),
            )

        st.success("Done!")

        with open(out_path, "rb") as f:
            detected_bytes = f.read()
        st.image(detected_bytes, caption="Detected", use_column_width=True)

        unique_labels: List[str] = sorted(set(labels))
        if unique_labels:
            st.subheader("Detected Labels")
            st.write(", ".join(unique_labels))
        else:
            st.info("No labels detected.")

        st.download_button(
            label="Download annotated image",
            data=detected_bytes,
            file_name=os.path.basename(out_path),
            mime="image/jpeg",
        )
    except Exception as e:
        st.error("Detection failed.")
        st.exception(e)
