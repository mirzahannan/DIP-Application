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

conf_threshold = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preview using PIL Image to avoid file-like edge cases
    try:
        preview_img = Image.open(uploaded_file).convert("RGB")
        st.image(preview_img, caption="Original", use_container_width=True)
    except Exception:
        st.warning("Could not preview image, but detection can still run.")

    if st.button("Run Detection"):
        with st.spinner("Running detection..."):
            # Write a temporary file only for the detection call
            uploaded_file.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            out_path, labels = run_torchvision_detection(
                image_path=tmp_path,
                output_dir=RESULTS_DIR,
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

        with open(out_path, "rb") as f:
            st.download_button(
                label="Download annotated image",
                data=f.read(),
                file_name=os.path.basename(out_path),
                mime="image/jpeg",
            )
