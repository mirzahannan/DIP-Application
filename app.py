from __future__ import annotations

import os
from datetime import datetime
from typing import List

from flask import Flask, render_template, request, redirect, url_for

from detect import run_yolo_detection, ensure_dir_exists

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_ROOT, "static", "uploads")
RESULTS_DIR = os.path.join(APP_ROOT, "static", "results")

# Default model weights; can be overridden by YOLO_WEIGHTS env var
MODEL_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")

ensure_dir_exists(UPLOAD_DIR)
ensure_dir_exists(RESULTS_DIR)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect_route():
    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    # Save upload with timestamp to avoid collisions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"upload_{timestamp}_{file.filename}"
    upload_path = os.path.join(UPLOAD_DIR, filename)
    file.save(upload_path)

    # Run detection
    output_path, class_names = run_yolo_detection(
        image_path=upload_path,
        output_dir=RESULTS_DIR,
        model_weights_path=MODEL_WEIGHTS,
        confidence_threshold=0.25,
    )

    # Build URLs for template
    upload_url = url_for("static", filename=f"uploads/{filename}")
    result_filename = os.path.basename(output_path)
    result_url = url_for("static", filename=f"results/{result_filename}")

    # Unique, sorted labels
    unique_labels: List[str] = sorted(set(class_names))

    return render_template(
        "index.html",
        upload_url=upload_url,
        result_url=result_url,
        labels=unique_labels,
    )


if __name__ == "__main__":
    app.run(debug=True)

