# Object Detection Web App (YOLO + Flask)

This project provides:
- A Flask web app to upload an image and run object detection.
- Pretrained YOLO inference (Ultralytics) with saved annotated results.
- A training script to fine-tune YOLO on your custom dataset and save the model.
- A Streamlit app as an alternative UI for easy local deployment.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Flask web app

```powershell
python app.py
```
Then open `http://127.0.0.1:5000` and upload an image.

## Run the Streamlit app

```powershell
streamlit run streamlit_app.py
```
Optionally set custom weights:

```powershell
$env:YOLO_WEIGHTS="runs/train\custom\weights\best.pt"
streamlit run streamlit_app.py
```

## Train on your data

Prepare your dataset in YOLO format and a `data.yaml`:

```yaml
train: path/to/train/images
val: path/to/val/images
nc: 3
names: ["class1", "class2", "class3"]
```

Run training:

```powershell
python train.py --data path\to\data.yaml --epochs 50 --imgsz 640 --model yolov8n.pt --project runs/train --name custom
```

Best weights will be at `runs/train/custom/weights/best.pt`.

## Use a trained model for inference

Set `YOLO_WEIGHTS` env var or update `MODEL_WEIGHTS` in `app.py`/`streamlit_app.py` to your trained weights path.

