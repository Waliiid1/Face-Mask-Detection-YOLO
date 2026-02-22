import streamlit as st
from PIL import Image
import numpy as np
import os
import ultralytics
import sys
import subprocess

# Force-install ultralytics if missing
try:
    import ultralytics
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    import ultralytics

from utils.detector import YOLOModel
from utils.visualization import draw_boxes

st.set_page_config(page_title="YOLOv8 Mask Detection", layout="centered")

MODEL_PATH = os.path.join("model", "best.pt")
DEMO_IMAGE = os.path.join("assets", "demo.png")

@st.cache_resource
def load_model():
    return YOLOModel(MODEL_PATH)

model = load_model()

st.title("😷 Face Mask Detection – YOLOv8")
st.markdown("Upload an image or use the demo example.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
else:
    image = Image.open(DEMO_IMAGE).convert("RGB")

st.image(image, caption="Input Image", use_container_width=True)

if st.button("Run Detection"):
    with st.spinner("Running YOLOv8 inference..."):
        results = model.predict(image)
        output = draw_boxes(image, results)

    st.image(output, caption="Detection Result", use_container_width=True)