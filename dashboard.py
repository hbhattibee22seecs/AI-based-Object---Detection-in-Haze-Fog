#!/usr/bin/env python3
"""
Streamlit Dashboard for YOLOX Object Detection Inference
Upload an image and get object detection results with bounding boxes.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import torch
from PIL import ImageDraw
import numpy as np
from PIL import Image
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info
from tools.demo import Predictor

# RTTS Classes
RTTS_CLASSES = ("person", "car", "bus", "bicycle", "motorbike")

def load_model(exp_file, ckpt_file, device="cpu", fuse=True):
    """Load YOLOX model from experiment and checkpoint."""
    exp = get_exp(exp_file)
    model = exp.get_model()
    model.eval()

    if device == "gpu":
        model.cuda()
    elif device == "cpu":
        model.cpu()

    # Load checkpoint
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    if fuse:
        model = fuse_model(model)

    return model, exp

def main():
    st.title("🌫️ YOLOX Object Detection Dashboard")
    st.markdown("Upload an image to detect objects in foggy/hazy conditions using the trained RTTS model.")

    # Sidebar for configuration
    st.sidebar.header("Model Configuration")
    exp_file = st.sidebar.text_input(
        "Experiment File",
        value="src/exps/example/custom/rtts_yolox_s.py",
        help="Path to the experiment configuration file"
    )
    ckpt_file = st.sidebar.text_input(
        "Checkpoint File",
        value="trained_models/last_epoch_ckpt.pth",
        help="Path to the trained model checkpoint"
    )
    device = st.sidebar.selectbox("Device", ["cpu", "gpu"], index=0)
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
    nms_threshold = st.sidebar.slider("NMS Threshold", 0.0, 1.0, 0.45, 0.01)

    # Load model button
    if st.sidebar.button("Load Model"):
        try:
            with st.spinner("Loading model..."):
                model, exp = load_model(exp_file, ckpt_file, device)
                predictor = Predictor(
                    model=model,
                    exp=exp,
                    cls_names=RTTS_CLASSES,
                    device=device
                )
                st.session_state.predictor = predictor
                st.sidebar.success("Model loaded successfully!")
                st.sidebar.text(f"Model: {get_model_info(model, exp.test_size)}")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")

    # Main content
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


    if uploaded_file is not None and 'predictor' in st.session_state:
        # Read image
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        # Run inference (assuming predictor.inference accepts RGB numpy array)
        with st.spinner("Running inference..."):
            outputs, img_info = st.session_state.predictor.inference(img_array)

        # Draw bounding boxes using PIL
        result_img = image.copy()
        draw = ImageDraw.Draw(result_img)
        detections = []
        if outputs[0] is not None:
            for output in outputs[0]:
                bbox = output[:4].cpu().numpy()
                score = (output[4] * output[5]).cpu().numpy()
                cls_id = int(output[6].cpu().numpy())

                if score > conf_threshold:
                    detections.append({
                        'class': RTTS_CLASSES[cls_id],
                        'confidence': float(score),
                        'bbox': bbox.tolist()
                    })
                    # Draw rectangle
                    x0, y0, x1, y1 = bbox
                    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                    draw.text((x0, y0), f"{RTTS_CLASSES[cls_id]}: {score:.2f}", fill="red")

        st.image(result_img, caption="Detection Results", use_column_width=True)

        # Show detections
        if detections:
            st.subheader("Detected Objects")
            for det in detections:
                st.write(f"**{det['class']}** - Confidence: {det['confidence']:.2f} - BBox: {det['bbox']}")
        else:
            st.info("No objects detected above the confidence threshold.")

        # Download button
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Download Result Image",
            data=byte_im,
            file_name="detection_result.png",
            mime="image/png"
        )

    elif uploaded_file is not None:
        st.warning("Please load the model first using the sidebar.")

if __name__ == "__main__":
    main()