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
from PIL import ImageDraw, ImageFont
import numpy as np
from PIL import Image
import io
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info
from tools.demo import Predictor

# RTTS Classes
RTTS_CLASSES = ("person", "car", "bus", "bicycle", "motorbike")

# Color palette for visualizations
_COLORS = np.array([
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
])

def vis_custom(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    """Custom visualization with improved label placement."""
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id % len(_COLORS)] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id % len(_COLORS)]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.5, 1)[0]
        
        # Draw bounding box
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        # Place label above the box with padding
        label_y = max(y0 - 10, txt_size[1] + 10)  # Ensure label stays within image
        txt_bk_color = (_COLORS[cls_id % len(_COLORS)] * 255 * 0.8).astype(np.uint8).tolist()
        
        cv2.rectangle(
            img,
            (x0, label_y - txt_size[1] - 8),
            (x0 + txt_size[0] + 8, label_y + 2),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0 + 4, label_y - 4), font, 0.5, txt_color, thickness=1)

    return img

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
        
        # Convert RGB to BGR for the model (OpenCV format expected)
        img_bgr = img_array[:, :, ::-1].copy()

        # Run inference with BGR image
        with st.spinner("Running inference..."):
            outputs, img_info = st.session_state.predictor.inference(img_bgr)

        # Use custom visualization with improved label placement
        result_img_bgr = img_info["raw_img"].copy()
        if outputs[0] is not None:
            output = outputs[0].cpu()
            bboxes = output[:, 0:4]
            # Scale bboxes back to original size
            bboxes /= img_info["ratio"]
            scores = output[:, 4] * output[:, 5]
            cls_ids = output[:, 6]
            result_img_bgr = vis_custom(result_img_bgr, bboxes, scores, cls_ids, conf_threshold, RTTS_CLASSES)
        
        # Convert BGR back to RGB for display
        result_img = Image.fromarray(result_img_bgr[:, :, ::-1])
        
        # Collect detections for display
        detections = []
        if outputs[0] is not None:
            for output in outputs[0]:
                score = (output[4] * output[5]).cpu().numpy()
                cls_id = int(output[6].cpu().numpy())

                if score > conf_threshold:
                    bbox = output[:4].cpu().numpy()
                    ratio = img_info["ratio"]
                    bbox /= ratio  # Scale back to original image size
                    detections.append({
                        'class': RTTS_CLASSES[cls_id],
                        'confidence': float(score),
                        'bbox': bbox.tolist()
                    })

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