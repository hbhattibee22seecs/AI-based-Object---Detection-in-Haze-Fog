---
title: Object Detection in Hazy and Foggy Conditions
emoji: 🌫️
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: "1.28.0"
app_file: dashboard.py
pinned: false
license: mit
---

# 🌫️ Object Detection in Hazy and Foggy Conditions

A lightweight dehazing-aware object detection framework using YOLOX trained on the RTTS dataset.

## Features
- Upload images and detect objects in foggy/hazy conditions
- Trained on 50K images from the RTTS dataset
- Detects: person, car, bus, bicycle, motorbike
- Real-time inference with confidence scoring

## Usage
1. Click "Load Model" in the sidebar
2. Upload an image (JPG, JPEG, or PNG)
3. View detection results with bounding boxes
4. Download the processed image

## Model
- Architecture: YOLOX-S
- Training Dataset: RTTS (Realistic Tasks in Traffic Scenarios)
- Classes: 5 object categories optimized for foggy conditions
