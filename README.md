# Auto_Annotator

Auto_Annotator is an AI-powered offline annotation tool specifically designed for Windows. It provides a seamless, efficient, and interactive environment for image and video data annotation, accelerating the process of creating datasets for computer vision models, specifically YOLOv8.

## Features

- **Offline YOLOv8 Support:** Completely offline detection and segmentation using base YOLOv8 models (`n`, `s`, `m`, `l`, `x`, `n-seg`, `s-seg`).
- **Interactive UI:** A highly responsive Dark Theme GUI built with CustomTkinter, featuring intuitive controls, real-time image rendering, and hardware-accelerated interfaces.
- **Smart Train & Propagate:** Easily train custom models directly within the app on a small subset of manually annotated data (10-15 images) and automatically propagate these annotations to the rest of your dataset.
- **Manual Annotation:** Fast click-and-drag bounding box drawing, class categorization, and easy modification or removal of labels.
- **Seamless Video Support:** Extract and annotate frames directly from video files at your chosen frame rate.
- **YOLO Dataset Export/Import:** Native support for YOLO format labels. Export your annotated datasets easily and import existing ones to resume progress.
- **Batch Processing:** Clear annotations, manage lists, and delete selected files efficiently in batches.

## Prerequisites

Ensure you have the following dependencies installed in your environment:

- Python 3.8+
- `ultralytics`
- `customtkinter`
- `opencv-python`
- `Pillow`
- `numpy`

## Usage

1. **Start the Application:**
   Run `main.py` to start the GUI.
   ```bash
   python src/main.py
   ```
2. **Load Data:** Use the Sidebar to open individual images, a folder of images, or extract frames from a video.
3. **Select Model:** Choose your preferred base YOLOv8 model from the dropdown. The model will download on the first run if you are online, but will remain strictly offline subsequently.
4. **Annotate:**
   - **Manual:** Add classes and drag on the canvas to draw bounding boxes.
   - **Auto:** Adjust confidence, let the AI suggest bounding boxes, and confirm or remove them via the detection list.
5. **Train Custom Model:** Once you have a few manual annotations, go to "Smart Train & Propagate" to train a custom model and automatically annotate remaining images.
6. **Export:** Export your finalized dataset to standard YOLO formatting, ready for full-scale training.

## Integrity

This project contains an integrity check integrated directly into its runtime source to ensure authenticity and proper functioning. This check evaluates silently and must remain intact to verify proper execution.

## License

All rights reserved. Use within authorized scope only.
