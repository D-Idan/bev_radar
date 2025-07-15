"""Configuration settings for the car detection system."""

import os
from pathlib import Path

# Paths
# INPUT_FOLDER = Path("path/to/your/images")  # Change this to your image folder
# INPUT_FOLDER = Path("/Volumes/ELEMENTS/datasets/radial/RadIal_Data/RECORD@2020-11-21_11.54.31/camera")  # Change this to your image folder
INPUT_FOLDER = Path("/mnt/data/datasets/radial/gd/raw_data/RadIal_Data/RECORD@2020-11-21_11.54.31/camera")  # Change this to your image folder
OUTPUT_CSV = INPUT_FOLDER / "new_labels.csv"

# Model settings
MODEL_NAME = "yolov8m.pt"
# Other options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt (slower but more accurate)

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Classes to detect (COCO dataset class IDs)
CAR_CLASSES = [2, 5, 7]  # 2: car, 5: bus, 7: truck
# Add or remove classes as needed

# Tracking settings
TRACKER_TYPE = "botsort.yaml"  # Built-in tracker
# Other options: "botsort.yaml", "bytetrack.yaml"

# Output settings
SAVE_VISUALIZATION = True  # Save images with bounding boxes
VISUALIZATION_FOLDER = "output_visualizations"