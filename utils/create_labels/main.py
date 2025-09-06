"""Main script for car detection and tracking."""

import os
import sys
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

from utils.create_labels.config import *
from utils.create_labels.detector import CarDetector
from utils.create_labels.labels_tracker import CarTracker
from utils.create_labels.labels_utils import (
    extract_sample_number,
    get_sorted_images,
    draw_bounding_boxes,
    save_results_to_csv
)


def process_images_with_tracking(
        input_folder: str,
        output_csv: str,
        model_name: str = MODEL_NAME,
        car_classes: List[int] = CAR_CLASSES,
        save_viz: bool = SAVE_VISUALIZATION
) -> pd.DataFrame:
    """
    Process all images in folder with tracking.

    Args:
        input_folder: Path to image folder
        output_csv: Path to output CSV file
        model_name: YOLO model to use
        car_classes: Classes to detect
        save_viz: Whether to save visualizations

    Returns:
        DataFrame with results
    """
    # Get sorted image list
    print("Loading images...")
    image_paths = get_sorted_images(input_folder)

    if not image_paths:
        print(f"No images found in {input_folder}")
        return pd.DataFrame()

    print(f"Found {len(image_paths)} images")

    # Initialize tracker
    print("Initializing tracker...")
    tracker = CarTracker(
        model_name=model_name,
        tracker_config=TRACKER_TYPE,
        confidence=CONFIDENCE_THRESHOLD
    )

    # Process images with tracking
    print("Processing images with tracking...")
    tracked_results = tracker.track_batch(image_paths, car_classes)

    # Create visualization folder if needed
    if save_viz:
        os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

    # Format results for CSV
    all_results = []

    for image_path, detections in tracked_results.items():
        filename = os.path.basename(image_path)
        sample_num = extract_sample_number(filename)

        # Save visualization if requested
        if save_viz:
            viz_path = os.path.join(VISUALIZATION_FOLDER, filename)
            if detections:
                draw_bounding_boxes(image_path, detections, viz_path)
            else:
                # Save original image without any annotations
                import shutil
                shutil.copy2(image_path, viz_path)

        # Add each detection as a row
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            all_results.append({
                'numSample': sample_num,
                'x1_pix': int(x1),
                'y1_pix': int(y1),
                'x2_pix': int(x2),
                'y2_pix': int(y2),
                'filename': filename,
                'ID': det.get('id', -1)
            })

        # If no detections, add empty row for this image
        if not detections:
            all_results.append({
                'numSample': sample_num,
                'x1_pix': None,
                'y1_pix': None,
                'x2_pix': None,
                'y2_pix': None,
                'filename': filename,
                'ID': None
            })

    # Save to CSV
    save_results_to_csv(all_results, output_csv)

    return pd.DataFrame(all_results)


def process_images_without_tracking(
        input_folder: str,
        output_csv: str,
        model_name: str = MODEL_NAME,
        car_classes: List[int] = CAR_CLASSES,
        save_viz: bool = SAVE_VISUALIZATION
) -> pd.DataFrame:
    """
    Process images without tracking (detection only).
    Each detected car gets a unique ID based on order.
    """
    # Get sorted image list
    print("Loading images...")
    image_paths = get_sorted_images(input_folder)

    if not image_paths:
        print(f"No images found in {input_folder}")
        return pd.DataFrame()

    print(f"Found {len(image_paths)} images")

    # Initialize detector
    print("Initializing detector...")
    detector = CarDetector(
        model_name=model_name,
        confidence=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD
    )

    # Create visualization folder if needed
    if save_viz:
        os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

    # Process images
    all_results = []
    global_id = 1  # Global ID counter

    print("Processing images...")
    for image_path in tqdm(image_paths):
        filename = os.path.basename(image_path)
        sample_num = extract_sample_number(filename)

        # Detect cars
        detections = detector.detect(image_path, car_classes)

        # Process detections
        viz_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']

            all_results.append({
                'numSample': sample_num,
                'x1_pix': int(x1),
                'y1_pix': int(y1),
                'x2_pix': int(x2),
                'y2_pix': int(y2),
                'filename': filename,
                'ID': global_id
            })

            viz_detections.append({
                'bbox': det['bbox'],
                'id': global_id
            })

            global_id += 1

        # Save visualization if requested
        if save_viz:
            viz_path = os.path.join(VISUALIZATION_FOLDER, filename)
            if viz_detections:
                draw_bounding_boxes(image_path, viz_detections, viz_path)
            else:
                # Save original image without any annotations
                import shutil
                shutil.copy2(image_path, viz_path)

        # If no detections, add empty row
        if not detections:
            all_results.append({
                'numSample': sample_num,
                'x1_pix': None,
                'y1_pix': None,
                'x2_pix': None,
                'y2_pix': None,
                'filename': filename,
                'ID': None
            })

    # Save to CSV
    save_results_to_csv(all_results, output_csv)

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    # Choose processing mode
    use_tracking = True  # Set to False for detection only

    # Update the input folder path
    # INPUT_FOLDER = "path/to/your/images"  # CHANGE THIS

    if use_tracking:
        print("Processing with tracking (consistent IDs across frames)...")
        df = process_images_with_tracking(INPUT_FOLDER, OUTPUT_CSV)
    else:
        print("Processing without tracking (unique ID per detection)...")
        df = process_images_without_tracking(INPUT_FOLDER, OUTPUT_CSV)

    # Display summary
    print(f"\nProcessing complete!")
    print(f"Total rows: {len(df)}")
    print(f"Unique samples: {df['numSample'].nunique()}")
    print(f"Detected objects: {df['ID'].notna().sum()}")

    if use_tracking:
        print(f"Unique tracked IDs: {df['ID'].nunique()}")