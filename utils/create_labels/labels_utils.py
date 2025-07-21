"""Utility functions for the car detection system."""

import os
import re
from typing import List, Tuple
import cv2
import pandas as pd


def extract_sample_number(filename: str) -> int:
    """
    Extract sample number from filename.

    Args:
        filename: Image filename (e.g., "image_006344.jpg")

    Returns:
        Sample number as integer
    """
    match = re.search(r'_(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return -1


def get_sorted_images(folder_path: str, extensions: List[str] = ['.jpg', '.jpeg', '.png']) -> List[str]:
    """
    Get sorted list of image files from folder.
    """
    images = []
    invalid_images = []

    for file in os.listdir(folder_path):
        # SKIP HIDDEN FILES (macOS metadata files)
        if file.startswith('._'):
            continue

        if any(file.lower().endswith(ext) for ext in extensions):
            full_path = os.path.join(folder_path, file)

            # Validate image
            if validate_image(full_path):
                images.append(full_path)
            else:
                invalid_images.append(full_path)

    if invalid_images:
        print(f"WARNING: Found {len(invalid_images)} invalid/corrupted images:")
        for img in invalid_images[:5]:  # Show first 5
            print(f"  - {img}")
        if len(invalid_images) > 5:
            print(f"  ... and {len(invalid_images) - 5} more")

    # Sort by filename
    images.sort(key=lambda x: os.path.basename(x))
    return images

    if invalid_images:
        print(f"WARNING: Found {len(invalid_images)} invalid/corrupted images:")
        for img in invalid_images[:5]:  # Show first 5
            print(f"  - {img}")
        if len(invalid_images) > 5:
            print(f"  ... and {len(invalid_images) - 5} more")

    # Sort by filename
    images.sort(key=lambda x: os.path.basename(x))
    return images


def draw_bounding_boxes(image_path: str, detections: List[dict], output_path: str):
    """
    Draw bounding boxes on image and save.

    Args:
        image_path: Path to input image
        detections: List of detections with bbox and id
        output_path: Path to save the annotated image
    """
    img = cv2.imread(image_path)

    for det in detections:
        x1, y1, x2, y2 = [int(x) for x in det['bbox']]
        track_id = det.get('id', -1)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw ID
        label = f"ID: {track_id}"
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)


def save_results_to_csv(results: List[dict], output_path: str):
    """
    Save detection results to CSV file.

    Args:
        results: List of detection results
        output_path: Path to save CSV file
    """
    df = pd.DataFrame(results)

    # Ensure columns are in the correct order
    columns = ['numSample', 'x1_pix', 'y1_pix', 'x2_pix', 'y2_pix', 'filename', 'ID']

    # Add any missing columns with None values
    for col in columns:
        if col not in df.columns:
            df[col] = None

    # Reorder columns
    df = df[columns]

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def validate_image(image_path: str) -> bool:
    """
    Validate that an image file is readable.

    Args:
        image_path: Path to image file

    Returns:
        True if image is valid, False otherwise
    """
    try:
        img = cv2.imread(image_path)
        return img is not None
    except Exception:
        return False


def merge_cvpr_labels(base_dir, labels_path, root_folder, iou_threshold=0.3):
    """
    Merge CVPR labels with existing labels.csv using IoU matching and timestamps

    Args:
        base_dir: Base directory containing labels.csv
        labels_path: Path to labels_CVPR.csv file
        root_folder: Root folder for SyncReader
        iou_threshold: Minimum IoU threshold for matching bounding boxes
    """
    import pandas as pd
    import numpy as np
    from ADCProcessing.DBReader.DBReader import SyncReader
    from utils.util import bbox_iou
    from shapely.geometry import Polygon

    # Read existing labels.csv
    existing_labels_path = os.path.join(base_dir, 'labels.csv')
    existing_df = pd.read_csv(existing_labels_path)
    dataset_name = existing_df['dataset'].unique().item()

    # Read CVPR labels
    cvpr_df = pd.read_csv(labels_path)
    cvpr_df = cvpr_df[cvpr_df['dataset'] == dataset_name]

    # Initialize SyncReader to get timestamps
    db = SyncReader(root_folder, master='radar', tolerance=20000, silent=True, camera_only=False)


    # Add column to track CVPR updates
    existing_df['cvpr_updated'] = False
    existing_df['cvpr_iou'] = 0.0

    # Create timestamp mapping for CVPR labels
    cvpr_timestamps = {}

    # Get timestamps for CVPR samples
    for _, row in cvpr_df.iterrows():
        idx = row['index']
        try:
            # Extract frame index from CVPR data (assuming it has a frame/sample identifier)
            frame_idx = int(row['numSample'])

            # Get timestamp for this frame
            if idx < len(db):
                sample = db.GetSensorData(idx)
                timestamp = sample['radar_ch1']['timestamp']

                if timestamp not in cvpr_timestamps:
                    cvpr_timestamps[timestamp] = []

                cvpr_timestamps[timestamp].append({
                    'cvpr_frame_idx': frame_idx,
                    'x1_pix': row.get('x1_pix', 0),
                    'y1_pix': row.get('y1_pix', 0),
                    'x2_pix': row.get('x2_pix', 0),
                    'y2_pix': row.get('y2_pix', 0),
                    'radar_R_m': row.get('radar_R_m', 0),
                    'radar_A_deg': row.get('radar_A_deg', 0),
                    'radar_X_m': row.get('radar_X_m', 0),
                    'radar_Y_m': row.get('radar_Y_m', 0),
                    'radar_D_mps': row.get('radar_D_mps', 0),
                    'radar_P_db': row.get('radar_P_db', 0)
                })

        except Exception as e:
            print(f"Error processing CVPR row index {idx}: {e}")
            continue

    # Process existing labels and match with CVPR
    updated_count = 0
    total_matches = 0

    for idx, row in existing_df.iterrows():
        timestamp = row['timestamp_us']

        # Skip rows without bounding box data
        if pd.isna(row['x1_pix']) or pd.isna(row['y1_pix']):
            continue

        # Check if we have CVPR data for this timestamp
        if timestamp in cvpr_timestamps:
            cvpr_detections = cvpr_timestamps[timestamp]

            # Calculate IoU with all CVPR detections at this timestamp
            best_iou = 0.0
            best_match = None

            # Current bounding box from existing labels
            curr_bbox = np.array([
                [row['x1_pix'], row['y1_pix']],  # top-left
                [row['x2_pix'], row['y1_pix']],  # top-right
                [row['x2_pix'], row['y2_pix']],  # bottom-right
                [row['x1_pix'], row['y2_pix']]  # bottom-left
            ]).flatten()

            for cvpr_det in cvpr_detections:
                # CVPR bounding box
                cvpr_bbox = np.array([
                    [cvpr_det['x1_pix'], cvpr_det['y1_pix']],  # top-left
                    [cvpr_det['x2_pix'], cvpr_det['y1_pix']],  # top-right
                    [cvpr_det['x2_pix'], cvpr_det['y2_pix']],  # bottom-right
                    [cvpr_det['x1_pix'], cvpr_det['y2_pix']]  # bottom-left
                ]).flatten()

                # Calculate IoU using existing function
                iou = bbox_iou(curr_bbox, cvpr_bbox.reshape(1, -1))[0]

                if iou > best_iou:
                    best_iou = iou
                    best_match = cvpr_det


            # Update if IoU is above threshold
            if best_iou >= iou_threshold and best_match is not None:
                # Update radar measurements with CVPR data
                existing_df.at[idx, 'radar_R_m'] = best_match['radar_R_m']
                existing_df.at[idx, 'radar_A_deg'] = best_match['radar_A_deg']
                existing_df.at[idx, 'radar_X_m'] = best_match['radar_X_m']
                existing_df.at[idx, 'radar_Y_m'] = best_match['radar_Y_m']
                existing_df.at[idx, 'radar_D_mps'] = best_match['radar_D_mps']
                existing_df.at[idx, 'radar_P_db'] = best_match['radar_P_db']

                # Mark as updated
                existing_df.at[idx, 'cvpr_updated'] = True
                existing_df.at[idx, 'cvpr_iou'] = best_iou

                updated_count += 1

            total_matches += 1

    # Save updated labels
    existing_df.to_csv(existing_labels_path, index=False)

    return updated_count