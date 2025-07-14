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