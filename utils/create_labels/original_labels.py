import pandas as pd
import cv2
import os
from pathlib import Path

from tqdm import tqdm

from utils.create_labels.config import *


def get_scale_factor(image, target_width=1920, target_height=1080):
    """Calculate scale factors for image coordinates."""
    height, width = image.shape[:2]
    scale_w = width / target_width
    scale_h = height / target_height
    return scale_w, scale_h


def save_cvpr_labeled_images(labels_csv_path, image_folder_path, output_folder_path):
    """
    Save images with original CVPR bounding boxes (cvpr_updated=FALSE).
    """
    # Load labels
    df = pd.read_csv(labels_csv_path)

    # Filter for CVPR labels only
    cvpr_labels = df[df['cvpr_updated'] == True]
    no_cvpr_labels = df[df['cvpr_updated'] == False]

    # Create output directory
    os.makedirs(output_folder_path, exist_ok=True)

    # Group by filename to process each image once
    for filename_index, group in tqdm(cvpr_labels.groupby('filename'), desc="Processing cvpr_labels images"):
        # Construct actual filename from index
        actual_filename = f"image_{filename_index:06d}.jpg"

        # Load image
        image_path = os.path.join(image_folder_path, actual_filename)
        image = cv2.imread(image_path)

        # Get scale factors
        scale_w, scale_h = get_scale_factor(image)

        # Draw bounding boxes for this image
        for _, row in group.iterrows():
            # Scale coordinates to actual image size
            x1 = int(row['x1_pix'] * scale_w)
            y1 = int(row['y1_pix'] * scale_h)
            x2 = int(row['x2_pix'] * scale_w)
            y2 = int(row['y2_pix'] * scale_h)

            # Draw green bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save labeled image
        output_path = os.path.join(output_folder_path, actual_filename)
        cv2.imwrite(output_path, image)

    # Group by filename to process each image once
    for filename_index, group in tqdm(no_cvpr_labels.groupby('filename'), desc="Processing no_cvpr_labels images"):
        # Construct actual filename from index
        actual_filename = f"image_{filename_index:06d}.jpg"

        # Load image
        image_path = os.path.join(image_folder_path, actual_filename)
        image = cv2.imread(image_path)

        # Save labeled image
        output_path = os.path.join(output_folder_path, actual_filename)
        cv2.imwrite(output_path, image)


if __name__ == "__main__":
    labels_csv = LABELS_FILE
    image_folder = INPUT_FOLDER
    output_folder = "./cvpr_labeled_images"

    save_cvpr_labeled_images(labels_csv, image_folder, output_folder)