import os
import json
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import imageio
from tqdm import tqdm
from rpl import RadarSignalProcessing
from DBReader.DBReader import SyncReader
from utils.create_labels.camera_to_RA import calculate_radar_coords

from utils.create_labels.main import process_images_with_tracking, process_images_without_tracking

def create_labels_file(base_dir, frame_data):
    """Create labels.csv file with timestamp, index, and filename info"""
    labels_df = pd.DataFrame(frame_data)
    labels_df = labels_df.sort_values('timestamp_us')
    # Extract dataset name from base_dir path
    dataset_name = Path(base_dir).name
    labels_df['dataset'] = dataset_name
    labels_df.to_csv(os.path.join(base_dir, 'labels.csv'), index=False)

def ensure_dirs(base_dir, subdirs):
    for sub in subdirs:
        path = os.path.join(base_dir, sub)
        os.makedirs(path, exist_ok=True)
    # create labels.csv placeholder
    open(os.path.join(base_dir, 'labels.csv'), 'w').close()

def safe_save_with_retry(save_func, filepath, data, max_retries=3, delay=1):
    """Safely save data with retry logic"""
    for attempt in range(max_retries):
        try:
            save_func(filepath, data)
            return True
        except (OSError, IOError) as e:
            print(f"Attempt {attempt + 1} failed for {filepath}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"Failed to save {filepath} after {max_retries} attempts")
                return False
    return False


def merge_labels(base_dir, car_labels_df):
    """Merge car detection results with existing labels.csv"""

    # Load existing labels
    existing_labels_path = os.path.join(base_dir, 'labels.csv')
    existing_df = pd.read_csv(existing_labels_path)

    # Extract sample numbers from car labels filenames
    car_labels_df['sample_index'] = car_labels_df['filename'].str.extract(r'image_(\d+)\.jpg').astype(int)

    # SCALE BOUNDING BOXES FROM 960x540 TO 1920x1080
    scale_x = 1920 / 960  # 2.0
    scale_y = 1080 / 540  # 2.0

    # Scale all bounding box coordinates
    car_labels_df['x1_pix'] = car_labels_df['x1_pix'] * scale_x
    car_labels_df['y1_pix'] = car_labels_df['y1_pix'] * scale_y
    car_labels_df['x2_pix'] = car_labels_df['x2_pix'] * scale_x
    car_labels_df['y2_pix'] = car_labels_df['y2_pix'] * scale_y

    # Convert to integers
    car_labels_df['x1_pix'] = car_labels_df['x1_pix'].astype('Int64')
    car_labels_df['y1_pix'] = car_labels_df['y1_pix'].astype('Int64')
    car_labels_df['x2_pix'] = car_labels_df['x2_pix'].astype('Int64')
    car_labels_df['y2_pix'] = car_labels_df['y2_pix'].astype('Int64')

    # Apply radar coordinate calculation to car labels (after scaling)
    radar_coords = car_labels_df.apply(calculate_radar_coords, axis=1)
    car_labels_df['radar_R_m'] = radar_coords['radar_R_m']
    car_labels_df['radar_A_deg'] = radar_coords['radar_A_deg']

    # Merge dataframes
    merged_df = existing_df.merge(
        car_labels_df[['sample_index', 'x1_pix', 'y1_pix', 'x2_pix', 'y2_pix', 'ID', 'radar_R_m', 'radar_A_deg']],
        left_on='index',
        right_on='sample_index',
        how='left'
    )

    # Drop the temporary sample_index column
    merged_df = merged_df.drop('sample_index', axis=1)

    # Add the missing columns with placeholder values
    merged_df['laser_X_m'] = 0.0
    merged_df['laser_Y_m'] = 0.0
    merged_df['radar_X_m'] = 0.0
    merged_df['radar_Y_m'] = 0.0
    merged_df['radar_D_mps'] = 0.0
    merged_df['radar_P_db'] = 0.0
    merged_df['dataset'] = Path(base_dir).name
    merged_df['Annotation'] = 'weak'  # default annotation
    merged_df['Difficult'] = 0  # default difficulty

    # Reorder columns to match the expected format
    column_order = [
        'numSample', 'x1_pix', 'y1_pix', 'x2_pix', 'y2_pix',
        'laser_X_m', 'laser_Y_m', 'radar_X_m', 'radar_Y_m', 'radar_R_m',
        'radar_A_deg', 'radar_D_mps', 'radar_P_db', 'dataset', 'index',
        'Annotation', 'Difficult', 'filename', 'timestamp_us'
    ]

    # Add numSample column (same as index)
    merged_df['numSample'] = merged_df['index']

    # Ensure all columns exist
    for col in column_order:
        if col not in merged_df.columns:
            merged_df[col] = None

    # Reorder columns
    merged_df = merged_df[column_order]

    # Fill all NaN values with 0
    merged_df = merged_df.fillna(0)

    # Save merged labels
    merged_df.to_csv(existing_labels_path, index=False)
    print(f"Updated labels.csv with car detection data")

def extract_all(config):
    # Load configuration
    cal_table = config['Calibration']
    output_dir = Path(config['Output_Folder'])
    record = config['target_value']
    root_folder = Path(config['Data_Dir'], record)

    # Prepare output folder structure
    base = os.path.join(output_dir, record)
    subdirs = [
        'ADC_Data', 'camera', 'laser_PCL',
        'radar_FFT', 'radar_Freespace', 'radar_PCL', 'radar_RD', 'radar_RA',
    ]
    ensure_dirs(base, subdirs)

    # Initialize readers and processors
    db = SyncReader(root_folder, master='radar', tolerance=20000, silent=True)
    RSP_RD = RadarSignalProcessing(cal_table, method='RD', lib='PyTorch')
    RSP_RA = RadarSignalProcessing(cal_table, method='RA', lib='PyTorch')
    RSP_ADC = RadarSignalProcessing(cal_table, method='ADC', lib='PyTorch')

    # Process all frames
    frame_data = []
    DEBUG = True
    db_len = 50 if DEBUG else len(db)
    for idx in tqdm(range(db_len), desc="Processing Samples", unit="sample", colour="magenta"):
        sample = db.GetSensorData(idx)
        tag = f"{idx:06d}"

        # Define all output file paths
        adc_path = os.path.join(base, 'ADC_Data', f'adc_{tag}.npy')
        img_path = os.path.join(base, 'camera', f'image_{tag}.jpg')
        rd_path = os.path.join(base, 'radar_RD', f'rd_{tag}.npy')
        ra_path = os.path.join(base, 'radar_RA', f'ra_{tag}.npy')

        # Extract timestamp
        timestamp = sample['radar_ch1']['timestamp']
        # Collect frame metadata - Add these lines
        frame_data.append({
            'timestamp_us': timestamp,
            'index': idx,
            'filename': tag
        })

        # Check if all files already exist - if so, skip processing
        files_exist = all(os.path.exists(path) for path in [adc_path, img_path, rd_path, ra_path])
        if files_exist:
            continue

        # 1. ADC_Data
        if not os.path.exists(adc_path):
            adc = RSP_ADC.run(
                sample['radar_ch0']['data'], sample['radar_ch1']['data'],
                sample['radar_ch2']['data'], sample['radar_ch3']['data']
            )
            if not safe_save_with_retry(np.save, adc_path, adc):
                print(f"Skipping sample {tag} due to save error")
                continue

        # 2. camera -> save jpg
        if not os.path.exists(img_path):
            cam = sample['camera']['data']
            resized = cv2.resize(cam, (960, 540))
            if not safe_save_with_retry(cv2.imwrite, img_path, resized):
                print(f"Skipping sample {tag} due to save error")
                continue

        # 3. radar_RD -> save as NPY
        if not os.path.exists(rd_path):
            rd = RSP_RD.run(sample['radar_ch0']['data'], sample['radar_ch1']['data'],
                            sample['radar_ch2']['data'], sample['radar_ch3']['data']
                            ).numpy()
            rd_map = np.log10(np.sum(np.abs(rd), axis=2) + 1e-6)
            if not safe_save_with_retry(np.save, rd_path, rd_map):
                print(f"Skipping sample {tag} due to save error")
                continue

        # 4. radar_RA -> save as NPY
        if not os.path.exists(ra_path):
            ra = RSP_RA.run(sample['radar_ch0']['data'], sample['radar_ch1']['data'],
                            sample['radar_ch2']['data'], sample['radar_ch3']['data']
                            )
            if not safe_save_with_retry(np.save, ra_path, ra):
                print(f"Skipping sample {tag} due to save error")
                continue

    create_labels_file(base, frame_data)
    print(f"Processed {len(db)} frames")

    print("\nStarting car detection and labeling...")
    try:
        # Path to the camera images folder
        camera_folder = os.path.join(base, 'camera')
        output_csv = os.path.join(base, 'car_labels.csv')

        # Run car detection with tracking
        car_labels_df = process_images_with_tracking(
            input_folder=camera_folder,
            output_csv=output_csv,
            save_viz=False  # This will create visualizations
        )

        # Merge with existing labels.csv
        merge_labels(base, car_labels_df)

        print(f"Car detection complete! Found {car_labels_df['ID'].notna().sum()} detections")

    except Exception as e:
        print(f"Error during car detection: {e}")
        print("Continuing without car labels...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and organize RadIal data')
    parser.add_argument('-c', '--config', default='./data_config.json', type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    extract_all(cfg)