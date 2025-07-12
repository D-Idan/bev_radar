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
from DBReader.DBReader.SensorsReaders import CANDecoder
from radar_tracking.data_structures import EgoMotion
from utils.odometry.odometry import get_custom_odometry, get_optimized_odometry


def labels_need_update(labels_path, expected_columns):
    """Check if labels.csv exists and has all expected columns"""
    if not os.path.exists(labels_path):
        return True

    try:
        existing_labels = pd.read_csv(labels_path)
        missing_columns = set(expected_columns) - set(existing_labels.columns)
        return len(missing_columns) > 0
    except Exception as e:
        print(f"Error reading existing labels: {e}")
        return True

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

def extract_all(config):
    # Load configuration
    cal_table = config['Calibration']
    labels_df = pd.read_csv(config['label_path'], sep=',')
    output_dir = Path(config['Output_Folder'])
    record = config['target_value']
    root_folder = Path(config['Data_Dir'], record)

    expected_label_columns = [
        'filename', 'timestamp_us', 'ego_steering_wheel_deg',
        'ego_yaw_rate_deg_s', 'ego_speed_kph'
        # Add other expected columns from original labels
    ]
    labels_update_needed = labels_need_update(config['label_path'], expected_label_columns)

    # Prepare output folder structure
    base = os.path.join(output_dir, record)
    subdirs = [
        'ADC_Data', 'camera', 'laser_PCL',
        'radar_FFT', 'radar_Freespace', 'radar_PCL', 'radar_RD', 'radar_RA',
    ]
    ensure_dirs(base, subdirs)

    # Initialize readers and processors
    db = SyncReader(root_folder, tolerance=20000, silent=True)
    RSP_PC = RadarSignalProcessing(cal_table, method='PC', lib='PyTorch')
    RSP_RD = RadarSignalProcessing(cal_table, method='RD', lib='PyTorch')
    RSP_RA = RadarSignalProcessing(cal_table, method='RA', lib='PyTorch')
    RSP_ADC = RadarSignalProcessing(cal_table, method='ADC', lib='PyTorch')

    # Initialize CAN decoder for odometry, Use the DBC file from DBReader examples
    can_dbc_path = Path(__file__).parent / 'DBReader' / 'examples' / 'can_database.dbc'
    assert can_dbc_path.exists(), f"CAN database not found at {can_dbc_path}"
    can_decoder = CANDecoder(str(can_dbc_path))

    # Filter labels for this record
    rec_labels = labels_df[labels_df['dataset'] == record]
    collected = []

    for idx in tqdm(rec_labels['index'].unique(), desc="Processing Samples", unit="sample"):
        sample = db.GetSensorData(int(idx))
        numSample = rec_labels[rec_labels['index'] == idx]['numSample'].iloc[0]
        tag = f"{numSample:06d}"

        # Define all output file paths
        adc_path = os.path.join(base, 'ADC_Data', f'adc_{tag}.npy')
        img_path = os.path.join(base, 'camera', f'image_{tag}.jpg')
        rd_path = os.path.join(base, 'radar_RD', f'rd_{tag}.npy')
        ra_path = os.path.join(base, 'radar_RA', f'ra_{tag}.npy')

        # Check if all files already exist
        files_exist = all(os.path.exists(path) for path in [adc_path, img_path, rd_path, ra_path])

        if files_exist:
            # Extract timestamp (using radar as reference)
            timestamp = sample['radar_ch1']['timestamp']

            # Collect labels for this sample
            boxes = rec_labels[rec_labels['index'] == idx]
            boxes_out = boxes.copy()
            boxes_out['filename'] = tag
            boxes_out['timestamp_us'] = timestamp

            if labels_update_needed:
                # Only extract odometry if labels need updating
                odometry_data = None
                if can_decoder is not None:
                    try:
                        odometry = get_optimized_odometry(db, can_decoder, timestamp)
                        odometry_data = {
                            'steering_wheel_angle_deg': odometry[0],
                            'yaw_rate_deg_per_sec': odometry[1],
                            'speed_kph': odometry[2]
                        }
                    except Exception as e:
                        print(f"Warning: Failed to extract odometry for sample {tag}: {e}")

                # Add odometry data (or NaN if extraction failed)
                if odometry_data is not None:
                    boxes_out['ego_steering_wheel_deg'] = odometry_data['steering_wheel_angle_deg']
                    boxes_out['ego_yaw_rate_deg_s'] = odometry_data['yaw_rate_deg_per_sec']
                    boxes_out['ego_speed_kph'] = odometry_data['speed_kph']
                else:
                    boxes_out['ego_steering_wheel_deg'] = np.nan
                    boxes_out['ego_yaw_rate_deg_s'] = np.nan
                    boxes_out['ego_speed_kph'] = np.nan
            # If labels don't need updating, don't add odometry columns at all
            # (or you could read existing values from the current labels.csv)

            collected.append(boxes_out)
            continue  # Skip to next iteration

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

        # 5. Extract odometry data
        odometry_data = None
        if can_decoder is not None:
            try:
                odometry = get_optimized_odometry(db, can_decoder, timestamp)
                odometry_data = {
                    'steering_wheel_angle_deg': odometry[0],  # Steering wheel angle (degrees)
                    'yaw_rate_deg_per_sec': odometry[1],      # Yaw rate (degrees/sec)
                    'speed_kph': odometry[2]                   # Speed (kph)
                }
            except Exception as e:
                print(f"Warning: Failed to extract odometry for sample {tag}: {e}")
                odometry_data = None


        # collect label entries with timestamp
        boxes = rec_labels[rec_labels['index'] == idx]
        boxes_out = boxes.copy()
        boxes_out['filename'] = tag
        boxes_out['timestamp_us'] = timestamp  # Add timestamp in microseconds

        # Add odometry data to labels
        if odometry_data is not None:
            boxes_out['ego_steering_wheel_deg'] = odometry_data['steering_wheel_angle_deg']
            boxes_out['ego_yaw_rate_deg_s'] = odometry_data['yaw_rate_deg_per_sec']
            boxes_out['ego_speed_kph'] = odometry_data['speed_kph']
        else:
            # Add NaN values if odometry is not available
            boxes_out['ego_steering_wheel_deg'] = np.nan
            boxes_out['ego_yaw_rate_deg_s'] = np.nan
            boxes_out['ego_speed_kph'] = np.nan

        collected.append(boxes_out)

    # write aggregated labels.csv
    if collected:
        all_labels = pd.concat(collected, ignore_index=True)
        # Sort by timestamp to ensure chronological order
        all_labels = all_labels.sort_values('timestamp_us')
        all_labels.to_csv(os.path.join(base, 'labels.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and organize RadIal data')
    parser.add_argument('-c', '--config', default='./data_config.json', type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    extract_all(cfg)