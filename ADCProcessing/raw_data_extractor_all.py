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

def create_labels_file(base_dir, frame_data):
    """Create labels.csv file with timestamp, index, and filename info"""
    labels_df = pd.DataFrame(frame_data)
    labels_df = labels_df.sort_values('timestamp_us')
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

    for idx in tqdm(range(len(db)), desc="Processing Samples", unit="sample", colour="magenta"):
        sample = db.GetSensorData(idx)
        tag = f"{idx:06d}"

        # Define all output file paths
        adc_path = os.path.join(base, 'ADC_Data', f'adc_{tag}.npy')
        img_path = os.path.join(base, 'camera', f'image_{tag}.jpg')
        rd_path = os.path.join(base, 'radar_RD', f'rd_{tag}.npy')
        ra_path = os.path.join(base, 'radar_RA', f'ra_{tag}.npy')

        # Check if all files already exist - if so, skip processing
        files_exist = all(os.path.exists(path) for path in [adc_path, img_path, rd_path, ra_path])
        if files_exist:
            continue

        # Extract timestamp
        timestamp = sample['radar_ch1']['timestamp']
        # Collect frame metadata - Add these lines
        frame_data.append({
            'timestamp_us': timestamp,
            'index': idx,
            'filename': tag
        })

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and organize RadIal data')
    parser.add_argument('-c', '--config', default='./data_config.json', type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    extract_all(cfg)