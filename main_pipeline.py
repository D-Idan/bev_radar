"""
Main pipeline script for radar data processing, prediction, and tracking.

This script orchestrates three main steps:
1. Raw data extraction
2. Model prediction
3. Tracking

All steps use direct function calls instead of subprocess calls for better integration.
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path
import traceback

# Add current directory and subdirs to Python path
def setup_python_path():
    """Setup Python path to include all necessary modules."""
    base_path = Path(__file__).parent

    # Add base path
    if str(base_path) not in sys.path:
        sys.path.insert(0, str(base_path))

    # Add subdirectories
    for folder in base_path.iterdir():
        if folder.is_dir() and not folder.name.startswith(('.', '_', '__pycache__')):
            path_str = str(folder)
            if path_str not in sys.path:
                sys.path.append(path_str)

# Setup paths before imports
setup_python_path()

# Now import our modules
from ADCProcessing.raw_data_extractor_all import extract_all
from offline_tracking import offline_tracking


class RadarProcessingPipeline:
    """Main pipeline class for radar data processing with direct function calls."""

    def _load_configurations(self):
        """Load and merge all configuration files."""
        # Load main config
        with open(self.config_path, 'r') as f:
            main_config = yaml.safe_load(f)

        # Load radar/model config if specified
        if 'config_files' in main_config and 'radar_model_config' in main_config['config_files']:
            radar_model_config_path = self.script_dir / main_config['config_files']['radar_model_config']

            if radar_model_config_path.exists():
                with open(radar_model_config_path, 'r') as f:
                    radar_model_config = yaml.safe_load(f)

                # Merge configurations
                main_config.update(radar_model_config)
            else:
                print(f"Warning: Radar/model config file not found: {radar_model_config_path}")

        return main_config

    def __init__(self, config_path: str):
        """
        Initialize the pipeline with YAML configuration.

        Args:
            config_path: Path to the main YAML configuration file
        """
        self.config_path = Path(config_path)
        self.script_dir = Path(__file__).parent

        # Load and merge configurations
        self.config = self._load_configurations()

        # Validate paths
        self._validate_paths()

        # Setup paths
        self.data_dir = Path(self.config['paths']['data_dir'])
        self.output_base = Path(self.config['paths']['output_base'])
        self.model_path = Path(self.config['paths']['model_path'])

    def _validate_paths(self):
        """Validate that required paths exist."""
        data_dir = Path(self.config['paths']['data_dir'])
        model_path = Path(self.config['paths']['model_path'])
        calibration_table = Path(self.config['paths']['calibration_table'])

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not calibration_table.exists():
            raise FileNotFoundError(f"Calibration table not found: {calibration_table}")

    def discover_datasets(self) -> List[str]:
        """Discover all RECORD* datasets in the data directory."""
        datasets = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name.startswith('RECORD'):
                datasets.append(item.name)
        return sorted(datasets)

    def check_step_completed(self, step: str, target_value: str) -> bool:
        """
        Check if a processing step has already been completed.

        Args:
            step: One of 'extraction', 'prediction', 'tracking'
            target_value: Dataset name (e.g., RECORD@2020-11-22_12.45.05)

        Returns:
            True if step is completed, False otherwise
        """
        dataset_output_dir = self.output_base / target_value

        if step == 'extraction':
            # Check if extraction completed - look for labels.csv and key directories
            required_files = [
                dataset_output_dir / 'labels.csv',
                dataset_output_dir / 'ADC_Data',
                dataset_output_dir / 'camera',
                dataset_output_dir / 'radar_RA'
            ]
            return all(path.exists() for path in required_files)

        elif step == 'prediction':
            # Check if predictions completed
            pred_file = dataset_output_dir / 'plots' / 'predictions' / 'all_predictions.csv'
            return pred_file.exists()

        elif step == 'tracking':
            # Check if tracking completed
            tracking_file = dataset_output_dir / 'plots' / 'tracking_output' / 'tracks' / 'tracking.csv'
            return tracking_file.exists()

        return False

    def _create_extraction_config(self, target_value: str) -> dict:
        """Create configuration dict for data extraction."""
        return {
            "Calibration": str(self.config['paths']['calibration_table']),
            "Method": self.config['radar']['method'],
            "label_path": str(self.config['paths']['labels_path']),
            "Data_Dir": str(self.data_dir),
            "Output_Folder": str(self.output_base),
            "target_value": target_value
        }

    def _create_model_config(self, target_value: str) -> dict:
        """Create configuration dict for model prediction."""
        dataset_root = self.output_base / target_value

        return {
            "name": self.config['model']['name'],
            "seed": 3,
            "data_mode": self.config['model']['data_mode'],
            "model": {
                "depths": self.config['model']['depths'],
                "channels": self.config['model']['channels'],
                "patch_size": self.config['model']['patch_size'],
                "in_chans": self.config['model']['in_chans'],
                "embed_dim": self.config['model']['embed_dim'],
                "drop_rates": self.config['model']['drop_rates'],
                "num_heads": self.config['model']['num_heads'],
                "DetectionHead": str(self.config['model']['detection_head']),
                "SegmentationHead": str(self.config['model']['segmentation_head'])
            },
            "dataset": {
                "root_dir": str(dataset_root),
                "geometry": self.config['radar']['geometry'],
                "statistics": self.config['radar']['statistics']
            }
        }

    def _create_tracking_config(self, target_value: str) -> dict:
        """Create configuration dict for tracking."""
        dataset_root = self.output_base / target_value

        # Safety check
        if 'radar' not in self.config:
            raise KeyError("'radar' configuration section missing from config")
        if 'tracking' not in self.config:
            raise KeyError("'tracking' configuration section missing from config")

        # Build tracker config from YAML
        tracker_config = {
            # Track lifecycle parameters
            'max_age': self.config['tracking']['lifecycle']['max_age'],
            'min_hits': self.config['tracking']['lifecycle']['min_hits'],
            'max_velocity_ms': self.config['tracking']['lifecycle']['max_velocity_ms'],

            # Timing parameters
            'base_dt': self.config['tracking']['timing']['base_dt'],
            'max_dt_gap': self.config['tracking']['timing']['max_dt_gap'],
            'max_time_without_update': self.config['tracking']['timing']['max_time_without_update'],
            'max_frame_gap_time': self.config['tracking']['timing']['max_frame_gap_time'],

            # Association strategy
            'association_strategy': self.config['tracking']['association']['strategy'],
            'min_confidence_init': self.config['tracking']['association']['min_confidence_init'],
            'min_confidence_assoc': self.config['tracking']['association']['min_confidence_assoc'],
            'confidence_weight': self.config['tracking']['association']['confidence_weight'],

            # Adaptive R matrix weighting
            'use_adaptive_r_in_association': self.config['tracking']['adaptive_r']['use_in_association'],
            'use_adaptive_r_in_update': self.config['tracking']['adaptive_r']['use_in_update'],
            'r_weighting_strategy': self.config['tracking']['adaptive_r']['weighting_strategy'],
            'r_weighting_config': self.config['tracking']['adaptive_r']['config'],

            # Mahalanobis distance parameters
            'default_chi2_threshold': self.config['tracking']['mahalanobis']['default_chi2_threshold'],

            # Range culling parameters
            'enable_range_culling': self.config['tracking']['range_culling']['enable'],
            'max_range': self.config['radar']['coverage']['max_range'],
            'min_azimuth_deg': self.config['radar']['coverage']['min_azimuth_deg'],
            'max_azimuth_deg': self.config['radar']['coverage']['max_azimuth_deg'],
            'range_buffer': self.config['tracking']['range_culling']['range_buffer'],
            'azimuth_buffer_deg': self.config['tracking']['range_culling']['azimuth_buffer_deg'],

            # Kalman filter parameters
            'kalman_config': {
                'process_noise_q_std': self.config['tracking']['kalman']['process_noise_q_std'],
                'measurement_noise_std': self.config['tracking']['kalman']['measurement_noise_std'],
                'initial_pos_std': self.config['tracking']['kalman']['initial_pos_std'],
                'initial_vel_std': self.config['tracking']['kalman']['initial_vel_std'],
            },

            # Evaluation parameters
            'max_distance_threshold': self.config['tracking']['evaluation']['max_distance_threshold'],
        }

        return {
            'preds_csv': str(dataset_root / 'plots' / 'predictions' / 'all_predictions.csv'),
            'labels_csv': str(dataset_root / 'labels.csv'),
            'output_dir': str(dataset_root / 'plots' / 'tracking_output'),
            'tracker_config': tracker_config,
            'create_video': self.config['tracking']['create_video'],
            'max_video_samples': self.config['tracking']['max_video_samples'],
            'max_frames': self.config['tracking']['max_frames'],
        }

    def run_extraction(self, target_value: str) -> bool:
        """
        Run the raw data extraction step using direct function call.

        Args:
            target_value: Dataset name to process

        Returns:
            True if successful, False otherwise
        """
        print(f"  Running data extraction for {target_value}...")

        try:
            # Create extraction config
            extraction_config = self._create_extraction_config(target_value)

            # Call extraction function directly
            extract_all(extraction_config)

            print(f"    Extraction completed successfully")
            return True

        except Exception as e:
            print(f"    Extraction failed: {e}")
            print("    Full traceback:")
            traceback.print_exc()
            return False

    def run_prediction(self, target_value: str) -> bool:
        """
        Run the model prediction step by calling the existing script's main function.

        Args:
            target_value: Dataset name to process

        Returns:
            True if successful, False otherwise
        """
        print(f"  Running model prediction for {target_value}...")

        try:
            # Import the prediction script's main function
            from T_FFTRadNet_predictions import main as run_prediction_main

            # Get model config
            model_config = self._create_model_config(target_value)
            dataset_root = self.output_base / target_value

            # Create output directory for predictions
            predictions_dir = dataset_root / 'plots' / 'predictions'
            predictions_dir.mkdir(parents=True, exist_ok=True)

            # Call the main function directly
            run_prediction_main(
                config=model_config,
                checkpoint_filename=str(self.model_path),
                difficult=True,
                output_dir=str(predictions_dir)
            )

            print(f"    Prediction completed successfully")
            return True

        except Exception as e:
            print(f"    Prediction failed: {e}")
            print("    Full traceback:")
            traceback.print_exc()
            return False

    def run_tracking(self, target_value: str) -> bool:
        """
        Run the offline tracking step using direct function call.

        Args:
            target_value: Dataset name to process

        Returns:
            True if successful, False otherwise
        """
        print(f"  Running offline tracking for {target_value}...")

        try:
            # Create tracking config
            tracking_config = self._create_tracking_config(target_value)

            # Verify input files exist
            predictions_csv = Path(tracking_config['preds_csv'])
            labels_csv = Path(tracking_config['labels_csv'])

            if not predictions_csv.exists():
                print(f"    Predictions file not found: {predictions_csv}")
                return False

            if not labels_csv.exists():
                print(f"    Labels file not found: {labels_csv}")
                return False

            # Call tracking function directly
            offline_tracking(**tracking_config)

            print(f"    Tracking completed successfully")
            return True

        except Exception as e:
            print(f"    Tracking failed: {e}")
            print("    Full traceback:")
            traceback.print_exc()  # This will show the full traceback
            return False

    def process_dataset(self, target_value: str, skip_existing: bool = True) -> bool:
        """
        Process a single dataset through the entire pipeline.

        Args:
            target_value: Dataset name to process
            skip_existing: If True, skip steps that are already completed

        Returns:
            True if all steps completed successfully, False otherwise
        """
        print(f"\nProcessing dataset: {target_value}")
        print("-" * 50)

        # Step 1: Data Extraction
        # if skip_existing and self.check_step_completed('extraction', target_value):
        if self.check_step_completed('extraction', target_value):
            print("  Data extraction already completed, skipping...")
        else:
            if not self.run_extraction(target_value):
                print(f"  Failed to extract data for {target_value}")
                return False

        # Step 2: Model Prediction
        # if skip_existing and self.check_step_completed('prediction', target_value):
        if self.check_step_completed('prediction', target_value):
            print("  Model prediction already completed, skipping...")
        else:
            if not self.run_prediction(target_value):
                print(f"  Failed to run prediction for {target_value}")
                return False

        # Step 3: Offline Tracking
        if skip_existing and self.check_step_completed('tracking', target_value):
            print("  Offline tracking already completed, skipping...")
        else:
            if not self.run_tracking(target_value):
                print(f"  Failed to run tracking for {target_value}")
                return False

        print(f"  ✓ All steps completed successfully for {target_value}")
        return True

    def run_pipeline(self, target_datasets: Optional[List[str]] = None,
                     skip_existing: bool = True) -> None:
        """
        Run the complete pipeline for specified datasets or all available datasets.

        Args:
            target_datasets: List of specific datasets to process, or None for all
            skip_existing: If True, skip steps that are already completed
        """
        # Discover available datasets
        available_datasets = self.discover_datasets()

        if not available_datasets:
            print("No RECORD* datasets found in the data directory.")
            return

        # Determine which datasets to process
        if target_datasets is None:
            datasets_to_process = available_datasets
            print(f"Found {len(available_datasets)} datasets to process:")
            for dataset in available_datasets:
                print(f"  - {dataset}")
        else:
            # Validate requested datasets exist
            missing_datasets = [d for d in target_datasets if d not in available_datasets]
            if missing_datasets:
                print(f"Warning: The following datasets were not found: {missing_datasets}")

            datasets_to_process = [d for d in target_datasets if d in available_datasets]
            print(f"Processing {len(datasets_to_process)} specified datasets:")
            for dataset in datasets_to_process:
                print(f"  - {dataset}")

        if not datasets_to_process:
            print("No valid datasets to process.")
            return

        # Process each dataset
        successful = 0
        failed = 0

        for dataset in datasets_to_process:
            if self.process_dataset(dataset, skip_existing):
                successful += 1
            else:
                failed += 1

        # Summary
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Total datasets processed: {len(datasets_to_process)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")

        if failed > 0:
            print(f"\nSome datasets failed to process completely.")
        else:
            print(f"\n✓ All datasets processed successfully!")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='Radar data processing pipeline: extraction → prediction → tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all datasets with default config
  python main_pipeline.py

  # Process specific dataset  
  python main_pipeline.py --target RECORD@2020-11-22_12.45.05

  # Process multiple specific datasets
  python main_pipeline.py --target RECORD@2020-11-22_12.45.05 RECORD@2020-11-22_08.45.18

  # Force reprocess all steps (skip existing = False)
  python main_pipeline.py --no-skip-existing

  # Use custom config file
  python main_pipeline.py --config custom_config.yaml
        """
    )

    parser.add_argument(
        '--config',
        default='config/pipeline_config.yaml',
        help='Path to YAML configuration file (default: config/pipeline_config.yaml)'
    )

    parser.add_argument(
        '--target',
        nargs='*',
        help='Specific dataset(s) to process. If not specified, all RECORD* datasets will be processed.'
    )

    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Do not skip existing outputs - reprocess all steps'
    )

    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List available datasets and exit'
    )

    args = parser.parse_args()

    try:
        # Initialize pipeline with config
        pipeline = RadarProcessingPipeline(config_path=args.config)

        # List datasets if requested
        if args.list_datasets:
            datasets = pipeline.discover_datasets()
            print(f"Available datasets in {pipeline.data_dir}:")
            for dataset in datasets:
                print(f"  - {dataset}")
            return

        # Run pipeline
        skip_existing = not args.no_skip_existing
        pipeline.run_pipeline(
            target_datasets=args.target,
            skip_existing=skip_existing
        )

    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

# Usage Examples
# ============================

# # Process all datasets
# python main_pipeline.py
#
# # Process specific dataset
# python main_pipeline.py --target RECORD@2020-11-22_12.45.05
#
# # List available datasets
# python main_pipeline.py --list-datasets
#
# # Force reprocess everything
# python main_pipeline.py --no-skip-existing
#
# # Use custom paths
# python main_pipeline.py --data-dir /path/to/data --output-dir /path/to/output