# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a thesis work focused on improving radar-based object detection performance by applying Kalman filter tracking over deep learning network outputs. The key research contribution is handling variable time intervals in real-world radar data (RadIal dataset) where frame timestamps are not constant, unlike synthetic datasets with fixed time steps.

The system combines detection and tracking to achieve better overall performance than detection alone. The codebase includes:

1. **T_FFTRadNet**: A Vision Transformer-based neural network for radar object detection using FFT features
2. **Radar Tracking System**: A SORT-like tracking implementation with Kalman filtering
3. **ADC Processing**: Tools for reading and processing raw radar data using the DBReader library
4. **Visualization Suite**: Comprehensive visualization tools for tracking results and performance analysis

## Key Architecture Components

### Neural Network Pipeline (T_FFTRadNet/)
- `FFTRadNet_ViT.py`: Main Vision Transformer model combining FFT processing with Swin Transformer
- `fourier_net.py`: FFT feature extraction network
- `dataset/`: Custom dataset loaders for radar data with specific train/validation splits
- `encoder.py`: Label encoding for radar detection tasks

### Tracking System (radar_tracking/)
- `tracker.py`: Main SORT-like tracking logic with Hungarian algorithm assignment
- `kalman_filter.py`: Kalman filter implementation for state estimation
- `tracklet_manager.py`: Complete track lifecycle management
- `data_structures.py`: Detection and Track data classes
- `coordinate_transforms.py`: Polar ↔ Cartesian coordinate conversions

### Data Processing (ADCProcessing/)
- `DBReader/`: Library for reading dataset sequences (install with `pip install .`)
- Supports both ASyncReader (temporal order) and SyncReader (synchronized) modes
- `raw_data_extractor_all.py`: Raw data processing pipeline

## Common Development Commands

### Installation
```bash
# Install DBReader library
cd ADCProcessing/DBReader && pip install .

# Uninstall if needed
pip uninstall db-reader
```

### Running Tracking
```python
# Main offline tracking function
from offline_tracking import offline_tracking

offline_tracking(
    preds_csv="predictions.csv",
    labels_csv="labels.csv", 
    output_dir="./tracking_results",
    tracker_config=custom_config
)
```

### Key Configuration Parameters

**Tracker Configuration:**
- `max_age`: Track death threshold (default: 3)
- `min_hits`: Minimum detections for track confirmation (default: 3)
- `min_confidence_init`: Confidence threshold for new tracks (default: 0.7)
- `min_confidence_assoc`: Confidence threshold for associations (default: 0.4)
- `association_strategy`: Options: 'distance_only', 'confidence_weighted', 'confidence_gated', 'hybrid_score'
- `enable_range_culling`: Remove tracks outside radar coverage (default: True)
- `max_range`: Radar maximum range (default: 103.0m)

**Output Structure:**
All tracking outputs are organized in centralized directories:
- `tracks/`: CSV tracking results
- `visualizations/frames/`: Individual frame images
- `visualizations/summary/`: Summary plots
- `logs/`: Text summaries and performance logs
- `config/`: Configuration files

### Notebook Examples
- `ADCProcessing/DBReader/examples/`: Example notebooks for data reading
- `ADCProcessing/radar processing.ipynb`: Radar data processing pipeline

## Code Organization Notes

- **Variable Time Handling**: The Kalman filter uses actual timestamp differences (`dt`) from the RadIal dataset rather than assuming constant frame rates
- All coordinate transformations handle polar radar coordinates properly
- Confidence-based tracking supports multiple association strategies
- Range culling prevents tracks from drifting outside sensor coverage
- Visualization functions are modular and support batch processing
- Track evaluation uses Hungarian algorithm for optimal assignment
- The system maintains backward compatibility with configuration defaults

## Research Context

This thesis investigates whether post-processing deep learning detection outputs with tracking algorithms improves overall radar object detection performance. The RadIal dataset provides real-world automotive radar data with variable frame timing, making it ideal for testing tracking robustness in practical scenarios where constant frame rates cannot be assumed.

## Algorithmic Documentation Notes

- All algorithmic implementations, techniques, and issues for the thesis paper should be documented in the `readme_offline_tracker.md` file to aid in thesis and paper writing