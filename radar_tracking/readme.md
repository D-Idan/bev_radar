# Radar Tracking System

Comprehensive radar object detection and tracking system

## File Structure
```
radar_tracking/
├── __init__.py
├── data_structures.py      # Data classes for detections and tracks
├── coordinate_transforms.py # Polar to Cartesian conversions
├── metrics.py             # Evaluation metrics using Hungarian algorithm
├── kalman_filter.py       # Kalman filter implementation
├── tracker.py             # Main tracking logic (SORT-like)
├── tracklet_manager.py    # Track management and lifecycle
└── example_usage.py       # Example usage and testing
```

## How to Use the System

Here's a simple example of how to use the complete system:

```python
from radar_tracking import TrackletManager, Detection
import numpy as np

# Create some sample detections
detections = [
    Detection(range_m=100.0, azimuth_rad=0.1, confidence=0.9, timestamp=1.0),
    Detection(range_m=150.0, azimuth_rad=-0.2, confidence=0.8, timestamp=1.0),
]

ground_truth = [
    Detection(range_m=98.0, azimuth_rad=0.12, confidence=1.0, timestamp=1.0),
    Detection(range_m=152.0, azimuth_rad=-0.18, confidence=1.0, timestamp=1.0),
]

# Initialize tracker
manager = TrackletManager()

# Process frame
tracks = manager.update(detections, ground_truth)

# Print results
manager.print_summary()
```

## Key Features

1. **Complete SORT-like Architecture**: Implements prediction, association, and update steps similar to SORT/DeepSORT
2. **Hungarian Algorithm**: Uses scipy.optimize.linear_sum_assignment for optimal detection-track assignment
3. **Kalman Filtering**: Full Kalman filter implementation for state estimation and prediction
4. **Coordinate Transforms**: Proper polar ↔ Cartesian coordinate handling for radar data
5. **Comprehensive Metrics**: Detailed evaluation using Euclidean distances and match ratios
6. **Track Management**: Full lifecycle management with statistics and historical data
7. **Confidence Integration**: Uses detection confidence scores in association costs
8. **Synthetic Data Generation**: Includes realistic test data generation with noise and false alarms

The system is modular, well-documented, and ready for real-world radar tracking applications. You can easily extend it with additional features like track merging, splitting, or more sophisticated motion models.