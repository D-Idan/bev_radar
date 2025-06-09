"""
Radar Object Detection and Tracking System

A comprehensive system for tracking radar objects using Kalman filtering,
Hungarian algorithm for association, and SORT-like tracking architecture.

Key Components:
- Detection and Track data structures
- Coordinate transformations (polar ↔ Cartesian)
- Hungarian algorithm-based metrics evaluation
- Kalman filter for object state estimation
- SORT-like tracker with association logic
- Tracklet manager for lifecycle management

Usage:
    from radar_tracking import TrackletManager, Detection

    # Initialize tracker
    manager = TrackletManager()

    # Process detections
    tracks = manager.update(detections, ground_truth)

    # Get results
    manager.print_summary()
"""

from .data_structures import Detection, Track, TrackingResult
from .coordinate_transforms import (
    polar_to_cartesian,
    cartesian_to_polar,
    batch_polar_to_cartesian,
    euclidean_distance
)
from .metrics import RadarMetrics
from .kalman_filter import RadarKalmanFilter
from .tracker import RadarTracker
from .tracklet_manager import TrackletManager, TrackletStatistics

__version__ = "1.0.0"
__author__ = "Radar Tracking System"

__all__ = [
    # Data structures
    'Detection',
    'Track',
    'TrackingResult',
    'TrackletStatistics',

    # Coordinate transforms
    'polar_to_cartesian',
    'cartesian_to_polar',
    'batch_polar_to_cartesian',
    'euclidean_distance',

    # Core components
    'RadarMetrics',
    'RadarKalmanFilter',
    'RadarTracker',
    'TrackletManager',
]