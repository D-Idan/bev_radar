"""
Centralized visualization configuration for radar tracking system.
Ensures consistent visual representation across all plots.
"""

import numpy as np


class VisualizationConfig:
    """Unified visualization configuration for all tracking plots."""

    # ===================
    # MARKER CONFIGURATION
    # ===================

    # Network output/detections (blue circles)
    DETECTIONS = {
        'color': 'blue',
        'marker': 'o',
        'size_base': 50,  # Base size for fixed-size markers
        'size_range': (50, 150),  # Min/max for confidence-scaled sizing
        'alpha_base': 0.4,
        'alpha_range': (0.4, 0.8),  # Min/max for confidence-scaled alpha
        'edgecolor': 'darkblue',
        'linewidth': 1,
        'label': 'Network Output',
        'camera_bbox_color': (0, 0, 255),  # BGR for OpenCV
        'camera_bbox_thickness': 3
    }

    # Ground truth/labels (green X)
    GROUND_TRUTH = {
        'color': 'green',
        'marker': 'x',
        'size': 100,
        'alpha': 0.9,
        'linewidth': 3,
        'label': 'Ground Truth',
        'camera_bbox_color': (0, 255, 0),  # BGR for OpenCV
        'camera_bbox_thickness': 2
    }

    # Track positions - confirmed tracks (red triangles)
    TRACKS = {
        'color': 'red',
        'marker': '^',
        'size': 50,
        'alpha': 0.9,
        'facecolors': 'none',
        'edgecolors': 'red',
        'linewidth': 0.8,
        'label': 'Track Position',
        'camera_bbox_color': (255, 0, 0),  # BGR for OpenCV
        'camera_bbox_thickness': 1,
        # Track state-specific colors
        'associated': {
            'text_color': 'darkred',
            'edge_color': 'red',
            'bg_color': 'white'
        },
        'unassociated': {
            'text_color': 'darkorange',
            'edge_color': 'orange',
            'bg_color': 'lightyellow'
        }
    }

    # Predictions (orange circles with specific style)
    PREDICTIONS = {
        'color': 'orange',
        'marker': 'o',
        'size': 15,
        'alpha': 0.7,
        'facecolor': 'orange',
        'edgecolor': 'darkorange',
        'label': 'Prediction',
        'linestyle': '--',  # For temporal plots
        'linewidth': 2
    }

    # Tentative tracks (purple diamonds)
    TENTATIVE_TRACKS = {
        'color': 'purple',
        'marker': 'D',
        'size': 60,
        'alpha': 0.8,
        'facecolors': 'none',
        'edgecolors': 'purple',
        'linewidth': 1.5,
        'label': 'Tentative Track',
        'text_color': 'purple',
        'bg_color': 'lavender',
        'edge_color': 'purple'
    }

    # Rejected/filtered items
    REJECTED = {
        'color': 'gray',
        'marker': 's',
        'size': 30,
        'alpha': 0.3,
        'text_color': 'gray',
        'bg_color': 'lightgray',
        'edge_color': 'gray'
    }

    # New track initialization
    NEW_TRACK = {
        'color': 'green',
        'marker': 'o',
        'size': 100,
        'facecolors': 'none',
        'edgecolors': 'green',
        'linewidth': 2,
        'alpha': 0.8,
        'text_color': 'green',
        'bg_color': 'lightgreen'
    }

    # Rejected track initialization
    REJECTED_INIT = {
        'color': 'orange',
        'marker': 'o',
        'size': 80,
        'linestyle': '--',
        'linewidth': 1.5,
        'alpha': 0.6,
        'text_color': 'orange',
        'bg_color': 'lightyellow',
        'edge_color': 'orange'
    }

    # ===================
    # CONFIDENCE VISUALIZATION
    # ===================

    # Confidence color bins (5 distinct blues)
    CONFIDENCE_COLORS = ['#c6dbef', '#9ecae1', '#6baed6', '#3182bd', '#08519c']
    CONFIDENCE_LABELS = ['0.0-0.4', '0.4-0.6', '0.6-0.8', '0.8-0.9', '0.9-1.0']
    CONFIDENCE_BINS = [0, 0.4, 0.6, 0.8, 0.9, 1.0]

    # ===================
    # ELLIPSE CONFIGURATION
    # ===================

    # Chi-square confidence ellipses
    CHI2_95 = {
        'threshold': 5.991,
        'color': 'red',
        'alpha': 0.15,
        'linestyle': '-',
        'linewidth': 1,
        'label': 'χ² 95%'
    }

    CHI2_99 = {
        'threshold': 9.210,
        'color': 'orange',
        'alpha': 0.12,
        'linestyle': '--',
        'linewidth': 1.5,
        'label': 'χ² 99%'
    }

    CHI2_99_9 = {
        'threshold': 13.816,
        'color': 'purple',
        'alpha': 0.1,
        'linestyle': ':',
        'linewidth': 2,
        'label': 'χ² 99.9%'
    }

    # ===================
    # COVERAGE & BOUNDS
    # ===================

    COVERAGE_BOUNDS = {
        'color': 'gray',
        'linestyle': '--',
        'alpha': 0.5,
        'linewidth': 2
    }

    OUT_OF_COVERAGE = {
        'color': 'red',
        'alpha': 0.1
    }

    # ===================
    # TEXT ANNOTATION CONFIGURATION
    # ===================

    TEXT_ANNOTATION = {
        'fontsize': {
            'default': 11,
            'zoomed': 10,
            'small': 6,
            'track_flag': 9
        },
        'offset': {
            'default': {'azimuth': 2.5, 'range': 8.0},
            'zoomed': {'azimuth': 0.5, 'range': 2.5}
        },
        'bbox_style': 'round,pad=0.3',
        'arrow_props': {
            'arrowstyle': '->',
            'connectionstyle': 'arc3,rad=0.1',
            'linewidth': 0.8,
            'alpha': 0.6
        }
    }

    # ===================
    # TIME GAP VISUALIZATION
    # ===================

    TIME_GAP = {
        'threshold': 0.5,  # seconds
        'color': 'red',
        'alpha': 0.15,
        'annotation_bg': 'yellow',
        'annotation_alpha': 0.8
    }

    # ===================
    # 3D PLOT CONFIGURATION
    # ===================

    PLOT_3D = {
        'track_linewidth': 2.5,
        'track_marker_size': 4,
        'prediction_linewidth': 1.5,
        'prediction_marker_size': 3,
        'detection_size': 15,
        'detection_alpha': 0.5,
        'ground_truth_size': 40,
        'ground_truth_alpha': 0.7
    }

    # ===================
    # TEMPORAL PLOT CONFIGURATION
    # ===================

    TEMPORAL = {
        'update_marker_size': 80,      # Larger since no lines (was 5)
        'prediction_marker_size': 60,  # Larger since no lines (was 4)
        'detection_scatter_size': 50,  # Slightly larger (was 30)
        'raw_confidence_size': 15,
        'raw_confidence_alpha': 0.4,
        'smoothed_marker_size': 3,
        'uncertainty_color': 'gray',
        'uncertainty_alpha': 0.5,
        'uncertainty_linewidth': 1
    }

    # ===================
    # RADAR MAP CONFIGURATION
    # ===================

    RADAR_COLORMAP = 'viridis'

    # ===================
    # ZOOM CONFIGURATION
    # ===================

    ZOOM_BORDER = {
        'azimuth': 2.0,  # degrees
        'range': 5.0  # meters
    }

    # ===================
    # ASSOCIATION DISTANCE THRESHOLDS
    # ===================

    ASSOCIATION_DISTANCE = {
        'general_threshold': 2.0,  # meters - for general association
        'azimuth_threshold': 5.0,  # degrees - for polar distance filtering
        'range_threshold': 1.0,  # meters - for polar distance filtering
    }