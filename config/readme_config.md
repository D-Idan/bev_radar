# Radar Tracking Configuration Guide

This document provides a comprehensive guide to configuring the radar tracking system using the YAML configuration file (`tracking_config.yaml`).

## Table of Contents

- [Radar Tracking Configuration Guide](#radar-tracking-configuration-guide)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Configuration Structure](#configuration-structure)
  - [Configuration Parameters](#configuration-parameters)
    - [Track Lifecycle Parameters](#track-lifecycle-parameters)
    - [Timing Parameters](#timing-parameters)
    - [Data Association Strategy](#data-association-strategy)
    - [Confidence-Based Parameters](#confidence-based-parameters)
    - [Adaptive Measurement Noise (R Matrix) Configuration](#adaptive-measurement-noise-r-matrix-configuration)
    - [Mahalanobis Distance Parameters](#mahalanobis-distance-parameters)
    - [Radar Coverage Parameters](#radar-coverage-parameters)
    - [Kalman Filter Parameters](#kalman-filter-parameters)
    - [Evaluation Parameters](#evaluation-parameters)
    - [External Configuration Imports](#external-configuration-imports)
    - [Visualization and Output Parameters](#visualization-and-output-parameters)
    - [Experimental Features](#experimental-features)
  - [Usage Examples](#usage-examples)
    - [Basic Usage](#basic-usage)
    - [Override Specific Parameters](#override-specific-parameters)
    - [Multiple Configuration Profiles](#multiple-configuration-profiles)
  - [Performance Tuning Guide](#performance-tuning-guide)
    - [High Frame Rate (\>15 FPS)](#high-frame-rate-15-fps)
    - [Low Frame Rate (\<5 FPS)](#low-frame-rate-5-fps)
    - [High-Noise Environment](#high-noise-environment)
    - [Real-Time Performance](#real-time-performance)
  - [Troubleshooting](#troubleshooting)
    - [Too Many False Positive Tracks](#too-many-false-positive-tracks)
    - [Missing True Tracks](#missing-true-tracks)
    - [Tracks Lost Too Quickly](#tracks-lost-too-quickly)
    - [Poor Association Performance](#poor-association-performance)
    - [Memory Usage Too High](#memory-usage-too-high)

## Overview

The radar tracking system uses a hierarchical YAML configuration file that organizes parameters into logical groups. This approach provides:

- **Clear documentation** with inline comments explaining each parameter
- **Modular configuration** allowing fine-tuning of specific subsystems
- **External imports** to merge existing JSON configurations
- **Version control friendly** format for tracking configuration changes

## Configuration Structure

The configuration is organized into the following main sections:

```yaml
track_lifecycle:     # Track creation, confirmation, and deletion
timing:             # Frame timing and temporal behavior  
association:        # Detection-to-track matching strategy
confidence:         # Confidence-based filtering and weighting
adaptive_noise:     # Dynamic measurement noise adjustment
mahalanobis:        # Statistical gating parameters
radar_coverage:     # Sensor coverage and range culling
kalman_filter:      # Motion model and filtering parameters
evaluation:         # Performance evaluation settings
external_configs:   # Import external JSON configurations
output:            # Visualization and output control
experimental:      # Advanced/experimental features
```

## Configuration Parameters

### Track Lifecycle Parameters

Controls when tracks are created, confirmed, and deleted.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_age` | int | 3 | Maximum consecutive frames without detection before track deletion |
| `min_hits` | int | 3 | Minimum detections required for track confirmation |
| `max_velocity_ms` | float | 83.3 | Maximum expected object velocity (m/s) for validation |

**Tuning Guidelines:**
- **Higher `max_age`**: More robust to missed detections, but slower response to disappeared objects
- **Lower `max_age`**: Faster track deletion, less memory usage
- **Higher `min_hits`**: Fewer false positive tracks, but slower confirmation
- **Lower `min_hits`**: Faster track confirmation, better for fast-moving objects

### Timing Parameters

Controls how the tracker handles time gaps and frame timing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_dt` | float | 0.2 | Base time step in seconds (200ms = 5 FPS) |
| `max_dt_gap` | float | 1.0 | Time gap threshold for multi-step prediction |
| `max_time_without_update` | float | 1.0 | Maximum time (seconds) before track deletion |
| `max_frame_gap_time` | float | 2.0 | Maximum frame gap before clearing all tracks |

**Use Cases:**
- **High frame rate** (>10 FPS): Reduce `base_dt` to 0.1s or lower
- **Low frame rate** (<5 FPS): Increase `base_dt` to 0.5s or higher
- **Irregular timing**: Increase `max_dt_gap` and `max_time_without_update`

### Data Association Strategy

Choose how detections are matched to existing tracks.

| Strategy | Description | Best For |
|----------|-------------|----------|
| `distance_only` | Pure Euclidean distance | Simple scenarios, fast processing |
| `confidence_weighted` | Distance weighted by confidence | When confidence scores are reliable |
| `confidence_gated` | Confidence filtering + distance | High-noise environments |
| `hybrid_score` | Combined distance and confidence | Balanced performance |
| `mahalanobis_distance` | Statistical distance with uncertainty | Most robust, recommended |

**Example:**
```yaml
association:
  strategy: "mahalanobis_distance"  # Recommended for best performance
```

### Confidence-Based Parameters

Control how detection confidence affects tracking behavior.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_confidence_init` | float | 0.5 | Minimum confidence to create new tracks |
| `min_confidence_assoc` | float | 0.2 | Minimum confidence for track association |
| `confidence_weight` | float | 0.3 | Weight factor in hybrid strategies |

**Confidence Thresholds:**
- **High precision**: `min_confidence_init: 0.7`, `min_confidence_assoc: 0.4`
- **High recall**: `min_confidence_init: 0.3`, `min_confidence_assoc: 0.1`
- **Balanced**: `min_confidence_init: 0.5`, `min_confidence_assoc: 0.2`

### Adaptive Measurement Noise (R Matrix) Configuration

Controls whether detection confidence modulates Kalman filter behavior.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_adaptive_r_in_association` | bool | false | Use confidence-weighted R in data association |
| `use_adaptive_r_in_update` | bool | false | Use confidence-weighted R in Kalman updates |
| `r_weighting_strategy` | string | "linear" | Strategy: "linear", "squared", "stepped" |

**R Weighting Configuration:**
```yaml
r_weighting_config:
  r_min_factor: 0.3    # High confidence: 30% of base noise
  r_max_factor: 15.0   # Low confidence: 15x base noise
  stepped_r_thresholds: [0.95, 0.85, 0.75, 0.60, 0.40]
  stepped_r_factors: [0.1, 0.3, 0.7, 1.5, 4.0, 20.0]
```

**When to Enable:**
- **Enable** when confidence scores are well-calibrated and reliable
- **Disable** when confidence scores are unreliable or poorly calibrated

### Mahalanobis Distance Parameters

Statistical gating parameters for association (when using `mahalanobis_distance` strategy).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_chi2_threshold` | float | 4.605 | Chi-squared threshold for gating |

**Common Thresholds:**
- **4.605**: 90% confidence (more permissive associations)
- **5.991**: 95% confidence (balanced)
- **6.635**: 99% confidence (more strict associations)

### Radar Coverage Parameters

Define radar sensor capabilities and automatic track culling.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_range_culling` | bool | true | Automatically delete tracks outside coverage |
| `max_range` | float | 103.0 | Maximum radar detection range (meters) |
| `min_azimuth_deg` | float | -90.0 | Minimum azimuth angle (degrees) |
| `max_azimuth_deg` | float | 90.0 | Maximum azimuth angle (degrees) |
| `range_buffer` | float | 10.0 | Range buffer for track deletion (meters) |
| `azimuth_buffer_deg` | float | 5.0 | Azimuth buffer for track deletion (degrees) |

**Configuration for Different Radars:**
```yaml
# Forward-looking automotive radar
radar_coverage:
  max_range: 200.0
  min_azimuth_deg: -60.0
  max_azimuth_deg: 60.0

# 360-degree surveillance radar  
radar_coverage:
  max_range: 1000.0
  min_azimuth_deg: -180.0
  max_azimuth_deg: 180.0
```

### Kalman Filter Parameters

Core filtering parameters for motion prediction and state estimation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `process_noise_q_std` | float | 1.0 | Process noise standard deviation |
| `measurement_noise_std` | float | 0.5 | Base measurement noise std (meters) |
| `initial_pos_std` | float | 2.0 | Initial position uncertainty (meters) |
| `initial_vel_std` | float | 5.0 | Initial velocity uncertainty (m/s) |

**Tuning Guidelines:**
- **Higher `process_noise_q_std`**: Better for unpredictable motion, more responsive
- **Lower `process_noise_q_std`**: Better for smooth motion, more stable
- **`measurement_noise_std`**: Should match your radar's accuracy specification

### Evaluation Parameters

Parameters for tracking performance evaluation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_distance_threshold` | float | 2.0 | Max distance for valid track-ground truth association |

### External Configuration Imports

Import and merge existing JSON configuration files.

```yaml
external_configs:
  import_paths:
    - "config/ADC_config.json"
    - "ADCProcessing/data_config.json"
  merge_strategy: "yaml_priority"  # "yaml_priority", "json_priority", "merge_deep"
```

**Merge Strategies:**
- **`yaml_priority`**: YAML values override JSON values
- **`json_priority`**: JSON values override YAML values  
- **`merge_deep`**: Deep merge with YAML taking precedence

### Visualization and Output Parameters

Control visualization and output generation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `create_video` | bool | true | Whether to create tracking visualization video |
| `max_video_samples` | int/null | null | Maximum samples in video (null = all) |
| `max_frames` | int/null | null | Maximum frames to process (null = all) |
| `save_frame_images` | bool | true | Save individual frame visualizations |
| `generate_summary_plots` | bool | true | Generate summary plots and statistics |

### Experimental Features

Advanced options for research and experimentation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_constant_velocity_model` | bool | true | Use constant velocity motion model |
| `enable_track_merging` | bool | false | Enable track merging (experimental) |
| `enable_track_splitting` | bool | false | Enable track splitting (experimental) |
| `use_global_nearest_neighbor` | bool | false | Use global nearest neighbor assignment |
| `use_multiple_hypothesis_tracking` | bool | false | Use MHT algorithm (experimental) |

## Usage Examples

### Basic Usage
```python
from offline_tracking import offline_tracking

# Use default configuration
offline_tracking(
    preds_csv="predictions.csv",
    labels_csv="labels.csv",
    config_file="config/tracking_config.yaml"
)
```

### Override Specific Parameters
```python
# Create custom config that overrides YAML values
custom_config = {
    "max_age": 5,
    "min_hits": 2,
    "association_strategy": "distance_only"
}

offline_tracking(
    preds_csv="predictions.csv", 
    labels_csv="labels.csv",
    tracker_config=custom_config  # Takes precedence over YAML
)
```

### Multiple Configuration Profiles
```yaml
# config/highway_tracking.yaml - Optimized for highway scenarios
track_lifecycle:
  max_age: 5          # Higher persistence for highway
  min_hits: 2         # Faster confirmation
  max_velocity_ms: 120.0  # Higher speed limit

# config/urban_tracking.yaml - Optimized for urban scenarios  
track_lifecycle:
  max_age: 2          # Lower persistence for city
  min_hits: 4         # More conservative confirmation
  max_velocity_ms: 60.0   # Lower speed limit
```

## Performance Tuning Guide

### High Frame Rate (>15 FPS)
```yaml
timing:
  base_dt: 0.067      # ~15 FPS
  max_dt_gap: 0.5
  
track_lifecycle:
  max_age: 5          # Allow more missed frames
  min_hits: 3
```

### Low Frame Rate (<5 FPS)
```yaml
timing:
  base_dt: 0.5        # 2 FPS
  max_dt_gap: 2.0
  max_time_without_update: 3.0

track_lifecycle:
  max_age: 2          # Fewer allowed missed frames
  min_hits: 2
```

### High-Noise Environment
```yaml
confidence:
  min_confidence_init: 0.7    # Require high confidence
  min_confidence_assoc: 0.4
  
mahalanobis:
  default_chi2_threshold: 5.991  # More strict gating

adaptive_noise:
  use_adaptive_r_in_update: true  # Trust high-confidence more
  r_weighting_strategy: "stepped"
```

### Real-Time Performance
```yaml
association:
  strategy: "distance_only"      # Fastest association
  
output:
  create_video: false           # Disable video creation
  save_frame_images: false     # Disable frame saving
  generate_summary_plots: false # Disable summary plots

experimental:
  use_global_nearest_neighbor: false  # Disable expensive algorithms
```

## Troubleshooting

### Too Many False Positive Tracks
**Symptoms:** Many short-lived tracks, tracks on noise
**Solutions:**
```yaml
confidence:
  min_confidence_init: 0.7  # Increase threshold
track_lifecycle:
  min_hits: 4              # Require more confirmations
```

### Missing True Tracks  
**Symptoms:** Real objects not being tracked
**Solutions:**
```yaml
confidence:
  min_confidence_init: 0.3  # Lower threshold
track_lifecycle:
  min_hits: 2              # Faster confirmation
  max_age: 5               # More persistence
```

### Tracks Lost Too Quickly
**Symptoms:** Valid tracks disappearing prematurely
**Solutions:**
```yaml
track_lifecycle:
  max_age: 5               # Allow more missed detections
timing:
  max_time_without_update: 2.0  # Allow longer gaps
```

### Poor Association Performance
**Symptoms:** Detections associated with wrong tracks
**Solutions:**
```yaml
association:
  strategy: "mahalanobis_distance"  # Use statistical gating
mahalanobis:
  default_chi2_threshold: 5.991    # Stricter gating
```

### Memory Usage Too High
**Symptoms:** System running out of memory
**Solutions:**
```yaml
track_lifecycle:
  max_age: 2               # Delete tracks faster
output:
  save_frame_images: false # Reduce output files
  max_frames: 1000        # Limit processing
```

---

For more detailed information about specific algorithms and implementation details, refer to the source code documentation and research papers referenced in the codebase.