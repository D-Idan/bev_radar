
## Summary

I've implemented both requested features:

### 1. Confidence-Based Track Initiation and Association

**New Parameters Added to `tracker_config`:**
- `min_confidence_init`: Minimum confidence required to start a new track (default: 0.7)
- `min_confidence_assoc`: Minimum confidence required for association (default: 0.4) 
- `confidence_weight`: Weight for confidence in cost calculation (default: 0.3)
- `association_strategy`: Strategy to use (default: "confidence_weighted")

**Four Association Strategies Implemented:**
1. **`distance_only`**: Traditional distance-based association
2. **`confidence_weighted`**: Distance weighted by detection confidence
3. **`confidence_gated`**: Hard confidence threshold with distance
4. **`hybrid_score`**: Sophisticated combination of distance, confidence, Mahalanobis distance, and track quality

**Key Features:**
- Prevents low-confidence detections from starting new tracks
- Uses different confidence thresholds for initiation vs association
- Multiple association strategies for different scenarios
- Configurable parameters exposed through tracker_config

### 2. Centralized Output Structure

**New Directory Structure:**
```
output_dir/
├── tracks/               # CSV tracking results
├── visualizations/
│   ├── frames/          # Individual frame images
│   └── summary/         # Summary plots
├── logs/                # Text summaries and logs
└── config/              # Configuration files
```

**Changes Made:**
- Added `output_dir` parameter to `offline_tracking()`
- Created `setup_output_directories()` function
- Updated all visualization functions to use centralized paths
- Added configuration saving functionality
- All outputs now go to specified subdirectories

**Usage Example:**
```python
# Example with confidence-based tracking
custom_config = {
    'min_confidence_init': 0.8,     # High threshold for new tracks
    'min_confidence_assoc': 0.5,    # Lower for associations
    'association_strategy': 'hybrid_score'  # Most sophisticated
}

offline_tracking(
    preds_csv="predictions.csv",
    labels_csv="labels.csv", 
    output_dir="./my_tracking_results",
    tracker_config=custom_config
)
```

The implementation is backward-compatible and provides sensible defaults while allowing full customization of confidence-based behavior and output organization.

---

A common issue in radar tracking where the Kalman filter can predict tracks outside the sensor's field of view. 
implement range culling to remove tracks that predict positions beyond the radar's coverage area.


## Key Features of Range Culling Implementation:

1. **Configurable Parameters:**
   - `enable_range_culling`: Toggle feature on/off
   - `max_range`, `min_azimuth_deg`, `max_azimuth_deg`: Radar coverage limits
   - `range_buffer`, `azimuth_buffer_deg`: Buffer zones to avoid aggressive culling

2. **Smart Culling Logic:**
   - Checks predicted track positions after Kalman prediction
   - Uses buffer zones to avoid killing tracks that are just barely out of range
   - Only applies to track predictions, not initial detections (those are filtered separately)

3. **Statistics Tracking:**
   - Counts how many tracks were culled for analysis
   - Optional logging of which tracks were culled and why

4. **Multiple Check Points:**
   - Initial detection filtering when creating new tracks
   - Predicted track position checking after Kalman prediction
   - Buffer zones prevent oscillating track creation/deletion

The range culling will help prevent tracks from drifting outside the sensor's field of view and consuming computational resources on impossible track states. The buffer zones provide tolerance for tracking objects near the edge of coverage.


# Gaps
- if the gap is higher than a certain value we kill all tracks because the scene changes completely alter a number of seconds as 
the radar platform is moving (driving car)
- if a track got no update for a certain value of time we kill the track, gaps may cause objects to disappear in the middle of the map and not only at its way to the borders

# Association and network confidence
- we have tries multiple association methods for example mahalanobis distance that dont use the network output confidence and a weighted-confidence method that do use the network output confidence value. gate first (reject impossible associations) then use costs for the remaining valid ones.

# Scenarios documentation
**Scenario:** `RECORD@2020-11-22_12.54.38`
**Frames 001975 - 001991:**
- Using **WEITHGET DISTANCE** leads to a swap in track 1.  
- Using the standard Mahalanobis (without considering confidence) avoids the track swap; however, an extra false track is initiated and then closed.
**Additional Note:**
- Some measurements have low confidence. We plan to develop a new cost function that incorporates both the Mahalanobis distance and confidence—possibly using an adaptive R in the Kalman filter.

## Kalman Filter Parameter Summary for Thesis

### Process Noise (q)
#### Vehicle acceleration scenarios:
- gentle_accel = 1.0   # m/s² (normal driving)
- moderate_accel = 3.0 # m/s² (assertive driving) 
- hard_accel = 6.0     # m/s² (aggressive driving)
- emergency = 10.0     # m/s² (emergency braking)

#### For Different Vehicle Types:
###### Highway vehicles (smooth motion): 
q_highway = 1.0      # σ_a = 1.0 m/s²

###### City vehicles (frequent stops/starts):  
q_city = 4.0         # σ_a = 2.0 m/s²

###### Off-road vehicles (erratic motion):
q_offroad = 9.0      # σ_a = 3.0 m/s²

###### Emergency vehicles (unpredictable):
q_emergency = 16.0   # σ_a = 4.0 m/s²

```python
q = 1.0 to 4.0  # (m²/s⁴)
# Rationale: σ_acceleration = √q = 1-2 m/s²
# Represents typical vehicle acceleration variations
# Lower = smoother tracking, Higher = more responsive to maneuvers
```

### Measurement Noise (R)
```python
R = diag([0.25, 0.25])  # std = 0.5m
# Rationale: Typical automotive radar accuracy ±0.5m
# Diagonal matrix assumes x,y measurement errors are independent
```

### Initial Position Uncertainty
```python
P_init_pos = 4.0  # (2m std dev)
# Rationale: Conservative estimate for first radar detection accuracy
# Not too confident (allows quick adaptation) but not too uncertain
```

### Initial Velocity Uncertainty  
```python
P_init_vel = 25.0  # (5m/s std dev)
# Rationale: Complete uncertainty in initial velocity (start from zero)
# Allows filter to learn actual velocity from subsequent detections
```

### Chi-Square Gating Threshold
```python
chi2_95 = 5.991  # 95% confidence for 2 DOF
# Rationale: Balance between accepting valid associations and rejecting false alarms
# Corresponds to ~2.5σ spatial gate for typical radar accuracy
```

**Key Design Principle**: Parameters reflect **physical constraints** of vehicle dynamics and **sensor characteristics** of automotive radar, tuned for **highway/urban driving scenarios**.


## Adaptive R Matrix Weighting for Kalman Filter Update

We have successfully implemented **confidence-based R matrix weighting** with **independent control** for both association and Kalman filter update phases in our radar tracking system. This provides fine-grained control over how detection confidence affects measurement trust in different parts of the tracking pipeline.

### ✅ **Completed Features:**

1. **Three R Weighting Strategies:**
   - **Squared**: `R = R_base / conf²` (inverse quadratic scaling)
   - **Linear**: `R = R_base × linear_interpolation(R_min_factor, R_max_factor, conf)`
   - **Stepped**: `R = R_base × stepped_factors[conf_range]` (optimized for datasets with many high-confidence detections)

2. **Independent Adaptive R Control:**
   - `use_adaptive_r_in_association`: Controls confidence weighting in Mahalanobis distance gating
   - `use_adaptive_r_in_update`: Controls confidence weighting in Kalman filter update
   - Both flags can be enabled/disabled independently for maximum flexibility

3. **Enhanced Gating Distance Calculation:**
   - `gating_distance()` method now supports optional confidence-weighted R matrix
   - Maintains backward compatibility with standard (unweighted) Mahalanobis distance

4. **Unified Configuration System:**
   - Single `r_weighting_config` shared between association and update phases
   - Consistent strategy parameters across both usage contexts
   - Clean separation of concerns with boolean control flags

### 🔄 **Current Behavior (4 Possible Configurations):**

| Association R | Update R | Use Case |
|---------------|----------|----------|
| Standard | Standard | **Original behavior** - no confidence weighting |
| **Adaptive** | Standard | **Confidence-aware gating** - stricter/looser association based on confidence |
| Standard | **Adaptive** | **Confidence-aware estimation** - measurement trust varies by confidence |
| **Adaptive** | **Adaptive** | **Full adaptive** - confidence affects both association and state estimation |

### 📋 **Configuration Examples:**

```python
# Configuration 1: No adaptive R (original behavior)
config = {
    'association_strategy': 'mahalanobis_distance',
    'use_adaptive_r_in_association': False,
    'use_adaptive_r_in_update': False,
}
```

# Multi-Configuration Tracking System Files

## **New Files Created**

### **Configuration Management**
- **`config/tracking_configurations.yaml`** - Defines tracking configuration variations (baseline, adaptive R settings)
- **`config/__init__.py`** - Package initialization
- **`config/tracking_configuration_manager.py`** - Loads YAML configs, generates configuration variations, validates requests

### **Analysis & Reporting**
- **`utils/metrics/configuration_comparison.py`** - Compares configurations within a single dataset, ranks performance, calculates improvements vs baseline
- **`utils/reports/__init__.py`** - Package initialization  
- **`utils/reports/aggregate_analysis.py`** - Aggregates analysis across multiple datasets, identifies best configurations globally

## **Modified Files**
- **`main_pipeline.py`** - Updated to run multiple tracking configurations, generate comparison reports, integrate new components

## **System Flow**
1. **YAML Config** → defines configuration variations
2. **Configuration Manager** → generates multiple tracking configs  
3. **Main Pipeline** → runs each config, calls comparison analysis
4. **Configuration Comparison** → analyzes single dataset results
5. **Aggregate Analysis** → combines results across all datasets

## **Key Features**
- ✅ **Configurable via YAML** - Easy to add new configurations
- ✅ **Separate outputs** - Each configuration gets its own directory
- ✅ **Comprehensive metrics** - Precision, recall, F1, DetA, distance, IoU
- ✅ **Flexible execution** - Run specific configs or defaults
- ✅ **Detailed reporting** - Per-dataset + aggregate analysis


## changed Algorithm tracking for established tracks

**Problem**: Original algorithm used Mahalanobis distance for all associations, causing failures when objects moved during the first 2 frames (velocity = 0).

**Solution**: Hybrid association strategy:

1. **Frames 1-2**: Use Euclidean distance only (handles movement with zero velocity)
2. **Frame 3+**: Switch to Mahalanobis distance (leverages velocity predictions)

**Implementation**:
- Modified `_create_cost_matrix()` to check `track.hits <= 1` 
- Updated `_is_valid_association()` to use different thresholds based on track maturity
- Distance threshold for early tracks, chi-squared threshold for mature tracks

**Result**: More robust association that prevents track loss during velocity bootstrap phase while maintaining sophisticated tracking for established tracks.
