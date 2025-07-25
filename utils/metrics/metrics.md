# Tracking and Detection Metrics

This implementation evaluates tracking performance using the following metrics:

## Primary Metrics

### 1. **HOTA (Higher Order Tracking Accuracy)**
- Combines detection and association accuracy
- Formula: `sqrt(DetA × AssA)`
- Provides a balanced measure of tracking performance
- Uses camera bounding boxes with all labels

### 2. **MOTA (Multiple Object Tracking Accuracy)**
- Formula: `1 - (FP + FN + ID_switches) / GT`
- Measures overall tracking accuracy including identity consistency
- Uses range-azimuth distance matching with all labels

### 3. **Precision**
- Formula: `Average distance to matched ground truth (meters)`
- **Range-Azimuth**: Based on distance threshold matching with Hungarian assignment
- **Camera**: Would be based on IoU threshold matching
- Average localization accuracy of matched predictions/tracks in meters (lower is better)

### 4. **DetA (Detection Accuracy)**
- Formula: `(TP - FP) / GT`
- Based on camera IoU threshold matching
- Uses all labels regardless of cvpr_updated flag

### 5. **IoU (Intersection over Union)**
- Mean IoU between predicted and ground truth bounding boxes
- Calculated in camera image space
- Uses all labels

### 6. **NCLE (Normalized Center Location Error)**
- Formula: `CLE / image_diagonal` where `CLE = sqrt((x_pred - x_gt)² + (y_pred - y_gt)²)`
- Measures center point accuracy normalized by image diagonal
- Lower values indicate better localization
- Uses camera bounding boxes with all labels

### 7. **True Positives (TP)**
- Number of detections/tracks with IoU ≥ threshold with any ground truth
- Based on camera bounding box matching
- Uses all labels

### 8. **False Positives (FP)**
- Number of detections/tracks with IoU < threshold for all ground truth
- Based on camera bounding box matching
- Uses all labels

## Matching Criteria

- **Range-Azimuth metrics**: Use Euclidean distance threshold (default: 5.0m)
- **Camera metrics**: Use IoU threshold (default: 0.2)

## Configuration

- `use_cvpr_labels_only`: 
  - If True: Only affects range-azimuth precision calculation
  - Camera metrics (DetA, IoU, NCLE, TP, FP) always use all labels
  - MOTA and HOTA always use all labels

- `iou_threshold`: Threshold for camera-based TP/FP determination (default: 0.2)
- `distance_threshold`: Threshold for range-azimuth matching (default: 5.0m)

## Implementation Details

- Uses `motmetrics` package for MOTA calculation
- HOTA is calculated as a simplified version focusing on detection and association accuracy
- Ground truth object IDs are used from the 'ID' column in labels.csv
- Tracking IDs come from the 'track_id' column in tracking results
- All camera-based metrics use consistent bounding box scaling to match visualization