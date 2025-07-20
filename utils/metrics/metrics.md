# Tracking and Detection Metrics

This implementation evaluates tracking performance using the following primary metrics:

## Primary Metrics

### 1. **HOTA (Higher Order Tracking Accuracy)**
- Combines detection and association accuracy
- Formula: `sqrt(DetA × AssA)`
- Provides a balanced measure of tracking performance

### 2. **MOTA (Multiple Object Tracking Accuracy)**
- Formula: `1 - (FP + FN + ID_switches) / GT`
- Measures overall tracking accuracy including identity consistency

### 3. **Precision**
- Formula: `TP / (TP + FP)`
- Proportion of predicted detections/tracks that are correct

### 4. **DetA (Detection Tracking Accuracy)**
- Formula: `(TP - FP) / GT`
- Detection quality with false positive penalty

### 5. **IoU (Intersection over Union)**
- For range-azimuth: `TP / (TP + FP + FN)` based on distance threshold
- For camera: Mean IoU between predicted and ground truth bounding boxes

## Configuration

- `use_cvpr_labels_only`: 
  - If True: Only uses labels with cvpr_updated=True for range-azimuth metrics
  - Camera metrics always use all labels regardless of this setting

## Implementation Details

- Uses `motmetrics` package for MOTA calculation
- HOTA is calculated as a simplified version focusing on detection and association accuracy
- Ground truth object IDs are used from the 'ID' column in labels.csv
- Tracking IDs come from the 'track_id' column in tracking results