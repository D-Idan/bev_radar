# Network Confidence Integration in Radar Tracking Algorithm

## Overview

This document explains how network detection confidence scores are integrated into the radar tracking algorithm for the `mahalanobis_distance` association strategy. Confidence scores influence detection filtering, track association, and state estimation.

## 1. Detection Filtering Parameters

### Minimum Confidence for Track Initialization

**Configuration**: `min_confidence_init: 0.5`

**Mathematical Condition**:
$$\text{Track Creation} = \begin{cases} 
\text{allowed} & \text{if } c_d \geq c_{\text{init}} \\
\text{blocked} & \text{otherwise}
\end{cases} \quad (1)$$

where $c_d$ is detection confidence and $c_{\text{init}}$ is the minimum threshold.

**Implementation**:
```python
if detection.confidence >= self.min_confidence_init:
    if self._is_within_radar_coverage(detection):
        self._initiate_track(detection)
```

**Effect**: Only detections with confidence ≥ 0.5 can initiate new tracks.

### Minimum Confidence for Association

**Configuration**: `min_confidence_assoc: 0.2`

**Mathematical Filter**:
$$\mathcal{D}_{\text{filtered}} = \{d \in \mathcal{D} \mid c_d \geq c_{\text{assoc}}\} \quad (2)$$

**Implementation**:
```python
high_conf_detections = [det for det in detections
                        if det.confidence >= self.min_confidence_assoc]
```

**Effect**: Only detections with confidence ≥ 0.2 are considered for track association.

## 2. Adaptive R Matrix Integration

### Association Gating Control

**Configuration**: `use_in_association: false`

**Standard Mahalanobis Distance**:
$$d^2 = (\mathbf{z} - \mathbf{H}\hat{\mathbf{x}})^T \mathbf{S}^{-1} (\mathbf{z} - \mathbf{H}\hat{\mathbf{x}}) \quad (3)$$

**Innovation Covariance**:
$$\mathbf{S} = \mathbf{H}\mathbf{P}\mathbf{H}^T + \mathbf{R}_{\text{gating}} \quad (4)$$

**Confidence-Weighted Gating**:
$$\mathbf{R}_{\text{gating}} = \frac{\mathbf{R}_{\text{base}}}{\alpha(c)} \quad (5)$$

where $\alpha(c)$ is the confidence factor.

**Implementation**:
```python
if self.use_adaptive_r_in_association:
    mahal_dist = self.kf.gating_distance(
        pred_state, pred_covariance, det_pos,
        confidence=detection.confidence,
        r_strategy=self.r_weighting_strategy,
        strategy_params=self.r_weighting_config
    )
```

**Effect**: Low-confidence detections produce larger Mahalanobis distances (stricter association); high-confidence detections produce smaller distances (easier association).

### Kalman Update Control

**Configuration**: `use_in_update: false`

**Kalman Gain**:
$$\mathbf{K} = \mathbf{P}_{k|k-1}\mathbf{H}^T(\mathbf{H}\mathbf{P}_{k|k-1}\mathbf{H}^T + \mathbf{R}_{\text{update}})^{-1} \quad (6)$$

**State Update**:
$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}(\mathbf{z}_k - \mathbf{H}\hat{\mathbf{x}}_{k|k-1}) \quad (7)$$

**Confidence-Weighted Update**:
$$\mathbf{R}_{\text{update}} = \mathbf{R}_{\text{base}} \times \alpha(c) \quad (8)$$

**Implementation**:
```python
if self.use_adaptive_r_in_update:
    updated_state, updated_covariance, innovation = self.kf.update(
        pred_state, pred_covariance, detection.cartesian_pos,
        confidence=detection.confidence,
        r_strategy=self.r_weighting_strategy,
        strategy_params=self.r_weighting_config,
    )
```

**Effect**: High-confidence detections have stronger influence on state updates; low-confidence detections have weaker influence.

## 3. Confidence Weighting Strategies

### Linear Strategy

**Configuration**: `weighting_strategy: "linear"`

**Confidence Factor**:
$$\alpha(c) = \alpha_{\max} + (\alpha_{\min} - \alpha_{\max}) \times c \quad (9)$$

**With Default Values** (`r_min_factor: 0.7`, `r_max_factor: 15.0`):

$$\alpha(c) = 15.0 + (0.7 - 15.0) \times c = 15.0 - 14.3c \quad (10)$$

**Extreme Cases**:
- High confidence ($c = 1.0$): $\alpha = 0.7$
- Low confidence ($c = 0.0$): $\alpha = 15.0$

**Implementation**:
```python
confidence_factor = R_max_factor + (R_min_factor - R_max_factor) * confidence_clamped
```

### Squared Strategy

**Confidence Factor**:
$$\alpha(c) = \frac{1}{c^2} \quad (11)$$

### Stepped Strategy

**Confidence Factor**:
$$\alpha(c) = \text{lookup}(c, \text{thresholds}, \text{factors}) \quad (12)$$

## 4. R Matrix Application

### For Association (Inverted Scaling)

$$\mathbf{R}_{\text{gating}} = \frac{\mathbf{R}_{\text{base}}}{\alpha(c)} \quad (13)$$

**Effect on Mahalanobis Distance**:
- High confidence ($\alpha = 0.7$): $\mathbf{R}_{\text{gating}} = 1.43 \times \mathbf{R}_{\text{base}}$ → smaller $d^2$
- Low confidence ($\alpha = 15.0$): $\mathbf{R}_{\text{gating}} = 0.067 \times \mathbf{R}_{\text{base}}$ → larger $d^2$

### For Updates (Direct Scaling)

$$\mathbf{R}_{\text{update}} = \mathbf{R}_{\text{base}} \times \alpha(c) \quad (14)$$

**Effect on Kalman Gain**:
- High confidence ($\alpha = 0.7$): Higher gain → stronger update influence
- Low confidence ($\alpha = 15.0$): Lower gain → weaker update influence

**Implementation**:
```python
def get_confidence_weighted_R(self, confidence, strategy="linear", 
                              strategy_params=None, return_confidence_factor=False):
    # Calculate confidence_factor using selected strategy
    if return_confidence_factor:  # For gating
        return self.R / confidence_factor
    return self.R * confidence_factor  # For update
```

## 5. Association Validation

### Chi-Squared Threshold

**Configuration**: `default_chi2_threshold: 4.605`

**Validation Condition**:
$$\text{Valid Association} = d^2 \leq \chi^2_{\text{threshold}} \quad (15)$$

where $\chi^2_{\text{threshold}} = 4.605$ (90% confidence ellipse).

**Implementation**:
```python
if self.association_strategy in [AssociationStrategy.MAHALANOBIS_DISTANCE]:
    return cost <= self.default_chi2_threshold
```

## 6. Complete Algorithm Flow

### Step 1: Detection Filtering
$$\mathcal{D}_{\text{valid}} = \{d \in \mathcal{D} \mid c_d \geq c_{\text{assoc}}\} \quad (16)$$

### Step 2: Cost Matrix Calculation
$$\mathbf{C}_{ij} = d^2_{ij} = (\mathbf{z}_j - \mathbf{H}\hat{\mathbf{x}}_i)^T \mathbf{S}_{ij}^{-1} (\mathbf{z}_j - \mathbf{H}\hat{\mathbf{x}}_i) \quad (17)$$

where:
$$\mathbf{S}_{ij} = \mathbf{H}\mathbf{P}_i\mathbf{H}^T + \mathbf{R}_{\text{gating}}(c_j) \quad (18)$$

### Step 3: Hungarian Assignment
Solve: $\min \sum_{i,j} \mathbf{C}_{ij} x_{ij}$ subject to assignment constraints.

### Step 4: Track Updates
For matched pairs, apply Kalman update with confidence-weighted $\mathbf{R}_{\text{update}}$.

## 7. Configuration Examples

### Conservative Tracking
```yaml
min_confidence_init: 0.8
min_confidence_assoc: 0.6
use_in_association: false
use_in_update: false
```
**Effect**: Only high-confidence detections create/update tracks without adaptive weighting.

### Confidence-Aware Association Only
```yaml
min_confidence_init: 0.5
min_confidence_assoc: 0.2
use_in_association: true
use_in_update: false
r_min_factor: 0.5
r_max_factor: 10.0
```
**Effect**: Adaptive gating based on confidence, but standard Kalman updates.

### Full Confidence Integration
```yaml
min_confidence_init: 0.4
min_confidence_assoc: 0.2
use_in_association: true
use_in_update: true
r_min_factor: 0.3
r_max_factor: 20.0
```
**Effect**: Both association and updates adapt to detection confidence levels.

### Permissive Tracking
```yaml
min_confidence_init: 0.3
min_confidence_assoc: 0.1
use_in_association: false
use_in_update: false
```
**Effect**: Accepts low-confidence detections without adaptive weighting, maximizing detection sensitivity.