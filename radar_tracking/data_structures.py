"""
Data structures for radar object detection and tracking system.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import numpy as np


@dataclass
class Detection:
    """
    Represents a single radar detection in polar coordinates.

    Attributes:
        range_m: Range distance in meters
        azimuth_rad: Azimuth angle in radians
        confidence: Detection confidence score (0.0 to 1.0)
        timestamp: Absolute timestamp in seconds
        frame_id: Frame/sample ID
        cartesian_pos: Converted Cartesian position (x, y)
        bbox_corners: Optional bounding box corners
    """
    range_m: float
    azimuth_rad: float
    confidence: float
    timestamp: float  # Absolute timestamp in seconds
    frame_id: Optional[int] = None
    cartesian_pos: Optional[Tuple[float, float]] = None
    bbox_corners: Optional[dict] = None

    def __post_init__(self):
        """Convert to Cartesian coordinates after initialization."""
        if self.cartesian_pos is None:
            from radar_tracking.coordinate_transforms import polar_to_cartesian
            self.cartesian_pos = polar_to_cartesian(self.range_m, self.azimuth_rad)


@dataclass
class Track:
    """
    Represents a tracked object with state history.

    Attributes:
        id: Unique track identifier
        state: Current state vector [x, y, vx, vy]
        covariance: State covariance matrix
        last_detection: Most recent associated detection
        age: Number of frames since track initialization
        hits: Number of successful detection associations
        time_since_update: Frames since last successful update
        confidence: Track confidence score
    """
    id: int
    state: np.ndarray  # Current state vector [x, y, vx, vy] (after update)
    covariance: np.ndarray  # Current covariance matrix (after update)

    # Prediction step state storage
    predicted_state: Optional[np.ndarray] = None  # State after prediction, before update
    predicted_covariance: Optional[np.ndarray] = None  # Covariance after prediction

    # State history for analysis
    state_history: List[Dict] = field(default_factory=list)

    last_detection: Optional[Detection] = None
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    confidence: float = 0.0

    @property
    def position(self) -> Tuple[float, float]:
        """Get current position estimate."""
        return (self.state[0], self.state[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate."""
        return (self.state[2], self.state[3])

    @property
    def kalman_polar_position(self) -> Tuple[float, float]:
        """Get current position in polar coordinates."""
        from radar_tracking.coordinate_transforms import cartesian_to_polar
        return cartesian_to_polar(self.state[0], self.state[1])

    def record_prediction_step(self, pred_state: np.ndarray, pred_cov: np.ndarray,
                               timestamp: Optional[float] = None, dt: float = None):
        """Record the prediction step results."""
        self.predicted_state = pred_state.copy()
        self.predicted_covariance = pred_cov.copy()

        # Add to history
        self.state_history.append({
            'step_type': 'prediction',
            'timestamp': timestamp,
            'dt': dt,
            'state': pred_state.copy(),
            'covariance': pred_cov.copy(),
            'uncertainty_trace': np.trace(pred_cov),
            'position_uncertainty': np.sqrt(pred_cov[0, 0] + pred_cov[1, 1])
        })

    def record_update_step(self, updated_state: np.ndarray, updated_cov: np.ndarray,
                           detection: Detection, innovation: Optional[np.ndarray] = None):
        """Record the update step results."""
        self.state = updated_state.copy()
        self.covariance = updated_cov.copy()

        # Add to history
        self.state_history.append({
            'step_type': 'update',
            'timestamp': detection.timestamp,
            'state': updated_state.copy(),
            'covariance': updated_cov.copy(),
            'detection': detection,
            'innovation': innovation.copy() if innovation is not None else None,
            'uncertainty_trace': np.trace(updated_cov),
            'position_uncertainty': np.sqrt(updated_cov[0, 0] + updated_cov[1, 1]),
            'uncertainty_reduction': (
                np.trace(self.predicted_covariance) - np.trace(updated_cov)
                if self.predicted_covariance is not None else 0.0
            )
        })

@dataclass
class TrackingResult:
    """
    Results from tracking evaluation.

    Attributes:
        total_distance: Sum of minimum distances between predictions and ground truth
        num_matches: Number of successful matches
        num_predictions: Total number of predictions
        num_ground_truth: Total number of ground truth detections
        match_ratio: Ratio of matches to ground truth
        average_distance: Average distance of matches
    """
    total_distance: float
    num_matches: int
    num_predictions: int
    num_ground_truth: int
    match_ratio: float
    average_distance: float
