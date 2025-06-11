# tracker.py
"""
Main tracking logic implementing SORT-like algorithm for radar objects.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from radar_tracking.data_structures import Detection, Track
from radar_tracking.kalman_filter import RadarKalmanFilter
from radar_tracking.metrics import RadarMetrics
from radar_tracking.coordinate_transforms import euclidean_distance, cartesian_to_polar
from scipy.optimize import linear_sum_assignment
from enum import Enum
from copy import deepcopy


class AssociationStrategy(Enum):
    """Available association strategies."""
    DISTANCE_ONLY = "distance_only"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    CONFIDENCE_GATED = "confidence_gated"
    HYBRID_SCORE = "hybrid_score"


class RadarTracker:
    """
    SORT-like tracker for radar objects using Kalman filtering and Hungarian assignment.
    """

    def __init__(self,
                 max_age: int = 5,
                 min_hits: int = 3,
                 iou_threshold: float = 5.0,  # Distance threshold in meters
                 dt: float = 0.1,  # Default time step for predictions
                 base_dt: float = 0.1,
                 max_dt_gap: float = 1.0,
                 # Confidence-based parameters
                 min_confidence_init: float = 0.5,
                 min_confidence_assoc: float = 0.3,
                 confidence_weight: float = 0.3,
                 association_strategy: str = "confidence_weighted",
                 # Range culling parameters
                 enable_range_culling: bool = True,
                 max_range: float = 103.0,
                 min_azimuth_deg: float = -90.0,
                 max_azimuth_deg: float = 90.0,
                 range_buffer: float = 10.0,
                 azimuth_buffer_deg: float = 5.0,
                 max_time_without_update: float = 2.0,  # Kill track after 2s without update
                 max_frame_gap_time: float = 5.0,  # Kill all tracks if gap > 5s
                 ):
        """
        Initialize radar tracker with timestamp support.

        Args:
            max_age: Maximum frames to keep track alive without detections
            min_hits: Minimum detections before track is considered confirmed
            iou_threshold: Maximum distance for association (meters)
            min_confidence_init: Minimum confidence required to initiate new track
            min_confidence_assoc: Minimum confidence required for association
            confidence_weight: Weight for confidence in association cost (0.0-1.0)
            association_strategy: Strategy for association
            enable_range_culling: Whether to remove tracks outside radar coverage
            max_range: Maximum radar range (meters)
            min_azimuth_deg: Minimum azimuth coverage (degrees)
            max_azimuth_deg: Maximum azimuth coverage (degrees)
            range_buffer: Buffer zone for range culling (meters)
            azimuth_buffer_deg: Buffer zone for azimuth culling (degrees)
            base_dt: Base time step for regular predictions (seconds)
            max_dt_gap: Maximum time gap before multi-step prediction (seconds)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.dt = dt
        self.base_dt = base_dt
        self.max_dt_gap = max_dt_gap
        self.prediction_history = {}  # Track ID -> list of predictions
        self.gap_predictions = {}     # Track ID -> predictions during gaps
        self.max_time_without_update = max_time_without_update
        self.max_frame_gap_time = max_frame_gap_time
        self.track_last_update_times = {}  # Track ID -> last update timestamp

        # Confidence-based parameters
        self.min_confidence_init = min_confidence_init
        self.min_confidence_assoc = min_confidence_assoc
        self.confidence_weight = confidence_weight

        # Range culling parameters
        self.enable_range_culling = enable_range_culling
        self.max_range = max_range
        self.min_azimuth_deg = min_azimuth_deg
        self.max_azimuth_deg = max_azimuth_deg
        self.range_buffer = range_buffer
        self.azimuth_buffer_deg = azimuth_buffer_deg

        # Validate and set association strategy
        try:
            self.association_strategy = AssociationStrategy(association_strategy)
        except ValueError:
            print(f"Warning: Unknown association strategy '{association_strategy}', using 'confidence_weighted'")
            self.association_strategy = AssociationStrategy.CONFIDENCE_WEIGHTED

        # Initialize components
        self.kf = RadarKalmanFilter(base_dt=base_dt)
        self.metrics = RadarMetrics(max_distance_threshold=iou_threshold)

        # Tracking state
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0

        # Timestamp tracking
        self.last_update_time = None
        self.track_last_update_times = {}  # Track ID -> last update timestamp

        # Statistics for range culling
        self.tracks_culled_by_range = 0

    def update(self, detections: List[Detection], dt: Optional[float] = None,
               current_time: Optional[float] = None) -> List[Track]:
        """
        Update tracker with new detections and dynamic time step.

        Args:
            detections: List of detections for current frame
            dt: Optional time step for this frame (if None, uses configured dt)
            current_time: Optional current timestamp in seconds

        Returns:
            List of active tracks
        """
        # Calculate frame time step with comprehensive logic
        if current_time is not None and self.last_update_time is not None:
            # Use actual time difference when timestamps are available
            frame_dt = current_time - self.last_update_time

            # If gap too large, kill all tracks
            if frame_dt > self.max_frame_gap_time:
                print(f"Large time gap detected ({frame_dt:.2f}s), killing all tracks")
                self.tracks = []
                self.track_last_update_times = {}

        elif dt is not None:
            # Use provided dt
            frame_dt = dt
        else:
            # Fall back to configured dt
            frame_dt = self.dt

        # Filter detections by confidence for association
        high_conf_detections = [det for det in detections
                                if det.confidence >= self.min_confidence_assoc]

        # Perform predictions based on actual time gap
        if current_time is not None:
            # Use timestamp-aware prediction for better handling of time gaps
            self._predict_tracks_with_timestamp(current_time, frame_dt)
        else:
            # Fall back to simple prediction when no timestamp available
            self._predict_tracks(frame_dt)

        # Remove tracks that are predicted to be outside radar coverage
        if self.enable_range_culling:
            self._remove_out_of_range_tracks()

        # Associate detections with tracks using high-confidence detections only
        matches, unmatched_detections, unmatched_tracks = self._associate(high_conf_detections)

        # Update matched tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx] = self._update_track(
                self.tracks[track_idx],
                high_conf_detections[det_idx],
                dt=frame_dt  # Pass the calculated time step
            )

        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            # Update current state to predicted state (if prediction was made)
            if hasattr(track, 'predicted_state') and track.predicted_state is not None:
                track.state = track.predicted_state.copy()
                track.covariance = track.predicted_covariance.copy()

            track.time_since_update += 1
            # Note: track.age is already incremented in prediction step

        # Create new tracks from unmatched detections (apply confidence threshold)
        for det_idx in unmatched_detections:
            detection = high_conf_detections[det_idx]
            if detection.confidence >= self.min_confidence_init:
                # Additional check: only create tracks for detections within radar coverage
                if self._is_within_radar_coverage(detection):
                    self._initiate_track(detection)

        # Remove dead tracks
        self._remove_dead_tracks()

        # Update last update time if provided
        if current_time is not None:
            # After processing matches, check time-based termination
            self._remove_time_expired_tracks(current_time)
            self.last_update_time = current_time

        return self._get_confirmed_tracks()

    def _remove_time_expired_tracks(self, current_time: float):
        """Remove tracks that haven't been updated within time threshold."""
        tracks_to_keep = []

        for track in self.tracks:
            last_update = self.track_last_update_times.get(track.id, current_time)
            time_since_update = current_time - last_update

            if time_since_update <= self.max_time_without_update:
                tracks_to_keep.append(track)
            else:
                print(f"Track {track.id} killed: {time_since_update:.2f}s without update")

        self.tracks = tracks_to_keep

    def _predict_tracks_with_timestamp(self, current_time: float, time_gap: float):
        """
        Predict tracks considering actual time gaps with multi-step prediction for large gaps.
        Remove those predicted outside radar coverage.

        Args:
            current_time: Current timestamp in seconds
            time_gap: Time since last update in seconds
        """
        tracks_to_keep = []

        for track in self.tracks:
            # Get time since last update for this specific track
            last_update = self.track_last_update_times.get(track.id, current_time - time_gap)
            track_time_gap = current_time - last_update

            # Perform prediction (either single-step or multi-step)
            if track_time_gap > self.max_dt_gap:
                # Multi-step prediction for large gaps
                predictions = self.kf.multi_step_predict(
                    track.state, track.covariance, track_time_gap, self.base_dt
                )

                # Record all intermediate prediction steps
                num_steps = len(predictions)
                step_dt = track_time_gap / num_steps
                for i, (pred_state, pred_cov) in enumerate(predictions):
                    # Calculate timestamp for this prediction step
                    step_time = last_update + (i + 1) * step_dt
                    track.record_prediction_step(pred_state, pred_cov, step_time, step_dt)

                # Use final prediction for culling check
                pred_state, pred_cov = predictions[-1]
            else:
                # Single-step prediction
                pred_state, pred_cov = self.kf.predict(
                    track.state, track.covariance, track_time_gap
                )
                # Record single prediction step
                track.record_prediction_step(pred_state, pred_cov, current_time, track_time_gap)

            # Check if predicted position is within coverage (applies to both cases)
            if self._is_track_within_coverage_predicted(pred_state):
                # Track is within coverage - keep it
                track.predicted_state = pred_state
                track.predicted_covariance = pred_cov
                track.age += 1
                tracks_to_keep.append(track)
                self.track_last_update_times[track.id] = current_time
            else:
                # Track predicted outside coverage - kill it
                self.tracks_culled_by_range += 1
                print(f"Track {track.id} culled: predicted outside radar coverage")

        # Update tracks list (IMPORTANT: correct indentation)
        self.tracks = tracks_to_keep

    def _is_track_within_coverage_predicted(self, predicted_state: np.ndarray) -> bool:
        """Check if predicted state is within radar coverage."""
        x, y = predicted_state[0], predicted_state[1]
        range_m, azimuth_rad = cartesian_to_polar(x, y)
        azimuth_deg = np.degrees(azimuth_rad)

        # Apply buffer zones
        max_range_check = self.max_range + self.range_buffer
        min_range_check = -self.range_buffer
        min_azimuth_check = self.min_azimuth_deg - self.azimuth_buffer_deg
        max_azimuth_check = self.max_azimuth_deg + self.azimuth_buffer_deg

        return (min_range_check <= range_m <= max_range_check and
                min_azimuth_check <= azimuth_deg <= max_azimuth_check)

    def _predict_tracks(self, dt: float):
        """
        Predict all existing tracks with specific time step (fallback method).

        Args:
            dt: Time step for prediction
        """
        for track in self.tracks:
            track.state, track.covariance = self.kf.predict(
                track.state,
                track.covariance,
                dt
            )
            track.age += 1

    def _is_within_radar_coverage(self, detection: Detection) -> bool:
        """
        Check if a detection is within radar coverage area.

        Args:
            detection: Detection to check

        Returns:
            True if detection is within coverage
        """
        range_m = detection.range_m
        azimuth_deg = np.degrees(detection.azimuth_rad)

        # Check range bounds
        if range_m < 0 or range_m > self.max_range:
            return False

        # Check azimuth bounds
        if azimuth_deg < self.min_azimuth_deg or azimuth_deg > self.max_azimuth_deg:
            return False

        return True

    def _is_track_within_coverage(self, track: Track, use_buffer: bool = True) -> bool:
        """
        Check if a track's predicted position is within radar coverage.

        Args:
            track: Track to check
            use_buffer: Whether to use buffer zones for tolerance

        Returns:
            True if track is within coverage (including buffer if enabled)
        """
        # Convert track Cartesian position to polar
        x, y = track.state[0], track.state[1]
        range_m, azimuth_rad = cartesian_to_polar(x, y)
        azimuth_deg = np.degrees(azimuth_rad)

        # Apply buffer zones if requested
        if use_buffer:
            max_range_check = self.max_range + self.range_buffer
            min_range_check = -self.range_buffer  # Allow slightly negative ranges with buffer
            min_azimuth_check = self.min_azimuth_deg - self.azimuth_buffer_deg
            max_azimuth_check = self.max_azimuth_deg + self.azimuth_buffer_deg
        else:
            max_range_check = self.max_range
            min_range_check = 0.0
            min_azimuth_check = self.min_azimuth_deg
            max_azimuth_check = self.max_azimuth_deg

        # Check range bounds
        if range_m < min_range_check or range_m > max_range_check:
            return False

        # Check azimuth bounds
        if azimuth_deg < min_azimuth_check or azimuth_deg > max_azimuth_check:
            return False

        return True

    def _remove_out_of_range_tracks(self):
        """Remove tracks that are predicted to be outside radar coverage."""
        tracks_to_keep = []

        for track in self.tracks:
            if self._is_track_within_coverage(track, use_buffer=True):
                tracks_to_keep.append(track)
            else:
                self.tracks_culled_by_range += 1
                # Optional: Log which track was culled and why
                range_m, azimuth_rad = cartesian_to_polar(track.state[0], track.state[1])
                azimuth_deg = np.degrees(azimuth_rad)
                print(
                    f"Track {track.id} culled: predicted position (r={range_m:.1f}m, az={azimuth_deg:.1f}°) outside coverage")

        self.tracks = tracks_to_keep

    def _associate(self, detections: List[Detection]) -> Tuple[List[Tuple[int, int]],
    List[int],
    List[int]]:
        """
        Associate detections with existing tracks using Hungarian algorithm.

        Args:
            detections: List of detections to associate

        Returns:
            Tuple of (matches, unmatched_detections, unmatched_tracks)
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))

        # Create cost matrix using selected strategy
        cost_matrix = self._create_cost_matrix(detections)

        # Apply Hungarian algorithm
        if cost_matrix.size > 0:
            track_indices, det_indices = linear_sum_assignment(cost_matrix)

            # Filter matches by distance threshold and confidence requirements
            matches = []
            for t_idx, d_idx in zip(track_indices, det_indices):
                if self._is_valid_association(cost_matrix[t_idx, d_idx], detections[d_idx]):
                    matches.append((t_idx, d_idx))

            # Find unmatched tracks and detections
            matched_tracks = {t_idx for t_idx, _ in matches}
            matched_detections = {d_idx for _, d_idx in matches}

            unmatched_tracks = [t for t in range(len(self.tracks))
                                if t not in matched_tracks]
            unmatched_detections = [d for d in range(len(detections))
                                    if d not in matched_detections]
        else:
            matches = []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_detections = list(range(len(detections)))

        return matches, unmatched_detections, unmatched_tracks

    def _create_cost_matrix(self, detections: List[Detection]) -> np.ndarray:
        """
        Create cost matrix using the selected association strategy.

        Args:
            detections: List of detections to associate

        Returns:
            Cost matrix for Hungarian algorithm
        """
        cost_matrix = np.zeros((len(self.tracks), len(detections)))

        for t, track in enumerate(self.tracks):
            for d, detection in enumerate(detections):
                track_pos = (track.state[0], track.state[1])
                det_pos = detection.cartesian_pos
                distance = euclidean_distance(track_pos, det_pos)

                if self.association_strategy == AssociationStrategy.DISTANCE_ONLY:
                    cost_matrix[t, d] = distance

                elif self.association_strategy == AssociationStrategy.CONFIDENCE_WEIGHTED:
                    # Weight distance by inverse confidence
                    confidence_factor = 1.0 / max(detection.confidence, 0.1)
                    cost_matrix[t, d] = distance * (1.0 + self.confidence_weight * (confidence_factor - 1.0))

                elif self.association_strategy == AssociationStrategy.CONFIDENCE_GATED:
                    # Use distance but apply confidence gating
                    if detection.confidence < self.min_confidence_assoc:
                        cost_matrix[t, d] = float('inf')  # Reject low confidence
                    else:
                        cost_matrix[t, d] = distance

                elif self.association_strategy == AssociationStrategy.HYBRID_SCORE:
                    # Combine distance, confidence, and track quality
                    try:
                        mahal_dist = self.kf.gating_distance(
                            track.state, track.covariance, det_pos
                        )
                        # Normalize Mahalanobis distance
                        mahal_norm = min(mahal_dist / 10.0, 3.0)
                    except:
                        mahal_norm = 1.0

                    # Track quality score (higher is better)
                    track_quality = min(track.hits / max(track.age, 1), 1.0)

                    # Confidence score (higher is better)
                    conf_score = detection.confidence

                    # Combined cost (lower is better)
                    distance_cost = distance / self.iou_threshold  # Normalize distance
                    confidence_cost = (1.0 - conf_score) * 2.0  # Penalty for low confidence
                    mahal_cost = mahal_norm * 0.5  # Mahalanobis component
                    track_bonus = (1.0 - track_quality) * 0.5  # Penalty for poor tracks

                    cost_matrix[t, d] = distance_cost + confidence_cost + mahal_cost + track_bonus

        return cost_matrix

    def _is_valid_association(self, cost: float, detection: Detection) -> bool:
        """
        Check if an association is valid based on cost and confidence.

        Args:
            cost: Association cost from cost matrix
            detection: Detection being associated

        Returns:
            True if association is valid
        """
        # Basic distance check
        if self.association_strategy == AssociationStrategy.HYBRID_SCORE:
            # For hybrid score, cost is normalized differently
            return cost <= 2.0  # Adjusted threshold for hybrid scoring
        else:
            # For other strategies, use distance threshold
            return cost <= self.iou_threshold

    def _update_track(self, track: Track, detection: Detection, dt: float = None) -> Track:
        """
        Update track with associated detection.

        Args:
            track: Track to update
            detection: Detection to use for update
            dt: Time step for this update (if None, uses self.dt)

        Returns:
            Updated track
        """
        # Use provided dt or fall back to tracker's dt
        if dt is None:
            dt = self.dt

        # Use the already-computed prediction state.
        pred_state = track.predicted_state
        pred_covariance = track.predicted_covariance

        # If this is the second detection, compute velocity estimate
        if track.hits == 1 and track.last_detection is not None:
            x_prev, y_prev = track.last_detection.cartesian_pos
            x_new, y_new = detection.cartesian_pos

            # Calculate actual time difference for velocity computation
            if hasattr(detection, 'timestamp') and hasattr(track.last_detection, 'timestamp'):
                actual_dt = detection.timestamp - track.last_detection.timestamp
                if actual_dt <= 0:
                    actual_dt = dt  # Fallback if timestamps are invalid
            else:
                actual_dt = dt

            # Compute "observed" velocity
            vx = (x_new - x_prev) / actual_dt
            vy = (y_new - y_prev) / actual_dt

            # First, run the usual KF-predict with the current dt
            pred_state, pred_covariance = self.kf.predict(track.state, track.covariance, dt)

            # Overwrite velocity entries with bootstrapped value
            pred_state[2] = vx
            pred_state[3] = vy

        else:
            # Normal predict→update cycle
            pred_state, pred_covariance = self.kf.predict(track.state, track.covariance, dt)


        # Perform update step
        updated_state, updated_covariance, innovation = self.kf.update(
            pred_state, pred_covariance, detection.cartesian_pos
        )

        # Record update step
        track.record_update_step(updated_state, updated_covariance, detection, innovation)

        # Update other track attributes
        track.last_detection = detection
        track.hits += 1
        track.time_since_update = 0
        track.confidence = detection.confidence

        # Update track timestamp
        if hasattr(detection, 'timestamp'):
            self.track_last_update_times[track.id] = detection.timestamp

        return track

    def _initiate_track(self, detection: Detection):
        """Create new track from unmatched detection."""
        state, covariance = self.kf.initiate(detection.cartesian_pos)

        new_track = Track(
            id=self.next_id,
            state=state,
            covariance=covariance,
            last_detection=detection,
            age=1,
            hits=1,
            time_since_update=0,
            confidence=detection.confidence
        )

        self.tracks.append(new_track)
        self.next_id += 1

    def _remove_dead_tracks(self):
        """Remove tracks that have been inactive for too long."""
        self.tracks = [track for track in self.tracks
                       if track.time_since_update < self.max_age]

    def _get_confirmed_tracks(self) -> List[Track]:
        """
        Return only those tracks that have seen at least `min_hits` detections.
        """
        return [
            track
            for track in self.tracks
            if track.hits >= self.min_hits
        ]

    def get_all_tracks(self) -> List[Track]:
        """Get all active tracks (confirmed and tentative)."""
        return deepcopy(self.tracks)

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        self.tracks_culled_by_range = 0

    def get_config_summary(self) -> Dict:
        """Get current configuration summary."""
        return {
            'max_age': self.max_age,
            'min_hits': self.min_hits,
            'iou_threshold': self.iou_threshold,
            'dt': self.dt,
            'min_confidence_init': self.min_confidence_init,
            'min_confidence_assoc': self.min_confidence_assoc,
            'confidence_weight': self.confidence_weight,
            'association_strategy': self.association_strategy.value,
            'enable_range_culling': self.enable_range_culling,
            'max_range': self.max_range,
            'min_azimuth_deg': self.min_azimuth_deg,
            'max_azimuth_deg': self.max_azimuth_deg,
            'range_buffer': self.range_buffer,
            'azimuth_buffer_deg': self.azimuth_buffer_deg,
            'tracks_culled_by_range': self.tracks_culled_by_range
        }