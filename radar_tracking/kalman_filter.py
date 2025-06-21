# kalman_filter.py

"""
Kalman filter implementation for radar object tracking.
Adapted for 2D position and velocity tracking.
"""
import numpy as np
from typing import Tuple, List, Optional
from copy import deepcopy


class RadarKalmanFilter:
    """
    Kalman filter for tracking radar objects in 2D space with variable time steps.

    State vector: [x, y, vx, vy] (position and velocity)
    Measurement vector: [x, y] (position only)
    """

    def __init__(self, base_dt: float = 0.1, config: dict = None):
        """
        Initialize Kalman filter.

        Args:
            base_dt: Base time step for process noise tuning (seconds)
            config: Optional configuration dictionary
        """
        self.base_dt = base_dt
        self.dim_x = 4  # State dimension: [x, y, vx, vy]
        self.dim_z = 2  # Measurement dimension: [x, y]

        # Default config
        default_config = {
            'process_noise_q_std': 1.0,  # Reduced from 3.0
            'measurement_noise_std': 0.5,  # Keep current
            'initial_pos_std': 2.0,  # Reduced from ~7m
            'initial_vel_std': 5.0,  # Reduced from ~7m/s
        }

        # Merge with provided config
        self.config = {**default_config, **(config or {})}

        # Set parameters from config
        self.q = self.config['process_noise_q_std'] ** 2
        self.R = np.eye(2) * (self.config['measurement_noise_std'] ** 2)

        # Different uncertainties for position and velocity
        pos_var = self.config['initial_pos_std'] ** 2
        vel_var = self.config['initial_vel_std'] ** 2
        self.P_init = np.diag([pos_var, pos_var, vel_var, vel_var])

        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

    def _get_F_matrix(self, dt: float) -> np.ndarray:
        """Get state transition matrix for given time step."""
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def _get_Q_matrix(self, dt: float) -> np.ndarray:
        """Get process noise matrix for given time step."""
        # Adaptive process noise based on time step
        # Larger time steps = more uncertainty
        q_scaled = self.q * (dt / self.base_dt)

        return q_scaled * np.array([
            [dt ** 4 / 4, 0, dt ** 3 / 2, 0],
            [0, dt ** 4 / 4, 0, dt ** 3 / 2],
            [dt ** 3 / 2, 0, dt ** 2, 0],
            [0, dt ** 3 / 2, 0, dt ** 2]
        ])

    def initiate(self, measurement: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize track state from first measurement.

        Args:
            measurement: Initial position measurement (x, y)

        Returns:
            Tuple of (initial_state, initial_covariance)
        """
        # Initialize state: [x, y, 0, 0] (zero initial velocity)
        state = np.array([measurement[0], measurement[1], 0.0, 0.0])
        covariance = deepcopy(self.P_init)

        return state, covariance

    def predict(self, state: np.ndarray, covariance: np.ndarray,
                dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict next state using motion model with specific time step.

        Args:
            state: Current state vector
            covariance: Current state covariance matrix
            dt: Time step for this prediction (seconds)

        Returns:
            Tuple of (predicted_state, predicted_covariance)
        """
        F = self._get_F_matrix(dt)
        Q = self._get_Q_matrix(dt)

        # Predict state: x_{k|k-1} = F * x_{k-1|k-1}
        state_pred = F @ state

        # Predict covariance: P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        covariance_pred = F @ covariance @ F.T + Q

        return state_pred, covariance_pred

    def multi_step_predict(self, state: np.ndarray, covariance: np.ndarray,
                           total_dt: float, step_dt: float) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Perform multiple prediction steps for large time gaps.

        Args:
            state: Current state vector
            covariance: Current state covariance matrix
            total_dt: Total time to predict ahead
            step_dt: Time step for each prediction

        Returns:
            List of (state, covariance) tuples for each step
        """
        predictions = []
        current_state = state
        current_cov = covariance

        num_steps = int(np.ceil(total_dt / step_dt))

        for i in range(num_steps):
            # Use remaining time for last step if needed
            dt = min(step_dt, total_dt - i * step_dt)
            current_state, current_cov = self.predict(current_state, current_cov, dt)
            predictions.append((deepcopy(current_state), deepcopy(current_cov)))

        return predictions

    def update(self,
               state: np.ndarray,
               covariance: np.ndarray,
               measurement: Tuple[float, float],
               confidence: Optional[float] = None,
               r_strategy: str = "squared",
               strategy_params: dict = None,

               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update state estimate with new measurement using confidence-weighted R.

        Args:
            state: Predicted state vector
            covariance: Predicted state covariance matrix
            measurement: New measurement (x, y)
            confidence: Detection confidence for R weighting (optional)
            r_strategy: Strategy for R weighting ("squared", "linear", "stepped")

        Returns:
            Tuple of (updated_state, updated_covariance, innovation)
        """
        # Convert measurement to numpy array
        z = np.array([measurement[0], measurement[1]])

        # Innovation: y = z - H * x_{k|k-1}
        innovation = z - self.H @ state

        # Get confidence-weighted R matrix
        if confidence is not None:
            R_weighted = self.get_confidence_weighted_R(confidence, r_strategy, strategy_params)
        else:
            R_weighted = self.R

        # Innovation covariance: S = H * P_{k|k-1} * H^T + R
        innovation_cov = self.H @ covariance @ self.H.T + R_weighted

        # Kalman gain: K = P_{k|k-1} * H^T * S^{-1}
        kalman_gain = covariance @ self.H.T @ np.linalg.inv(innovation_cov)

        # Update state: x_{k|k} = x_{k|k-1} + K * y
        state_updated = state + kalman_gain @ innovation

        # Update covariance: P_{k|k} = (I - K * H) * P_{k|k-1}
        I_KH = np.eye(self.dim_x) - kalman_gain @ self.H
        covariance_updated = I_KH @ covariance

        return state_updated, covariance_updated, innovation

    def gating_distance(self,
                        state: np.ndarray,
                        covariance: np.ndarray,
                        measurement: Tuple[float, float]) -> float:
        """
        Calculate Mahalanobis distance for gating.

        Args:
            state: State vector
            covariance: State covariance matrix
            measurement: Measurement (x, y)

        Returns:
            Mahalanobis distance
        """
        # Convert measurement to numpy array
        z = np.array([measurement[0], measurement[1]])

        # Predicted measurement
        z_pred = self.H @ state

        # Innovation
        innovation = z - z_pred

        # Innovation covariance
        innovation_cov = self.H @ covariance @ self.H.T + self.R

        # Mahalanobis distance
        distance = innovation.T @ np.linalg.inv(innovation_cov) @ innovation

        return float(distance)

    def get_confidence_weighted_R(self, confidence: float, strategy: str = "squared",
                                  strategy_params: dict = None) -> np.ndarray:
        """
        Get measurement noise covariance matrix weighted by detection confidence.

        Args:
            confidence: Detection confidence score (0.0 to 1.0)
            strategy: Weighting strategy ("squared", "linear", "stepped")
            strategy_params: Parameters for the specific strategy

        Returns:
            Confidence-weighted R matrix
        """
        # Default parameters if none provided
        if strategy_params is None:
            strategy_params = {}

        # Clamp confidence to reasonable range
        confidence_clamped = np.clip(confidence, 0.01, 1.0)

        if strategy == "squared":
            # R = R_base / conf^2 (lower confidence = higher noise)
            R_max_factor = strategy_params.get('r_max_factor', 100.0)
            confidence_factor = 1.0 / (confidence_clamped ** 2)
            confidence_factor = min(confidence_factor, R_max_factor)

        elif strategy == "linear":
            # Linear scaling between R_min and R_max based on confidence
            R_min_factor = strategy_params.get('r_min_factor', 0.5)  # High confidence
            R_max_factor = strategy_params.get('r_max_factor', 10.0)  # Low confidence

            # Linear interpolation: factor = R_max + (R_min - R_max) * confidence
            confidence_factor = R_max_factor + (R_min_factor - R_max_factor) * confidence_clamped

        elif strategy == "stepped":
            # Stepped confidence levels with different factors
            thresholds = strategy_params.get('stepped_r_thresholds', [0.9, 0.8, 0.7, 0.5, 0.3])
            factors = strategy_params.get('stepped_r_factors', [0.25, 0.5, 1.0, 2.0, 5.0, 15.0])

            # Ensure we have the right number of factors (one more than thresholds)
            if len(factors) != len(thresholds) + 1:
                raise ValueError(f"Need {len(thresholds) + 1} factors for {len(thresholds)} thresholds")

            # Find appropriate factor based on confidence level
            confidence_factor = factors[-1]  # Default to lowest confidence factor
            for i, threshold in enumerate(thresholds):
                if confidence_clamped >= threshold:
                    confidence_factor = factors[i]
                    break
        else:
            confidence_factor = 1.0

        return self.R * confidence_factor