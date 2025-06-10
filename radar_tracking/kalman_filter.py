# kalman_filter.py

"""
Kalman filter implementation for radar object tracking.
Adapted for 2D position and velocity tracking.
"""
import numpy as np
from typing import Tuple, List
from copy import deepcopy


class RadarKalmanFilter:
    """
    Kalman filter for tracking radar objects in 2D space with variable time steps.

    State vector: [x, y, vx, vy] (position and velocity)
    Measurement vector: [x, y] (position only)
    """

    def __init__(self, base_dt: float = 0.1):
        """
        Initialize Kalman filter.

        Args:
            base_dt: Base time step for process noise tuning (seconds)
        """
        self.base_dt = base_dt
        self.dim_x = 4  # State dimension: [x, y, vx, vy]
        self.dim_z = 2  # Measurement dimension: [x, y]

        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Base process noise magnitude
        self.q = 10.0  # Adjust based on expected acceleration

        # Measurement noise covariance matrix
        self.R = np.eye(2) * 0.5  # 0.5 meter standard deviation

        # Initial state covariance matrix
        self.P_init = np.eye(4) * 50.0

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
               measurement: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update state estimate with new measurement.

        Args:
            state: Predicted state vector
            covariance: Predicted state covariance matrix
            measurement: New measurement (x, y)

        Returns:
            Tuple of (updated_state, updated_covariance)
        """
        # Convert measurement to numpy array
        z = np.array([measurement[0], measurement[1]])

        # Innovation: y = z - H * x_{k|k-1}
        innovation = z - self.H @ state

        # Innovation covariance: S = H * P_{k|k-1} * H^T + R
        innovation_cov = self.H @ covariance @ self.H.T + self.R

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