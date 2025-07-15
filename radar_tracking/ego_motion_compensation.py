# radar_tracking/ego_motion_compensation.py

import numpy as np
from typing import Tuple, Optional
from radar_tracking.data_structures import OdometryData

class EgoMotionCompensator:
    """Ego motion compensation for predicted track states."""

    def __init__(self, radar_offset_x=0.0, radar_offset_y=3.5):
        self.radar_offset = np.array([radar_offset_x, radar_offset_y])

    def integrate_ego_motion(self, odometry: OdometryData, dt: float) -> Tuple[float, float, float, float]:
        if dt <= 0:
            return 0.0, 0.0, 0.0, 0.0

        v = odometry.speed_mps
        omega = odometry.yaw_rate_rad_s

        delta_psi = omega * dt

        if abs(omega) < 1e-6:  # Straight motion
            delta_x = 0.0
            delta_y = v * dt
        else:
            radius = v / omega
            delta_x = radius * (1 - np.cos(delta_psi))
            delta_y = radius * np.sin(delta_psi)

        return delta_x, delta_y, delta_psi, v

    def compensate_track_state(self, state: np.ndarray,
                               odometry: OdometryData,
                               dt: float) -> np.ndarray:
        if dt <= 0:
            return state

        # Ego motion calculation
        delta_x_ego, delta_y_ego, delta_psi, vel_radar = self.integrate_ego_motion(odometry, dt)
        delta_translation = np.array([delta_x_ego, delta_y_ego])

        # Rotation matrix for coordinate transformation
        cos_psi = np.cos(delta_psi)
        sin_psi = np.sin(delta_psi)
        R = np.array([
            [cos_psi, sin_psi],
            [-sin_psi, cos_psi]
        ])

        # Position compensation
        pos_obj = np.array([state[0], state[1]])
        pos_vehicle = pos_obj + self.radar_offset
        pos_vehicle_comp = R @ (pos_vehicle - delta_translation)
        pos_radar_comp = pos_vehicle_comp - self.radar_offset

        compensated_state = state.copy()
        compensated_state[0:2] = pos_radar_comp

        # Velocity compensation: rotate relative velocities to new radar frame
        if len(state) > 2:
            vel_relative = np.array([state[2], state[3]])
            vel_relative_rotated = R @ vel_relative
            compensated_state[2:4] = vel_relative_rotated

        return compensated_state