import numpy as np
from typing import List, Tuple, Optional
from radar_tracking.data_structures import Detection, Track, EgoMotion


class EgoMotionCompensator:
    """
    Compensates for ego vehicle motion in radar detections and tracks.
    """

    def __init__(self, radar_mounting_angle: float = 0.0):
        """
        Initialize ego motion compensator.

        Args:
            radar_mounting_angle: Radar mounting angle relative to vehicle (radians)
        """
        self.radar_mounting_angle = radar_mounting_angle

    def compensate_detection(self, detection: Detection, ego_motion: EgoMotion,
                             dt: float) -> Detection:
        """
        Compensate a single detection for ego motion.

        Args:
            detection: Original detection
            ego_motion: Ego vehicle motion data
            dt: Time elapsed since detection

        Returns:
            Motion-compensated detection
        """
        # Get original position
        x, y = detection.cartesian_pos

        # Calculate ego motion displacement
        # For small time intervals, we can use simple motion model
        ego_dx = ego_motion.speed_mps * dt * np.sin(ego_motion.yaw_rate_rps * dt)
        ego_dy = ego_motion.speed_mps * dt * np.cos(ego_motion.yaw_rate_rps * dt)
        ego_dtheta = ego_motion.yaw_rate_rps * dt

        # Rotate and translate to compensate for ego motion
        cos_theta = np.cos(ego_dtheta)
        sin_theta = np.sin(ego_dtheta)

        # Apply rotation
        x_rot = x * cos_theta - y * sin_theta
        y_rot = x * sin_theta + y * cos_theta

        # Apply translation
        x_comp = x_rot - ego_dx
        y_comp = y_rot - ego_dy

        # Create compensated detection
        comp_detection = Detection(
            range_m=detection.range_m,
            azimuth_rad=detection.azimuth_rad,
            confidence=detection.confidence,
            timestamp=detection.timestamp,
            frame_id=detection.frame_id,
            cartesian_pos=(x_comp, y_comp),
            ego_speed_mps=ego_motion.speed_mps,
            ego_yaw_rate_rps=ego_motion.yaw_rate_rps,
            ego_steering_rad=ego_motion.steering_rad
        )

        # Update polar coordinates to match compensated position
        from radar_tracking.coordinate_transforms import cartesian_to_polar
        comp_detection.range_m, comp_detection.azimuth_rad = cartesian_to_polar(x_comp, y_comp)

        return comp_detection

    def compensate_detections(self, detections: List[Detection],
                              ego_motion: EgoMotion, dt: float) -> List[Detection]:
        """Compensate multiple detections for ego motion."""
        return [self.compensate_detection(det, ego_motion, dt) for det in detections]

    def predict_with_ego_motion(self, state: np.ndarray, ego_motion: EgoMotion,
                                dt: float) -> np.ndarray:
        """
        Predict state with ego motion compensation.

        Args:
            state: Current state [x, y, vx, vy]
            ego_motion: Ego vehicle motion
            dt: Time step

        Returns:
            Predicted state accounting for ego motion
        """
        # Copy state
        pred_state = state.copy()

        # Calculate ego motion effects
        v_ego = ego_motion.speed_mps
        omega = ego_motion.yaw_rate_rps

        # For a turning vehicle, the apparent velocity of static objects is:
        # v_apparent = -v_ego * [sin(theta), cos(theta)] - omega * [-y, x]

        # Position prediction with ego compensation
        pred_state[0] = state[0] + state[2] * dt  # x position
        pred_state[1] = state[1] + state[3] * dt  # y position

        # Adjust for ego motion (simplified for small dt)
        pred_state[0] -= v_ego * np.sin(omega * dt) * dt
        pred_state[1] -= v_ego * (1 - np.cos(omega * dt))

        # Rotate due to ego yaw
        cos_omega = np.cos(omega * dt)
        sin_omega = np.sin(omega * dt)

        x_rot = pred_state[0] * cos_omega - pred_state[1] * sin_omega
        y_rot = pred_state[0] * sin_omega + pred_state[1] * cos_omega

        pred_state[0] = x_rot
        pred_state[1] = y_rot

        # Velocity stays the same in world frame
        # (assuming tracked objects maintain their velocity)

        return pred_state