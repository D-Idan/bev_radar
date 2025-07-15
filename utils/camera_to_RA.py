import cv2
import numpy as np
from utils.util import camera_matrix, tvecs, dist_coeffs, rvecs


def imageToWorld(u, v, z=0):
    """
    Convert image coordinates to world coordinates at a given height z.
    """
    # Undistort the point
    pts = np.array([[u, v]], dtype=np.float32)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)

    # Convert to normalized camera coordinates
    x_norm = undistorted[0][0][0]
    y_norm = undistorted[0][0][1]

    # Get rotation matrix
    rotation_matrix = cv2.Rodrigues(rvecs)[0]

    # Create ray in camera coordinates
    ray_cam = np.array([x_norm, y_norm, 1.0])

    # Transform ray to world coordinates
    ray_world = np.linalg.inv(rotation_matrix) @ ray_cam

    # Camera position in world coordinates
    cam_pos = -np.linalg.inv(rotation_matrix) @ tvecs

    # Find intersection with plane at height z
    # Parametric line: P = cam_pos + t * ray_world
    # We want P[2] = z
    t = (z - cam_pos[2]) / ray_world[2]

    # Get intersection point
    world_point = cam_pos + t * ray_world

    return world_point[0], world_point[1]


def cameraBoxToRadar(u1, v1, u2, v2):
    """
    Convert camera bounding box to radar coordinates.
    u1,v1: bottom-left corner in image
    u2,v2: top-right corner in image
    """
    # Get world coordinates of bottom corners (z=0)
    x1_cam, y1_cam = imageToWorld(u1, v1, z=0)
    x2_cam, y2_cam = imageToWorld(u2, v1, z=0)  # Same v coordinate for bottom edge

    # Transform from camera world frame to radar frame
    # Camera X → Radar Y
    # Camera -Y → Radar X
    radar_x1 = -y1_cam
    radar_y1 = x1_cam
    radar_x2 = -y2_cam
    radar_y2 = x2_cam

    # Get center of the box in radar coordinates
    radar_x_center = (radar_x1 + radar_x2) / 2
    radar_y_center = (radar_y1 + radar_y2) / 2

    # Convert to polar coordinates
    range_meters = np.sqrt(radar_x_center ** 2 + radar_y_center ** 2)
    azimuth_radians = np.arctan2(radar_x_center, radar_y_center)
    azimuth_degrees = np.degrees(azimuth_radians)

    return range_meters, azimuth_degrees


def cameraBoxToRadarWithUncertainty(u1, v1, u2, v2):
    """
    Enhanced version that considers the full bounding box area.
    Returns range and azimuth with uncertainty estimates.
    """
    # Sample multiple points along the bottom edge of the box
    num_samples = 5
    u_samples = np.linspace(u1, u2, num_samples)

    ranges = []
    azimuths = []

    for u in u_samples:
        x_cam, y_cam = imageToWorld(u, v1, z=0)

        # Transform to radar frame
        radar_x = -y_cam
        radar_y = x_cam

        # Convert to polar
        r = np.sqrt(radar_x ** 2 + radar_y ** 2)
        a = np.degrees(np.arctan2(radar_x, radar_y))

        ranges.append(r)
        azimuths.append(a)

    # Return mean and standard deviation
    range_mean = np.mean(ranges)
    range_std = np.std(ranges)
    azimuth_mean = np.mean(azimuths)
    azimuth_std = np.std(azimuths)

    return range_mean, azimuth_mean, range_std, azimuth_std