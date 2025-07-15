import cv2
import numpy as np
import pandas as pd

from utils.util import camera_matrix, tvecs, dist_coeffs, rvecs, worldToImage, ImageWidth, ImageHeight


def imageToWorld(u, v, z=0):
    """
    Convert image coordinates to world coordinates at a given height z.
    """
    # Create point array
    pts = np.array([[u, v]], dtype=np.float32)

    # Undistort to get normalized coordinates
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs)

    x_norm = undistorted[0][0][0]
    y_norm = undistorted[0][0][1]

    # Get rotation matrix and its inverse
    rotation_matrix = cv2.Rodrigues(rvecs)[0]
    R_inv = rotation_matrix.T  # Transpose = inverse for rotation matrix

    # Camera position in world coordinates
    cam_pos_world = -R_inv @ tvecs

    # Ray direction in camera coordinates (normalized)
    ray_cam = np.array([x_norm, y_norm, 1.0])

    # Transform ray direction to world coordinates
    ray_world = R_inv @ ray_cam

    # Find intersection with plane z = constant
    if abs(ray_world[2]) < 1e-10:  # Ray parallel to z-plane
        return None, None

    t = (z - cam_pos_world[2]) / ray_world[2]
    world_point = cam_pos_world + t * ray_world

    return world_point[0], world_point[1]


def is_point_in_camera_fov(x, y, z):
    """Check if a 3D point is within camera FOV and image bounds"""
    u, v = worldToImage(x, y, z)
    return 0 <= u < ImageWidth and 0 <= v < ImageHeight


def cameraBoxToRadar(u1, v1, u2, v2, vehicle_height=1.6):
    """
    Convert camera bounding box to radar coordinates using vehicle height.
    u1,v1: bottom-left corner in image
    u2,v2: top-right corner in image
    vehicle_height: assumed height of vehicle (default 1.6m as used in util.py)
    """
    # Get the center bottom point of the bounding box
    u_bottom = (u1 + u2) / 2
    v_bottom = v1  # v1 is the bottom edge

    # Get the center top point of the bounding box
    u_top = (u1 + u2) / 2
    v_top = v2  # v2 is the top edge

    # Get world coordinates at ground level (z=0)
    x_bottom, y_bottom = imageToWorld(u_bottom, v_bottom, z=0)

    if x_bottom is None or y_bottom is None:
        return None, None

    # Get world coordinates at vehicle height
    x_top, y_top = imageToWorld(u_top, v_top, z=vehicle_height)

    if x_top is None or y_top is None:
        # If top point fails, fall back to bottom only
        x_center, y_center = x_bottom, y_bottom
    else:
        # The radar likely detects somewhere between bottom and center of vehicle
        # Use a weighted average favoring the bottom (where radar reflections are stronger)
        weight_bottom = 0.7
        weight_top = 0.3
        x_center = weight_bottom * x_bottom + weight_top * x_top
        y_center = weight_bottom * y_bottom + weight_top * y_top

    # Check if point is in camera FOV
    if not is_point_in_camera_fov(x_center, y_center, z=vehicle_height / 2):
        return 0.0, 0.0

    # Convert to polar coordinates
    range_m = np.sqrt(x_center ** 2 + y_center ** 2)
    azimuth_rad = np.arctan2(x_center, y_center)
    azimuth_deg = np.degrees(azimuth_rad)

    return range_m, -azimuth_deg



# Calculate radar coordinates for each bounding box
def calculate_radar_coords(row):
    """Calculate radar coordinates from bounding box"""
    if pd.isna(row['x1_pix']) or pd.isna(row['y1_pix']) or pd.isna(row['x2_pix']) or pd.isna(row['y2_pix']):
        return pd.Series({'radar_R_m': None, 'radar_A_deg': None})

    # Note: In image coordinates, y increases downward
    # So y2 > y1 means y2 is the bottom, y1 is the top
    u1, v1 = int(row['x1_pix']), int(row['y2_pix'])  # Left-bottom
    u2, v2 = int(row['x2_pix']), int(row['y1_pix'])  # Right-top

    # Calculate radar coordinates with vehicle height consideration
    range_m, azimuth_deg = cameraBoxToRadar(u1, v1, u2, v2, vehicle_height=1.6)#1.6)

    # if range_m <= 0.0 or range_m >= 103.0 or azimuth_deg <= -60.0 or azimuth_deg >= 60.0:
    #     range_m, azimuth_deg = None, None

    return pd.Series({'radar_R_m': range_m, 'radar_A_deg': azimuth_deg})
