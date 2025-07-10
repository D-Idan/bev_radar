import numpy as np

from utils.util import worldToImage


def is_radar_point_in_camera_view(range_m, azimuth_deg):
    """
    Check if a radar detection is within camera view using projection.

    Args:
        range_m: Radar range in meters
        azimuth_deg: Radar azimuth in degrees

    Returns:
        bool: True if the point projects to valid image coordinates
    """
    # Convert to world coordinates
    x = np.sin(np.deg2rad(azimuth_deg)) * range_m
    y = np.cos(np.deg2rad(azimuth_deg)) * range_m

    # Use same projection as visualization
    # Check multiple heights to ensure coverage
    heights = [0, 0.8, 1.6]  # Ground, mid, top of typical object

    for h in heights:
        u, v = worldToImage(-x, y, h)  # Note: -x as in your visualization

        # Check if within image bounds (accounting for scaling)
        u_scaled = u / 2  # Same scaling as in your code
        v_scaled = v / 2

        if 0 <= u_scaled < 960 and 0 <= v_scaled < 540:
            return True

    return False