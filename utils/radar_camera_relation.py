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


def is_within_radar_coverage(azimuth_deg, range_m, radar_config):
    """Check if a point is within radar coverage boundaries."""
    max_range = radar_config.get('max_range', 103.0)
    min_azimuth = radar_config.get('min_azimuth_deg', -90.0)
    max_azimuth = radar_config.get('max_azimuth_deg', 90.0)

    return (range_m <= max_range and
            min_azimuth <= azimuth_deg <= max_azimuth)

def is_within_camera_radar_coverage(azimuth_deg, range_m, radar_config):
    """Check if a point is within radar coverage boundaries."""
    max_range = 103.0
    min_azimuth = -60.0
    max_azimuth = 60.0

    return (range_m <= max_range and
            min_azimuth <= azimuth_deg <= max_azimuth)