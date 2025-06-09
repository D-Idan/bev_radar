"""
Coordinate transformation functions for radar data.
"""
import numpy as np
from typing import Tuple, List


def polar_to_cartesian(range_m: float, azimuth_rad: float) -> Tuple[float, float]:
    """
    Convert polar coordinates to Cartesian coordinates.

    Args:
        range_m: Range distance in meters
        azimuth_rad: Azimuth angle in radians (0 = North, clockwise positive)

    Returns:
        Tuple of (x, y) coordinates in meters

    Note:
        Assumes standard radar convention: 0° = North, clockwise positive
        Converts to standard Cartesian: x = East, y = North
    """
    x = range_m * np.sin(azimuth_rad)  # East component
    y = range_m * np.cos(azimuth_rad)  # North component
    return (x, y)


def cartesian_to_polar(x: float, y: float) -> Tuple[float, float]:
    """
    Convert Cartesian coordinates to polar coordinates.

    Args:
        x: East coordinate in meters
        y: North coordinate in meters

    Returns:
        Tuple of (range_m, azimuth_rad)
    """
    range_m = np.sqrt(x ** 2 + y ** 2)
    azimuth_rad = np.arctan2(x, y)  # atan2(East, North) for radar convention
    return (range_m, azimuth_rad)


def batch_polar_to_cartesian(ranges: List[float],
                             azimuths: List[float]) -> List[Tuple[float, float]]:
    """
    Convert multiple polar coordinates to Cartesian coordinates.

    Args:
        ranges: List of range distances in meters
        azimuths: List of azimuth angles in radians

    Returns:
        List of (x, y) coordinate tuples
    """
    return [polar_to_cartesian(r, a) for r, a in zip(ranges, azimuths)]


def euclidean_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        pos1: First position (x, y)
        pos2: Second position (x, y)

    Returns:
        Euclidean distance
    """
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)