"""Geometric calculations for 2D joint angles and orientations."""

from typing import Tuple
import numpy as np


def calculate_angle_3points(
    p1: Tuple[float, float] | np.ndarray,
    vertex: Tuple[float, float] | np.ndarray,
    p2: Tuple[float, float] | np.ndarray,
) -> float:
    """
    Calculate internal angle between three points with `vertex` at the center.

    Angle is returned in degrees in range [0.0, 180.0].
    180.0 corresponds to a completely straight line (full extension).
    """
    v1 = np.array(p1, dtype=float) - np.array(vertex, dtype=float)
    v2 = np.array(p2, dtype=float) - np.array(vertex, dtype=float)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return float("nan")

    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle_rad = np.arccos(cosine)
    return float(np.degrees(angle_rad))


def calculate_angle_with_horizontal(
    p_lower: Tuple[float, float] | np.ndarray,
    p_upper: Tuple[float, float] | np.ndarray,
) -> float:
    """
    Calculate acute angle of a body segment relative to the horizontal plane.

    Useful for torso inclination (hip -> shoulder) and foot angle (heel -> toe).
    In image coordinates, y increases downwards.

    Returns:
        Angle in degrees in [0.0, 90.0] relative to horizontal.
    """
    p1 = np.array(p_lower, dtype=float)
    p2 = np.array(p_upper, dtype=float)

    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])

    if dx < 1e-6 and dy < 1e-6:
        return float("nan")

    angle_rad = np.arctan2(dy, dx)
    return float(np.degrees(angle_rad))


def calculate_segment_length(
    p1: Tuple[float, float] | np.ndarray,
    p2: Tuple[float, float] | np.ndarray,
) -> float:
    """Euclidean distance in pixels between two points."""
    return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
