"""Scale calibration using bicycle crank arm length."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class ScaleCalibration:
    """Scale calibration factor converting pixel measurements to millimeters."""
    crank_arm_mm: float
    crank_radius_px: float
    pixels_per_mm: float

    @property
    def mm_per_pixel(self) -> float:
        return 1.0 / self.pixels_per_mm if self.pixels_per_mm > 0 else 0.0

    def px_to_mm(self, px: float) -> float:
        return px * self.mm_per_pixel

    def mm_to_px(self, mm: float) -> float:
        return mm * self.pixels_per_mm


def fit_circle_radius(
    points: np.ndarray,
    center: Optional[Tuple[float, float]] = None,
) -> float:
    """
    Fit radius of circular/orbital motion.

    If center is provided (e.g. known bottom bracket), calculates median distance
    of points to the center. Otherwise, performs least-squares circle fit.

    Args:
        points: (N, 2) array of coordinates (e.g. pedal/foot trajectory)
        center: known (x, y) center if available
    """
    if len(points) < 3:
        return 0.0

    if center is not None:
        cx, cy = center
        distances = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
        # 50th percentile (median) is robust against occasional misdetections
        return float(np.median(distances))

    # Algebraic circle fit (Kåsa method)
    x = points[:, 0]
    y = points[:, 1]
    a_mat = np.column_stack([x, y, np.ones_like(x)])
    b_vec = x**2 + y**2
    try:
        c, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        radius = np.sqrt(4 * c[2] + c[0]**2 + c[1]**2) / 2
        return float(radius)
    except Exception:
        return 0.0


def calibrate_scale_from_crank(
    foot_points: np.ndarray,
    crank_center: Tuple[float, float],
    crank_arm_mm: float = 172.5,
) -> ScaleCalibration:
    """
    Compute pixel-to-millimeter scale from circular crank motion.

    Args:
        foot_points: (N, 2) array of tracked foot/pedal coordinates
        crank_center: (x, y) coordinates of the bottom bracket
        crank_arm_mm: known crank length in mm (standard: 165, 170, 172.5, 175)

    Returns:
        ScaleCalibration instance
    """
    if crank_arm_mm <= 0:
        raise ValueError(f"Crank arm length must be positive: {crank_arm_mm} mm")

    radius_px = fit_circle_radius(foot_points, center=crank_center)
    if radius_px <= 1e-3:
        # Fallback if no valid points
        return ScaleCalibration(
            crank_arm_mm=crank_arm_mm,
            crank_radius_px=0.0,
            pixels_per_mm=1.0,
        )

    pixels_per_mm = radius_px / crank_arm_mm
    return ScaleCalibration(
        crank_arm_mm=crank_arm_mm,
        crank_radius_px=radius_px,
        pixels_per_mm=pixels_per_mm,
    )
