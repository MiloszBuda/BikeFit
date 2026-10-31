"""Kinematics, geometry, filtering, and biomechanics package."""

from bikefit.kinematics.angles import KinematicsResult, extract_kinematics
from bikefit.kinematics.filtering import butterworth_lowpass_filter, filter_trajectory, interpolate_missing_coordinates
from bikefit.kinematics.geometry import calculate_angle_3points, calculate_angle_with_horizontal, calculate_segment_length
from bikefit.kinematics.scale import ScaleCalibration, calibrate_scale_from_crank, fit_circle_radius
from bikefit.kinematics.side import detect_facing_side

__all__ = [
    "KinematicsResult",
    "extract_kinematics",
    "butterworth_lowpass_filter",
    "filter_trajectory",
    "interpolate_missing_coordinates",
    "calculate_angle_3points",
    "calculate_angle_with_horizontal",
    "calculate_segment_length",
    "ScaleCalibration",
    "calibrate_scale_from_crank",
    "fit_circle_radius",
    "detect_facing_side",
]
