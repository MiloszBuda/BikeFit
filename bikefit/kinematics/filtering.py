"""Signal filtering, outlier rejection, and interpolation for joint trajectories."""

from typing import Tuple
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def interpolate_missing_coordinates(
    coords: np.ndarray,
    scores: np.ndarray,
    score_threshold: float = 0.35,
) -> np.ndarray:
    """
    Replace coordinates with confidence below threshold with interpolated values.

    Args:
        coords: 1D array of coordinate values (e.g. x or y over time)
        scores: 1D array of confidence scores for each frame
        score_threshold: minimum confidence to consider a coordinate valid

    Returns:
        Interpolated 1D numpy array with no NaNs (unless all points were invalid).
    """
    clean = coords.astype(float).copy()
    low_conf_mask = (scores < score_threshold) | np.isnan(clean) | np.isinf(clean)
    clean[low_conf_mask] = np.nan

    series = pd.Series(clean)
    # If all values are NaN, return zeros
    if series.isna().all():
        return np.zeros_like(clean)

    # Linear interpolation + boundary backfill / forward fill
    interpolated = series.interpolate(method="linear").bfill().ffill()
    return interpolated.to_numpy()


def butterworth_lowpass_filter(
    data: np.ndarray,
    fps: float,
    cutoff_hz: float = 5.0,
    order: int = 2,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth lowpass filter to remove tracking jitter
    without shifting peak locations (critical for accurate Bottom/Top Dead Center).

    Args:
        data: 1D array of coordinates or angles
        fps: sampling rate / video frame rate
        cutoff_hz: lowpass cutoff frequency in Hz (cycling motions typically < 5 Hz)
        order: filter order (default 2 -> 4th order equivalent with filtfilt)

    Returns:
        Filtered 1D numpy array
    """
    if len(data) < 3 * (order * 2) or fps <= 0:
        return data  # Not enough samples to filter reliably

    nyquist = 0.5 * fps
    norm_cutoff = min(cutoff_hz / nyquist, 0.99)
    if norm_cutoff <= 0:
        return data

    b, a = butter(order, norm_cutoff, btype="low", analog=False)
    filtered = filtfilt(b, a, data)
    return filtered


def filter_trajectory(
    coords: np.ndarray,
    scores: np.ndarray,
    fps: float,
    cutoff_hz: float = 5.0,
    score_threshold: float = 0.35,
) -> np.ndarray:
    """
    Complete cleaning pipeline:
    1. Threshold & interpolate low confidence points
    2. Zero-phase Butterworth lowpass filter
    """
    interpolated = interpolate_missing_coordinates(coords, scores, score_threshold=score_threshold)
    filtered = butterworth_lowpass_filter(interpolated, fps=fps, cutoff_hz=cutoff_hz)
    return filtered
