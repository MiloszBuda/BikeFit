import numpy as np
import pytest

from bikefit.kinematics.filtering import (
    butterworth_lowpass_filter,
    filter_trajectory,
    interpolate_missing_coordinates,
)


def test_interpolate_missing_coordinates():
    # Signal with middle value missing (low score)
    coords = np.array([10.0, 999.0, 30.0])
    scores = np.array([0.9, 0.1, 0.9])
    interpolated = interpolate_missing_coordinates(coords, scores, score_threshold=0.35)
    assert interpolated[1] == pytest.approx(20.0)


def test_interpolate_all_nan():
    coords = np.array([float("nan"), float("nan")])
    scores = np.array([0.1, 0.1])
    res = interpolate_missing_coordinates(coords, scores)
    assert np.all(res == 0.0)


def test_butterworth_lowpass_filter_removes_jitter():
    # 30 fps, 1 second of 1.5 Hz cycling motion + 12 Hz jitter
    t = np.linspace(0, 1, 30, endpoint=False)
    cycling_signal = 100.0 + 20.0 * np.sin(2 * np.pi * 1.5 * t)
    jitter = 5.0 * np.sin(2 * np.pi * 12.0 * t)
    noisy_signal = cycling_signal + jitter

    filtered = butterworth_lowpass_filter(noisy_signal, fps=30.0, cutoff_hz=4.0)

    # Filtered signal should be much closer to pure cycling signal than noisy signal
    noisy_error = np.mean((noisy_signal - cycling_signal) ** 2)
    filtered_error = np.mean((filtered - cycling_signal) ** 2)

    assert filtered_error < noisy_error * 0.4


def test_filter_trajectory_end_to_end():
    coords = np.linspace(10, 50, 25)
    scores = np.ones(25)
    # introduce a dropout
    scores[10] = 0.0
    coords[10] = 500.0

    cleaned = filter_trajectory(coords, scores, fps=30.0)
    assert len(cleaned) == 25
    assert not np.any(np.isnan(cleaned))
    # Dropout should be smoothed close to the linear trend
    assert abs(cleaned[10] - 26.66) < 5.0
