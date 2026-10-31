import numpy as np
import pytest

from bikefit.kinematics.angles import KinematicsResult
from bikefit.segmentation.cycles import CycleDetector


def create_synthetic_kinematics(num_seconds: float = 4.0, fps: float = 30.0, cadence_rpm: float = 90.0) -> KinematicsResult:
    """Create synthetic kinematics with regular sinusoidal pedal strokes."""
    num_frames = int(num_seconds * fps)
    t = np.linspace(0, num_seconds, num_frames, endpoint=False)
    freq_hz = cadence_rpm / 60.0  # 1.5 Hz for 90 RPM

    # Knee internal angle oscillating between 70 deg (GMP) and 150 deg (DMP)
    knee_angles = 110.0 + 40.0 * np.cos(2 * np.pi * freq_hz * t)
    # Hip angle oscillating between 42 deg (GMP) and 95 deg (DMP)
    hip_angles = 68.5 + 26.5 * np.cos(2 * np.pi * freq_hz * t)
    torso_angles = np.full(num_frames, 45.0)
    elbow_angles = np.full(num_frames, 155.0)
    arm_trunk_angles = np.full(num_frames, 88.0)
    ankle_angles = np.full(num_frames, 90.0)
    foot_angles = np.full(num_frames, 8.0)

    # Dummy trajectories
    trajectories = {
        "right_ankle": np.column_stack([np.full(num_frames, 300.0), 300.0 + 80.0 * np.cos(2 * np.pi * freq_hz * t)])
    }

    return KinematicsResult(
        frame_indices=np.arange(num_frames),
        timestamps_s=t,
        side="right",
        knee_angles=knee_angles,
        hip_angles=hip_angles,
        torso_angles=torso_angles,
        elbow_angles=elbow_angles,
        arm_trunk_angles=arm_trunk_angles,
        ankle_angles=ankle_angles,
        foot_angles=foot_angles,
        filtered_trajectories=trajectories,
    )


def test_cycle_detector_cadence_and_peaks():
    fps = 30.0
    cadence = 90.0
    kinematics = create_synthetic_kinematics(num_seconds=5.0, fps=fps, cadence_rpm=cadence)

    detector = CycleDetector(discard_warmup_cycles=1, discard_cooldown_cycles=1)
    res = detector.segment(kinematics, fps=fps)

    # 5 seconds at 1.5 rev/sec = approx 7.5 revolutions
    assert len(res.all_cycles) >= 6
    assert res.discarded_count == 2
    assert len(res.retained_cycles) == len(res.all_cycles) - 2

    # Cadence should be close to 90 RPM
    assert res.cadence_rpm_median == pytest.approx(cadence, abs=3.0)

    # Knee angle at DMP should be approx 150 deg (flexion ~ 30 deg)
    assert res.knee_angle_bdc_median == pytest.approx(150.0, abs=2.0)
    assert res.knee_flexion_bdc_median == pytest.approx(30.0, abs=2.0)

    # Repeatability std should be very low on clean synthetic data
    assert res.knee_flexion_bdc_std < 1.0


def test_cycle_detector_few_frames():
    # Only 5 frames -> cannot form full cycle
    kinematics = create_synthetic_kinematics(num_seconds=0.2, fps=30.0)
    detector = CycleDetector()
    res = detector.segment(kinematics, fps=30.0)
    assert len(res.all_cycles) == 0
