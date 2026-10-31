import numpy as np
import pytest

from bikefit.kinematics.angles import extract_kinematics
from bikefit.models.base import FramePoseResult, Keypoint
from bikefit.models.halpe26 import HALPE26_KEYPOINTS


def generate_mock_cyclist_results(num_frames: int = 30) -> list[FramePoseResult]:
    results = []
    for i in range(num_frames):
        phase = 2 * np.pi * (i / 15.0)  # 2 full pedal cycles
        kpts = {}

        # Right side kinematics
        # Hip at (300, 200)
        kpts["right_hip"] = Keypoint("right_hip", 300.0, 200.0, 0.95)
        # Knee swinging
        knee_x = 330.0 + 20.0 * np.sin(phase)
        knee_y = 320.0 + 30.0 * np.cos(phase)
        kpts["right_knee"] = Keypoint("right_knee", knee_x, knee_y, 0.95)
        # Ankle rotating around pedal
        ank_x = 340.0 + 50.0 * np.sin(phase)
        ank_y = 440.0 + 50.0 * np.cos(phase)
        kpts["right_ankle"] = Keypoint("right_ankle", ank_x, ank_y, 0.95)

        # Torso & Arms
        kpts["right_shoulder"] = Keypoint("right_shoulder", 420.0, 140.0, 0.95)
        kpts["right_elbow"] = Keypoint("right_elbow", 460.0, 190.0, 0.95)
        kpts["right_wrist"] = Keypoint("right_wrist", 490.0, 170.0, 0.95)

        # Foot
        kpts["right_heel"] = Keypoint("right_heel", ank_x - 15.0, ank_y + 10.0, 0.90)
        kpts["right_big_toe"] = Keypoint("right_big_toe", ank_x + 25.0, ank_y + 10.0, 0.90)

        # Center / other side with low score
        for name in HALPE26_KEYPOINTS:
            if name not in kpts:
                kpts[name] = Keypoint(name, 100.0, 100.0, 0.2)

        results.append(
            FramePoseResult(
                frame_idx=i,
                timestamp_s=i / 30.0,
                keypoints=kpts,
                crank_center=(340.0, 440.0),
            )
        )
    return results


def test_extract_kinematics():
    results = generate_mock_cyclist_results(30)
    kinematics = extract_kinematics(results, fps=30.0, side="right")

    assert kinematics.side == "right"
    assert len(kinematics.knee_angles) == 30
    assert len(kinematics.hip_angles) == 30
    assert len(kinematics.torso_angles) == 30
    assert len(kinematics.elbow_angles) == 30

    # Knee flexion relation
    assert np.allclose(kinematics.knee_flexion, 180.0 - kinematics.knee_angles)

    # DataFrame conversion
    df = kinematics.to_dataframe()
    assert len(df) == 30
    assert "knee_angle" in df.columns
    assert "knee_flexion" in df.columns
    assert "torso_angle" in df.columns
    assert "hip_angle" in df.columns
    assert not df.isna().any().any()
