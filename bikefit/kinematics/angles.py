"""Kinematic angle calculations for cycling biomechanics."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from bikefit.kinematics.filtering import filter_trajectory
from bikefit.kinematics.geometry import calculate_angle_3points, calculate_angle_with_horizontal
from bikefit.kinematics.side import detect_facing_side
from bikefit.models.base import FramePoseResult
from bikefit.models.halpe26 import SIDE_KEYPOINTS


@dataclass
class KinematicsResult:
    """Calculated angles and trajectories over time for a video."""
    frame_indices: np.ndarray
    timestamps_s: np.ndarray
    side: str

    # Joint angles in degrees (internal convention: 180 = straight)
    knee_angles: np.ndarray
    hip_angles: np.ndarray
    torso_angles: np.ndarray
    elbow_angles: np.ndarray
    arm_trunk_angles: np.ndarray
    ankle_angles: np.ndarray
    foot_angles: np.ndarray

    # Filtered 2D pixel coordinates per keypoint: dict[keypoint_name -> ndarray of shape (N, 2)]
    filtered_trajectories: Dict[str, np.ndarray]

    @property
    def knee_flexion(self) -> np.ndarray:
        """Knee flexion angle (0 = straight leg, typical 25-35 deg in Bottom Dead Center)."""
        return 180.0 - self.knee_angles

    @property
    def elbow_flexion(self) -> np.ndarray:
        """Elbow flexion angle."""
        return 180.0 - self.elbow_angles

    def to_dataframe(self) -> pd.DataFrame:
        """Convert angle time series to pandas DataFrame."""
        return pd.DataFrame({
            "frame_idx": self.frame_indices,
            "timestamp_s": self.timestamps_s,
            "knee_angle": self.knee_angles,
            "knee_flexion": self.knee_flexion,
            "hip_angle": self.hip_angles,
            "torso_angle": self.torso_angles,
            "elbow_angle": self.elbow_angles,
            "elbow_flexion": self.elbow_flexion,
            "arm_trunk_angle": self.arm_trunk_angles,
            "ankle_angle": self.ankle_angles,
            "foot_angle": self.foot_angles,
        })


def extract_kinematics(
    results: List[FramePoseResult],
    fps: float,
    side: Optional[str] = None,
    cutoff_hz: float = 5.0,
    score_threshold: float = 0.35,
) -> KinematicsResult:
    """
    Extract, clean, filter, and calculate all joint angles for the rider.

    Args:
        results: List of FramePoseResult from Stage 1
        fps: video frames per second
        side: 'right', 'left', or None (auto-detected)
        cutoff_hz: Butterworth filter cutoff in Hz (default: 5.0)
        score_threshold: minimum confidence to accept keypoint coordinates
    """
    if not results:
        raise ValueError("Cannot extract kinematics from empty results.")

    if side is None or side == "auto":
        side = detect_facing_side(results)

    kpt_names = SIDE_KEYPOINTS[side]
    num_frames = len(results)
    frame_indices = np.array([r.frame_idx for r in results], dtype=int)
    timestamps = np.array([r.timestamp_s for r in results], dtype=float)

    # 1. Collect and filter trajectories for required joints
    filtered_trajectories: Dict[str, np.ndarray] = {}
    needed_joints = set(kpt_names.values()) | {"hip_center", "neck", "head"}

    for joint in needed_joints:
        raw_x = np.zeros(num_frames, dtype=float)
        raw_y = np.zeros(num_frames, dtype=float)
        scores = np.zeros(num_frames, dtype=float)

        for i, res in enumerate(results):
            kp = res.get(joint)
            if kp:
                raw_x[i] = kp.x
                raw_y[i] = kp.y
                scores[i] = kp.score

        filt_x = filter_trajectory(raw_x, scores, fps=fps, cutoff_hz=cutoff_hz, score_threshold=score_threshold)
        filt_y = filter_trajectory(raw_y, scores, fps=fps, cutoff_hz=cutoff_hz, score_threshold=score_threshold)
        filtered_trajectories[joint] = np.column_stack([filt_x, filt_y])

    # 2. Extract specific side keypoint trajectories
    sh_pts = filtered_trajectories[kpt_names["shoulder"]]
    el_pts = filtered_trajectories[kpt_names["elbow"]]
    wr_pts = filtered_trajectories[kpt_names["wrist"]]
    hip_pts = filtered_trajectories[kpt_names["hip"]]
    knee_pts = filtered_trajectories[kpt_names["knee"]]
    ank_pts = filtered_trajectories[kpt_names["ankle"]]
    heel_pts = filtered_trajectories[kpt_names["heel"]]
    toe_pts = filtered_trajectories[kpt_names["big_toe"]]

    # 3. Calculate angles for each frame
    knee_angles = np.zeros(num_frames, dtype=float)
    hip_angles = np.zeros(num_frames, dtype=float)
    torso_angles = np.zeros(num_frames, dtype=float)
    elbow_angles = np.zeros(num_frames, dtype=float)
    arm_trunk_angles = np.zeros(num_frames, dtype=float)
    ankle_angles = np.zeros(num_frames, dtype=float)
    foot_angles = np.zeros(num_frames, dtype=float)

    for i in range(num_frames):
        # Knee: hip - knee - ankle
        knee_angles[i] = calculate_angle_3points(hip_pts[i], knee_pts[i], ank_pts[i])

        # Hip: shoulder - hip - knee
        hip_angles[i] = calculate_angle_3points(sh_pts[i], hip_pts[i], knee_pts[i])

        # Torso inclination: hip -> shoulder vs horizontal
        torso_angles[i] = calculate_angle_with_horizontal(hip_pts[i], sh_pts[i])

        # Elbow: shoulder - elbow - wrist
        elbow_angles[i] = calculate_angle_3points(sh_pts[i], el_pts[i], wr_pts[i])

        # Arm-Trunk: elbow - shoulder - hip
        arm_trunk_angles[i] = calculate_angle_3points(el_pts[i], sh_pts[i], hip_pts[i])

        # Ankle: knee - ankle - big_toe
        ankle_angles[i] = calculate_angle_3points(knee_pts[i], ank_pts[i], toe_pts[i])

        # Foot angle: heel -> toe vs horizontal
        foot_angles[i] = calculate_angle_with_horizontal(heel_pts[i], toe_pts[i])

    return KinematicsResult(
        frame_indices=frame_indices,
        timestamps_s=timestamps,
        side=side,
        knee_angles=knee_angles,
        hip_angles=hip_angles,
        torso_angles=torso_angles,
        elbow_angles=elbow_angles,
        arm_trunk_angles=arm_trunk_angles,
        ankle_angles=ankle_angles,
        foot_angles=foot_angles,
        filtered_trajectories=filtered_trajectories,
    )
