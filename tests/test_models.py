import numpy as np
from bikefit.models.base import FramePoseResult, Keypoint
from bikefit.models.halpe26 import HALPE26_KEYPOINTS, SIDE_KEYPOINTS
from bikefit.models.rtmpose import RTMPoseEstimator


def test_keypoint_validity():
    kp = Keypoint("test", 10.0, 20.0, 0.9)
    assert kp.is_valid
    assert kp.name == "test"

    kp_nan = Keypoint("nan_pt", float("nan"), 20.0, 0.0)
    assert not kp_nan.is_valid


def test_frame_pose_result_get():
    res = FramePoseResult(
        frame_idx=1,
        timestamp_s=0.033,
        keypoints={"left_knee": Keypoint("left_knee", 50.0, 80.0, 0.85)},
    )
    assert res.get("left_knee") is not None
    assert res.get("left_knee").x == 50.0
    assert res.get("right_knee") is None


def test_halpe26_keypoints_coverage():
    assert len(HALPE26_KEYPOINTS) == 26
    required_joints = [
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_big_toe", "right_big_toe",
        "left_heel", "right_heel",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "neck", "head",
    ]
    for joint in required_joints:
        assert joint in HALPE26_KEYPOINTS

    assert "right" in SIDE_KEYPOINTS
    assert "left" in SIDE_KEYPOINTS


def test_rtmpose_select_primary_person():
    # Empty
    kpts_empty = np.empty((0, 26, 2))
    scores_empty = np.empty((0, 26))
    k, s = RTMPoseEstimator.select_primary_person(kpts_empty, scores_empty)
    assert k.shape == (26, 2)
    assert s.shape == (26,)

    # Two persons: one with high score, one with low
    kpts_two = np.zeros((2, 26, 2))
    scores_two = np.array([[0.2] * 26, [0.8] * 26])
    kpts_two[1, :, :] = 100.0

    # Without reference point: selects highest mean score (index 1)
    k, s = RTMPoseEstimator.select_primary_person(kpts_two, scores_two)
    assert np.allclose(k, 100.0)

    # With reference point near person 0 (at 0,0)
    kpts_two[0, :, :] = 5.0
    scores_two[0, :] = 0.7
    k, s = RTMPoseEstimator.select_primary_person(kpts_two, scores_two, reference_point=(0.0, 0.0))
    assert np.allclose(k, 5.0)
