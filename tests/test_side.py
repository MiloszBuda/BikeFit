from bikefit.kinematics.side import detect_facing_side
from bikefit.models.base import FramePoseResult, Keypoint


def test_detect_facing_side_right():
    res = FramePoseResult(
        frame_idx=0,
        timestamp_s=0.0,
        keypoints={
            "right_knee": Keypoint("right_knee", 100, 200, 0.95),
            "right_ankle": Keypoint("right_ankle", 100, 300, 0.90),
            "left_knee": Keypoint("left_knee", 100, 200, 0.15),
            "left_ankle": Keypoint("left_ankle", 100, 300, 0.10),
        },
    )
    side = detect_facing_side([res])
    assert side == "right"


def test_detect_facing_side_left():
    res = FramePoseResult(
        frame_idx=0,
        timestamp_s=0.0,
        keypoints={
            "right_knee": Keypoint("right_knee", 100, 200, 0.10),
            "right_ankle": Keypoint("right_ankle", 100, 300, 0.15),
            "left_knee": Keypoint("left_knee", 100, 200, 0.90),
            "left_ankle": Keypoint("left_ankle", 100, 300, 0.92),
        },
    )
    side = detect_facing_side([res])
    assert side == "left"
