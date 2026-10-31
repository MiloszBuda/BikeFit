import numpy as np
from bikefit.io.visualization import VideoVisualizer
from bikefit.models.base import FramePoseResult, Keypoint


def test_visualizer_draw_frame():
    visualizer = VideoVisualizer(kpt_score_threshold=0.3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = FramePoseResult(
        frame_idx=10,
        timestamp_s=0.33,
        keypoints={
            "right_hip": Keypoint("right_hip", 300.0, 200.0, 0.9),
            "right_knee": Keypoint("right_knee", 320.0, 300.0, 0.9),
            "right_ankle": Keypoint("right_ankle", 310.0, 400.0, 0.9),
        },
        crank_center=(310.0, 420.0),
    )

    rendered = visualizer.draw_frame(frame, result, side_preference="right")
    assert rendered.shape == frame.shape
    # Ensure drawing happened (non-zero pixels)
    assert np.count_nonzero(rendered) > 0
