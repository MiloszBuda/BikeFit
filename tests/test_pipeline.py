from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import pytest

from bikefit.models.base import BasePoseEstimator
from bikefit.models.halpe26 import HALPE26_KEYPOINTS
from bikefit.pipeline.video import VideoProcessor, WebCompatibleVideoWriter


class MockCyclistPoseEstimator(BasePoseEstimator):
    """Deterministic mock pose estimator for testing."""

    def __init__(self):
        self._step = 0

    @property
    def keypoint_names(self) -> List[str]:
        return HALPE26_KEYPOINTS

    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._step += 1
        num_kpts = len(HALPE26_KEYPOINTS)
        kpts = np.zeros((1, num_kpts, 2), dtype=np.float32)
        scores = np.ones((1, num_kpts), dtype=np.float32) * 0.95

        # Place keypoints around realistic pixel locations
        for i in range(num_kpts):
            kpts[0, i] = [100.0 + i * 5, 150.0 + (self._step % 10) * 2]

        return kpts, scores


def create_synthetic_video(file_path: Path, num_frames: int = 15, fps: int = 30) -> None:
    """Create a small valid MP4 video for pipeline testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(file_path), fourcc, fps, (320, 240))
    for i in range(num_frames):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        # Draw a moving circle
        cv2.circle(img, (50 + i * 5, 100), 10, (255, 255, 255), -1)
        writer.write(img)
    writer.release()


def test_video_processor_pipeline(tmp_path: Path):
    video_path = tmp_path / "synthetic.mp4"
    out_video_path = tmp_path / "annotated.mp4"
    create_synthetic_video(video_path, num_frames=12, fps=30)

    estimator = MockCyclistPoseEstimator()
    processor = VideoProcessor(pose_estimator=estimator)

    results, metadata = processor.process_video(
        video_path=video_path,
        crank=(160.0, 180.0),
        output_video_path=out_video_path,
        max_frames=10,
    )

    assert len(results) == 10
    assert metadata.width == 320
    assert metadata.height == 240
    assert metadata.fps == 30.0

    # Check keypoints inside first result
    first_res = results[0]
    assert first_res.frame_idx == 0
    assert first_res.crank_center == (160.0, 180.0)
    assert "right_knee" in first_res.keypoints
    assert first_res.keypoints["right_knee"].score == pytest.approx(0.95)

    # Check annotated video was written
    assert out_video_path.exists()
    assert out_video_path.stat().st_size > 0


def test_web_compatible_video_writer(tmp_path: Path):
    out_path = tmp_path / "h264_test.mp4"
    writer = WebCompatibleVideoWriter(out_path, fps=30.0, width=320, height=240)
    for _ in range(5):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        writer.write_frame_bgr(frame)
    writer.release()

    assert out_path.exists()
    assert out_path.stat().st_size > 500

    # Verify OpenCV can read it back
    cap = cv2.VideoCapture(str(out_path))
    assert cap.isOpened()
    ret, frame = cap.read()
    assert ret
    assert frame.shape == (240, 320, 3)
    cap.release()


def test_video_processor_missing_file():
    processor = VideoProcessor(pose_estimator=MockCyclistPoseEstimator())
    with pytest.raises(FileNotFoundError):
        processor.process_video("non_existent_file.mp4")
