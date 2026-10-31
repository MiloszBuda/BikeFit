"""Video processing pipeline: orchestrates pose tracking, crank calibration, and output."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

from bikefit.calibration.crank import CrankCalibration, CrankCalibrator
from bikefit.io.visualization import VideoVisualizer
from bikefit.models.base import BasePoseEstimator, FramePoseResult, Keypoint
from bikefit.models.rtmpose import RTMPoseEstimator


@dataclass
class VideoMetadata:
    """Basic video stream properties."""
    fps: float
    total_frames: int
    width: int
    height: int
    duration_s: float


class VideoProcessor:
    """Processes video frames through pose estimation and generates tracking records."""

    def __init__(
        self,
        pose_estimator: BasePoseEstimator,
        visualizer: Optional[VideoVisualizer] = None,
    ) -> None:
        self.pose_estimator = pose_estimator
        self.visualizer = visualizer or VideoVisualizer()

    @staticmethod
    def get_metadata(cap: cv2.VideoCapture) -> VideoMetadata:
        """Extract metadata from opened VideoCapture."""
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 30.0  # standard fallback
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = total_frames / fps if fps > 0 else 0.0
        return VideoMetadata(
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            duration_s=duration_s,
        )

    def process_video(
        self,
        video_path: Path | str,
        crank: Optional[CrankCalibration | Tuple[float, float]] = None,
        interactive_crank: bool = False,
        output_video_path: Optional[Path | str] = None,
        side_preference: Optional[str] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[FramePoseResult], VideoMetadata]:
        """
        Run the complete Stage 1 video processing pipeline.

        Args:
            video_path: Path to input MP4/video file.
            crank: Optional CrankCalibration or (x, y) coordinates.
            interactive_crank: If True, prompt user on first frame if crank is not provided.
            output_video_path: Optional path to write annotated output video.
            side_preference: 'right', 'left', or None.
            max_frames: Optional frame limit for testing or fast preview.
            progress_callback: Optional callback(current_frame, total_frames).

        Returns:
            Tuple of (list of FramePoseResults, VideoMetadata).
        """
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file}")

        try:
            metadata = self.get_metadata(cap)
            total_frames = (
                min(metadata.total_frames, max_frames)
                if max_frames and metadata.total_frames > 0
                else (max_frames or metadata.total_frames)
            )

            # Read first frame to calibrate crank if interactive
            crank_point: Optional[Tuple[float, float]] = None
            if isinstance(crank, CrankCalibration):
                crank_point = crank.as_tuple
            elif isinstance(crank, (tuple, list)) and len(crank) == 2:
                crank_point = (float(crank[0]), float(crank[1]))
            elif interactive_crank:
                ret, first_frame = cap.read()
                if not ret:
                    raise RuntimeError("Failed to read first frame for crank calibration.")
                calib = CrankCalibrator.calibrate_interactive(first_frame)
                crank_point = calib.as_tuple
                # Rewind to start of video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Setup video writer if output path requested
            writer: Optional[cv2.VideoWriter] = None
            if output_video_path:
                out_path = Path(output_video_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(out_path),
                    fourcc,
                    metadata.fps,
                    (metadata.width, metadata.height),
                )

            results: List[FramePoseResult] = []
            frame_idx = 0

            with tqdm(total=total_frames if total_frames > 0 else None, desc="Processing video") as pbar:
                while cap.isOpened():
                    if max_frames and frame_idx >= max_frames:
                        break

                    ret, frame = cap.read()
                    if not ret:
                        break

                    timestamp_s = frame_idx / metadata.fps if metadata.fps > 0 else 0.0

                    # Run model inference
                    kpts_batch, scores_batch = self.pose_estimator.predict(frame)

                    # Select primary cyclist
                    if isinstance(self.pose_estimator, RTMPoseEstimator):
                        kpts, scores = RTMPoseEstimator.select_primary_person(
                            kpts_batch, scores_batch, reference_point=crank_point
                        )
                    else:
                        kpts = kpts_batch[0] if len(kpts_batch) > 0 else np.zeros((len(self.pose_estimator.keypoint_names), 2))
                        scores = scores_batch[0] if len(scores_batch) > 0 else np.zeros(len(self.pose_estimator.keypoint_names))

                    # Build keypoint map
                    keypoints_dict: Dict[str, Keypoint] = {}
                    for i, name in enumerate(self.pose_estimator.keypoint_names):
                        keypoints_dict[name] = Keypoint(
                            name=name,
                            x=float(kpts[i, 0]),
                            y=float(kpts[i, 1]),
                            score=float(scores[i]),
                        )

                    mean_score = float(np.mean(scores)) if len(scores) > 0 else 0.0
                    frame_result = FramePoseResult(
                        frame_idx=frame_idx,
                        timestamp_s=timestamp_s,
                        keypoints=keypoints_dict,
                        crank_center=crank_point,
                        person_score=mean_score,
                    )
                    results.append(frame_result)

                    # Render output video frame
                    if writer is not None:
                        annotated = self.visualizer.draw_frame(
                            frame, frame_result, side_preference=side_preference
                        )
                        writer.write(annotated)

                    if progress_callback:
                        progress_callback(frame_idx + 1, total_frames)

                    frame_idx += 1
                    pbar.update(1)

            return results, metadata

        finally:
            cap.release()
            if "writer" in locals() and writer is not None:
                writer.release()
