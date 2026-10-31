"""RTMPose pose estimator implementation using rtmlib (Halpe26 BodyWithFeet)."""

from typing import List, Optional, Tuple
import numpy as np

from bikefit.models.base import BasePoseEstimator
from bikefit.models.halpe26 import HALPE26_KEYPOINTS


class RTMPoseEstimator(BasePoseEstimator):
    """
    RTMPose estimator backed by OpenMMLab's rtmlib with Halpe26 format.
    Includes full body + foot keypoints (big toe, small toe, heel) necessary for bike fitting.
    """

    def __init__(
        self,
        mode: str = "balanced",
        device: str = "cpu",
        backend: str = "onnxruntime",
    ) -> None:
        """
        Initialize the RTMPose Halpe26 model.

        Args:
            mode: 'lightweight' (fastest), 'balanced' (recommended), or 'performance' (highest accuracy).
            device: 'cpu' or 'cuda'.
            backend: 'onnxruntime' or 'opencv'.
        """
        try:
            from rtmlib import BodyWithFeet
        except ImportError as e:
            raise ImportError(
                "rtmlib is required for RTMPoseEstimator. Install via `pip install rtmlib`."
            ) from e

        self.mode = mode
        self.device = device
        self.backend = backend
        self._model = BodyWithFeet(
            mode=mode,
            device=device,
            backend=backend,
            to_openpose=False,
        )

    @property
    def keypoint_names(self) -> List[str]:
        return HALPE26_KEYPOINTS

    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on a single frame.

        Returns:
            Tuple of:
                keypoints: (N, 26, 2)
                scores: (N, 26)
        """
        keypoints, scores = self._model(frame)
        if keypoints is None or len(keypoints) == 0:
            return np.empty((0, len(HALPE26_KEYPOINTS), 2), dtype=np.float32), np.empty(
                (0, len(HALPE26_KEYPOINTS)), dtype=np.float32
            )
        return keypoints, scores

    @staticmethod
    def select_primary_person(
        keypoints: np.ndarray,
        scores: np.ndarray,
        reference_point: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select single cyclist from multiple detected persons.
        Prioritizes person closest to the reference point (e.g. crank center) or highest mean score.
        """
        if len(keypoints) == 0:
            return np.zeros((len(HALPE26_KEYPOINTS), 2), dtype=np.float32), np.zeros(
                len(HALPE26_KEYPOINTS), dtype=np.float32
            )
        if len(keypoints) == 1:
            return keypoints[0], scores[0]

        if reference_point is not None:
            ref_x, ref_y = reference_point
            min_dist = float("inf")
            best_idx = 0
            for i, person_kpts in enumerate(keypoints):
                valid_mask = scores[i] > 0.3
                if not np.any(valid_mask):
                    continue
                dists = np.hypot(person_kpts[valid_mask, 0] - ref_x, person_kpts[valid_mask, 1] - ref_y)
                dist = float(np.min(dists))
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            return keypoints[best_idx], scores[best_idx]

        # Fallback to highest mean confidence score
        mean_scores = [float(np.mean(s)) for s in scores]
        best_idx = int(np.argmax(mean_scores))
        return keypoints[best_idx], scores[best_idx]
