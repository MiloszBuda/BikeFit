"""Base definitions and abstract interface for pose estimation models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Keypoint:
    """Individual tracked keypoint."""
    name: str
    x: float
    y: float
    score: float

    @property
    def is_valid(self) -> bool:
        """Check if coordinates are non-NaN and finite."""
        return not (np.isnan(self.x) or np.isnan(self.y) or np.isinf(self.x) or np.isinf(self.y))


@dataclass
class FramePoseResult:
    """Pose estimation output for a single video frame."""
    frame_idx: int
    timestamp_s: float
    keypoints: Dict[str, Keypoint] = field(default_factory=dict)
    crank_center: Optional[Tuple[float, float]] = None
    person_score: float = 0.0

    def get(self, name: str) -> Optional[Keypoint]:
        """Convenience getter for a keypoint by semantic name."""
        return self.keypoints.get(name)


class BasePoseEstimator(ABC):
    """Abstract pose estimator interface."""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run pose estimation on a single BGR image/frame.

        Returns:
            Tuple of:
                - keypoints: np.ndarray of shape (num_persons, num_kpts, 2)
                - scores: np.ndarray of shape (num_persons, num_kpts)
        """
        pass

    @property
    @abstractmethod
    def keypoint_names(self) -> List[str]:
        """List of keypoint names in model index order."""
        pass
