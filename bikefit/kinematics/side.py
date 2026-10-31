"""Automatic and manual side selection for sagittal bike fitting analysis."""

from typing import List
import numpy as np

from bikefit.models.base import FramePoseResult
from bikefit.models.halpe26 import SIDE_KEYPOINTS


def detect_facing_side(results: List[FramePoseResult]) -> str:
    """
    Detect whether the rider's right or left side is facing the camera.

    Compares average confidence scores of right vs left limb keypoints.
    The side facing the camera has consistently higher scores and fewer occlusions.

    Returns:
        'right' or 'left'
    """
    if not results:
        return "right"  # default fallback

    right_scores: List[float] = []
    left_scores: List[float] = []

    right_kpt_names = list(SIDE_KEYPOINTS["right"].values())
    left_kpt_names = list(SIDE_KEYPOINTS["left"].values())

    for res in results:
        for name in right_kpt_names:
            kpt = res.get(name)
            if kpt:
                right_scores.append(kpt.score)

        for name in left_kpt_names:
            kpt = res.get(name)
            if kpt:
                left_scores.append(kpt.score)

    mean_right = float(np.mean(right_scores)) if right_scores else 0.0
    mean_left = float(np.mean(left_scores)) if left_scores else 0.0

    return "right" if mean_right >= mean_left else "left"
