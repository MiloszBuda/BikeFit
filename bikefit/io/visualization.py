"""Visualization and video rendering for bike fitting keypoints and skeletons."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from bikefit.models.base import FramePoseResult, Keypoint
from bikefit.models.halpe26 import HALPE26_SKELETON_PAIRS


# High visibility BGR color scheme
COLOR_RIGHT = (245, 130, 48)     # Bright orange/amber for right side
COLOR_LEFT = (60, 180, 75)       # Bright green for left side
COLOR_TORSO = (255, 225, 25)     # Yellow
COLOR_HEAD = (230, 25, 75)       # Red
COLOR_CRANK = (0, 255, 255)      # Cyan for bottom bracket
COLOR_TEXT = (255, 255, 255)     # White


class VideoVisualizer:
    """Renders pose skeletons, keypoints, crank reticle, and informational overlays."""

    def __init__(
        self,
        kpt_score_threshold: float = 0.35,
        line_thickness: int = 2,
        point_radius: int = 4,
    ) -> None:
        self.kpt_score_threshold = kpt_score_threshold
        self.line_thickness = line_thickness
        self.point_radius = point_radius

    def _get_bone_color(self, p1: str, p2: str) -> Tuple[int, int, int]:
        if "right" in p1 and "right" in p2:
            return COLOR_RIGHT
        if "left" in p1 and "left" in p2:
            return COLOR_LEFT
        if "head" in p1 or "head" in p2 or "neck" in p1 or "neck" in p2 or "ear" in p1 or "eye" in p1:
            return COLOR_HEAD
        return COLOR_TORSO

    def _get_point_color(self, name: str) -> Tuple[int, int, int]:
        if "right" in name:
            return COLOR_RIGHT
        if "left" in name:
            return COLOR_LEFT
        if "head" in name or "neck" in name or "nose" in name or "ear" in name or "eye" in name:
            return COLOR_HEAD
        return COLOR_TORSO

    def draw_frame(
        self,
        frame: np.ndarray,
        result: FramePoseResult,
        side_preference: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw keypoints, skeleton bones, and crank center on an image frame.
        """
        canvas = frame.copy()
        kpts = result.keypoints

        # 1. Draw bones / skeleton connections
        for p1_name, p2_name in HALPE26_SKELETON_PAIRS:
            k1 = kpts.get(p1_name)
            k2 = kpts.get(p2_name)
            if not k1 or not k2:
                continue
            if k1.score < self.kpt_score_threshold or k2.score < self.kpt_score_threshold:
                continue

            pt1 = (int(round(k1.x)), int(round(k1.y)))
            pt2 = (int(round(k2.x)), int(round(k2.y)))
            color = self._get_bone_color(p1_name, p2_name)
            cv2.line(canvas, pt1, pt2, color, self.line_thickness, cv2.LINE_AA)

        # 2. Draw keypoint circles
        for name, kpt in kpts.items():
            if kpt.score < self.kpt_score_threshold:
                continue
            pt = (int(round(kpt.x)), int(round(kpt.y)))
            color = self._get_point_color(name)
            cv2.circle(canvas, pt, self.point_radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, self.point_radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. Draw calibrated crank center
        if result.crank_center is not None:
            cx, cy = int(round(result.crank_center[0])), int(round(result.crank_center[1]))
            # Crank reticle
            cv2.circle(canvas, (cx, cy), 10, COLOR_CRANK, 2, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 2, COLOR_CRANK, -1, cv2.LINE_AA)
            cv2.line(canvas, (cx - 15, cy), (cx + 15, cy), COLOR_CRANK, 1, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy - 15), (cx, cy + 15), COLOR_CRANK, 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                "BB",
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLOR_CRANK,
                1,
                cv2.LINE_AA,
            )

        # 4. HUD / Info overlay in top-left
        time_str = f"{result.timestamp_s:.2f}s"
        hud_text = f"Frame: {result.frame_idx} | Time: {time_str}"
        if side_preference:
            hud_text += f" | Side: {side_preference.upper()}"

        cv2.rectangle(canvas, (10, 10), (320, 42), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            hud_text,
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        return canvas
