"""Manual calibration of the bicycle bottom bracket (crank axis center / suport)."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class CrankCalibration:
    """Bottom bracket / crank center position in image coordinates."""
    x: float
    y: float

    @property
    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_dict(self) -> dict:
        return {"crank_x": self.x, "crank_y": self.y}

    def save(self, path: Path | str) -> None:
        """Save crank calibration to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "CrankCalibration":
        """Load crank calibration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(x=float(data["crank_x"]), y=float(data["crank_y"]))


class CrankCalibrator:
    """Interactive and programmatic calibrator for crank center."""

    @staticmethod
    def from_coordinates(x: float, y: float) -> CrankCalibration:
        """Create calibration from known coordinates."""
        if x < 0 or y < 0:
            raise ValueError(f"Crank coordinates must be non-negative: ({x}, {y})")
        return CrankCalibration(x=float(x), y=float(y))

    @staticmethod
    def calibrate_interactive(
        frame: np.ndarray,
        window_name: str = "BikeFit - Click Crank Center (Suport)",
    ) -> CrankCalibration:
        """
        Interactively pick crank center on the given frame.

        Controls:
            - Left click: place / update crank center
            - Enter / Space: confirm selection
            - 'r': reset selection
            - 'q' / ESC: cancel selection (raises RuntimeError)
        """
        selected_point: Optional[Tuple[int, int]] = None
        confirmed = False
        canceled = False

        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_point
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_point = (x, y)

        display_frame = frame.copy()
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, mouse_callback)

        try:
            while True:
                canvas = display_frame.copy()

                # Instructions header banner
                header = "Click on Crank Axis (Suport). ENTER/SPACE: Confirm | R: Reset | Q/ESC: Cancel"
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (20, 20, 20), -1)
                cv2.putText(
                    canvas,
                    header,
                    (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                if selected_point is not None:
                    px, py = selected_point
                    # Target reticle
                    cv2.circle(canvas, (px, py), 12, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 2, (0, 0, 255), -1, cv2.LINE_AA)
                    cv2.line(canvas, (px - 20, py), (px + 20, py), (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.line(canvas, (px, py - 20), (px, py + 20), (0, 255, 255), 1, cv2.LINE_AA)
                    coords_text = f"Crank: ({px}, {py})"
                    cv2.putText(
                        canvas,
                        coords_text,
                        (px + 15, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(window_name, canvas)
                key = cv2.waitKey(20) & 0xFF

                if key in (13, 32):  # Enter or Space
                    if selected_point is not None:
                        confirmed = True
                        break
                elif key in (ord("r"), ord("R")):
                    selected_point = None
                elif key in (27, ord("q"), ord("Q")):
                    canceled = True
                    break
        finally:
            cv2.destroyWindow(window_name)

        if canceled or not confirmed or selected_point is None:
            raise RuntimeError("Interactive crank calibration was cancelled or no point selected.")

        return CrankCalibration(x=float(selected_point[0]), y=float(selected_point[1]))
