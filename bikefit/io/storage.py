"""Serialization and storage of keypoints to CSV and Parquet."""

from pathlib import Path
from typing import List, Optional
import pandas as pd

from bikefit.models.base import FramePoseResult


def results_to_dataframe(results: List[FramePoseResult]) -> pd.DataFrame:
    """
    Convert a list of FramePoseResult instances to a tidy long-form pandas DataFrame.

    Columns:
        - frame_idx: integer index of the frame (0-based)
        - timestamp_s: timestamp in seconds
        - keypoint: semantic name of the keypoint
        - x: pixel x-coordinate
        - y: pixel y-coordinate
        - score: model detection confidence (0.0 to 1.0)
        - crank_x: calibrated crank center x (or NaN)
        - crank_y: calibrated crank center y (or NaN)
    """
    rows = []
    for frame_res in results:
        crank_x = frame_res.crank_center[0] if frame_res.crank_center else float("nan")
        crank_y = frame_res.crank_center[1] if frame_res.crank_center else float("nan")

        for kpt_name, kpt in frame_res.keypoints.items():
            rows.append({
                "frame_idx": frame_res.frame_idx,
                "timestamp_s": frame_res.timestamp_s,
                "keypoint": kpt_name,
                "x": kpt.x,
                "y": kpt.y,
                "score": kpt.score,
                "crank_x": crank_x,
                "crank_y": crank_y,
            })

    if not rows:
        return pd.DataFrame(
            columns=[
                "frame_idx",
                "timestamp_s",
                "keypoint",
                "x",
                "y",
                "score",
                "crank_x",
                "crank_y",
            ]
        )

    df = pd.DataFrame(rows)
    df["frame_idx"] = df["frame_idx"].astype(int)
    df["timestamp_s"] = df["timestamp_s"].astype(float)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df["score"] = df["score"].astype(float)
    return df


def save_keypoints(
    results: List[FramePoseResult],
    output_path: Path | str,
    export_format: str = "both",
) -> List[Path]:
    """
    Save keypoint tracking results to disk.

    Args:
        results: List of FramePoseResult objects
        output_path: Base path or file path (e.g. 'results/keypoints.csv' or 'results/keypoints')
        export_format: 'csv', 'parquet', or 'both'

    Returns:
        List of saved file Paths.
    """
    out_path = Path(output_path)
    if out_path.is_dir() or str(output_path).endswith(("\\", "/")):
        out_dir = out_path
        stem = "keypoints"
    else:
        out_dir = out_path.parent
        stem = out_path.stem

    out_dir.mkdir(parents=True, exist_ok=True)

    df = results_to_dataframe(results)
    saved_files: List[Path] = []

    if export_format in ("csv", "both"):
        csv_file = out_dir / f"{stem}.csv"
        df.to_csv(csv_file, index=False)
        saved_files.append(csv_file)

    if export_format in ("parquet", "both"):
        parquet_file = out_dir / f"{stem}.parquet"
        df.to_parquet(parquet_file, index=False)
        saved_files.append(parquet_file)

    return saved_files
