from pathlib import Path
import pandas as pd
from bikefit.io.storage import results_to_dataframe, save_keypoints
from bikefit.models.base import FramePoseResult, Keypoint


def test_results_to_dataframe_empty():
    df = results_to_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "frame_idx" in df.columns
    assert "keypoint" in df.columns


def test_results_to_dataframe_with_data():
    res = FramePoseResult(
        frame_idx=0,
        timestamp_s=0.0,
        keypoints={
            "right_knee": Keypoint("right_knee", 100.0, 200.0, 0.95),
            "right_ankle": Keypoint("right_ankle", 105.0, 300.0, 0.88),
        },
        crank_center=(120.0, 310.0),
        person_score=0.91,
    )
    df = results_to_dataframe([res])
    assert len(df) == 2
    assert set(df["keypoint"]) == {"right_knee", "right_ankle"}
    assert df.loc[df["keypoint"] == "right_knee", "x"].iloc[0] == 100.0
    assert df.loc[df["keypoint"] == "right_knee", "crank_x"].iloc[0] == 120.0


def test_save_keypoints_both_formats(tmp_path: Path):
    res = FramePoseResult(
        frame_idx=0,
        timestamp_s=0.0,
        keypoints={"nose": Keypoint("nose", 50.0, 60.0, 0.9)},
        crank_center=(100.0, 150.0),
    )
    out_base = tmp_path / "test_out"
    saved = save_keypoints([res], out_base, export_format="both")

    assert len(saved) == 2
    csv_file = tmp_path / "test_out.csv"
    parquet_file = tmp_path / "test_out.parquet"

    assert csv_file.exists()
    assert parquet_file.exists()

    df_csv = pd.read_csv(csv_file)
    df_parquet = pd.read_parquet(parquet_file)

    assert len(df_csv) == 1
    assert len(df_parquet) == 1
    assert df_csv["keypoint"].iloc[0] == "nose"
