import pytest
from pathlib import Path
from bikefit.calibration.crank import CrankCalibration, CrankCalibrator


def test_crank_calibration_creation():
    calib = CrankCalibration(x=500.5, y=700.25)
    assert calib.as_tuple == (500.5, 700.25)
    assert calib.to_dict() == {"crank_x": 500.5, "crank_y": 700.25}


def test_crank_calibration_save_and_load(tmp_path: Path):
    calib = CrankCalibration(x=640.0, y=480.0)
    json_path = tmp_path / "crank.json"
    calib.save(json_path)

    loaded = CrankCalibration.load(json_path)
    assert loaded.x == 640.0
    assert loaded.y == 480.0


def test_crank_calibrator_from_coordinates():
    calib = CrankCalibrator.from_coordinates(100, 200)
    assert calib.x == 100.0
    assert calib.y == 200.0

    with pytest.raises(ValueError, match="non-negative"):
        CrankCalibrator.from_coordinates(-10, 20)
