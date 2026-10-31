import math
import numpy as np
import pytest

from bikefit.kinematics.geometry import (
    calculate_angle_3points,
    calculate_angle_with_horizontal,
    calculate_segment_length,
)


def test_calculate_angle_3points_right_angle():
    p1 = (0.0, 10.0)
    vertex = (0.0, 0.0)
    p2 = (10.0, 0.0)
    angle = calculate_angle_3points(p1, vertex, p2)
    assert angle == pytest.approx(90.0, abs=1e-4)


def test_calculate_angle_3points_straight_line():
    p1 = (-10.0, 0.0)
    vertex = (0.0, 0.0)
    p2 = (10.0, 0.0)
    angle = calculate_angle_3points(p1, vertex, p2)
    assert angle == pytest.approx(180.0, abs=1e-4)


def test_calculate_angle_3points_acute_45():
    p1 = (10.0, 10.0)
    vertex = (0.0, 0.0)
    p2 = (10.0, 0.0)
    angle = calculate_angle_3points(p1, vertex, p2)
    assert angle == pytest.approx(45.0, abs=1e-4)


def test_calculate_angle_3points_coincident_points():
    p1 = (0.0, 0.0)
    vertex = (0.0, 0.0)
    p2 = (10.0, 0.0)
    assert math.isnan(calculate_angle_3points(p1, vertex, p2))


def test_calculate_angle_with_horizontal():
    # Horizontal segment
    assert calculate_angle_with_horizontal((0.0, 100.0), (50.0, 100.0)) == pytest.approx(0.0)

    # Vertical segment
    assert calculate_angle_with_horizontal((100.0, 100.0), (100.0, 200.0)) == pytest.approx(90.0)

    # 45 degrees
    assert calculate_angle_with_horizontal((0.0, 0.0), (10.0, 10.0)) == pytest.approx(45.0)


def test_calculate_segment_length():
    assert calculate_segment_length((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
