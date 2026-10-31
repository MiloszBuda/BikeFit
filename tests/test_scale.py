import numpy as np
import pytest

from bikefit.kinematics.scale import calibrate_scale_from_crank, fit_circle_radius


def test_fit_circle_radius_with_known_center():
    # Circle with radius 100 around center (500, 500)
    theta = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    pts = np.column_stack([500.0 + 100.0 * np.cos(theta), 500.0 + 100.0 * np.sin(theta)])

    radius = fit_circle_radius(pts, center=(500.0, 500.0))
    assert radius == pytest.approx(100.0, abs=1e-3)


def test_fit_circle_radius_algebraic_fit():
    # Circle with radius 50 around (200, 200)
    theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    pts = np.column_stack([200.0 + 50.0 * np.cos(theta), 200.0 + 50.0 * np.sin(theta)])

    radius = fit_circle_radius(pts)
    assert radius == pytest.approx(50.0, abs=0.5)


def test_calibrate_scale_from_crank():
    # 172.5 pixels radius for 172.5 mm crank -> 1.0 px/mm
    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    pts = np.column_stack([300.0 + 172.5 * np.cos(theta), 400.0 + 172.5 * np.sin(theta)])

    scale = calibrate_scale_from_crank(pts, crank_center=(300.0, 400.0), crank_arm_mm=172.5)
    assert scale.pixels_per_mm == pytest.approx(1.0, abs=1e-3)
    assert scale.mm_per_pixel == pytest.approx(1.0, abs=1e-3)
    assert scale.px_to_mm(50.0) == pytest.approx(50.0)
