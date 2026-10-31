import sys
from pathlib import Path

# Add tests directory to sys.path if not present
TESTS_DIR = Path(__file__).parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from bikefit.io.plotting import plot_kinematics_summary
from bikefit.kinematics.angles import extract_kinematics
from test_angles import generate_mock_cyclist_results


def test_plot_kinematics_summary(tmp_path: Path):
    results = generate_mock_cyclist_results(25)
    kinematics = extract_kinematics(results, fps=30.0, side="right")

    chart_path = tmp_path / "test_chart.png"
    out = plot_kinematics_summary(kinematics, chart_path)

    assert out.exists()
    assert out.stat().st_size > 5000  # valid image file generated
