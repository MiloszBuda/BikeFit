import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from test_rules import create_mock_cycle_stats
from bikefit.reporting.html_report import generate_html_report
from bikefit.rules.engine import FittingRuleEngine


def test_generate_html_report(tmp_path: Path):
    stats = create_mock_cycle_stats(knee_flexion_bdc=21.0, hip_angle_tdc=42.0)
    engine = FittingRuleEngine()
    report = engine.evaluate(stats)

    out_file = tmp_path / "test_report.html"
    html = generate_html_report(report, output_path=out_file)

    assert "<!DOCTYPE html>" in html
    assert "BikeFit AI" in html
    assert "Zgięcie kolana w DMP" in html
    assert "Obniż siodełko" in html
    assert out_file.exists()
    assert out_file.stat().st_size > 1000
