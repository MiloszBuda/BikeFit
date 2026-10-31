from pathlib import Path
import pytest

from bikefit.reporting.summary import format_markdown_summary, format_text_summary, save_reports
from bikefit.rules.config import FittingConfig
from bikefit.rules.engine import ConfidenceLevel, FittingRuleEngine, RecommendationCategory
from bikefit.segmentation.cycles import CycleMetrics, CycleSegmentationResult


def create_mock_cycle_stats(
    knee_flexion_bdc: float = 30.0,
    hip_angle_tdc: float = 50.0,
    elbow_angle: float = 155.0,
    foot_angle_bdc: float = 5.0,
) -> CycleSegmentationResult:
    dummy_cycle = CycleMetrics(
        cycle_idx=0,
        start_frame=0,
        end_frame=20,
        duration_s=0.66,
        cadence_rpm=90.0,
        bdc_frame=0,
        tdc_frame=10,
        knee_angle_bdc=180.0 - knee_flexion_bdc,
        knee_flexion_bdc=knee_flexion_bdc,
        knee_angle_tdc=70.0,
        knee_angle_range=80.0,
        hip_angle_tdc=hip_angle_tdc,
        hip_angle_bdc=90.0,
        torso_angle_mean=45.0,
        elbow_angle_mean=elbow_angle,
        arm_trunk_angle_mean=88.0,
        ankle_angle_bdc=90.0,
        foot_angle_bdc=foot_angle_bdc,
    )
    return CycleSegmentationResult(
        side="right",
        all_cycles=[dummy_cycle],
        retained_cycles=[dummy_cycle],
        discarded_count=0,
        cadence_rpm_median=90.0,
        cadence_rpm_std=1.0,
        knee_flexion_bdc_median=knee_flexion_bdc,
        knee_flexion_bdc_std=0.8,
        knee_angle_bdc_median=180.0 - knee_flexion_bdc,
        knee_angle_bdc_std=0.8,
        hip_angle_tdc_median=hip_angle_tdc,
        hip_angle_tdc_std=1.2,
        torso_angle_median=45.0,
        torso_angle_std=1.0,
        elbow_angle_median=elbow_angle,
        elbow_angle_std=1.0,
        ankle_angle_bdc_median=90.0,
        ankle_angle_bdc_std=1.0,
        foot_angle_bdc_median=foot_angle_bdc,
        foot_angle_bdc_std=1.0,
    )


def test_fitting_config_yaml_roundtrip(tmp_path: Path):
    cfg = FittingConfig()
    cfg.saddle_height_mm_per_deg = 6.0
    yaml_file = tmp_path / "test_config.yaml"
    cfg.to_yaml(yaml_file)

    loaded = FittingConfig.from_yaml(yaml_file)
    assert loaded.saddle_height_mm_per_deg == 6.0
    assert loaded.knee_flexion_bdc.min_val == 25.0


def test_rule_empty_cycles():
    empty_stats = CycleSegmentationResult(
        side="right",
        all_cycles=[],
        retained_cycles=[],
        discarded_count=0,
        cadence_rpm_median=0.0,
        cadence_rpm_std=0.0,
        knee_flexion_bdc_median=0.0,
        knee_flexion_bdc_std=0.0,
        knee_angle_bdc_median=0.0,
        knee_angle_bdc_std=0.0,
        hip_angle_tdc_median=0.0,
        hip_angle_tdc_std=0.0,
        torso_angle_median=0.0,
        torso_angle_std=0.0,
        elbow_angle_median=0.0,
        elbow_angle_std=0.0,
        ankle_angle_bdc_median=0.0,
        ankle_angle_bdc_std=0.0,
        foot_angle_bdc_median=0.0,
        foot_angle_bdc_std=0.0,
    )
    engine = FittingRuleEngine()
    report = engine.evaluate(empty_stats)
    assert len(report.recommendations) == 1
    assert "brak wyraźnych cykli" in report.recommendations[0].action_summary


def test_rule_saddle_too_high():
    stats = create_mock_cycle_stats(knee_flexion_bdc=20.0)  # leg too straight
    engine = FittingRuleEngine()
    report = engine.evaluate(stats)

    saddle_recs = [r for r in report.recommendations if r.category == RecommendationCategory.SADDLE_HEIGHT]
    assert len(saddle_recs) == 1
    rec = saddle_recs[0]
    assert "Obniż siodełko" in rec.action_summary
    assert rec.adjustment_mm < 0
    assert rec.confidence == ConfidenceLevel.HIGH


def test_rule_saddle_too_low():
    stats = create_mock_cycle_stats(knee_flexion_bdc=40.0)  # leg too bent
    engine = FittingRuleEngine()
    report = engine.evaluate(stats)

    saddle_recs = [r for r in report.recommendations if r.category == RecommendationCategory.SADDLE_HEIGHT]
    assert len(saddle_recs) == 1
    rec = saddle_recs[0]
    assert "Podnieś siodełko" in rec.action_summary
    assert rec.adjustment_mm > 0
    assert rec.confidence == ConfidenceLevel.HIGH


def test_rule_saddle_in_range():
    stats = create_mock_cycle_stats(knee_flexion_bdc=30.0)
    engine = FittingRuleEngine()
    report = engine.evaluate(stats)

    saddle_recs = [r for r in report.recommendations if r.category == RecommendationCategory.SADDLE_HEIGHT]
    assert len(saddle_recs) == 1
    assert "prawidłowa" in saddle_recs[0].action_summary
    assert saddle_recs[0].adjustment_mm == 0.0


def test_rule_hip_closure_crank_advice():
    # Hip angle 38 deg in GMP (< 45 deg)
    stats = create_mock_cycle_stats(hip_angle_tdc=38.0)
    engine = FittingRuleEngine()
    report = engine.evaluate(stats, known_crank_length_mm=175.0)

    crank_recs = [r for r in report.recommendations if r.category == RecommendationCategory.CRANK_LENGTH]
    assert len(crank_recs) == 1
    assert "skrócenie ramion korby" in crank_recs[0].action_summary


def test_rule_elbow_cockpit_advice():
    # Elbow angle 170 deg (straight / locked)
    stats = create_mock_cycle_stats(elbow_angle=170.0)
    engine = FittingRuleEngine()
    report = engine.evaluate(stats)

    cockpit_recs = [r for r in report.recommendations if r.category == RecommendationCategory.COCKPIT_REACH_STACK]
    assert len(cockpit_recs) == 1
    assert "Skróć zasięg" in cockpit_recs[0].action_summary


def test_reporting_formatting_and_saving(tmp_path: Path):
    stats = create_mock_cycle_stats(knee_flexion_bdc=22.0)
    engine = FittingRuleEngine()
    report = engine.evaluate(stats)

    txt = format_text_summary(report)
    assert "RAPORT ANALIZY BIKE FITTINGU" in txt
    assert "Obniż siodełko" in txt

    md = format_markdown_summary(report)
    assert "# Raport analizy Bike Fittingu" in md

    paths = save_reports(report, tmp_path, "sample")
    assert paths["txt"].exists()
    assert paths["md"].exists()
    assert paths["json"].exists()
