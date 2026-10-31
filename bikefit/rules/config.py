"""Configuration and target ranges for bike fitting recommendations."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class TargetRange:
    """Range definition for a specific kinematic parameter."""
    min_val: float
    max_val: float
    ideal_val: float
    unit: str = "deg"


@dataclass
class FittingConfig:
    """Target biomechanical ranges and adjustment coefficients."""

    # 1. Knee flexion at Bottom Dead Center (flexion = 180 - internal angle)
    knee_flexion_bdc: TargetRange = field(
        default_factory=lambda: TargetRange(min_val=25.0, max_val=35.0, ideal_val=30.0, unit="deg")
    )

    # 2. Hip angle at Top Dead Center (GMP - closure)
    hip_angle_tdc: TargetRange = field(
        default_factory=lambda: TargetRange(min_val=45.0, max_val=65.0, ideal_val=52.0, unit="deg")
    )

    # 3. Torso inclination vs horizontal (road endurance profile)
    torso_angle: TargetRange = field(
        default_factory=lambda: TargetRange(min_val=40.0, max_val=50.0, ideal_val=45.0, unit="deg")
    )

    # 4. Elbow angle (180 = straight, typical slight bend 150-165)
    elbow_angle: TargetRange = field(
        default_factory=lambda: TargetRange(min_val=148.0, max_val=165.0, ideal_val=155.0, unit="deg")
    )

    # 5. Arm-Trunk angle
    arm_trunk_angle: TargetRange = field(
        default_factory=lambda: TargetRange(min_val=80.0, max_val=95.0, ideal_val=88.0, unit="deg")
    )

    # 6. Foot angle vs horizontal at BDC (toes down is positive)
    foot_angle_bdc: TargetRange = field(
        default_factory=lambda: TargetRange(min_val=-5.0, max_val=15.0, ideal_val=5.0, unit="deg")
    )

    # Adjustment coefficients
    saddle_height_mm_per_deg: float = 5.5
    max_single_saddle_step_mm: float = 6.0
    stem_reach_step_mm: float = 10.0

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file."""
        data = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "FittingConfig":
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(
            knee_flexion_bdc=TargetRange(**data["knee_flexion_bdc"]),
            hip_angle_tdc=TargetRange(**data["hip_angle_tdc"]),
            torso_angle=TargetRange(**data["torso_angle"]),
            elbow_angle=TargetRange(**data["elbow_angle"]),
            arm_trunk_angle=TargetRange(**data["arm_trunk_angle"]),
            foot_angle_bdc=TargetRange(**data["foot_angle_bdc"]),
            saddle_height_mm_per_deg=float(data.get("saddle_height_mm_per_deg", 5.5)),
            max_single_saddle_step_mm=float(data.get("max_single_saddle_step_mm", 6.0)),
            stem_reach_step_mm=float(data.get("stem_reach_step_mm", 10.0)),
        )
