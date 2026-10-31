"""Plotting and chart generation for cycling kinematics and angle time series."""

from pathlib import Path
from typing import Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for CLI / pipelines
import matplotlib.pyplot as plt
import numpy as np

from bikefit.kinematics.angles import KinematicsResult


def plot_kinematics_summary(
    kinematics: KinematicsResult,
    output_path: Path | str,
    title: Optional[str] = None,
) -> Path:
    """
    Generate a 3-panel publication-grade chart of joint angles over time:
      1. Knee angle (internal and flexion) with target BDC zone (145-155 deg)
      2. Hip angle and Torso inclination angle
      3. Foot angle and Ankle angle
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    t = kinematics.timestamps_s
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    side_label = kinematics.side.capitalize()
    main_title = title or f"BikeFit Kinematics Analysis ({side_label} Side)"
    fig.suptitle(main_title, fontsize=14, fontweight="bold", y=0.98)

    # 1. Knee Angle
    ax1 = axes[0]
    ax1.plot(t, kinematics.knee_angles, color="#1f77b4", lw=2, label="Knee Angle (internal)")
    # Target zone for Bottom Dead Center (145-155 deg internal = 25-35 deg flexion)
    ax1.axhspan(145, 155, color="green", alpha=0.15, label="Target BDC Range (145°-155°)")
    ax1.set_ylabel("Knee Angle [°]")
    ax1.set_title("Knee Kinematics (Extension & Flexion Cycles)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Secondary y-axis for flexion
    ax1_flex = ax1.twinx()
    ax1_flex.set_ylabel("Flexion [°] (180° - angle)", color="#555555")
    y_min, y_max = ax1.get_ylim()
    ax1_flex.set_ylim(180 - y_min, 180 - y_max)
    ax1_flex.tick_params(axis="y", labelcolor="#555555")

    # 2. Hip and Torso Angle
    ax2 = axes[1]
    ax2.plot(t, kinematics.hip_angles, color="#d62728", lw=1.8, label="Hip Angle (shoulder-hip-knee)")
    ax2.plot(t, kinematics.torso_angles, color="#ff7f0e", lw=1.8, linestyle="--", label="Torso Inclination vs Horizontal")
    ax2.axhspan(40, 50, color="#ff7f0e", alpha=0.1, label="Typical Road Torso Range (40°-50°)")
    ax2.set_ylabel("Angle [°]")
    ax2.set_title("Trunk and Hip Biomechanics")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.9)

    # 3. Foot and Ankle Angle
    ax3 = axes[2]
    ax3.plot(t, kinematics.ankle_angles, color="#2ca02c", lw=1.8, label="Ankle Angle (knee-ankle-toe)")
    ax3.plot(t, kinematics.foot_angles, color="#9467bd", lw=1.8, linestyle=":", label="Foot Angle vs Horizontal (heel-toe)")
    ax3.set_ylabel("Angle [°]")
    ax3.set_xlabel("Time [seconds]")
    ax3.set_title("Pedaling Technique & Foot Dynamics")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return out_file
