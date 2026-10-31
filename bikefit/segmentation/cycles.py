"""Pedal stroke cycle segmentation, Dead Center detection, and repeatability statistics."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.signal import find_peaks

from bikefit.kinematics.angles import KinematicsResult


@dataclass
class CycleMetrics:
    """Kinematic metrics for a single pedal revolution."""
    cycle_idx: int
    start_frame: int
    end_frame: int
    duration_s: float
    cadence_rpm: float

    # Key event frames
    bdc_frame: int  # Bottom Dead Center (DMP - max knee extension)
    tdc_frame: int  # Top Dead Center (GMP - min knee angle / max hip closure)

    # Knee angles [degrees]
    knee_angle_bdc: float       # Internal angle at DMP (target: 145-155 deg)
    knee_flexion_bdc: float     # Flexion at DMP = 180 - internal (target: 25-35 deg)
    knee_angle_tdc: float       # Internal angle at GMP
    knee_angle_range: float     # Max - Min in cycle

    # Hip angles [degrees]
    hip_angle_tdc: float        # Hip closure at GMP (target: >= 45 deg)
    hip_angle_bdc: float

    # Trunk & Arms [degrees]
    torso_angle_mean: float     # Mean torso angle vs horizontal
    elbow_angle_mean: float     # Mean elbow angle
    arm_trunk_angle_mean: float

    # Foot & Ankle [degrees]
    ankle_angle_bdc: float
    foot_angle_bdc: float       # Foot inclination at DMP


@dataclass
class CycleSegmentationResult:
    """Aggregated statistics across multiple pedal revolutions."""
    side: str
    all_cycles: List[CycleMetrics]
    retained_cycles: List[CycleMetrics]  # after discarding warmup/cooldown
    discarded_count: int

    # Aggregate metrics (median +/- std of retained cycles)
    cadence_rpm_median: float
    cadence_rpm_std: float

    knee_flexion_bdc_median: float
    knee_flexion_bdc_std: float
    knee_angle_bdc_median: float
    knee_angle_bdc_std: float

    hip_angle_tdc_median: float
    hip_angle_tdc_std: float

    torso_angle_median: float
    torso_angle_std: float

    elbow_angle_median: float
    elbow_angle_std: float

    ankle_angle_bdc_median: float
    ankle_angle_bdc_std: float

    foot_angle_bdc_median: float
    foot_angle_bdc_std: float

    # Frame indices for all detected DMP (BDC) and GMP (TDC)
    all_bdc_frames: List[int] = field(default_factory=list)
    all_tdc_frames: List[int] = field(default_factory=list)


class CycleDetector:
    """
    Detects pedal stroke cycles and bottom/top dead centers from kinematic trajectories.
    """

    def __init__(
        self,
        min_cadence_rpm: float = 40.0,
        max_cadence_rpm: float = 140.0,
        discard_warmup_cycles: int = 1,
        discard_cooldown_cycles: int = 1,
    ) -> None:
        self.min_cadence_rpm = min_cadence_rpm
        self.max_cadence_rpm = max_cadence_rpm
        self.discard_warmup_cycles = discard_warmup_cycles
        self.discard_cooldown_cycles = discard_cooldown_cycles

    def segment(self, kinematics: KinematicsResult, fps: float) -> CycleSegmentationResult:
        """
        Segment the continuous time series into discrete pedal strokes.

        Uses the filtered knee angle or ankle vertical position to identify
        Bottom Dead Center (DMP - maximum leg extension) and Top Dead Center (GMP).
        """
        knee_angles = kinematics.knee_angles
        num_frames = len(knee_angles)

        # Expected frames between successive DMP peaks based on cadence
        # E.g. at 30 fps and 140 RPM: 30 / (140/60) = 12.8 frames
        min_distance = max(4, int(fps * (60.0 / self.max_cadence_rpm)))

        # DMP corresponds to peak of internal knee angle (most extended leg)
        # Prominence of at least 8 degrees to reject minor noise fluctuations
        bdc_peaks, properties = find_peaks(
            knee_angles,
            distance=min_distance,
            prominence=6.0,
        )

        # Fallback if knee angle had low amplitude: use ankle vertical y-position
        if len(bdc_peaks) < 2:
            ankle_key = "right_ankle" if kinematics.side == "right" else "left_ankle"
            if ankle_key in kinematics.filtered_trajectories:
                ankle_y = kinematics.filtered_trajectories[ankle_key][:, 1]
                # In image coordinates, y increases downwards -> max y is lowest point in physical space (DMP)
                bdc_peaks, _ = find_peaks(ankle_y, distance=min_distance, prominence=15.0)

        # TDC (GMP) corresponds to minimum knee angle (peaks of inverted knee angle)
        tdc_peaks, _ = find_peaks(
            -knee_angles,
            distance=min_distance,
            prominence=6.0,
        )

        all_cycles: List[CycleMetrics] = []

        # Each cycle is bounded by two successive BDC (DMP) events
        for i in range(len(bdc_peaks) - 1):
            start_f = int(bdc_peaks[i])
            end_f = int(bdc_peaks[i + 1])
            duration_s = (end_f - start_f) / fps
            cadence_rpm = (60.0 / duration_s) if duration_s > 0 else 0.0

            # Ignore unphysiological cycles
            if cadence_rpm < self.min_cadence_rpm or cadence_rpm > self.max_cadence_rpm:
                continue

            # Find corresponding TDC (GMP) between start_f and end_f
            tdc_in_cycle = [t for t in tdc_peaks if start_f < t < end_f]
            if tdc_in_cycle:
                tdc_f = int(tdc_in_cycle[0])
            else:
                # If peak detector missed TDC, use argmin of knee angle in the window
                tdc_f = start_f + int(np.argmin(knee_angles[start_f:end_f]))

            cycle_slice = slice(start_f, end_f)

            knee_bdc = float(knee_angles[start_f])
            knee_tdc = float(knee_angles[tdc_f])
            knee_range = float(np.max(knee_angles[cycle_slice]) - np.min(knee_angles[cycle_slice]))

            hip_tdc = float(kinematics.hip_angles[tdc_f])
            hip_bdc = float(kinematics.hip_angles[start_f])

            torso_mean = float(np.mean(kinematics.torso_angles[cycle_slice]))
            elbow_mean = float(np.mean(kinematics.elbow_angles[cycle_slice]))
            arm_trunk_mean = float(np.mean(kinematics.arm_trunk_angles[cycle_slice]))

            ankle_bdc = float(kinematics.ankle_angles[start_f])
            foot_bdc = float(kinematics.foot_angles[start_f])

            cycle_metric = CycleMetrics(
                cycle_idx=len(all_cycles),
                start_frame=start_f,
                end_frame=end_f,
                duration_s=duration_s,
                cadence_rpm=cadence_rpm,
                bdc_frame=start_f,
                tdc_frame=tdc_f,
                knee_angle_bdc=knee_bdc,
                knee_flexion_bdc=180.0 - knee_bdc,
                knee_angle_tdc=knee_tdc,
                knee_angle_range=knee_range,
                hip_angle_tdc=hip_tdc,
                hip_angle_bdc=hip_bdc,
                torso_angle_mean=torso_mean,
                elbow_angle_mean=elbow_mean,
                arm_trunk_angle_mean=arm_trunk_mean,
                ankle_angle_bdc=ankle_bdc,
                foot_angle_bdc=foot_bdc,
            )
            all_cycles.append(cycle_metric)

        # Discard warmup and cooldown cycles if enough cycles are available
        total_detected = len(all_cycles)
        if total_detected > (self.discard_warmup_cycles + self.discard_cooldown_cycles + 1):
            start_idx = self.discard_warmup_cycles
            end_idx = total_detected - self.discard_cooldown_cycles
            retained = all_cycles[start_idx:end_idx]
            discarded = total_detected - len(retained)
        else:
            retained = all_cycles
            discarded = 0

        # Calculate robust statistics (median and standard deviation)
        def calc_stat(values: List[float]) -> Tuple[float, float]:
            if not values:
                return 0.0, 0.0
            return float(np.median(values)), float(np.std(values))

        cadence_med, cadence_std = calc_stat([c.cadence_rpm for c in retained])
        knee_flx_med, knee_flx_std = calc_stat([c.knee_flexion_bdc for c in retained])
        knee_ang_med, knee_ang_std = calc_stat([c.knee_angle_bdc for c in retained])
        hip_tdc_med, hip_tdc_std = calc_stat([c.hip_angle_tdc for c in retained])
        torso_med, torso_std = calc_stat([c.torso_angle_mean for c in retained])
        elbow_med, elbow_std = calc_stat([c.elbow_angle_mean for c in retained])
        ankle_med, ankle_std = calc_stat([c.ankle_angle_bdc for c in retained])
        foot_med, foot_std = calc_stat([c.foot_angle_bdc for c in retained])

        return CycleSegmentationResult(
            side=kinematics.side,
            all_cycles=all_cycles,
            retained_cycles=retained,
            discarded_count=discarded,
            cadence_rpm_median=cadence_med,
            cadence_rpm_std=cadence_std,
            knee_flexion_bdc_median=knee_flx_med,
            knee_flexion_bdc_std=knee_flx_std,
            knee_angle_bdc_median=knee_ang_med,
            knee_angle_bdc_std=knee_ang_std,
            hip_angle_tdc_median=hip_tdc_med,
            hip_angle_tdc_std=hip_tdc_std,
            torso_angle_median=torso_med,
            torso_angle_std=torso_std,
            elbow_angle_median=elbow_med,
            elbow_angle_std=elbow_std,
            ankle_angle_bdc_median=ankle_med,
            ankle_angle_bdc_std=ankle_std,
            foot_angle_bdc_median=foot_med,
            foot_angle_bdc_std=foot_std,
            all_bdc_frames=[int(p) for p in bdc_peaks],
            all_tdc_frames=[int(p) for p in tdc_peaks],
        )
