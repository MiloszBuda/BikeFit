"""Command line interface for BikeFit pipeline (Stage 1 & Stage 2)."""

import argparse
from pathlib import Path
import sys
from typing import Optional, Tuple

from bikefit.calibration.crank import CrankCalibration
from bikefit.io.plotting import plot_kinematics_summary
from bikefit.io.storage import save_keypoints
from bikefit.io.visualization import VideoVisualizer
from bikefit.kinematics.angles import extract_kinematics
from bikefit.kinematics.scale import calibrate_scale_from_crank
from bikefit.models.rtmpose import RTMPoseEstimator
from bikefit.pipeline.video import VideoProcessor


def parse_crank_arg(crank_str: Optional[str]) -> Optional[Tuple[float, float]]:
    """Parse 'x,y' crank string into tuple."""
    if not crank_str:
        return None
    try:
        parts = crank_str.split(",")
        if len(parts) != 2:
            raise ValueError()
        return (float(parts[0].strip()), float(parts[1].strip()))
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid crank format '{crank_str}'. Expected 'x,y' (e.g. '640,720')"
        ) from e


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bikefit",
        description="BikeFit: AI-Powered Bike Fitting Analysis (RTMPose Halpe26 + Kinematics)",
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to input video file (e.g. riding on indoor trainer)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Directory to save keypoints, angles, charts, and annotated video (default: ./output)",
    )
    parser.add_argument(
        "--mode",
        choices=["lightweight", "balanced", "performance"],
        default="balanced",
        help="RTMPose model complexity mode (default: balanced)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Inference device (default: cpu)",
    )
    parser.add_argument(
        "--crank",
        type=str,
        default=None,
        help="Bottom bracket center coordinates as 'x,y' (e.g. --crank 540,680)",
    )
    parser.add_argument(
        "--interactive-crank",
        action="store_true",
        help="Open GUI window to click on bottom bracket / crank center on first frame",
    )
    parser.add_argument(
        "--crank-length",
        type=float,
        default=172.5,
        help="Physical crank arm length in mm for scale calibration (default: 172.5)",
    )
    parser.add_argument(
        "--side",
        choices=["right", "left", "auto"],
        default="auto",
        help="Rider side facing camera (default: auto)",
    )
    parser.add_argument(
        "--filter-cutoff",
        type=float,
        default=5.0,
        help="Butterworth lowpass filter cutoff frequency in Hz (default: 5.0)",
    )
    parser.add_argument(
        "--export-format",
        choices=["csv", "parquet", "both"],
        default="both",
        help="Keypoints & angles serialization format (default: both)",
    )
    parser.add_argument(
        "--no-video-out",
        action="store_true",
        help="Skip generating annotated skeleton MP4 video",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating kinematic matplotlib charts",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most N frames (useful for quick testing)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Confidence threshold for keypoints visualization (default: 0.35)",
    )
    return parser


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    crank_coords = parse_crank_arg(args.crank)
    interactive = args.interactive_crank or (crank_coords is None and sys.stdin.isatty())

    print(f"[BikeFit] Initializing RTMPose (mode={args.mode}, device={args.device})...")
    estimator = RTMPoseEstimator(mode=args.mode, device=args.device)

    visualizer = VideoVisualizer(kpt_score_threshold=args.threshold)
    processor = VideoProcessor(pose_estimator=estimator, visualizer=visualizer)

    video_stem = args.video.stem
    output_video_path = None if args.no_video_out else (output_dir / f"{video_stem}_annotated.mp4")

    print(f"[BikeFit] Processing video: {args.video}")
    if crank_coords:
        print(f"[BikeFit] Using specified crank center: {crank_coords}")
    elif interactive:
        print("[BikeFit] Interactive crank calibration enabled. A window will open to click the crank center.")

    results, metadata = processor.process_video(
        video_path=args.video,
        crank=crank_coords,
        interactive_crank=interactive,
        output_video_path=output_video_path,
        side_preference=args.side if args.side != "auto" else None,
        max_frames=args.max_frames,
    )

    print(f"[BikeFit] Processed {len(results)} frames ({metadata.fps:.1f} FPS, {metadata.width}x{metadata.height}).")

    # 1. Save raw tracked keypoints
    saved_keypoints = save_keypoints(
        results=results,
        output_path=output_dir / f"{video_stem}_keypoints",
        export_format=args.export_format,
    )
    for f in saved_keypoints:
        print(f"[BikeFit] Saved keypoints: {f}")

    if output_video_path and output_video_path.exists():
        print(f"[BikeFit] Saved annotated video: {output_video_path}")

    # 2. Extract kinematics & joint angles (Stage 2)
    print(f"[BikeFit] Filtering trajectories & calculating joint angles (cutoff={args.filter_cutoff} Hz)...")
    kinematics = extract_kinematics(
        results=results,
        fps=metadata.fps,
        side=args.side,
        cutoff_hz=args.filter_cutoff,
        score_threshold=args.threshold,
    )
    print(f"[BikeFit] Analyzed side: {kinematics.side.upper()}")

    # 3. Export angles DataFrame
    df_angles = kinematics.to_dataframe()
    if args.export_format in ("csv", "both"):
        csv_angles = output_dir / f"{video_stem}_angles.csv"
        df_angles.to_csv(csv_angles, index=False)
        print(f"[BikeFit] Saved angles: {csv_angles}")

    if args.export_format in ("parquet", "both"):
        parquet_angles = output_dir / f"{video_stem}_angles.parquet"
        df_angles.to_parquet(parquet_angles, index=False)
        print(f"[BikeFit] Saved angles: {parquet_angles}")

    # 4. Scale calibration if crank center is available
    if results and results[0].crank_center is not None:
        foot_key = "right_big_toe" if kinematics.side == "right" else "left_big_toe"
        foot_pts = kinematics.filtered_trajectories.get(foot_key)
        if foot_pts is not None:
            scale = calibrate_scale_from_crank(
                foot_points=foot_pts,
                crank_center=results[0].crank_center,
                crank_arm_mm=args.crank_length,
            )
            print(
                f"[BikeFit] Scale calibration: {scale.pixels_per_mm:.3f} px/mm "
                f"({scale.mm_per_pixel:.3f} mm/px) based on {args.crank_length} mm crank"
            )

    # 5. Generate Kinematics Chart
    if not args.no_plots:
        chart_path = output_dir / f"{video_stem}_kinematics.png"
        plot_kinematics_summary(kinematics, output_path=chart_path)
        print(f"[BikeFit] Saved kinematics chart: {chart_path}")

    print("[BikeFit] Analysis complete.")


if __name__ == "__main__":
    main()
