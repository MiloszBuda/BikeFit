"""Command line interface for BikeFit Stage 1 video processing."""

import argparse
from pathlib import Path
import sys
from typing import Optional, Tuple

from bikefit.calibration.crank import CrankCalibration
from bikefit.io.storage import save_keypoints
from bikefit.io.visualization import VideoVisualizer
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
        description="BikeFit: Video-to-Keypoints Stage 1 Pipeline (RTMPose Halpe26)",
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
        help="Directory to save keypoints data and annotated video (default: ./output)",
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
        "--side",
        choices=["right", "left", "auto"],
        default="auto",
        help="Rider side facing camera (default: auto)",
    )
    parser.add_argument(
        "--export-format",
        choices=["csv", "parquet", "both"],
        default="both",
        help="Keypoints serialization format (default: both)",
    )
    parser.add_argument(
        "--no-video-out",
        action="store_true",
        help="Skip generating annotated skeleton MP4 video",
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

    saved_files = save_keypoints(
        results=results,
        output_path=output_dir / f"{video_stem}_keypoints",
        export_format=args.export_format,
    )
    for f in saved_files:
        print(f"[BikeFit] Saved keypoints: {f}")

    if output_video_path and output_video_path.exists():
        print(f"[BikeFit] Saved annotated video: {output_video_path}")


if __name__ == "__main__":
    main()
