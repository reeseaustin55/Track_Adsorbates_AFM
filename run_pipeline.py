"""Command line entry point for adsorbate tracking."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.track_adsorbates.pipeline import PipelineConfig, run_pipeline
from src.track_adsorbates.video import load_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track adsorbates in atomic-resolution videos")
    parser.add_argument("video", type=Path, help="Path to the mp4 video")
    parser.add_argument("--approx-a", type=float, required=True, help="Approximate lattice vector a length (Å)")
    parser.add_argument("--approx-b", type=float, required=True, help="Approximate lattice vector b length (Å)")
    parser.add_argument("--angle", type=float, required=True, help="Angle between vectors (degrees)")
    parser.add_argument("--frame-width", type=float, required=True, help="Frame width (nm)")
    parser.add_argument("--wiggle", type=float, default=10.0, help="Wiggle room around lattice parameters (%)")
    parser.add_argument("--atom-diameter", type=float, default=1.5, help="Atom diameter for overlay (Å)")
    parser.add_argument("--drift", type=float, default=2.0, help="Drift allowance between frames (atoms)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Brightness threshold for adsorbates")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory")
    return parser.parse_args()


def main() -> None:
    if len(sys.argv) == 1:
        from src.track_adsorbates.gui import launch_app

        launch_app()
        return

    args = parse_args()
    video = load_video(str(args.video))
    config = PipelineConfig(
        approx_a_angstrom=args.approx_a,
        approx_b_angstrom=args.approx_b,
        approx_angle_deg=args.angle,
        frame_width_nm=args.frame_width,
        wiggle_percent=args.wiggle,
        atom_diameter_angstrom=args.atom_diameter,
        drift_allowance_atoms=args.drift,
        brightness_threshold=args.threshold,
    )
    run_pipeline(video, config, args.output)


if __name__ == "__main__":
    main()
