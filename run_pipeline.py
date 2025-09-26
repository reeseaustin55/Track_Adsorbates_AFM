"""Command line entry point for adsorbate tracking."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.track_adsorbates.pipeline import PipelineConfig, run_pipeline
from src.track_adsorbates.video import load_video
from src.track_adsorbates.selection import select_initial_lattice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track adsorbates in atomic-resolution videos")
    parser.add_argument("video", type=Path, help="Path to the mp4 video")
    parser.add_argument("--frame-width", type=float, required=True, help="Frame width (nm)")
    parser.add_argument(
        "--first-wiggle",
        type=float,
        default=35.0,
        help="Initial lattice wiggle for consolidation (%)",
    )
    parser.add_argument(
        "--second-wiggle",
        type=float,
        default=10.0,
        help="Refinement wiggle around consolidated lattice (%)",
    )
    parser.add_argument("--atom-diameter", type=float, default=1.5, help="Atom diameter for overlay (Å)")
    parser.add_argument("--drift", type=float, default=2.0, help="Drift allowance between frames (atoms)")
    return parser.parse_args()


def main() -> None:
    if len(sys.argv) == 1:
        from src.track_adsorbates.gui import launch_app

        launch_app()
        return

    args = parse_args()
    video = load_video(str(args.video))
    config = PipelineConfig(
        frame_width_nm=args.frame_width,
        first_pass_wiggle_percent=args.first_wiggle,
        second_pass_wiggle_percent=args.second_wiggle,
        atom_diameter_angstrom=args.atom_diameter,
        drift_allowance_atoms=args.drift,
    )
    initial_lattice = select_initial_lattice(video, frame_width_nm=args.frame_width)
    result = run_pipeline(video, config, initial_lattice)
    print(f"True lattice: {result.lattice_json}")
    print(f"Overlay video: {result.overlay_video_path}")
    print(f"Diffusion summary: {result.diffusion_json}")
    print(f"Frame diffusion CSV: {result.diffusion_csv}")


if __name__ == "__main__":
    main()
