# Adsorbate Tracking Toolkit

This project provides a GUI and command line interface for analysing atomic-resolution videos with adsorbates. The workflow performs:

1. Frame-by-frame lattice estimation using FFT peak detection.
2. Robust consolidation of lattice parameters across the video.
3. Refined lattice fitting with configurable "wiggle" tolerance.
4. Adsorbate detection based on the intensity beneath lattice sites.
5. Drift correction between frames and tracking of adsorbates.
6. Diffusion coefficient computation from the resulting trajectories.
7. Export of diffusion statistics alongside a side-by-side overlay video.

> **Angle convention:** the `--angle` parameter (and its GUI counterpart) represents the angle *between* the two lattice vectors. The in-frame orientation of the lattice is determined automatically from the footage.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## GUI Usage

```bash
python -m src.track_adsorbates.gui
```

Select an input video and adjust parameters to suit your dataset. The GUI automatically saves all artefacts in the same directory as the source video, so no output folder selection is required.

## Command Line Usage

Running `python run_pipeline.py` with no additional arguments launches the GUI directly.

To process a video from the terminal instead, supply the required lattice parameters:

```bash
python run_pipeline.py /path/to/video.mp4 \
    --approx-a 3.0 \
    --approx-b 3.0 \
    --angle 90 \
    --frame-width 10 \
    --wiggle 10 \
    --atom-diameter 1.5 \
    --drift 2.0
```

The command produces artefacts alongside the source video:

- `<video_name>_true_lattice.json` – consolidated lattice parameters (including the in-frame orientation).
- `<video_name>_diffusion_summary.json` – overall diffusion statistics.
- `<video_name>_diffusion_frame_values.csv` – per-frame diffusion coefficients.
- `<video_name>_overlay.mp4` – side-by-side video with the colour original on the left and the standalone artificial lattice on the right.
