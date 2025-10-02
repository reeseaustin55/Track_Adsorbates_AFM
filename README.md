# Adsorbate Tracking Toolkit

This project provides a GUI and command line interface for analysing atomic-resolution videos with adsorbates. The workflow performs:

1. Interactive FFT peak selection to seed the lattice basis.
2. Frame-by-frame lattice estimation guided by the selected peaks.
3. Robust consolidation of lattice parameters across the video with configurable "wiggle" tolerances.
4. Adsorbate detection based on the intensity beneath lattice sites.
5. Drift correction between frames and tracking of adsorbates.
6. Diffusion coefficient computation from the resulting trajectories.
7. Export of diffusion statistics alongside a side-by-side overlay video.

> **Angle convention:** the interactive FFT selection determines both the lattice spacing and the angle *between* the lattice vectors. The in-frame orientation relative to the image axes is recovered automatically from the footage.

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

Select an input video and adjust parameters to suit your dataset. After loading the first frame, a window appears showing the original image, the Difference-of-Gaussians (DoG) filtered view, and the log-magnitude FFT. Click two reciprocal lattice peaks on the FFT to initialise the lattice, optionally cycling through frames with the **Next →** button if the first frame is unsuitable. The GUI automatically saves all artefacts in the same directory as the source video, so no output folder selection is required.

## Command Line Usage

Running `python run_pipeline.py` with no additional arguments launches the GUI directly.

To process a video from the terminal instead, provide the physical metadata and interactively choose the lattice peaks when prompted:

```bash
python run_pipeline.py /path/to/video.mp4 \
    --frame-width 10 \
    --first-wiggle 35 \
    --second-wiggle 10 \
    --atom-diameter 1.5 \
    --drift 2.0
```

The command produces artefacts alongside the source video:

- `<video_name>_true_lattice.json` – consolidated lattice parameters (including the in-frame orientation).
- `<video_name>_diffusion_summary.json` – overall diffusion statistics.
- `<video_name>_diffusion_frame_values.csv` – per-frame diffusion coefficients.
- `<video_name>_overlay.mp4` – side-by-side video with the colour original on the left and the standalone artificial lattice on the right.
