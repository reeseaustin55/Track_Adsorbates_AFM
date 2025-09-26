# Adsorbate Tracking Toolkit

This project provides a GUI and command line interface for analysing atomic-resolution videos with adsorbates. The workflow performs:

1. Frame-by-frame lattice estimation using FFT peak detection.
2. Robust consolidation of lattice parameters across the video.
3. Refined lattice fitting with configurable "wiggle" tolerance.
4. Adsorbate detection based on the intensity beneath lattice sites.
5. Drift correction between frames and tracking of adsorbates.
6. Diffusion coefficient computation from the resulting trajectories.
7. Export of diffusion statistics alongside a side-by-side overlay video.

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

Select an input video and adjust parameters to suit your dataset. Outputs are saved to the chosen directory.

## Command Line Usage

```bash
python run_pipeline.py /path/to/video.mp4 \
    --approx-a 3.0 \
    --approx-b 3.0 \
    --angle 60 \
    --frame-width 10 \
    --wiggle 10 \
    --atom-diameter 1.5 \
    --drift 2.0 \
    --threshold 0.5 \
    --output results
```

The command produces:

- `results/true_lattice.json` – consolidated lattice parameters.
- `results/diffusion_summary.json` – overall diffusion statistics.
- `results/diffusion_frame_values.csv` – per-frame diffusion coefficients.
- `results/<video_name>_overlay.mp4` – video showing the original data with the artificial lattice overlay.
