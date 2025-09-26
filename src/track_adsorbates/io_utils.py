"""I/O helpers for exporting analysis results."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .adsorbates import DiffusionResults
from .lattice import LatticeParameters


def export_lattice(path: Path, params: LatticeParameters) -> None:
    path.write_text(json.dumps(params.as_dict(), indent=2))


def export_diffusion(path: Path, results: DiffusionResults, fps: float) -> None:
    data = {
        "average_diffusion_coefficient": results.average,
        "msd_slope": results.slope,
        "frame_rate": fps,
    }
    path.write_text(json.dumps(data, indent=2))


def export_frame_diffusion(path: Path, frame_values: Iterable[float]) -> None:
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "diffusion_coefficient"])
        for idx, value in enumerate(frame_values):
            writer.writerow([idx, value])
