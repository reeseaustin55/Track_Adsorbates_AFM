"""Core processing pipeline for adsorbate tracking."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .adsorbates import AdsorbateDetection, DiffusionResults, assign_tracks, compute_diffusion, detect_adsorbates
from .io_utils import export_diffusion, export_frame_diffusion, export_lattice
from .lattice import (
    LatticeParameters,
    build_lattice_points,
    consolidate_lattice,
    estimate_lattice_from_frame,
    lattice_vectors_in_pixels,
    refine_translation,
)
from .video import VideoData
from .visualization import combine_frames, draw_lattice_overlay, write_video


@dataclass
class PipelineConfig:
    approx_a_angstrom: float
    approx_b_angstrom: float
    approx_angle_deg: float
    frame_width_nm: float
    wiggle_percent: float = 10.0
    atom_diameter_angstrom: float = 1.5
    drift_allowance_atoms: float = 2.0
    brightness_threshold: float = 0.5


@dataclass
class PipelineResult:
    lattice: LatticeParameters
    diffusion: DiffusionResults
    detections: List[AdsorbateDetection]
    overlay_video_path: Path
    diffusion_json: Path
    diffusion_csv: Path


def run_pipeline(video: VideoData, config: PipelineConfig, output_dir: Path) -> PipelineResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    approx_lattice = LatticeParameters(
        a_length=config.approx_a_angstrom,
        b_length=config.approx_b_angstrom,
        angle_deg=config.approx_angle_deg,
    )

    width_pixels = video.frame_shape[1]
    pixel_size_nm = config.frame_width_nm / width_pixels

    per_frame_lattices: List[LatticeParameters] = []

    for frame in video.frames:
        params = estimate_lattice_from_frame(frame, approx_lattice, pixel_size_nm)
        if params is None:
            params = approx_lattice
        per_frame_lattices.append(params)

    true_lattice = consolidate_lattice(per_frame_lattices)
    export_lattice(output_dir / "true_lattice.json", true_lattice)

    wiggle = config.wiggle_percent / 100.0
    detections: List[AdsorbateDetection] = []

    a_vec, b_vec = lattice_vectors_in_pixels(true_lattice, pixel_size_nm)
    atom_diameter_pixels = config.atom_diameter_angstrom * 0.1 / pixel_size_nm

    for frame in video.frames:
        a_scaled = a_vec * (1.0 + wiggle)
        b_scaled = b_vec * (1.0 + wiggle)
        translation = refine_translation(frame, a_scaled, b_scaled)
        points = build_lattice_points(frame.shape, a_vec, b_vec, translation)
        detection = detect_adsorbates(
            frame,
            points,
            atom_diameter_pixels=atom_diameter_pixels,
            brightness_threshold=config.brightness_threshold,
        )
        detections.append(detection)

    # Align frames using translation drift correction
    aligned_detections = _align_detections(detections, config.drift_allowance_atoms, a_vec, b_vec)

    tracks = assign_tracks(aligned_detections, max_distance=config.drift_allowance_atoms * np.linalg.norm(a_vec))
    diffusion = compute_diffusion(tracks, pixel_size_nm=pixel_size_nm, fps=video.fps)

    overlay_frames = []
    for frame, detection in zip(video.frames, aligned_detections):
        overlay = draw_lattice_overlay(frame, detection, atom_diameter_pixels)
        overlay_frames.append(combine_frames(frame, overlay))

    overlay_path = output_dir / f"{video.path.stem}_overlay.mp4"
    write_video(overlay_path, overlay_frames, fps=video.fps)

    diffusion_json = output_dir / "diffusion_summary.json"
    diffusion_csv = output_dir / "diffusion_frame_values.csv"
    export_diffusion(diffusion_json, diffusion, fps=video.fps)
    export_frame_diffusion(diffusion_csv, diffusion.frame_values)

    return PipelineResult(
        lattice=true_lattice,
        diffusion=diffusion,
        detections=aligned_detections,
        overlay_video_path=overlay_path,
        diffusion_json=diffusion_json,
        diffusion_csv=diffusion_csv,
    )


def _align_detections(
    detections: List[AdsorbateDetection],
    drift_allowance_atoms: float,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
) -> List[AdsorbateDetection]:
    """Align lattice detections using correlation-based drift correction."""
    if not detections:
        return detections

    aligned = [detections[0]]
    base_points = detections[0].points
    grid_spacing = (np.linalg.norm(a_vec) + np.linalg.norm(b_vec)) / 2.0
    max_shift = drift_allowance_atoms * grid_spacing

    for detection in detections[1:]:
        shift = _estimate_shift(base_points, detection.points, max_shift)
        shifted_points = detection.points + shift
        aligned.append(AdsorbateDetection(points=shifted_points, present=detection.present))
        base_points = shifted_points

    return aligned


def _estimate_shift(
    reference_points: np.ndarray,
    target_points: np.ndarray,
    max_shift: float,
) -> np.ndarray:
    """Estimate shift between two sets of points by simple centroid alignment."""
    ref_centroid = np.mean(reference_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    shift = ref_centroid - target_centroid
    shift_norm = np.linalg.norm(shift)
    if shift_norm > max_shift:
        shift = shift * (max_shift / shift_norm)
    return shift
