"""Python port of the `simple_lattice_occupancy` routine."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import imageio
import imageio.v3 as iio
import numpy as np
from scipy.io import savemat

from .helpers import (
    angle_between_deg,
    angle_diff_deg,
    angle_mode_deg,
    draw_spots,
    enhance_dog,
    gaussian_fft_peaks,
    kmeans_1d,
    length_mode_px,
    nm_cover_image,
    rescale,
    sample_at_points,
)
from .helpers import integer_shift


@dataclass
class LatticeParams:
    gauss_lo_hi: Tuple[float, float] = (0.8, 3.0)
    ring_slack_scan: float = 0.40
    top_k_per_band: int = 140
    wiggle_pct: float = 0.05
    min_tol_deg: float = 2.0
    phase_grid: int = 18
    spot_radius_frac: float = 0.25
    gray_dark: float = 0.30
    gray_light: float = 0.85
    max_sites_per_frame: int = 30000
    ad_percentile: float = 75.0
    z_pos_thresh: float = 0.25
    min_occ_for_frame: int = 3
    fallback_min_pct: float = 50.0
    fallback_step: int = 5
    top_frac_rescue: float = 0.03
    top_min_rescue: int = 5
    cover_margin: int = 3
    fwhm_probe_diam_a: float = 1.5
    fwhm_dr_px: float = 0.5
    fwhm_outer_frac: float = 0.3
    export_fwhm_hists: bool = True
    max_frames: int | None = None


@dataclass
class FrameBundle:
    nodes_px_locked: np.ndarray
    nodes_px_measured: np.ndarray
    occ: np.ndarray
    fwhm_px: np.ndarray
    a_len_px_pf: Tuple[float, float]
    a_len_px_locked: Tuple[float, float]
    r_spot: int
    phi_pf: Tuple[float, float]
    phi_locked: Tuple[float, float]


@dataclass
class BundleOutputs:
    occ_frames: List[FrameBundle]
    fps: float
    params: LatticeParams
    ax_star: float
    ay_star: float
    gamma_star: float
    theta_star: float
    nm_per_px: float
    sxspath: str
    onlypath: str
    fwhm_all_px: np.ndarray
    fwhm_all_a: np.ndarray
    png_a: str | None
    png_p: str | None
    canvas_height: int
    canvas_width: int
    shift_canvas: Tuple[int, int]
    frame_shape: Tuple[int, int]


def simple_lattice_occupancy(
    vid_path: str | Path,
    vid_width_nm: float,
    ax_a: float,
    ay_a: float,
    gamma_deg_guess: float,
    *,
    probe_diam_a: float | None = None,
    max_frames: int | None = None,
) -> BundleOutputs:
    """Port of the MATLAB ``simple_lattice_occupancy`` routine."""
    params = LatticeParams()
    if probe_diam_a is not None:
        params.fwhm_probe_diam_a = probe_diam_a
    if max_frames is not None:
        params.max_frames = max_frames

    vid_path = Path(vid_path)
    folder = vid_path.parent if vid_path.parent != Path("") else Path.cwd()
    base = vid_path.stem

    reader = imageio.get_reader(str(vid_path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", meta.get("fps_numerator", 1)) / max(meta.get("fps_denominator", 1), 1))
    if not np.isfinite(fps) or fps <= 0:
        fps = float(meta.get("fps", 1.0))
    frames: List[np.ndarray] = []
    for idx, frame in enumerate(reader):
        if params.max_frames is not None and idx >= params.max_frames:
            break
        if frame.ndim == 3:
            frame = np.mean(frame[..., :3], axis=2)
        frame = frame.astype(np.float64)
        frame = rescale(frame)
        frames.append(frame)
    reader.close()
    if not frames:
        raise ValueError(f"No frames could be read from {vid_path}")
    frames = [np.asarray(f, dtype=np.float64) for f in frames]
    n_frames = len(frames)
    height, width = frames[0].shape

    nm_per_px = vid_width_nm / width
    a_per_px = 10.0 * nm_per_px
    px_per_a = 1.0 / a_per_px
    ax_px_guess = ax_a * px_per_a
    ay_px_guess = ay_a * px_per_a

    theta_list = np.full(n_frames, np.nan)
    gamma_list = np.full(n_frames, np.nan)
    len_a1 = np.full(n_frames, np.nan)
    len_a2 = np.full(n_frames, np.nan)

    prev_g1: np.ndarray | None = None
    prev_g2: np.ndarray | None = None

    for i, image in enumerate(frames):
        enh = enhance_dog(image, *params.gauss_lo_hi)
        g1, g2 = _fft_pair_scan(
            enh,
            ax_px_guess,
            ay_px_guess,
            gamma_deg_guess,
            params,
        )
        if g1 is None or g2 is None:
            if prev_g1 is not None and prev_g2 is not None:
                g1, g2 = prev_g1, prev_g2
            else:
                theta_guess = np.radians(0.0)
                g1 = np.array([
                    np.cos(theta_guess) / max(ax_px_guess, 1e-9),
                    np.sin(theta_guess) / max(ax_px_guess, 1e-9),
                ])
                g2 = np.array([
                    np.cos(theta_guess + np.radians(gamma_deg_guess))
                    / max(ay_px_guess, 1e-9),
                    np.sin(theta_guess + np.radians(gamma_deg_guess))
                    / max(ay_px_guess, 1e-9),
                ])
        prev_g1, prev_g2 = g1, g2
        lattice = np.linalg.pinv(np.vstack([g1, g2]))
        a1 = lattice[:, 0]
        a2 = lattice[:, 1]
        len_a1[i] = np.linalg.norm(a1)
        len_a2[i] = np.linalg.norm(a2)
        gamma_list[i] = angle_between_deg(a1, a2)
        theta_list[i] = float(np.mod(np.degrees(np.arctan2(a1[1], a1[0])), 180.0))

    gamma_star = angle_mode_deg(gamma_list, 1.0)
    if not np.isfinite(gamma_star):
        gamma_star = float(gamma_deg_guess)
    theta_star = angle_mode_deg(theta_list, 1.0)
    if not np.isfinite(theta_star):
        theta_star = 0.0
    ax_star = length_mode_px(len_a1)
    if not np.isfinite(ax_star):
        ax_star = float(ax_px_guess)
    ay_star = length_mode_px(len_a2)
    if not np.isfinite(ay_star):
        ay_star = float(ay_px_guess)

    tol_gamma = max(params.min_tol_deg, params.wiggle_pct * max(gamma_star, 1.0))
    tol_theta = max(params.min_tol_deg, params.wiggle_pct * max(theta_star, 1.0))
    len_slack = max(0.05, params.wiggle_pct)

    u1_lock = np.array([np.cos(np.radians(theta_star)), np.sin(np.radians(theta_star))])
    u2_lock = np.array(
        [
            np.cos(np.radians(theta_star + gamma_star)),
            np.sin(np.radians(theta_star + gamma_star)),
        ]
    )
    a1_lock = ax_star * u1_lock
    a2_lock = ay_star * u2_lock

    phi0 = np.array([0.5, 0.5])
    base0 = np.array([1.0, 1.0]) + phi0[0] * a1_lock + phi0[1] * a2_lock
    n_range, m_range = nm_cover_image((height, width), base0, a1_lock, a2_lock, params.cover_margin)
    nn_glob, mm_glob = np.meshgrid(n_range, m_range)
    nn_glob = nn_glob.ravel()
    mm_glob = mm_glob.ravel()

    if nn_glob.size > params.max_sites_per_frame:
        idx = np.linspace(0, nn_glob.size - 1, params.max_sites_per_frame, dtype=int)
        nn_glob = nn_glob[idx]
        mm_glob = mm_glob[idx]

    r_spot_locked = int(max(1, round(params.spot_radius_frac * min(ax_star, ay_star))))

    phi_ref = np.array([0.0, 0.0])
    base_ref = np.array([1.0, 1.0]) + phi_ref[0] * a1_lock + phi_ref[1] * a2_lock
    xy_ref = np.column_stack(
        [
            base_ref[0] + nn_glob * a1_lock[0] + mm_glob * a2_lock[0],
            base_ref[1] + nn_glob * a1_lock[1] + mm_glob * a2_lock[1],
        ]
    )
    min_x, min_y = np.floor(np.min(xy_ref, axis=0)).astype(int)
    max_x, max_y = np.ceil(np.max(xy_ref, axis=0)).astype(int)
    pad_canvas = int(np.ceil(np.linalg.norm(a1_lock) + np.linalg.norm(a2_lock))) + r_spot_locked + 2
    min_xc = min_x - pad_canvas
    min_yc = min_y - pad_canvas
    max_xc = max_x + pad_canvas
    max_yc = max_y + pad_canvas
    canvas_width = max(1, max_xc - min_xc + 1)
    canvas_height = max(1, max_yc - min_yc + 1)
    shift_canvas = (min_xc, min_yc)

    sxspath = str(folder / f"{base}_artificial_lattice_SxS.mp4")
    onlypath = str(folder / f"{base}_artificial_lattice_only.mp4")
    side_writer = imageio.get_writer(sxspath, fps=fps)
    only_writer = imageio.get_writer(onlypath, fps=fps)

    occ_frames: List[FrameBundle] = []
    prev_phi = np.array([0.5, 0.5])
    fwhm_all_px: List[float] = []
    fwhm_all_a: List[float] = []

    probe_r_px = 0.5 * params.fwhm_probe_diam_a * px_per_a

    for i, image in enumerate(frames):
        enh = enhance_dog(image, *params.gauss_lo_hi)
        g1, g2 = _fft_pair_lock(
            enh,
            ax_star,
            ay_star,
            gamma_star,
            theta_star,
            len_slack,
            tol_gamma,
            tol_theta,
            params,
        )
        if g1 is None or g2 is None:
            u1 = np.array([np.cos(np.radians(theta_star)), np.sin(np.radians(theta_star))])
            u2 = np.array(
                [
                    np.cos(np.radians(theta_star + gamma_star)),
                    np.sin(np.radians(theta_star + gamma_star)),
                ]
            )
            g1 = u1 / max(ax_star, 1e-9)
            g2 = u2 / max(ay_star, 1e-9)
        lattice = np.linalg.pinv(np.vstack([g1, g2]))
        a1_pf = lattice[:, 0]
        a2_pf = lattice[:, 1]
        ax_pf = float(np.linalg.norm(a1_pf))
        ay_pf = float(np.linalg.norm(a2_pf))
        if not np.isfinite(ax_pf) or ax_pf <= 0 or not np.isfinite(ay_pf) or ay_pf <= 0:
            a1_pf = a1_lock
            a2_pf = a2_lock

        best_score = -np.inf
        best_phi = prev_phi
        K = params.phase_grid
        offsets = np.linspace(0, 1, K, endpoint=False)
        for p1 in offsets:
            for p2 in offsets:
                phi = np.array([p1, p2])
                base_pf = np.array([1.0, 1.0]) + phi[0] * a1_pf + phi[1] * a2_pf
                xy_pf = np.column_stack(
                    [
                        base_pf[0] + nn_glob * a1_pf[0] + mm_glob * a2_pf[0],
                        base_pf[1] + nn_glob * a1_pf[1] + mm_glob * a2_pf[1],
                    ]
                )
                mask = (
                    (xy_pf[:, 0] >= 1)
                    & (xy_pf[:, 0] <= width)
                    & (xy_pf[:, 1] >= 1)
                    & (xy_pf[:, 1] <= height)
                )
                if not np.any(mask):
                    continue
                score = float(np.sum(sample_at_points(enh, xy_pf[mask])))
                if score > best_score:
                    best_score = score
                    best_phi = phi
        prev_phi = best_phi

        base_pf = np.array([1.0, 1.0]) + best_phi[0] * a1_pf + best_phi[1] * a2_pf
        xy_pf = np.column_stack(
            [
                base_pf[0] + nn_glob * a1_pf[0] + mm_glob * a2_pf[0],
                base_pf[1] + nn_glob * a1_pf[1] + mm_glob * a2_pf[1],
            ]
        )
        t_pix = best_phi[0] * a1_pf + best_phi[1] * a2_pf
        phi_star = np.linalg.lstsq(np.column_stack([a1_lock, a2_lock]), t_pix, rcond=None)[0]
        base_lock = np.array([1.0, 1.0]) + phi_star[0] * a1_lock + phi_star[1] * a2_lock
        xy_lock = np.column_stack(
            [
                base_lock[0] + nn_glob * a1_lock[0] + mm_glob * a2_lock[0],
                base_lock[1] + nn_glob * a1_lock[1] + mm_glob * a2_lock[1],
            ]
        )
        in_img = (
            (xy_lock[:, 0] >= 1)
            & (xy_lock[:, 0] <= width)
            & (xy_lock[:, 1] >= 1)
            & (xy_lock[:, 1] <= height)
        )
        xy_lock_in = xy_lock[in_img]
        xy_pf_in = xy_pf[in_img]

        occ_in, fwhm_px = _adsorbates_from_fwhm(
            image,
            xy_pf_in,
            probe_r_px,
            r_spot_locked,
            params,
        )
        valid = np.isfinite(fwhm_px)
        fwhm_all_px.extend(fwhm_px[valid])
        fwhm_all_a.extend(fwhm_px[valid] * a_per_px)

        xy_canvas = np.column_stack(
            [xy_lock[:, 0] - shift_canvas[0] + 1, xy_lock[:, 1] - shift_canvas[1] + 1]
        )
        art = np.zeros((canvas_height, canvas_width), dtype=float)
        if xy_canvas.size:
            art = draw_spots(art, xy_canvas, r_spot_locked, params.gray_dark)
            if np.any(in_img) and np.any(occ_in == 1):
                art = draw_spots(art, xy_canvas[in_img & (occ_in == 1)], r_spot_locked, params.gray_light)
        art = np.clip(art, 0, 1)
        art_rgb = np.repeat((art * 255).astype(np.uint8)[..., None], 3, axis=2)

        left = image
        if left.shape[0] < canvas_height:
            pad_total = canvas_height - left.shape[0]
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            left = np.pad(left, ((pad_top, pad_bottom), (0, 0)), mode="edge")
        elif left.shape[0] > canvas_height:
            start = (left.shape[0] - canvas_height) // 2
            left = left[start : start + canvas_height, :]
        combo = np.concatenate([left, art], axis=1)
        combo = np.clip(combo, 0, 1)
        combo_rgb = np.repeat((combo * 255).astype(np.uint8)[..., None], 3, axis=2)

        side_writer.write(combo_rgb)
        only_writer.write(art_rgb)

        bundle = FrameBundle(
            nodes_px_locked=xy_lock_in,
            nodes_px_measured=xy_pf_in,
            occ=occ_in.astype(float),
            fwhm_px=fwhm_px,
            a_len_px_pf=(ax_pf, ay_pf),
            a_len_px_locked=(ax_star, ay_star),
            r_spot=r_spot_locked,
            phi_pf=(float(best_phi[0]), float(best_phi[1])),
            phi_locked=(float(phi_star[0]), float(phi_star[1])),
        )
        occ_frames.append(bundle)

    side_writer.close()
    only_writer.close()

    png_a = None
    png_p = None
    if params.export_fwhm_hists and fwhm_all_px:
        png_a = str(folder / f"{base}_fwhm_hist_A.png")
        png_p = str(folder / f"{base}_fwhm_hist_px.png")
        _save_histogram(np.array(fwhm_all_a), "FWHM (Å)", png_a, base)
        _save_histogram(np.array(fwhm_all_px), "FWHM (px)", png_p, base)

    occ_serialised = [
        {
            "nodes_px_locked": bundle.nodes_px_locked,
            "nodes_px_measured": bundle.nodes_px_measured,
            "occ": bundle.occ,
            "fwhm_px": bundle.fwhm_px,
            "a_len_px_pf": np.array(bundle.a_len_px_pf, dtype=float),
            "a_len_px_locked": np.array(bundle.a_len_px_locked, dtype=float),
            "r_spot": bundle.r_spot,
            "phi_pf": np.array(bundle.phi_pf, dtype=float),
            "phi_locked": np.array(bundle.phi_locked, dtype=float),
        }
        for bundle in occ_frames
    ]

    data = {
        "vidPath": str(vid_path),
        "fps": fps,
        "P": asdict(params),
        "ax_A": ax_a,
        "ay_A": ay_a,
        "gamma_deg_guess": gamma_deg_guess,
        "ax_px_guess": ax_px_guess,
        "ay_px_guess": ay_px_guess,
        "nm_per_px": nm_per_px,
        "occFrames": np.array(occ_serialised, dtype=object),
        "sxspath": sxspath,
        "onlypath": onlypath,
        "ax_star": ax_star,
        "ay_star": ay_star,
        "gamma_star": gamma_star,
        "theta_star": theta_star,
        "FWHM_all_px": np.array(fwhm_all_px),
        "FWHM_all_A": np.array(fwhm_all_a),
        "pngA": png_a or "",
        "pngP": png_p or "",
        "Hc": canvas_height,
        "Wc": canvas_width,
        "shiftCanvas": np.array(shift_canvas),
        "frame_shape": np.array([height, width]),
    }
    savemat(folder / "lattice_occupancy_locked.mat", data)

    return BundleOutputs(
        occ_frames=occ_frames,
        fps=fps,
        params=params,
        ax_star=ax_star,
        ay_star=ay_star,
        gamma_star=gamma_star,
        theta_star=theta_star,
        nm_per_px=nm_per_px,
        sxspath=sxspath,
        onlypath=onlypath,
        fwhm_all_px=np.array(fwhm_all_px),
        fwhm_all_a=np.array(fwhm_all_a),
        png_a=png_a,
        png_p=png_p,
        canvas_height=canvas_height,
        canvas_width=canvas_width,
        shift_canvas=shift_canvas,
        frame_shape=(height, width),
    )


def _fft_pair_scan(
    image: np.ndarray,
    ax_px: float,
    ay_px: float,
    gamma_deg: float,
    params: LatticeParams,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    r1 = 1.0 / max(ax_px, 1e-9)
    r2 = 1.0 / max(ay_px, 1e-9)
    _, freqs, values = gaussian_fft_peaks(
        image,
        r1,
        r2,
        params.ring_slack_scan,
        params.top_k_per_band,
    )
    best = -np.inf
    best_pair: Tuple[np.ndarray | None, np.ndarray | None] = (None, None)
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            u = freqs[i]
            v = freqs[j]
            lattice = np.linalg.pinv(np.vstack([u, v]))
            if not np.all(np.isfinite(lattice)):
                continue
            a1 = lattice[:, 0]
            a2 = lattice[:, 1]
            gamma = angle_between_deg(a1, a2)
            diff = angle_diff_deg(gamma, gamma_deg)
            if diff > 25:
                continue
            score = (values[i] + values[j]) / (1 + (diff / 10) ** 2)
            if score > best:
                best = score
                best_pair = (u, v)
    if best_pair[0] is None:
        angles = np.mod(np.degrees(np.arctan2(freqs[:, 1], freqs[:, 0])), 180.0)
        best = -np.inf
        for i in range(len(freqs)):
            for j in range(i + 1, len(freqs)):
                sep = min(abs(angles[i] - angles[j]), 180.0 - abs(angles[i] - angles[j]))
                if sep < 25:
                    continue
                score = values[i] + values[j]
                if score > best:
                    best = score
                    best_pair = (freqs[i], freqs[j])
    if best_pair[0] is None:
        return None, None
    return np.array(best_pair[0]), np.array(best_pair[1])


def _fft_pair_lock(
    image: np.ndarray,
    ax_star: float,
    ay_star: float,
    gamma_star: float,
    theta_star: float,
    len_slack: float,
    tol_gamma: float,
    tol_theta: float,
    params: LatticeParams,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    r1 = 1.0 / max(ax_star, 1e-9)
    r2 = 1.0 / max(ay_star, 1e-9)
    _, freqs, values = gaussian_fft_peaks(
        image,
        r1,
        r2,
        len_slack,
        params.top_k_per_band,
    )
    best = -np.inf
    best_pair: Tuple[np.ndarray | None, np.ndarray | None] = (None, None)
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            u = freqs[i]
            v = freqs[j]
            lattice = np.linalg.pinv(np.vstack([u, v]))
            if not np.all(np.isfinite(lattice)):
                continue
            a1 = lattice[:, 0]
            a2 = lattice[:, 1]
            ax_t = np.linalg.norm(a1)
            ay_t = np.linalg.norm(a2)
            if abs(ax_t - ax_star) / ax_star > len_slack:
                continue
            if abs(ay_t - ay_star) / ay_star > len_slack:
                continue
            gamma = angle_between_deg(a1, a2)
            theta = np.mod(np.degrees(np.arctan2(a1[1], a1[0])), 180.0)
            if angle_diff_deg(gamma, gamma_star) > tol_gamma:
                continue
            if angle_diff_deg(theta, theta_star) > tol_theta:
                continue
            penal = 1 + 0.5 * (angle_diff_deg(gamma, gamma_star) / tol_gamma) ** 2
            penal += 0.5 * (angle_diff_deg(theta, theta_star) / tol_theta) ** 2
            penal += 0.25 * ((ax_t - ax_star) / (len_slack * ax_star)) ** 2
            penal += 0.25 * ((ay_t - ay_star) / (len_slack * ay_star)) ** 2
            score = (values[i] + values[j]) / penal
            if score > best:
                best = score
                best_pair = (u, v)
    if best_pair[0] is None:
        u1 = np.array([np.cos(np.radians(theta_star)), np.sin(np.radians(theta_star))])
        u2 = np.array(
            [
                np.cos(np.radians(theta_star + gamma_star)),
                np.sin(np.radians(theta_star + gamma_star)),
            ]
        )
        return u1 / max(ax_star, 1e-9), u2 / max(ay_star, 1e-9)
    return np.array(best_pair[0]), np.array(best_pair[1])


def _adsorbates_from_fwhm(
    image: np.ndarray,
    xy_pf_in: np.ndarray,
    probe_r_px: float,
    r_spot_locked: int,
    params: LatticeParams,
) -> Tuple[np.ndarray, np.ndarray]:
    occ = np.zeros(len(xy_pf_in), dtype=int)
    fwhm_px = np.full(len(xy_pf_in), np.nan)
    if len(xy_pf_in) == 0:
        return occ, fwhm_px

    height, width = image.shape
    r_max = max(
        int(round(min(min(width, height) * 0.05, 10 * max(1, probe_r_px)))),
        r_spot_locked * 2,
    )
    dr = params.fwhm_dr_px

    for idx, (cx, cy) in enumerate(xy_pf_in):
        xmin = max(0, int(np.floor(cx - r_max)))
        xmax = min(width - 1, int(np.ceil(cx + r_max)))
        ymin = max(0, int(np.floor(cy - r_max)))
        ymax = min(height - 1, int(np.ceil(cy + r_max)))
        if xmin > xmax or ymin > ymax:
            continue
        xs = np.arange(xmin, xmax + 1)
        ys = np.arange(ymin, ymax + 1)
        xx, yy = np.meshgrid(xs, ys)
        rr = np.hypot(xx - cx, yy - cy)
        block = image[ymin : ymax + 1, xmin : xmax + 1]
        rvals = rr.ravel()
        ivals = block.ravel()
        nbins = max(3, int(np.ceil(np.max(rvals) / dr)))
        edges = np.linspace(0, nbins * dr, nbins + 1)
        bin_idx = np.minimum(nbins - 1, (rvals / dr).astype(int))
        prof_sum = np.bincount(bin_idx, weights=ivals, minlength=nbins)
        prof_cnt = np.bincount(bin_idx, minlength=nbins)
        with np.errstate(invalid="ignore"):
            prof = prof_sum / np.maximum(prof_cnt, 1)
        rmid = (edges[:-1] + edges[1:]) * 0.5
        if not np.isfinite(prof[0]) or np.sum(np.isfinite(prof)) < 3:
            continue
        tail_start = max(0, int(round((1 - params.fwhm_outer_frac) * len(prof))))
        baseline = np.nanmedian(prof[tail_start:])
        if not np.isfinite(baseline):
            baseline = np.nanmin(prof)
        peak = prof[0]
        if not np.isfinite(peak):
            continue
        half = baseline + 0.5 * (peak - baseline)
        finite_mask = np.isfinite(prof)
        indices = np.where((prof <= half) & finite_mask)[0]
        if len(indices) == 0 or indices[0] == 0:
            continue
        idx0 = indices[0]
        x1 = rmid[idx0 - 1]
        x2 = rmid[idx0]
        y1 = prof[idx0 - 1]
        y2 = prof[idx0]
        if not np.isfinite(y1) or not np.isfinite(y2) or y2 == y1:
            continue
        t = (half - y1) / (y2 - y1)
        r_hwhm = x1 + t * (x2 - x1)
        fwhm_px[idx] = max(0.0, 2.0 * r_hwhm)

    valid = np.isfinite(fwhm_px)
    if np.sum(valid) >= 4:
        labels, centers = kmeans_1d(fwhm_px[valid], 2)
        idx_small = int(np.argmin(centers))
        occ_valid = labels == idx_small
        occ[valid] = occ_valid.astype(int)
        if np.sum(occ) < params.min_occ_for_frame:
            order = np.argsort(fwhm_px[valid])
            k = max(params.top_min_rescue, int(round(params.top_frac_rescue * np.sum(valid))))
            occ[:] = 0
            occ_idx = np.where(valid)[0][order[: min(k, len(order))]]
            occ[occ_idx] = 1
    elif np.any(valid):
        order = np.argsort(fwhm_px[valid])
        k = max(params.top_min_rescue, int(round(params.top_frac_rescue * np.sum(valid))))
        occ_idx = np.where(valid)[0][order[: min(k, len(order))]]
        occ[occ_idx] = 1

    return occ, fwhm_px


def _save_histogram(data: np.ndarray, xlabel: str, path: str, base: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.hist(data, bins="fd")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(f"{base} – FWHM distribution (all frames)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
