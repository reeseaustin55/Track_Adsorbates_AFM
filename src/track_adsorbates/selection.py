"""Interactive helpers for selecting lattice peaks from FFT visualisations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from .lattice import LatticeParameters, enhance_frame
from .video import VideoData


class SelectionCancelledError(RuntimeError):
    """Raised when the user closes the picker without confirming a selection."""


@dataclass
class _SelectionState:
    frame_index: int
    selections: List[tuple[float, float]]


def select_initial_lattice(video: VideoData, frame_width_nm: float) -> LatticeParameters:
    """Launch an interactive picker to derive the initial lattice parameters.

    The user is shown the original colour frame, a filtered grayscale frame, and
    the log-magnitude FFT. Clicking two reciprocal peaks on the FFT establishes
    the lattice basis used for the first fitting pass. A "Next →" button lets
    the user browse frames before committing to a selection.
    """

    if video.frame_count == 0:
        raise ValueError("Video has no frames available for selection")

    if frame_width_nm <= 0:
        raise ValueError("Frame width must be positive to derive lattice parameters")

    pixel_size_nm = frame_width_nm / max(video.frame_shape[1], 1)

    state = _SelectionState(frame_index=0, selections=[])
    result: dict[str, LatticeParameters] = {}

    frames_gray = video.frames_gray
    frames_color = video.frames_color

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    plt.subplots_adjust(bottom=0.2, wspace=0.08)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    original_im = axes[0].imshow(_frame_to_rgb(frames_color[state.frame_index]))
    axes[0].set_title("Original frame")

    filtered = enhance_frame(frames_gray[state.frame_index])
    filtered_im = axes[1].imshow(filtered, cmap="gray")
    axes[1].set_title("Filtered (DoG)")

    fft_image = _fft_image(filtered)
    fft_im = axes[2].imshow(fft_image, cmap="magma")
    axes[2].set_title("FFT (log magnitude)")

    scatter = axes[2].scatter([], [], c="cyan", marker="x", s=80, linewidths=2)

    status = fig.text(0.02, 0.04, _status_message(state, video.frame_count))

    def update_images() -> None:
        frame_idx = state.frame_index
        original_im.set_data(_frame_to_rgb(frames_color[frame_idx]))
        axes[0].set_title(f"Original frame #{frame_idx + 1}")

        filtered_frame = enhance_frame(frames_gray[frame_idx])
        filtered_im.set_data(filtered_frame)
        axes[1].set_title("Filtered (DoG)")

        fft_data = _fft_image(filtered_frame)
        fft_im.set_data(fft_data)
        axes[2].set_title("FFT (log magnitude)")

        axes[2].set_xlim(0, fft_data.shape[1])
        axes[2].set_ylim(fft_data.shape[0], 0)

        state.selections.clear()
        _update_scatter()
        status.set_text(_status_message(state, video.frame_count))
        fig.canvas.draw_idle()

    def _update_scatter() -> None:
        if state.selections:
            scatter.set_offsets(np.array(state.selections))
        else:
            scatter.set_offsets(np.empty((0, 2)))
        status.set_text(_status_message(state, video.frame_count))
        fig.canvas.draw_idle()

    def _handle_click(event) -> None:
        if event.inaxes != axes[2]:
            return
        if event.xdata is None or event.ydata is None:
            return
        if len(state.selections) >= 2:
            return
        state.selections.append((float(event.xdata), float(event.ydata)))
        _update_scatter()

    def _clear(event) -> None:  # noqa: ARG001 - signature required by Matplotlib
        del event
        state.selections.clear()
        _update_scatter()

    def _next_frame(event) -> None:  # noqa: ARG001 - signature required by Matplotlib
        del event
        state.frame_index = (state.frame_index + 1) % video.frame_count
        update_images()

    def _accept(event) -> None:  # noqa: ARG001 - signature required by Matplotlib
        del event
        if len(state.selections) != 2:
            status.set_text("Select exactly two FFT peaks before continuing")
            fig.canvas.draw_idle()
            return
        try:
            params = _parameters_from_selection(
                state.selections,
                frames_gray[state.frame_index].shape,
                pixel_size_nm,
            )
        except ValueError as exc:
            status.set_text(str(exc))
            fig.canvas.draw_idle()
            return

        result["params"] = params
        plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _handle_click)

    btn_done_ax = plt.axes([0.38, 0.05, 0.16, 0.08])
    btn_done = Button(btn_done_ax, "Done")
    btn_done.on_clicked(_accept)

    btn_reset_ax = plt.axes([0.56, 0.05, 0.16, 0.08])
    btn_reset = Button(btn_reset_ax, "Reset")
    btn_reset.on_clicked(_clear)

    btn_next_ax = plt.axes([0.74, 0.05, 0.2, 0.08])
    btn_next = Button(btn_next_ax, "Next →")
    btn_next.on_clicked(_next_frame)

    def _on_close(event) -> None:  # noqa: ARG001
        if "params" not in result:
            status.set_text("Selection cancelled")

    fig.canvas.mpl_connect("close_event", _on_close)

    update_images()
    plt.show()

    if "params" not in result:
        raise SelectionCancelledError("Initial lattice selection was cancelled")
    return result["params"]


def _status_message(state: _SelectionState, total_frames: int) -> str:
    remaining = 2 - len(state.selections)
    if remaining > 0:
        return (
            f"Frame {state.frame_index + 1}/{total_frames} – "
            f"select {remaining} more FFT peak{'s' if remaining > 1 else ''}."
        )
    return "Two peaks selected – press Done to continue or Reset to adjust."


def _frame_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _fft_image(frame: np.ndarray) -> np.ndarray:
    shifted = np.fft.fftshift(np.fft.fft2(frame - np.mean(frame)))
    magnitude = np.log1p(np.abs(shifted))
    if magnitude.max() > 0:
        magnitude /= magnitude.max()
    return magnitude


def _parameters_from_selection(
    points: Sequence[tuple[float, float]],
    shape: tuple[int, int],
    pixel_size_nm: float,
) -> LatticeParameters:
    if len(points) != 2:
        raise ValueError("Exactly two FFT peaks are required to initialise the lattice")

    height, width = shape
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float64)

    frequencies: List[np.ndarray] = []
    for x, y in points:
        vec = np.array([x, y], dtype=np.float64) - center
        freq = vec / np.array([width, height], dtype=np.float64)
        frequencies.append(freq)

    reciprocal = np.stack(frequencies, axis=0)
    if abs(np.linalg.det(reciprocal)) < 1e-10:
        raise ValueError("Selected peaks are colinear; choose two non-colinear points")

    direct_basis = np.linalg.inv(reciprocal)
    a_vec = direct_basis[:, 0]
    b_vec = direct_basis[:, 1]

    len_a = float(np.linalg.norm(a_vec))
    len_b = float(np.linalg.norm(b_vec))

    if len_a <= 0 or len_b <= 0:
        raise ValueError("Unable to compute lattice lengths from the selected peaks")

    scale = pixel_size_nm * 10.0
    a_length_angstrom = len_a * scale
    b_length_angstrom = len_b * scale

    cos_angle = np.clip(np.dot(a_vec, b_vec) / (len_a * len_b), -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_angle)))
    orientation_deg = float((np.degrees(np.arctan2(a_vec[1], a_vec[0])) + 360.0) % 360.0)

    return LatticeParameters(
        a_length=a_length_angstrom,
        b_length=b_length_angstrom,
        angle_deg=angle_deg,
        orientation_deg=orientation_deg,
    )

