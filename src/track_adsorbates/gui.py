"""Tkinter GUI for running the adsorbate tracking pipeline."""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox

from .pipeline import PipelineConfig, run_pipeline
from .video import load_video


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Adsorbate Tracker")
        self.geometry("480x600")
        self.resizable(False, False)
        self.video_path_var = tk.StringVar()
        self.approx_a_var = tk.DoubleVar(value=3.0)
        self.approx_b_var = tk.DoubleVar(value=3.0)
        self.angle_var = tk.DoubleVar(value=90.0)
        self.frame_width_var = tk.DoubleVar(value=10.0)
        self.wiggle_var = tk.DoubleVar(value=10.0)
        self.atom_diameter_var = tk.DoubleVar(value=1.5)
        self.drift_allowance_var = tk.DoubleVar(value=2.0)
        self.threshold_var = tk.DoubleVar(value=0.5)

        self._build_ui()

    def _build_ui(self) -> None:
        padding = {"padx": 10, "pady": 5, "sticky": "we"}

        tk.Label(self, text="Video file").grid(row=0, column=0, **padding)
        entry = tk.Entry(self, textvariable=self.video_path_var)
        entry.grid(row=0, column=1, **padding)
        tk.Button(self, text="Browse", command=self._browse_video).grid(row=0, column=2, **padding)

        row = 1
        for label, var in [
            ("Approx. a (Å)", self.approx_a_var),
            ("Approx. b (Å)", self.approx_b_var),
            ("Angle (deg)", self.angle_var),
            ("Frame width (nm)", self.frame_width_var),
            ("Wiggle (%)", self.wiggle_var),
            ("Atom diameter (Å)", self.atom_diameter_var),
            ("Drift allowance (atoms)", self.drift_allowance_var),
            ("Brightness threshold", self.threshold_var),
        ]:
            tk.Label(self, text=label).grid(row=row, column=0, **padding)
            tk.Entry(self, textvariable=var).grid(row=row, column=1, columnspan=2, **padding)
            row += 1

        self.progress = tk.StringVar(value="Idle")
        tk.Label(self, textvariable=self.progress, anchor="w").grid(row=row, column=0, columnspan=3, **padding)
        row += 1

        tk.Button(self, text="Run", command=self._run).grid(row=row, column=0, columnspan=3, pady=15)

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if path:
            self.video_path_var.set(path)

    def _run(self) -> None:
        path = self.video_path_var.get()
        if not path:
            messagebox.showerror("Error", "Please select a video file")
            return

        config = PipelineConfig(
            approx_a_angstrom=float(self.approx_a_var.get()),
            approx_b_angstrom=float(self.approx_b_var.get()),
            approx_angle_deg=float(self.angle_var.get()),
            frame_width_nm=float(self.frame_width_var.get()),
            wiggle_percent=float(self.wiggle_var.get()),
            atom_diameter_angstrom=float(self.atom_diameter_var.get()),
            drift_allowance_atoms=float(self.drift_allowance_var.get()),
            brightness_threshold=float(self.threshold_var.get()),
        )

        threading.Thread(
            target=self._run_pipeline,
            args=(path, config),
            daemon=True,
        ).start()

    def _run_pipeline(self, path: str, config: PipelineConfig) -> None:
        try:
            self.progress.set("Loading video...")
            video = load_video(path)
            self.progress.set("Processing frames...")
            result = run_pipeline(video, config)
            self.progress.set("Done")
            messagebox.showinfo(
                "Complete",
                "\n".join(
                    [
                        f"True lattice: {result.lattice_json}",
                        f"Overlay video: {result.overlay_video_path}",
                        f"Diffusion summary: {result.diffusion_json}",
                        f"Frame diffusion CSV: {result.diffusion_csv}",
                    ]
                ),
            )
        except Exception as exc:  # pragma: no cover - GUI feedback
            self.progress.set("Error")
            messagebox.showerror("Processing error", str(exc))


def launch_app() -> None:
    App().mainloop()


if __name__ == "__main__":  # pragma: no cover
    launch_app()
