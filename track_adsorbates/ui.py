"""Simple Tkinter GUI for the adsorbate diffusion analysis pipeline."""
from __future__ import annotations

import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Dict

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from track_adsorbates.analysis import analyze_adsorbate_diffusion
else:
    from .analysis import analyze_adsorbate_diffusion


class AnalysisGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Adsorbate Diffusion Analysis")
        self.root.geometry("480x420")
        self.root.resizable(False, False)

        self.video_folder_var = tk.StringVar()
        self.video_name_var = tk.StringVar()
        self.video_file_var = tk.StringVar()
        self.width_nm_var = tk.StringVar(value="10.0")
        self.ax_var = tk.StringVar(value="3.0")
        self.ay_var = tk.StringVar(value="3.0")
        self.gamma_var = tk.StringVar(value="60.0")
        self.probe_var = tk.StringVar(value="1.5")
        self.max_frames_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Idle")

        self._build_ui()

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill=tk.BOTH, expand=True)

        row = 0
        tk.Label(frame, text="Video folder:").grid(row=row, column=0, sticky="w")
        entry_folder = tk.Entry(frame, textvariable=self.video_folder_var, width=40)
        entry_folder.grid(row=row, column=1, sticky="ew", padx=(6, 6))
        tk.Button(frame, text="Browse", command=self._browse_folder).grid(row=row, column=2, sticky="ew")

        row += 1
        tk.Label(frame, text="Video file name:").grid(row=row, column=0, sticky="w")
        tk.Entry(frame, textvariable=self.video_name_var, width=40).grid(row=row, column=1, columnspan=2, sticky="ew", padx=(6, 6))

        row += 1
        tk.Label(frame, text="Or choose file directly:").grid(row=row, column=0, sticky="w")
        tk.Entry(frame, textvariable=self.video_file_var, width=40).grid(row=row, column=1, sticky="ew", padx=(6, 6))
        tk.Button(frame, text="Browse", command=self._browse_file).grid(row=row, column=2, sticky="ew")

        separators = [
            ("Video width (nm):", self.width_nm_var),
            ("a₁ spacing (Å):", self.ax_var),
            ("a₂ spacing (Å):", self.ay_var),
            ("γ between a₁/a₂ (deg):", self.gamma_var),
            ("Probe diameter (Å):", self.probe_var),
            ("Max frames (optional):", self.max_frames_var),
        ]

        for label_text, variable in separators:
            row += 1
            tk.Label(frame, text=label_text).grid(row=row, column=0, sticky="w")
            tk.Entry(frame, textvariable=variable, width=15).grid(row=row, column=1, sticky="w", padx=(6, 6))

        row += 1
        tk.Button(frame, text="Run analysis", command=self._run_analysis).grid(row=row, column=0, columnspan=3, pady=(12, 6), sticky="ew")

        row += 1
        tk.Label(frame, textvariable=self.status_var, anchor="w").grid(row=row, column=0, columnspan=3, sticky="ew")

        frame.columnconfigure(1, weight=1)

    def _browse_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select video folder")
        if folder:
            self.video_folder_var.set(folder)

    def _browse_file(self) -> None:
        file_path = filedialog.askopenfilename(title="Select AFM video", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if file_path:
            self.video_file_var.set(file_path)
            self.video_folder_var.set(str(Path(file_path).parent))
            self.video_name_var.set(Path(file_path).name)

    def _run_analysis(self) -> None:
        try:
            params = self._collect_parameters()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self.status_var.set("Running analysis…")
        self.root.update_idletasks()

        thread = threading.Thread(target=self._execute_analysis, args=(params,), daemon=True)
        thread.start()

    def _collect_parameters(self) -> Dict[str, object]:
        video_file = self.video_file_var.get().strip()
        if video_file:
            video_path = Path(video_file)
        else:
            folder = self.video_folder_var.get().strip()
            name = self.video_name_var.get().strip()
            if not folder or not name:
                raise ValueError("Please provide a video file or folder and filename.")
            video_path = Path(folder) / name
        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")

        try:
            width_nm = float(self.width_nm_var.get())
            ax = float(self.ax_var.get())
            ay = float(self.ay_var.get())
            gamma = float(self.gamma_var.get())
            probe = float(self.probe_var.get()) if self.probe_var.get() else None
            max_frames = int(self.max_frames_var.get()) if self.max_frames_var.get() else None
        except ValueError as exc:
            raise ValueError("Numeric fields must contain valid numbers.") from exc

        if width_nm <= 0 or ax <= 0 or ay <= 0:
            raise ValueError("Video width and lattice spacings must be positive.")

        return {
            "vid_path": str(video_path),
            "vid_width_nm": width_nm,
            "ax_a": ax,
            "ay_a": ay,
            "gamma_deg_guess": gamma,
            "probe_diam_a": probe,
            "max_frames": max_frames,
        }

    def _execute_analysis(self, params: Dict[str, object]) -> None:
        try:
            results = analyze_adsorbate_diffusion(**params)
        except Exception as exc:  # noqa: BLE001
            self.status_var.set("Analysis failed. See console for details.")
            messagebox.showerror("Analysis failed", str(exc))
            raise
        else:
            msg = (
                f"D = {results.D:.4g} nm²/s\n"
                f"Steps used: {results.N_steps}\n"
                f"Δt = {results.delta_t:.4g} s"
            )
            self.status_var.set("Analysis completed successfully.")
            messagebox.showinfo("Analysis complete", msg)

    def run(self) -> None:
        self.root.mainloop()


def launch_gui() -> None:
    """Launch the Tkinter GUI."""
    gui = AnalysisGUI()
    gui.run()


if __name__ == "__main__":
    launch_gui()
