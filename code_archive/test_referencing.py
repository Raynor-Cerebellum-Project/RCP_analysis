from pathlib import Path
import numpy as np
from scipy.io import loadmat

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.exporters as sexp
from spikeinterface.core import concatenate_recordings
import matplotlib.pyplot as plt
from probeinterface import Probe
from probeinterface.plotting import plot_probe

# ==============================
# Config
# ==============================
DATA_DIR = Path(
    "/home/bryan/mnt/cullen/Current Project Databases - NHP/2025 Cerebellum prosthesis/Bryan/data"
)
GEOM_MAT = DATA_DIR / "BL_RW_003_Session_1/ImecPrimateStimRec128_042421.mat"
TARGET_STREAM = "RHS2000 amplifier channel"

OUT_BASE = DATA_DIR / "BL_RW_003_Session_1/Sorting Results/combined_all"
OUT_BASE.mkdir(parents=True, exist_ok=True)

intan_root = DATA_DIR / "BL_RW_003_Session_1/Intan"
SESSION_FOLDERS = sorted([p for p in intan_root.iterdir() if p.is_dir()])
print("Found session folders:")
for sf in SESSION_FOLDERS:
    print("  ", sf)

# Global SI job settings
si.set_global_job_kwargs(n_jobs=4, chunk_duration="1s", progress_bar=True)


# ==============================
# Helpers
# ==============================
def load_probe_from_mat(mat_path: Path) -> Probe:
    """Build a ProbeInterface Probe from a Kilosort-style .mat file."""
    mat = loadmat(mat_path)
    x = mat["xcoords"].astype(float).ravel()
    y = mat["ycoords"].astype(float).ravel()
    chanmap0 = mat["chanMap0ind"].astype(int).ravel()  # 0-based device indices
    kcoords = mat.get("kcoords", None)
    kcoords = None if kcoords is None else kcoords.ravel().astype(int)

    pr = Probe(ndim=2)
    pr.set_contacts(positions=np.c_[x, y], shapes="circle", shape_params={"radius": 5})
    pr.set_device_channel_indices(chanmap0)
    if kcoords is not None:
        try:
            pr.set_shank_ids(kcoords)
        except Exception:
            pass
    return pr


def ensure_signed(rec):
    """Convert unsigned dtype to signed (or float) before filtering."""
    if rec.get_dtype().kind == "u":
        rec = spre.unsigned_to_signed(rec)
        rec = spre.astype(rec, dtype="int16")
    return rec


# ---- helper: run sorter + analyze + export to Phy ----
def run_sorter_and_export(
    name: str, recording, base_folder: Path, sorter_params: dict | None = None
):
    sorter_params = sorter_params or {}
    sort_out = base_folder / f"{name}_output"

    sorting = ss.run_sorter(
        name,
        recording,
        folder=sort_out,
        remove_existing_folder=True,
        delete_output_folder=False,
        verbose=True,
        **sorter_params,  # safe if empty
    )

    an_dir = base_folder / f"analyzer_{name}"
    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="binary_folder",
        folder=an_dir,
        sparse=False,
        overwrite=True,
    )
    analyzer.compute(
        [
            "random_spikes",
            "waveforms",
            "templates",
            "noise_levels",
            "spike_amplitudes",
            "principal_components",
        ],
        extension_params={
            "random_spikes": {"method": "uniform", "max_spikes_per_unit": 500},
            "waveforms": {"ms_before": 1.0, "ms_after": 2.0},
            "spike_amplitudes": {"peak_sign": "neg"},
            "principal_components": {"n_components": 3, "mode": "by_channel_local"},
        },
        n_jobs=8,
        chunk_duration="1s",
        progress_bar=True,
        save=True,
    )

    phy_dir = base_folder / f"phy_output_{name}"
    try:
        sexp.export_to_phy(analyzer, output_folder=phy_dir, remove_if_exists=True)
    except TypeError:
        we = analyzer.get_extension("waveforms").get_waveform_extractor()
        sexp.export_to_phy(
            we, output_folder=phy_dir, compute_pc_features=True, compute_amplitudes=True
        )
    return sorting, analyzer


# Load recordings from all subfolders
recordings = [
    se.read_split_intan_files(
        folder_path=folder,
        mode="concatenate",
        stream_name=TARGET_STREAM,
        use_names_as_ids=True,
    )
    for folder in SESSION_FOLDERS
]

rec_concat = concatenate_recordings(recordings)


print(
    f"Loaded Intan: fs={rec_concat.get_sampling_frequency()} Hz, chans={rec_concat.get_num_channels()}"
)

rec = ensure_signed(rec_concat)
rec_pp = spre.highpass_filter(rec, freq_min=300)
rec_pp = spre.common_reference(rec_pp, reference="global", operator="median")

# Add ring reference if needed / stim channels vs nonstim channels

# Add python version of artifact correction here if needed

rec_pp = rec_pp.save(folder=OUT_BASE / "preprocessed", overwrite=True)

# 3) Attach probe geometry
try:
    assert GEOM_MAT.exists(), f"Missing geometry file: {GEOM_MAT}"
    probe = load_probe_from_mat(GEOM_MAT)
    rec_pp = rec_pp.set_probe(probe, in_place=False)
    print("Attached probe geometry from .mat")
except Exception as e:
    print(
        f"Warning: failed to attach probe from {GEOM_MAT} ({e}). "
        "Proceeding without geometry; MS5 will use adjacency_radius=-1."
    )

fig, ax = plt.subplots()
plot_probe(probe, ax=ax)  # no extra kwargs in older probeinterface
ax.set_aspect("equal", adjustable="box")
out_png = OUT_BASE / "probe_layout.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print("Saved:", out_png)
plt.close(fig)


import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class TraceBrowser:
    def __init__(
        self,
        recording,
        out_dir: Path,
        channels_per_page=5,
        duration_s=5.0,
        t0_s=0.0,
        scale="auto",
    ):
        """
        recording: SpikeInterface Recording (e.g., rec_pp)
        out_dir: where screenshots are saved
        channels_per_page: rows per page (default 5)
        duration_s: initial window length in seconds
        t0_s: initial start time (seconds)
        scale: 'auto' or float (uV) for y-lims symmetric about 0
        """
        self.rec = recording
        self.fs = float(recording.get_sampling_frequency())
        self.n_ch = int(recording.get_num_channels())
        self.ch_ids = list(recording.get_channel_ids())
        self.page = 0
        self.cpp = int(channels_per_page)
        self.duration = float(duration_s)
        self.t0 = float(t0_s)
        self.scale = scale
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.fig, self.axes = plt.subplots(
            self.cpp, 1, sharex=True, figsize=(12, 8), constrained_layout=True
        )
        if self.cpp == 1:
            self.axes = [self.axes]
        self.fig.canvas.manager.set_window_title(
            "TraceBrowser (←/→ pan, ↑/↓ page, [/] duration, g goto, s save, q quit)"
        )
        self._draw()

        # key bindings
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ---------------- internal helpers ----------------
    def _page_channels(self):
        i0 = self.page * self.cpp
        i1 = min(i0 + self.cpp, self.n_ch)
        return self.ch_ids[i0:i1]

    def _draw(self):
        # clamp time range
        n_frames = (
            int(self.rec.get_total_duration() * self.fs)
            if hasattr(self.rec, "get_total_duration")
            else None
        )
        start = max(0, int(self.t0 * self.fs))
        stop = int((self.t0 + self.duration) * self.fs)
        if n_frames is not None:
            stop = min(stop, n_frames)

        chans = self._page_channels()
        traces = self.rec.get_traces(
            start_frame=start, end_frame=stop, channel_ids=chans
        )
        t = np.arange(traces.shape[0]) / self.fs + (start / self.fs)

        # auto scale per channel
        if self.scale == "auto":
            # robust scale: median abs dev
            mad = (
                np.median(
                    np.abs(traces - np.median(traces, axis=0, keepdims=True)), axis=0
                )
                + 1e-9
            )
            ylim = 6 * mad  # ~±6*MAD
        else:
            ylim = np.full(traces.shape[1], float(self.scale))

        for ax in self.axes:
            ax.cla()
        for i, ch in enumerate(chans):
            ax = self.axes[i]
            ax.plot(t, traces[:, i], linewidth=0.8)
            ax.set_ylabel(f"ch {ch}", rotation=0, ha="right", va="center")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-ylim[i], +ylim[i])
        # hide unused axes on the last page
        for j in range(len(chans), self.cpp):
            self.axes[j].cla()
            self.axes[j].axis("off")

        self.axes[-1].set_xlabel("Time (s)")
        self.fig.suptitle(
            f"TraceBrowser | page {self.page + 1}/{math.ceil(self.n_ch / self.cpp)} | "
            f"channels {self.page * self.cpp}-{min((self.page + 1) * self.cpp - 1, self.n_ch - 1)} | "
            f"t=[{start / self.fs:.3f}, {stop / self.fs:.3f}] s | fs={self.fs:.1f} Hz"
        )

        self.fig.canvas.draw_idle()

    def _pan(self, frac):
        self.t0 = max(0.0, self.t0 + frac * self.duration)
        self._draw()

    def _dur(self, factor):
        self.duration = max(0.05, self.duration * factor)
        self._draw()

    def _goto(self):
        try:
            import tkinter as tk
            from tkinter.simpledialog import askfloat

            root = tk.Tk()
            root.withdraw()
            val = askfloat("Go to time", "Start time t0 (seconds):", minvalue=0.0)
            root.destroy()
        except Exception:
            # fallback to console input
            val = None
            try:
                val = float(input("Start time t0 (s): "))
            except Exception:
                pass
        if val is not None and val >= 0:
            self.t0 = val
            self._draw()

    def _save(self):
        fname = (
            self.out_dir
            / f"trace_view_p{self.page + 1}_t{self.t0:.3f}s_{self.duration:.3f}s.png"
        )
        self.fig.savefig(fname, dpi=200, bbox_inches="tight")
        print("Saved:", fname)

    def _on_key(self, ev):
        if ev.key == "right":
            self._pan(+0.5)
        elif ev.key == "left":
            self._pan(-0.5)
        elif ev.key == "up":
            if self.page > 0:
                self.page -= 1
                self._draw()
        elif ev.key == "down":
            if (self.page + 1) * self.cpp < self.n_ch:
                self.page += 1
                self._draw()
        elif ev.key == "[":
            self._dur(0.5)
        elif ev.key == "]":
            self._dur(2.0)
        elif ev.key == "g":
            self._goto()
        elif ev.key == "s":
            self._save()
        elif ev.key in ("q", "escape"):
            plt.close(self.fig)

    def show(self):
        plt.show()


# ---- usage (put this after you create `rec_pp` and `OUT_BASE`) ----
browser = TraceBrowser(
    recording=rec_pp, out_dir=OUT_BASE, channels_per_page=5, duration_s=5.0, t0_s=0.0
)
browser.show()
