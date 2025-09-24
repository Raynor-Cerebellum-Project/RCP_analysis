from pathlib import Path
import re, numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
import io, json, zipfile
from probeinterface.plotting import plot_probe

import spikeinterface as si

# import helpers from sibling modules
from ..functions.intan_preproc import load_stim_geometry, get_chanmap_perm_from_geom, reorder_recording_to_geometry, read_intan_recording, make_identity_probe_from_geom, load_stim_detection

def _first_stim_time_from_npz(stim_npz: Path) -> float | None:
    """
    Return first stim time (s) using 'trigger_pairs' and 'meta' in the NPZ.
    If no triggers are present, returns None.
    """
    stim_npz = Path(stim_npz)
    if not stim_npz.exists():
        return None

    with np.load(stim_npz, allow_pickle=False) as z:
        # fs from meta
        fs_hz = 1.0
        if "meta" in z:
            try:
                meta_raw = z["meta"].item()
                meta = json.loads(meta_raw) if isinstance(meta_raw, (str, bytes)) else dict(meta_raw)
                fs_hz = float(meta.get("fs_hz", 1.0))
            except Exception:
                pass

        # first trigger start sample
        if "trigger_pairs" in z and z["trigger_pairs"].size:
            tp = z["trigger_pairs"]
            first_start = int(tp[0, 0])
            return first_start / fs_hz

    return None

def _detect_stim_channels_from_npz(
    stim_npz_path: Path,
    eps: float = 1e-12,
    min_edges: int = 1,
) -> np.ndarray:
    """
    Return GEOMETRY-ORDERED indices of stimulated channels by counting 0→nonzero
    rising edges over the full 'stim_traces' array stored in the NPZ.

    Note: extract_and_save_stim_npz() sets meta['order'] to 'geometry' when
    it can reorder using chanmap_perm, so indices should already be in geometry order.
    """
    stim_npz_path = Path(stim_npz_path)
    if not stim_npz_path.exists():
        return np.array([], dtype=int)

    with np.load(stim_npz_path, allow_pickle=False) as z:
        if "stim_traces" not in z:
            return np.array([], dtype=int)

        X = z["stim_traces"]  # (n_channels, n_samples)
        if X.ndim != 2 or X.shape[1] < 2:
            return np.array([], dtype=int)

        # per-channel thresholds: midpoint between 5th & 95th percentiles
        p5  = np.nanpercentile(X, 5, axis=1)
        p95 = np.nanpercentile(X, 95, axis=1)
        thr = 0.5 * (p5 + p95)

        above = (X > (thr[:, None] + eps))
        rising = above[:, 1:] & (~above[:, :-1])    # (ch, frames-1)
        counts = rising.sum(axis=1)                 # per-channel counts

        active = np.where(counts >= int(min_edges))[0].astype(int)

        # If meta says order=='geometry', active are geometry indices already.
        # If order=='device' we don't have a perm in this NPZ; return as-is.
        # (extract_and_save_stim_npz normally sets geometry when it can.)
        return np.unique(active)


def _find_interp_dir_for_session(preproc_root: Path, sess_name: str) -> Path | None:
    """
    Find the artifact-corrected checkpoint directory like:
      pp_local_<rmin>_<rmax>__interp_<sess_name>
    Returns the newest match if multiple exist.
    """
    root = Path(preproc_root) / "NPRW"
    if not root.exists():
        return None
    pat = re.compile(rf"^pp_.*__interp_{re.escape(sess_name)}$")
    cands = [p for p in root.iterdir() if p.is_dir() and pat.match(p.name)]
    if not cands:
        return None
    # pick the most recent
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]

def _load_cached_recording(ac_dir: Path):
    """Robust loader for SI cache folders."""
    ac_dir = Path(ac_dir)
    # Try the folder directly (SI >= 0.97 usually works)
    try:
        return si.load(ac_dir)
    except Exception:
        pass

    # Fallback to explicit si_folder.json (older saves)
    j = ac_dir / "si_folder.json"
    if j.exists():
        return si.load_extractor(j)

    # Nothing worked
    raise FileNotFoundError(f"Could not load SpikeInterface recording from {ac_dir} "
                            f"(no si_folder.json found and si.load() failed).")

def _pick_window_frames(fs, n_total, t0, pre_s, post_s, view_s=None):
    if view_s is not None:
        v0, v1 = float(view_s[0]), float(view_s[1])
        if v1 < v0:
            v0, v1 = v1, v0  # swap
        start = max(0, int(round((t0 + v0) * fs)))
        end   = min(n_total, int(round((t0 + v1) * fs)))
    else:
        start = max(0, int(round((t0 - pre_s) * fs)))
        end   = min(n_total, int(round((t0 + post_s) * fs)))
    if end <= start:
        end = min(n_total, start + int(round(0.3 * fs)))  # 300 ms fallback
    return start, end


def plot_all_quads_for_session(
    sess_folder: Path,
    geom_path: Path,
    neural_stream: str,
    stim_stream: str,
    out_dir: Path,
    duration_s: float = 1.0,
    stim_npz_path: Path | None = None,
    pre_s: float = 0.3,
    post_s: float = 0.3,
    probe_ratio: float = 0.4,
    preproc_root: Path = Path("/home/bryan/mnt/cullen/Current Project Databases - NHP/2025 Cerebellum prosthesis/Nike/NRR_RW001/results/checkpoints"),
    show_pca_windows: bool = True,
    show_blocks: bool = True,
    template_samples_before: float | None = None,
    template_samples_after: float | None = None,
    view_s: tuple[float, float] | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load AC checkpoint
    ac_dir = _find_interp_dir_for_session(preproc_root, sess_folder.name)
    if ac_dir is None:
        raise FileNotFoundError(
            f"No artifact-corrected checkpoint found for session '{sess_folder.name}' "
            f"in {preproc_root/'NPRW'} (looked for 'pp_*__interp_{sess_folder.name}')"
        )
    rec = _load_cached_recording(ac_dir)

    # 2) ensure probe present / geometry order
    try:
        probe = rec.get_probe()
    except Exception:
        probe = None
    if probe is None:
        geom = load_stim_geometry(geom_path)
        probe = make_identity_probe_from_geom(geom, radius_um=5.0)
        rec = rec.set_probe(probe, in_place=False)

    fs = float(rec.get_sampling_frequency())
    try:
        n_total = rec.get_num_frames()
    except TypeError:
        n_total = rec.get_num_frames(0) if hasattr(rec, "get_num_frames") else rec.get_num_samples()

    locs = rec.get_channel_locations()
    chan_ids = [str(x) for x in rec.get_channel_ids()]

    # 3) pick window around first stim
    t0 = None
    if stim_npz_path is not None:
        try:
            t0 = _first_stim_time_from_npz(stim_npz_path)  # keep None if no triggers
        except Exception:
            print(f"[WARN] could not read stim_npz at {stim_npz_path}; centering at t=0")
            t0 = None

    if t0 is None:
        print("[WARN] no stim found; centering at t=0 and ignoring view_s")
        t0 = 0.0
        has_stim = False
    else:
        has_stim = True

    start, end = _pick_window_frames(fs, n_total, t0, pre_s, post_s, view_s=(view_s if has_stim else None))
    Xwin = rec.get_traces(start_frame=start, end_frame=end, return_in_uV=True)
    t = (np.arange(Xwin.shape[0], dtype=float) + start) / fs - t0  # axis still relative to stim
    
    # --- compute a global y-axis range across all channels ---
    meds = np.median(Xwin, axis=0)
    mads = np.median(np.abs(Xwin - meds), axis=0) + 1e-9
    ymins = meds - 4 * mads
    ymaxs = meds + 4 * mads
    global_ylim = (float(ymins.min()), float(ymaxs.max()))

    # --- Load triggers/blocks + build visible trigger times
    trig_t = np.array([], dtype=float)
    block_edges_t = None

    if stim_npz_path is not None:
        try:
            stim = load_stim_detection(stim_npz_path)

            # triggers: take start column if present
            tp = np.asarray(stim.get("trigger_pairs", []))
            if tp.ndim == 2 and tp.shape[1] == 2 and tp.size:
                trigs_start = tp[:, 0].astype(np.int64)
                trig_t = trigs_start.astype(float) / fs - t0
            else:
                trig_t = np.array([], dtype=float)

            # blocks: use (B,2) sample-space directly
            bb = np.asarray(stim.get("block_bounds_samples", []))
            if bb.ndim == 2 and bb.shape[1] == 2 and bb.size:
                starts_s = bb[:, 0].astype(np.int64) / fs - t0  # seconds
                ends_s   = bb[:, 1].astype(np.int64) / fs - t0
                # flatten to edges [start1, end1, start2, end2, ...] and sort
                block_edges_t = np.sort(np.concatenate([starts_s, ends_s]))
            else:
                block_edges_t = None

        except Exception as e:
            print(f"[WARN] could not load trig/block info: {e}")
            trig_t = np.array([], dtype=float)
            block_edges_t = None

    # Template window (ms) – prefer explicit args, else fall back to global params if present
    if template_samples_before is None or template_samples_after is None:
        tb, ta = 15, 15   # samples (fallback)
    else:
        tb, ta = int(template_samples_before), int(template_samples_after)

    tb_s, ta_s = tb / fs, ta / fs   # samples -> seconds

    if trig_t.size:
        vis = (trig_t >= t[0] - tb_s) & (trig_t <= t[-1] + ta_s)
        trig_t_vis = trig_t[vis]
    else:
        trig_t_vis = np.array([], dtype=float)

    # --- Stim-channel highlighting (optional)
    stim_idx = np.array([], dtype=int)
    if stim_npz_path is not None:
        try:
            # removed max_seconds kwarg; keep eps/min_edges
            stim_idx = _detect_stim_channels_from_npz(
                stim_npz_path, eps=1e-12, min_edges=1
            )
        except Exception as e:
            print(f"[WARN] stim site detection failed: {e}")

    # 5) group channels by geometry order and plot
    order = np.lexsort((locs[:, 0], locs[:, 1]))  # ascending y, then x
    nrows, ncols = 4, 4
    group = nrows * ncols
    groups = [order[i:i + group] for i in range(0, order.size, group)]

    for gi, sel in enumerate(groups):
        fig = plt.figure(figsize=(20, 10), constrained_layout=True)
        gs_main = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[5, probe_ratio], wspace=0.15)
        gs_grid = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs_main[0, 0],
                                                   wspace=0.05, hspace=0.15)
        axes = np.array([[plt.subplot(gs_grid[r, c]) for c in range(ncols)] for r in range(nrows)])
        ax_probe = plt.subplot(gs_main[0, 1])

        for k_idx in range(group):
            r0, c0 = divmod(k_idx, ncols)
            r = nrows - 1 - r0  # flip vertically
            ax = axes[r, c0]
            if k_idx >= sel.size:
                ax.axis("off"); continue
            ch = int(sel[k_idx])
            y = Xwin[:, ch]
            med = np.median(y)
            mad = np.median(np.abs(y - med)) + 1e-9

            # --- Block shading (behind trace)
            if block_edges_t is not None and block_edges_t.size >= 2:
                x0, x1 = float(t[0]), float(t[-1])
                edges = np.concatenate([[x0], block_edges_t, [x1]])
                for si_ in range(0, len(edges) - 1):
                    a, b = edges[si_], edges[si_ + 1]
                    if b <= x0 or a >= x1:
                        continue
                    a = max(a, x0); b = min(b, x1)
                    if si_ % 2 == 1:
                        ax.axvspan(a, b, color="0.92", alpha=0.6, zorder=0)

            # --- Trace + reference lines
            ax.plot(t, y, lw=0.8, zorder=1)
            ax.axhline(med, lw=0.3, alpha=0.3, zorder=1)
            ax.axvline(0.0, linestyle="--", linewidth=0.8, zorder=2)

            # --- PCA template windows
            if show_pca_windows and trig_t_vis.size:
                for tt in trig_t_vis:
                    a = tt - tb_s
                    b = tt + ta_s
                    if b < t[0] or a > t[-1]:
                        continue
                    ax.axvspan(max(a, t[0]), min(b, t[-1]),
                               color="tab:green", alpha=0.15, lw=0, zorder=0.5)
                    ax.axvline(tt, color="tab:green", lw=0.6, alpha=0.6, zorder=2)

            ax.set_title(chan_ids[ch], fontsize=9)
            ax.set_ylim(*global_ylim)
            if r != nrows - 1:
                ax.tick_params(labelbottom=False)
            if c0 != 0:
                ax.tick_params(labelleft=False)

        axes[-1, 0].set_xlabel("Time (s) rel. first stim")
        axes[-1, 0].set_ylabel("µV")

        # --- probe coloring for stim sites
        n_contacts = probe.get_contact_count()
        contacts_colors = np.array(["none"] * n_contacts, dtype=object)
        if stim_idx.size:
            stim_idx_clipped = stim_idx[(stim_idx >= 0) & (stim_idx < n_contacts)]
            contacts_colors[stim_idx_clipped] = "tab:red"
        plot_probe(probe, ax=ax_probe, with_contact_id=False, contacts_colors=contacts_colors)

        # blue box around the 16 channels shown
        xs, ys = locs[sel, 0], locs[sel, 1]
        pad = max(6.0, 0.03 * float(max(xs.max() - xs.min(), ys.max() - ys.min())))
        rect = patches.Rectangle(
            (xs.min() - pad, ys.min() - pad),
            (xs.max() - xs.min()) + 2 * pad,
            (ys.max() - ys.min()) + 2 * pad,
            linewidth=2.0, edgecolor="tab:blue", facecolor="none", zorder=4
        )
        ax_probe.add_patch(rect)
        ax_probe.set_aspect("equal", adjustable="box")
        x_min, x_max = float(locs[:, 0].min()), float(locs[:, 0].max())
        y_min, y_max = float(locs[:, 1].min()), float(locs[:, 1].max())
        mx, my = 0.06 * (x_max - x_min + 1e-6), 0.06 * (y_max - y_min + 1e-6)
        ax_probe.set_xlim(x_min - mx, x_max + mx)
        ax_probe.set_ylim(y_min - my, y_max + my)
        ax_probe.set_xticks([]); ax_probe.set_yticks([])

        n_stim = int(stim_idx.size)
        if hasattr(fig, "set_constrained_layout_pads"):
            fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.06, wspace=0.20, hspace=0.30)

        st = fig.suptitle(
            f"{sess_folder.name} — AC ({neural_stream}) (panel {gi:02d}) • stim sites: {n_stim}",
            y=1.02, fontsize=12
        )

        out_png = out_dir / f"{sess_folder.name}_interp_probe+4x4_panel{gi:02d}.png"
        fig.savefig(out_png, dpi=180, bbox_inches="tight", bbox_extra_artists=[st])
        plt.close(fig)
        
def plot_all_quads_firing_rate(
    fr_hz: np.ndarray,         # (n_units, n_bins)
    t: np.ndarray,             # (n_bins,)
    locs: np.ndarray,          # (n_units, 2) geometry positions
    stim_times: np.ndarray = None,
    out_dir: Path = Path("fr_plots"),
    probe_ratio: float = 0.4,
    title: str = "Firing rate (Hz)"
):
    out_dir.mkdir(exist_ok=True, parents=True)

    order = np.lexsort((locs[:,0], locs[:,1]))
    nrows, ncols = 4, 4
    group = nrows * ncols
    groups = [order[i:i+group] for i in range(0, order.size, group)]

    for gi, sel in enumerate(groups):
        fig = plt.figure(figsize=(20,10), constrained_layout=True)
        gs_main = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[5, probe_ratio])
        gs_grid = gridspec.GridSpecFromSubplotSpec(
            nrows, ncols, subplot_spec=gs_main[0,0], wspace=0.05, hspace=0.15
        )
        axes = np.array([[plt.subplot(gs_grid[r,c]) for c in range(ncols)] for r in range(nrows)])
        ax_probe = plt.subplot(gs_main[0,1])

        for k_idx, unit in enumerate(sel):
            r0, c0 = divmod(k_idx, ncols)
            r = nrows - 1 - r0
            ax = axes[r,c0]
            if k_idx >= sel.size:
                ax.axis("off"); continue
            y = fr_hz[unit,:]
            ax.plot(t, y, lw=1.0)
            if stim_times is not None:
                for st in stim_times:
                    ax.axvline(st, color="red", lw=0.5, alpha=0.5)
            ax.set_title(f"unit {unit}", fontsize=8)

        ax_probe.scatter(locs[:,0], locs[:,1], s=10, c="k")
        xs, ys = locs[sel,0], locs[sel,1]
        rect = patches.Rectangle((xs.min(), ys.min()), xs.ptp(), ys.ptp(),
                                 edgecolor="blue", facecolor="none")
        ax_probe.add_patch(rect)
        ax_probe.set_aspect("equal")

        fig.suptitle(f"{title} – panel {gi}")
        out_png = out_dir / f"fr_panel{gi:02d}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
