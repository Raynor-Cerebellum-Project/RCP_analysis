from pathlib import Path
import re, numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
import io, json, zipfile
from probeinterface.plotting import plot_probe
import spikeinterface as si

# import helpers from sibling modules
from ..functions.intan_preproc import load_stim_geometry, make_identity_probe_from_geom, load_stim_detection

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

def detect_stim_channels_from_npz(
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

def build_probe_and_locs_from_geom(geom_path: Path, radius_um: float = 5.0):
    """Load your saved geometry -> ProbeInterface Probe + (n_ch,2) locs."""
    geom = load_stim_geometry(geom_path)                  # your project format
    probe = make_identity_probe_from_geom(geom, radius_um=radius_um)  # ProbeInterface Probe
    locs  = probe.contact_positions.astype(float)         # (n_ch, 2)
    return probe, locs

def plot_all_quads_for_session(
    sess_folder: Path,
    geom_path: Path,
    neural_stream: str,
    out_dir: Path,
    peaks: np.ndarray | None = None,
    peak_t_s: np.ndarray | None = None,
    stim_npz_path: Path | None = None,
    pre_s: float = 0.3,
    post_s: float = 0.3,
    probe_ratio: float = 0.1,
    preproc_root: Path = Path("/home/bryan/mnt/cullen/Current Project Databases - NHP/2025 Cerebellum prosthesis/Nike/NRR_RW001/results/checkpoints"),
    show_pca_windows: bool = True,
    show_blocks: bool = True,
    template_samples_before: float | None = None,
    template_samples_after: float | None = None,
    view_s: tuple[float, float] | None = None,
    padding: float = 15.0,
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
    
    zero_in_view = (t[0] <= 0.0 <= t[-1])
    
    # Convert absolute peak times to "seconds rel. first stim"
    if peak_t_s is not None:
        peak_t_rel = peak_t_s - float(t0)   # shape (n_peaks,)
    else:
        peak_t_rel = None

    # Convenience: grab channel indices from peaks if provided
    if peaks is not None and 'channel_index' in peaks.dtype.names:
        peak_ch = peaks['channel_index'].astype(int, copy=False)
    else:
        peak_ch = None
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
            stim_idx = detect_stim_channels_from_npz(
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
        gs_main = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[5, probe_ratio], wspace=0.05)
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
            if zero_in_view:
                ax.axvline(0.0, linestyle="--", linewidth=0.8, zorder=2)

            if (peak_t_rel is not None) and (peak_ch is not None):
                in_ch = (peak_ch == ch)
                if np.any(in_ch):
                    # restrict to the current time window shown
                    in_win = (peak_t_rel >= t[0]) & (peak_t_rel <= t[-1])
                    sel_peaks = np.where(in_ch & in_win)[0]
                    if sel_peaks.size:
                        # draw small markers at the trace; using nearest sample for y-position
                        # (fast + visually informative)
                        # Map each peak time to nearest index in current window:
                        idx = np.clip(np.searchsorted(t, peak_t_rel[sel_peaks]) - 1, 0, t.size - 1)
                        ax.scatter(
                            peak_t_rel[sel_peaks], y[idx],
                            s=8, zorder=3, alpha=0.75, marker='o',
                            color='red'
                        )
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

        # bottom-left axis label
        if has_stim:
            axes[-1, 0].set_xlabel("Time (s) rel. first stim")
        else:
            axes[-1, 0].set_xlabel("Time (s)")

        axes[-1, 0].set_ylabel("µV")

                # --- probe coloring for stim sites (unchanged) ---
        n_contacts = probe.get_contact_count()
        contacts_colors = np.array(["none"] * n_contacts, dtype=object)
        if stim_idx.size:
            stim_idx_clipped = stim_idx[(stim_idx >= 0) & (stim_idx < n_contacts)]
            contacts_colors[stim_idx_clipped] = "tab:red"
        plot_probe(probe, ax=ax_probe, with_contact_id=False, contacts_colors=contacts_colors)

        # blue box around the 16 channels shown (unchanged)
        xs, ys = locs[sel, 0], locs[sel, 1]
        pad = max(6.0, 0.03 * float(max(xs.max() - xs.min(), ys.max() - ys.min())))
        rect = patches.Rectangle(
            (xs.min() - pad, ys.min() - pad),
            (xs.max() - xs.min()) + 2 * pad,
            (ys.max() - ys.min()) + 2 * pad,
            linewidth=2.0, edgecolor="tab:blue", facecolor="none", zorder=4
        )
        ax_probe.add_patch(rect)

        # --- widen the probe view with fixed padding ---
        ax_probe.set_aspect("equal", adjustable="box")
        x_min = float(locs[:, 0].min()) - padding
        x_max = float(locs[:, 0].max()) + padding
        y_min = float(locs[:, 1].min())
        y_max = float(locs[:, 1].max())
        mx = 0.06 * (x_max - x_min + 1e-6)
        my = 0.06 * (y_max - y_min + 1e-6)
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
        

def _probe_from_locs(locs, radius_um: float = 5.0):
    """Create a ProbeInterface Probe from (n_ch, 2) contact positions."""
    from probeinterface import Probe
    pr = Probe(ndim=2)
    pr.set_contacts(
        positions=np.asarray(locs, float),
        shapes="circle",
        shape_params={"radius": float(radius_um)},
    )
    try:
        pr.create_auto_shape()
    except Exception:
        pass
    return pr

def load_rate_npz(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    rate_hz = d["rate_hz"]           # (n_ch, n_bins_total)
    t_ms    = d["t_ms"]              # (n_bins_total,)
    meta    = d.get("meta", None)
    return rate_hz, t_ms, (meta.item() if hasattr(meta, "item") else meta)

def load_stim_ms_from_stimstream(stim_npz_path: Path) -> np.ndarray:
    """
    Load stimulation *block onset times* in ms from stim_stream.npz.
    Uses 'block_bounds_samples[:,0]' (Intan sample indices) and converts
    to milliseconds using meta['fs_hz'].
    """
    with np.load(stim_npz_path, allow_pickle=False) as z:
        if "block_bounds_samples" not in z or "meta" not in z:
            raise KeyError(f"{stim_npz_path} missing 'block_bounds_samples' or 'meta'.")

        blocks = z["block_bounds_samples"].astype(np.int64)  # (n_blocks, 2)
        if blocks.size == 0:
            raise ValueError(f"No block boundaries found in {stim_npz_path}.")

        # Parse meta
        meta_json = z["meta"].item() if hasattr(z["meta"], "item") else z["meta"]
        meta = json.loads(meta_json)
        fs_hz = float(meta.get("fs_hz"))
        if not np.isfinite(fs_hz):
            raise ValueError("meta['fs_hz'] missing or non-finite in stim_stream.npz")

        # Use block starts only
        onset_samples = blocks[:, 0]
        t_ms = onset_samples.astype(float) * (1000.0 / fs_hz)
        return np.unique(np.sort(t_ms))

def extract_peristim_segments(
    rate_hz: np.ndarray,
    t_ms: np.ndarray,
    stim_ms: np.ndarray,
    win_ms=(-800.0, 1200.0),
    min_trials: int = 1,
):
    """
    Return segments shape (n_trials, n_ch, n_twin) and rel_time_ms shape (n_twin,).
    Skips triggers whose window falls outside t_ms.
    """
    t_min, t_max = float(t_ms[0]), float(t_ms[-1])
    seg_len_ms = win_ms[1] - win_ms[0]

    # Relative timebase for a *perfectly* aligned segment (used only for plotting/logic)
    # We will slice on t_ms for each trial, so segment lengths are equal if t_ms is uniform.
    # Assume uniform binning for rates:
    dt = float(np.median(np.diff(t_ms)))  # ms per bin
    n_twin = int(round(seg_len_ms / dt))
    rel_time_ms = np.arange(n_twin) * dt + win_ms[0]

    segments = []
    kept = 0
    for s in np.asarray(stim_ms, dtype=float):
        start_ms = s + win_ms[0]
        end_ms   = s + win_ms[1]
        if start_ms < t_min or end_ms > t_max:
            continue  # skip partial windows

        # slice indices on t_ms
        i0 = int(np.searchsorted(t_ms, start_ms, side="left"))
        i1 = int(np.searchsorted(t_ms, end_ms,   side="left"))
        seg = rate_hz[:, i0:i1]  # (n_ch, n_twin)
        # Safety: ensure equal length (can happen if boundary falls between bins)
        if seg.shape[1] != n_twin:
            continue

        segments.append(seg)
        kept += 1

    if kept < min_trials:
        raise RuntimeError(f"Only {kept} peri-stim segments available (min_trials={min_trials}).")

    segments = np.stack(segments, axis=0)  # (n_trials, n_ch, n_twin)
    return segments, rel_time_ms

def baseline_zero_each_trial(
    segments: np.ndarray,
    rel_time_ms: np.ndarray,
    baseline_first_ms: float = 100.0,
):
    """
    For each trial & channel, subtract the mean over the first `baseline_first_ms`
    of the segment (i.e., from window start to start+baseline_first_ms).
    segments: (n_trials, n_ch, n_twin)
    """
    # mask for first 100 ms of the segment window (e.g., [-800, -700))
    t0 = rel_time_ms[0]
    mask = (rel_time_ms >= t0) & (rel_time_ms < t0 + baseline_first_ms)
    if mask.sum() < 1:
        raise ValueError("Baseline window has 0 bins — check your timebase and dt.")
    base = segments[:, :, mask].mean(axis=2, keepdims=True)  # (n_trials, n_ch, 1)
    return segments - base

def average_across_trials(zeroed_segments: np.ndarray):
    """
    Average across trials per channel.
    Input: (n_trials, n_ch, n_twin) -> returns (n_ch, n_twin)
    """
    return zeroed_segments.mean(axis=0)

def plot_channel_heatmap(
    avg_change: np.ndarray,           # (n_ch, n_twin)
    rel_time_ms: np.ndarray,          # (n_twin,)
    out_png: Path,
    n_trials: int,
    title: str = "Avg Δ in firing rate (baseline=first 100 ms)",
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
    intan_file: int | None = None,    # int is fine
    locs: np.ndarray | None = None,   # (n_ch, 2) positions for probe
    probe_ratio: float = 0.1,
    stim_idx: np.ndarray | None = None,  # optional highlight indices
    probe=None,
    padding: float = 15.0,
):
    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[5, probe_ratio], wspace=0.05)

    # --- heatmap (left) ---
    ax_hm = fig.add_subplot(gs[0, 0])
    n_ch = avg_change.shape[0]

    im = ax_hm.imshow(
        avg_change,
        aspect="auto",
        cmap=cmap,
        extent=[rel_time_ms[0], rel_time_ms[-1], 0, n_ch],  # normal (0 .. n_ch)
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    
    ax_hm.set_ylim(0, n_ch)
    cbar = fig.colorbar(im, ax=ax_hm); cbar.set_label("Δ Firing rate (Hz)")
    ax_hm.set_xlabel("Time (ms) rel. stim")
    ax_hm.set_ylabel("Channel index")
    suffix = f" (file {intan_file})" if intan_file is not None else ""
    ax_hm.set_title(f"{title} (n={n_trials} trials){suffix}")
    ax_hm.axvline(0.0, color="k", alpha=0.8, linewidth=1.2)
    ax_hm.axvspan(0.0, 100.0, color="gray", alpha=0.2, zorder=2)

    # --- probe (right), same way you do elsewhere ---
    ax_probe = fig.add_subplot(gs[0, 1])
    if probe is None and (locs is not None and locs.ndim == 2 and locs.shape[1] == 2):
        # fall back to building a simple probe from locs if caller didn’t pass one
        probe = _probe_from_locs(locs)

    if probe is not None:
        # if caller didn’t pass locs, get them from the probe
        if locs is None:
            locs = probe.contact_positions.astype(float)

        n_contacts = locs.shape[0]
        contacts_colors = np.array(["none"] * n_contacts, dtype=object)
        if stim_idx is not None and np.size(stim_idx):
            s = np.asarray(stim_idx, int)
            s = s[(s >= 0) & (s < n_contacts)]
            contacts_colors[s] = "tab:red"

        plot_probe(probe, ax=ax_probe, with_contact_id=False, contacts_colors=contacts_colors)

        # nice framing (same padding logic you use)
        ax_probe.set_aspect("equal", adjustable="box")
        x_min, x_max = float(locs[:, 0].min()-padding), float(locs[:, 0].max()+padding)
        y_min, y_max = float(locs[:, 1].min()), float(locs[:, 1].max())
        mx = 0.06 * (x_max - x_min + 1e-6)
        my = 0.06 * (y_max - y_min + 1e-6)
        ax_probe.set_xlim(x_min - mx, x_max + mx)
        ax_probe.set_ylim(y_min - my, y_max + my)
        ax_probe.set_xticks([]); ax_probe.set_yticks([])
        ax_probe.set_title("Probe layout", fontsize=10)
    else:
        ax_probe.axis("off")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap -> {out_png}")

def plot_debug_channel_traces(
    zeroed_segments: np.ndarray,   # (n_trials, n_ch, n_twin)
    rel_time_ms: np.ndarray,       # (n_twin,)
    ch: int,
    out_png_base: Path,
):
    n_trials, n_ch, n_twin = zeroed_segments.shape
    assert 0 <= ch < n_ch, f"debug channel {ch} out of range [0, {n_ch-1}]"

    y = zeroed_segments[:, ch, :]  # (n_trials, n_twin)

    # 1) overlay all trials + mean (x-axis in ms)
    plt.figure(figsize=(10, 5))
    for i in range(n_trials):
        plt.plot(rel_time_ms, y[i], alpha=0.25, linewidth=0.8)
    plt.plot(rel_time_ms, y.mean(axis=0), linewidth=2.0, label="mean")
    plt.axvline(0.0, color="k", alpha=0.6, linewidth=1.0)
    plt.xlabel("Time (ms) rel. stim")
    plt.ylabel("Δ Firing rate (Hz)")
    plt.title(f"Channel {ch}: peri-stim trials (n={n_trials})")
    plt.legend(loc="upper right")
    plt.tight_layout()
    out_png = out_png_base.with_name(out_png_base.stem + f"__ch{ch:03d}_overlay.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[debug] Saved overlay for ch {ch} -> {out_png}")

def run_one_Intan_FR_heatmap(
    npz_path: Path,
    out_dir: Path,
    win_ms=(-800.0, 1200.0),
    baseline_first_ms=100.0,
    min_trials=1,
    save_npz=True,
    stim_ms: np.ndarray | None = None,
    intan_file=None,
    geom_path: Path | None = None,
    stim_idx: np.ndarray | None = None,
    debug_chans: list[int] | None = None,                 # e.g. [0..15]
    debug_window_ms: tuple[float, float] | None = None,   # e.g. (0, 250)
):
    rate_hz, t_ms, _ = load_rate_npz(npz_path)
    stim_ms = np.asarray(stim_ms, dtype=float).ravel()

    segments, rel_time_ms = extract_peristim_segments(
        rate_hz=rate_hz, t_ms=t_ms, stim_ms=stim_ms, win_ms=win_ms, min_trials=min_trials
    )

    zeroed = baseline_zero_each_trial(
        segments=segments, rel_time_ms=rel_time_ms, baseline_first_ms=baseline_first_ms
    )
    
    # -------- NEW: multi-channel, windowed trial debug --------
    if debug_chans is not None and len(debug_chans):
        w0, w1 = (debug_window_ms if debug_window_ms is not None
                  else (rel_time_ms[0], rel_time_ms[-1]))
        mask = (rel_time_ms >= float(w0)) & (rel_time_ms <= float(w1))
        if not np.any(mask):
            print(f"[debug] requested window {w0}..{w1} ms has 0 bins — skipping.")
        else:
            z = zeroed[:, :, mask]
            rt = rel_time_ms[mask]
            out_base = (Path(out_dir) / (npz_path.stem + f"__debug_trials_{int(w0)}to{int(w1)}ms"))
            for ch in debug_chans:
                try:
                    plot_debug_channel_traces(z, rt, ch, out_base)
                except AssertionError as e:
                    print(f"[debug] skip ch {ch}: {e}")
    # -----------------------------------------------------------

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = npz_path.stem
    avg_change = average_across_trials(zeroed)
    n_trials = segments.shape[0]   # number of kept trials

    out_png = out_dir / f"{stem}__peri_stim_heatmap.png"
    
    probe, locs = build_probe_and_locs_from_geom(geom_path)
    
    plot_channel_heatmap(
        avg_change, rel_time_ms, out_png,
        n_trials=n_trials,
        vmin=-500, vmax=1000,
        intan_file=intan_file,
        locs=locs,
        probe=probe,
        probe_ratio=1.2,
        stim_idx=stim_idx,
        padding=15.0,
    )
    if save_npz:
        out_npz = out_dir / f"{stem}__peri_stim_avg.npz"
        np.savez_compressed(
            out_npz,
            avg_change=avg_change.astype(np.float32),
            rel_time_ms=rel_time_ms.astype(np.float32),
            meta=dict(
                source_file=str(npz_path),
                win_ms=win_ms,
                baseline_first_ms=baseline_first_ms,
                n_trials=int(segments.shape[0]),
            ),
        )
        print(f"Saved averaged peri-stim data -> {out_npz}")

def _extract_trial_windows_raw(rec, fs, stim_start_samples, ch, window_ms=(0.0, 250.0), n_show=4):
    w0_ms, w1_ms = float(window_ms[0]), float(window_ms[1])
    w0_s,  w1_s  = w0_ms / 1000.0, w1_ms / 1000.0
    n_total = rec.get_num_frames() if hasattr(rec, "get_num_frames") else rec.get_num_samples()

    starts, ends, s0_list = [], [], []   # <— keep the stim start for each chosen trial
    for s0 in np.asarray(stim_start_samples, dtype=np.int64):
        i0 = int(s0 + round(w0_s * fs))
        i1 = int(s0 + round(w1_s * fs))
        if 0 <= i0 < i1 <= n_total:
            starts.append(i0); ends.append(i1); s0_list.append(int(s0))
            if len(starts) >= n_show:
                break
    if not starts:
        raise RuntimeError("No trials fit into the requested window.")

    n_time = ends[0] - starts[0]
    t_ms = np.arange(n_time, dtype=float) / fs * 1000.0 + w0_ms

    traces = np.empty((len(starts), n_time), dtype=float)
    ch_id = rec.get_channel_ids()[ch]

    for k, (i0, i1) in enumerate(zip(starts, ends)):
        try:
            y = rec.get_traces(start_frame=i0, end_frame=i1, channel_ids=[ch_id], return_in_uV=True)
        except TypeError:
            y = rec.get_traces(start_frame=i0, end_frame=i1, channel_ids=[ch_id], return_scaled=True)
        traces[k, :] = np.asarray(y).squeeze()

    return traces, t_ms, np.asarray(s0_list, dtype=np.int64)   # <— also return s0 per trial

def plot_single_channel_trial_quad_raw(
    rec,
    fs: float,
    stim_start_samples: np.ndarray,
    ch: int,
    out_png: Path,
    window_ms=(90.0, 300.0),
    n_show: int = 4,
    title_prefix: str = "Stim-aligned raw traces",
    peak_ch: np.ndarray | None = None,
    peak_t_ms: np.ndarray | None = None,
):
    """
    2×2 panel: 4 trials of raw voltage for one channel, with optional spike overlays.
    """
        # ---- Normalize overlay inputs to 1-D arrays ----
    if peak_ch is not None:
        peak_ch = np.atleast_1d(np.asarray(peak_ch)).astype(int)
    if peak_t_ms is not None:
        peak_t_ms = np.atleast_1d(np.asarray(peak_t_ms)).astype(float)

    # Handle length mismatches / broadcasting-friendly behavior
    if peak_ch is not None and peak_t_ms is not None:
        if peak_ch.size != peak_t_ms.size:
            if peak_ch.size == 1:
                peak_ch = np.full_like(peak_t_ms, int(peak_ch[0]), dtype=int)
            elif peak_t_ms.size == 1:
                peak_t_ms = np.full_like(peak_ch, float(peak_t_ms[0]))
            else:
                n = min(peak_ch.size, peak_t_ms.size)
                peak_ch = peak_ch[:n]
                peak_t_ms = peak_t_ms[:n]
                
    Y, t_ms, s0_list = _extract_trial_windows_raw(
        rec, fs, stim_start_samples, ch, window_ms=window_ms, n_show=n_show
    )

    n_show = Y.shape[0]
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 6), sharex=True, sharey=True)

    # y-lims (you can keep your robust MAD logic if you prefer)
    ylim = (-100, 100)
    dt_ms = 1000.0 / fs
    w0_ms, w1_ms = float(window_ms[0]), float(window_ms[1])

    for k in range(nrows * ncols):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        if k < n_show:
            ax.plot(t_ms, Y[k], lw=1.0)
            # draw vline at 0 only if inside the window
            if w0_ms <= 0.0 <= w1_ms:
                ax.axvline(0.0, color="k", alpha=0.6, lw=0.8)
            ax.set_ylim(*ylim)
            ax.set_title(f"Trial {k}", fontsize=9)

            # --------- spike overlay (optional) ----------
            if (peak_ch is not None) and (peak_t_ms is not None) and (peak_ch.size > 0) and (peak_t_ms.size > 0):
                s0_ms = (float(s0_list[k]) / fs) * 1000.0
                m = (peak_ch == int(ch))
                t_abs = peak_t_ms[m]
                in_win = (t_abs >= s0_ms + w0_ms) & (t_abs <= s0_ms + w1_ms)
                if np.any(in_win):
                    x_ms = t_abs[in_win] - s0_ms
                    idx = np.clip(np.round((x_ms - w0_ms) / dt_ms).astype(int), 0, Y.shape[1] - 1)
                    ax.scatter(x_ms, Y[k, idx], s=12, zorder=3, alpha=0.9, color="red")


        else:
            ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (ms) rel. stim")
    for ax in axes[:, 0]:
        ax.set_ylabel("µV")

    fig.suptitle(f"{title_prefix} • ch {ch} • {w0_ms:.0f}–{w1_ms:.0f} ms", y=0.98)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

