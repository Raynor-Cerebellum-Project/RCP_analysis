from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
import io, json, zipfile
from probeinterface.plotting import plot_probe

# import helpers from sibling modules
from ..functions.intan_preproc import load_stim_geometry, get_chanmap_perm_from_geom, reorder_recording_to_geometry, read_intan_recording, make_identity_probe_from_geom

def get_stimulated_channel_ids_from_geom(sess_folder: Path, rec_geom) -> set[str]:
    """
    Detect active stim rows from a GEOMETRY-ordered stim matrix,
    return the corresponding channel IDs from rec_geom (also geometry-ordered).
    """
    stim_geom = load_stim_geometry(sess_folder)
    if stim_geom is None or stim_geom.ndim != 2:
        return set()

    # any nonzero at any time -> stimulated row
    active_rows = np.flatnonzero(np.any(stim_geom != 0, axis=1)).astype(int)
    if active_rows.size == 0:
        return set()

    ids_geom = list(rec_geom.get_channel_ids())  # geometry order
    return {ids_geom[i] for i in active_rows if 0 <= i < len(ids_geom)}

def _first_stim_time_from_npz(stim_npz: Path) -> float | None:
    """
    Returns first stim time (seconds from start), or None if not found.
    Uses rising-edge detection per chunk (5th/95th percentile midpoint).
    """
    if not stim_npz.exists():
        return None
    with zipfile.ZipFile(str(stim_npz), "r") as zf:
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        fs = float(meta.get("fs_hz", 1.0))
        chunk_s = float(meta.get("chunk_seconds", 0.0)) or None
        chunk_files = sorted(n for n in zf.namelist() if n.startswith("chunk_") and n.endswith(".npy"))
        if not chunk_files:
            return None

        t_offset = 0.0
        for cf in chunk_files:
            X = np.load(io.BytesIO(zf.read(cf)))  # (frames, channels)
            # thresholds per channel
            p5  = np.nanpercentile(X, 5, axis=0)
            p95 = np.nanpercentile(X, 95, axis=0)
            thr = 0.5 * (p5 + p95)
            above = X > thr
            if X.shape[0] >= 2:
                rising = above[1:] & (~above[:-1])  # (frames-1, ch)
                idx = np.where(rising.any(axis=1))[0]
                if idx.size:
                    edge_idx = int(idx[0] + 1)
                    t_chunk = edge_idx / fs
                    return t_offset + t_chunk
            # advance offset
            if chunk_s is not None:
                t_offset += chunk_s
            else:
                t_offset += X.shape[0] / fs
    return None
def _detect_stim_channels_from_npz(
    stim_npz_path: Path,
    max_seconds: float = 2.0,   # scan up to this much data
    eps: float = 1e-12,         # near-zero threshold
    min_edges: int = 1,         # require at least this many 0→nonzero edges
) -> np.ndarray:
    """
    Return GEOMETRY-ORDERED indices of channels that actually received stim,
    detected from stim_npz_path by counting 0→nonzero rising edges.

    Uses per-channel mid-threshold: 0.5 * (p5 + p95) on each scanned chunk.
    Respects meta['order'] and perm_* mappings to emit GEOMETRY indices.
    """
    stim_npz_path = Path(stim_npz_path)
    if not stim_npz_path.exists():
        return np.array([], dtype=int)

    import io, json, zipfile
    with zipfile.ZipFile(str(stim_npz_path), "r") as zf:
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        fs = float(meta.get("fs_hz", 1.0))
        order = meta.get("order", "device")
        pg2d = meta.get("perm_geom_to_device")         # geometry -> device
        pd2g = meta.get("perm_device_to_geom")         # device   -> geometry
        if pd2g is None and pg2d is not None:
            # build inverse if only forward is present
            inv = np.argsort(np.asarray(pg2d, dtype=int))
            pd2g = inv.tolist()

        # which chunks to read?
        chunk_s = float(meta.get("chunk_seconds", 0.0)) or None
        chunks = sorted(n for n in zf.namelist() if n.startswith("chunk_") and n.endswith(".npy"))
        if not chunks:
            return np.array([], dtype=int)

        # limit how much we scan
        max_chunks = len(chunks)
        if chunk_s is not None and max_seconds is not None:
            max_chunks = max(1, int(np.ceil(max_seconds / chunk_s)))

        # accumulate rising-edge counts per channel
        # lazily sized after first chunk read
        rising_counts = None

        for cf in chunks[:max_chunks]:
            X = np.load(io.BytesIO(zf.read(cf)))  # (frames, channels)
            if X.ndim != 2 or X.shape[0] < 2:
                continue

            # per-channel threshold: midpoint between p5 and p95
            p5  = np.nanpercentile(X, 5, axis=0)
            p95 = np.nanpercentile(X, 95, axis=0)
            thr = 0.5 * (p5 + p95)

            above  = X > thr
            rising = above[1:] & (~above[:-1])    # (frames-1, ch)
            rc = rising.sum(axis=0)               # count per channel this chunk

            if rising_counts is None:
                rising_counts = rc.astype(np.int64, copy=True)
            else:
                rising_counts += rc.astype(np.int64)

        if rising_counts is None:
            return np.array([], dtype=int)

        # which columns are active in the data *as stored*?
        active_cols = np.where(rising_counts >= int(min_edges))[0].astype(int)

        # map to GEOMETRY index space
        if order == "geometry":
            geom_idx = active_cols
        elif order == "device" and pd2g is not None:
            pd2g = np.asarray(pd2g, dtype=int)
            geom_idx = pd2g[active_cols]
        else:
            # no mapping info → best effort: return as-is
            geom_idx = active_cols

        geom_idx = np.asarray(geom_idx, dtype=int)
        # keep only valid indices
        geom_idx = geom_idx[(geom_idx >= 0)]
        return np.unique(geom_idx)


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
    probe_ratio: float = 2.5,   # ← probe/sidebar width relative to grid (larger = wider probe)
):
    """
    Layout matches quicklook_stim_grid:
      - 1×2 figure: left = 4×4 traces, right = probe sidebar
      - Channels grouped by geometry order (y, then x)
      - Vertical flip of 4×4 grid (first channels at bottom row)
      - Probe sidebar shows all contacts, circles likely 'stim' channels, and boxes the 16 shown
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- geometry + reorder to geometry order (identity probe mapping) ---
    geom = load_stim_geometry(geom_path)
    perm = get_chanmap_perm_from_geom(geom)
    probe = make_identity_probe_from_geom(geom, radius_um=5.0)

    rec = read_intan_recording(sess_folder, stream_name=neural_stream)
    rec = reorder_recording_to_geometry(rec, perm)
    rec = rec.set_probe(probe, in_place=False)

    fs = float(rec.get_sampling_frequency())
    # SI versions differ: get_num_frames([seg]) vs get_num_samples()
    try:
        n_total = rec.get_num_frames()
    except TypeError:
        n_total = rec.get_num_frames(0) if hasattr(rec, "get_num_frames") else rec.get_num_samples()

    locs = rec.get_channel_locations()
    chan_ids = [str(x) for x in rec.get_channel_ids()]

    if stim_npz_path is not None:
        try:
            t0 = _first_stim_time_from_npz(stim_npz_path) or 0.0
        except Exception:
            print(f"[WARN] could not read stim_npz at {stim_npz_path}; centering at t=0")
            t0 = 0.0
    else:
        print("[WARN] stim_npz_path=None; centering at t=0 and skipping stim site coloring.")
        t0 = 0.0

    # window around stim
    start = max(0, int(round((t0 - pre_s) * fs)))
    end   = min(n_total, int(round((t0 + post_s) * fs)))
    if end <= start:
        end = min(n_total, start + int(round(max(0.3, pre_s + post_s) * fs)))

    Xwin = rec.get_traces(start_frame=start, end_frame=end, return_in_uV=True)
    t = (np.arange(Xwin.shape[0], dtype=float) + start) / fs - t0

    # --- actual stim site detection from NPZ (geometry indices) ---
    stim_idx = np.array([], dtype=int)
    if stim_npz_path is not None:
        try:
            stim_idx = _detect_stim_channels_from_npz(
                stim_npz_path, max_seconds=2.0, eps=1e-12, min_edges=1
            )
        except Exception as e:
            print(f"[WARN] stim site detection failed: {e}")
            stim_idx = np.array([], dtype=int)


    # --- group channels by geometry order (y then x) into 16s ---
    order = np.lexsort((locs[:, 0], locs[:, 1]))  # ascending y, then x
    nrows, ncols = 4, 4
    group = nrows * ncols
    groups = [order[i:i + group] for i in range(0, order.size, group)]
    # If the last group is short, we still render it (unused axes are blanked)

    for gi, sel in enumerate(groups):
        # --- figure: 1×2 (grid + probe) like quicklook, with adjustable probe width ---
        fig = plt.figure(figsize=(20, 10), constrained_layout=True)
        # left pane weight = 5, right/probe = probe_ratio (e.g., 2.5 → nice wide probe)
        gs_main = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[5, probe_ratio], wspace=0.15)
        gs_grid = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs_main[0, 0],
                                                   wspace=0.05, hspace=0.15)
        axes = np.array([[plt.subplot(gs_grid[r, c]) for c in range(ncols)] for r in range(nrows)])
        ax_probe = plt.subplot(gs_main[0, 1])

        # --- 4×4 traces, vertical flip like quicklook_stim_grid ---
        for k_idx in range(group):
            r0, c0 = divmod(k_idx, ncols)
            r = nrows - 1 - r0  # flip vertically
            ax = axes[r, c0]
            if k_idx >= sel.size:
                ax.axis("off")
                continue
            ch = int(sel[k_idx])
            y = Xwin[:, ch]
            med = np.median(y)
            mad = np.median(np.abs(y - med)) + 1e-9

            ax.plot(t, y, lw=0.8)
            ax.axhline(med, lw=0.3, alpha=0.3)
            ax.axvline(0.0, linestyle="--", linewidth=0.8)

            ax.set_title(chan_ids[ch], fontsize=9)
            ax.set_ylim(med - 6 * mad, med + 6 * mad)
            # ↓ labels only on the true bottom row/left column after flipping
            if r != nrows - 1:
                ax.tick_params(labelbottom=False)
            if c0 != 0:
                ax.tick_params(labelleft=False)

        axes[-1, 0].set_xlabel("Time (s) rel. first stim")
        axes[-1, 0].set_ylabel("µV")

        # --- probe sidebar using probeinterface (red = stim) ---
        n_contacts = probe.get_contact_count()
        contacts_colors = np.array(["none"] * n_contacts, dtype=object)
        if stim_idx.size:
            # clip in case of any stray indices
            stim_idx_clipped = stim_idx[(stim_idx >= 0) & (stim_idx < n_contacts)]
            contacts_colors[stim_idx_clipped] = "tab:red"

        plot_probe(
            probe,
            ax=ax_probe,
            with_contact_id=False,
            contacts_colors=contacts_colors,
        )

        # blue box around the 16 channels shown in the 4×4 grid
        xs, ys = locs[sel, 0], locs[sel, 1]
        pad = max(6.0, 0.03 * float(max(xs.max() - xs.min(), ys.max() - ys.min())))
        rect = patches.Rectangle(
            (xs.min() - pad, ys.min() - pad),
            (xs.max() - xs.min()) + 2 * pad,
            (ys.max() - ys.min()) + 2 * pad,
            linewidth=2.0, edgecolor="tab:blue", facecolor="none", zorder=4
        )
        ax_probe.add_patch(rect)

        # set aspect/limits AFTER plot_probe (it can reset them)
        ax_probe.set_aspect("equal", adjustable="box")
        x_min, x_max = float(locs[:, 0].min()), float(locs[:, 0].max())
        y_min, y_max = float(locs[:, 1].min()), float(locs[:, 1].max())
        mx, my = 0.06 * (x_max - x_min + 1e-6), 0.06 * (y_max - y_min + 1e-6)
        ax_probe.set_xlim(x_min - mx, x_max + mx)
        ax_probe.set_ylim(y_min - my, y_max + my)
        ax_probe.set_xticks([]); ax_probe.set_yticks([])

        n_stim = int(stim_idx.size)
        if hasattr(fig, "set_constrained_layout_pads"):
            # inches between elements + relative hspace/wspace
            fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.06, wspace=0.20, hspace=0.30)

        st = fig.suptitle(
            f"{sess_folder.name} — {neural_stream} (panel {gi:02d}) • stim sites: {n_stim}",
            y=1.02, fontsize=12
        )

        out_png = out_dir / f"{sess_folder.name}_probe+4x4_panel{gi:02d}.png"
        fig.savefig(out_png, dpi=180, bbox_inches="tight", bbox_extra_artists=[st])
        plt.close(fig)
