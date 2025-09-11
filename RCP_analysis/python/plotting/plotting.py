from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# import helpers from sibling modules
from .data_loading import _load_intan_idx
from ..functions.params_loading import load_stim_geometry

def _probe_sidebar(ax, rec_sess, sel_indices=None, stim_ids=None, title="Probe"):
    # TODO: draw probe; for now keep it minimal
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    
def _pulse_onsets_from_stim_row(stim_row, eps=1e-12, min_gap=1):
    """
    Return indices where the stimulation goes from ~0 to non-zero (pulse onsets).
    eps: zero threshold. min_gap: debounce in samples (>=1).
    """
    x = np.asarray(stim_row, float).ravel()
    is_on = np.abs(x) > eps
    idx = np.flatnonzero((~is_on[:-1]) & (is_on[1:])) + 1
    if idx.size and min_gap > 1:
        keep = [idx[0]]
        for j in idx[1:]:
            if j - keep[-1] >= min_gap:
                keep.append(j)
        idx = np.asarray(keep, dtype=int)
    return idx

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
    
def quicklook_stim_grid(
    rec_sess, sess_folder: Path, out_dir: Path, fs: float,
    stim_num: int, nrows: int = 4, ncols: int = 4,
    pre_s: float = 0.015, post_s: float = 0.020, bar_ms: float = 20,
    dig_line=None, pick="top", sel_indices=None, name_suffix="",
    show_probe=True
):
    out_dir.mkdir(parents=True, exist_ok=True)
    sess_name = sess_folder.name
    bar = float(bar_ms) / 1000.0

    # ---- load stim_data (geometry-ordered) ----
    stim_geom_full = load_stim_geometry(sess_folder)
    if stim_geom_full is None:
        print(f"[{sess_name}] stim_data missing; skipping.")
        return

    # ---- OPTIONAL sanity check against Data.Intan_idx (if present) ----
    intan_idx_0 = _load_intan_idx(sess_folder)
    if intan_idx_0 is not None:
        n_rec  = int(rec_sess.get_num_frames())
        n_idx  = int(intan_idx_0.size)
        tstim  = int(stim_geom_full.shape[1])
        # 1) same number of indices as Intan frames
        assert n_idx == n_rec, (
            f"[{sess_name}] ERROR: Data.Intan_idx has {n_idx} samples "
            f"but Intan recording has {n_rec} frames."
        )
        # 2) all indices in-bounds for stim_data
        max_idx = int(np.max(intan_idx_0))
        min_idx = int(np.min(intan_idx_0))
        assert 0 <= min_idx and max_idx < tstim, (
            f"[{sess_name}] ERROR: Data.Intan_idx range [{min_idx}, {max_idx}] "
            f"exceeds stim_data length {tstim}."
        )

    # ---- choose first active stim row (as in your MATLAB) ----
    active_rows = np.flatnonzero(np.any(stim_geom_full != 0, axis=1))
    if active_rows.size == 0:
        print(f"[{sess_name}] no active stim rows; skipping.")
        return
    chosen_row = int(active_rows[0])
    stim_trace = stim_geom_full[chosen_row, :]

    # ---- onsets from stim_data only (first falling edge / 0->nonzero) ----
    onsets = _pulse_onsets_from_stim_row(stim_trace, eps=1e-12, min_gap=1)
    if onsets.size < stim_num:
        print(f"[{sess_name}] only {onsets.size} pulses in stim_data; need {stim_num}. Skipping.")
        return

    # pick the requested pulse as time-zero
    stim_idx = int(onsets[stim_num - 1])

    # ---- slice window centered on chosen stim ----
    s0 = max(0,                stim_idx - int(pre_s  * fs))
    s1 = min(int(rec_sess.get_num_frames()), stim_idx + int(post_s * fs))
    n  = s1 - s0
    if n <= 1:
        print(f"[{sess_name}] WARN: empty window; skipping.")
        return
    t_rel = (np.arange(n) + s0 - stim_idx) / fs

    # onsets in this window, relative to t=0
    ts = ((onsets[(onsets >= s0) & (onsets < s1)] - stim_idx) / fs).astype(float)
    ts.sort()

    # ----- GEOMETRY-ORDERED CHANNEL SELECTION -----
    ids = list(rec_sess.get_channel_ids())
    pos = rec_sess.get_probe().contact_positions  # (nch, 2): [:,1] = y
    order = np.lexsort((pos[:, 0], pos[:, 1]))    # y then x (ascending)

    k = nrows * ncols
    if sel_indices is None:
        if pick == "bottom":
            sel = order[-k:][::-1]
        elif pick == "center":
            y = pos[:, 1]
            near = np.argsort(np.abs(y - np.median(y)))[:k]
            sel = near[np.lexsort((pos[near, 0], y[near]))]
        else:
            sel = order[:k]  # 'top'
    else:
        sel = np.array(sel_indices, dtype=int)
        if sel.size != k:
            print(f"[{sess_name}] sel_indices has size {sel.size}, expected {k}. Skipping.")
            return

    ch_ids = [ids[i] for i in sel]
    traces = rec_sess.get_traces(start_frame=s0, end_frame=s1, channel_ids=ch_ids)

    stim_set_total = get_stimulated_channel_ids_from_geom(sess_folder, rec_sess)

    # ---------- figure layout: grid + optional probe sidebar ----------
    if show_probe:
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1], wspace=0.15)
        gs_grid = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs[0, 0], wspace=0.05, hspace=0.15)
        axes = np.array([[plt.subplot(gs_grid[r, c]) for c in range(ncols)] for r in range(nrows)])
        ax_probe = plt.subplot(gs[0, 1])
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10), sharex=True)
        ax_probe = None

    axes = np.atleast_2d(axes)

    # ----- plot the nrows×ncols traces -----
    for k_idx, cid in enumerate(ch_ids):
        r, c = divmod(k_idx, ncols)
        r = nrows - 1 - r   # flip vertical indexing
        ax = axes[r, c]
        ych = traces[:, k_idx]
        med = np.median(ych)
        mad = np.median(np.abs(ych - med)) + 1e-9

        ax.plot(t_rel, ych, lw=0.8)
        ax.axhline(med, lw=0.3, alpha=0.3)

        # dashed lines + shaded spans at stim onsets (from stim_data only)
        if ts.size:
            for t in ts:
                ax.axvline(t, color="tab:blue", linestyle="--", lw=0.6, alpha=0.6)

            gap_thresh = max(2.0 * bar, 0.010)
            diffs = np.diff(ts)
            cut = np.where(diffs > gap_thresh)[0]
            starts = np.r_[0, cut + 1]
            ends   = np.r_[cut, ts.size - 1]
            for a_i, b_i in zip(starts, ends):
                ax.axvspan(ts[a_i], ts[b_i] + bar, facecolor="tab:blue", alpha=0.10, zorder=-1)

        stim_tag = " (Stim)" if cid in stim_set_total else ""
        ax.set_title(f"{cid}{stim_tag}", fontsize=9)
        ax.set_ylim(med - 6 * mad, med + 6 * mad)
        if r < nrows - 1: ax.tick_params(labelbottom=False)
        if c > 0:         ax.tick_params(labelleft=False)

    axes[-1, 0].set_xlabel(f"Time relative to stim #{stim_num} (s)")
    axes[-1, 0].set_ylabel("µV")

    fig.suptitle(
        f"{sess_name} — {nrows}×{ncols} channels by geometry "
        f"(pre {pre_s:.3f}s / post {post_s:.3f}s){name_suffix}",
        y=0.995
    )

    if show_probe and ax_probe is not None:
        _probe_sidebar(ax_probe, rec_sess, sel_indices=sel, stim_ids=stim_set_total,
                       title="Probe (stim=red, viewed=blue box)")

    fig.tight_layout()
    suffix = f"_{name_suffix.strip('_')}" if name_suffix else ""
    out_png = out_dir / f"quicklook_grid_{sess_name}_stim{stim_num}_{nrows}x{ncols}{suffix}.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[{sess_name}] grid quicklook -> {out_png}")

def quicklook_stim_grid_all(
    rec_sess, sess_folder: Path, out_dir: Path, fs: float,
    stim_num: int, nrows: int = 4, ncols: int = 4,
    pre_s: float = 0.015, post_s: float = 0.020, bar_ms: float = 20,
    dig_line=None, stride: int | None = None, show_probe=True
):
    """Generate multiple 4x4 panels down the probe (top -> bottom).
    stride: how many channels to move between panels (default = nrows*ncols for no overlap).
            Use smaller (e.g., 8) for 50% overlap.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = list(rec_sess.get_channel_ids())
    pos = rec_sess.get_probe().contact_positions
    order = np.lexsort((pos[:, 0], pos[:, 1]))  # y then x
    group = nrows * ncols
    if stride is None:
        stride = group  # no overlap by default

    n = len(ids)
    panel_idx = 0
    for start in range(0, max(1, n - group + 1), stride):
        sel = order[start:start + group]
        if sel.size < group:
            break
        name_suffix = f"_p{panel_idx:02d}_ch{start:03d}-{start+group-1:03d}"
        quicklook_stim_grid(
            rec_sess=rec_sess, sess_folder=sess_folder, out_dir=out_dir, fs=fs,
            stim_num=stim_num, nrows=nrows, ncols=ncols,
            pre_s=pre_s, post_s=post_s, bar_ms=bar_ms, dig_line=dig_line,
            pick="top", sel_indices=sel, name_suffix=name_suffix,
            show_probe=show_probe,
        )
        panel_idx += 1