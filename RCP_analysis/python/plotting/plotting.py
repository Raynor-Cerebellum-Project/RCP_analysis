from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
    
from probeinterface.plotting import plot_probe
from probeinterface import Probe
from mpl_toolkits.axes_grid1 import make_axes_locatable

def _prepare_cam(lines: np.ndarray | None,
                 rel_t: np.ndarray | None,
                 *,
                 min_trace_coverage: float = 0.05  # ≥5% finite samples in time
                 ) -> tuple[np.ndarray | None, bool]:
    """
    Returns (clean_lines, has_data).
    - Accepts (D,T) or (T,D); transposes if needed.
    - Drops traces that are all-NaN or have < min_trace_coverage finite values.
    - Ensures time dimension == rel_t.size.
    """
    if rel_t is None or lines is None:
        return None, False

    L = np.asarray(lines)
    if L.size == 0:
        return None, False

    # Ensure 2D
    if L.ndim == 1:
        # interpret as a single trace over time
        if rel_t is not None and L.size == rel_t.size:
            L = L[None, :]  # -> (1, T)
        else:
            return None, False

    # Coerce orientation so that axis-1 is time
    T = rel_t.size
    if L.shape[1] != T and L.shape[0] == T:
        L = L.T  # assume transposed input
    elif L.shape[1] != T:
        # neither axis matches time; can't plot
        return None, False

    # Drop traces with too few finite samples
    finite_frac = np.isfinite(L).mean(axis=1)
    keep = finite_frac >= float(min_trace_coverage)
    if not np.any(keep):
        return None, False
    L = L[keep, :]

    return L, True

# import helpers from sibling modules
from ..functions.utils import load_rate_npz, median_across_trials, extract_peristim_segments, build_probe_and_locs_from_geom, baseline_zero_each_trial

# ---- Plotting Intan only ----
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
    normalize_first_ms=100.0,
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
        segments=segments, rel_time_ms=rel_time_ms, normalize_first_ms=normalize_first_ms
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
    med_change = median_across_trials(zeroed)
    n_trials = segments.shape[0]   # number of kept trials

    out_png = out_dir / f"{stem}__peri_stim_heatmap.png"
    probe, locs = build_probe_and_locs_from_geom(geom_path)
    
    plot_channel_heatmap(
        med_change, rel_time_ms, out_png,
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
            avg_change=med_change.astype(np.float32),
            rel_time_ms=rel_time_ms.astype(np.float32),
            meta=dict(
                source_file=str(npz_path),
                win_ms=win_ms,
                normalize_first_ms=normalize_first_ms,
                n_trials=int(segments.shape[0]),
            ),
        )
        print(f"Saved median peri-stim data -> {out_npz}")

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

    return traces, t_ms, np.asarray(s0_list, dtype=np.int64)

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

def add_ua_region_bar(
    ax,
    n_rows: int,
    ua_chan_ids_1based: np.ndarray | None = None,  # length == n_rows; values in 1..256
    x: float = -0.06,   # axes coords (left of plot)
    width: float = 0.02
):
    """
    Draw a contiguous vertical bar subdivided into the 4 UA regions.
    If ua_chan_ids_1based is provided, segment heights are proportional
    to how many rows belong to each region; otherwise equal quarters.
    """
    regions = [("SMA", 1, 64),
               ("Dorsal premotor", 65, 128),
               ("M1 inferior", 129, 192),
               ("M1 superior", 193, 256)]

    if ua_chan_ids_1based is not None and len(ua_chan_ids_1based) == n_rows:
        ids = np.asarray(ua_chan_ids_1based, dtype=int)
        counts = [int(((ids >= lo) & (ids <= hi)).sum()) for _, lo, hi in regions]
        total = max(1, sum(counts))
        splits = [c / total for c in counts]
    else:
        splits = [0.25, 0.25, 0.25, 0.25]

    # Outer frame
    ax.add_patch(Rectangle((x, 0.0), width, 1.0,
                           transform=ax.transAxes, facecolor="none",
                           edgecolor="k", lw=1.2, clip_on=False))

    # Fill segments + labels
    y0 = 0.0
    for (label, _, _), frac in zip(regions, splits):
        y1 = y0 + frac
        ax.add_patch(Rectangle((x, y0), width, frac,
                               transform=ax.transAxes, facecolor="0.92",
                               edgecolor="none", clip_on=False))
        ax.text(x - 0.01, 0.5*(y0 + y1), label, transform=ax.transAxes,
                ha="right", va="center", fontsize=9)
        y0 = y1

    # 3 tick marks at the internal boundaries
    for y in np.cumsum(splits)[:-1]:
        ax.plot([x, x - 0.012], [y, y], transform=ax.transAxes,
                color="k", lw=1.2, clip_on=False)

# ---- Plotting FR for both ----
def _ua_region_from_elec(e: int) -> int:
    if e <= 0:      return -1
    if e <= 64:     return 0  # SMA
    if e <= 128:    return 1  # Dorsal premotor
    if e <= 192:    return 2  # M1 inferior
    return 3                  # M1 superior

def stacked_heatmaps_two_t(
    intan_med, ua_med, t_intan, t_ua, out_svg, title_top, title_bot,
    cmap="jet", vmin_intan=None, vmax_intan=None, vmin_ua=None, vmax_ua=None,
    probe=None, probe_locs=None, stim_idx=None,
    probe_title="Probe (Intan)", probe_width_ratio=0.35, probe_marker_size=28,
    ua_ids_1based: np.ndarray | None = None,
    ua_sort: str = "none",   # "none" | "elec" | "region_then_elec"
):

    have_probe = (probe is not None) or (
        probe_locs is not None and np.asarray(probe_locs).ndim == 2 and len(probe_locs) > 0
    )

    if have_probe:
        fig = plt.figure(figsize=(14, 8), constrained_layout=False)
        gs = gridspec.GridSpec(nrows=2, ncols=3, figure=fig,
                      width_ratios=[1.0, 0.0001, probe_width_ratio])
        ax_top   = fig.add_subplot(gs[0, 0])
        ax_bot   = fig.add_subplot(gs[1, 0], sharex=ax_top)
        ax_probe = fig.add_subplot(gs[:, 2])
    else:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=False)
        ax_probe = None

    # ----- Intan heatmap -----
    im0 = ax_top.imshow(
        intan_med, aspect="auto", cmap=cmap, origin="lower",
        extent=[t_intan[0], t_intan[-1], 0, intan_med.shape[0]],
        vmin=vmin_intan, vmax=vmax_intan
    )
    ax_top.axvline(0.0, color="k", alpha=0.8, linewidth=1.2)
    ax_top.axvspan(0.0, 100.0, color="gray", alpha=0.2)
    ax_top.set_title(title_top); ax_top.set_ylabel("Intan ch")
    fig.colorbar(im0, ax=ax_top).set_label("Δ FR (Hz)")

    # ----- UA heatmap -----
    # Reorder indices
    ua_plot = ua_med
    ids_plot = ua_ids_1based
    if ua_ids_1based is not None and ua_sort != "none":
        ids = np.asarray(ua_ids_1based, int)
        valid = ids > 0
        if ua_sort == "elec":
            order_valid = np.argsort(ids[valid], kind="stable")
        elif ua_sort == "region_then_elec":
            regs = np.array([_ua_region_from_elec(int(e)) for e in ids], int)
            order_valid = np.lexsort((ids[valid], regs[valid]))  # by (region, electrode)
        order = np.r_[np.where(valid)[0][order_valid], np.where(~valid)[0]]
        ua_plot = ua_med[order, :]
        ids_plot = ids[order]
        
    im1 = ax_bot.imshow(
        ua_plot, aspect="auto", cmap=cmap, origin="lower",
        extent=[t_ua[0], t_ua[-1], 0, ua_plot.shape[0]],
        vmin=vmin_ua, vmax=vmax_ua
    )
    ax_bot.axvline(0.0, color="k", alpha=0.8, linewidth=1.2)
    ax_bot.axvspan(0.0, 100.0, color="gray", alpha=0.2)
    ax_bot.set_title(title_bot)
    ax_bot.set_xlabel("Time (ms) rel. stim")

    # Hide numeric y labels/ticks for UA panel
    ax_bot.set_ylabel("")
    ax_bot.set_yticks([])
    ax_bot.tick_params(left=False, labelleft=False)

    # Contiguous region bar; if ua_ids_1based is None it will fall back to equal quarters
    add_ua_region_bar(ax_bot, ua_plot.shape[0], ua_chan_ids_1based=ids_plot)

    fig.colorbar(im1, ax=ax_bot).set_label("Δ FR (Hz) NOTE: SCALE")
    # ----- Probe inset: outline all contacts, fill ONLY stim sites -----
    if have_probe:
        if probe is None and Probe is not None:
            # build a simple probe from locs
            pr = Probe(ndim=2)
            pr.set_contacts(positions=np.asarray(probe_locs, float),
                            shapes="circle", shape_params={"radius": 5.0})
            try: pr.create_auto_shape()
            except Exception: pass
            probe = pr

        # derive locs if missing
        if probe_locs is None:
            try:
                probe_locs = probe.contact_positions.astype(float)
            except Exception:
                probe_locs = None

        n_contacts = (probe.get_contact_count()
                      if probe is not None else (0 if probe_locs is None else probe_locs.shape[0]))

        # default: no fill everywhere
        contacts_colors = ["none"] * n_contacts
        # highlight stim contacts in red
        if stim_idx is not None:
            s = np.atleast_1d(np.asarray(stim_idx, int))
            s = s[(s >= 0) & (s < n_contacts)]
            for i in s:
                contacts_colors[i] = "tab:red"

        if plot_probe is not None and probe is not None:
            plot_probe(
                probe, ax=ax_probe, with_contact_id=False,
                contacts_colors=contacts_colors,
                probe_shape_kwargs={"facecolor": "none", "edgecolor": "black", "linewidth": 0.6},
                contacts_kargs={"edgecolor": "k", "linewidth": 0.3},
            )
        else:
            # fallback to scatter if plot_probe not available
            locs = np.asarray(probe_locs)
            ax_probe.scatter(locs[:, 0], locs[:, 1], s=probe_marker_size,
                             facecolors="none", edgecolors="k", linewidths=0.3)
            if stim_idx is not None and locs.size:
                s = np.atleast_1d(np.asarray(stim_idx, int))
                s = s[(s >= 0) & (s < locs.shape[0])]
                ax_probe.scatter(locs[s, 0], locs[s, 1], s=probe_marker_size,
                                 c="tab:red", edgecolors="k", linewidths=0.3)

        ax_probe.set_title(probe_title, fontsize=10)
        ax_probe.set_aspect("equal", adjustable="box")
        ax_probe.set_xticks([]); ax_probe.set_yticks([])


    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, dpi=300, bbox_inches="tight"); plt.close(fig)

def stacked_heatmaps_plus_behv(
    intan_med: np.ndarray,              # (n_intan_ch, T_intan)
    ua_med: np.ndarray,                 # (n_ua_rows, T_ua)
    t_intan: np.ndarray,                # (T_intan,)
    t_ua: np.ndarray,                   # (T_ua,)
    out_svg: Path,
    title_kinematics: str,
    title_top: str,
    title_bot: str,
    *,
    cmap: str = "jet",
    vmin_intan: float | None = None,
    vmax_intan: float | None = None,
    vmin_ua: float | None = None,
    vmax_ua: float | None = None,
    probe=None,
    probe_locs=None,
    stim_idx=None,
    probe_title: str = "Probe (Intan)",
    probe_width_ratio: float = 0.35,
    probe_marker_size: int = 28,
    ua_ids_1based: np.ndarray | None = None,
    ua_sort: str = "region_then_elec",  # "none" | "elec" | "region_then_elec"
    # --- behavior (each optional) ---
    beh_rel_time: np.ndarray | None = None,   # (T_beh,)
    beh_cam0_lines: np.ndarray | None = None, # (6, T_beh) or empty
    beh_cam1_lines: np.ndarray | None = None, # (6, T_beh) or empty
    beh_labels: list[str] | None = None,      # length 6
    sess: str | None = None,
    overall_title: str | None = None,
    title_cam1: str | None = None,
):
    """
    Dynamic layout:
      - 2 cams available: 4 rows = [cam0, cam1, Intan, UA]
      - 1 cam available:  3 rows = [that cam, Intan, UA]
      - 0 cams:           2 rows = [Intan, UA]
    Right-side probe inset is included if `probe` or `probe_locs` are provided.
    """

    # Sanitize camera blocks first
    beh_cam0_lines, have_cam0 = _prepare_cam(beh_cam0_lines, beh_rel_time, min_trace_coverage=0.05)
    beh_cam1_lines, have_cam1 = _prepare_cam(beh_cam1_lines, beh_rel_time, min_trace_coverage=0.05)

    n_beh_rows = int(have_cam0) + int(have_cam1)
    # ---- figure & gridspec (with optional probe column) ----
    have_probe = (probe is not None) or (
        probe_locs is not None and np.asarray(probe_locs).ndim == 2 and len(probe_locs) > 0
    )

    # rows = n_beh_rows + 2 (Intan + UA)
    nrows = n_beh_rows + 2
    if nrows == 4:
        height_ratios = [1, 1, 1, 1]
    elif nrows == 3:
        height_ratios = [1, 1, 1]
    else:  # 2
        height_ratios = [1, 1]

    if have_probe:
        fig = plt.figure(figsize=(14, 8 + 2*n_beh_rows), layout='constrained')
        gs = gridspec.GridSpec(
            nrows=nrows, ncols=3, figure=fig,
            width_ratios=[1.0, 0.02, probe_width_ratio],
            height_ratios=height_ratios
        )
        _main_col = (slice(None), 0)
        ax_probe = fig.add_subplot(gs[:, 2])
    else:
        fig = plt.figure(figsize=(14, 8 + 2*n_beh_rows), constrained_layout=True)
        gs = gridspec.GridSpec(nrows=nrows, ncols=1, figure=fig, height_ratios=height_ratios)
        _main_col = (slice(None), 0)
        ax_probe = None

    # Allocate axes in order
    axes_beh = []
    next_row = 0
    if have_cam0:
        ax_b0 = fig.add_subplot(gs[next_row, 0])
        axes_beh.append(("cam0", ax_b0))
        next_row += 1
    else:
        ax_b0 = None

    if have_cam1:
        ax_b1 = fig.add_subplot(gs[next_row, 0])
        axes_beh.append(("cam1", ax_b1))
        next_row += 1
    else:
        ax_b1 = None

    ax_i  = fig.add_subplot(gs[next_row, 0]); next_row += 1
    ax_ua = fig.add_subplot(gs[next_row, 0])

    # ---------------- Behavior rows ----------------
    labs = beh_labels or []  # can be any length (0, 2, 6, 8, ...)

    def _plot_cam(ax, rel_t, lines, title, linestyle="-"):
        D = lines.shape[0]  # number of traces
        for k in range(D):
            y = lines[k]
            if np.isfinite(y).any():
                label = labs[k] if k < len(labs) else f"feat_{k+1}"
                ax.plot(rel_t, y, lw=1.25, ls=linestyle, label=label, alpha=0.95)
        ax.axvline(0.0, linewidth=1.2)
        ax.axvspan(0.0, 100.0, alpha=0.2)
        ax.set_ylabel("Δ (z)")
        ax.set_title(title)
        if D > 0:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=False, fontsize=8, ncols=1, borderaxespad=0.0)

    if have_cam0:
        _plot_cam(ax_b0, beh_rel_time, beh_cam0_lines, title_kinematics)
    if have_cam1:
        _plot_cam(ax_b1, beh_rel_time, beh_cam1_lines, title_cam1 or "Median Kinematics (Cam-1)")

    # ---------------- Intan heatmap ----------------
    im0 = ax_i.imshow(
        intan_med, aspect="auto", cmap=cmap, origin="lower",
        extent=[t_intan[0], t_intan[-1], 0, intan_med.shape[0]],
        vmin=vmin_intan, vmax=vmax_intan
    )
    ax_i.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2)
    ax_i.axvspan(-20.0, 120.0, color="gray", alpha=1)
    ax_i.set_title(title_top); ax_i.set_ylabel("Intan ch")
    cb0 = fig.colorbar(im0, ax=ax_i, fraction=0.046, pad=0.02)
    cb0.set_label("Δ FR (Hz)")

    # ---------------- UA heatmap (+ optional reordering) ----------------
    ua_plot = ua_med
    ids_plot = ua_ids_1based
    if ua_ids_1based is not None and ua_sort != "none":
        ids = np.asarray(ua_ids_1based, int)
        valid = ids > 0
        if ua_sort == "elec":
            order_valid = np.argsort(ids[valid], kind="stable")
        elif ua_sort == "region_then_elec":
            regs = np.array([_ua_region_from_elec(int(e)) for e in ids], int)
            order_valid = np.lexsort((ids[valid], regs[valid]))  # (region, elec)
        else:
            order_valid = np.arange(valid.sum())
        order = np.r_[np.where(valid)[0][order_valid], np.where(~valid)[0]]
        ua_plot = ua_med[order, :]
        ids_plot = ids[order]

    im1 = ax_ua.imshow(
        ua_plot, aspect="auto", cmap=cmap, origin="lower",
        extent=[t_ua[0], t_ua[-1], 0, ua_plot.shape[0]],
        vmin=vmin_ua, vmax=vmax_ua
    )
    ax_ua.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2)
    ax_ua.axvspan(-5.0, 105.0, color="gray", alpha=1)
    ax_ua.set_title(title_bot)
    ax_ua.set_xlabel("Time (ms) rel. stim")
    ax_ua.set_ylabel(""); ax_ua.set_yticks([]); ax_ua.tick_params(left=False, labelleft=False)

    add_ua_region_bar(ax_ua, ua_plot.shape[0], ua_chan_ids_1based=ids_plot)

    cb1 = fig.colorbar(im1, ax=ax_ua, fraction=0.046, pad=0.02)
    cb1.set_label("Δ FR (Hz) NOTE SCALE")

    # ---------------- Probe inset (optional) ----------------
    if have_probe:
        try:
            if probe is None and Probe is not None:
                pr = Probe(ndim=2)
                pr.set_contacts(positions=np.asarray(probe_locs, float),
                                shapes="circle", shape_params={"radius": 5.0})
                try:
                    pr.create_auto_shape()
                except Exception:
                    pass
                probe_local = pr
            else:
                probe_local = probe
        except Exception:
            probe_local = None

        if probe_locs is None and probe_local is not None:
            try:
                probe_locs = probe_local.contact_positions.astype(float)
            except Exception:
                probe_locs = None

        n_contacts = (probe_local.get_contact_count()
                      if probe_local is not None else (0 if probe_locs is None else probe_locs.shape[0]))
        contacts_colors = ["none"] * n_contacts
        if stim_idx is not None and np.size(stim_idx):
            s = np.asarray(stim_idx, int)
            s = s[(s >= 0) & (s < n_contacts)]
            for i in s:
                contacts_colors[i] = "tab:red"

        if (probe_local is not None) and (plot_probe is not None):
            plot_probe(
                probe_local, ax=ax_probe, with_contact_id=False,
                contacts_colors=contacts_colors,
                probe_shape_kwargs={"facecolor": "none", "edgecolor": "black", "linewidth": 0.6},
                contacts_kargs={"edgecolor": "k", "linewidth": 0.3},
            )
        else:
            if probe_locs is not None:
                L = np.asarray(probe_locs)
                ax_probe.scatter(L[:, 0], L[:, 1], s=probe_marker_size,
                                 facecolors="none", edgecolors="k", linewidths=0.3)
        ax_probe.set_title(probe_title, fontsize=10)
        ax_probe.set_aspect("equal", adjustable="box")
        ax_probe.set_xticks([]); ax_probe.set_yticks([])

    # ---------------- Sync x-lims across available axes ----------------
    x_ranges = [(t_intan[0], t_intan[-1]), (t_ua[0], t_ua[-1])]
    if have_cam0:
        x_ranges.append((beh_rel_time[0], beh_rel_time[-1]))
    if have_cam1:
        x_ranges.append((beh_rel_time[0], beh_rel_time[-1]))

    xmin = float(min(lo for lo, _ in x_ranges))
    xmax = float(max(hi for _, hi in x_ranges))

    for ax in [ax for ax in (ax_b0, ax_b1, ax_i, ax_ua) if ax is not None]:
        ax.set_xlim(xmin, xmax)

    # ---------------- Overall title ----------------
    if overall_title:
        fig.suptitle(overall_title, fontsize=13, fontweight="bold", y=1.02)

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


