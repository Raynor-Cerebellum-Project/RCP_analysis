import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from pathlib import Path
import numpy.ma as ma
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm

from probeinterface.plotting import plot_probe
from probeinterface import Probe

# ---- knobs: UA vmin/vmax per group/label ----
UA_VRANGE_BY_LABEL = {
    "M1i": (-30, 125),
    "M1s": (-30, 125),
    "PMd": (-30, 150),
    "SMA": (-10, 40),
    "M1 (upper limb)": (-30, 125),
    "M1 (lower limb)": (-30, 125),
    "Premotor (dorsal)": (-30, 150),
    "UA (other)": (-50, 150),
}
UA_VRANGE_DEFAULT = (-50, 150)  # fallback

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
                ha="right", va="center", fontsize=12, fontweight="bold")
        y0 = y1

    # 3 tick marks at the internal boundaries
    for y in np.cumsum(splits)[:-1]:
        ax.plot([x, x - 0.012], [y, y], transform=ax.transAxes,
                color="k", lw=1.2, clip_on=False)


def _centers_to_edges(t, widths=None):
    t = np.asarray(t, float).ravel()

    if t.size == 0:
        return np.zeros(0, float)

    if t.size == 1:
        if widths is not None and np.size(widths):
            w_arr = np.asarray(widths, float).ravel()
            w = float(w_arr[0]) if w_arr.size else 1.0
        else:
            w = 1.0
        return np.array([t[0] - w / 2.0, t[0] + w / 2.0], float)

    edges = np.empty(t.size + 1, float)
    edges[1:-1] = 0.5 * (t[:-1] + t[1:])

    widths_ok = False
    if widths is not None and np.size(widths):
        w_arr = np.asarray(widths, float).ravel()
        if w_arr.size == t.size and np.all(np.isfinite(w_arr)):
            edges[0] = t[0] - w_arr[0] / 2.0
            edges[-1] = t[-1] + w_arr[-1] / 2.0
            widths_ok = True

    if not widths_ok:
        edges[0] = t[0] - (t[1] - t[0]) / 2.0
        edges[-1] = t[-1] + (t[-1] - t[-2]) / 2.0

    return edges


def _mask_time_window_by_edges(mat, x_edges, blank_ms):
    """
    Mask columns whose bin overlaps [blank_start, blank_end].
    mat: (n_ch, T)
    x_edges: (T+1,)
    blank_ms: tuple(start_ms, end_ms) or None
    """
    if blank_ms is None:
        return ma.masked_invalid(mat)

    b0, b1 = map(float, blank_ms)
    if b1 <= b0:
        return ma.masked_invalid(mat)

    mat_masked = ma.masked_invalid(mat)

    # bin overlaps blanking window if right edge > b0 and left edge < b1
    col_mask = (x_edges[1:] > b0) & (x_edges[:-1] < b1)

    if col_mask.size != mat_masked.shape[1]:
        raise ValueError(
            f"Blank mask mismatch: mat has {mat_masked.shape[1]} columns, "
            f"x_edges gives {col_mask.size} bins"
        )

    mat_masked[:, col_mask] = ma.masked
    return mat_masked


def sort_matrix_by_peak(
    mat: np.ndarray,
    rel_t: np.ndarray,
    search_window: tuple[float, float] | None = None,
    direction: str = "top_to_bottom",
    z_scored: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort rows (channels) of a 2D matrix (n_channels, n_time) by time of peak activity.

    Parameters
    ----------
    mat : np.ndarray
        (n_channels, n_time) 2D array.
    rel_t : np.ndarray
        (n_time,) time vector in ms.
    search_window : tuple[float, float] | None
        (t_min, t_max) in ms to search for peak, or None for full window.
    direction : str
        "top_to_bottom" (earliest peak at top of plot) or "bottom_to_top" (earliest peak at bottom).
    z_scored : bool
        Whether data is z-scored.

    Returns
    -------
    mat_sorted : np.ndarray
        Sorted matrix with same shape as input.
    order : np.ndarray
        1D permutation indices applied to rows.
    """
    if mat is None or mat.size == 0 or mat.ndim != 2:
        return mat, np.arange(0 if mat is None else mat.shape[0])

    n_ch, n_time = mat.shape
    if n_ch <= 1:
        return mat, np.arange(n_ch)

    if search_window is not None and rel_t is not None and len(rel_t) == n_time:
        w_min, w_max = search_window
        mask = (rel_t >= w_min) & (rel_t <= w_max)
        if np.any(mask):
            sub = mat[:, mask]
            t_sub = np.asarray(rel_t)[mask]
        else:
            sub = mat
            t_sub = np.asarray(rel_t)
    else:
        sub = mat
        t_sub = np.asarray(rel_t) if rel_t is not None and len(rel_t) == n_time else np.arange(n_time)

    # Replace NaNs/Infs with -inf so argmax selects the finite maximum
    sub_finite = np.where(np.isfinite(sub), sub, -np.inf)
    max_vals = np.max(sub_finite, axis=1)
    peak_idx = np.argmax(sub_finite, axis=1)
    peak_times = t_sub[peak_idx]

    # Responsive channel criteria: finite, non-zero peak, and not flat
    responsive = np.isfinite(max_vals) & (max_vals > 0.0) & (np.nanstd(sub, axis=1) > 1e-6)

    idx_resp = np.where(responsive)[0]
    idx_unresp = np.where(~responsive)[0]

    if len(idx_resp) > 0:
        order_resp = idx_resp[np.argsort(peak_times[idx_resp], kind="stable")]
    else:
        order_resp = np.array([], dtype=int)

    # In matplotlib pcolormesh, row 0 is drawn at the bottom (y=0) and row n_ch-1 at the top (y=n_ch).
    # If direction == "top_to_bottom":
    #   We want earliest peak at top (row n_ch-1) and latest towards bottom.
    #   Unresponsive channels go at the very bottom (rows 0, 1, ...).
    #   order: [idx_unresp, order_resp[::-1]]
    # If direction == "bottom_to_top":
    #   We want earliest peak at bottom (row 0) and latest towards top.
    #   order: [order_resp, idx_unresp]
    if direction == "top_to_bottom":
        order = np.concatenate([idx_unresp, order_resp[::-1]])
    else:
        order = np.concatenate([order_resp, idx_unresp])

    return mat[order, :], order


# ---- Plotting FR for both ----
def stacked_heatmaps_plus_behv(
    nprw_med, ua_med, t_nprw, t_ua, nprw_edges_ms, ua_edges_ms, out_svg,
    title_kinematics, title_NA,
    *, cmap="jet",
    cmap_nprw=None, cmap_ua=None,
    norm_nprw=None, norm_ua=None,
    center_zero: bool = False,
    vmin_nprw=None, vmax_nprw=None,
    vmin_ua=None, vmax_ua=None,
    probe=None, probe_locs=None, stim_idx=None,
    probe_title="Probe (NPRW)", probe_width_ratio=0.15, probe_marker_size=28,
    probe_gap_ratio=0.04,
    ua_ids_1based=None, ua_sort="region_then_elec",
    beh_rel_time=None, beh_cam0_pos=None, beh_cam1_pos=None,
    beh_labels=None, beh_cam0_vel=None, beh_cam1_vel=None,
    beh_pos_ylabel="Position Δ (z)",
    beh_variability=None,
    beh_variability_stds=None,
    title_variability=None,
    beh_variability_ylabel="Pairwise\nDistance",
    beh_variability_ylim=None,
    # Optional per-timepoint STD arrays for shaded plotting of position traces
    beh_cam0_pos_stds=None, beh_cam1_pos_stds=None,
    beh_cam0_vel_stds=None, beh_cam1_vel_stds=None,
    target_pos_cam0=None, 
    target_pos_cam1=None, 
    title_cam1=None, title_cam0_vel=None, title_cam1_vel=None,
    sess="",
    overall_title="",
    beh_ylim=None,
    # New Layout Knobs (defaults match original constants)
    beh_ratio: float = 0.6,
    ch_ratio_per_row: float = 0.015,
    min_heatmap_ratio: float = 0.6,
    ua_compact_factor: float = 0.95,
    nprw_scale: float = 0.6,
    gap_beh_nprw: float = 0.22, # originally GAP_HEIGHT
    fig_width_in: float = 16.0,
    height_per_ratio_in: float = 4.0,
    cb_label_nprw: str = "Δ FR (Hz)",
    cb_label_ua: str = "Δ FR (Hz)",
    nprw_blank_ms=None,
    ua_blank_ms=None,
    stim_dur_ms: float | None = None,
    nprw_sort: str = "none",
    peak_order_direction: str = "top_to_bottom",
    peak_search_window_ms: tuple[float, float] | None = None,
    nprw_show_y_ticks: bool = False,
    beh_show_box: bool = False,
    beh_show_y_axis: bool = False,
    beh_line_width: float = 2.4,
    beh_scale_bar_ms: float | None = 200.0,
    beh_in_plot_label: str | None = "Position",
    xlim: tuple[float, float] | None = None,
):
    """
    s: session ID
    """
    # ---------------- Sanitize presence ----------------
    has_nprw = isinstance(nprw_med, np.ndarray) and nprw_med.ndim == 2 and nprw_med.size > 0 and (t_nprw is not None)
    has_ua    = isinstance(ua_med,    np.ndarray) and ua_med.ndim    == 2 and ua_med.size    > 0 and (t_ua    is not None)

    # ---------- Behavior prep (make rows + flags) ----------
    def _prepare_cam(lines, rel_t, min_trace_coverage=0.05):
        if lines is None or rel_t is None:
            return None, False
        arr = np.asarray(lines, float)
        if arr.ndim == 1:
            arr = arr[None, :]
        ok = (arr.size > 0) and (np.asarray(rel_t).size == arr.shape[1])
        if not ok:
            return None, False
        finite_frac = np.isfinite(arr).mean()
        return (arr if finite_frac >= min_trace_coverage else None), (finite_frac >= min_trace_coverage)

    labs = (beh_labels or [])

    beh_variability, have_variability = _prepare_cam(beh_variability, beh_rel_time)
    beh_cam0_pos, have_cam0_pos = _prepare_cam(beh_cam0_pos, beh_rel_time)
    beh_cam1_pos, have_cam1_pos = _prepare_cam(beh_cam1_pos, beh_rel_time)
    beh_cam0_vel, have_cam0_vel = _prepare_cam(beh_cam0_vel, beh_rel_time)
    beh_cam1_vel, have_cam1_vel = _prepare_cam(beh_cam1_vel, beh_rel_time)

    beh_rows = []
    if have_variability: beh_rows.append(("beh", "variability"))
    if have_cam0_pos: beh_rows.append(("beh", "cam0_pos"))
    if have_cam1_pos: beh_rows.append(("beh", "cam1_pos"))
    if have_cam0_vel: beh_rows.append(("beh", "cam0_vel"))
    if have_cam1_vel: beh_rows.append(("beh", "cam1_vel"))

    rowspec = list(beh_rows)  # [("beh","cam0_pos"), ("beh","cam1_pos"), ...]
    if (gap_beh_nprw > 0.0) and (has_nprw or has_ua) and len(beh_rows):
        rowspec.append(("gap", None))   # <-- spacer row comes after behavior

    # ---------- UA grouping by regions (handle None safely) ----------
    ua_groups = []
    ua_plot   = ua_med if has_ua else None
    ids_plot  = ua_ids_1based if has_ua and (ua_ids_1based is not None) else None

    def _ua_region_code(elec) -> int:
        try:
            e = int(elec)
        except (TypeError, ValueError):
            return 1_000_000  # unknown / bad value
        if e <= 0:
            return 1_000_000  # invalid / non-positive
        if e <= 64:
            return 0  # SMA
        if e <= 128:
            return 1  # Dorsal premotor
        if e <= 192:
            return 2  # M1 inferior
        return 3      # M1 superior

    if has_ua:
        if ids_plot is not None:
            ids_arr = np.asarray(ids_plot, float)

            # define valid once: positive and finite
            valid = np.isfinite(ids_arr) & (ids_arr > 0)

            # compute regions once, with a large sentinel for bad IDs
            regs = np.array(
                [_ua_region_code(int(e)) if np.isfinite(e) and (e > 0) else 1_000_000
                for e in ids_arr],
                int,
            )

            if ua_sort != "none":
                if ua_sort == "elec":
                    order_valid = np.argsort(ids_arr[valid], kind="stable")

                elif ua_sort == "region_then_elec":
                    # sort by region, then electrode id
                    order_valid = np.lexsort((ids_arr[valid], regs[valid]))

                elif ua_sort == "peak":
                    # sort all UA electrodes globally by peak latency
                    sub_data = ua_plot[valid, :]
                    _, order_valid = sort_matrix_by_peak(
                        sub_data,
                        t_ua,
                        search_window=peak_search_window_ms,
                        direction=peak_order_direction,
                        z_scored=center_zero,
                    )

                else:
                    order_valid = np.arange(valid.sum())

                order = np.r_[np.where(valid)[0][order_valid], np.where(~valid)[0]]
                ua_plot = ua_plot[order, :]
                ids_arr = ids_arr[order]
                regs = regs[order]

            # now group using the same regs
            ua_groups = []

            def _append(mask, code, label, display_name):
                if mask.sum():
                    ua_groups.append({
                        "mat": ua_plot[mask, :],
                        "ids": ids_arr[mask],
                        "code": code,
                        "label": label,
                        "display_name": display_name,
                    })

            _append((regs == 2), "M1i", "M1 (upper limb)", "M1\n(upper limb)")
            _append((regs == 3), "M1s", "M1 (lower limb)", "M1\n(lower limb)")
            _append((regs == 1), "PMd", "Premotor (dorsal)", "Premotor\n(dorsal)")
            _append((regs == 0), "SMA", "SMA", "SMA")

            other_mask = (regs >= 1_000_000)
            if other_mask.any():
                _append(other_mask, "UA_other", "UA (other)", "UA (other)")

            if ua_sort == "region_then_peak":
                for group in ua_groups:
                    mat_g = group["mat"]
                    if mat_g is not None and mat_g.size > 0:
                        mat_sorted, ord_g = sort_matrix_by_peak(
                            mat_g,
                            t_ua,
                            search_window=peak_search_window_ms,
                            direction=peak_order_direction,
                            z_scored=center_zero,
                        )
                        group["mat"] = mat_sorted
                        if group["ids"] is not None:
                            group["ids"] = group["ids"][ord_g]

        else:
            ua_groups = [{"mat": ua_plot, "ids": None, "code": "UA", "label": "UA (all)", "display_name": "UA (all)"}]


    # ---------- Figure sizing ----------
    n_nprw_rows = 1 if has_nprw else 0
    n_ua_rows    = len(ua_groups) if has_ua else 0
    has_any_beh  = len(beh_rows) > 0

    # If literally nothing to draw, exit early
    if (not has_any_beh) and (n_nprw_rows == 0) and (n_ua_rows == 0):
        print(f"[warn] {sess or ''}: nothing to render (no behavior, no NPRW, no UA).")
        return

    height_ratios = []

    # behavior rows
    for _ in beh_rows:
        height_ratios.append(beh_ratio)

    # optional gap row
    if ("gap", None) in rowspec:
        height_ratios.append(gap_beh_nprw)

    # nprw row
    if has_nprw:
        nprw_rows = nprw_med.shape[0]
        nprw_ratio = max(min_heatmap_ratio, ch_ratio_per_row * nprw_rows) * float(nprw_scale)
        height_ratios.append(nprw_ratio)

    # UA rows
    if has_ua:
        for group in ua_groups:
            rows = group["mat"].shape[0] if group["mat"] is not None else 0
            base = max(min_heatmap_ratio, ch_ratio_per_row * rows)
            # PMd, SMA, and M1s (lower limb) are reduced in size relative to M1i (upper limb)
            grp_key = (group.get("code", "") + " " + group.get("label", "")).lower()
            scale = 0.5 if any(k in grp_key for k in ("pmd", "premotor", "sma", "m1s", "lower")) else 1.0
            height_ratios.append(base * scale * ua_compact_factor)

    have_probe = (probe is not None) or (
        probe_locs is not None and np.asarray(probe_locs).ndim == 2 and len(probe_locs) > 0
    )

    total_ratio = sum(height_ratios)
    fig_height  = height_per_ratio_in * total_ratio

    if have_probe:
        fig_width = fig_width_in * (1.0 + probe_gap_ratio + probe_width_ratio)
    else:
        fig_width = fig_width_in

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)

    if have_probe:
        # Outer grid: [main stack | skinny spacer | probe]
        outer = gridspec.GridSpec(
            nrows=1, ncols=3, figure=fig,
            width_ratios=[1.0, probe_gap_ratio, probe_width_ratio],
            wspace=0.05
        )

        # Inner grid lives entirely in the left (main) column; carries height_ratios
        gs = gridspec.GridSpecFromSubplotSpec(
            nrows=len(height_ratios), ncols=2, subplot_spec=outer[0],
            width_ratios=[1.0, 0.02],  # plot + colorbar
            height_ratios=height_ratios, hspace=0.18, wspace=0.05
        )

        # Spacer column (middle) is a blank axis
        ax_gap = fig.add_subplot(outer[1])
        ax_gap.axis("off")

        # Probe axis on the right
        ax_probe = fig.add_subplot(outer[2])
        ax_probe.set_xticks([]); ax_probe.set_yticks([])
        for sp in ax_probe.spines.values():
            sp.set_visible(False)
    else:
        gs = gridspec.GridSpec(
            nrows=len(height_ratios), ncols=2, figure=fig,
            width_ratios=[1.0, 0.02],
            height_ratios=height_ratios, hspace=0.18, wspace=0.05
        )
        ax_probe = None



    # ---------- Render behavior rows ----------
    def _ylabel_horizontal(ax, text: str, pad: float = 12.0, fontsize: float = 14, fontweight: str = "bold"):
        t = ax.set_ylabel(text, rotation=0, labelpad=pad, fontsize=fontsize, fontweight=fontweight)
        t.set_horizontalalignment("right")
        t.set_verticalalignment("center")
        return t

    def _plot_lines(ax, rel_t, lines, title, ylabel, sub, place_legend: bool, stds=None, ylim=None, target=None, is_last_beh: bool = False):
        """
        Plot lines (D x T). If `stds` is provided and has same shape as `lines`,
        draw a filled band ±std behind each mean line.
        """
        if lines is None:
            ax.axis("off")
            return
        D = lines.shape[0]
        if ylim is not None:
            ax.set_ylim(ylim)
        elif sub == "variability":
            finite_lines = lines[np.isfinite(lines)]
            if finite_lines.size > 0:
                max_val = float(np.nanmax(lines))
                if stds is not None:
                    try:
                        s_arr = np.asarray(stds, float)
                        if s_arr.size > 0 and np.isfinite(s_arr).any():
                            max_val = max(max_val, float(np.nanmax(lines + s_arr)))
                    except Exception:
                        pass
                ax.set_ylim(bottom=0.0, top=max(1.0, max_val * 1.15))
            else:
                ax.set_ylim(bottom=0.0)

        # Remove box outline (spines) if beh_show_box is False
        if not beh_show_box:
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.grid(False)

        for i in range(D):
            y = lines[i]
            if not np.isfinite(y).any():
                continue
            if sub == "variability":
                lab = ""
            else:
                lab = labs[i] if i < len(labs) else f"trace_{i+1}"
            # plot line with thicker linewidth
            ln, = ax.plot(rel_t, y, lw=beh_line_width, alpha=0.95, label=lab)
            # if stds available and matching shape, draw shaded band
            if stds is not None:
                try:
                    s = stds[i]
                    col = ln.get_color()
                    if isinstance(s, (tuple, list)) and len(s) == 2:
                        lo, hi = np.asarray(s[0], float), np.asarray(s[1], float)
                    elif isinstance(s, np.ndarray) and s.ndim == 2 and s.shape[0] == 2:
                        lo, hi = s[0], s[1]
                    elif np.shape(s) == np.shape(y):
                        lo = y - s
                        hi = y + s
                    else:
                        lo, hi = None, None
                    if lo is not None and hi is not None:
                        if sub == "variability":
                            lo = np.maximum(0.0, lo)
                        ax.fill_between(rel_t, lo, hi, color=col, alpha=0.22, linewidth=0.0)
                except Exception:
                    pass

        if beh_show_box and sub in ("cam0_pos", "cam1_pos") and "normalized" in str(ylabel).lower():
            ax.axhline(0.0, color="0.5", linestyle=":", alpha=0.5, linewidth=0.8)
            ax.axhline(1.0, color="0.5", linestyle=":", alpha=0.5, linewidth=0.8)

        ax.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2, ls="--")
        if stim_dur_ms is not None and np.isfinite(stim_dur_ms) and stim_dur_ms > 0:
            ax.axvspan(
                0.0,
                float(stim_dur_ms),
                color="#383838",
                alpha=0.35,
                zorder=0,
            )

        ax._is_time_axis = True

        # Y-axis ticks and labels
        if not beh_show_y_axis:
            ax.set_yticks([])
            ax.tick_params(left=False, labelleft=False)
            ax.set_ylabel("")
        else:
            ax.tick_params(axis="y", labelsize=12, width=1.4)
            _ylabel_horizontal(ax, ylabel)

        # Draw time scale bar replacing x-axis on the last behavior row
        t_arr = np.asarray(rel_t, float) if rel_t is not None else np.array([0.0])
        t_min = float(xlim[0]) if (xlim is not None) else (float(np.nanmin(t_arr)) if t_arr.size else 0.0)
        x_start = t_min + 40.0
        scale_val = float(beh_scale_bar_ms or 200.0)
        x_end = x_start + scale_val
        if x_end > -20.0:
            x_start = -20.0 - scale_val
            x_end = -20.0

        if beh_scale_bar_ms is not None and is_last_beh:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.spines["bottom"].set_visible(False)
            y_bar = -0.06
            ax.plot(
                [x_start, x_end], [y_bar, y_bar],
                transform=ax.get_xaxis_transform(),
                color="black", lw=3.0, clip_on=False, solid_capstyle="butt"
            )
            ax.text(
                (x_start + x_end) / 2.0, y_bar - 0.09,
                f"{int(scale_val)} ms",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=14, fontweight="bold",
                color="black", clip_on=False
            )
        elif beh_scale_bar_ms is not None:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.spines["bottom"].set_visible(False)

        if beh_in_plot_label:
            in_label = "Variability" if sub == "variability" else beh_in_plot_label
            ax.text(
                x_start, 0.85,
                in_label,
                transform=ax.get_xaxis_transform(),
                fontsize=18, fontweight="bold",
                color="#222222", ha="left", va="center", clip_on=False
            )

        if title and not beh_in_plot_label:
            ax.set_title(title, fontsize=15, fontweight="bold")
        if place_legend and D:
            handles, labels = ax.get_legend_handles_labels()
            valid = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")]

            if valid:
                handles, labels = zip(*valid)
                ax.legend(handles, labels,
                        loc="center left",
                        bbox_to_anchor=(1.02, 0.5),
                        fontsize=12,
                        frameon=False)
        # ax.grid(alpha=0.15, linestyle=":")

    row = 0
    # before the loop
    total_beh_rows = sum(1 for k, _ in rowspec if k == "beh")
    legend_row_idx = min(3, total_beh_rows)  # place legend on 3rd beh row, or last if <3
    beh_row_i = 0

    legend_placed = False
    for kind, sub in rowspec:
        if kind == "beh":
            beh_row_i += 1
            ax  = fig.add_subplot(gs[row, 0])
            cax = fig.add_subplot(gs[row, 1]); cax.axis("off")

            # Only place legend on the chosen behavior row
            place_legend_now = (beh_row_i == legend_row_idx) and (not legend_placed)
            is_last_beh = (beh_row_i == total_beh_rows)

            if sub == "variability":
                var_title = title_variability if title_variability is not None else (title_kinematics or "")
                _plot_lines(ax, beh_rel_time, beh_variability, var_title,
                            beh_variability_ylabel, sub, place_legend_now, stds=beh_variability_stds, ylim=beh_variability_ylim, target=None, is_last_beh=is_last_beh)
            elif sub == "cam0_pos":
                c0_title = "" if have_variability else (title_kinematics or "")
                _plot_lines(ax, beh_rel_time, beh_cam0_pos, c0_title,
                            f"Cam-0\n{beh_pos_ylabel}", sub, place_legend_now, stds=beh_cam0_pos_stds, ylim=beh_ylim, target=target_pos_cam0, is_last_beh=is_last_beh)
            elif sub == "cam1_pos":
                c1_title = "" if (have_variability or have_cam0_pos) else (title_kinematics or "")
                if title_cam1:
                    c1_title = title_cam1
                _plot_lines(ax, beh_rel_time, beh_cam1_pos, c1_title,
                            f"Cam-1\n{beh_pos_ylabel}", sub, place_legend_now, stds=beh_cam1_pos_stds, ylim=beh_ylim, target=target_pos_cam1, is_last_beh=is_last_beh)
            elif sub == "cam0_vel":
                _plot_lines(ax, beh_rel_time, beh_cam0_vel, title_cam0_vel or "",
                            "Cam-0\nVelocity (z/ms)", sub, place_legend_now, stds=beh_cam0_vel_stds, ylim = tuple(np.array(beh_ylim) / 200.0), is_last_beh=is_last_beh)
            elif sub == "cam1_vel":
                _plot_lines(ax, beh_rel_time, beh_cam1_vel, title_cam1_vel or "",
                            "Cam-1\nVelocity (z/ms)", sub, place_legend_now, stds=beh_cam1_vel_stds, ylim = tuple(np.array(beh_ylim) / 200.0), is_last_beh=is_last_beh)

            if place_legend_now:
                legend_placed = True

            # hide x labels if another behavior row follows or if scale bar is active
            if beh_row_i < total_beh_rows or beh_scale_bar_ms is not None:
                ax.set_xlabel(""); ax.tick_params(axis="x", labelbottom=False)
            row += 1
            continue

        if kind == "gap":
            fig.add_subplot(gs[row, 0]).axis("off")
            fig.add_subplot(gs[row, 1]).axis("off")
            row += 1
            continue


    # ---------- NPRW heatmap (optional) ----------
    if has_nprw:
        ax_nprw     = fig.add_subplot(gs[row, 0])
        ax_nprw_cax = fig.add_subplot(gs[row, 1])

        nprw_plot_mat = nprw_med
        if nprw_sort == "peak":
            mat_for_sort = np.copy(nprw_med)
            if nprw_blank_ms is not None and t_nprw is not None and len(t_nprw) == mat_for_sort.shape[1]:
                b0, b1 = float(nprw_blank_ms[0]), float(nprw_blank_ms[1])
                if b1 > b0:
                    mat_for_sort[:, (t_nprw >= b0) & (t_nprw <= b1)] = np.nan
            _, order_nprw = sort_matrix_by_peak(
                mat_for_sort,
                t_nprw,
                search_window=peak_search_window_ms,
                direction=peak_order_direction,
                z_scored=center_zero,
            )
            nprw_plot_mat = nprw_med[order_nprw, :]

        # Mask NaNs
        nprw_x_edges = _centers_to_edges(t_nprw, nprw_edges_ms)

        nprw_masked = _mask_time_window_by_edges(
            nprw_plot_mat,
            nprw_x_edges,
            nprw_blank_ms,
        )

        # Make a copy of the cmap and set NaN color to gray
        cmap_choice_nprw = cmap_nprw if cmap_nprw is not None else cmap
        try:
            cmap_local_nprw = plt.get_cmap(cmap_choice_nprw).copy()
        except Exception:
            cmap_local_nprw = cm.get_cmap(cmap_choice_nprw).copy()
        cmap_local_nprw.set_bad(color="#383838")

        # Determine normalization for NPRW
        norm_local_nprw = norm_nprw
        if norm_local_nprw is None and center_zero and (vmin_nprw is not None) and (vmax_nprw is not None):
            if vmin_nprw < 0.0 < vmax_nprw:
                norm_local_nprw = TwoSlopeNorm(vcenter=0.0, vmin=float(vmin_nprw), vmax=float(vmax_nprw))
        
        # nprw_masked: (n_ch, T)
        n_ch, T = nprw_masked.shape

        # 4) Build channel edges (0..n_ch)
        y_edges = np.arange(n_ch + 1)

        # 5) Replace imshow with pcolormesh
        nprw_x_edges = _centers_to_edges(t_nprw, nprw_edges_ms)

        if nprw_masked.shape[1] != len(nprw_x_edges) - 1:
            raise ValueError(
                f"NPRW pcolormesh mismatch: C={nprw_masked.shape}, "
                f"len(t_nprw)={len(t_nprw)}, len(x_edges)={len(nprw_x_edges)}, "
                f"len(nprw_edges_ms)={len(nprw_edges_ms)}"
            )

        pcolor_kwargs_nprw = dict(
            cmap=cmap_local_nprw,
            shading="flat",
            antialiased=False,
            rasterized=True,
        )
        if norm_local_nprw is not None:
            pcolor_kwargs_nprw["norm"] = norm_local_nprw
        else:
            pcolor_kwargs_nprw["vmin"] = vmin_nprw
            pcolor_kwargs_nprw["vmax"] = vmax_nprw

        im0 = ax_nprw.pcolormesh(
            nprw_x_edges,
            y_edges,
            nprw_masked,
            **pcolor_kwargs_nprw,
        )
        im0.set_edgecolor('none')
        # im1.set_rasterized(True)
        ax_nprw.grid(False)

        ax_nprw.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2, ls="--")
        # ax_nprw.axvspan(-20.0, 120.0, color="gray", alpha=1)
        if title_NA:
            ax_nprw.set_title(title_NA, fontsize=15, fontweight="bold")
        else:
            ax_nprw.set_title("")
        _ylabel_horizontal(ax_nprw, f"IntA\n{nprw_med.shape[0]} chs")
        if not nprw_show_y_ticks:
            ax_nprw.set_yticks([])
            ax_nprw.tick_params(left=False, labelleft=False)
        else:
            ax_nprw.tick_params(axis="y", labelsize=12, width=1.4)
        ax_nprw_cax.cla(); [sp.set_visible(False) for sp in ax_nprw_cax.spines.values()]
        ax_nprw_cax.set_xticks([]); ax_nprw_cax.set_yticks([])
        if not has_ua:
            ax_nprw.tick_params(axis="x", labelbottom=True, labelsize=13, width=1.4)
            ax_nprw.set_xlabel("Time (ms)", fontsize=15, fontweight="bold")
        else:
            ax_nprw.tick_params(axis="x", labelbottom=False)
        ax_nprw._is_time_axis = True
        cb0 = fig.colorbar(im0, cax=ax_nprw_cax)
        try: cb0.solids.set_edgecolor('face')
        except: pass
        cb0.ax.tick_params(width=1.4, labelsize=12)
        cb0.outline.set_linewidth(1.2)
        if cb_label_nprw:
            cb0.set_label(cb_label_nprw, fontsize=14, fontweight="bold", labelpad=8)
        else:
            cb0.set_label("")
        row += 1

    # ---------- UA heatmaps (optional) ----------
    if has_ua:
        for group_idx, group in enumerate(ua_groups):
            mat_group   = group["mat"]
            label_group = group.get("label", "")
            code_group  = group.get("code", "")
            disp_group  = group.get("display_name", label_group)

            # 1) start from your existing per-label default
            vmin_ua_group, vmax_ua_group = UA_VRANGE_BY_LABEL.get(
                code_group, UA_VRANGE_BY_LABEL.get(label_group, UA_VRANGE_DEFAULT)
            )

            # 2) apply call-time overrides (scalar or dict). Call-time wins.
            def _pick(override, current, label, code=""):
                # allow: scalar number OR dict of label->value
                if override is None:
                    return current
                if isinstance(override, dict):
                    # allow dict[str, (min,max)] or separate dicts
                    val = override.get(code, override.get(label, None))
                    if val is None:
                        return current
                    if isinstance(val, (tuple, list)) and len(val) == 2:
                        # when dict maps label -> (min,max)
                        # caller would set only vmin_ua OR vmax_ua, so guard:
                        return val[0] if current is vmin_ua_group else val[1]
                    return float(val)
                # scalar
                return float(override)

            vmin_ua_group = _pick(vmin_ua, vmin_ua_group, label_group, code_group)
            vmax_ua_group = _pick(vmax_ua, vmax_ua_group, label_group, code_group)

            ax_ua     = fig.add_subplot(gs[row, 0])
            ax_ua_cax = fig.add_subplot(gs[row, 1])

            # Mask NaNs
            ua_masked = ma.masked_invalid(mat_group)

            # Determine normalization for UA
            norm_local_ua = None
            if isinstance(norm_ua, dict):
                norm_local_ua = norm_ua.get(code_group, norm_ua.get(label_group, None))
            elif norm_ua is not None:
                norm_local_ua = norm_ua

            if norm_local_ua is None and center_zero and (vmin_ua_group is not None) and (vmax_ua_group is not None):
                if vmin_ua_group < 0.0 < vmax_ua_group:
                    norm_local_ua = TwoSlopeNorm(vcenter=0.0, vmin=float(vmin_ua_group), vmax=float(vmax_ua_group))

            # Make a copy of the cmap and set NaN color to gray
            cmap_choice_ua = cmap_ua if cmap_ua is not None else cmap
            try:
                cmap_local_ua = plt.get_cmap(cmap_choice_ua).copy()
            except Exception:
                cmap_local_ua = cm.get_cmap(cmap_choice_ua).copy()
            cmap_local_ua.set_bad(color="#383838")
            
            # ua_masked: (n_ch, T)
            n_ch, T = ua_masked.shape

            if ua_edges_ms.size > T + 1:
                i = int(np.searchsorted(ua_edges_ms, 0.0, side="right") - 1)
                i = np.clip(i, 0, T - 1)

                if not (ua_edges_ms[i] <= 0.0 <= ua_edges_ms[i + 1]):
                    print(
                        f"[warn] 0 ms not within edges range "
                        f"[{ua_edges_ms[0]:.3f}, {ua_edges_ms[-1]:.3f}] – inserting anyway at bin {i}"
                    )
                    
                # 2) Insert a zero column in the data at column index i+1
                #    (this new column will live *between* bins i and i+1)
                ua_masked = np.insert(ua_masked, i, np.nan, axis=1)  # (n_ch, T+1)
            # 4) Build channel edges (0..n_ch)
            y_edges = np.arange(n_ch + 1)

            # 5) Replace imshow with pcolormesh
            ua_x_edges = _centers_to_edges(t_ua, ua_edges_ms)

            if ua_masked.shape[1] != len(ua_x_edges) - 1:
                raise ValueError(
                    f"UA pcolormesh mismatch: C={ua_masked.shape}, "
                    f"len(t_ua)={len(t_ua)}, len(x_edges)={len(ua_x_edges)}, "
                    f"len(ua_edges_ms)={len(ua_edges_ms)}"
                )

            pcolor_kwargs_ua = dict(
                cmap=cmap_local_ua,
                shading="flat",
                antialiased=False,
                rasterized=True,
            )
            if norm_local_ua is not None:
                pcolor_kwargs_ua["norm"] = norm_local_ua
            else:
                pcolor_kwargs_ua["vmin"] = vmin_ua_group
                pcolor_kwargs_ua["vmax"] = vmax_ua_group

            im1 = ax_ua.pcolormesh(
                ua_x_edges,
                y_edges,
                ua_masked,
                **pcolor_kwargs_ua,
            )
            im1.set_edgecolor('none')
            # im1.set_rasterized(True)
            ax_ua.grid(False)
            
            ax_ua.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2, ls="--")
            # ax_u.axvspan(-5.0, 105.0, color="gray", alpha=1)
            _ylabel_horizontal(ax_ua, f"{disp_group}\n{mat_group.shape[0]} chs")
            ax_ua.set_yticks([]); ax_ua.tick_params(left=False, labelleft=False)
            ax_ua._is_time_axis = True

            is_last = (group_idx == len(ua_groups) - 1)
            if is_last:
                ax_ua.set_xlabel("Time (ms)", fontsize=15, fontweight="bold")
                ax_ua.tick_params(axis="x", labelbottom=True, labelsize=13, width=1.4)
            else:
                ax_ua.set_xlabel("")
                ax_ua.tick_params(axis="x", labelbottom=False)

            ax_ua_cax.cla(); [sp.set_visible(False) for sp in ax_ua_cax.spines.values()]
            ax_ua_cax.set_xticks([]); ax_ua_cax.set_yticks([])
            cb1 = fig.colorbar(im1, cax=ax_ua_cax)
            cb1.ax.tick_params(width=1.4, labelsize=12)
            cb1.outline.set_linewidth(1.2)
            if not has_nprw and group_idx == 0 and cb_label_ua:
                cb1.set_label(cb_label_ua, fontsize=14, fontweight="bold", labelpad=8)
            else:
                cb1.set_label("")
            row += 1

    # ---------- Probe inset (optional) ----------
    if have_probe:
        try:
            if probe is None and Probe is not None:
                pr = Probe(ndim=2)
                pr.set_contacts(positions=np.asarray(probe_locs, float),
                                shapes="circle", shape_params={"radius": 5.0})
                try: pr.create_auto_shape()
                except Exception: pass
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
        if probe_local is not None and getattr(probe_local, "probe_shape", None) is None:
            try:
                probe_local.create_auto_shape()   # e.g., rectangular shank
            except Exception:
                pass

        # --- draw (note plural kwargs in contacts_kargs) ---
        if (probe_local is not None) and ('plot_probe' in globals()) and (plot_probe is not None):
            plot_probe(
                probe_local, ax=ax_probe, with_contact_id=False,
                contacts_colors=contacts_colors,
                probe_shape_kwargs={"facecolor": "none", "edgecolor": "black", "linewidth": 2.2},
                contacts_kargs={"edgecolors": "black", "linewidths": 1.6, "zorder": 3},
            )
        else:
            if probe_locs is not None:
                L = np.asarray(probe_locs)
                ax_probe.scatter(L[:, 0], L[:, 1], s=probe_marker_size,
                                facecolors="none", edgecolors="black", linewidths=1.6, zorder=3)
        # Remove x (um), y (um), ticks, ticklabels, and probe title
        ax_probe.set_xlabel("")
        ax_probe.set_ylabel("")
        ax_probe.set_xticks([])
        ax_probe.set_yticks([])
        ax_probe.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax_probe.set_title("")
        # Fill the full gridspec cell (avoid letterboxing)
        ax_probe.set_aspect("auto")
        ax_probe.set_box_aspect(None)      # let the grid cell dictate height
        ax_probe.margins(x=0.05,y=0.05)   # small padding


    if xlim is not None:
        xmin, xmax = float(xlim[0]), float(xlim[1])
    else:
        x_ranges = []
        if has_nprw: x_ranges.append((t_nprw[0], t_nprw[-1]))
        if has_ua:    x_ranges.append((t_ua[0],    t_ua[-1]))
        if beh_rel_time is not None and (have_cam0_pos or have_cam1_pos or have_cam0_vel or have_cam1_vel):
            x_ranges.append((beh_rel_time[0], beh_rel_time[-1]))
        if x_ranges:
            xmin = float(min(lo for lo, _ in x_ranges))
            xmax = float(max(hi for _, hi in x_ranges))
        else:
            xmin, xmax = None, None

    if xmin is not None and xmax is not None:
        # Sync x-lims
        for ax in [a for a in fig.axes if isinstance(a, plt.Axes)]:
            if getattr(ax, "_is_time_axis", False):
                ax.set_xlim(xmin, xmax)

    # ---------- Overall title ----------
    out_svg = Path(out_svg)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    if overall_title:
        fig.suptitle(overall_title, fontsize=16, fontweight="bold", y=0.995, va="top")
    fig.subplots_adjust(top=0.96)           # pull axes up toward the top
    fig.savefig(out_svg, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
