import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from pathlib import Path
import numpy.ma as ma
from matplotlib import cm

from probeinterface.plotting import plot_probe
from probeinterface import Probe

# ---- knobs ----
GAP_AFTER_BEHAVIOR = True
GAP_HEIGHT = 0.22   # relative height of the spacer row (tune to taste)

BEH_RATIO = 0.6
CH_RATIO_PER_ROW = 0.015
MIN_HEATMAP_RATIO = 0.6
UA_COMPACT_FACTOR = 0.95
NPRW_SCALE      = 0.6   # < 1.0 shrinks NPRW height (e.g., 0.6 = 60% of previous)
FIG_WIDTH_IN        = 16.0
HEIGHT_PER_RATIO_IN = 4.0

# ---- knobs: UA vmin/vmax per group/label ----
UA_VRANGE_BY_LABEL = {
    "M1i+M1s": (-30, 125),
    "PMd":     (-30, 150),
    "SMA":     (-10, 40),
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
                ha="right", va="center", fontsize=9)
        y0 = y1

    # 3 tick marks at the internal boundaries
    for y in np.cumsum(splits)[:-1]:
        ax.plot([x, x - 0.012], [y, y], transform=ax.transAxes,
                color="k", lw=1.2, clip_on=False)

# ---- Plotting FR for both ----
def stacked_heatmaps_plus_behv(
    nprw_med, ua_med, t_nprw, t_ua, out_svg,
    title_kinematics, title_NA,
    *, cmap="jet",
    vmin_nprw=None, vmax_nprw=None,
    vmin_ua=None, vmax_ua=None,
    probe=None, probe_locs=None, stim_idx=None,
    probe_title="Probe (NPRW)", probe_width_ratio=0.15, probe_marker_size=28,
    probe_gap_ratio=0.04,
    ua_ids_1based=None, ua_sort="region_then_elec",
    beh_rel_time=None, beh_cam0_lines=None, beh_cam1_lines=None,
    beh_labels=None, beh_cam0_vel_lines=None, beh_cam1_vel_lines=None,
    # Optional per-timepoint SEM arrays for shaded plotting of position traces
    beh_cam0_pos_sems=None, beh_cam1_pos_sems=None,
    title_cam1=None, title_cam0_vel=None, title_cam1_vel=None,
    sess="",
    overall_title="",
    beh_ylim=None,
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

    beh_cam0_pos, have_cam0_pos = _prepare_cam(beh_cam0_lines, beh_rel_time)
    beh_cam1_pos, have_cam1_pos = _prepare_cam(beh_cam1_lines, beh_rel_time)
    beh_cam0_vel, have_cam0_vel = _prepare_cam(beh_cam0_vel_lines, beh_rel_time)
    beh_cam1_vel, have_cam1_vel = _prepare_cam(beh_cam1_vel_lines, beh_rel_time)

    beh_rows = []
    if have_cam0_pos: beh_rows.append(("beh", "cam0_pos"))
    if have_cam1_pos: beh_rows.append(("beh", "cam1_pos"))
    if have_cam0_vel: beh_rows.append(("beh", "cam0_vel"))
    if have_cam1_vel: beh_rows.append(("beh", "cam1_vel"))

    rowspec = list(beh_rows)  # [("beh","cam0_pos"), ("beh","cam1_pos"), ...]
    if GAP_AFTER_BEHAVIOR and (has_nprw or has_ua) and len(beh_rows):
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

                else:
                    order_valid = np.arange(valid.sum())

                order = np.r_[np.where(valid)[0][order_valid], np.where(~valid)[0]]
                ua_plot = ua_plot[order, :]
                ids_arr = ids_arr[order]
                regs = regs[order]

            # now group using the same regs
            ua_groups = []

            def _append(mask, label):
                if mask.sum():
                    ua_groups.append({
                        "mat": ua_plot[mask, :],
                        "ids": ids_arr[mask],
                        "label": label,
                    })

            _append((regs == 2) | (regs == 3), "M1i+M1s")
            _append((regs == 1), "PMd")
            _append((regs == 0), "SMA")

            other_mask = (regs >= 1_000_000)
            if other_mask.any():
                _append(other_mask, "UA (other)")

        else:
            ua_groups = [{"mat": ua_plot, "ids": None, "label": "UA (all)"}]


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
        height_ratios.append(BEH_RATIO)

    # optional gap row
    if ("gap", None) in rowspec:
        height_ratios.append(GAP_HEIGHT)

    # nprw row
    if has_nprw:
        nprw_rows = nprw_med.shape[0]
        nprw_ratio = max(MIN_HEATMAP_RATIO, CH_RATIO_PER_ROW * nprw_rows) * float(NPRW_SCALE)
        height_ratios.append(nprw_ratio)

    # UA rows
    if has_ua:
        for group in ua_groups:
            rows = group["mat"].shape[0] if group["mat"] is not None else 0
            base = max(MIN_HEATMAP_RATIO, CH_RATIO_PER_ROW * rows)
            scale = 0.5 if any(k in group["label"].lower() for k in ("pmd", "sma")) else 1.0
            height_ratios.append(base * scale * UA_COMPACT_FACTOR)

    have_probe = (probe is not None) or (
        probe_locs is not None and np.asarray(probe_locs).ndim == 2 and len(probe_locs) > 0
    )

    total_ratio = sum(height_ratios)
    fig_height  = HEIGHT_PER_RATIO_IN * total_ratio

    if have_probe:
        fig_width = FIG_WIDTH_IN * (1.0 + probe_gap_ratio + probe_width_ratio)
    else:
        fig_width = FIG_WIDTH_IN

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
    def _ylabel_horizontal(ax, text: str, pad: float = 8.0):
        t = ax.set_ylabel(text, rotation=0, labelpad=pad)
        t.set_horizontalalignment("right")
        t.set_verticalalignment("center")
        return t

    def _plot_lines(ax, rel_t, lines, title, ylabel, sub, place_legend: bool, sems=None, ylim=None):
        """
        Plot lines (D x T). If `sems` is provided and has same shape as `lines`,
        draw a filled band ±sem behind each mean line.
        """
        if lines is None:
            ax.axis("off")
            return
        D = lines.shape[0]
        if ylim is not None:
            ax.set_ylim(ylim)

        for i in range(D):
            y = lines[i]
            if not np.isfinite(y).any():
                continue
            lab = labs[i] if i < len(labs) else f"trace_{i+1}"
            # plot mean line first to capture color from matplotlib cycle
            ln, = ax.plot(rel_t, y, lw=1.25, alpha=0.95, label=lab)
            # if sems available and matching shape, draw shaded band
            if sems is not None:
                try:
                    s = sems[i]
                    if np.shape(s) == np.shape(y):
                        col = ln.get_color()
                        lo = y - s
                        hi = y + s
                        ax.fill_between(rel_t, lo, hi, color=col, alpha=0.22, linewidth=0.0)
                except Exception:
                    pass

        ax.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2, ls="--")
        ax.axvspan(0.0, 100.0, color="0.7", alpha=0.15, zorder=0)  # light, behind data
        ax._is_time_axis = True
        _ylabel_horizontal(ax, ylabel)
        if title:
            ax.set_title(title)
        if place_legend and D:
            ax.legend(loc="center left",
                    bbox_to_anchor=(1.02, 0.5),  # was 1.02
                    frameon=False, fontsize=8, ncols=1, borderaxespad=0.0)
        ax.grid(alpha=0.15, linestyle=":")

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

            if sub == "cam0_pos":
                _plot_lines(ax, beh_rel_time, beh_cam0_pos, title_kinematics or "",
                            "Cam-0\nPosition Δ (z)", sub, place_legend_now, sems=beh_cam0_pos_sems, ylim=beh_ylim)
            elif sub == "cam1_pos":
                _plot_lines(ax, beh_rel_time, beh_cam1_pos, title_cam1 or "",
                            "Cam-1\nPosition Δ (z)", sub, place_legend_now, sems=beh_cam1_pos_sems, ylim=beh_ylim)
            elif sub == "cam0_vel":
                _plot_lines(ax, beh_rel_time, beh_cam0_vel, title_cam0_vel or "",
                            "Cam-0\nVelocity (z/ms)", sub, place_legend_now, ylim=beh_ylim)
            elif sub == "cam1_vel":
                _plot_lines(ax, beh_rel_time, beh_cam1_vel, title_cam1_vel or "",
                            "Cam-1\nVelocity (z/ms)", sub, place_legend_now, ylim=beh_ylim)

            if place_legend_now:
                legend_placed = True

            # hide x labels if another behavior row follows
            if beh_row_i < total_beh_rows:
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
        # Mask NaNs
        nprw_masked = ma.masked_invalid(nprw_med)

        # Make a copy of the cmap and set NaN color to gray
        cmap_local_nprw = cm.get_cmap(cmap).copy()
        cmap_local_nprw.set_bad(color="gray")

        im0 = ax_nprw.imshow(
            nprw_masked,
            aspect="auto",
            cmap=cmap_local_nprw,
            origin="lower",
            extent=[t_nprw[0], t_nprw[-1], 0, nprw_med.shape[0]],
            vmin=vmin_nprw,
            vmax=vmax_nprw,
        )
        ax_nprw.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2, ls="--")
        # ax_i.axvspan(-20.0, 120.0, color="gray", alpha=1)
        ax_nprw.set_title(title_NA); _ylabel_horizontal(ax_nprw, f"IP {nprw_med.shape[0]} chs")
        ax_nprw_cax.cla(); [sp.set_visible(False) for sp in ax_nprw_cax.spines.values()]
        ax_nprw_cax.set_xticks([]); ax_nprw_cax.set_yticks([])
        ax_nprw.tick_params(axis="x", labelbottom=False)
        ax_nprw._is_time_axis = True
        cb0 = fig.colorbar(im0, cax=ax_nprw_cax)
        try: cb0.solids.set_edgecolor('face')
        except: pass
        cb0.ax.tick_params(width=1.2, labelsize=9)
        cb0.outline.set_linewidth(1.0)
        cb0.set_label("Δ FR (Hz)")
        row += 1

    # ---------- UA heatmaps (optional) ----------
    if has_ua:
        for group_idx, group in enumerate(ua_groups):
            mat_group   = group["mat"]
            label_group = group.get("label", "")

            # 1) start from your existing per-label default
            vmin_ua_group, vmax_ua_group = UA_VRANGE_BY_LABEL.get(label_group, UA_VRANGE_DEFAULT)

            # 2) apply call-time overrides (scalar or dict). Call-time wins.
            def _pick(override, current, label):
                # allow: scalar number OR dict of label->value
                if override is None:
                    return current
                if isinstance(override, dict):
                    # allow dict[str, (min,max)] or separate dicts
                    val = override.get(label, None)
                    if val is None:
                        return current
                    if isinstance(val, (tuple, list)) and len(val) == 2:
                        # when dict maps label -> (min,max)
                        # caller would set only vmin_ua OR vmax_ua, so guard:
                        return val[0] if current is vmin_ua_group else val[1]
                    return float(val)
                # scalar
                return float(override)

            vmin_ua_group = _pick(vmin_ua, vmin_ua_group, label_group)
            vmax_ua_group = _pick(vmax_ua, vmax_ua_group, label_group)

            ax_ua     = fig.add_subplot(gs[row, 0])
            ax_ua_cax = fig.add_subplot(gs[row, 1])

            # Mask NaNs
            mat_masked = ma.masked_invalid(mat_group)

            # Make a copy of the cmap and set NaN color
            cmap_local = cm.get_cmap(cmap).copy()
            cmap_local.set_bad(color="gray")   # <- NaN = grey
            
            im1 = ax_ua.imshow(
                mat_masked, aspect="auto", cmap=cmap_local, origin="lower",
                extent=[t_ua[0], t_ua[-1], 0, mat_group.shape[0]],
                vmin=vmin_ua_group, vmax=vmax_ua_group
            )
            
            ax_ua.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2, ls="--")
            # ax_u.axvspan(-5.0, 105.0, color="gray", alpha=1)
            _ylabel_horizontal(ax_ua, f"{label_group} {mat_group.shape[0]} chs")
            ax_ua.set_yticks([]); ax_ua.tick_params(left=False, labelleft=False)
            ax_ua._is_time_axis = True

            is_last = (group_idx == len(ua_groups) - 1)
            if is_last:
                ax_ua.set_xlabel("Time (ms) rel. stim")
            else:
                ax_ua.set_xlabel("")
                ax_ua.tick_params(axis="x", labelbottom=False)

            ax_ua_cax.cla(); [sp.set_visible(False) for sp in ax_ua_cax.spines.values()]
            ax_ua_cax.set_xticks([]); ax_ua_cax.set_yticks([])
            cb1 = fig.colorbar(im1, cax=ax_ua_cax)
            cb1.ax.tick_params(width=1.2, labelsize=9)
            cb1.outline.set_linewidth(1.0)
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
                probe_shape_kwargs={"facecolor": "none", "edgecolor": "black", "linewidth": 0.8},
                contacts_kargs={"edgecolors": "k", "linewidths": 0.6, "zorder": 3},
            )
        else:
            if probe_locs is not None:
                L = np.asarray(probe_locs)
                ax_probe.scatter(L[:, 0], L[:, 1], s=probe_marker_size,
                                facecolors="none", edgecolors="k", linewidths=0.6, zorder=3)
        ax_probe.set_title(probe_title, fontsize=10)
        # Fill the full gridspec cell (avoid letterboxing)
        ax_probe.set_aspect("auto")
        ax_probe.set_box_aspect(None)      # let the grid cell dictate height
        ax_probe.margins(x=0.05, y=0.05)   # small padding

        # ax_probe.set_xticks([])
        # ax_probe.set_yticks([])

    # ---------- Sync x-lims ----------
    x_ranges = []
    if has_nprw: x_ranges.append((t_nprw[0], t_nprw[-1]))
    if has_ua:    x_ranges.append((t_ua[0],    t_ua[-1]))
    if beh_rel_time is not None and (have_cam0_pos or have_cam1_pos or have_cam0_vel or have_cam1_vel):
        x_ranges.append((beh_rel_time[0], beh_rel_time[-1]))
    if x_ranges:
        xmin = float(min(lo for lo, _ in x_ranges))
        xmax = float(max(hi for _, hi in x_ranges))
        # Sync x-lims
        for ax in [a for a in fig.axes if isinstance(a, plt.Axes)]:
            if getattr(ax, "_is_time_axis", False):
                ax.set_xlim(xmin, xmax)

    # ---------- Overall title ----------
    out_svg = Path(out_svg)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    if overall_title:
        fig.suptitle(overall_title, fontsize=13, fontweight="bold", y=0.995, va="top")
    fig.subplots_adjust(top=0.96)           # pull axes up toward the top
    fig.savefig(out_svg, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
