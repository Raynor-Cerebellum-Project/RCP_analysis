from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from probeinterface.plotting import plot_probe
from probeinterface import Probe
matplotlib.use("Agg")
import RCP_analysis as rcp

def _ua_region_from_elec(e: int) -> int:
    if e <= 0:      return -1
    if e <= 64:     return 0  # SMA
    if e <= 128:    return 1  # Dorsal premotor
    if e <= 192:    return 2  # M1 inferior
    return 3                  # M1 superior

def _load_combined_npz(p: Path):
    d = np.load(p, allow_pickle=True)
    # Intan
    i_rate = d["intan_rate_hz"]
    i_t    = d["intan_t_ms_aligned"] if "intan_t_ms_aligned" in d.files else d["intan_t_ms"]
    # UA
    u_rate = d["ua_rate_hz"]
    u_t    = d["ua_t_ms_aligned"] if "ua_t_ms_aligned" in d.files else d["ua_t_ms"]
    # stim (absolute Intan ms); we’ll align it below
    stim_ms = d["stim_ms"] if "stim_ms" in d.files else np.array([], dtype=float)
    # alignment meta (JSON)
    meta = json.loads(d["align_meta"].item()) if "align_meta" in d.files else {}
    return i_rate, i_t.astype(float), u_rate, u_t.astype(float), stim_ms.astype(float), meta

def _aligned_stim_ms(stim_ms_abs: np.ndarray, meta: dict) -> np.ndarray:
    """
    Convert absolute Intan stim times (ms) into the aligned timebase used in the
    combined file:
      intan_t_ms_aligned = intan_t_ms - t0_intan_ms
    where t0_intan_ms = anchor_sample * 1000 / fs_intan
    """
    if stim_ms_abs.size == 0:
        return stim_ms_abs
    if "anchor_ms" in meta:
        return stim_ms_abs - float(meta["anchor_ms"])
    # fallback if some files only had anchor_sample
    if "anchor_sample" in meta:
        fs_intan = float(meta.get("fs_intan", 30000.0))
        return stim_ms_abs - (float(meta["anchor_sample"]) * 1000.0 / fs_intan)
    return stim_ms_abs  # nothing to do

def _add_ua_region_bar(
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

def _stacked_heatmaps_two_t(
    intan_avg, ua_avg, t_intan, t_ua, out_png, title_top, title_bot,
    cmap="jet", vmin=None, vmax=None,
    probe=None, probe_locs=None, stim_idx=None,
    probe_title="Probe (Intan)", probe_width_ratio=0.35, probe_marker_size=28,
    ua_ids_1based: np.ndarray | None = None,
    ua_sort: str = "none",   # NEW: "none" | "elec" | "region_then_elec"
):

    have_probe = (probe is not None) or (
        probe_locs is not None and np.asarray(probe_locs).ndim == 2 and len(probe_locs) > 0
    )

    if have_probe:
        fig = plt.figure(figsize=(14, 8), constrained_layout=True)
        gs = GridSpec(nrows=2, ncols=3, figure=fig,
                      width_ratios=[1.0, 0.0001, probe_width_ratio])
        ax_top   = fig.add_subplot(gs[0, 0])
        ax_bot   = fig.add_subplot(gs[1, 0], sharex=ax_top)
        ax_probe = fig.add_subplot(gs[:, 2])
    else:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
        ax_probe = None

    # ----- Intan heatmap -----
    im0 = ax_top.imshow(
        intan_avg, aspect="auto", cmap=cmap, origin="lower",
        extent=[t_intan[0], t_intan[-1], 0, intan_avg.shape[0]],
        vmin=vmin, vmax=vmax
    )
    ax_top.axvline(0.0, color="k", alpha=0.8, linewidth=1.2)
    ax_top.axvspan(0.0, 100.0, color="gray", alpha=0.2)
    ax_top.set_title(title_top); ax_top.set_ylabel("Intan ch")
    fig.colorbar(im0, ax=ax_top).set_label("Δ FR (Hz)")

    # ----- UA heatmap -----
    # Reorder indices
    ua_plot = ua_avg
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
        ua_plot = ua_avg[order, :]
        ids_plot = ids[order]
        
    im1 = ax_bot.imshow(
        ua_plot, aspect="auto", cmap=cmap, origin="lower",
        extent=[t_ua[0], t_ua[-1], 0, ua_plot.shape[0]],
        vmin=vmin, vmax=vmax
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
    _add_ua_region_bar(ax_bot, ua_plot.shape[0], ua_chan_ids_1based=ids_plot)

    fig.colorbar(im1, ax=ax_bot).set_label("Δ FR (Hz)")
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

    
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close(fig)

def _ua_title_from_meta(meta: dict) -> str:
    """
    Build Utah array title from metadata.
    """
    if "br_idx" in meta and meta["br_idx"] is not None:
        return f"Blackrock / UA: NRR_RW_001_{int(meta['br_idx']):03d}"
    return "Utah/BR"

if __name__ == "__main__":
    # ---------- CONFIG ----------
    REPO_ROOT = Path(__file__).resolve().parents[1]
    PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
    OUT_BASE  = rcp.resolve_output_root(PARAMS)

    ALIGNED_ROOT = OUT_BASE / "checkpoints" / "Aligned"
    FIG_ROOT     = OUT_BASE / "figures" / "peri_stim" / "Aligned"
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    # Peri-stim settings — identical to your NPRW run_one_Intan_FR_heatmap defaults
    WIN_MS            = (-800.0, 1200.0)
    BASELINE_FIRST_MS = 100.0
    MIN_TRIALS        = 1

    xls = rcp.ua_excel_path(REPO_ROOT, PARAMS.probes)
    ua_map = rcp.load_UA_mapping_from_excel(xls) if xls else None

    # Intan probe (optional, only used if you later want a probe panel like NPRW)
    try:
        GEOM_PATH = (
            Path(PARAMS.geom_mat_rel).resolve()
            if getattr(PARAMS, "geom_mat_rel", None) and str(PARAMS.geom_mat_rel).startswith("/")
            else (REPO_ROOT / PARAMS.geom_mat_rel).resolve()
            if getattr(PARAMS, "geom_mat_rel", None)
            else rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
        )
        probe, locs = rcp.build_probe_and_locs_from_geom(GEOM_PATH)
    except Exception:
        probe, locs = None, None

    files = sorted(ALIGNED_ROOT.glob("aligned__*.npz"))
    if not files:
        raise SystemExit(f"[error] No combined aligned NPZs found at {ALIGNED_ROOT}")

    for file in files:
        try:
            NPRW_rate, NPRW_t, UA_rate, UA_t, stim_ms_abs, meta = _load_combined_npz(file)
            # Try to get per-row UA electrode IDs (1..256), + NSP mapping (1..128/256)
            ua_ids_1based = None
            ua_row_to_nsp = None
            ua_region_names = None

            with np.load(file, allow_pickle=True) as z:
                if "ua_row_to_elec" in z.files:
                    ua_ids_1based = np.asarray(z["ua_row_to_elec"], dtype=int).ravel()
                else:
                    # legacy fallbacks just in case
                    for key in ("ua_ids_1based", "ua_elec_per_row", "ua_electrodes", "ua_chan_ids_1based"):
                        if key in z.files:
                            ua_ids_1based = np.asarray(z[key], dtype=int).ravel()
                            break
                if "ua_row_to_nsp" in z.files:
                    ua_row_to_nsp = np.asarray(z["ua_row_to_nsp"], dtype=int).ravel()
                if "ua_region_names" in z.files:
                    ua_region_names = np.asarray(z["ua_region_names"], dtype=object).ravel()

            # sanity: ensure length matches UA rows (plot uses row order)
            if ua_ids_1based is not None and ua_ids_1based.size != UA_rate.shape[0]:
                print(f"[warn] ua_row_to_elec len={ua_ids_1based.size} != UA rows={UA_rate.shape[0]} — ignoring")
                ua_ids_1based = None

            sess = meta.get("session", file.stem)

            # stim times aligned to Intan aligned timebase
            stim_ms = _aligned_stim_ms(stim_ms_abs, meta)

            # --- NPRW/Intan ---
            NPRW_segments, rel_time_ms_i = rcp.extract_peristim_segments(
                rate_hz=NPRW_rate, t_ms=NPRW_t, stim_ms=stim_ms,
                win_ms=WIN_MS, min_trials=MIN_TRIALS
            )
            NPRW_zeroed = rcp.baseline_zero_each_trial(
                NPRW_segments, rel_time_ms_i, baseline_first_ms=BASELINE_FIRST_MS
            )
            NPRW_avg = rcp.average_across_trials(NPRW_zeroed)

            # ---------- UA: same logic but keep its own rel_time_ms ----------
            UA_segments, ua_rel_time_ms = rcp.extract_peristim_segments(
                rate_hz=UA_rate, t_ms=UA_t, stim_ms=stim_ms,
                win_ms=WIN_MS, min_trials=MIN_TRIALS
            )
            UA_zeroed = rcp.baseline_zero_each_trial(
                UA_segments, ua_rel_time_ms, baseline_first_ms=BASELINE_FIRST_MS
            )
            UA_avg = rcp.average_across_trials(UA_zeroed)

            # ---------- produce stacked heatmap figure ----------
            out_png = FIG_ROOT / f"{sess}__Intan_vs_UA__peri_stim_heatmaps.png"
            title_top = f"Avg Δ in firing rate (baseline = first 100ms)\nNPRW/Intan: {sess} (n={NPRW_segments.shape[0]} trials)"
            title_bot = f"{_ua_title_from_meta(meta)} (n={UA_segments.shape[0]} trials)"

            # sanity checks
            assert NPRW_avg.shape[1] == rel_time_ms_i.size, "Intan avg vs time mismatch"
            assert UA_avg.shape[1]   == ua_rel_time_ms.size, "UA avg vs time mismatch"

            VMIN, VMAX = -500, 1000  # keep heatmap range if you like

            # locate the Intan stim_stream.npz and detect stimulated channels
            bundles_root = OUT_BASE / "bundles" / "NPRW"
            stim_npz = bundles_root / f"{sess}_Intan_bundle" / "stim_stream.npz"
            stim_idx = None
            if stim_npz.exists():
                try:
                    stim_idx = rcp.detect_stim_channels_from_npz(stim_npz, eps=1e-12, min_edges=1)
                except Exception as e:
                    print(f"[warn] stim-site detection failed for {sess}: {e}")
        
            _stacked_heatmaps_two_t(
                NPRW_avg, UA_avg, rel_time_ms_i, ua_rel_time_ms,
                out_png, title_top, title_bot, cmap="jet", vmin=VMIN, vmax=VMAX,
                probe=probe, probe_locs=locs, stim_idx=stim_idx,
                probe_title="NPRW probe (stim sites highlighted)",
                ua_ids_1based=ua_ids_1based,
                ua_sort="region_then_elec",
            )
            print(f"[PLOT] Plotted {sess} and saved at {out_png.parent}")
            
        except Exception as e:
            print(f"[error] Failed on {file.name}: {e}")
            continue
