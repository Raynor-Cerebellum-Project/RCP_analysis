from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
import RCP_analysis as rcp
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'

# ---------- CONFIG ----------
WIN_MS            = (-800.0, 1200.0)
BASELINE_FIRST_MS = 200.0
MIN_TRIALS        = 1

# ---- Resolving paths ----
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
OUT_BASE  = rcp.resolve_output_root(PARAMS); OUT_BASE.mkdir(parents=True, exist_ok=True)

ALIGNED_ROOT = OUT_BASE / "checkpoints" / "Aligned"
FIG_ROOT     = OUT_BASE / "figures" / "peri_stim" / "Aligned"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

XLS = rcp.ua_excel_path(REPO_ROOT, PARAMS.probes)
UA_MAP = rcp.load_UA_mapping_from_excel(XLS) if XLS else None

def main():
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
            NPRW_rate, NPRW_t, UA_rate, UA_t, stim_ms_abs, meta = rcp.load_combined_npz(file)
            # Try to get per-row UA electrode IDs (1..256), + NSP mapping (1..128/256)
            ua_ids_1based = None

            with np.load(file, allow_pickle=True) as z:
                if "ua_row_to_elec" in z.files:
                    ua_ids_1based = np.asarray(z["ua_row_to_elec"], dtype=int).ravel()
                else:
                    # legacy fallbacks just in case
                    for key in ("ua_ids_1based", "ua_elec_per_row", "ua_electrodes", "ua_chan_ids_1based"):
                        if key in z.files:
                            ua_ids_1based = np.asarray(z[key], dtype=int).ravel()
                            break

            # sanity: ensure length matches UA rows (plot uses row order)
            if ua_ids_1based is not None and ua_ids_1based.size != UA_rate.shape[0]:
                print(f"[warn] ua_row_to_elec len={ua_ids_1based.size} != UA rows={UA_rate.shape[0]} — ignoring")
                ua_ids_1based = None

            sess = meta.get("session", file.stem)

            # stim times aligned to Intan aligned timebase
            stim_ms = rcp.aligned_stim_ms(stim_ms_abs, meta)

            # --- NPRW/Intan ---
            NPRW_segments, rel_time_ms_i = rcp.extract_peristim_segments(
                rate_hz=NPRW_rate, t_ms=NPRW_t, stim_ms=stim_ms,
                win_ms=WIN_MS, min_trials=MIN_TRIALS
            )
            NPRW_zeroed = rcp.baseline_zero_each_trial(
                NPRW_segments, rel_time_ms_i, baseline_first_ms=BASELINE_FIRST_MS
            )
            NPRW_med = rcp.median_across_trials(NPRW_zeroed)

            # ---------- UA: same logic but keep its own rel_time_ms ----------
            UA_segments, ua_rel_time_ms = rcp.extract_peristim_segments(
                rate_hz=UA_rate, t_ms=UA_t, stim_ms=stim_ms,
                win_ms=WIN_MS, min_trials=MIN_TRIALS
            )
            UA_zeroed = rcp.baseline_zero_each_trial(
                UA_segments, ua_rel_time_ms, baseline_first_ms=BASELINE_FIRST_MS
            )
            UA_med = rcp.median_across_trials(UA_zeroed)

            # ---------- produce stacked heatmap figure ----------
            out_svg = FIG_ROOT / f"{sess}__Intan_vs_UA__peri_stim_heatmaps.svg"
            title_top = f"Median Δ in firing rate (baseline = first 100ms)\nNPRW/Intan: {sess} (n={NPRW_segments.shape[0]} trials)"
            title_bot = f"{rcp.ua_title_from_meta(meta)} (n={UA_segments.shape[0]} trials)"

            # sanity checks
            assert NPRW_med.shape[1] == rel_time_ms_i.size, "Intan med vs time mismatch"
            assert UA_med.shape[1]   == ua_rel_time_ms.size, "UA med vs time mismatch"

            VMIN_INTAN, VMAX_INTAN = -500, 1000  # keep heatmap range if you like
            VMIN_UA, VMAX_UA = -250, 500  # keep heatmap range if you like

            # locate the Intan stim_stream.npz and locate stimulated channels
            bundles_root = OUT_BASE / "bundles" / "NPRW"
            stim_npz = bundles_root / f"{sess}_Intan_bundle" / "stim_stream.npz"
            stim_locs = None
            if stim_npz.exists():
                try:
                    stim_locs = rcp.detect_stim_channels_from_npz(stim_npz, eps=1e-12, min_edges=1)
                except Exception as e:
                    print(f"[warn] stim-site detection failed for {sess}: {e}")
        
            rcp.stacked_heatmaps_two_t(
                NPRW_med, UA_med, rel_time_ms_i, ua_rel_time_ms,
                out_svg, title_top, title_bot, cmap="jet", vmin_intan=VMIN_INTAN, vmax_intan=VMAX_INTAN,
                vmin_ua=VMIN_UA, vmax_ua=VMAX_UA,
                probe=probe, probe_locs=locs, stim_idx=stim_locs,
                probe_title="NPRW probe (stim sites highlighted)",
                ua_ids_1based=ua_ids_1based,
                ua_sort="region_then_elec",
            )
            print(f"[PLOT] Plotted {sess} and saved at {out_svg.parent}")
            
        except Exception as e:
            print(f"[error] Failed on {file.name}: {e}")
            continue

if __name__ == "__main__":
    main()
