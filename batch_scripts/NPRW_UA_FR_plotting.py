from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib
import RCP_analysis as rcp
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'

# ---------- CONFIG ----------
WIN_MS            = (-800.0, 1200.0)
BASELINE_FIRST_MS = 100.0
MIN_TRIALS        = 1

# ---- Resolving paths ----
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
OUT_BASE  = rcp.resolve_output_root(PARAMS)
OUT_BASE.mkdir(parents=True, exist_ok=True)

ALIGNED_ROOT = OUT_BASE / "checkpoints" / "Aligned"
FIG_ROOT     = OUT_BASE / "figures" / "peri_stim" / "Aligned"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

XLS = rcp.ua_excel_path(REPO_ROOT, PARAMS.probes)
UA_MAP = rcp.load_UA_mapping_from_excel(XLS) if XLS else None
        
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

def _ua_title_from_meta(meta: dict) -> str:
    """
    Build Utah array title from metadata.
    """
    if "br_idx" in meta and meta["br_idx"] is not None:
        return f"Blackrock / UA: NRR_RW_001_{int(meta['br_idx']):03d}"
    return "Utah/BR"

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
            title_bot = f"{_ua_title_from_meta(meta)} (n={UA_segments.shape[0]} trials)"

            # sanity checks
            assert NPRW_med.shape[1] == rel_time_ms_i.size, "Intan med vs time mismatch"
            assert UA_med.shape[1]   == ua_rel_time_ms.size, "UA med vs time mismatch"

            VMIN_INTAN, VMAX_INTAN = -500, 1000  # keep heatmap range if you like
            VMIN_UA, VMAX_UA = -250, 500  # keep heatmap range if you like

            # locate the Intan stim_stream.npz and detect stimulated channels
            bundles_root = OUT_BASE / "bundles" / "NPRW"
            stim_npz = bundles_root / f"{sess}_Intan_bundle" / "stim_stream.npz"
            stim_idx = None
            if stim_npz.exists():
                try:
                    stim_idx = rcp.detect_stim_channels_from_npz(stim_npz, eps=1e-12, min_edges=1)
                except Exception as e:
                    print(f"[warn] stim-site detection failed for {sess}: {e}")
        
            stacked_heatmaps_two_t(
                NPRW_med, UA_med, rel_time_ms_i, ua_rel_time_ms,
                out_svg, title_top, title_bot, cmap="jet", vmin_intan=VMIN_INTAN, vmax_intan=VMAX_INTAN,
                vmin_ua=VMIN_UA, vmax_ua=VMAX_UA,
                probe=probe, probe_locs=locs, stim_idx=stim_idx,
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
