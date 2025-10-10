from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
import RCP_analysis as rcp
matplotlib.use("Agg")
matplotlib.rcParams['svg.fonttype'] = 'none'

# ---------- CONFIG ----------
WIN_MS            = (-750.0, 750.0)
BASELINE_FIRST_MS = 150.0
MIN_TRIALS        = 1

VMIN_INTAN, VMAX_INTAN = 0, 500  # keep heatmap range if you like
VMIN_UA, VMAX_UA = 0, 200  # keep heatmap range if you like
COLORMAP = "jet"

# ---- Resolving paths ----
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
OUT_BASE  = rcp.resolve_output_root(PARAMS); OUT_BASE.mkdir(parents=True, exist_ok=True)

ALIGNED_ROOT = OUT_BASE / "checkpoints" / "Aligned"
FIG_ROOT     = OUT_BASE / "figures" / "peri_stim" / "Aligned"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

XLS = rcp.ua_excel_path(REPO_ROOT, PARAMS.probes)
UA_MAP = rcp.load_UA_mapping_from_excel(XLS) if XLS else None

METADATA_CSV  = (rcp.resolve_data_root(PARAMS) / PARAMS.metadata_rel).resolve(); METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)

PARAMS = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
METADATA_CSV = (rcp.resolve_data_root(PARAMS) / PARAMS.metadata_rel).resolve()
METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)

FOUR_COLS = ["Stim_Frequency_Hz", "Current_uA", "Depth_mm", "Stim_Duration_ms"]


_KEYPOINTS_ORDER = ("wrist", "middle_finger_base", "middle_finger_tip")  # order you want
def _as_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)

def _median_behavior_line(series_on_common: np.ndarray,
                          t_common_ms: np.ndarray,
                          stim_ms: np.ndarray,
                          win_ms: Tuple[float,float],
                          baseline_ms: float,
                          min_trials: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = series_on_common[None, :]  # (1, T)
    segs, rel_t = rcp.extract_peristim_segments(arr, t_common_ms, stim_ms,
                                                win_ms=win_ms, min_trials=min_trials)
    if segs.size == 0:
        return np.full(int(max(1, np.round((win_ms[1]-win_ms[0]) /
                                           (np.median(np.diff(t_common_ms)) if t_common_ms.size>1 else 1.0)))),
                       np.nan, float), rel_t
    segs = rcp.baseline_zero_each_trial(segs, rel_t, baseline_first_ms=baseline_ms)
    segs = segs[:, 0, :]                  # (n_trials, n_t)
    med  = rcp.median_across_trials(
            segs[:, None, :]           # (n_trials, 1, n_t)  ← channel dim in axis 1
        ).ravel()                       # -> (n_t,)

    return med, rel_t

# ---------- Behavior (already mapped to common grid in the NPZ) ----------
def _ordered_xy_indices(cam_cols: List[str]) -> Tuple[List[int], List[str]]:
    """
    Return indices for 6 columns in the order:
      wrist_x, wrist_y, middle_finger_base_x, middle_finger_base_y, middle_finger_tip_x, middle_finger_tip_y
    Also return their names in the same order.
    """
    idx: List[int] = []
    names: List[str] = []
    cols_lc = [c.lower() for c in cam_cols]

    def _find_first(predicate):
        for i, c in enumerate(cols_lc):
            if predicate(i, c):
                return i
        return None

    for kp in _KEYPOINTS_ORDER:
        ix = _find_first(lambda i, c: c.endswith("_x") and kp in c)
        iy = _find_first(lambda i, c: c.endswith("_y") and kp in c)
        if ix is None or iy is None:
            # If any keypoint missing, keep place with None so shapes remain predictable; we fill NaNs later.
            idx.extend([-1, -1])
            names.extend([f"{kp}_x(MISSING)", f"{kp}_y(MISSING)"])
        else:
            idx.extend([ix, iy])
            names.extend([cam_cols[ix], cam_cols[iy]])
    return idx, names

def _z_per_column(M: np.ndarray) -> np.ndarray:
    out = M.astype(float, copy=True)
    for j in range(out.shape[1]):
        col = out[:, j]
        m, s = np.nanmean(col), np.nanstd(col)
        out[:, j] = (col - m) / s if np.isfinite(s) and s > 0 else np.full_like(col, np.nan)
    return out

def _aggregate_columns_to_bins(idx_bins: np.ndarray, values_2d: np.ndarray, n_bins: int) -> np.ndarray:
    """
    idx_bins: (N,) target bin per sample (-1 to ignore)
    values_2d: (N, D) per-sample values
    returns: (n_bins, D) mean per bin
    """
    out = np.full((n_bins, values_2d.shape[1]), np.nan, float)
    valid = (idx_bins >= 0) & (idx_bins < n_bins)
    if not np.any(valid):
        return out
    for d in range(values_2d.shape[1]):
        vals = values_2d[:, d]
        ok = valid & np.isfinite(vals)
        if not np.any(ok):
            continue
        counts = np.zeros(n_bins, dtype=np.int32)
        sums = np.zeros(n_bins, dtype=float)
        np.add.at(counts, idx_bins[ok], 1)
        np.add.at(sums,   idx_bins[ok], vals[ok])
        hit = counts > 0
        out[hit, d] = sums[hit] / counts[hit]
    return out

def _median_lines_for_columns(series_on_common: np.ndarray, t_common_ms: np.ndarray,
                              stim_ms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    series_on_common: (T, D) on common grid
    returns (D, T_window), rel_t
    """
    D = series_on_common.shape[1]
    lines = []
    rel_t_out = None
    for d in range(D):
        line, rel_t = _median_behavior_line(series_on_common[:, d], t_common_ms, stim_ms,
                                            WIN_MS, BASELINE_FIRST_MS, MIN_TRIALS)
        lines.append(line)
        if rel_t_out is None:
            rel_t_out = rel_t
    return np.vstack(lines), rel_t_out  # (D, T_window), rel_t

# Build the 6-column matrices in the requested order for each camera
def _select_matrix(cam, cols):
    idx, names = _ordered_xy_indices(cols)
    if cam.size == 0:
        M = np.zeros((0, len(idx)), float)
    else:
        M = np.empty((cam.shape[0], len(idx)), float)
        for k, j in enumerate(idx):
            if j == -1:
                M[:, k] = np.nan
            else:
                M[:, k] = cam[:, j]
    return M, names

def _fmt_num(x) -> str:
    if pd.isna(x):
        return "n/a"
    try:
        v = float(x)
        return f"{int(v)}" if v.is_integer() else f"{v:g}"
    except Exception:
        s = str(x).strip()
        return s if s else "n/a"

def _parse_delay_ms(val) -> int:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0
    s = str(val).strip()
    if not s:
        return 0
    if s.casefold() == "random":
        return 0
    try:
        return int(float(s))
    except Exception:
        return 0

def build_title_from_csv(
    csv_path: Path, *, sess: Optional[str] = None, br_file: Optional[int] = None
) -> str:
    df = pd.read_csv(csv_path)

    # Skip the first (attributes) row
    df_data = df.iloc[1:].reset_index(drop=True)
    if df_data.empty:
        return "Condition: n/a, n/a Hz, n/a µA, n/a mm, n/a ms, Delay: 0 ms"

    # Optional filters
    idx = pd.Series(True, index=df_data.index)
    if "UA_port" in df_data.columns:
        idx &= (df_data["UA_port"].astype(str).str.upper().str.strip() == "A")
    if "Movement_Trigger" in df_data.columns:
        idx &= ~(df_data["Movement_Trigger"].astype(str).str.strip().str.casefold() == "velocity")
    if sess and "Session" in df_data.columns:
        idx &= df_data["Session"].astype(str).str.contains(str(sess), regex=False)
    if br_file is not None and "BR_File" in df_data.columns:
        idx &= (pd.to_numeric(df_data["BR_File"], errors="coerce") == int(br_file))

    row = df_data.loc[idx].iloc[0] if idx.any() else df_data.iloc[0]

    # Delay: hard-coded column; default 0 (including 'random')
    delay_ms = _parse_delay_ms(row.get("Delay", None))

    parts = []
    if br_file is not None:
        parts.append(f"Condition: {_fmt_num(br_file)}")
    parts.extend([
        f"Freq: {_fmt_num(row.get('Stim_Frequency_Hz'))} Hz",
        f"Current: {_fmt_num(row.get('Current_uA'))} µA",
        f"Depth: {_fmt_num(row.get('Depth_mm'))} mm",
        f"Duration: {_fmt_num(row.get('Stim_Duration_ms'))} ms",
        f"Delay: {delay_ms} ms",
    ])
    if "UA_port" in df_data.columns:
        parts.append(f"UA Port: {_fmt_num(row.get('UA_port'))}")

    return ", ".join(parts), int(row.get('Video_File'))

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

            # ---------- Behavior (already mapped to common grid in the NPZ) ----------
            with np.load(file, allow_pickle=True) as z:
                common_t = z["intan_t_ms_aligned"].astype(float)  # common Intan grid
                beh_idx   = z.get("beh_common_idx", np.array([], dtype=np.int64)).astype(np.int64)
                beh_valid = z.get("beh_common_valid", np.array([], dtype=bool)).astype(bool)
                cam0      = z.get("beh_cam0", np.zeros((0,0), np.float32)).astype(float)
                cam1      = z.get("beh_cam1", np.zeros((0,0), np.float32)).astype(float)
                raw0 = z.get("beh_cam0_cols", None)
                raw1 = z.get("beh_cam1_cols", None)
                cam0_cols = [str(x) for x in _as_list(raw0)]
                cam1_cols = [str(x) for x in _as_list(raw1)]

            # Build (N,6) in the order: wrist_x, wrist_y, middle_finger_base_x, middle_finger_base_y, middle_finger_tip_x, middle_finger_tip_y
            cam0_M, cam0_names = _select_matrix(cam0, cam0_cols)
            cam1_M, cam1_names = _select_matrix(cam1, cam1_cols)

            # z-score each column independently
            cam0_M = _z_per_column(cam0_M) if cam0_M.size else cam0_M
            cam1_M = _z_per_column(cam1_M) if cam1_M.size else cam1_M

            # Map samples to bins using precomputed indices; mask out-of-tolerance
            bin_idx = np.where(beh_valid, beh_idx, -1)

            # Aggregate to the common grid → (T,6)
            cam0_on_grid = _aggregate_columns_to_bins(bin_idx, cam0_M, common_t.size)
            cam1_on_grid = _aggregate_columns_to_bins(bin_idx, cam1_M, common_t.size)

            # Peri-stim medians for each of the 6 traces per camera → (6, Twindow)
            cam0_lines, beh_rel_t = _median_lines_for_columns(cam0_on_grid, common_t, stim_ms)
            cam1_lines, _         = _median_lines_for_columns(cam1_on_grid, common_t, stim_ms)

            # Labels for legend/axis in the figure function
            beh_labels = [
                "wrist_x","wrist_y",
                "middle_finger_base_x","middle_finger_base_y",
                "middle_finger_tip_x","middle_finger_tip_y",
            ]

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
            out_svg = FIG_ROOT / f"{sess}__Intan_vs_UA__peri_stim_heatmaps.png"
            
            title_top = f"Median Δ in firing rate (baseline = first 100ms)\nNPRW/Intan: {sess} (n={NPRW_segments.shape[0]} trials)"
            title_bot = f"{rcp.ua_title_from_meta(meta)} (n={UA_segments.shape[0]} trials)"

            # sanity checks
            assert NPRW_med.shape[1] == rel_time_ms_i.size, "Intan med vs time mismatch"
            assert UA_med.shape[1]   == ua_rel_time_ms.size, "UA med vs time mismatch"

            # locate the Intan stim_stream.npz and locate stimulated channels
            bundles_root = OUT_BASE / "bundles" / "NPRW"
            stim_npz = bundles_root / f"{sess}_Intan_bundle" / "stim_stream.npz"
            stim_locs = None
            if stim_npz.exists():
                try:
                    stim_locs = rcp.detect_stim_channels_from_npz(stim_npz, eps=1e-12, min_edges=1)
                except Exception as e:
                    print(f"[warn] stim-site detection failed for {sess}: {e}")

            overall_title, video_file = build_title_from_csv(
                METADATA_CSV, br_file=meta.get("br_idx")
            )
    
            title_kinematics = (
                f"Median Kinematics Right Camera (Cam-0) "
                f"Video: {video_file if video_file else 'n/a'}"
            )
            
            rcp.stacked_heatmaps_plus_behv(
                NPRW_med, UA_med, rel_time_ms_i, ua_rel_time_ms,
                out_svg, title_kinematics, title_top, title_bot, cmap=COLORMAP,
                vmin_intan=VMIN_INTAN, vmax_intan=VMAX_INTAN,
                vmin_ua=VMIN_UA, vmax_ua=VMAX_UA,
                probe=probe, probe_locs=locs, stim_idx=stim_locs,
                probe_title="NPRW probe (stim sites highlighted)",
                ua_ids_1based=ua_ids_1based,
                ua_sort="region_then_elec",
                # behavior args:
                beh_rel_time=beh_rel_t,          # (Twindow,)
                beh_cam0_lines=cam0_lines,       # (6, Twindow)
                beh_cam1_lines=cam1_lines,       # (6, Twindow)
                beh_labels=beh_labels,           # list of 6 stringns
                sess=sess,
                overall_title=overall_title
            )

            print(f"[PLOT] Plotted {sess} and saved at {out_svg.parent}")
            
        except Exception as e:
            print(f"[error] Failed on {file.name}: {e}")
            continue

if __name__ == "__main__":
    main()

