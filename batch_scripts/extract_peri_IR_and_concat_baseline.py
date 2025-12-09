from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import json, re, csv
import numpy as np
import pandas as pd
from scipy.io import savemat
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import RCP_analysis as rcp

"""
Baseline peristim extraction:
- Uses IR events instead of stim times as peri-event centers.
- Concatenates all baseline trials per (UA_port, Depth_mm) group.
- Produces per-group Target A/B NPZ + MAT files with the SAME structure
  as the peristim script's per-BR Target A/B outputs.
"""

# ---------------------------------------------------------------------
# CONFIG / ROOTS  (same pattern as peristim script)
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS = rcp.load_experiment_params(
    REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT
)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE = SESSION_LOC / "results"
OUT_BASE.mkdir(parents=True, exist_ok=True)

KEYPOINTS_ORDER = tuple(PARAMS.kinematics.get("keypoints", []))

# aligned combined bundles (same as peristim)
ALIGNED_ROOT = OUT_BASE / "checkpoints" / "Aligned"

# metadata CSV (same as peristim)
METADATA_ROOT = SESSION_LOC / "Metadata"
METADATA_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_CSV = METADATA_ROOT / f"{Path(PARAMS.session)}_metadata.csv"

# baseline peristim outputs
PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim"
PERI_ROOT.mkdir(parents=True, exist_ok=True)

# same windows as peristim
WIN_MS = (-600.0, 600.0)
NORMALIZE_FIRST_MS = 150.0
MIN_TRIALS = 1

# ---------- Intan + shifts config (copied from original baseline script) ----------
INTAN_ROOT   = SESSION_LOC / "Intan"
INTAN_ROOT.mkdir(parents=True, exist_ok=True)

METADATA_ROOT = SESSION_LOC / "Metadata"
METADATA_ROOT.mkdir(parents=True, exist_ok=True)
SHIFTS_CSV   = METADATA_ROOT / "br_to_intan_shifts.csv"

IR_STREAM    = "USB board digital input channel"  # same as original
DEFAULT_BR_FS = 30000.0  # only used in your behavior script; not critical here

def _load_br_to_intan_shifts(shifts_csv: Path) -> dict[int, dict]:
    """
    Return {BR_idx: {"session": <str>, "anchor_ms": <float or 0.0>}}.
    Accepts header variations.
    """
    if not shifts_csv.exists():
        raise SystemExit(f"[error] shifts CSV not found: {shifts_csv}")
    with shifts_csv.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            raise SystemExit(f"[error] {shifts_csv.name} has no header")
        # normalize header names -> original
        cols = { re.sub(r'[^a-z0-9]+','', c.lower()): c for c in rdr.fieldnames }

        def col(*cands):
            for cand in cands:
                k = re.sub(r'[^a-z0-9]+','', cand.lower())
                if k in cols: 
                    return cols[k]
            return None

        c_br    = col("br_idx")
        c_sess  = col("session")
        c_shift = col("anchor_ms")

        if not c_br or not c_sess:
            raise SystemExit(
                f"[error] {shifts_csv.name} missing BR and/or Session columns "
                f"(have: {rdr.fieldnames})"
            )

        out: dict[int, dict] = {}
        for row in rdr:
            try:
                br = int(float(str(row[c_br]).strip()))
            except Exception:
                continue
            sess = str(row.get(c_sess, "")).strip()
            if not sess:
                continue
            try:
                shift_ms = float(str(row.get(c_shift, "0")).strip()) if c_shift else 0.0
            except Exception:
                shift_ms = 0.0
            out[br] = {"session": sess, "anchor_ms": shift_ms}
    return out

BR_TO_INTAN_SHIFTS = _load_br_to_intan_shifts(SHIFTS_CSV)
   
# ---------------------------------------------------------------------
# Small helpers (identical to peristim or trivial)
# ---------------------------------------------------------------------
def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def _end_or_nan(arr):
    arr = np.asarray(arr)
    return float(arr[-1]) if arr.size else float("nan")

def _stims_mask_for_stream(stim_ms: np.ndarray, t_ms: np.ndarray,
                           win_ms: tuple[float, float]) -> np.ndarray:
    """Boolean mask over stim_ms for which [stim+win0, stim+win1] lies in t_ms."""
    if stim_ms is None or t_ms is None or np.size(stim_ms) == 0 or np.size(t_ms) < 2:
        return np.zeros(stim_ms.size if stim_ms is not None else 0, dtype=bool)
    t0, t1 = float(t_ms[0]), float(t_ms[-1])
    w0, w1 = float(win_ms[0]), float(win_ms[1])
    return (stim_ms + w0 >= t0) & (stim_ms + w1 <= t1)

def _filter_stims_for_stream(stim_ms: np.ndarray, t_ms: np.ndarray,
                             win_ms: tuple[float, float]) -> np.ndarray:
    """Keep only stims whose [stim+win0, stim+win1] lies fully within t_ms range."""
    if stim_ms is None or t_ms is None or np.size(stim_ms) == 0 or np.size(t_ms) < 2:
        return np.array([], dtype=float)
    mask = _stims_mask_for_stream(stim_ms, t_ms, win_ms)
    return np.asarray(stim_ms, float)[mask]

def _safe_extract_segments(rate_hz, counts, t_ms, stim_ms_in, win_ms,
                           min_trials, normalize_first_ms):
    """Filter to in-range stims, then extract without crashing if 0-kept."""
    st = _filter_stims_for_stream(stim_ms_in, t_ms, win_ms)
    if st.size == 0:
        dt = float(np.nanmedian(np.diff(t_ms))) if np.size(t_ms) > 1 else 1.0
        rel_t = np.arange(win_ms[0], win_ms[1] + 1e-9, dt, dtype=float)
        # med=None, var=None, n_trials=0, zeroed=None
        return None, None, rel_t, 0, None, None

    try:
        rate_segs, count_segs, rel_t, _ = rcp.extract_peristim_segments(
            rate_hz=rate_hz,
            counts=counts,
            t_ms=t_ms,
            stim_ms=st,
            win_ms=win_ms,
            min_trials=min_trials,
        )
    except RuntimeError as e:
        if "Only 0 peri-stim segments" in str(e):
            dt = float(np.nanmedian(np.diff(t_ms))) if np.size(t_ms) > 1 else 1.0
            rel_t = np.arange(win_ms[0], win_ms[1] + 1e-9, dt, dtype=float)
            return None, None, rel_t, 0, None, None
        raise

    zeroed_rates = rcp.baseline_zero_each_trial(
        rate_segs, rel_t, normalize_first_ms=normalize_first_ms
    )

    # drop bins with too few valid trials per channel
    valid_counts = np.isfinite(zeroed_rates).sum(axis=0)      # (n_channels, T)
    n_events = zeroed_rates.shape[0]
    # require at least 30% of events, but never more than n_events
    min_bin_trials = max(1, int(0.3 * n_events))
    min_bin_trials = min(min_bin_trials, n_events)

    med_rates = rcp.median_across_trials(zeroed_rates)              # (n_channels, T)
    var_rates = rcp.variance_across_trials(zeroed_rates)
    med_rates[valid_counts < min_bin_trials] = np.nan
    var_rates[valid_counts < min_bin_trials] = np.nan

    return med_rates, var_rates, rel_t, int(n_events), zeroed_rates, count_segs

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)

def _median_behavior_line(series_on_common: np.ndarray,
                          t_common_ms: np.ndarray,
                          stim_ms: np.ndarray,
                          win_ms: Tuple[float, float],
                          baseline_ms: float,
                          min_trials: int
                          ) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Return:
      line : (T,) median peri-stim trace across kept trials
      rel_t: (T,) relative time axis
      n_kept: int, number of kept trials
      segs: (n_kept, T) baseline-zeroed single-trial segments
    """
    s = np.asarray(series_on_common, float)

    try:
        rate_segs, _, rel_t, _ = rcp.extract_peristim_segments(
            rate_hz=s[None, :],
            counts=None,
            t_ms=t_common_ms,
            stim_ms=stim_ms,
            win_ms=win_ms,
            min_trials=min_trials,
        )
    except RuntimeError as e:
        if "Only 0 peri-stim segments" in str(e):
            dt = np.nanmedian(np.diff(t_common_ms)) if t_common_ms.size > 1 else 1.0
            rel_t = np.arange(win_ms[0], win_ms[1] + 1e-9, dt, dtype=float)
            return np.full(rel_t.size, np.nan, float), rel_t, 0, np.zeros((0, rel_t.size), float)
        else:
            raise

    if rate_segs.size == 0:
        dt = np.nanmedian(np.diff(t_common_ms)) if t_common_ms.size > 1 else 1.0
        rel_t = np.arange(win_ms[0], win_ms[1] + 1e-9, dt, dtype=float)
        return np.full(rel_t.size, np.nan, float), rel_t, 0, np.zeros((0, rel_t.size), float)

    segs = rate_segs[:, 0, :]  # (n_trials, T)

    bl_mask = (rel_t >= rel_t[0]) & (rel_t <= baseline_ms)
    if not bl_mask.any():
        step = max(1, int(round(baseline_ms / (np.nanmedian(np.diff(rel_t)) if rel_t.size > 1 else 1.0))))
        bl_mask = np.zeros_like(rel_t, bool)
        bl_mask[:step] = True

    keep = np.isfinite(segs[:, bl_mask]).any(axis=1)
    n_kept = int(np.count_nonzero(keep))
    if n_kept == 0:
        return np.full(segs.shape[1], np.nan, float), rel_t, 0, np.zeros((0, segs.shape[1]), float)

    segs = segs[keep]
    segs = segs - np.nanmedian(segs[:, bl_mask], axis=1, keepdims=True)
    line = np.nanmedian(segs, axis=0)
    return line, rel_t, n_kept, segs

def _median_lines_for_columns(series_on_common: np.ndarray,
                              t_common_ms: np.ndarray,
                              stim_ms: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Return:
      lines      : (D, T) median line per column
      rel_t_out  : (T,) relative time axis
      n_trials   : int, max number of kept trials across columns
      segs_all   : (n_trials, D, T) baseline-zeroed segments per column,
                   padded with NaNs where a column has fewer trials
    """
    if series_on_common is None or np.size(series_on_common) == 0:
        return np.zeros((0, 0), float), np.array([], float), 0, np.zeros((0, 0, 0), float)
    if series_on_common.ndim == 1:
        series_on_common = series_on_common[:, None]

    T, D = series_on_common.shape
    if D == 0:
        return np.zeros((0, 0), float), np.array([], float), 0, np.zeros((0, 0, 0), float)

    lines: list[np.ndarray] = []
    kept_counts: list[int] = []
    segs_list: list[np.ndarray] = []
    rel_t_out: Optional[np.ndarray] = None

    for d in range(D):
        line, rel_t, n_kept, segs = _median_behavior_line(
            series_on_common[:, d],
            t_common_ms, stim_ms,
            WIN_MS, NORMALIZE_FIRST_MS, MIN_TRIALS
        )
        lines.append(line)
        kept_counts.append(int(n_kept))
        segs_list.append(segs)
        if rel_t_out is None:
            rel_t_out = rel_t

    if rel_t_out is None or rel_t_out.size == 0:
        dt = np.nanmedian(np.diff(t_common_ms)) if t_common_ms.size > 1 else 1.0
        rel_t_out = np.arange(WIN_MS[0], WIN_MS[1] + 1e-9, dt, dtype=float)

    # stack median lines
    if lines:
        lines_arr = np.vstack(lines)  # (D, T)
    else:
        lines_arr = np.zeros((0, rel_t_out.size), float)

    # build segs_all: (max_trials, D, T)
    max_trials = max((s.shape[0] for s in segs_list), default=0)
    if max_trials == 0:
        segs_all = np.zeros((0, D, rel_t_out.size), float)
    else:
        segs_all = np.full((max_trials, D, rel_t_out.size), np.nan, float)
        for d, segs in enumerate(segs_list):
            if segs.size == 0:
                continue
            n_d, T_d = segs.shape
            # if T_d differs from rel_t_out.size, you might want to warn;
            # assuming rcp.extract_peristim_segments keeps T consistent.
            segs_all[:n_d, d, :T_d] = segs

    n_trials_used = int(max(kept_counts)) if kept_counts else 0
    return lines_arr, rel_t_out, n_trials_used, segs_all

# ---------- Behavior (already mapped to common grid in the NPZ) ----------
def _ordered_xy_indices(cam_cols: List[str],
                        keypoints: Tuple[str, ...] = KEYPOINTS_ORDER
                        ) -> Tuple[List[int], List[str]]:
    idx: List[int] = []
    names: List[str] = []
    cols_lc = [c.lower() for c in cam_cols]

    def _clean_kp(s: str) -> str:
        s = str(s).strip()
        s = s.strip("[]").strip("'\"")   # handle "'Wrist'" → Wrist
        return s.lower()

    def _find_xy_for(kp: str) -> Tuple[Optional[int], Optional[int]]:
        kp_lc = _clean_kp(kp).lower()
        ix = next((i for i, c in enumerate(cols_lc) if c.endswith("_x") and kp_lc in c), None)
        iy = next((i for i, c in enumerate(cols_lc) if c.endswith("_y") and kp_lc in c), None)
        return ix, iy

    for kp in keypoints:
        ix, iy = _find_xy_for(kp)
        base = _clean_kp(kp)
        if ix is None or iy is None:
            idx.extend([-1, -1])
            names.extend([f"{base}_x(MISSING)", f"{base}_y(MISSING)"])
        else:
            idx.extend([ix, iy])
            names.extend([f"{base}_x", f"{base}_y"])
    return idx, names

def _select_matrix(cam: np.ndarray, cols: List[str],
                   keypoints: Tuple[str, ...] = KEYPOINTS_ORDER
                   ) -> Tuple[np.ndarray, List[str]]:
    idx, names = _ordered_xy_indices(cols or [], keypoints=keypoints)

    if cam is None or cam.size == 0:
        M = np.zeros((0, len(idx)), float)
        return M, names

    if cam.ndim == 1:
        cam = cam[:, None]

    # Guard against column index overflow if CSV is malformed
    n_rows, n_cols = cam.shape
    M = np.empty((n_rows, len(idx)), float)
    for k, j in enumerate(idx):
        if j == -1 or j >= n_cols:
            M[:, k] = np.nan
        else:
            M[:, k] = cam[:, j]
    return M, names

def _z_per_column(M: np.ndarray) -> np.ndarray:
    if M is None or M.size == 0:
        return np.asarray(M, dtype=float)
    if M.ndim == 1:
        M = M[:, None]
    out = M.astype(float, copy=True)
    if out.shape[1] == 0:
        return out
    for j in range(out.shape[1]):
        col = out[:, j]
        m = np.nanmean(col) if np.isfinite(col).any() else np.nan
        s = np.nanstd(col)  if np.isfinite(col).any() else np.nan
        if np.isfinite(s) and s > 0:
            out[:, j] = (col - m) / s
        else:
            out[:, j] = np.full_like(col, np.nan)
    return out

# delay parse
def _parse_delay_ms(val) -> int:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0
    s = str(val).strip()
    if not s or s.casefold() == "random":
        return 0
    try:
        return int(float(s))
    except Exception:
        return 0

def _fmt_num(x) -> str:
    if pd.isna(x):
        return "n/a"
    try:
        v = float(x); return f"{int(v)}" if v.is_integer() else f"{v:g}"
    except Exception:
        s = str(x).strip(); return s if s else "n/a"

def _interp_nans_1d_gentle(x: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """Fill only interior NaN runs up to max_gap by linear interp. 
    Leading/trailing NaNs stay NaN; large gaps stay NaN."""
    x = np.asarray(x, float)
    if x.ndim != 1 or x.size == 0:
        return x
    out = x.copy()
    isn = ~np.isfinite(out)
    if not isn.any():
        return out

    starts = np.where(isn & ~np.r_[False, isn[:-1]])[0]
    ends   = np.where(isn & ~np.r_[isn[1:],  False])[0]
    for s, e in zip(starts, ends):
        gap = e - s + 1
        left, right = s - 1, e + 1
        if (gap <= max_gap and left >= 0 and right < out.size 
                and np.isfinite(out[left]) and np.isfinite(out[right])):
            out[s:e+1] = np.interp(np.arange(s, e+1), [left, right], [out[left], out[right]])
    return out

def _interp_nans_2d_by_col(arr: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """Apply _interp_nans_1d_gentle to each column of a 2D array."""
    arr = np.asarray(arr, float)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    out = arr.copy()
    for j in range(out.shape[1]):
        out[:, j] = _interp_nans_1d_gentle(out[:, j], max_gap=max_gap)
    return out

def _read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Try several common encodings and tolerate a few bad rows.
    """
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1", "mac_roman")
    last_err = None
    for enc in encodings:
        try:
            # engine='python' + on_bad_lines='skip' is more forgiving with odd quotes
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
        except Exception as e:
            last_err = e
            continue
    # final fallback: read bytes → decode as latin-1 (never fails) → StringIO
    try:
        from io import StringIO
        txt = Path(path).read_bytes().decode("latin-1", errors="ignore")
        return pd.read_csv(StringIO(txt), engine="python", on_bad_lines="skip")
    except Exception:
        raise last_err or RuntimeError(f"Could not read CSV: {path}")

def build_title_from_csv(
    csv_path: Path, *, sess: Optional[str] = None, br_file: Optional[int] = None
) -> Tuple[str, Optional[int]]:
    df_raw = _read_csv_robust(csv_path)

    # normalize column names: lower, strip, spaces->underscores
    df = df_raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # drop the first row if it's a header-like row (your original iloc[1:])
    df_data = df.iloc[1:].reset_index(drop=True) if len(df) > 1 else df.copy()
    if df_data.empty:
        return "Condition: n/a, n/a Hz, n/a µA, n/a mm, n/a ms, Delay: 0 ms", None

    def col(name: str) -> Optional[str]:
        n = name.lower()
        return n if n in df_data.columns else None

    # columns we might use
    c_session   = col("session")
    c_brfile    = col("br_file")
    c_uaport    = col("ua_port")
    c_trigger   = col("movement_trigger")
    c_freq      = col("stim_frequency_hz")
    c_current   = col("current_ua")
    c_depth     = col("depth_mm")
    c_duration  = col("stim_duration_ms")
    c_delay     = col("delay")

    # ---- build masks stepwise (recording each stage) ----
    mask = pd.Series(True, index=df_data.index)

    # prefer matching BR_File exactly (if provided and column exists)
    if br_file is not None and c_brfile:
        mask_br = (pd.to_numeric(df_data[c_brfile], errors="coerce") == int(br_file))
        if mask_br.any():
            mask &= mask_br

    # then try session (substring match)
    if c_session and sess:
        m_sess = df_data[c_session].astype(str).str.contains(str(sess), regex=False, na=False)
        if (mask & m_sess).any():
            mask &= m_sess

    # prefer UA_port == 'A' if it doesn’t eliminate all rows
    if c_uaport:
        m_portA = df_data[c_uaport].astype(str).str.upper().str.strip().eq("A")
        if (mask & m_portA).any():
            mask &= m_portA

    # avoid Movement_Trigger == 'velocity' when possible
    if c_trigger:
        m_not_vel = ~df_data[c_trigger].astype(str).str.strip().str.casefold().eq("velocity")
        if (mask & m_not_vel).any():
            mask &= m_not_vel

    # if still empty, relax progressively (order: remove trigger, then port, then session, then BR)
    if not mask.any():
        mask = pd.Series(True, index=df_data.index)
        if br_file is not None and c_brfile:
            mask_br = (pd.to_numeric(df_data[c_brfile], errors="coerce") == int(br_file))
            mask &= mask_br if mask_br.any() else True
        if c_session and sess:
            m_sess = df_data[c_session].astype(str).str.contains(str(sess), regex=False, na=False)
            mask &= m_sess if m_sess.any() else True
        if c_uaport:
            m_portA = df_data[c_uaport].astype(str).str.upper().str.strip().eq("A")
            mask &= m_portA if m_portA.any() else True
        if c_trigger:
            m_not_vel = ~df_data[c_trigger].astype(str).str.strip().str.casefold().eq("velocity")
            mask &= m_not_vel if m_not_vel.any() else True
        if not mask.any():
            mask = pd.Series(True, index=df_data.index)  # final fallback: any row

    row = df_data.loc[mask].iloc[0]

    delay_ms = _parse_delay_ms(row.get(c_delay, None) if c_delay else None)

    parts = []
    if br_file is not None:
        parts.append(f"Condition: {_fmt_num(br_file)}")
    parts.extend([
        f"Freq: {_fmt_num(row.get(c_freq))} Hz"      if c_freq else "Freq: n/a Hz",
        f"Current: {_fmt_num(row.get(c_current))} µA" if c_current else "Current: n/a µA",
        f"Depth: {_fmt_num(row.get(c_depth))} mm"     if c_depth else "Depth: n/a mm",
        f"Duration: {_fmt_num(row.get(c_duration))} ms" if c_duration else "Duration: n/a ms",
        f"Delay: {delay_ms} ms",
    ])
    if c_uaport:
        parts.append(f"UA Port: {_fmt_num(row.get(c_uaport))}")

    # ---- robust video_file extraction ----
    # try multiple candidate columns and parse the first integer
    video_cols = [c for c in ("video_file", "video", "video_index", "video#", "vid", "videoid") if c in df_data.columns]
    video_file: Optional[int] = None
    for c in video_cols:
        val = row.get(c)
        if pd.isna(val):
            continue
        # numeric first
        num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
        if pd.notna(num):
            video_file = int(num)
            break
        # otherwise, pull the first integer in the string
        m = re.search(r"\d+", str(val))
        if m:
            video_file = int(m.group(0))
            break

    return ", ".join(parts), video_file

def _compute_trial_labels_from_ts_state(
    stim_ms: np.ndarray,
    UA_t: np.ndarray,
    ts_state_num_full: np.ndarray,
    ts_state_char_arr: Optional[np.ndarray],
    win_ms: tuple[float, float] = WIN_MS,
) -> np.ndarray:
    """
    For each stim, look at ts_state in [stim+win0, stim+win1] on UA timebase
    and assign majority label 'A', 'B', or 'N' (other/none).
    Returns: labels : (n_stims,) array of 'A'/'B'/'N'.
    """
    stim_ms = np.asarray(stim_ms, float).ravel()
    UA_t = np.asarray(UA_t, float).ravel()
    ts_state_num_full = np.asarray(ts_state_num_full, int).ravel()

    if stim_ms.size == 0 or UA_t.size == 0 or ts_state_num_full.size == 0:
        return np.full(stim_ms.shape, "N", dtype="U1")

    # mapping from state number -> char
    mapping: dict[int, str] = {}
    if ts_state_char_arr is not None:
        arr = np.asarray(ts_state_char_arr)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        n_min = min(arr.size, ts_state_num_full.size)
        for n, c in zip(ts_state_num_full[:n_min], arr[:n_min]):
            mapping[int(n)] = str(c)[0]
    mapping_default = " "

    labels = np.full(stim_ms.shape, "N", dtype="U1")
    w0, w1 = float(win_ms[0]), float(win_ms[1])

    for i, s in enumerate(stim_ms):
        lo = s + w0
        hi = s + w1
        mask = (UA_t >= lo) & (UA_t <= hi)
        if not mask.any():
            continue
        nums = ts_state_num_full[mask]
        if nums.size == 0:
            continue
        chars = np.array([mapping.get(int(n), mapping_default) for n in nums], dtype="U1")
        chars = chars[chars != " "]
        if chars.size == 0:
            continue
        vals, counts = np.unique(chars, return_counts=True)
        majority = vals[np.argmax(counts)]
        if majority in ("A", "B"):
            labels[i] = majority
        else:
            labels[i] = "N"
    return labels

def _subset_neural_from_zeroed(
    zeroed: Optional[np.ndarray],
    counts: Optional[np.ndarray],
    labels_stream: Optional[np.ndarray],
    target_label: str,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Given baseline-zeroed rate segments (n_events, n_channels, T),
    optional baseline-zeroed count segments with the same shape,
    and per-event labels on this stream, compute median+var for
    events with label == target_label.

    Returns:
        med          : (n_channels, T) median across selected events
        var          : (n_channels, T) variance across selected events
        n_events_lbl : int, number of selected events
        zeroed_sub   : (n_events_lbl, n_channels, T) subset of `zeroed`
        counts_sub   : (n_events_lbl, n_channels, T) subset of `counts`
                       (empty if counts is None or shape-mismatched)
    """
    empty_med = np.zeros((0, 0), float)
    empty_zeroed = np.zeros((0, 0, 0), float)

    if zeroed is None or zeroed.size == 0 or labels_stream is None:
        return empty_med, empty_med, 0, empty_zeroed, empty_zeroed

    zeroed = np.asarray(zeroed, float)
    labels_stream = np.asarray(labels_stream)

    if zeroed.shape[0] != labels_stream.size:
        # shape mismatch; bail gracefully
        return empty_med, empty_med, 0, empty_zeroed, empty_zeroed

    # Prepare counts (optional, must match zeroed shape)
    if counts is not None and np.size(counts) != 0:
        counts_arr = np.asarray(counts)
        if counts_arr.shape != zeroed.shape:
            counts_arr = None
    else:
        counts_arr = None

    idx = np.where(labels_stream == target_label)[0]
    if idx.size == 0:
        # med/var shapes follow zeroed's channel/time dims
        med = np.zeros((zeroed.shape[1], zeroed.shape[2]), float)
        var = np.zeros((zeroed.shape[1], zeroed.shape[2]), float)
        zeroed_sub = np.zeros((0, zeroed.shape[1], zeroed.shape[2]), float)
        counts_sub = np.zeros_like(zeroed_sub)
        return med, var, 0, zeroed_sub, counts_sub

    sub = zeroed[idx]  # (n_label, n_ch, T)

    if counts_arr is not None:
        counts_sub = counts_arr[idx]
    else:
        counts_sub = np.zeros_like(sub)

    # coverage mask from rates
    valid_counts = np.isfinite(sub).sum(axis=0)  # (n_ch, T)
    n_events = sub.shape[0]
    min_bin_trials = max(1, int(0.3 * n_events))
    min_bin_trials = min(min_bin_trials, n_events)

    med = np.nanmedian(sub, axis=0)
    var = np.nanvar(sub, axis=0)

    med[valid_counts < min_bin_trials] = np.nan
    var[valid_counts < min_bin_trials] = np.nan

    return med, var, int(n_events), sub, counts_sub

def _behavior_medians_for_label(
    cam_z: np.ndarray,
    behv_t: np.ndarray,
    stim_ms: np.ndarray,
    labels: np.ndarray,
    target_label: str,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Recompute behavior medians using only stims with labels==target_label.
    """
    if cam_z is None or cam_z.size == 0 or stim_ms.size == 0:
        return (np.zeros((0, 0), float),
                np.zeros(0, float),
                0,
                np.zeros((0, 0, 0), float))
    mask = (labels == target_label)
    if not mask.any():
        return (np.zeros((cam_z.shape[1] if cam_z.ndim == 2 else 0, 0), float),
                np.zeros(0, float),
                0,
                np.zeros((0, 0, 0), float))

    stim_label = stim_ms[mask]
    lines, rel_t, n_trials, segs = _median_lines_for_columns(cam_z, behv_t, stim_label)
    return lines, rel_t, n_trials, segs

def _behavior_valid_mask_from_cam0(
    cam0_z: np.ndarray,
    behv_t: np.ndarray,
    stim_ms: np.ndarray,
    win_ms: Tuple[float, float] = WIN_MS,
    baseline_ms: float = NORMALIZE_FIRST_MS,
    min_trials: int = MIN_TRIALS,
) -> np.ndarray:
    """
    Decide which stim events have 'good' behavior traces based on cam0_z.

    Steps:
      1. Use RCP_analysis.extract_peristim_segments to find the subset of stims
         that produce valid, full-length windows on the behavior timebase.
      2. Among those, keep only events whose baseline window has at least one finite value.
      3. Map that decision back to a boolean mask over the original stim_ms array.

    Returns:
      mask : (n_stims,) bool, True where behavior is usable.
    """
    stim_all = np.asarray(stim_ms, float).ravel()
    if stim_all.size == 0:
        return np.zeros(0, dtype=bool)

    if cam0_z is None or cam0_z.size == 0 or behv_t.size < 2:
        # no behavior info to gate with → keep all stims
        return np.ones(stim_all.shape, dtype=bool)

    cam0_z = np.asarray(cam0_z, float)
    if cam0_z.ndim == 2:
        series = cam0_z[:, 0]  # first column as proxy
    elif cam0_z.ndim == 1:
        series = cam0_z
    else:
        return np.ones(stim_all.shape, dtype=bool)

    try:
        rate_segs, _, rel_t, stim_valid = rcp.extract_peristim_segments(
            rate_hz=series[None, :],   # (1, T)
            counts=None,
            t_ms=behv_t,
            stim_ms=stim_all,
            win_ms=win_ms,
            min_trials=min_trials,
        )
    except RuntimeError as e:
        if "Only 0 peri-stim segments" in str(e):
            # nothing extractable on behavior → consider all bad
            return np.zeros(stim_all.shape, dtype=bool)
        raise

    if rate_segs.size == 0 or stim_valid.size == 0:
        return np.zeros(stim_all.shape, dtype=bool)

    # rate_segs: (n_events_valid, 1, T) → (n_events_valid, T)
    segs = rate_segs[:, 0, :]

    # baseline window on the relative time axis
    bl_mask = (rel_t >= rel_t[0]) & (rel_t <= baseline_ms)
    if not bl_mask.any():
        step = max(
            1,
            int(round(baseline_ms / (np.nanmedian(np.diff(rel_t)) if rel_t.size > 1 else 1.0))),
        )
        bl_mask = np.zeros_like(rel_t, bool)
        bl_mask[:step] = True

    # For each valid stim, require at least one finite baseline sample
    keep_valid = np.isfinite(segs[:, bl_mask]).any(axis=1)

    # Now map back to a mask over the original stim_ms
    mask_beh = np.zeros(stim_all.shape, dtype=bool)
    stim_kept = stim_valid[keep_valid]

    for s in np.asarray(stim_kept, float):
        idx = np.where(np.isclose(stim_all, s, rtol=0.0, atol=1e-6))[0]
        if idx.size:
            # If duplicates ever existed, this marks the first occurrence.
            mask_beh[idx[0]] = True

    return mask_beh

# ---------------------------------------------------------------------
# IR event loader
# --------------------------------------------------------------------- 
def _detect_IR_crossings(x: np.ndarray, fs: float | None, refractory_sec: float = 0.0005) -> np.ndarray:
    """
    Binarize at mid-point and return sample indices of 1→0 edges.
    Debounce via refractory (default 0.5 ms).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([], dtype=np.int64)

    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    thr = 0.5 * (lo + hi)
    b = (x > thr).astype(np.int8)

    # edges where b[t-1]=1 and b[t]=0
    db = np.diff(b, prepend=b[0])
    edges = np.flatnonzero(db == -1)

    if edges.size == 0:
        return edges.astype(np.int64)

    if fs and fs > 0:
        refr = max(1, int(round(refractory_sec * fs)))
    else:
        refr = 1

    keep = []
    last = -10**12
    for i in edges:
        if i - last >= refr:
            keep.append(i)
            last = i
    return np.asarray(keep, dtype=np.int64)

def _load_ir_ms_from_aligned(aligned_path: Path, meta: Dict[str, Any]) -> np.ndarray:
    """
    Compute IR event times (in BR-aligned ms) for this aligned bundle.

    This mirrors the original baseline script logic:
      - read the Intan IR stream for the matching Intan session
      - detect IR 1→0 edges
      - convert sample indices to seconds, then to BR-aligned ms
        using anchor_ms from br_to_intan_shifts.csv.

    Returns:
        ir_ms_br : np.ndarray of shape (N,) in *BR ms* timebase,
                   suitable to use directly as event centers with
                   NPRW_t, UA_t, beh_t_ms, vog_t_ms from the combined bundle.
    """
    br_idx = int(meta.get("br_idx", -1))
    if br_idx < 0:
        print(f"[baseline-IR] {aligned_path.name}: meta has no valid br_idx; skipping IR.")
        return np.array([], float)

    shift_entry = BR_TO_INTAN_SHIFTS.get(br_idx)
    if shift_entry is None:
        print(f"[baseline-IR] BR {br_idx}: no entry in br_to_intan_shifts; skipping IR.")
        return np.array([], float)

    intan_session = shift_entry.get("session")
    shift_ms = float(shift_entry.get("anchor_ms", 0.0))
    if not intan_session:
        print(f"[baseline-IR] BR {br_idx}: empty Intan session; skipping IR.")
        return np.array([], float)

    intan_dir = INTAN_ROOT / intan_session
    try:
        rec_ir = se.read_split_intan_files(
            intan_dir,
            mode="concatenate",
            stream_name=IR_STREAM,
            use_names_as_ids=True,
        )
        rec_ir = spre.unsigned_to_signed(rec_ir)  # UInt16 -> int16
    except Exception as e:
        print(f"[baseline-IR] Reading IR stream failed for Intan={intan_session}: {e}")
        return np.array([], float)

    fs = float(rec_ir.sampling_frequency)
    sig = np.asarray(rec_ir.get_traces()).squeeze()
    if sig is None or sig.size == 0:
        print(f"[baseline-IR] Intan={intan_session}: empty IR signal.")
        return np.array([], float)

    ir_idx = _detect_IR_crossings(sig, fs, refractory_sec=0.0005)
    if ir_idx.size == 0:
        print(f"[baseline-IR] Intan={intan_session}: no IR crossings.")
        return np.array([], float)

    # Intan seconds -> BR-aligned ms (same as original script: subtract anchor_ms)
    ir_sec = ir_idx / fs
    ir_ms_br = ir_sec * 1000.0 - shift_ms

    print(
        f"[baseline-IR] {aligned_path.name}: BR={br_idx}, Intan={intan_session}, "
        f"shift_ms={shift_ms:g}, n_IR={ir_ms_br.size}"
    )
    return ir_ms_br.astype(float)

# ---------------------------------------------------------------------
# Per-file baseline extraction (IR-based), NO saving, only returns data.
# This mirrors extract_one_file(...) in your peristim script but:
#   - uses ir_ms instead of stim_ms
#   - does not write output; just returns a dict to the caller.
# ---------------------------------------------------------------------
def extract_baseline_from_file(
    aligned_path: Path,
    ir_ms: np.ndarray,
) -> Dict[str, Any]:
    """
    Return a dictionary with all per-file data needed for later concatenation.

    The structure is very close to what your peristim extract_one_file builds,
    but we keep A and B pieces in memory instead of saving immediately.
    """
    # ---- load combined (NPRW/UA + stim + meta) ----
    aligned_npz = np.load(aligned_path, allow_pickle=True)
    NPRW_rate = aligned_npz["nprw_rate_hz"]
    NPRW_counts = aligned_npz["nprw_counts"]
    NPRW_t    = aligned_npz["nprw_t_ms_aligned"] if "nprw_t_ms_aligned" in aligned_npz.files else aligned_npz["nprw_t_ms"]
    UA_rate = aligned_npz["ua_rate_hz"]
    UA_counts = aligned_npz["ua_counts"]
    UA_t    = aligned_npz["ua_t_ms_aligned"] if "ua_t_ms_aligned" in aligned_npz.files else aligned_npz["ua_t_ms"]
    # UA ids
    ua_elec_1based = aligned_npz["ua_elec"]
    if ua_elec_1based is not None:
        ua_elec_1based = np.asarray(ua_elec_1based, dtype=int).ravel()
    
    ua_region=aligned_npz["ua_region"]
    ua_region_names=aligned_npz["ua_region_names"]
    ua_port=aligned_npz["ua_port"]
    ua_nsp=aligned_npz["ua_nsp"]
    ua_idx_rows=aligned_npz["ua_idx_rows"]
    
    # ---- behavior / VOG / ts_state from the bundle ----
    beh_cam0 = aligned_npz["beh_cam0"]
    beh_cam1 = aligned_npz["beh_cam1"]
    beh_cam0_cols = aligned_npz["beh_cam0_cols"]
    beh_cam1_cols = aligned_npz["beh_cam1_cols"]
    cam0_cols = [str(x) for x in _as_list(beh_cam0_cols)]
    cam1_cols = [str(x) for x in _as_list(beh_cam1_cols)]
    behv_t = aligned_npz["beh_t_ms"]

    # VOG
    vog_cols = aligned_npz["vog_cols"]
    vog_t_ms = aligned_npz["vog_t_ms"]
    vog_raw_names = aligned_npz["vog_col_names"]
    vog_col_names = [str(x) for x in _as_list(vog_raw_names)]

    # ts_state on UA timebase
    ts_state_num = aligned_npz["ts_state_num"]
    ts_state_char = aligned_npz["ts_state_char"]
        
    # alignment meta (JSON)
    meta = json.loads(aligned_npz["align_meta"].item()) if "align_meta" in aligned_npz.files else {}
    sess = meta.get("session", aligned_path.stem)
    br_idx = int(meta.get("br_idx", -1))

    # IR times in the same aligned ms frame as NPRW_t / UA_t / beh_t_ms / vog_t_ms
    ir_ms = np.asarray(ir_ms, float).ravel()

    # normalize ts_state arrays
    ts_state_num_full = None
    ts_state_char_arr = None
    if ts_state_num is not None and np.size(ts_state_num):
        ts_state_num_full = np.asarray(ts_state_num).astype(int).ravel()
    if ts_state_char is not None:
        ts_state_char_arr = np.asarray(ts_state_char)
        if ts_state_char_arr.ndim > 1:
            ts_state_char_arr = ts_state_char_arr.reshape(-1)

    print(
        f"[baseline-debug] {aligned_path.name}: "
        f"IR_n={ir_ms.size}, "
        f"vog_t_end={_end_or_nan(vog_t_ms):.3f} ms, "
        f"behv_t_end={_end_or_nan(behv_t):.3f} ms, "
        f"NPRW_t_end={_end_or_nan(NPRW_t):.3f} ms, "
        f"UA_t_end={_end_or_nan(UA_t):.3f} ms"
    )

    # ---- behavior Preprocessing (same as peristim) ----
    if behv_t.size == 0:
        print(f"[baseline-extract] {aligned_path.name}: no behavior time axis.")

    if beh_cam0.size:
        beh_cam0 = _interp_nans_2d_by_col(beh_cam0, max_gap=4)
    if beh_cam1.size:
        beh_cam1 = _interp_nans_2d_by_col(beh_cam1, max_gap=4)

    cam0_M, cam0_names = _select_matrix(beh_cam0, cam0_cols)
    cam1_M, cam1_names = _select_matrix(beh_cam1, cam1_cols)

    cam0_z = _z_per_column(cam0_M) if cam0_M.size else cam0_M
    cam1_z = _z_per_column(cam1_M) if cam1_M.size else cam1_M

    ir_ms = np.asarray(ir_ms, float).ravel()

    # ---- behavior-based IR gating (analogous to stim gating) ----
    if ir_ms.size and behv_t.size and cam0_z.size:
        beh_mask = _behavior_valid_mask_from_cam0(
            cam0_z,
            behv_t,
            ir_ms,
            win_ms=WIN_MS,
            baseline_ms=NORMALIZE_FIRST_MS,
            min_trials=MIN_TRIALS,
        )
        n_keep = int(beh_mask.sum())
        if n_keep == 0:
            print(
                f"[baseline-extract] {aligned_path.name}: behavior gating would remove all IRs; "
                "keeping original IR list."
            )
        elif n_keep < ir_ms.size:
            print(
                f"[baseline-extract] {aligned_path.name}: behavior gating kept "
                f"{n_keep}/{ir_ms.size} IRs."
            )
            ir_ms = ir_ms[beh_mask]

    # if no IR left, bail with empty payload
    if ir_ms.size == 0:
        print(f"[baseline-extract] {aligned_path.name}: no usable IR events after gating.")
        return {
            "sess": sess,
            "br_idx": br_idx,
            "meta": meta,
            "ua_ids_1based": ua_elec_1based,
            "ir_ms": np.array([], float),
            "has_data": False,
        }

    # pos+vel (same as peristim)
    cam0_pos_filt, cam0_vel, _ = rcp.butter_lowpass_pos_and_vel(
        cam0_z, behv_t, cutoff_hz=10.0, order=3
    )
    cam1_pos_filt, cam1_vel, _ = rcp.butter_lowpass_pos_and_vel(
        cam1_z, behv_t, cutoff_hz=10.0, order=3
    )

    # ---- compute A/B labels from ts_state, but using IR events ----
    if ts_state_num_full is not None and np.size(UA_t) and ir_ms.size:
        trial_labels = _compute_trial_labels_from_ts_state(
            stim_ms=ir_ms,  # reusing arg name, but these are IR centers
            UA_t=UA_t,
            ts_state_num_full=ts_state_num_full,
            ts_state_char_arr=ts_state_char_arr,
            win_ms=WIN_MS,
        )
    else:
        trial_labels = np.full(ir_ms.shape, "N", dtype="U1")

    # ---- Behavior medians & segments (A/B) ----
    beh_cam0_pos_med_A, beh_rel_t_A, n_beh_A, beh_cam0_segs_A = _behavior_medians_for_label(
        cam0_z, behv_t, ir_ms, trial_labels, "A"
    )
    beh_cam0_pos_med_B, beh_rel_t_B, n_beh_B, beh_cam0_segs_B = _behavior_medians_for_label(
        cam0_z, behv_t, ir_ms, trial_labels, "B"
    )

    beh_cam1_pos_med_A, _, _, beh_cam1_segs_A = _behavior_medians_for_label(
        cam1_z, behv_t, ir_ms, trial_labels, "A"
    )
    beh_cam1_pos_med_B, _, _, beh_cam1_segs_B = _behavior_medians_for_label(
        cam1_z, behv_t, ir_ms, trial_labels, "B"
    )

    beh_cam0_vel_med_A, _, _, _ = _behavior_medians_for_label(
        cam0_vel, behv_t, ir_ms, trial_labels, "A"
    )
    beh_cam0_vel_med_B, _, _, _ = _behavior_medians_for_label(
        cam0_vel, behv_t, ir_ms, trial_labels, "B"
    )
    beh_cam1_vel_med_A, _, _, _ = _behavior_medians_for_label(
        cam1_vel, behv_t, ir_ms, trial_labels, "A"
    )
    beh_cam1_vel_med_B, _, _, _ = _behavior_medians_for_label(
        cam1_vel, behv_t, ir_ms, trial_labels, "B"
    )

    # ---- VOG medians (event list is IR-based but we still use stim_ms_abs as origin) ----
    if vog_cols.size and vog_t_ms.size:
        vog_cols = _interp_nans_2d_by_col(vog_cols, max_gap=4)
        vog_z = _z_per_column(vog_cols)

        # we use IR events in absolute ms space; if your IR times are already absolute
        # BR ms, you can pass them directly. If they are relative, adjust accordingly.
        vog_med, vog_rel_t, n_vog_trials, vog_segs = _median_lines_for_columns(
            vog_z, vog_t_ms, ir_ms
        )
    else:
        vog_med = np.zeros((0, 0), float)
        vog_rel_t = np.zeros(0, float)
        vog_segs = np.zeros((0, 0, 0), float)
        n_vog_trials = 0

    # ---- ts_state peri-event segments (numeric + char) ----
    ts_state_rel_t = np.zeros(0, float)
    ts_state_segs = np.zeros((0, 0), float)
    ts_state_char_segs = np.zeros((0, 0), dtype="U1")
    n_ts_state_trls = 0

    if ts_state_num_full is not None and np.size(UA_t):
        try:
            ts_rate_segs, _, ts_state_rel_t, _ = rcp.extract_peristim_segments(
                rate_hz=ts_state_num_full.reshape(1, -1),
                counts=None,
                t_ms=UA_t,
                stim_ms=ir_ms,
                win_ms=WIN_MS,
                min_trials=MIN_TRIALS,
            )
            if ts_rate_segs.size:
                ts_state_segs = ts_rate_segs[:, 0, :]
                n_ts_state_trls = int(ts_state_segs.shape[0])
        except RuntimeError as e:
            if "Only 0 peri-stim segments" not in str(e):
                raise

        if ts_state_segs.size and ts_state_char_arr is not None:
            mapping: Dict[int, str] = {}
            if ts_state_char_arr.size == ts_state_num_full.size:
                for n, c in zip(ts_state_num_full, ts_state_char_arr):
                    n_int = int(n)
                    ch = str(c)[0]
                    if n_int not in mapping:
                        mapping[n_int] = ch
            mapping_default = " "

            n_events, T = ts_state_segs.shape
            ts_state_char_segs = np.empty((n_events, T), dtype="U1")
            for i in range(n_events):
                for j in range(T):
                    n_ij = int(ts_state_segs[i, j])
                    ts_state_char_segs[i, j] = mapping.get(n_ij, mapping_default)

    # ---- ts_state A/B subsets ----
    ts_state_segs_A = ts_state_segs[:0, :]
    ts_state_segs_B = ts_state_segs[:0, :]
    ts_state_char_segs_A = ts_state_char_segs[:0, :]
    ts_state_char_segs_B = ts_state_char_segs[:0, :]
    n_ts_state_trls_A = 0
    n_ts_state_trls_B = 0

    if ts_state_segs.size and ir_ms.size and np.size(UA_t):
        mask_ts = _stims_mask_for_stream(ir_ms, UA_t, WIN_MS)
        if (
            mask_ts.size == trial_labels.size
            and mask_ts.sum() == ts_state_segs.shape[0]
        ):
            labels_ts = trial_labels[mask_ts]

            idx_A = np.where(labels_ts == "A")[0]
            if idx_A.size:
                ts_state_segs_A = ts_state_segs[idx_A]
                if ts_state_char_segs.size and ts_state_char_segs.shape == ts_state_segs.shape:
                    ts_state_char_segs_A = ts_state_char_segs[idx_A]
                n_ts_state_trls_A = int(idx_A.size)

            idx_B = np.where(labels_ts == "B")[0]
            if idx_B.size:
                ts_state_segs_B = ts_state_segs[idx_B]
                if ts_state_char_segs.size and ts_state_char_segs.shape == ts_state_segs.shape:
                    ts_state_char_segs_B = ts_state_char_segs[idx_B]
                n_ts_state_trls_B = int(idx_B.size)
        else:
            print(
                f"[baseline-warn] ts_state mapping mismatch for {aligned_path.name}: "
                f"mask_ts.sum()={mask_ts.sum()}, "
                f"ts_state_segs.shape[0]={ts_state_segs.shape[0]}, "
                f"trial_labels.size={trial_labels.size}"
            )

    # ---- NPRW / UA peri-event segments (baseline-zeroed) ----
    _, _, NPRW_rel_t, _, NPRW_rates_zeroed, NPRW_counts_segs = _safe_extract_segments(
        NPRW_rate, NPRW_counts, NPRW_t, ir_ms, WIN_MS, MIN_TRIALS, NORMALIZE_FIRST_MS
    )
    _, _, UA_rel_t, _, UA_rates_zeroed, UA_counts_segs = _safe_extract_segments(
        UA_rate, UA_counts, UA_t, ir_ms, WIN_MS, MIN_TRIALS, NORMALIZE_FIRST_MS
    )

    # labels on NPRW event axis
    mask_nprw = _stims_mask_for_stream(ir_ms, NPRW_t, WIN_MS)
    labels_nprw = trial_labels[mask_nprw] if trial_labels is not None else None

    # labels on UA event axis
    mask_ua = _stims_mask_for_stream(ir_ms, UA_t, WIN_MS)
    labels_ua = trial_labels[mask_ua] if trial_labels is not None else None

    # A/B subsets for NPRW
    NPRW_med_A, NPRW_var_A, n_nprw_A, NPRW_rates_zeroed_A, NPRW_counts_segs_A = _subset_neural_from_zeroed(
        NPRW_rates_zeroed, NPRW_counts_segs, labels_nprw, "A"
    )
    NPRW_med_B, NPRW_var_B, n_nprw_B, NPRW_rates_zeroed_B, NPRW_counts_segs_B = _subset_neural_from_zeroed(
        NPRW_rates_zeroed, NPRW_counts_segs, labels_nprw, "B"
    )

    # A/B subsets for UA
    UA_med_A, UA_var_A, n_ua_A, UA_rates_zeroed_A, UA_counts_segs_A = _subset_neural_from_zeroed(
        UA_rates_zeroed, UA_counts_segs, labels_ua, "A"
    )
    UA_med_B, UA_var_B, n_ua_B, UA_rates_zeroed_B, UA_counts_segs_B = _subset_neural_from_zeroed(
        UA_rates_zeroed, UA_counts_segs, labels_ua, "B"
    )

    # ---- metadata title (still useful; we’ll later override for grouped baseline) ----
    try:
        overall_title, _ = build_title_from_csv(METADATA_CSV, br_file=br_idx)
    except Exception as e:
        print(f"[baseline-warn] metadata parse failed for BR {br_idx}: {e}")
        overall_title, _ = (
            "Condition: n/a, n/a Hz, n/a µA, n/a mm, n/a ms, Delay: 0 ms",
            None,
        )

    return {
        "sess": sess,
        "br_idx": br_idx,
        "meta": meta,
        "overall_title": overall_title,
        "ir_ms": ir_ms,
        "trial_labels": np.array(trial_labels, dtype="U1"),
        "beh": {
            "cam0_names": np.array(cam0_names, dtype=object),
            "cam1_names": np.array(cam1_names, dtype=object),
            "cam0_pos_med_A": beh_cam0_pos_med_A,
            "cam0_pos_med_B": beh_cam0_pos_med_B,
            "cam1_pos_med_A": beh_cam1_pos_med_A,
            "cam1_pos_med_B": beh_cam1_pos_med_B,
            "cam0_vel_med_A": beh_cam0_vel_med_A,
            "cam0_vel_med_B": beh_cam0_vel_med_B,
            "cam1_vel_med_A": beh_cam1_vel_med_A,
            "cam1_vel_med_B": beh_cam1_vel_med_B,
            "cam0_segs_A": beh_cam0_segs_A,
            "cam0_segs_B": beh_cam0_segs_B,
            "cam1_segs_A": beh_cam1_segs_A,
            "cam1_segs_B": beh_cam1_segs_B,
            "rel_t_A": beh_rel_t_A,
            "rel_t_B": beh_rel_t_B,
            "n_beh_A": int(n_beh_A),
            "n_beh_B": int(n_beh_B),
            "behv_t": behv_t,
        },
        "vog": {
            "vog_rel_t": vog_rel_t,
            "vog_med": vog_med,
            "vog_segs": vog_segs,
            "vog_cols": vog_cols,
            "vog_col_names": np.array(vog_col_names, dtype=object),
            "n_vog_trials": int(n_vog_trials),
        },
        "ts_state": {
            "ts_state_rel_t": ts_state_rel_t,
            "ts_state_segs_A": ts_state_segs_A,
            "ts_state_segs_B": ts_state_segs_B,
            "ts_state_char_segs_A": ts_state_char_segs_A,
            "ts_state_char_segs_B": ts_state_char_segs_B,
            "ts_state_char": ts_state_char,
            "ts_state_num_full": ts_state_num_full
            if ts_state_num is not None
            else np.array([], int),
            "n_ts_state_trials_A": int(n_ts_state_trls_A),
            "n_ts_state_trials_B": int(n_ts_state_trls_B),
        },
        "neural": {
            "NPRW_med_A": NPRW_med_A,
            "NPRW_var_A": NPRW_var_A,
            "NPRW_rates_zeroed_A": NPRW_rates_zeroed_A,
            "NPRW_counts_A": NPRW_counts_segs_A,
            "n_nprw_A": int(n_nprw_A),
            "NPRW_med_B": NPRW_med_B,
            "NPRW_var_B": NPRW_var_B,
            "NPRW_rates_zeroed_B": NPRW_rates_zeroed_B,
            "NPRW_counts_B": NPRW_counts_segs_B,
            "n_nprw_B": int(n_nprw_B),
            "NPRW_rel_t": NPRW_rel_t,

            "UA_med_A": UA_med_A,
            "UA_var_A": UA_var_A,
            "UA_rates_zeroed_A": UA_rates_zeroed_A,
            "UA_counts_A": UA_counts_segs_A,
            "n_ua_A": int(n_ua_A),
            "UA_med_B": UA_med_B,
            "UA_var_B": UA_var_B,
            "UA_rates_zeroed_B": UA_rates_zeroed_B,
            "UA_counts_B": UA_counts_segs_B,
            "n_ua_B": int(n_ua_B),
            "UA_rel_t": UA_rel_t,
        },
        "ua_ids_1based": ua_elec_1based if ua_elec_1based is not None else np.array([], int),
        "ua_region": ua_region,
        "ua_region_names": ua_region_names,
        "ua_port": ua_port,
        "ua_nsp": ua_nsp,
        "ua_idx_rows": ua_idx_rows,
        "has_data": True,
    }

# ---------------------------------------------------------------------
# Grouping & concatenation across baseline trials per UA site
# ---------------------------------------------------------------------
def _normalize_metadata(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _find_baseline_groups(df_norm: pd.DataFrame) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Return mapping (ua_port_norm, depth_mm_norm) -> row indices of baseline rows.
    This mirrors the grouping logic from your baseline IR script.
    """
    if "notes" not in df_norm.columns:
        raise RuntimeError("No 'notes' column in metadata (after normalization).")

    is_baseline = df_norm["notes"].astype(str).str.lower().eq("baseline")
    df_base = df_norm.loc[is_baseline].copy()

    if df_base.empty:
        return {}

    # UA_port
    if "ua_port" in df_base.columns:
        df_base["_ua_port_norm"] = (
            df_base["ua_port"].astype(str).str.upper().str.strip().replace({"": "UNKNOWN"})
        )
    else:
        df_base["_ua_port_norm"] = "UNKNOWN"

    # Depth_mm
    if "depth_mm" in df_base.columns:
        depth_num = pd.to_numeric(df_base["depth_mm"], errors="coerce")
        df_base["_depth_mm_norm"] = depth_num.where(
            depth_num.notna(), df_base["depth_mm"].astype(str)
        ).replace({"": "n/a"})
    else:
        df_base["_depth_mm_norm"] = "n/a"

    groups: Dict[Tuple[str, str], np.ndarray] = {}
    for (port, depth), g in df_base.groupby(["_ua_port_norm", "_depth_mm_norm"], sort=True):
        key = (str(port), str(depth))
        groups[key] = g.index.to_numpy()
    return groups

def main():
    files = sorted(ALIGNED_ROOT.glob("aligned__*.npz"))
    if not files:
        raise SystemExit(f"[error] No combined aligned NPZs found at {ALIGNED_ROOT}")

    if not METADATA_CSV.exists():
        raise SystemExit(f"[error] Metadata CSV not found: {METADATA_CSV}")

    df_meta_raw = _read_csv_robust(METADATA_CSV)
    df_meta = _normalize_metadata(df_meta_raw)

    # find baseline UA_port×Depth groups based on metadata
    groups = _find_baseline_groups(df_meta)
    if not groups:
        raise SystemExit("[baseline] No baseline rows found in metadata.")

    print(f"[baseline] found {len(groups)} UA_port×Depth baseline groups.")

    # Build mapping BR -> (ua_port, depth) for quick lookup
    br_to_group: Dict[int, Tuple[str, str]] = {}
    if "br_file" in df_meta.columns:
        br_col = pd.to_numeric(df_meta["br_file"], errors="coerce")
    else:
        br_col = pd.Series(np.nan, index=df_meta.index)

    for (port, depth), idxs in groups.items():
        for i in idxs:
            br = br_col.iloc[i]
            if pd.isna(br):
                continue
            br_to_group[int(br)] = (port, depth)

    # Accumulators: per (port, depth) and per target A/B
    # Each target entry will store lists of pieces that we later concatenate.
    acc: Dict[Tuple[str, str], Dict[str, Dict[str, List[Any]]]] = {}

    for aligned_path in files:
        # load meta to know which BR this file corresponds to
        aligned_npz = np.load(aligned_path, allow_pickle=True)
        meta = json.loads(aligned_npz["align_meta"].item()) if "align_meta" in aligned_npz.files else {}
        br_idx = int(meta.get("br_idx", -1))

        if br_idx not in br_to_group:
            # not a baseline file
            continue

        group_key = br_to_group[br_idx]
        port, depth = group_key
        print(
            f"[baseline] processing {aligned_path.name} for group UA_port={port}, Depth={depth}, BR={br_idx}"
        )

        ir_ms = _load_ir_ms_from_aligned(aligned_path, meta)
        if ir_ms.size == 0:
            print(f"[baseline] {aligned_path.name}: no IR events found; skipping.")
            continue

        res = extract_baseline_from_file(aligned_path, ir_ms)
        if not res.get("has_data", False):
            continue
        ua_meta = {
            "ua_region":      res["ua_region"],
            "ua_region_names": res["ua_region_names"],
            "ua_port":        res["ua_port"],
            "ua_nsp":         res["ua_nsp"],
            "ua_idx_rows":    res["ua_idx_rows"],
            "ua_ids_1based":  res["ua_ids_1based"],
        }

        # Prepare group accumulator
        if group_key not in acc:
            acc[group_key] = {
                "A": {
                    "beh_cam0_segs": [],
                    "beh_cam1_segs": [],
                    "beh_rel_t": None,
                    "beh_cam0_vel_med_list": [],
                    "beh_cam1_vel_med_list": [],
                    "n_beh": 0,
                    "trial_labels": [],
                    "ir_ms": [],
                    "NPRW_rates_zeroed": [],
                    "NPRW_counts": [],
                    "UA_rates_zeroed": [],
                    "UA_counts": [],
                    "NPRW_rel_t": None,
                    "UA_rel_t": None,
                    "ts_state_segs": [],
                    "ts_state_char_segs": [],
                    "ts_state_rel_t": None,
                    "n_ts_state_trials": 0,
                },
                "B": {
                    "beh_cam0_segs": [],
                    "beh_cam1_segs": [],
                    "beh_rel_t": None,
                    "beh_cam0_vel_med_list": [],
                    "beh_cam1_vel_med_list": [],
                    "n_beh": 0,
                    "trial_labels": [],
                    "ir_ms": [],
                    "NPRW_rates_zeroed": [],
                    "NPRW_counts": [],
                    "UA_rates_zeroed": [],
                    "UA_counts": [],
                    "NPRW_rel_t": None,
                    "UA_rel_t": None,
                    "ts_state_segs": [],
                    "ts_state_char_segs": [],
                    "ts_state_rel_t": None,
                    "n_ts_state_trials": 0,
                },
                "shared": {
                    "sess_list": [],
                    "br_list": [],
                    "overall_title_list": [],
                    "vog_rel_t": None,
                    "vog_med": [],
                    "vog_segs": [],
                    "vog_cols": None,
                    "vog_col_names": None,
                    "n_vog_trials": 0,
                    "cam0_names": res["beh"]["cam0_names"],
                    "cam1_names": res["beh"]["cam1_names"],
                    "ua_ids_1based": res["ua_ids_1based"],
                },
                "ua_meta": ua_meta,
            }

        g = acc[group_key]

        # ---------- accumulate Target A ----------
        beh_A = res["beh"]
        neu = res["neural"]
        ts = res["ts_state"]
        vog = res["vog"]

        # A
        if beh_A["n_beh_A"] > 0 and beh_A["cam0_segs_A"].size:
            gA = g["A"]
            gA["beh_cam0_segs"].append(beh_A["cam0_segs_A"])
            gA["beh_cam1_segs"].append(beh_A["cam1_segs_A"])
            gA["beh_cam0_vel_med_list"].append(beh_A["cam0_vel_med_A"])
            gA["beh_cam1_vel_med_list"].append(beh_A["cam1_vel_med_A"])
            gA["n_beh"] += beh_A["n_beh_A"]
            gA["ir_ms"].append(res["ir_ms"])  # raw IR centers
            gA["trial_labels"].append(res["trial_labels"])

            if gA["beh_rel_t"] is None:
                gA["beh_rel_t"] = beh_A["rel_t_A"]

            # NPRW neural (A)
            if neu["NPRW_rates_zeroed_A"].size:
                gA["NPRW_rates_zeroed"].append(neu["NPRW_rates_zeroed_A"])
                if gA["NPRW_rel_t"] is None:
                    gA["NPRW_rel_t"] = neu["NPRW_rel_t"]
                if "NPRW_counts_A" in neu and neu["NPRW_counts_A"].size:
                    gA["NPRW_counts"].append(neu["NPRW_counts_A"])

            # UA neural (A)
            if neu["UA_rates_zeroed_A"].size:
                gA["UA_rates_zeroed"].append(neu["UA_rates_zeroed_A"])
                if gA["UA_rel_t"] is None:
                    gA["UA_rel_t"] = neu["UA_rel_t"]
                if "UA_counts_A" in neu and neu["UA_counts_A"].size:
                    gA["UA_counts"].append(neu["UA_counts_A"])

            # ts_state
            if ts["ts_state_segs_A"].size:
                gA["ts_state_segs"].append(ts["ts_state_segs_A"])
                gA["ts_state_char_segs"].append(ts["ts_state_char_segs_A"])
                gA["n_ts_state_trials"] += ts["n_ts_state_trials_A"]
                if gA["ts_state_rel_t"] is None:
                    gA["ts_state_rel_t"] = ts["ts_state_rel_t"]

        # B
        if beh_A["n_beh_B"] > 0 and beh_A["cam0_segs_B"].size:
            gB = g["B"]
            gB["beh_cam0_segs"].append(beh_A["cam0_segs_B"])
            gB["beh_cam1_segs"].append(beh_A["cam1_segs_B"])
            gB["beh_cam0_vel_med_list"].append(beh_A["cam0_vel_med_B"])
            gB["beh_cam1_vel_med_list"].append(beh_A["cam1_vel_med_B"])
            gB["n_beh"] += beh_A["n_beh_B"]
            gB["ir_ms"].append(res["ir_ms"])
            gB["trial_labels"].append(res["trial_labels"])
            
            if gB["beh_rel_t"] is None:
                gB["beh_rel_t"] = beh_A["rel_t_B"]

            if neu["NPRW_rates_zeroed_B"].size:
                gB["NPRW_rates_zeroed"].append(neu["NPRW_rates_zeroed_B"])
                if gB["NPRW_rel_t"] is None:
                    gB["NPRW_rel_t"] = neu["NPRW_rel_t"]
                if "NPRW_counts_B" in neu and neu["NPRW_counts_B"].size:
                    gB["NPRW_counts"].append(neu["NPRW_counts_B"])
                    
            if neu["UA_rates_zeroed_B"].size:
                gB["UA_rates_zeroed"].append(neu["UA_rates_zeroed_B"])
                if gB["UA_rel_t"] is None:
                    gB["UA_rel_t"] = neu["UA_rel_t"]
                if "UA_counts_B" in neu and neu["UA_counts_B"].size:
                    gB["UA_counts"].append(neu["UA_counts_B"])

            if ts["ts_state_segs_B"].size:
                gB["ts_state_segs"].append(ts["ts_state_segs_B"])
                gB["ts_state_char_segs"].append(ts["ts_state_char_segs_B"])
                gB["n_ts_state_trials"] += ts["n_ts_state_trials_B"]
                if gB["ts_state_rel_t"] is None:
                    gB["ts_state_rel_t"] = ts["ts_state_rel_t"]

        # shared (VOG / metadata)
        gs = g["shared"]
        gs["sess_list"].append(res["sess"])
        gs["br_list"].append(res["br_idx"])
        gs["overall_title_list"].append(res["overall_title"])

        if vog["vog_segs"].size:
            gs["vog_segs"].append(vog["vog_segs"])
            gs["n_vog_trials"] += vog["n_vog_trials"]
            if gs["vog_rel_t"] is None:
                gs["vog_rel_t"] = vog["vog_rel_t"]
            if gs["vog_cols"] is None:
                gs["vog_cols"] = vog["vog_cols"]
                gs["vog_col_names"] = vog["vog_col_names"]

    # -----------------------------------------------------------------
    # Finalize each group: concatenate across files and save NPZ/MAT
    # -----------------------------------------------------------------
    for (port, depth), g in acc.items():
        for target_label, tag_name in [("A", "Target A"), ("B", "Target B")]:
            G = g[target_label]
            if not G["beh_cam0_segs"] and not G["NPRW_rates_zeroed"]:
                print(
                    f"[baseline] UA_port={port}, Depth={depth}, target {target_label}: no data; skipping."
                )
                continue

            # concat behavior segments
            beh_cam0_segs = (
                np.concatenate(G["beh_cam0_segs"], axis=0)
                if G["beh_cam0_segs"]
                else np.zeros((0, 0, 0), float)
            )
            beh_cam1_segs = (
                np.concatenate(G["beh_cam1_segs"], axis=0)
                if G["beh_cam1_segs"]
                else np.zeros((0, 0, 0), float)
            )
            n_beh = int(beh_cam0_segs.shape[0])
            beh_rel_t = G["beh_rel_t"]
            if beh_rel_t is None:
                beh_rel_t = np.zeros(0, float)

            # aggregated behavior medians (pos)
            if beh_cam0_segs.size:
                beh_cam0_pos_med = np.nanmedian(beh_cam0_segs, axis=0)
            else:
                beh_cam0_pos_med = np.zeros((0, 0), float)
            if beh_cam1_segs.size:
                beh_cam1_pos_med = np.nanmedian(beh_cam1_segs, axis=0)
            else:
                beh_cam1_pos_med = np.zeros((0, 0), float)

            # aggregated behavior medians (vel) – simple mean of per-file medians
            if G["beh_cam0_vel_med_list"]:
                beh_cam0_vel_med = np.nanmedian(
                    np.stack(G["beh_cam0_vel_med_list"], axis=0), axis=0
                )
            else:
                beh_cam0_vel_med = np.zeros_like(beh_cam0_pos_med)
            if G["beh_cam1_vel_med_list"]:
                beh_cam1_vel_med = np.nanmedian(
                    np.stack(G["beh_cam1_vel_med_list"], axis=0), axis=0
                )
            else:
                beh_cam1_vel_med = np.zeros_like(beh_cam1_pos_med)

            # concat neural segments
            NPRW_rates_zeroed = (
                np.concatenate(G["NPRW_rates_zeroed"], axis=0)
                if G["NPRW_rates_zeroed"]
                else np.zeros((0, 0, 0), float)
            )
            UA_rates_zeroed = (
                np.concatenate(G["UA_rates_zeroed"], axis=0)
                if G["UA_rates_zeroed"]
                else np.zeros((0, 0, 0), float)
            )
            NPRW_counts = (
                np.concatenate(G["NPRW_counts"], axis=0)
                if G["NPRW_counts"]
                else np.zeros((0, 0, 0), float)
            )
            UA_counts = (
                np.concatenate(G["UA_counts"], axis=0)
                if G["UA_counts"]
                else np.zeros((0, 0, 0), float)
            )

            n_nprw = int(NPRW_rates_zeroed.shape[0])
            n_ua = int(UA_rates_zeroed.shape[0])

            if NPRW_rates_zeroed.size:
                NPRW_med = np.nanmedian(NPRW_rates_zeroed, axis=0)
                NPRW_var = np.nanvar(NPRW_rates_zeroed, axis=0)
            else:
                NPRW_med = np.zeros((0, 0), float)
                NPRW_var = np.zeros((0, 0), float)

            if UA_rates_zeroed.size:
                UA_med = np.nanmedian(UA_rates_zeroed, axis=0)
                UA_var = np.nanvar(UA_rates_zeroed, axis=0)
            else:
                UA_med = np.zeros((0, 0), float)
                UA_var = np.zeros((0, 0), float)

            NPRW_rel_t = G["NPRW_rel_t"] if G["NPRW_rel_t"] is not None else np.zeros(0)
            UA_rel_t = G["UA_rel_t"] if G["UA_rel_t"] is not None else np.zeros(0)

            # concat ts_state
            ts_segs = (
                np.concatenate(G["ts_state_segs"], axis=0)
                if G["ts_state_segs"]
                else np.zeros((0, 0), float)
            )
            ts_char_segs = (
                np.concatenate(G["ts_state_char_segs"], axis=0)
                if G["ts_state_char_segs"]
                else np.zeros((0, 0), dtype="U1")
            )
            n_ts_state_trials = int(ts_segs.shape[0])
            ts_state_rel_t = (
                G["ts_state_rel_t"] if G["ts_state_rel_t"] is not None else np.zeros(0)
            )

            # concat IR event centers and trial_labels
            ir_all = (
                np.concatenate(G["ir_ms"], axis=0)
                if G["ir_ms"]
                else np.zeros(0, float)
            )
            labels_all = (
                np.concatenate(G["trial_labels"], axis=0)
                if G["trial_labels"]
                else np.zeros(0, dtype="U1")
            )

            # shared fields
            gs = g["shared"]
            cam0_names = gs["cam0_names"]
            cam1_names = gs["cam1_names"]
            um = g["ua_meta"]
            ua_region      = um["ua_region"]
            ua_region_names = um["ua_region_names"]
            ua_port        = um["ua_port"]
            ua_nsp         = um["ua_nsp"]
            ua_idx_rows    = um["ua_idx_rows"]
            ua_ids_1based  = um["ua_ids_1based"]

            vog_rel_t = gs["vog_rel_t"] if gs["vog_rel_t"] is not None else np.zeros(0)
            vog_cols = gs["vog_cols"] if gs["vog_cols"] is not None else np.zeros(
                (0, 0), float
            )
            vog_col_names = (
                gs["vog_col_names"]
                if gs["vog_col_names"] is not None
                else np.array([], dtype=object)
            )
            if gs["vog_segs"]:
                vog_segs = np.concatenate(gs["vog_segs"], axis=0)
            else:
                vog_segs = np.zeros((0, 0, 0), float)
            n_vog_trials = int(gs["n_vog_trials"])

            # overall title: combine & label as baseline for this UA site
            overall_title = (
                f"Baseline (UA_port={port}, Depth={depth} mm); "
                f"{len(gs['br_list'])} BR files"
            )

            # For baseline, set sess/br_idx to sentinel values for compatibility
            sess = f"{PARAMS.session}_baseline"
            br_idx = -1

            # --- choose out dir by target: PeriStim / Target_X ---
            target_dir = PERI_ROOT / f"Target_{target_label}"
            target_dir.mkdir(parents=True, exist_ok=True)

            # strip trailing "_baseline" from sess for nicer naming
            sess_core = str(sess)
            if sess_core.endswith("_baseline"):
                sess_core = sess_core[:-len("_baseline")]

            depth_str = _sanitize(str(depth))
            port_str  = _sanitize(str(port))

            # filename base with _target_X appended
            fname_base = f"baseline__{sess_core}_Depth_{depth_str}_port_{port_str}_target_{target_label}"

            out_npz = target_dir / f"{fname_base}.npz"
            out_mat = target_dir / f"{fname_base}.mat"

            # ---- NPZ ----            
            np.savez_compressed(
                out_npz,
                sess=sess,
                br_idx=int(br_idx),
                overall_title=overall_title,
                stim_ms=ir_all,
                meta={},
                n_beh=int(n_beh),
                beh_rel_t=beh_rel_t,
                beh_cam0_pos_med=beh_cam0_pos_med,
                beh_cam1_pos_med=beh_cam1_pos_med,
                beh_cam0_vel_med=beh_cam0_vel_med,
                beh_cam1_vel_med=beh_cam1_vel_med,
                beh_cam0_segs=beh_cam0_segs,
                beh_cam1_segs=beh_cam1_segs,
                beh_cam0_names=cam0_names,
                beh_cam1_names=cam1_names,
                vog_rel_t=vog_rel_t,
                vog_med=np.nanmedian(vog_segs, axis=0)
                        if vog_segs.size else np.zeros((0, 0), float),
                vog_segs=vog_segs,
                vog_cols=vog_cols,
                vog_col_names=vog_col_names,
                n_vog_trials=int(n_vog_trials),
                ts_state_rel_t=ts_state_rel_t,
                ts_state_segs=ts_segs,
                ts_state_char_segs=ts_char_segs,
                ts_state_char=np.array([], dtype=object),
                ts_state_num=np.array([], int),
                n_ts_state_trials=int(n_ts_state_trials),
                trial_labels=labels_all,
                NPRW_med=NPRW_med,
                NPRW_var=NPRW_var,
                NPRW_rates_zeroed=NPRW_rates_zeroed,
                NPRW_counts=NPRW_counts,
                NPRW_rel_t=NPRW_rel_t,
                n_nprw=int(n_nprw),
                UA_med=UA_med,
                UA_var=UA_var,
                UA_rates_zeroed=UA_rates_zeroed,
                UA_counts=UA_counts,
                UA_rel_t=UA_rel_t,
                n_ua=int(n_ua),
                ua_ids_1based=ua_ids_1based,

                ua_region=ua_region,
                ua_region_names=ua_region_names,
                ua_port=ua_port,
                ua_nsp=ua_nsp,
                ua_idx_rows=ua_idx_rows,
            )
            print(f"[baseline] wrote {out_npz}")

            # ---- MAT ----
            mat_dict = {
                "sess": np.array(sess, dtype=object),
                "br_idx": int(br_idx),
                "overall_title": np.array(overall_title, dtype=object),
                "stim_ms": ir_all,
                "n_beh": int(n_beh),
                "beh_rel_t": beh_rel_t,
                "beh_cam0_pos_med": beh_cam0_pos_med,
                "beh_cam1_pos_med": beh_cam1_pos_med,
                "beh_cam0_vel_med": beh_cam0_vel_med,
                "beh_cam1_vel_med": beh_cam1_vel_med,
                "beh_cam0_segs": beh_cam0_segs,
                "beh_cam1_segs": beh_cam1_segs,
                "beh_cam0_names": cam0_names,
                "beh_cam1_names": cam1_names,
                "vog_rel_t": vog_rel_t,
                "vog_med": np.nanmedian(vog_segs, axis=0)
                    if vog_segs.size else np.zeros((0, 0), float),
                "vog_segs": vog_segs,
                "vog_cols": vog_cols,
                "vog_col_names": vog_col_names,
                "n_vog_trials": int(n_vog_trials),
                "ts_state_rel_t": ts_state_rel_t,
                "ts_state_segs": ts_segs,
                "ts_state_char_segs": ts_char_segs,
                "ts_state_char": np.array([], dtype=object),
                "ts_state_num": np.array([], int),
                "n_ts_state_trials": int(n_ts_state_trials),
                "trial_labels": labels_all,
                "NPRW_med": NPRW_med,
                "NPRW_var": NPRW_var,
                "NPRW_rates_zeroed": NPRW_rates_zeroed,
                "NPRW_counts": NPRW_counts,
                "NPRW_rel_t": NPRW_rel_t,
                "n_nprw": int(n_nprw),
                "UA_med": UA_med,
                "UA_var": UA_var,
                "UA_rates_zeroed": UA_rates_zeroed,
                "UA_counts": UA_counts,
                "UA_rel_t": UA_rel_t,
                "n_ua": int(n_ua),
                "ua_ids_1based": ua_ids_1based,

                "ua_region": ua_region,
                "ua_region_names": ua_region_names,
                "ua_port": ua_port,
                "ua_nsp": ua_nsp,
                "ua_idx_rows": ua_idx_rows,
            }
            savemat(out_mat, mat_dict, do_compression=True)
            print(f"[baseline] wrote {out_mat}")

if __name__ == "__main__":
    main()
