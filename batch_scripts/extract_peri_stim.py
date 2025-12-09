from pathlib import Path
import numpy as np, pandas as pd, re, json
from scipy.io import savemat
from typing import List, Tuple, Optional
import RCP_analysis as rcp

# Reuse the same config as your plotting script

REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE  = SESSION_LOC / "results"; OUT_BASE.mkdir(parents=True, exist_ok=True)

KEYPOINTS_ORDER = tuple(PARAMS.kinematics.get("keypoints", []))  # [] if missing

# ---------- checkpoints / inputs ----------
ALIGNED_ROOT = OUT_BASE / "checkpoints" / "Aligned"
METADATA_ROOT = SESSION_LOC / "Metadata"; METADATA_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_CSV  = METADATA_ROOT / f"{Path(PARAMS.session)}_metadata.csv"
PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim"; PERI_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- CONFIG ----------
WIN_MS            = (-600.0, 600.0)
NORMALIZE_FIRST_MS = 150.0
MIN_TRIALS        = 1

MIN_BIN_COVERAGE_FRAC = 0.9  # require at least 50% of trials finite per bin

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

def _safe_extract_segments(rate_hz, t_ms, stim_ms_in, win_ms, min_trials, normalize_first_ms):
    """Filter to in-range stims, then extract without crashing if 0-kept."""
    st = _filter_stims_for_stream(stim_ms_in, t_ms, win_ms)
    if st.size == 0:
        dt = float(np.nanmedian(np.diff(t_ms))) if np.size(t_ms) > 1 else 1.0
        rel_t = np.arange(win_ms[0], win_ms[1] + 1e-9, dt, dtype=float)
        # med=None, var=None, n_trials=0, zeroed=None
        return None, None, rel_t, 0, None

    try:
        segs, rel_t, _ = rcp.extract_peristim_segments(
            rate_hz=rate_hz, t_ms=t_ms, stim_ms=st, win_ms=win_ms, min_trials=min_trials
        )
    except RuntimeError as e:
        if "Only 0 peri-stim segments" in str(e):
            dt = float(np.nanmedian(np.diff(t_ms))) if np.size(t_ms) > 1 else 1.0
            rel_t = np.arange(win_ms[0], win_ms[1] + 1e-9, dt, dtype=float)
            return None, None, rel_t, 0, None
        raise
    
    zeroed = rcp.baseline_zero_each_trial(segs, rel_t, normalize_first_ms=normalize_first_ms)
    
    # drop / down-weight bins with too few valid trials per channel
    # zeroed: (n_events, n_channels, T)
    valid_counts = np.isfinite(zeroed).sum(axis=0)      # (n_channels, T)
    n_events = zeroed.shape[0]

    # require at least MIN_BIN_COVERAGE_FRAC of events, but never more than n_events
    min_bin_trials = max(1, int(np.ceil(MIN_BIN_COVERAGE_FRAC * n_events)))
    min_bin_trials = min(min_bin_trials, n_events)

    med = rcp.median_across_trials(zeroed)              # (n_channels, T)
    var = rcp.variance_across_trials(zeroed)            # (n_channels, T)

    bad_mask = valid_counts < min_bin_trials           # (n_channels, T)

    if bad_mask.any():
        # mask med/var
        med[bad_mask] = np.nan
        var[bad_mask] = np.nan

        # also NaN-out poorly covered bins in the trial stack itself,
        # so later label-specific subsets don't try to use them
        for ch in range(zeroed.shape[1]):
            bad_t = bad_mask[ch]                       # (T,)
            if bad_t.any():
                zeroed[:, ch, bad_t] = np.nan

    return med, var, rel_t, int(segs.shape[0]), zeroed


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
        segs, rel_t, _ = rcp.extract_peristim_segments(
            s[None, :], t_common_ms, stim_ms, win_ms=win_ms, min_trials=min_trials
        )
    except RuntimeError as e:
        if "Only 0 peri-stim segments" in str(e):
            dt = np.nanmedian(np.diff(t_common_ms)) if t_common_ms.size > 1 else 1.0
            rel_t = np.arange(win_ms[0], win_ms[1] + 1e-9, dt, dtype=float)
            return np.full(rel_t.size, np.nan, float), rel_t, 0, np.zeros((0, rel_t.size), float)
        else:
            raise

    if segs.size == 0:
        dt = np.nanmedian(np.diff(t_common_ms)) if t_common_ms.size > 1 else 1.0
        rel_t = np.arange(win_ms[0], win_ms[1] + 1e-9, dt, dtype=float)
        return np.full(rel_t.size, np.nan, float), rel_t, 0, np.zeros((0, rel_t.size), float)

    segs = segs[:, 0, :]  # (n_trials, T)

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
    labels_stream: Optional[np.ndarray],
    target_label: str,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Given baseline-zeroed segments (n_events, n_channels, T) and per-event labels,
    compute median+var for events with label == target_label.
    Returns (med, var, n_events_label, zeroed_subset).
    """
    if zeroed is None or zeroed.size == 0 or labels_stream is None:
        return (np.zeros((0, 0), float),
                np.zeros((0, 0), float),
                0,
                np.zeros((0, 0, 0), float))
    zeroed = np.asarray(zeroed, float)
    labels_stream = np.asarray(labels_stream)
    if zeroed.shape[0] != labels_stream.size:
        # shape mismatch; bail gracefully
        return (np.zeros((0, 0), float),
                np.zeros((0, 0), float),
                0,
                np.zeros((0, 0, 0), float))

    idx = np.where(labels_stream == target_label)[0]
    if idx.size == 0:
        return (np.zeros((zeroed.shape[1], zeroed.shape[2]), float),
                np.zeros((zeroed.shape[1], zeroed.shape[2]), float),
                0,
                np.zeros((0, zeroed.shape[1], zeroed.shape[2]), float))

    sub = zeroed[idx]  # (n_label, n_ch, T)

    # coverage mask within this label
    valid_counts = np.isfinite(sub).sum(axis=0)  # (n_ch, T)
    n_events = sub.shape[0]
    min_bin_trials = max(1, int(np.ceil(MIN_BIN_COVERAGE_FRAC * n_events)))
    min_bin_trials = min(min_bin_trials, n_events)

    med = np.nanmedian(sub, axis=0)  # (n_ch, T)
    var = np.nanvar(sub, axis=0)

    bad_mask = valid_counts < min_bin_trials  # (n_ch, T)

    if bad_mask.any():
        med[bad_mask] = np.nan
        var[bad_mask] = np.nan
        # propagate to the per-trial stack as well
        for ch in range(sub.shape[1]):
            bad_t = bad_mask[ch]
            if bad_t.any():
                sub[:, ch, bad_t] = np.nan

    return med, var, int(n_events), sub


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
        segs, rel_t, stim_valid = rcp.extract_peristim_segments(
            series[None, :],   # (1, T)
            behv_t,
            stim_all,
            win_ms=win_ms,
            min_trials=min_trials,
        )
    except RuntimeError as e:
        if "Only 0 peri-stim segments" in str(e):
            # nothing extractable on behavior → consider all bad
            return np.zeros(stim_all.shape, dtype=bool)
        raise

    if segs.size == 0 or stim_valid.size == 0:
        return np.zeros(stim_all.shape, dtype=bool)

    # segs: (n_events_valid, 1, T) → (n_events_valid, T)
    segs = segs[:, 0, :]

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

def extract_one_file(aligned_path: Path) -> None:
    """
    Load aligned__*.npz, compute peri-stim med/var for NPRW+UA,
    + behavior/VOG/ts_state peri-stim summaries, and save to PERI_ROOT.
    """
    # ---- load combined npz (for NPRW/UA + stim) ----
    aligned_npz = np.load(aligned_path, allow_pickle=True)
    NPRW_rate = aligned_npz["nprw_rate_hz"]
    NPRW_t    = aligned_npz["nprw_t_ms_aligned"]
    UA_rate = aligned_npz["ua_rate_hz"]
    UA_t    = aligned_npz["ua_t_ms_aligned"]
    # stim (absolute Intan ms)
    stim_ms_abs = aligned_npz["stim_ms"]
        # alignment meta (JSON)
    meta = json.loads(aligned_npz["align_meta"].item()) if "align_meta" in aligned_npz.files else {}

    sess   = meta.get("session", aligned_path.stem)
    br_idx = int(meta.get("br_idx", -1))

    # stim times on aligned Intan timebase
    stim_ms = rcp.aligned_stim_ms(stim_ms_abs, meta)

    # ---- load aligned file for behavior / VOG / ts_state ----
    aligned_npz = np.load(aligned_path, allow_pickle=True)
    beh_cam0      = aligned_npz["beh_cam0"]
    beh_cam1      = aligned_npz["beh_cam1"]
    raw0      = aligned_npz["beh_cam0_cols"]
    raw1      = aligned_npz["beh_cam1_cols"]
    beh_cam0_cols = [str(x) for x in _as_list(raw0)]
    beh_cam1_cols = [str(x) for x in _as_list(raw1)]
    behv_t    = aligned_npz["beh_t_ms"].astype(float) if "beh_t_ms" in aligned_npz.files else np.arange(0.0)

    # VOG
    vog_cols      = aligned_npz["vog_cols"]
    vog_t_ms      = aligned_npz["vog_t_ms"]
    vog_raw_names = aligned_npz["vog_col_names"]
    vog_col_names = [str(x) for x in _as_list(vog_raw_names)]

    # ts_state on UA timebase
    ts_state_num  = aligned_npz["ts_state_num"]
    ts_state_char = aligned_npz["ts_state_char"]
 
    # UA ids
        # UA ids
    ua_ids_1based = aligned_npz["ua_elec"]
    if ua_ids_1based is not None:
        ua_ids_1based = np.asarray(ua_ids_1based, dtype=int).ravel()

    # ---- UA meta
    ua_region = aligned_npz["ua_region"]
    ua_region_names = aligned_npz["ua_region_names"]
    ua_port = aligned_npz["ua_port"]
    ua_nsp = aligned_npz["ua_nsp"]
    ua_idx_rows = aligned_npz["ua_idx_rows"]
    
    # prepare ts_state arrays but delay label computation until after stim gating
    ts_state_num_full = None
    ts_state_char_arr = None
    if ts_state_num is not None and np.size(ts_state_num):
        ts_state_num_full = np.asarray(ts_state_num).astype(int).ravel()
    if ts_state_char is not None:
        ts_state_char_arr = np.asarray(ts_state_char)
        if ts_state_char_arr.ndim > 1:
            ts_state_char_arr = ts_state_char_arr.reshape(-1)


    # ---- quick alignment debug prints ----
    def _end_or_nan(arr):
        arr = np.asarray(arr)
        return float(arr[-1]) if arr.size else float("nan")

    print(
        f"[debug] {aligned_path.name}: "
        f"vog_t_ms_end={_end_or_nan(vog_t_ms):.3f} ms, "
        f"behv_t_end={_end_or_nan(behv_t):.3f} ms, "
        f"NPRW_t_end={_end_or_nan(NPRW_t):.3f} ms, "
        f"UA_t_end={_end_or_nan(UA_t):.3f} ms"
    )

    # ---- basic behavior checks ----
    if behv_t.size == 0:
        print(f"[extract] {aligned_path.name}: no behavior time axis; skipping behavior.")
    # interpolate small NaN gaps
    if beh_cam0.size:
        beh_cam0 = _interp_nans_2d_by_col(beh_cam0, max_gap=4)
    if beh_cam1.size:
        beh_cam1 = _interp_nans_2d_by_col(beh_cam1, max_gap=4)

    cam0_M, cam0_names = _select_matrix(beh_cam0, beh_cam0_cols)
    cam1_M, cam1_names = _select_matrix(beh_cam1, beh_cam1_cols)

    cam0_z = _z_per_column(cam0_M) if cam0_M.size else cam0_M
    cam1_z = _z_per_column(cam1_M) if cam1_M.size else cam1_M

    # ---- behavior-based stim gating: drop bad behavior trials everywhere ----
    stim_ms     = np.asarray(stim_ms, float).ravel()
    stim_ms_abs = np.asarray(stim_ms_abs, float).ravel()

    if stim_ms.size and behv_t.size and cam0_z.size:
        beh_mask = _behavior_valid_mask_from_cam0(
            cam0_z,
            behv_t,
            stim_ms,
            win_ms=WIN_MS,
            baseline_ms=NORMALIZE_FIRST_MS,
            min_trials=MIN_TRIALS,
        )
        n_keep = int(beh_mask.sum())
        if n_keep == 0:
            print(
                f"[extract] {aligned_path.name}: behavior gating would remove all stims; "
                "keeping original stim list."
            )
        elif n_keep < stim_ms.size:
            print(
                f"[extract] {aligned_path.name}: behavior gating kept "
                f"{n_keep}/{stim_ms.size} stims."
            )
            stim_ms = stim_ms[beh_mask]
            if stim_ms_abs.size == beh_mask.size:
                stim_ms_abs = stim_ms_abs[beh_mask]
            else:
                print(
                    f"[extract] {aligned_path.name}: warning stim_ms_abs length mismatch; "
                    "not gating stim_ms_abs."
                )
        # if n_keep == stim_ms.size: nothing to change
    
    cam0_pos_filt, cam0_vel, _ = rcp.butter_lowpass_pos_and_vel(cam0_z, behv_t, cutoff_hz=10.0, order=3)
    cam1_pos_filt, cam1_vel, _ = rcp.butter_lowpass_pos_and_vel(cam1_z, behv_t, cutoff_hz=10.0, order=3)

    # ---- behavior median lines (pos+vel) + segments ----

    # ---- compute labels A/B/N per stim from ts_state on UA timebase (after gating) ----
    if ts_state_num_full is not None and np.size(UA_t) and stim_ms.size:
        trial_labels = _compute_trial_labels_from_ts_state(
            stim_ms=stim_ms,
            UA_t=UA_t,
            ts_state_num_full=ts_state_num_full,
            ts_state_char_arr=ts_state_char_arr,
            win_ms=WIN_MS,
        )
    else:
        trial_labels = np.full(stim_ms.shape, "N", dtype="U1")

    # ---- A/B behavior medians (position) ----
    beh_cam0_pos_med_A, beh_rel_t_A, n_beh_A, beh_cam0_segs_A = _behavior_medians_for_label(cam0_z, behv_t, stim_ms, trial_labels, "A")
    beh_cam0_pos_med_B, beh_rel_t_B, n_beh_B, beh_cam0_segs_B = _behavior_medians_for_label(cam0_z, behv_t, stim_ms, trial_labels, "B")

    beh_cam1_pos_med_A, _, _, beh_cam1_segs_A = _behavior_medians_for_label(cam1_z, behv_t, stim_ms, trial_labels, "A")
    beh_cam1_pos_med_B, _, _, beh_cam1_segs_B = _behavior_medians_for_label(cam1_z, behv_t, stim_ms, trial_labels, "B")

    # ---- A/B behavior medians (velocity) ----
    beh_cam0_vel_med_A, _, _, _ = _behavior_medians_for_label(cam0_vel, behv_t, stim_ms, trial_labels, "A")
    beh_cam0_vel_med_B, _, _, _ = _behavior_medians_for_label(cam0_vel, behv_t, stim_ms, trial_labels, "B")

    beh_cam1_vel_med_A, _, _, _ = _behavior_medians_for_label(cam1_vel, behv_t, stim_ms, trial_labels, "A")
    beh_cam1_vel_med_B, _, _, _ = _behavior_medians_for_label(cam1_vel, behv_t, stim_ms, trial_labels, "B")

    # ---- VOG peri-stim (median lines + segments) ----
    if vog_cols.size and vog_t_ms.size:
        vog_cols = _interp_nans_2d_by_col(vog_cols, max_gap=4)
        vog_z = _z_per_column(vog_cols)

        # keep only stims whose [stim+WIN0, stim+WIN1] lies fully within VOG range
        stim_vog = _filter_stims_for_stream(stim_ms_abs, vog_t_ms, WIN_MS)

        vog_med, vog_rel_t, n_vog_trials, vog_segs = _median_lines_for_columns(vog_z, vog_t_ms, stim_vog)
    else:
        vog_med       = np.zeros((0, 0), float)
        vog_rel_t     = np.zeros(0, float)
        vog_segs      = np.zeros((0, 0, 0), float)
        n_vog_trials  = 0


    # ---- ts_state peri-stim segments (numeric + char) ----
    ts_state_rel_t      = np.zeros(0, float)
    ts_state_segs       = np.zeros((0, 0), float)
    ts_state_char_segs  = np.zeros((0, 0), dtype="U1")
    n_ts_state_trls     = 0

    if ts_state_num_full is not None and np.size(UA_t):
        # peri-stim numeric segments on UA timebase
        try:
            segs_ts, ts_state_rel_t, _ = rcp.extract_peristim_segments(
                ts_state_num_full.reshape(1, -1),  # (1, T)
                UA_t,
                stim_ms,
                win_ms=WIN_MS,
                min_trials=MIN_TRIALS,
            )
            if segs_ts.size:
                # (n_events, 1, T) -> (n_events, T)
                ts_state_segs = segs_ts[:, 0, :]
                n_ts_state_trls = int(ts_state_segs.shape[0])
        except RuntimeError as e:
            if "Only 0 peri-stim segments" not in str(e):
                raise

        # build char segments if we have a char time series
        if ts_state_segs.size and ts_state_char_arr is not None:
            mapping: dict[int, str] = {}
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
    ts_state_segs_A      = np.zeros_like(ts_state_segs[:0, :])  # (0, T) if T known, else (0, 0)
    ts_state_segs_B      = np.zeros_like(ts_state_segs[:0, :])
    ts_state_char_segs_A = np.zeros_like(ts_state_char_segs[:0, :])
    ts_state_char_segs_B = np.zeros_like(ts_state_char_segs[:0, :])
    n_ts_state_trls_A    = 0
    n_ts_state_trls_B    = 0

    if ts_state_segs.size and stim_ms.size and np.size(UA_t):
        # map trial_labels (per-stim) onto ts_state event axis via the same mask used for UA
        mask_ts = _stims_mask_for_stream(stim_ms, UA_t, WIN_MS)
        if mask_ts.size == trial_labels.size and mask_ts.sum() == ts_state_segs.shape[0]:
            labels_ts = trial_labels[mask_ts]

            # A subset
            idx_A = np.where(labels_ts == "A")[0]
            if idx_A.size:
                ts_state_segs_A = ts_state_segs[idx_A]
                if ts_state_char_segs.size and ts_state_char_segs.shape == ts_state_segs.shape:
                    ts_state_char_segs_A = ts_state_char_segs[idx_A]
                n_ts_state_trls_A = int(idx_A.size)

            # B subset
            idx_B = np.where(labels_ts == "B")[0]
            if idx_B.size:
                ts_state_segs_B = ts_state_segs[idx_B]
                if ts_state_char_segs.size and ts_state_char_segs.shape == ts_state_segs.shape:
                    ts_state_char_segs_B = ts_state_char_segs[idx_B]
                n_ts_state_trls_B = int(idx_B.size)
        else:
            # shape mismatch; keep A/B as empty
            print(
                f"[warn] ts_state label mapping mismatch for {aligned_path.name}: "
                f"mask_ts.sum()={mask_ts.sum()}, ts_state_segs.shape[0]={ts_state_segs.shape[0]}, "
                f"trial_labels.size={trial_labels.size}"
            )

    # ---- NPRW/UA peri-stim med + var ----
    _, _, nprw_rel_t, _, nprw_zeroed = _safe_extract_segments(
        NPRW_rate, NPRW_t, stim_ms, WIN_MS, MIN_TRIALS, NORMALIZE_FIRST_MS
    )
    _, _, ua_rel_t, _, ua_zeroed = _safe_extract_segments(
        UA_rate, UA_t, stim_ms, WIN_MS, MIN_TRIALS, NORMALIZE_FIRST_MS
    )

    # labels on NPRW event axis
    mask_nprw = _stims_mask_for_stream(stim_ms, NPRW_t, WIN_MS)
    labels_nprw = trial_labels[mask_nprw] if trial_labels is not None else None

    # labels on UA event axis
    mask_ua = _stims_mask_for_stream(stim_ms, UA_t, WIN_MS)
    labels_ua = trial_labels[mask_ua] if trial_labels is not None else None

    # A/B subsets for NPRW
    NPRW_med_A, NPRW_var_A, n_nprw_A, NPRW_zeroed_A = _subset_neural_from_zeroed(nprw_zeroed, labels_nprw, "A")
    NPRW_med_B, NPRW_var_B, n_nprw_B, NPRW_zeroed_B = _subset_neural_from_zeroed(nprw_zeroed, labels_nprw, "B")

    # A/B subsets for UA
    UA_med_A, UA_var_A, n_ua_A, UA_zeroed_A = _subset_neural_from_zeroed(ua_zeroed, labels_ua, "A")
    UA_med_B, UA_var_B, n_ua_B, UA_zeroed_B = _subset_neural_from_zeroed(ua_zeroed, labels_ua, "B")

    # ---- metadata for titles ----
    try:
        overall_title, _ = build_title_from_csv(
            METADATA_CSV, br_file=meta.get("br_idx")
        )
    except Exception as e:
        print(f"[warn] metadata parse failed for BR {br_idx}: {e}")
        overall_title, _ = (
            "Condition: n/a, n/a Hz, n/a µA, n/a mm, n/a ms, Delay: 0 ms",
            None,
        )

    
    # ------------------------------------------------------------------
    # Per-target files: Target A and Target B
    # ------------------------------------------------------------------
    targetA_dir = PERI_ROOT / "Target_A"
    targetB_dir = PERI_ROOT / "Target_B"
    targetA_dir.mkdir(parents=True, exist_ok=True)
    targetB_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Target A ----------
    peribase_A = f"peristim__{sess}__BR_{int(br_idx):03d}_target_A"
    out_npz_A  = targetA_dir / f"{peribase_A}.npz"
    out_mat_A  = targetA_dir / f"{peribase_A}.mat"

    np.savez_compressed(
        out_npz_A,
        sess=sess,
        br_idx=int(br_idx),
        overall_title=overall_title,
        stim_ms=stim_ms,
        meta=meta,

        # behavior (A only, mapped to generic field names)
        n_beh=n_beh_A,
        beh_rel_t=beh_rel_t_A,
        beh_cam0_pos_med=beh_cam0_pos_med_A,
        beh_cam1_pos_med=beh_cam1_pos_med_A,
        beh_cam0_vel_med=beh_cam0_vel_med_A,
        beh_cam1_vel_med=beh_cam1_vel_med_A,
        beh_cam0_segs=beh_cam0_segs_A,
        beh_cam1_segs=beh_cam1_segs_A,
        beh_cam0_names=np.array(cam0_names, dtype=object),
        beh_cam1_names=np.array(cam1_names, dtype=object),

        # VOG and ts_state (shared)
        vog_rel_t=vog_rel_t,
        vog_med=vog_med,
        vog_segs=vog_segs,
        vog_cols=vog_cols,
        vog_col_names=np.array(vog_col_names, dtype=object),
        n_vog_trials=int(n_vog_trials),

        ts_state_rel_t=ts_state_rel_t,
        ts_state_segs=ts_state_segs_A,
        ts_state_char_segs=ts_state_char_segs_A,
        ts_state_char=ts_state_char,
        ts_state_num=ts_state_num_full if ts_state_num is not None else np.array([], int),
        n_ts_state_trials=int(n_ts_state_trls_A),
        trial_labels=np.array(trial_labels, dtype="U1"),  # still full stim labels

        # NPRW / UA (A only)
        NPRW_med=NPRW_med_A,
        NPRW_var=NPRW_var_A,
        NPRW_zeroed=NPRW_zeroed_A,
        NPRW_rel_t=nprw_rel_t,
        n_nprw=int(n_nprw_A),

        UA_med=UA_med_A,
        UA_var=UA_var_A,
        UA_zeroed=UA_zeroed_A,
        UA_rel_t=ua_rel_t,
        n_ua=int(n_ua_A),
        ua_ids_1based=ua_ids_1based if ua_ids_1based is not None else np.array([], int),
        
        # UA meta (per aligned file)
        ua_region=ua_region,
        ua_region_names=ua_region_names,
        ua_port=ua_port,
        ua_nsp=ua_nsp,
        ua_idx_rows=ua_idx_rows,
    )

    mat_A = {
        "sess":          np.array(sess, dtype=object),
        "br_idx":        int(br_idx),
        "overall_title": np.array(overall_title, dtype=object),
        "stim_ms":       stim_ms,
        
        "n_beh":       n_beh_A,
        "beh_rel_t":     beh_rel_t_A,
        "beh_cam0_pos_med": beh_cam0_pos_med_A,
        "beh_cam1_pos_med": beh_cam1_pos_med_A,
        "beh_cam0_vel_med": beh_cam0_vel_med_A,
        "beh_cam1_vel_med": beh_cam1_vel_med_A,
        "beh_cam0_segs": beh_cam0_segs_A,
        "beh_cam1_segs": beh_cam1_segs_A,
        "beh_cam0_names": np.array(cam0_names, dtype=object),
        "beh_cam1_names": np.array(cam1_names, dtype=object),

        "vog_rel_t":     vog_rel_t,
        "vog_med":       vog_med,
        "vog_segs":      vog_segs,
        "vog_cols":      vog_cols,
        "vog_col_names": np.array(vog_col_names, dtype=object),
        "n_vog_trials":  int(n_vog_trials),

        "ts_state_rel_t":      ts_state_rel_t,
        "ts_state_segs":       ts_state_segs_A,
        "ts_state_char_segs":  ts_state_char_segs_A,
        "ts_state_char":       ts_state_char,
        "ts_state_num":        ts_state_num_full if ts_state_num is not None else np.array([], int),
        "n_ts_state_trials":   int(n_ts_state_trls_A),
        "trial_labels":        trial_labels,

        "NPRW_med":      NPRW_med_A,
        "NPRW_var":      NPRW_var_A,
        "NPRW_zeroed":   NPRW_zeroed_A,
        "NPRW_rel_t":    nprw_rel_t,
        "n_nprw":        int(n_nprw_A),

        "UA_med":        UA_med_A,
        "UA_var":        UA_var_A,
        "UA_zeroed":     UA_zeroed_A,
        "UA_rel_t":      ua_rel_t,
        "n_ua":          int(n_ua_A),
        "ua_ids_1based": ua_ids_1based if ua_ids_1based is not None else np.array([], int),
        
        # UA meta
        "ua_region":      ua_region,
        "ua_region_names": ua_region_names,
        "ua_port":        ua_port,
        "ua_nsp":         ua_nsp,
        "ua_idx_rows":    ua_idx_rows,
    }
    savemat(out_mat_A, mat_A, do_compression=True)

    # ---------- Target B ----------
    peribase_B = f"peristim__{sess}__BR_{int(br_idx):03d}_target_B"
    out_npz_B  = targetB_dir / f"{peribase_B}.npz"
    out_mat_B  = targetB_dir / f"{peribase_B}.mat"

    np.savez_compressed(
        out_npz_B,
        sess=sess,
        br_idx=int(br_idx),
        overall_title=overall_title,
        stim_ms=stim_ms,
        meta=meta,

        n_beh=n_beh_B,
        beh_rel_t=beh_rel_t_B,
        beh_cam0_pos_med=beh_cam0_pos_med_B,
        beh_cam1_pos_med=beh_cam1_pos_med_B,
        beh_cam0_vel_med=beh_cam0_vel_med_B,
        beh_cam1_vel_med=beh_cam1_vel_med_B,
        beh_cam0_segs=beh_cam0_segs_B,
        beh_cam1_segs=beh_cam1_segs_B,
        beh_cam0_names=np.array(cam0_names, dtype=object),
        beh_cam1_names=np.array(cam1_names, dtype=object),

        vog_rel_t=vog_rel_t,
        vog_med=vog_med,
        vog_segs=vog_segs,
        vog_cols=vog_cols,
        vog_col_names=np.array(vog_col_names, dtype=object),
        n_vog_trials=int(n_vog_trials),

        ts_state_rel_t=ts_state_rel_t,
        ts_state_segs=ts_state_segs_B,
        ts_state_char_segs=ts_state_char_segs_B,
        ts_state_char=ts_state_char,
        ts_state_num=ts_state_num_full if ts_state_num is not None else np.array([], int),
        n_ts_state_trials=int(n_ts_state_trls_B),
        trial_labels=np.array(trial_labels, dtype="U1"),

        NPRW_med=NPRW_med_B,
        NPRW_var=NPRW_var_B,
        NPRW_zeroed=NPRW_zeroed_B,
        NPRW_rel_t=nprw_rel_t,
        n_nprw=int(n_nprw_B),

        UA_med=UA_med_B,
        UA_var=UA_var_B,
        UA_zeroed=UA_zeroed_B,
        UA_rel_t=ua_rel_t,
        n_ua=int(n_ua_B),
        ua_ids_1based=ua_ids_1based if ua_ids_1based is not None else np.array([], int),

        # UA meta (per aligned file)
        ua_region=ua_region,
        ua_region_names=ua_region_names,
        ua_port=ua_port,
        ua_nsp=ua_nsp,
        ua_idx_rows=ua_idx_rows,

    )

    mat_B = {
        "sess":          np.array(sess, dtype=object),
        "br_idx":        int(br_idx),
        "overall_title": np.array(overall_title, dtype=object),
        "stim_ms":       stim_ms,

        "n_beh":         n_beh_B,
        "beh_rel_t":     beh_rel_t_B,
        "beh_cam0_pos_med": beh_cam0_pos_med_B,
        "beh_cam1_pos_med": beh_cam1_pos_med_B,
        "beh_cam0_vel_med": beh_cam0_vel_med_B,
        "beh_cam1_vel_med": beh_cam1_vel_med_B,
        "beh_cam0_segs": beh_cam0_segs_B,
        "beh_cam1_segs": beh_cam1_segs_B,
        "beh_cam0_names": np.array(cam0_names, dtype=object),
        "beh_cam1_names": np.array(cam1_names, dtype=object),

        "vog_rel_t":     vog_rel_t,
        "vog_med":       vog_med,
        "vog_segs":      vog_segs,
        "vog_cols":      vog_cols,
        "vog_col_names": np.array(vog_col_names, dtype=object),
        "n_vog_trials":  int(n_vog_trials),

        "ts_state_rel_t":      ts_state_rel_t,
        "ts_state_segs":       ts_state_segs_B,
        "ts_state_char_segs":  ts_state_char_segs_B,
        "ts_state_char":       ts_state_char,
        "ts_state_num":        ts_state_num_full if ts_state_num is not None else np.array([], int),
        "n_ts_state_trials":   int(n_ts_state_trls_B),
        "trial_labels":        trial_labels,

        "NPRW_med":      NPRW_med_B,
        "NPRW_var":      NPRW_var_B,
        "NPRW_zeroed":   NPRW_zeroed_B,
        "NPRW_rel_t":    nprw_rel_t,
        "n_nprw":        int(n_nprw_B),

        "UA_med":        UA_med_B,
        "UA_var":        UA_var_B,
        "UA_zeroed":     UA_zeroed_B,
        "UA_rel_t":      ua_rel_t,
        "n_ua":          int(n_ua_B),
        "ua_ids_1based": ua_ids_1based if ua_ids_1based is not None else np.array([], int),

        # UA meta
        "ua_region":      ua_region,
        "ua_region_names": ua_region_names,
        "ua_port":        ua_port,
        "ua_nsp":         ua_nsp,
        "ua_idx_rows":    ua_idx_rows,

    }
    savemat(out_mat_B, mat_B, do_compression=True)
    print(f"[extract] wrote peristim__{sess}__BR_{int(br_idx)}")
    
def main():
    files = sorted(ALIGNED_ROOT.glob("aligned__*.npz"))
    if not files:
        raise SystemExit(f"[error] No combined aligned NPZs found at {ALIGNED_ROOT}")
    for f in files:
        extract_one_file(f)

if __name__ == "__main__":
    main()
