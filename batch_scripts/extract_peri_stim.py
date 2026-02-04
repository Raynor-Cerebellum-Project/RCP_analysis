import re, json
from scipy.io import savemat
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

SHIFTS_CSV = METADATA_ROOT / "br_to_intan_shifts.csv"

# Config
# Base paths from config_loading
CONTROL_ROOT = ALIGNED_CKPT_ROOT / "control_reaches"
AT_REST_ROOT  = ALIGNED_CKPT_ROOT / "at_rest"
STIM_ROOT  = ALIGNED_CKPT_ROOT / "stim_reaches"

CONTROL_ROOT.mkdir(parents=True, exist_ok=True)
AT_REST_ROOT.mkdir(parents=True, exist_ok=True)
STIM_ROOT.mkdir(parents=True, exist_ok=True)

CONTROL_PERI_ROOT = PERI_ROOT / "control_reaches"
AT_REST_PERI_ROOT  = PERI_ROOT / "at_rest"
STIM_PERI_ROOT  = PERI_ROOT / "stim_reaches"

CONTROL_PERI_ROOT.mkdir(parents=True, exist_ok=True)
AT_REST_PERI_ROOT.mkdir(parents=True, exist_ok=True)
STIM_PERI_ROOT.mkdir(parents=True, exist_ok=True)

NPRW_RATES = PARAMS.NPRW_rate_est
NPRW_BIN_MS     = NPRW_RATES.get("bin_ms")
NPRW_SIGMA_MS   = NPRW_RATES.get("sigma_ms")
NPRW_MS_BEFORE = float(NPRW_RATES.get("remove_ms_before", 20.0))
NPRW_TAIL_MS   = float(NPRW_RATES.get("remove_tail_ms_after", 20.0))
NPRW_DEDUP_MS = float(NPRW_RATES.get("dedup_ms", 0.5))

UA_RATES = PARAMS.UA_rate_est
UA_BIN_MS     = UA_RATES.get("bin_ms")
UA_SIGMA_MS   = UA_RATES.get("sigma_ms")
UA_MS_BEFORE = float(UA_RATES.get("remove_ms_before", 5.0))
UA_TAIL_MS   = float(UA_RATES.get("remove_tail_ms_after", 5.0))
UA_DEDUP_MS = float(UA_RATES.get("dedup_ms", 0.2))
MAX_CLUSTER_MS = 0.5

HAS_BR = PARAMS.preprocessing.get("has_BR")
HAS_KINEMATICS = PARAMS.preprocessing.get("has_kinematics")
HAS_VOG = PARAMS.preprocessing.get("has_VOG")
HAS_HR = PARAMS.preprocessing.get("has_HR")
SAVE_MAT_FILE = PARAMS.preprocessing.get("output_aligned_mat")
PROCESS_ONLY = PARAMS.preprocessing.get("process_only")  # list of BR indices to process; empty = all

KEYPOINTS_ORDER = tuple(PARAMS.kinematics.get("keypoints", []))

# Plotting config
WIN_MS            = (-600.0, 600.0)
NORMALIZE_FIRST_MS = 150.0
MIN_TRIALS        = 1
MIN_BIN_COVERAGE_FRAC = 0.9  # require at least x% of trials finite per bin

def _dedup_peaks(
    peaks: dict[int, np.ndarray],
    amps: dict[int, np.ndarray],
    dedup_ms: float = 0.5,
    max_cluster_ms: float = 1.0,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:

    """Deduplicate peaks per (segment, channel), keeping strongest within short windows"""
    peaks_dedup = {}
    amps_dedup = {}
    for ch in peaks:
        t_ch = peaks[ch]
        a_ch = amps[ch]
        num_peaks = len(t_ch)
        if num_peaks == 0:
            peaks_dedup[ch] = t_ch
            amps_dedup[ch]  = a_ch
            continue

        keep = np.zeros(num_peaks, dtype = bool)
        cluster_start = 0 # index counter
        for i in range(1, num_peaks):
            isi = t_ch[i] - t_ch[i - 1]
            if isi > dedup_ms or (t_ch[i] - t_ch[cluster_start]) > max_cluster_ms: # If isi violation or cluster too big has to be between 0.5ms (ISI violation) or max threshold: 1 ms (could be another MUA)
                best = cluster_start + np.argmax(np.abs(a_ch[cluster_start: i])) # Find max and give index
                keep[best] = True # Keep this spike
                cluster_start = i

        # finalize last cluster
        best = cluster_start + np.argmax(a_ch[cluster_start: num_peaks])
        keep[best] = True
        peaks_dedup[ch] = peaks[ch][np.argwhere(keep == True).flatten()]
        amps_dedup[ch] = amps[ch][np.argwhere(keep == True).flatten()]
    return peaks_dedup, amps_dedup

def _bin_counts_around_stim(
    peaks_ms: dict[int, np.ndarray],
    bin_ms: float,
    stim_times_ms: np.ndarray,
    art_before_ms: float,
    art_after_ms: float,
    win_ms: tuple[float, float] = WIN_MS,
):
    """
    Bin spike counts around each stimulation event.
    """
    window_start_ms, window_end_ms = float(win_ms[0]), float(win_ms[1])

    if peaks_ms is None or len(peaks_ms) == 0:
        return None, None, None, 0

    stim_times_ms = np.asarray(stim_times_ms, float).ravel()
    if stim_times_ms.size == 0:
        return None, None, None, 0

    # Left/right edges & centers
    left_window_edges_ms: list[float] = []
    edge_value = -art_before_ms
    while edge_value >= window_start_ms:
        left_window_edges_ms.append(edge_value)
        edge_value -= bin_ms
    left_window_edges_ms = np.array(left_window_edges_ms[::-1])

    right_window_edges_ms: list[float] = []
    edge_value = art_after_ms
    while edge_value <= window_end_ms:
        right_window_edges_ms.append(edge_value)
        edge_value += bin_ms
    right_window_edges_ms = np.array(right_window_edges_ms)

    # Bin centers on each side
    left_bin_centers_ms  = (
        left_window_edges_ms[:-1] + 0.5 * bin_ms
        if left_window_edges_ms.size > 1 else np.array([], float)
    )
    right_bin_centers_ms = (
        right_window_edges_ms[:-1] + 0.5 * bin_ms
        if right_window_edges_ms.size > 1 else np.array([], float)
    )

    bin_edges_ms   = np.concatenate([left_window_edges_ms, right_window_edges_ms])
    bin_centers_ms = np.concatenate([left_bin_centers_ms, right_bin_centers_ms])
    n_bins         = bin_centers_ms.size

    # Number of bins on the left side (used to offset right-side indices)
    n_left_bins = max(left_window_edges_ms.size - 1, 0)

    n_trials    = len(stim_times_ms)
    ch_keys = sorted(peaks_ms.keys())
    n_channels = len(ch_keys)

    # counts[trial_index, channel_index, bin_index]
    counts = np.zeros((n_trials, n_channels, n_bins), float)

    # Binning
    for trial_index, stim_time_ms in enumerate(stim_times_ms):
        trial_window_start_ms = stim_time_ms + window_start_ms
        trial_window_end_ms   = stim_time_ms + window_end_ms
        for ch_i, ch_id in enumerate(ch_keys):
            spike_times_ms = np.asarray(peaks_ms[ch_id], float).ravel()
            if spike_times_ms.size == 0:
                continue

            # Spikes within the overall peri-stim window (absolute)
            in_window_mask = (
                (spike_times_ms >= trial_window_start_ms) &
                (spike_times_ms <= trial_window_end_ms)
            )
            if not in_window_mask.any():
                continue

            # Relative spike times in ms, centered at the stim
            spike_times_rel_ms = spike_times_ms[in_window_mask] - stim_time_ms

            # Exclude spikes in the artifact gap [-art_before_ms, +art_after_ms]
            left_side_mask  = (
                (spike_times_rel_ms >= window_start_ms) &
                (spike_times_rel_ms < -art_before_ms)
            )
            right_side_mask = (
                (spike_times_rel_ms > art_after_ms) &
                (spike_times_rel_ms <= window_end_ms)
            )

            # ---- LEFT side binning ----
            if left_window_edges_ms.size > 1 and left_side_mask.any():
                spike_times_left_ms = spike_times_rel_ms[left_side_mask]
                # Find bin index j such that edges[j] <= t < edges[j+1]
                left_bin_indices = np.searchsorted(
                    left_window_edges_ms, spike_times_left_ms, side="right"
                ) - 1
                valid_left_bins = (
                    (left_bin_indices >= 0) &
                    (left_bin_indices < left_window_edges_ms.size - 1)
                )
                if valid_left_bins.any():
                    np.add.at(counts[trial_index, ch_i], left_bin_indices[valid_left_bins], 1.0)

            # ---- RIGHT side binning ----
            if right_window_edges_ms.size > 1 and right_side_mask.any():
                spike_times_right_ms = spike_times_rel_ms[right_side_mask]
                right_bin_indices_local = np.searchsorted(
                    right_window_edges_ms, spike_times_right_ms, side="right"
                ) - 1
                valid_right_bins = (
                    (right_bin_indices_local >= 0) &
                    (right_bin_indices_local < right_window_edges_ms.size - 1)
                )
                if valid_right_bins.any():
                    # Map right-side bins into global bin indices
                    right_bin_indices_global = (
                        right_bin_indices_local[valid_right_bins] + n_left_bins
                    )
                    np.add.at(counts[trial_index, ch_i], right_bin_indices_global, 1.0)
    return counts, bin_centers_ms, bin_edges_ms, n_left_bins

def _smooth_segment(seg_counts, seg_edges, seg_centers, sigma_ms):
        """
        """
        L = seg_centers.size
        if L == 0 or sigma_ms <= 0:
            return seg_counts.copy()

        # bin widths
        dt_ms = np.diff(seg_edges)
        dt_sec = np.clip(dt_ms / 1000.0, 1e-12, None)

        # convert counts → rates
        rates = seg_counts / dt_sec[None, None, :]   # broadcast dt

        # Gaussian weights
        diff = seg_centers[:, None] - seg_centers[None, :]
        W = np.exp(-(diff**2) / (2.0 * sigma_ms**2))   # (L, L)

        # output
        out = np.full_like(rates, np.nan)

        for tr in range(rates.shape[0]):
            for ch in range(rates.shape[1]):
                r = rates[tr, ch]
                valid = np.isfinite(r)
                if not valid.any():
                    continue

                r_valid = np.where(valid, r, 0.0)
                num = r_valid @ W.T
                den = (valid.astype(float) @ W.T).clip(1e-12, None)
                out[tr, ch] = num / den

        return out

def _smooth_counts_gauss(
    counts: np.ndarray,
    edges: np.ndarray,
    bin_centers_ms: np.ndarray,
    sigma_ms: float,
    left_bins: int,
) -> np.ndarray:
    """
    Smooth spike counts -> firing rates (Hz) separately on left and right sides
    of the stim-blank gap, using a Gaussian in ms (NaN-aware at the per-bin level).
    """
    if counts is None or np.size(counts) == 0:
        return np.asarray(counts, float)

    gap_bin = int(left_bins)
    gap_bin = max(0, min(gap_bin, counts.shape[-1]))  # clamp

    left_counts   = counts[:, :, :gap_bin]
    right_counts  = counts[:, :, gap_bin:]
    left_centers  = bin_centers_ms[:gap_bin]
    right_centers = bin_centers_ms[gap_bin:]

    # edges is length (T+2): left edges (L+1) + right edges (R+1)
    left_edges  = edges[:gap_bin + 1]
    right_edges = edges[gap_bin + 1:]

    left_sm  = _smooth_segment(left_counts,  left_edges,  left_centers,  sigma_ms)
    right_sm = _smooth_segment(right_counts, right_edges, right_centers, sigma_ms)

    out = np.full_like(counts, np.nan, dtype=float)
    out[:, :, :gap_bin] = left_sm
    out[:, :, gap_bin:] = right_sm
    return out

def _get_valid_events(event_ms: np.ndarray,
                      rec_time: tuple[float, float],
                      win_ms: tuple[float, float]) -> np.ndarray:
    """
    Return mask over event_ms where [event+win0, event+win1] is in recording time.
    """
    event_ms = np.asarray(event_ms, float).ravel()
    if event_ms.size == 0:
        return np.zeros(0, dtype=bool)
    w0, w1 = float(win_ms[0]), float(win_ms[1])
    if rec_time is None or len(rec_time) < 2:
        return np.zeros(event_ms.size, dtype=bool)
    t0, t1 = float(rec_time[0]), float(rec_time[1])
    return (event_ms + w0 >= t0) & (event_ms + w1 <= t1)

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)

def _last(x):
    x = np.asarray(x)
    return float(x.reshape(-1)[-1]) if x.size else float("nan")

def _median_behavior_line(series_on_common: np.ndarray,
                          t_common_ms: np.ndarray,
                          event_ms: np.ndarray,
                          win_ms: tuple[float, float],
                          baseline_ms: float,
                          min_trials: int
                          ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
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
            stim_ms=event_ms,
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

    rate_segs = rate_segs[:, 0, :]  # (n_trials, T)

    bl_mask = (rel_t >= rel_t[0]) & (rel_t <= baseline_ms)
    if not bl_mask.any():
        step = max(1, int(round(baseline_ms / (np.nanmedian(np.diff(rel_t)) if rel_t.size > 1 else 1.0))))
        bl_mask = np.zeros_like(rel_t, bool)
        bl_mask[:step] = True

    keep = np.isfinite(rate_segs[:, bl_mask]).any(axis=1)
    n_kept = int(np.count_nonzero(keep))
    if n_kept == 0:
        return np.full(rate_segs.shape[1], np.nan, float), rel_t, 0, np.zeros((0, rate_segs.shape[1]), float)

    rate_segs = rate_segs[keep]
    rate_segs = rate_segs - np.nanmedian(rate_segs[:, bl_mask], axis=1, keepdims=True)
    line = np.nanmedian(rate_segs, axis=0)
    return line, rel_t, n_kept, rate_segs

def _median_lines_for_columns(series_on_common: np.ndarray,
                              t_common_ms: np.ndarray,
                              event_ms: np.ndarray
                              ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
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
    rel_t_out: np.ndarray | None = None

    for d in range(D):
        line, rel_t, n_kept, segs = _median_behavior_line(
            series_on_common[:, d],
            t_common_ms, event_ms,
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

# Behavior
def _ordered_xy_indices(cam_cols: list[str],
                        keypoints: tuple[str, ...] = KEYPOINTS_ORDER
                        ) -> tuple[list[int], list[str]]:
    idx: list[int] = []
    names: list[str] = []
    cols_lc = [c.lower() for c in cam_cols]

    def _clean_kp(s: str) -> str:
        s = str(s).strip()
        s = s.strip("[]").strip("'\"")   # handle "'Wrist'" → Wrist
        return s.lower()

    def _find_xy_for(kp: str) -> tuple[int | None, int | None]:
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

def _select_matrix(cam: np.ndarray, cols: list[str],
                   keypoints: tuple[str, ...] = KEYPOINTS_ORDER
                   ) -> tuple[np.ndarray, list[str]]:
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

def build_title_from_csv(csv_path: Path, *,  sess: str | None = None, br_file: int | None = None) -> tuple[str, int | None]:
    df_raw = _read_csv_robust(csv_path)

    # normalize column names: lower, strip, spaces->underscores
    df = df_raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # drop the first row if it's a header-like row (your original iloc[1:])
    df_data = df.iloc[1:].reset_index(drop=True) if len(df) > 1 else df.copy()
    if df_data.empty:
        return "Condition: n/a, n/a Hz, n/a µA, n/a mm, n/a ms, Delay: 0 ms", None

    def col(name: str) -> str | None:
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
    video_file: int | None = None
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

def _compute_trial_labels(
    event_ms: np.ndarray,
    t_ms: np.ndarray,
    ts_char: np.ndarray | None,
    ts_num: np.ndarray,
    win_ms: tuple[float, float] = WIN_MS,
    *,
    num_to_label: dict[int, str] | None = None,   # e.g. {1:"A", 2:"B"}
) -> np.ndarray:
    event_ms = np.asarray(event_ms, float).ravel()
    t_ms     = np.asarray(t_ms, float).ravel()
    ts_num   = np.asarray(ts_num).ravel()
    ts_char_arr = None if ts_char is None else np.asarray(ts_char).reshape(-1)

    labels = np.full(event_ms.shape, "N", dtype="U1")
    w0, w1 = map(float, win_ms)

    # must be same length
    if t_ms.size == 0 or ts_num.size == 0 or t_ms.size != ts_num.size:
        return labels

    use_char = (ts_char_arr is not None and ts_char_arr.size == t_ms.size)

    for i, s in enumerate(event_ms):
        lo, hi = s + w0, s + w1

        i0 = np.searchsorted(t_ms, lo, side="left")
        i1 = np.searchsorted(t_ms, hi, side="right")
        if i1 <= i0:
            continue

        if use_char:
            chars = ts_char_arr[i0:i1].astype("U1")
            chars = chars[(chars == "A") | (chars == "B")]
            if chars.size:
                vals, cnts = np.unique(chars, return_counts=True)
                labels[i] = vals[np.argmax(cnts)]
        else:
            if not num_to_label:
                continue
            nums = ts_num[i0:i1].astype(int, copy=False)
            # map numeric codes to A/B and take majority
            mapped = np.array([num_to_label.get(int(n), "N") for n in nums], dtype="U1")
            mapped = mapped[(mapped == "A") | (mapped == "B")]
            if mapped.size:
                vals, cnts = np.unique(mapped, return_counts=True)
                labels[i] = vals[np.argmax(cnts)]

    return labels

def _behavior_medians_for_label(
    cam_z: np.ndarray,
    behv_t: np.ndarray,
    event_ms: np.ndarray,
    labels: np.ndarray,
    target_label: str,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Recompute behavior medians using only stims with labels==target_label.
    """
    if cam_z is None or cam_z.size == 0 or event_ms.size == 0:
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

    stim_label = event_ms[mask]
    lines, rel_t, n_trials, segs = _median_lines_for_columns(cam_z, behv_t, stim_label)
    return lines, rel_t, n_trials, segs

def _behavior_valid_mask_from_cam0(
    cam0_z: np.ndarray,
    behv_t: np.ndarray,
    event_ms: np.ndarray,
    win_ms: tuple[float, float] = WIN_MS,
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
    events_all = np.asarray(event_ms, float).ravel()
    if events_all.size == 0:
        return np.zeros(0, dtype=bool)

    if cam0_z is None or cam0_z.size == 0 or behv_t.size < 2:
        # no behavior info to gate with → keep all stims
        return np.ones(events_all.shape, dtype=bool)

    cam0_z = np.asarray(cam0_z, float)
    if cam0_z.ndim == 2:
        series = cam0_z[:, 0]  # first column as proxy
    elif cam0_z.ndim == 1:
        series = cam0_z
    else:
        return np.ones(events_all.shape, dtype=bool)

    try:
        rate_segs, _, rel_t, stim_valid_ts = rcp.extract_peristim_segments(
            rate_hz=series[None, :],   # (1, T)
            counts=None,
            t_ms=behv_t,
            stim_ms=events_all,
            win_ms=win_ms,
            min_trials=min_trials,
        )
    except RuntimeError as e:
        if "Only 0 peri-stim segments" in str(e):
            # nothing extractable on behavior → consider all bad
            return np.zeros(events_all.shape, dtype=bool)
        raise

    if rate_segs.size == 0 or stim_valid_ts.size == 0:
        return np.zeros(events_all.shape, dtype=bool)

    # segs: (n_events_valid, 1, T) → (n_events_valid, T)
    rate_segs = rate_segs[:, 0, :]

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
    keep_valid = np.isfinite(rate_segs[:, bl_mask]).any(axis=1)

    # Now map back to a mask over the original events
    mask_beh = np.zeros(events_all.shape, dtype=bool)
    stim_kept = stim_valid_ts[keep_valid]

    for s in np.asarray(stim_kept, float):
        idx = np.where(np.isclose(events_all, s, rtol=0.0, atol=1e-6))[0]
        if idx.size:
            # If duplicates ever existed, this marks the first occurrence.
            mask_beh[idx[0]] = True

    return mask_beh

def _save_peristim(
    *,
    out_dir: Path,
    target_name: str | None,
    intan_filename: str,
    br_idx: int,
    overall_title: str,

    meta: dict,
    align_meta_raw: str,
    nprw_meta: dict,
    ua_meta: dict | None,

    # events + labels (must match lengths)
    event_ms_sub: np.ndarray,
    labels_sub: np.ndarray,
    labels_all: np.ndarray,

    # NPRW
    NPRW_med_sub: np.ndarray,
    NPRW_var_sub: np.ndarray,
    NPRW_med_counts_sub: np.ndarray,
    NPRW_rates_hz_sub: np.ndarray,
    NPRW_rates_zeroed_sub: np.ndarray,
    NPRW_counts_sub: np.ndarray,
    NPRW_edges_ms: np.ndarray,
    NPRW_rel_t: np.ndarray,
    NPRW_peak_ms_dedup: dict[int, np.ndarray],
    NPRW_amps_ms_dedup: dict[int, np.ndarray],

    # UA (optional)
    HAS_BR: bool,
    UA_med_sub: np.ndarray | None = None,
    UA_var_sub: np.ndarray | None = None,
    UA_med_counts_sub: np.ndarray | None = None,
    UA_rates_hz_sub: np.ndarray | None = None,
    UA_rates_zeroed_sub: np.ndarray | None = None,
    UA_counts_sub: np.ndarray | None = None,
    UA_edges_ms: np.ndarray | None = None,
    UA_rel_t: np.ndarray | None = None,
    UA_peak_ms_dedup: dict[int, np.ndarray] | None = None,
    UA_amps_ms_dedup: dict[int, np.ndarray] | None = None,
    ua_ids_1based: np.ndarray | None = None,
    ua_region=None,
    ua_region_names=None,
    ua_port=None,
    ua_nsp=None,
    ua_idx_rows=None,
    
    # optional signals
    hr_sig: np.ndarray,
    vog_sig: np.ndarray,

    # ts_state (already subsetted to A/B if you want, but can be empty)
    ts_state_rel_t: np.ndarray,
    ts_state_segs_sub: np.ndarray,
    ts_state_char_segs_sub: np.ndarray,
    ts_state_char: np.ndarray | None,
    ts_state_num_full: np.ndarray,
    n_ts_state_trials_sub: int,
    
    # behavior
    n_beh_sub: int,
    beh_rel_t_sub: np.ndarray,
    beh_cam0_pos_med_sub: np.ndarray,
    beh_cam1_pos_med_sub: np.ndarray,
    beh_cam0_vel_med_sub: np.ndarray,
    beh_cam1_vel_med_sub: np.ndarray,
    beh_cam0_segs_sub: np.ndarray,
    beh_cam1_segs_sub: np.ndarray,
    beh_cam0_vel_segs_sub: np.ndarray,
    beh_cam1_vel_segs_sub: np.ndarray,
    beh_cam0_names: list[str],
    beh_cam1_names: list[str],
) -> None:
    """
    Save one .npz and/or .mat
      - Target subset (A/B) or
      - all events (target_name=None), e.g. AtRest
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if target_name is None:
        peribase = f"peristim__{intan_filename}__BR_{int(br_idx):03d}"
    else:
        peribase = f"peristim__{intan_filename}__BR_{int(br_idx):03d}_target_{target_name}"

    out_npz = out_dir / f"{peribase}.npz"
    out_mat = out_dir / f"{peribase}.mat"

    meta_json      = json.dumps(meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    ua_meta_json   = json.dumps(ua_meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) if ua_meta else "{}"
    nprw_meta_json = json.dumps(nprw_meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    np.savez_compressed(
        out_npz,
        sess=intan_filename,
        br_idx=int(br_idx),
        overall_title=overall_title,

        meta=meta,
        ua_meta=ua_meta if HAS_BR else None,
        nprw_meta=nprw_meta,
        meta_json=meta_json,
        ua_meta_json=ua_meta_json,
        nprw_meta_json=nprw_meta_json,

        align_meta_raw=align_meta_raw,

        event_ms=np.asarray(event_ms_sub, float),
        trial_labels=np.asarray(labels_sub, dtype="U1"),        # aligned to event_ms_sub
        trial_labels_all=np.asarray(labels_all, dtype="U1"),    # full labels after behavior gating

        # behavior
        n_beh=int(n_beh_sub),
        beh_rel_t=np.asarray(beh_rel_t_sub, float),
        beh_cam0_pos_med=np.asarray(beh_cam0_pos_med_sub, float),
        beh_cam1_pos_med=np.asarray(beh_cam1_pos_med_sub, float),
        beh_cam0_vel_med=np.asarray(beh_cam0_vel_med_sub, float),
        beh_cam1_vel_med=np.asarray(beh_cam1_vel_med_sub, float),
        beh_cam0_segs=np.asarray(beh_cam0_segs_sub, float),
        beh_cam1_segs=np.asarray(beh_cam1_segs_sub, float),
        beh_cam0_vel_segs=np.asarray(beh_cam0_vel_segs_sub, float),
        beh_cam1_vel_segs=np.asarray(beh_cam1_vel_segs_sub, float),
        beh_cam0_names=np.array(list(beh_cam0_names), dtype=object),
        beh_cam1_names=np.array(list(beh_cam1_names), dtype=object),

        hr_sig=hr_sig,
        vog_sig=vog_sig,

        # ts_state (already subsetted if you choose)
        ts_state_rel_t=np.asarray(ts_state_rel_t, float),
        ts_state_segs=np.asarray(ts_state_segs_sub, float),
        ts_state_char_segs=np.asarray(ts_state_char_segs_sub, dtype="U1"),
        ts_state_char=ts_state_char if ts_state_char is not None else np.array([], dtype="U1"),
        ts_state_num=np.asarray(ts_state_num_full, int) if ts_state_num_full is not None else np.array([], int),
        n_ts_state_trials=int(n_ts_state_trials_sub),

        # NPRW
        NPRW_med=np.asarray(NPRW_med_sub, float),
        NPRW_var=np.asarray(NPRW_var_sub, float),
        NPRW_med_counts=np.asarray(NPRW_med_counts_sub, float),
        NPRW_rates_hz=np.asarray(NPRW_rates_hz_sub, float),
        NPRW_rates_zeroed=np.asarray(NPRW_rates_zeroed_sub, float),
        NPRW_counts=np.asarray(NPRW_counts_sub, float),
        NPRW_edges_ms=np.asarray(NPRW_edges_ms, float),
        NPRW_rel_t=np.asarray(NPRW_rel_t, float),
        NPRW_peak_ms_dedup=NPRW_peak_ms_dedup,
        NPRW_amps_ms_dedup=NPRW_amps_ms_dedup,

        # UA (optional)
        HAS_BR=bool(HAS_BR),
        UA_med=np.asarray(UA_med_sub, float) if (HAS_BR and UA_med_sub is not None) else np.zeros((0, 0), float),
        UA_var=np.asarray(UA_var_sub, float) if (HAS_BR and UA_var_sub is not None) else np.zeros((0, 0), float),
        UA_med_counts=np.asarray(UA_med_counts_sub, float) if (HAS_BR and UA_med_counts_sub is not None) else np.zeros((0, 0), float),
        UA_rates_hz=np.asarray(UA_rates_hz_sub, float) if (HAS_BR and UA_rates_hz_sub is not None) else np.zeros((0, 0, 0), float),
        UA_rates_zeroed=np.asarray(UA_rates_zeroed_sub, float) if (HAS_BR and UA_rates_zeroed_sub is not None) else np.zeros((0, 0, 0), float),
        UA_counts=np.asarray(UA_counts_sub, float) if (HAS_BR and UA_counts_sub is not None) else np.zeros((0, 0, 0), float),
        UA_edges_ms=np.asarray(UA_edges_ms, float) if (HAS_BR and UA_edges_ms is not None) else np.zeros(0, float),
        UA_rel_t=np.asarray(UA_rel_t, float) if (HAS_BR and UA_rel_t is not None) else np.zeros(0, float),
        UA_peak_ms_dedup=UA_peak_ms_dedup if (HAS_BR and UA_peak_ms_dedup is not None) else {},
        UA_amps_ms_dedup=UA_amps_ms_dedup if (HAS_BR and UA_amps_ms_dedup is not None) else {},
        ua_ids_1based=np.asarray(ua_ids_1based, int) if (HAS_BR and ua_ids_1based is not None) else np.array([], int),

        ua_region=ua_region,
        ua_region_names=ua_region_names,
        ua_port=ua_port,
        ua_nsp=ua_nsp,
        ua_idx_rows=ua_idx_rows,

        n_trials=int(np.asarray(event_ms_sub).size),
    )
    if SAVE_MAT_FILE:
        mat = {
            "sess": np.array(intan_filename, dtype=object),
            "br_idx": int(br_idx),
            "overall_title": np.array(overall_title, dtype=object),
            "stim_ms": np.asarray(event_ms_sub, float),
            "trial_labels": np.asarray(labels_sub, dtype="U1"),
            "trial_labels_all": np.asarray(labels_all, dtype="U1"),

            "ua_meta": ua_meta if ua_meta is not None else {},
            "nprw_meta": nprw_meta,
            "align_meta_raw": align_meta_raw,

            "n_beh": int(n_beh_sub),
            "beh_rel_t": np.asarray(beh_rel_t_sub, float),
            "beh_cam0_pos_med": np.asarray(beh_cam0_pos_med_sub, float),
            "beh_cam1_pos_med": np.asarray(beh_cam1_pos_med_sub, float),
            "beh_cam0_vel_med": np.asarray(beh_cam0_vel_med_sub, float),
            "beh_cam1_vel_med": np.asarray(beh_cam1_vel_med_sub, float),
            "beh_cam0_segs": np.asarray(beh_cam0_segs_sub, float),
            "beh_cam1_segs": np.asarray(beh_cam1_segs_sub, float),
            "beh_cam0_vel_segs": np.asarray(beh_cam0_vel_segs_sub, float),
            "beh_cam1_vel_segs": np.asarray(beh_cam1_vel_segs_sub, float),
            "beh_cam0_names": np.array(list(beh_cam0_names), dtype=object),
            "beh_cam1_names": np.array(list(beh_cam1_names), dtype=object),

            "hr_sig": hr_sig,
            "vog_sig": vog_sig,

            "ts_state_rel_t": np.asarray(ts_state_rel_t, float),
            "ts_state_segs": np.asarray(ts_state_segs_sub, float),
            "ts_state_char_segs": np.asarray(ts_state_char_segs_sub, dtype="U1"),
            "ts_state_char": ts_state_char if ts_state_char is not None else np.array([], dtype="U1"),
            "ts_state_num": np.asarray(ts_state_num_full, int) if ts_state_num_full is not None else np.array([], int),
            "n_ts_state_trials": int(n_ts_state_trials_sub),

            "NPRW_med": np.asarray(NPRW_med_sub, float),
            "NPRW_med_counts": np.asarray(NPRW_med_counts_sub, float),
            "NPRW_var": np.asarray(NPRW_var_sub, float),
            "NPRW_rates_hz": np.asarray(NPRW_rates_hz_sub, float),
            "NPRW_rates_zeroed": np.asarray(NPRW_rates_zeroed_sub, float),
            "NPRW_counts": np.asarray(NPRW_counts_sub, float),
            "NPRW_edges_ms": np.asarray(NPRW_edges_ms, float),
            "NPRW_rel_t": np.asarray(NPRW_rel_t, float),

            "HAS_BR": bool(HAS_BR),
            "UA_med": np.asarray(UA_med_sub, float) if (HAS_BR and UA_med_sub is not None) else np.zeros((0, 0), float),
            "UA_med_counts": np.asarray(UA_med_counts_sub, float) if (HAS_BR and UA_med_counts_sub is not None) else np.zeros((0, 0), float),
            "UA_var": np.asarray(UA_var_sub, float) if (HAS_BR and UA_var_sub is not None) else np.zeros((0, 0), float),
            "UA_rates_hz": np.asarray(UA_rates_hz_sub, float) if (HAS_BR and UA_rates_hz_sub is not None) else np.zeros((0, 0, 0), float),
            "UA_rates_zeroed": np.asarray(UA_rates_zeroed_sub, float) if (HAS_BR and UA_rates_zeroed_sub is not None) else np.zeros((0, 0, 0), float),
            "UA_counts": np.asarray(UA_counts_sub, float) if (HAS_BR and UA_counts_sub is not None) else np.zeros((0, 0, 0), float),
            "UA_edges_ms": np.asarray(UA_edges_ms, float) if (HAS_BR and UA_edges_ms is not None) else np.zeros(0, float),
            "UA_rel_t": np.asarray(UA_rel_t, float) if (HAS_BR and UA_rel_t is not None) else np.zeros(0, float),
            "ua_ids_1based": np.asarray(ua_ids_1based, int) if (HAS_BR and ua_ids_1based is not None) else np.array([], int),

            "ua_region": ua_region,
            "ua_region_names": ua_region_names,
            "ua_port": ua_port,
            "ua_nsp": ua_nsp,
            "ua_idx_rows": ua_idx_rows,

            "n_trials": int(np.asarray(event_ms_sub).size),
        }
        savemat(out_mat, mat, do_compression=True)

def extract_one_file(aligned_path: Path, out_dir: Path, use_ir_ms: bool = False, split_targets: bool = True) -> None:
    """
    Load aligned .npz, compute peri-stim med/var for NPRW+UA,
    + behavior/VOG/ts_state peri-stim summaries, and save to PERI_ROOT.
    """
    # Load NPZ
    aligned_npz     = np.load(aligned_path, allow_pickle=True)
    
    # meta
    meta            = json.loads(aligned_npz["align_meta"].item()) if "align_meta" in aligned_npz.files else {}
    intan_filename  = meta.get("intan_filename", aligned_path.stem)
    br_idx          = int(meta.get("br_idx", -1))
    stim_dur        = float(meta.get("recording_stim_dur", 0.0))
    
    nprw_meta       = aligned_npz["nprw_meta"].item() #TODO 
    nprw_peak_ms    = aligned_npz["nprw_peak_ms"].reshape(-1)[0]
    nprw_peak_amps  = aligned_npz["nprw_peak_amps"].reshape(-1)[0]
    nprw_rec_ms     = (float(nprw_meta["rec_start_ms_aligned"]), float(nprw_meta["rec_end_ms_aligned"]))
    nprw_rec_dur    = nprw_meta['rec_dur']
    
    if use_ir_ms == True:
        event_ms = aligned_npz["ir_ms"]
    else:
        event_ms = aligned_npz["stim_ms"]
    if event_ms.size == 0:
        print(f"[extract] {aligned_path.name}: no events to align to, skipping peri-stim extraction.")
        return
    
    # ---- metadata for titles ----
    try:
        overall_title, _ = build_title_from_csv( METADATA_CSV, br_file=meta.get("br_idx"))
    except Exception as e:
        print(f"[warn] metadata parse failed for BR {br_idx}: {e}")
        overall_title = "Condition: n/a, n/a Hz, n/a µA, n/a mm, n/a ms, Delay: 0 ms"
    
    if HAS_BR:
        ua_meta         = aligned_npz["ua_meta"].item()
        ua_peak_ms      = aligned_npz["ua_peak_ms"].reshape(-1)[0]
        ua_peak_amps    = aligned_npz["ua_peak_amps"].reshape(-1)[0]
        ua_rec_ms       = (float(ua_meta["rec_start_ms"]), float(ua_meta["rec_end_ms"]))

        ua_rec_dur      = ua_meta['rec_dur']
        fs_ua           = ua_meta['fs_ua']
        fs_ts           = ua_meta['fs_ts']
        ua_samples      = ua_meta['n_samples']
        UA_t            = ua_rec_ms[0] + np.arange(ua_samples) * (1000.0 / fs_ua)
        
        # UA ids
        ua_ids_1based = aligned_npz["ua_elec"]
        if ua_ids_1based is not None:
            ua_ids_1based = np.asarray(ua_ids_1based, dtype=int).ravel()

        # UA meta
        ua_region = aligned_npz["ua_region"]
        ua_region_names = aligned_npz["ua_region_names"]
        ua_port = aligned_npz["ua_port"]
        ua_nsp = aligned_npz["ua_nsp"]
        ua_idx_rows = aligned_npz["ua_idx_rows"]
        
        hr_sig  = aligned_npz["hr_sig"]  if "hr_sig"  in aligned_npz.files else np.array([], float)
        vog_sig = aligned_npz["vog_sig"] if "vog_sig" in aligned_npz.files else np.array([], float)
       
    if HAS_KINEMATICS:
        beh_cam0        = aligned_npz["beh_cam0"]
        beh_cam1        = aligned_npz["beh_cam1"]
        raw0            = aligned_npz["beh_cam0_cols"]
        raw1            = aligned_npz["beh_cam1_cols"]
        beh_cam0_cols   = [str(x) for x in _as_list(raw0)]
        beh_cam1_cols   = [str(x) for x in _as_list(raw1)]
        behv_t          = aligned_npz["beh_t_ms"].astype(float) if "beh_t_ms" in aligned_npz.files else np.arange(0.0)

        # ---- basic behavior checks ----
        if behv_t.size == 0:
            print(f"[extract] {aligned_path.name}: no behavior time axis; skipping behavior.")
        # interpolate small NaN gaps
        if beh_cam0.size:
            beh_cam0 = _interp_nans_2d_by_col(beh_cam0, max_gap=4)
        if beh_cam1.size:
            beh_cam1 = _interp_nans_2d_by_col(beh_cam1, max_gap=4)

        cam0_M, beh_cam0_names = _select_matrix(beh_cam0, beh_cam0_cols)
        cam1_M, beh_cam1_names = _select_matrix(beh_cam1, beh_cam1_cols)

        cam0_z = _z_per_column(cam0_M) if cam0_M.size else cam0_M
        cam1_z = _z_per_column(cam1_M) if cam1_M.size else cam1_M

        # ---- behavior-based stim gating: drop bad behavior trials everywhere ----
        if behv_t.size and cam0_z.size:
            beh_mask = _behavior_valid_mask_from_cam0(
                cam0_z,
                behv_t,
                event_ms,
                win_ms=WIN_MS,
                baseline_ms=NORMALIZE_FIRST_MS,
                min_trials=MIN_TRIALS,
            )
            n_keep = int(beh_mask.sum())
            
            if 0 < n_keep < event_ms.size:
                print(f"[extract] {aligned_path.name}: behavior gating kept {n_keep}/{event_ms.size} stims.")
                event_ms = event_ms[beh_mask]
                if "trial_labels" in locals() and trial_labels is not None and np.size(trial_labels) == beh_mask.size:
                    trial_labels = np.asarray(trial_labels)[beh_mask]
        
        cam0_pos_filt, cam0_vel, _ = rcp.butter_lowpass_pos_and_vel(cam0_z, behv_t, cutoff_hz=10.0, order=3)
        cam1_pos_filt, cam1_vel, _ = rcp.butter_lowpass_pos_and_vel(cam1_z, behv_t, cutoff_hz=10.0, order=3)
    
    behv_dur = _last(behv_t) if ("behv_t" in locals() and np.size(behv_t)) else float("nan")
    nprw_dur = float(nprw_rec_dur) if ("nprw_rec_dur" in locals() and nprw_rec_dur is not None) else float("nan")
    ua_dur   = float(ua_rec_dur)   if ("ua_rec_dur"   in locals() and ua_rec_dur   is not None) else float("nan")

    print(
        f"[debug] {aligned_path.name}: "
        f"Behavioral rec duration={behv_dur:.3f} ms, "
        f"NPRW rec duration={nprw_dur:.3f} ms, "
        f"UA rec duration={ua_dur:.3f} ms"
    )
        
    # VOG
    # if HAS_VOG:
    #     vog_cols      = aligned_npz["vog_cols"]
    #     vog_t_ms      = aligned_npz["vog_t_ms"]
    #     vog_raw_names = aligned_npz["vog_col_names"]
    #     vog_col_names = [str(x) for x in _as_list(vog_raw_names)]
    
    #     # ---- VOG peri-stim (median lines + segments) ----
    #     if vog_cols.size and vog_t_ms.size:
    #         vog_cols = _interp_nans_2d_by_col(vog_cols, max_gap=4)
    #         vog_z = _z_per_column(vog_cols)

    #         # keep only stims whose [stim+WIN0, stim+WIN1] lies fully within VOG range
    #         stim_vog = _filter_stims_for_stream(event_ms, vog_t_ms, WIN_MS)

    #         vog_med, vog_rel_t, n_vog_trials, vog_segs = _median_lines_for_columns(vog_z, vog_t_ms, stim_vog)
    #     else:
    #         vog_med       = np.zeros((0, 0), float)
    #         vog_rel_t     = np.zeros(0, float)
    #         vog_segs      = np.zeros((0, 0, 0), float)
    #         n_vog_trials  = 0
    
    # labels corresponding to the original behavior-gated event_ms
    if "trial_labels" not in locals():
        trial_labels = np.full(event_ms.shape, "N", dtype="U1")

    # Mask valid trials
    if HAS_BR:
        mask_valid = _get_valid_events(event_ms, ua_rec_ms, WIN_MS)
    else:
        mask_valid = _get_valid_events(event_ms, nprw_rec_ms, WIN_MS)

    event_ms = np.asarray(event_ms[mask_valid], float)
    trial_labels = np.asarray(trial_labels[mask_valid], dtype="U1")  # aligns with events_valid
    
    # Touchscreen state based target selection
    if HAS_BR:
        ts_state_rel_t      = np.zeros(0, float)
        ts_state_segs       = np.zeros((0, 0), float)
        ts_state_char_segs  = np.zeros((0, 0), dtype="U1")
        n_ts_state_trls     = 0

        ts_state_num  = aligned_npz["ts_state_num"]
        ts_state_char = aligned_npz["ts_state_char"]

        ts_state_num_full = None
        ts_state_char_arr = None
        if ts_state_num is not None and np.size(ts_state_num):
            ts_state_num_full = np.asarray(ts_state_num).astype(int).ravel()
        if ts_state_char is not None:
            ts_state_char_arr = np.asarray(ts_state_char)
            if ts_state_char_arr.ndim > 1:
                ts_state_char_arr = ts_state_char_arr.reshape(-1)

        ns2_t_ms = np.zeros(0, float)

        # Compute labels A/B/N per stim from ts_state
        if ts_state_num_full is not None and event_ms.size:
            ns2_t_ms = ua_rec_ms[0] + np.arange(ts_state_num_full.size, dtype=float) * (1000.0 / fs_ts)

            trial_labels = _compute_trial_labels(
                event_ms=event_ms,
                t_ms=ns2_t_ms,
                ts_char=ts_state_char_arr,
                ts_num=ts_state_num_full,
                win_ms=(-500.0, 0.0),
                num_to_label={1: "A", 2: "B"},
            )
        else:
            trial_labels = np.full(event_ms.shape, "N", dtype="U1")

        vals, cnts = np.unique(trial_labels, return_counts=True)
        print(dict(zip(vals, cnts)))

        # extract ts_state peristim segments
        if ts_state_num_full is not None and ns2_t_ms.size and event_ms.size:
            try:
                ts_rate_segs, _, ts_state_rel_t, event_valid_ts = rcp.extract_peristim_segments(
                    rate_hz=ts_state_num_full.reshape(1, -1),
                    counts=None,
                    t_ms=ns2_t_ms,
                    stim_ms=event_ms,
                    win_ms=WIN_MS,
                    min_trials=MIN_TRIALS,
                )
            except RuntimeError as e:
                if "Only 0 peri-stim segments" in str(e):
                    ts_rate_segs = np.zeros((0, 1, 0), float)
                    event_valid_ts = np.zeros(0, float)
                else:
                    raise

            if ts_rate_segs.size:
                ts_state_segs = ts_rate_segs[:, 0, :]
                n_ts_state_trls = int(ts_state_segs.shape[0])

                # Select correct events
                event_valid_ts = np.asarray(event_valid_ts, float).ravel()
                idx_map = {float(s): i for i, s in enumerate(np.asarray(event_ms, float).ravel())}
                keep_idx = np.array([idx_map.get(float(s), -1) for s in event_valid_ts], dtype=int)
                keep_idx = keep_idx[keep_idx >= 0]
                event_ms = event_ms[keep_idx]
                trial_labels = trial_labels[keep_idx]
            else:
                ts_state_segs = np.zeros((0, 0), float)
                ts_state_char_segs = np.zeros((0, 0), dtype="U1")
                n_ts_state_trls = 0

        if ts_state_segs.size and ts_state_char_arr is not None and ts_state_num_full is not None:
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
        else:
            ts_state_char_segs = np.zeros_like(ts_state_segs, dtype="U1")

        idx_A = np.where(trial_labels == "A")[0]
        idx_B = np.where(trial_labels == "B")[0]

        ts_state_segs_A = ts_state_segs[idx_A] if ts_state_segs.size else np.zeros_like(ts_state_segs[:0, :])
        ts_state_segs_B = ts_state_segs[idx_B] if ts_state_segs.size else np.zeros_like(ts_state_segs[:0, :])

        ts_state_char_segs_A = ts_state_char_segs[idx_A] if ts_state_char_segs.size else np.zeros_like(ts_state_char_segs[:0, :])
        ts_state_char_segs_B = ts_state_char_segs[idx_B] if ts_state_char_segs.size else np.zeros_like(ts_state_char_segs[:0, :])

        n_ts_state_trls_A = int(idx_A.size)
        n_ts_state_trls_B = int(idx_B.size)

        # dedup peaks
        ua_peak_ms_dedup, ua_amps_ms_dedup = _dedup_peaks(ua_peak_ms, ua_peak_amps, UA_DEDUP_MS, MAX_CLUSTER_MS)

        # bin ONLY on valid stims per stream
        ua_counts, ua_rel_t, ua_edges_ms, ua_left_bins = _bin_counts_around_stim(
            ua_peak_ms_dedup, UA_BIN_MS, event_ms,
            UA_MS_BEFORE, (stim_dur + UA_TAIL_MS), WIN_MS
        )
        ua_rates_hz = _smooth_counts_gauss(ua_counts, ua_edges_ms, ua_rel_t, UA_SIGMA_MS, ua_left_bins)
        ua_rates_hz_baselined = rcp.baseline_zero_each_trial(ua_rates_hz, ua_rel_t, normalize_first_ms=NORMALIZE_FIRST_MS)

    # dedup peaks
    nprw_peak_ms_dedup, nprw_amps_ms_dedup = _dedup_peaks(nprw_peak_ms, nprw_peak_amps, NPRW_DEDUP_MS, MAX_CLUSTER_MS)

    # bin ONLY on valid stims per stream
    nprw_counts, nprw_rel_t, nprw_edges_ms, nprw_left_bins = _bin_counts_around_stim(
        nprw_peak_ms_dedup, NPRW_BIN_MS, event_ms,
        NPRW_MS_BEFORE, (stim_dur + NPRW_TAIL_MS), WIN_MS
    )
    nprw_rates_hz = _smooth_counts_gauss(nprw_counts, nprw_edges_ms, nprw_rel_t, NPRW_SIGMA_MS, nprw_left_bins)
    nprw_rates_hz_baselined = rcp.baseline_zero_each_trial(nprw_rates_hz, nprw_rel_t, normalize_first_ms=NORMALIZE_FIRST_MS)

    # 2) Define subsets: A/B or all-events
    if split_targets and HAS_BR:
        subsets = [
            ("A", np.where(trial_labels == "A")[0]),
            ("B", np.where(trial_labels == "B")[0]),
        ]
    else:
        subsets = [(None, np.arange(event_ms.size))]

    # 4) Loop subsets and save
    for target_name, idx in subsets:
        idx = np.asarray(idx, int)
        if idx.size == 0:
            continue

        event_ms_sub = event_ms[idx]
        labels_sub   = trial_labels[idx]

        # ---- subset ts_state segments (if you already computed ts_state_segs_A/B) ----
        # If split_targets: use your precomputed *_A/*_B (preferred)
        # Else: keep full ts_state_segs (valid subset), or empty if you didn’t compute it.
        if split_targets:
            if target_name == "A":
                ts_state_segs_sub = ts_state_segs_A
                ts_state_char_segs_sub = ts_state_char_segs_A
                n_ts_state_trials_sub = int(n_ts_state_trls_A)
            else:
                ts_state_segs_sub = ts_state_segs_B
                ts_state_char_segs_sub = ts_state_char_segs_B
                n_ts_state_trials_sub = int(n_ts_state_trls_B)
        else:
            # AtRest: easiest is to NOT split ts_state; store the overall (already computed) ts_state_segs
            ts_state_segs_sub = ts_state_segs
            ts_state_char_segs_sub = ts_state_char_segs
            n_ts_state_trials_sub = int(n_ts_state_trls)

        # ---- subset UA/NPRW trials ----
        NPRW_rates_hz_sub     = nprw_rates_hz[idx]
        NPRW_rates_zeroed_sub = nprw_rates_hz_baselined[idx]
        NPRW_counts_sub       = nprw_counts[idx]

        NPRW_med_counts_sub = np.nanmedian(NPRW_counts_sub, axis=0)
        NPRW_med_sub = np.nanmedian(NPRW_rates_zeroed_sub, axis=0)
        NPRW_var_sub = np.nanvar(NPRW_rates_zeroed_sub, axis=0)

        # UA (if present): same logic
        if HAS_BR:
            UA_rates_hz_sub     = ua_rates_hz[idx]
            UA_rates_zeroed_sub = ua_rates_hz_baselined[idx]
            UA_counts_sub       = ua_counts[idx]

            UA_med_counts_sub = np.nanmedian(UA_counts_sub, axis=0)
            UA_med_sub = np.nanmedian(UA_rates_zeroed_sub, axis=0)
            UA_var_sub = np.nanvar(UA_rates_zeroed_sub, axis=0)
        else:
            UA_rates_zeroed_sub = None
            UA_counts_sub = None
            UA_med_counts_sub = None
            UA_med_sub = None
            UA_var_sub = None

        # ---- behavior medians recomputed directly from event_ms_sub ----
        if HAS_KINEMATICS and behv_t.size and cam0_z.size:
            beh_cam0_pos_med_sub, beh_rel_t_sub, n_beh_sub, beh_cam0_segs_sub = _median_lines_for_columns(
                cam0_z, behv_t, event_ms_sub
            )
            beh_cam1_pos_med_sub, _, _, beh_cam1_segs_sub = _median_lines_for_columns(
                cam1_z, # cam1_z exists if you built it above
                behv_t, event_ms_sub
            )
            beh_cam0_vel_med_sub, _, _, beh_cam0_vel_segs_sub = _median_lines_for_columns(
                cam0_vel, behv_t, event_ms_sub
            )
            beh_cam1_vel_med_sub, _, _, beh_cam1_vel_segs_sub = _median_lines_for_columns(
                cam1_vel, behv_t, event_ms_sub
            )
        else:
            beh_rel_t_sub = np.zeros(0, float)
            beh_cam0_pos_med_sub = np.zeros((0, 0), float)
            beh_cam1_pos_med_sub = np.zeros((0, 0), float)
            beh_cam0_vel_med_sub = np.zeros((0, 0), float)
            beh_cam1_vel_med_sub = np.zeros((0, 0), float)
            beh_cam0_segs_sub = np.zeros((0, 0, 0), float)
            beh_cam1_segs_sub = np.zeros((0, 0, 0), float)
            beh_cam0_vel_segs_sub = np.zeros((0, 0, 0), float)
            beh_cam1_vel_segs_sub = np.zeros((0, 0, 0), float)
            n_beh_sub = 0
            
        if split_targets and target_name is not None:
            out_dir = out_dir / f"target_{target_name}"   # .../target_A
            
        # save NPZ + MAT
        _save_peristim(
            out_dir=out_dir,
            target_name=target_name,
            intan_filename=intan_filename,
            br_idx=int(br_idx),
            overall_title=overall_title,

            meta=meta,
            align_meta_raw=aligned_npz["align_meta"].item() if "align_meta" in aligned_npz.files else "{}",
            nprw_meta=nprw_meta,
            ua_meta=ua_meta if HAS_BR else None,

            NPRW_med_sub=NPRW_med_sub,
            NPRW_var_sub=NPRW_var_sub,
            NPRW_med_counts_sub=NPRW_med_counts_sub,
            NPRW_rates_hz_sub=NPRW_rates_hz_sub,
            NPRW_rates_zeroed_sub=NPRW_rates_zeroed_sub,
            NPRW_counts_sub=NPRW_counts_sub,
            NPRW_edges_ms=nprw_edges_ms,
            NPRW_rel_t=nprw_rel_t,
            NPRW_peak_ms_dedup=nprw_peak_ms_dedup,
            NPRW_amps_ms_dedup=nprw_amps_ms_dedup,

            HAS_BR=bool(HAS_BR),
            UA_med_sub=UA_med_sub,
            UA_var_sub=UA_var_sub,
            UA_med_counts_sub=UA_med_counts_sub,
            UA_rates_hz_sub=UA_rates_hz_sub,
            UA_rates_zeroed_sub=UA_rates_zeroed_sub,
            UA_counts_sub=UA_counts_sub,
            UA_edges_ms=ua_edges_ms if HAS_BR else None,
            UA_rel_t=ua_rel_t if HAS_BR else None,
            UA_peak_ms_dedup=ua_peak_ms_dedup if HAS_BR else None,
            UA_amps_ms_dedup=ua_amps_ms_dedup if HAS_BR else None,
            ua_ids_1based=ua_ids_1based if HAS_BR else None,

            ua_region=ua_region if HAS_BR else None,
            ua_region_names=ua_region_names if HAS_BR else None,
            ua_port=ua_port if HAS_BR else None,
            ua_nsp=ua_nsp if HAS_BR else None,
            ua_idx_rows=ua_idx_rows if HAS_BR else None,
            
            hr_sig=hr_sig if "hr_sig" in locals() else np.array([], float),
            vog_sig=vog_sig if "vog_sig" in locals() else np.array([], float),

            ts_state_rel_t=ts_state_rel_t,
            ts_state_segs_sub=ts_state_segs_sub,
            ts_state_char_segs_sub=ts_state_char_segs_sub,
            ts_state_char=ts_state_char,
            ts_state_num_full=ts_state_num_full if ts_state_num_full is not None else np.array([], int),
            n_ts_state_trials_sub=int(n_ts_state_trials_sub),

            event_ms_sub=event_ms_sub,
            labels_sub=labels_sub,
            labels_all=np.asarray(trial_labels, dtype="U1"),

            n_beh_sub=int(n_beh_sub),
            beh_rel_t_sub=beh_rel_t_sub,
            beh_cam0_pos_med_sub=beh_cam0_pos_med_sub,
            beh_cam1_pos_med_sub=beh_cam1_pos_med_sub,
            beh_cam0_vel_med_sub=beh_cam0_vel_med_sub,
            beh_cam1_vel_med_sub=beh_cam1_vel_med_sub,
            beh_cam0_segs_sub=beh_cam0_segs_sub,
            beh_cam1_segs_sub=beh_cam1_segs_sub,
            beh_cam0_vel_segs_sub=beh_cam0_vel_segs_sub,
            beh_cam1_vel_segs_sub=beh_cam1_vel_segs_sub,
            beh_cam0_names=beh_cam0_names if HAS_KINEMATICS else [],
            beh_cam1_names=beh_cam1_names if HAS_KINEMATICS else [],
        )

    print(f"[extract] wrote peristim__{intan_filename}__BR_{int(br_idx)}")
    
def main():
    control_files = sorted(CONTROL_ROOT.glob("aligned__*.npz"))
    stim_files = sorted(STIM_ROOT.glob("aligned__*.npz"))
    at_rest_files = sorted(AT_REST_ROOT.glob("aligned__*.npz"))
    
    if not control_files and not stim_files and not at_rest_files:
        raise SystemExit(f"[error] No combined aligned NPZs found at {ALIGNED_CKPT_ROOT}")
    
    # # normal: per-file processing
    for file in stim_files:
        extract_one_file(file, out_dir = STIM_PERI_ROOT, use_ir_ms=False, split_targets=True)
        # Extract files normally - done
        # Split A and B reaches - done
        # Aggregate and save - done
        
        # td = extract_trials_from_npz(f)
        # aggregate_and_save_normal(td, PERI_ROOT)

    # at_rest: per-file processing (no A/B)
    for file in at_rest_files:
        extract_one_file(file, out_dir = AT_REST_PERI_ROOT, use_ir_ms=False, split_targets=False)
        # Extract files without splitting by target - done
        # Aggregate and save - done
        
        # td = extract_trials_from_npz(f)
        # aggregate_and_save_at_rest(td, PERI_ROOT)

    # # control: split A/B, align to ir_ms
    for file in control_files:
        extract_one_file(file, out_dir = CONTROL_PERI_ROOT, use_ir_ms=True, split_targets=True)
        # Extract files normally but use ir_ms as alignment - done
        # Concatenate trials across files per group - TODO
        # Split A and B reaches - done
        # Aggregate and save per group - done
        
        # tds = [extract_trials_from_npz(f) for f in group_files]
        # td_concat = concat_trials(tds)
        # aggregate_and_save_baseline(td_concat, PERI_ROOT)

if __name__ == "__main__":
    main()
