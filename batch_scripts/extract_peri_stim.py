import re, json
from scipy.io import savemat
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

# Config
# Base paths from config_loading
ALIGNED_ROOT = ALIGNED_CKPT_ROOT


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
UA_DEDUP_MS = float(UA_RATES.get("dedup_ms", 0.5))
MAX_CLUSTER_MS = 0.5

# NOTE: Stimulation duration is now calculated dynamically per session from actual stim data
# instead of using a hardcoded 100ms value. This supports single-pulse and variable-duration stims.

KEYPOINTS_ORDER = tuple(PARAMS.kinematics.get("keypoints", []))  # [] if missing

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
                best = cluster_start + np.argmax(a_ch[cluster_start: i]) # Find max and give index
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
                          stim_ms: np.ndarray,
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
                              stim_ms: np.ndarray
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
        rate_segs, _, rel_t, stim_valid_ts = rcp.extract_peristim_segments(
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

    if rate_segs.size == 0 or stim_valid_ts.size == 0:
        return np.zeros(stim_all.shape, dtype=bool)

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

    # Now map back to a mask over the original stim_ms
    mask_beh = np.zeros(stim_all.shape, dtype=bool)
    stim_kept = stim_valid_ts[keep_valid]

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
    nprw_meta    = aligned_npz["nprw_meta"].item() #TODO 
    nprw_peak_ms = aligned_npz["nprw_peak_ms"].reshape(-1)[0]
    nprw_peak_amps = aligned_npz["nprw_peak_amps"].reshape(-1)[0]
    nprw_rec_ms = (float(nprw_meta["rec_start_ms_aligned"]), float(nprw_meta["rec_end_ms_aligned"]))
    nprw_rec_dur = nprw_meta['rec_dur']
    
    stim_ms = aligned_npz["stim_ms"]
    if stim_ms.size == 0:
        print(f"[extract] {aligned_path.name}: stim_ms is empty; skipping peri-stim extraction.")
        return
    
    ua_meta    = aligned_npz["ua_meta"].item()
    ua_peak_ms   = aligned_npz["ua_peak_ms"].reshape(-1)[0]
    ua_peak_amps   = aligned_npz["ua_peak_amps"].reshape(-1)[0]
    ua_rec_ms   = (float(ua_meta["rec_start_ms"]), float(ua_meta["rec_end_ms"]))

    ua_rec_dur = ua_meta['rec_dur']
    fs_ua = ua_meta['fs_ua']
    fs_ts = ua_meta['fs_ts']
    ua_samples = ua_meta['n_samples']
    UA_t = ua_rec_ms[0] + np.arange(ua_samples) * (1000.0 / fs_ua)
    
    # alignment meta (JSON)
    meta = json.loads(aligned_npz["align_meta"].item()) if "align_meta" in aligned_npz.files else {}
    sess   = meta.get("session", aligned_path.stem)
    br_idx = int(meta.get("br_idx", -1))
    
    # ---- Calculate actual stimulation duration from stim stream data ----
    # This replaces the old hardcoded STIM_DUR = 100.0 to support single pulses and variable durations
    stim_dur_ms = 0.0  # Default for sessions without stim
    intan_idx = int(meta.get("intan_idx", -1))
    
    # Try to load stim stream data to get actual pulse durations
    stim_npz_path = None
    if br_idx >= 0:
        try:
            stim_npz_path, _ = rcp.stim_npz_path_from_br_idx(br_idx, METADATA_CSV, NPRW_AUX_DATA)
        except Exception:
            pass
    
    if stim_npz_path is not None and Path(stim_npz_path).exists():
        try:
            stim_data = rcp.load_stim_detection(stim_npz_path)
            block_bounds_samp = stim_data.get("block_bounds_samples", np.empty((0, 2)))
            stim_meta_str = stim_data.get("meta", "{}")
            stim_meta = json.loads(stim_meta_str) if isinstance(stim_meta_str, str) else stim_meta_str
            fs_stim = float(stim_meta.get("fs_hz", meta.get("fs_nprw", 30000.0)))
            
            if block_bounds_samp.size > 0:
                # Calculate duration of each stim block/pulse
                durations_ms = (block_bounds_samp[:, 1] - block_bounds_samp[:, 0]) * 1000.0 / fs_stim
                stim_dur_ms = float(np.max(durations_ms))  # Use max for artifact window
                print(f"[stim_dur] {aligned_path.name}: Detected stim duration = {stim_dur_ms:.2f} ms (from {len(durations_ms)} blocks)")
            else:
                print(f"[stim_dur] {aligned_path.name}: No stim blocks found")
        except Exception as e:
            print(f"[stim_dur] {aligned_path.name}: Could not load stim data: {e}; using default duration=0.0")
    else:
        print(f"[stim_dur] {aligned_path.name}: No stim NPZ found; using default duration=0.0")


    # ---- load aligned file for behavior / VOG / ts_state ----
    beh_cam0      = aligned_npz["beh_cam0"]
    beh_cam1      = aligned_npz["beh_cam1"]
    raw0      = aligned_npz["beh_cam0_cols"]
    raw1      = aligned_npz["beh_cam1_cols"]
    beh_cam0_cols = [str(x) for x in _as_list(raw0)]
    beh_cam1_cols = [str(x) for x in _as_list(raw1)]
    behv_t    = aligned_npz["beh_t_ms"].astype(float) if "beh_t_ms" in aligned_npz.files else np.arange(0.0)

    # VOG
    # vog_cols      = aligned_npz["vog_cols"]
    # vog_t_ms      = aligned_npz["vog_t_ms"]
    # vog_raw_names = aligned_npz["vog_col_names"]
    # vog_col_names = [str(x) for x in _as_list(vog_raw_names)]

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
    
    hr_sig = aligned_npz["hr_sig"]
    vog_sig = aligned_npz["vog_sig"]
    
    # ts_state on UA timebase
    ts_state_num = aligned_npz["ts_state_num"]
    ts_state_char = aligned_npz["ts_state_char"]

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
        f"[debug] {aligned_path.name}: "
        # f"vog_t_end={_last(vog_t_ms):.3f} ms, "
        f"Behavioral rec duration={_last(behv_t):.3f} ms, "
        f"NPRW rec duration={_last(nprw_rec_dur):.3f} ms, "
        f"UA rec duration={_last(ua_rec_dur):.3f} ms"
    )

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
            # behavior gating
            if 0 < beh_mask.sum() < stim_ms.size:
                stim_ms = stim_ms[beh_mask]
        # if n_keep == stim_ms.size: nothing to change
    
    cam0_pos_filt, cam0_vel, _ = rcp.butter_lowpass_pos_and_vel(cam0_z, behv_t, cutoff_hz=10.0, order=3)
    cam1_pos_filt, cam1_vel, _ = rcp.butter_lowpass_pos_and_vel(cam1_z, behv_t, cutoff_hz=10.0, order=3)

    # ---- compute labels A/B/N per stim from ts_state on UA timebase (after gating) ----
    if ts_state_num_full is not None and stim_ms.size:
        ns2_t_ms = ua_rec_ms[0] + np.arange(ts_state_num_full.size, dtype=float) * (1000.0 / fs_ts) #TODO can make more general
        trial_labels = _compute_trial_labels(
            event_ms=stim_ms,
            t_ms=ns2_t_ms,
            ts_char=ts_state_char_arr,
            ts_num=ts_state_num_full,
            win_ms=(-500.0, 0.0),
            num_to_label={1: "A", 2: "B"},
        )
    else:
        trial_labels = np.full(stim_ms.shape, "N", dtype="U1")

    vals, cnts = np.unique(trial_labels, return_counts=True)
    print(dict(zip(vals, cnts)))
    
    # ---- A/B behavior medians (position) ----
    beh_cam0_pos_med_A, beh_rel_t_A, n_beh_A, beh_cam0_segs_A = _behavior_medians_for_label(cam0_z, behv_t, stim_ms, trial_labels, "A")
    beh_cam0_pos_med_B, beh_rel_t_B, n_beh_B, beh_cam0_segs_B = _behavior_medians_for_label(cam0_z, behv_t, stim_ms, trial_labels, "B")

    beh_cam1_pos_med_A, _, _, beh_cam1_segs_A = _behavior_medians_for_label(cam1_z, behv_t, stim_ms, trial_labels, "A")
    beh_cam1_pos_med_B, _, _, beh_cam1_segs_B = _behavior_medians_for_label(cam1_z, behv_t, stim_ms, trial_labels, "B")

    # ---- A/B behavior medians (velocity) ----
    beh_cam0_vel_med_A, _, _, beh_cam0_vel_segs_A = _behavior_medians_for_label(cam0_vel, behv_t, stim_ms, trial_labels, "A")
    beh_cam0_vel_med_B, _, _, beh_cam0_vel_segs_B = _behavior_medians_for_label(cam0_vel, behv_t, stim_ms, trial_labels, "B")

    beh_cam1_vel_med_A, _, _, beh_cam1_vel_segs_A = _behavior_medians_for_label(cam1_vel, behv_t, stim_ms, trial_labels, "A")
    beh_cam1_vel_med_B, _, _, beh_cam1_vel_segs_B = _behavior_medians_for_label(cam1_vel, behv_t, stim_ms, trial_labels, "B")

    # X = rate_hz.T  # (n_bins, n_channels)
    # X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # n_bins, n_ch = X.shape
    # n_comp = min(5, n_bins, n_ch)

    # total_var = np.var(X, axis=0).sum()
    # if n_comp >= 1 and total_var > 0.0:
    #     pca = PCA(n_components=n_comp, random_state=0)
    #     pcs = pca.fit_transform(X)  # (n_bins, n_comp)
    #     explained_var = np.nan_to_num(
    #         pca.explained_variance_ratio_, nan=0.0
    #     ).astype(np.float32)
    #     pcs_T = pcs.T.astype(np.float32)  # (n_comp, n_bins)
    # else:
    #     pcs_T = np.empty((0, n_bins), dtype=np.float32)
    #     explained_var = np.empty((0,), dtype=np.float32)
        
    # # ---- VOG peri-stim (median lines + segments) ----
    # if vog_cols.size and vog_t_ms.size:
    #     vog_cols = _interp_nans_2d_by_col(vog_cols, max_gap=4)
    #     vog_z = _z_per_column(vog_cols)

    #     # keep only stims whose [stim+WIN0, stim+WIN1] lies fully within VOG range
    #     stim_vog = _filter_stims_for_stream(stim_ms, vog_t_ms, WIN_MS)

    #     vog_med, vog_rel_t, n_vog_trials, vog_segs = _median_lines_for_columns(vog_z, vog_t_ms, stim_vog)
    # else:
    #     vog_med       = np.zeros((0, 0), float)
    #     vog_rel_t     = np.zeros(0, float)
    #     vog_segs      = np.zeros((0, 0, 0), float)
    #     n_vog_trials  = 0


    # ---- ts_state peri-stim segments (numeric + char) ----
    
        # ts_state_num_binned = np.zeros(t_cat_ms.shape, dtype=np.int8)
        # ts_state_char_binned = np.full(t_cat_ms.shape, 'N', dtype='U1')

        # if ts_state_num.size > 0:
        #     # t_cat_ms is in milliseconds since start of BR recording
        #     # ts_state_num is per touchscreen sample at ts_fs_hz
        #     # Map each bin center to nearest touchscreen sample index
        #     ts_idx = np.round((t_cat_ms / 1000.0) * FS_NS2).astype(np.int64)

        #     # Clip to valid range
        #     ts_idx = np.clip(ts_idx, 0, ts_state_num.size - 1)

        #     ts_state_num_binned = ts_state_num[ts_idx]
        #     ts_state_char_binned = ts_state_char[ts_idx]

        #     print(
        #         f"[touchscreen] Binned states: shape={ts_state_num_binned.shape}, "
        #         f"unique={np.unique(ts_state_num_binned)}"
        #     )
        # else:
        #     print("[touchscreen] No touchscreen states to bin; using all 'N'.")


    ts_state_rel_t      = np.zeros(0, float)
    ts_state_segs       = np.zeros((0, 0), float)
    ts_state_char_segs  = np.zeros((0, 0), dtype="U1")
    n_ts_state_trls     = 0

    if ts_state_num_full is not None and np.size(UA_t):
        try:
            ts_rate_segs, _, ts_state_rel_t, _ = rcp.extract_peristim_segments(
                rate_hz=ts_state_num_full.reshape(1, -1),
                counts=None,
                t_ms=ns2_t_ms,
                stim_ms=stim_ms,
                win_ms=WIN_MS,
                min_trials=MIN_TRIALS,
            )
            if ts_rate_segs.size:
                ts_state_segs = ts_rate_segs[:, 0, :]
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
                    
    ts_state_segs_A = np.zeros_like(ts_state_segs[:0, :])
    ts_state_segs_B = np.zeros_like(ts_state_segs[:0, :])
    ts_state_char_segs_A = np.zeros_like(ts_state_char_segs[:0, :])
    ts_state_char_segs_B = np.zeros_like(ts_state_char_segs[:0, :])
    n_ts_state_trls_A = 0
    n_ts_state_trls_B = 0

    if ts_state_segs.size and stim_ms.size and UA_t.size:
        mask_ts = _get_valid_events(stim_ms, ua_rec_ms, WIN_MS)
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
            
    # stream-valid stim lists (after behavior gating)
    mask_nprw = _get_valid_events(stim_ms, nprw_rec_ms, WIN_MS)
    mask_ua   = _get_valid_events(stim_ms, ua_rec_ms,   WIN_MS)

    stim_ms_ua_all   = stim_ms[mask_ua]
    stim_ms_nprw_all = stim_ms[mask_nprw]

    # dedup peaks
    ua_peak_ms_dedup, ua_amps_ms_dedup = _dedup_peaks(ua_peak_ms, ua_peak_amps, UA_DEDUP_MS, MAX_CLUSTER_MS)
    nprw_peak_ms_dedup, nprw_amps_ms_dedup = _dedup_peaks(nprw_peak_ms, nprw_peak_amps, NPRW_DEDUP_MS, MAX_CLUSTER_MS)

    # bin ONLY on valid stims per stream
    ua_counts, ua_rel_t, ua_edges_ms, ua_left_bins = _bin_counts_around_stim(
        ua_peak_ms_dedup, UA_BIN_MS, stim_ms_ua_all,
        UA_MS_BEFORE, (stim_dur_ms + UA_TAIL_MS), WIN_MS
    )
    nprw_counts, nprw_rel_t, nprw_edges_ms, nprw_left_bins = _bin_counts_around_stim(
        nprw_peak_ms_dedup, NPRW_BIN_MS, stim_ms_nprw_all,
        NPRW_MS_BEFORE, (stim_dur_ms + NPRW_TAIL_MS), WIN_MS
    )

    ua_rate_hz = _smooth_counts_gauss(ua_counts, ua_edges_ms, ua_rel_t, UA_SIGMA_MS, ua_left_bins)
    nprw_rates_hz = _smooth_counts_gauss(nprw_counts, nprw_edges_ms, nprw_rel_t, NPRW_SIGMA_MS, nprw_left_bins)

    ua_rate_hz_baselined = rcp.baseline_zero_each_trial(ua_rate_hz, ua_rel_t, normalize_first_ms=NORMALIZE_FIRST_MS)
    nprw_rates_hz_baselined = rcp.baseline_zero_each_trial(nprw_rates_hz, nprw_rel_t, normalize_first_ms=NORMALIZE_FIRST_MS)

    # labels corresponding to the original behavior-gated stim_ms
    trial_labels = np.asarray(trial_labels, dtype="U1")

    # restrict labels to each stream’s valid stim subset
    labels_ua   = trial_labels[mask_ua]
    labels_nprw = trial_labels[mask_nprw]

    idxA_ua   = np.where(labels_ua == "A")[0]
    idxB_ua   = np.where(labels_ua == "B")[0]
    idxA_nprw = np.where(labels_nprw == "A")[0]
    idxB_nprw = np.where(labels_nprw == "B")[0]

    stim_ms_A_ua   = stim_ms_ua_all[idxA_ua]
    stim_ms_B_ua   = stim_ms_ua_all[idxB_ua]
    stim_ms_A_nprw = stim_ms_nprw_all[idxA_nprw]
    stim_ms_B_nprw = stim_ms_nprw_all[idxB_nprw]
    
    # UA
    n_ua_A = len(idxA_ua)
    UA_rates_zeroed_A  = ua_rate_hz_baselined[idxA_ua]
    UA_counts_A        = ua_counts[idxA_ua]
    UA_med_counts_A    = np.nanmedian(UA_counts_A, axis=0)
    UA_med_A           = np.nanmedian(UA_rates_zeroed_A, axis=0)
    UA_var_A           = np.nanvar(UA_rates_zeroed_A, axis=0)

    n_ua_B = len(idxB_ua)
    UA_rates_zeroed_B  = ua_rate_hz_baselined[idxB_ua]
    UA_counts_B        = ua_counts[idxB_ua]
    UA_med_counts_B    = np.nanmedian(UA_counts_B, axis=0)
    UA_med_B           = np.nanmedian(UA_rates_zeroed_B, axis=0)
    UA_var_B           = np.nanvar(UA_rates_zeroed_B, axis=0)

    # NPRW
    n_nprw_A = len(idxA_nprw)
    NPRW_rates_zeroed_A = nprw_rates_hz_baselined[idxA_nprw]
    NPRW_counts_A       = nprw_counts[idxA_nprw]
    NPRW_med_counts_A   = np.nanmedian(NPRW_counts_A, axis=0)
    NPRW_med_A          = np.nanmedian(NPRW_rates_zeroed_A, axis=0)
    NPRW_var_A          = np.nanvar(NPRW_rates_zeroed_A, axis=0)

    n_nprw_B = len(idxB_nprw)
    NPRW_rates_zeroed_B = nprw_rates_hz_baselined[idxB_nprw]
    NPRW_counts_B       = nprw_counts[idxB_nprw]
    NPRW_med_counts_B   = np.nanmedian(NPRW_counts_B, axis=0)
    NPRW_med_B          = np.nanmedian(NPRW_rates_zeroed_B, axis=0)
    NPRW_var_B          = np.nanvar(NPRW_rates_zeroed_B, axis=0)

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

    # Per target
    targetA_dir = PERI_ROOT / "Target_A"
    targetB_dir = PERI_ROOT / "Target_B"
    targetA_dir.mkdir(parents=True, exist_ok=True)
    targetB_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Target A ----------
    peribase_A = f"peristim__{sess}__BR_{int(br_idx):03d}_target_A"
    out_npz_A  = targetA_dir / f"{peribase_A}.npz"
    out_mat_A  = targetA_dir / f"{peribase_A}.mat"
    
    meta_json     = json.dumps(meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    ua_meta_json  = json.dumps(ua_meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    nprw_meta_json= json.dumps(nprw_meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    
    np.savez_compressed(
        out_npz_A,
        sess=sess,
        br_idx=int(br_idx),
        overall_title=overall_title,

        meta=meta,
        ua_meta=ua_meta,
        nprw_meta=nprw_meta,
        meta_json=meta_json,
        ua_meta_json=ua_meta_json,
        nprw_meta_json=nprw_meta_json,

        align_meta_raw=aligned_npz["align_meta"].item() if "align_meta" in aligned_npz.files else "{}",    

        mask_ua=mask_ua,
        mask_nprw=mask_nprw,
        labels_ua=np.array(labels_ua, dtype="U1"),
        labels_nprw=np.array(labels_nprw, dtype="U1"),
        
        # behavior (A only, mapped to generic field names)
        n_beh=n_beh_A,
        beh_rel_t=beh_rel_t_A,
        beh_cam0_pos_med=beh_cam0_pos_med_A,
        beh_cam1_pos_med=beh_cam1_pos_med_A,
        beh_cam0_vel_med=beh_cam0_vel_med_A,
        beh_cam1_vel_med=beh_cam1_vel_med_A,
        beh_cam0_segs=beh_cam0_segs_A,
        beh_cam1_segs=beh_cam1_segs_A,
        beh_cam0_vel_segs=beh_cam0_vel_segs_A,
        beh_cam1_vel_segs=beh_cam1_vel_segs_A,
        beh_cam0_names=np.array(beh_cam0_names, dtype=object),
        beh_cam1_names=np.array(beh_cam1_names, dtype=object),
        
        hr_sig=hr_sig,
        vog_sig=vog_sig,

        # VOG and ts_state (shared)
        # vog_rel_t=vog_rel_t,
        # vog_med=vog_med,
        # vog_segs=vog_segs,
        # vog_cols=vog_cols,
        # vog_col_names=np.array(vog_col_names, dtype=object),
        # n_vog_trials=int(n_vog_trials),

        ts_state_rel_t=ts_state_rel_t,
        ts_state_segs=ts_state_segs_A,
        ts_state_char_segs=ts_state_char_segs_A,
        ts_state_char=ts_state_char,
        ts_state_num=ts_state_num_full if ts_state_num is not None else np.array([], int),
        n_ts_state_trials=int(n_ts_state_trls_A),
        
        stim_ms_ua=stim_ms_A_ua,
        stim_ms_nprw=stim_ms_A_nprw,

        stim_ms=stim_ms,
        trial_labels_all=np.array(trial_labels, dtype="U1"),

        NPRW_med=NPRW_med_A,
        NPRW_var=NPRW_var_A,
        NPRW_med_counts=NPRW_med_counts_A,
        
        NPRW_rates_zeroed=NPRW_rates_zeroed_A,
        NPRW_counts=NPRW_counts_A,
        NPRW_edges_ms=nprw_edges_ms,
        NPRW_rel_t=nprw_rel_t,
        NPRW_peak_ms_dedup=nprw_peak_ms_dedup,
        NPRW_amps_ms_dedup=nprw_amps_ms_dedup,
        n_nprw=int(n_nprw_A),

        UA_med=UA_med_A,
        UA_med_counts=UA_med_counts_A,
        UA_var=UA_var_A,
        UA_rates_zeroed=UA_rates_zeroed_A,
        UA_counts=UA_counts_A,
        UA_edges_ms=ua_edges_ms,
        UA_peak_ms_dedup=ua_peak_ms_dedup,
        UA_amps_ms_dedup=ua_amps_ms_dedup,
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
        "sess":     np.array(sess, dtype=object),
        "br_idx":        int(br_idx),
        "overall_title": np.array(overall_title, dtype=object),
        "stim_ms":       stim_ms,
        "ua_meta":ua_meta,
        "nprw_meta": nprw_meta,
        "align_meta_raw": aligned_npz["align_meta"].item() if "align_meta" in aligned_npz.files else "{}",
        
        "n_beh":       n_beh_A,
        "beh_rel_t":     beh_rel_t_A,
        "beh_cam0_pos_med": beh_cam0_pos_med_A,
        "beh_cam1_pos_med": beh_cam1_pos_med_A,
        "beh_cam0_vel_med": beh_cam0_vel_med_A,
        "beh_cam1_vel_med": beh_cam1_vel_med_A,
        "beh_cam0_segs": beh_cam0_segs_A,
        "beh_cam1_segs": beh_cam1_segs_A,
        "beh_cam0_vel_segs":beh_cam0_vel_segs_A,
        "beh_cam1_vel_segs":beh_cam1_vel_segs_A,
        "beh_cam0_names": np.array(beh_cam0_names, dtype=object),
        "beh_cam1_names": np.array(beh_cam1_names, dtype=object),

        "hr_sig": hr_sig,
        "vog_sig": vog_sig,
        
        # "vog_rel_t":     vog_rel_t,
        # "vog_med":       vog_med,
        # "vog_segs":      vog_segs,
        # "vog_cols":      vog_cols,
        # "vog_col_names": np.array(vog_col_names, dtype=object),
        # "n_vog_trials":  int(n_vog_trials),

        "ts_state_rel_t":      ts_state_rel_t,
        "ts_state_segs":       ts_state_segs_A,
        "ts_state_char_segs":  ts_state_char_segs_A,
        "ts_state_char":       ts_state_char,
        "ts_state_num":        ts_state_num_full if ts_state_num is not None else np.array([], int),
        "n_ts_state_trials":   int(n_ts_state_trls_A),
        "trial_labels":        trial_labels,

        "NPRW_med":      NPRW_med_A,
        "NPRW_med_counts":      NPRW_med_counts_A,
        "NPRW_var":      NPRW_var_A,
        "NPRW_rates_zeroed":   NPRW_rates_zeroed_A,
        "NPRW_counts":   NPRW_counts_A,
        "NPRW_edges_ms": nprw_edges_ms,
        "NPRW_rel_t":    nprw_rel_t,
        "n_nprw":        int(n_nprw_A),

        "UA_med":        UA_med_A,
        "UA_med_counts": UA_med_counts_A,
        "UA_var":        UA_var_A,
        "UA_rates_zeroed":     UA_rates_zeroed_A,
        "UA_counts":     UA_counts_A,
        "UA_edges_ms":   ua_edges_ms,
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

        meta=meta,
        ua_meta=ua_meta,
        nprw_meta=nprw_meta,
        meta_json=meta_json,
        ua_meta_json=ua_meta_json,
        nprw_meta_json=nprw_meta_json,

        align_meta_raw=aligned_npz["align_meta"].item() if "align_meta" in aligned_npz.files else "{}",    

        mask_ua=mask_ua,
        mask_nprw=mask_nprw,
        labels_ua=np.array(labels_ua, dtype="U1"),
        labels_nprw=np.array(labels_nprw, dtype="U1"),
        
        n_beh=n_beh_B,
        beh_rel_t=beh_rel_t_B,
        beh_cam0_pos_med=beh_cam0_pos_med_B,
        beh_cam1_pos_med=beh_cam1_pos_med_B,
        beh_cam0_vel_med=beh_cam0_vel_med_B,
        beh_cam1_vel_med=beh_cam1_vel_med_B,
        beh_cam0_segs=beh_cam0_segs_B,
        beh_cam1_segs=beh_cam1_segs_B,
        beh_cam0_vel_segs=beh_cam0_vel_segs_B,
        beh_cam1_vel_segs=beh_cam1_vel_segs_B,
        beh_cam0_names=np.array(beh_cam0_names, dtype=object),
        beh_cam1_names=np.array(beh_cam1_names, dtype=object),

        hr_sig=hr_sig,
        vog_sig=vog_sig,
        
        # vog_rel_t=vog_rel_t,
        # vog_med=vog_med,
        # vog_segs=vog_segs,
        # vog_cols=vog_cols,
        # vog_col_names=np.array(vog_col_names, dtype=object),
        # n_vog_trials=int(n_vog_trials),

        ts_state_rel_t=ts_state_rel_t,
        ts_state_segs=ts_state_segs_B,
        ts_state_char_segs=ts_state_char_segs_B,
        ts_state_char=ts_state_char,
        ts_state_num=ts_state_num_full if ts_state_num is not None else np.array([], int),
        n_ts_state_trials=int(n_ts_state_trls_B),

        stim_ms_ua=stim_ms_B_ua,
        stim_ms_nprw=stim_ms_B_nprw,

        stim_ms=stim_ms,
        trial_labels_all=np.array(trial_labels, dtype="U1"),

        NPRW_med=NPRW_med_B,
        NPRW_var=NPRW_var_B,
        NPRW_med_counts=NPRW_med_counts_B,
        
        NPRW_rates_zeroed=NPRW_rates_zeroed_B,
        NPRW_counts=NPRW_counts_B,
        NPRW_edges_ms=nprw_edges_ms,
        NPRW_rel_t=nprw_rel_t,
        NPRW_peak_ms_dedup=nprw_peak_ms_dedup,
        NPRW_amps_ms_dedup=nprw_amps_ms_dedup,
        n_nprw=int(n_nprw_B),

        UA_med=UA_med_B,
        UA_med_counts=UA_med_counts_B,
        UA_var=UA_var_B,
        UA_rates_zeroed=UA_rates_zeroed_B,
        UA_counts=UA_counts_B,
        UA_edges_ms=ua_edges_ms,
        UA_peak_ms_dedup=ua_peak_ms_dedup,
        UA_amps_ms_dedup=ua_amps_ms_dedup,
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
        "ua_meta":ua_meta,
        "nprw_meta": nprw_meta,
        "align_meta_raw": aligned_npz["align_meta"].item() if "align_meta" in aligned_npz.files else "{}",

        "n_beh":         n_beh_B,
        "beh_rel_t":     beh_rel_t_B,
        "beh_cam0_pos_med": beh_cam0_pos_med_B,
        "beh_cam1_pos_med": beh_cam1_pos_med_B,
        "beh_cam0_vel_med": beh_cam0_vel_med_B,
        "beh_cam1_vel_med": beh_cam1_vel_med_B,
        "beh_cam0_segs": beh_cam0_segs_B,
        "beh_cam1_segs": beh_cam1_segs_B,
        "beh_cam0_vel_segs":beh_cam0_vel_segs_B,
        "beh_cam1_vel_segs":beh_cam1_vel_segs_B,
        "beh_cam0_names": np.array(beh_cam0_names, dtype=object),
        "beh_cam1_names": np.array(beh_cam1_names, dtype=object),

        "hr_sig": hr_sig,
        "vog_sig": vog_sig,
        
        # "vog_rel_t":     vog_rel_t,
        # "vog_med":       vog_med,
        # "vog_segs":      vog_segs,
        # "vog_cols":      vog_cols,
        # "vog_col_names": np.array(vog_col_names, dtype=object),
        # "n_vog_trials":  int(n_vog_trials),

        "ts_state_rel_t":      ts_state_rel_t,
        "ts_state_segs":       ts_state_segs_B,
        "ts_state_char_segs":  ts_state_char_segs_B,
        "ts_state_char":       ts_state_char,
        "ts_state_num":        ts_state_num_full if ts_state_num is not None else np.array([], int),
        "n_ts_state_trials":   int(n_ts_state_trls_B),
        "trial_labels":        trial_labels,

        "NPRW_med":      NPRW_med_B,
        "NPRW_med_counts":      NPRW_med_counts_B,
        "NPRW_var":      NPRW_var_B,
        "NPRW_rates_zeroed":   NPRW_rates_zeroed_B,
        "NPRW_counts":   NPRW_counts_B,
        "NPRW_edges_ms": nprw_edges_ms,
        "NPRW_rel_t":    nprw_rel_t,
        "n_nprw":        int(n_nprw_B),

        "UA_med":        UA_med_B,
        "UA_med_counts": UA_med_counts_B,
        "UA_var":        UA_var_B,
        "UA_rates_zeroed":     UA_rates_zeroed_B,
        "UA_counts":     UA_counts_B,
        "UA_edges_ms":   ua_edges_ms,
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
