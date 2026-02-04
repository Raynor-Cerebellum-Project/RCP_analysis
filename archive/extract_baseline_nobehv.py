from typing import Any
import csv
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre




import re, json
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *








IR_STREAM    = "USB board digital input channel"
SHIFTS_CSV   = METADATA_ROOT / "br_to_intan_shifts.csv"
def _load_br_to_intan_shifts(shifts_csv: Path) -> dict[int, dict]:
    """
    Return {BR_idx: {"session": <str>, "shifts_ms": <float or 0.0>}}.
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
        c_shift = col("shift_ms")

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
            out[br] = {"session": sess, "shift_ms": shift_ms}
    return out

BR_TO_INTAN_SHIFTS = _load_br_to_intan_shifts(SHIFTS_CSV)

# IR event loader
def _detect_IR_crossings(x: np.ndarray, fs: float | None, refractory_sec: float = 0.0005) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    if x.size == 0:
        return np.array([], np.int64)

    # fill NaNs without changing indexing
    if not np.isfinite(x).all():
        med = np.nanmedian(x)
        x = np.nan_to_num(x, nan=med)

    lo, hi = float(np.min(x)), float(np.max(x))
    thr = 0.5 * (lo + hi)
    b = (x > thr).astype(np.int8)

    db = np.diff(b, prepend=b[0])
    edges = np.flatnonzero(db == -1)

    if edges.size == 0:
        return edges.astype(np.int64)

    refr = max(1, int(round(refractory_sec * fs))) if (fs and fs > 0) else 1
    keep = [edges[0]]
    for e in edges[1:]:
        if e - keep[-1] >= refr:
            keep.append(e)
    return np.asarray(keep, np.int64)

def _load_ir_ms_from_aligned(aligned_path: Path, meta: dict[str, Any]) -> np.ndarray:
    """
    Compute IR event times (in BR-aligned ms)
      - read the Intan IR stream for the matching Intan session
      - detect IR 1→0 edges
      - convert sample indices to seconds, then to BR-aligned ms
        using shift_ms from br_to_intan_shifts.csv.

    Returns:
        ir_ms_br : np.ndarray of shape (N,) in BR time
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
    shift_ms = float(shift_entry.get("shift_ms", 0.0))
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

    # Intan seconds -> BR-aligned ms (same as original script: subtract shift_ms)
    ir_sec = ir_idx / fs
    ir_ms_br = ir_sec * 1000.0 - shift_ms

    print(
        f"[baseline-IR] {aligned_path.name}: BR={br_idx}, Intan={intan_session}, "
        f"shift_ms={shift_ms:g}, n_IR={ir_ms_br.size}"
    )
    return ir_ms_br.astype(float)



# Config
# Base paths from config_loading
ALIGNED_ROOT = ALIGNED_CKPT_ROOT


NPRW_RATES = PARAMS.NPRW_rate_est
NPRW_BIN_MS     = NPRW_RATES.get("bin_ms")
NPRW_SIGMA_MS   = NPRW_RATES.get("sigma_ms")
NPRW_MS_BEFORE = float(NPRW_RATES.get("remove_ms_before", 20.0)) # Need to remove these for baseline
NPRW_TAIL_MS   = float(NPRW_RATES.get("remove_tail_ms_after", 20.0))
NPRW_DEDUP_MS = float(NPRW_RATES.get("dedup_ms", 0.5))

UA_RATES = PARAMS.UA_rate_est
UA_BIN_MS     = UA_RATES.get("bin_ms")
UA_SIGMA_MS   = UA_RATES.get("sigma_ms")
UA_MS_BEFORE = float(UA_RATES.get("remove_ms_before", 5.0))
UA_TAIL_MS   = float(UA_RATES.get("remove_tail_ms_after", 5.0))
UA_DEDUP_MS = float(UA_RATES.get("dedup_ms", 0.5))
MAX_CLUSTER_MS = 0.5

STIM_DUR = 20

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
    Load aligned__*.npz (BR↔Intan only), compute peri-stim med/var for NPRW+UA,
    and save to PERI_ROOT.
    """
    aligned_npz = np.load(aligned_path, allow_pickle=True)

    # ---- required fields from the new aligned NPZ ----
    nprw_meta      = aligned_npz["nprw_meta"].item()
    nprw_peak_ms   = aligned_npz["nprw_peak_ms"].reshape(-1)[0]
    nprw_peak_amps = aligned_npz["nprw_peak_amps"].reshape(-1)[0]
    nprw_rec_ms    = (float(nprw_meta["rec_start_ms_aligned"]), float(nprw_meta["rec_end_ms_aligned"]))

    ua_meta      = aligned_npz["ua_meta"].item()
    ua_peak_ms   = aligned_npz["ua_peak_ms"].reshape(-1)[0]
    ua_peak_amps = aligned_npz["ua_peak_amps"].reshape(-1)[0]
    ua_rec_ms    = (float(ua_meta["rec_start_ms"]), float(ua_meta["rec_end_ms"]))



    # alignment meta (JSON)
    meta = json.loads(aligned_npz["align_meta"].item()) if "align_meta" in aligned_npz.files else {}
    sess   = meta.get("session", aligned_path.stem)
    br_idx = int(meta.get("br_idx", -1))

    # ---- stim / IR times (prefer stim_ms, fall back to ir_ms) ----
    stim_ms = None
    
    

    if "stim_ms" in aligned_npz.files:
        stim_ms = np.asarray(aligned_npz["stim_ms"], float).ravel()
        if stim_ms.size == 0:
            stim_ms = None

    if stim_ms is None:
        stim_ms = ir_ms = _load_ir_ms_from_aligned(aligned_path, meta)
        
        if stim_ms.size:
            print(f"[extract] {aligned_path.name}: using ir_ms fallback ({stim_ms.size} events)")
        else:
            stim_ms = None
    
    # UA ids / labels (still present in new aligned NPZ)
    ua_ids_1based = np.asarray(aligned_npz["ua_elec"], dtype=int).ravel() if "ua_elec" in aligned_npz.files else np.array([], int)
    ua_region = aligned_npz["ua_region"] if "ua_region" in aligned_npz.files else np.array([], int)
    ua_region_names = aligned_npz["ua_region_names"] if "ua_region_names" in aligned_npz.files else np.array([], dtype=object)
    ua_port = aligned_npz["ua_port"] if "ua_port" in aligned_npz.files else np.array([], dtype=object)
    ua_nsp = aligned_npz["ua_nsp"] if "ua_nsp" in aligned_npz.files else np.array([], int)
    ua_idx_rows = aligned_npz["ua_idx_rows"] if "ua_idx_rows" in aligned_npz.files else np.array([], int)

    print(
        f"[debug] {aligned_path.name}: "
        f"stim_n={stim_ms.size}, "
        f"NPRW_rec={nprw_rec_ms[0]:.1f}-{nprw_rec_ms[1]:.1f} ms, "
        f"UA_rec={ua_rec_ms[0]:.1f}-{ua_rec_ms[1]:.1f} ms"
    )

    # ---- valid stims for each stream ----
    mask_nprw = _get_valid_events(stim_ms, nprw_rec_ms, WIN_MS)
    mask_ua   = _get_valid_events(stim_ms, ua_rec_ms,   WIN_MS)

    stim_ms_nprw = stim_ms[mask_nprw]
    stim_ms_ua   = stim_ms[mask_ua]

    if stim_ms_nprw.size == 0 and stim_ms_ua.size == 0:
        print(f"[extract] {aligned_path.name}: no valid stims in either stream; skipping.")
        return

    # ---- dedup peaks ----
    ua_peak_ms_dedup, ua_amps_ms_dedup = _dedup_peaks(
        ua_peak_ms, ua_peak_amps, UA_DEDUP_MS, MAX_CLUSTER_MS
    )
    nprw_peak_ms_dedup, nprw_amps_ms_dedup = _dedup_peaks(
        nprw_peak_ms, nprw_peak_amps, NPRW_DEDUP_MS, MAX_CLUSTER_MS
    )

    # ---- bin around stim (only valid stims per stream) ----
    ua_counts, ua_rel_t, ua_edges_ms, ua_left_bins = _bin_counts_around_stim(
        ua_peak_ms_dedup, UA_BIN_MS, stim_ms_ua,
        UA_MS_BEFORE, (STIM_DUR + UA_TAIL_MS), WIN_MS
    )
    nprw_counts, nprw_rel_t, nprw_edges_ms, nprw_left_bins = _bin_counts_around_stim(
        nprw_peak_ms_dedup, NPRW_BIN_MS, stim_ms_nprw,
        NPRW_MS_BEFORE, (STIM_DUR + NPRW_TAIL_MS), WIN_MS
    )

    # ---- smooth to rates ----
    ua_rate_hz   = _smooth_counts_gauss(ua_counts,   ua_edges_ms,   ua_rel_t,   UA_SIGMA_MS,   ua_left_bins)
    nprw_rate_hz = _smooth_counts_gauss(nprw_counts, nprw_edges_ms, nprw_rel_t, NPRW_SIGMA_MS, nprw_left_bins)

    # ---- baseline zero each trial ----
    ua_rate_hz_baselined   = rcp.baseline_zero_each_trial(ua_rate_hz,   ua_rel_t,   normalize_first_ms=NORMALIZE_FIRST_MS)
    nprw_rate_hz_baselined = rcp.baseline_zero_each_trial(nprw_rate_hz, nprw_rel_t, normalize_first_ms=NORMALIZE_FIRST_MS)

    # ---- compute median/var across trials ----
    # shapes:
    #   ua_rate_hz_baselined:   (n_trials_ua,   n_channels_ua,   n_bins_ua)
    #   nprw_rate_hz_baselined: (n_trials_nprw, n_channels_nprw, n_bins_nprw)
    UA_med        = np.nanmedian(ua_rate_hz_baselined, axis=0) if ua_rate_hz_baselined is not None and np.size(ua_rate_hz_baselined) else np.zeros((0, 0), float)
    UA_var        = np.nanvar(ua_rate_hz_baselined, axis=0)    if ua_rate_hz_baselined is not None and np.size(ua_rate_hz_baselined) else np.zeros((0, 0), float)
    UA_med_counts = np.nanmedian(ua_counts, axis=0)            if ua_counts is not None and np.size(ua_counts) else np.zeros((0, 0), float)

    NPRW_med        = np.nanmedian(nprw_rate_hz_baselined, axis=0) if nprw_rate_hz_baselined is not None and np.size(nprw_rate_hz_baselined) else np.zeros((0, 0), float)
    NPRW_var        = np.nanvar(nprw_rate_hz_baselined, axis=0)    if nprw_rate_hz_baselined is not None and np.size(nprw_rate_hz_baselined) else np.zeros((0, 0), float)
    NPRW_med_counts = np.nanmedian(nprw_counts, axis=0)            if nprw_counts is not None and np.size(nprw_counts) else np.zeros((0, 0), float)

    # ---- metadata title (optional) ----
    try:
        overall_title, _ = build_title_from_csv(METADATA_CSV, br_file=br_idx)
    except Exception as e:
        print(f"[warn] title parse failed for BR {br_idx}: {e}")
        overall_title = "Condition: n/a"

    # ---- output paths ----
    PERI_ROOT.mkdir(parents=True, exist_ok=True)
    peribase = f"peristim__{sess}__BR_{int(br_idx):03d}"
    out_npz  = PERI_ROOT / f"{peribase}.npz"
    out_mat  = PERI_ROOT / f"{peribase}.mat"

    meta_json      = json.dumps(meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    ua_meta_json   = json.dumps(ua_meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    nprw_meta_json = json.dumps(nprw_meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    np.savez_compressed(
        out_npz,
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

        stim_ms=stim_ms,
        stim_ms_ua=stim_ms_ua,
        stim_ms_nprw=stim_ms_nprw,
        mask_ua=mask_ua,
        mask_nprw=mask_nprw,

        # NPRW
        NPRW_med=NPRW_med,
        NPRW_var=NPRW_var,
        NPRW_med_counts=NPRW_med_counts,
        NPRW_rates_zeroed=nprw_rate_hz_baselined,
        NPRW_counts=nprw_counts,
        NPRW_edges_ms=nprw_edges_ms,
        NPRW_rel_t=nprw_rel_t,
        NPRW_peak_ms_dedup=nprw_peak_ms_dedup,
        NPRW_amps_ms_dedup=nprw_amps_ms_dedup,
        n_nprw=int(stim_ms_nprw.size),

        # UA
        UA_med=UA_med,
        UA_var=UA_var,
        UA_med_counts=UA_med_counts,
        UA_rates_zeroed=ua_rate_hz_baselined,
        UA_counts=ua_counts,
        UA_edges_ms=ua_edges_ms,
        UA_rel_t=ua_rel_t,
        UA_peak_ms_dedup=ua_peak_ms_dedup,
        UA_amps_ms_dedup=ua_amps_ms_dedup,
        n_ua=int(stim_ms_ua.size),

        ua_ids_1based=ua_ids_1based,
        ua_region=ua_region,
        ua_region_names=ua_region_names,
        ua_port=ua_port,
        ua_nsp=ua_nsp,
        ua_idx_rows=ua_idx_rows,
    )


def main():
    files = sorted(ALIGNED_ROOT.glob("aligned__*.npz"))
    if not files:
        raise SystemExit(f"[error] No combined aligned NPZs found at {ALIGNED_ROOT}")
    PERI_ROOT.mkdir(parents=True, exist_ok=True)
    for f in files:
        extract_one_file(f)

if __name__ == "__main__":
    main()
