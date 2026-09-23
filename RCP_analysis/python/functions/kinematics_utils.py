from __future__ import annotations

import warnings
from typing import Optional, Tuple, Any, List

import numpy as np
from scipy import stats
from scipy.signal import butter, filtfilt

def strip_cam_prefix(name: str) -> str:
    """Turn 'cam0_wrist_x' or 'cam1_wrist_x' into 'wrist_x'."""
    name = str(name)

    for prefix in ("cam0_", "cam1_"):
        if name.startswith(prefix):
            return name[len(prefix):]

    return name

def simple_beh_labels(
    names: list[str],
    keypoints: tuple[str, ...],
) -> list[str]:
    """
    Map raw DLC-style names to readable labels.

    Examples
    --------
    'cam0_middle_x' -> 'Middle X'
    'DLC_..._wrist_y' -> 'Wrist Y'
    """
    out: list[str] = []
    kps_lc = tuple(kp.lower() for kp in keypoints)

    for name in names:
        s = str(name).lower()

        kp = next((kp for kp in kps_lc if kp in s), None)

        axis = (
            "x" if ("_x" in s or s.endswith("x")) else
            "y" if ("_y" in s or s.endswith("y")) else
            None
        )

        if kp and axis:
            base = kp.replace("_", " ").title()
            out.append(f"{base} {axis.upper()}")
        else:
            out.append(str(name))

    return out

def get_subset_indices(
    labels: list[str],
    keys: tuple[str, ...] | list[str] | str | None,
) -> list[int]:
    """
    Keep behavior traces whose label contains one of the requested key substrings.

    keys can be:
        None
        "ALL"
        "middle"
        ("middle", "wrist")
    """
    if keys is None:
        return []

    if isinstance(keys, str):
        if keys.upper() == "ALL":
            return list(range(len(labels)))
        keys = (keys,)

    keys = tuple(str(k).lower() for k in keys)

    return [
        i for i, lab in enumerate(labels)
        if any(k in str(lab).lower() for k in keys)
    ]

def find_xy_indices_for_keypoint(
    labels: list[str],
    keypoint: str,
) -> tuple[int | None, int | None]:
    """
    Find x/y trace indices for a keypoint from labels like:
        'Middle X', 'Middle Y'
        'middle_x', 'middle_y'
        'middle x', 'middle y'
    """
    kp = str(keypoint).lower()
    x_idx = None
    y_idx = None

    for i, lab in enumerate(labels):
        s = str(lab).lower().replace("_", " ")
        tokens = s.split()

        if kp not in s:
            continue

        if ("x" in tokens or s.endswith(" x") or s.endswith("x")) and x_idx is None:
            x_idx = i

        if ("y" in tokens or s.endswith(" y") or s.endswith("y")) and y_idx is None:
            y_idx = i

    return x_idx, y_idx

def value_at_ref_time(
    trace: np.ndarray,
    t_ms: np.ndarray,
    ref_time_ms: float,
) -> float:
    """
    Return trace value at ref_time_ms.

    Uses interpolation if possible, otherwise nearest valid sample.
    """
    trace = np.asarray(trace, dtype=float)
    t_ms = np.asarray(t_ms, dtype=float)

    valid = np.isfinite(trace) & np.isfinite(t_ms)

    if not np.any(valid):
        return np.nan

    tv = t_ms[valid]
    xv = trace[valid]

    if xv.size >= 2 and np.nanmin(tv) <= ref_time_ms <= np.nanmax(tv):
        order = np.argsort(tv)
        return float(np.interp(ref_time_ms, tv[order], xv[order]))

    j = int(np.nanargmin(np.abs(tv - ref_time_ms)))
    return float(xv[j])

def normalize_distance_trace(
    dist: np.ndarray,
    mode: str = "max_abs",
) -> np.ndarray:
    """
    Normalize one distance trace.

    Parameters
    ----------
    dist:
        Shape (T,)

    mode:
        'max_abs'   : divide by max(abs(distance))
        'final_abs' : divide by abs(final valid distance)
        'max'       : divide by max(distance), useful for nonnegative distance
        'none'      : no normalization
    """
    dist = np.asarray(dist, dtype=float)
    out = dist.copy()

    mode = str(mode).lower()

    if mode == "none":
        return out

    if mode == "max_abs":
        denom = np.nanmax(np.abs(out))
    elif mode == "max":
        denom = np.nanmax(out)
    elif mode == "final_abs":
        valid = np.where(np.isfinite(out))[0]
        denom = np.abs(out[valid[-1]]) if valid.size else np.nan
    else:
        raise ValueError(
            f"Unknown normalization mode={mode!r}. "
            "Use 'max_abs', 'max', 'final_abs', or 'none'."
        )

    if not np.isfinite(denom) or denom == 0:
        return np.full_like(out, np.nan, dtype=float)

    return out / denom

def compute_normalized_distance_from_xy_3d(
    arr_nkt: np.ndarray | None,
    labels: list[str],
    t_ms: np.ndarray,
    keypoint: str = "middle",
    ref_time_ms: float = -600.0,
    mode: str = "max_abs",
) -> np.ndarray | None:
    """
    Compute normalized distance per trial from a 3-D behavior array.

    Parameters
    ----------
    arr_nkt:
        Shape (n_trials, K, T)

    labels:
        Behavior labels corresponding to K.

    t_ms:
        Time vector, shape (T,)

    keypoint:
        Example: 'middle'

    ref_time_ms:
        Time used as zero-distance reference.

    mode:
        Distance normalization mode.

    Returns
    -------
    dist_nt:
        Shape (n_trials, T)
    """
    if arr_nkt is None:
        return None

    arr_nkt = np.asarray(arr_nkt, dtype=float)

    if arr_nkt.ndim != 3:
        raise ValueError(f"Expected 3-D array (n_trials,K,T), got shape {arr_nkt.shape}")

    x_idx, y_idx = find_xy_indices_for_keypoint(labels, keypoint)

    if x_idx is None or y_idx is None:
        print(
            f"[WARN] Could not find x/y indices for keypoint={keypoint!r} "
            f"in labels={labels}. Normalized distance will not be plotted."
        )
        return None

    n_trials = arr_nkt.shape[0]
    dist_nt = np.full((n_trials, arr_nkt.shape[2]), np.nan, dtype=float)

    for tr in range(n_trials):
        x = arr_nkt[tr, x_idx, :]
        y = arr_nkt[tr, y_idx, :]

        x0 = value_at_ref_time(x, t_ms, ref_time_ms)
        y0 = value_at_ref_time(y, t_ms, ref_time_ms)

        dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        dist_nt[tr, :] = normalize_distance_trace(dist, mode=mode)

    return dist_nt


def extract_normalized_distance(
    pos_segs: Optional[np.ndarray],
    names: Any,
    beh_rel_t: Optional[np.ndarray],
    preferred_keypoints: Tuple[str, ...] = ("middle", "wrist", "hand", "index"),
    ref_time_ms: float = -600.0,
    mode: str = "max_abs",
) -> Optional[np.ndarray]:
    """
    Computes normalized distance traces (n_trials, T) from 3D behavior segments.
    Tries keypoints in preferred_keypoints in order until a valid distance array is computed.
    """
    if pos_segs is None or beh_rel_t is None or names is None:
        return None
    pos_segs = np.asarray(pos_segs, dtype=float)
    if pos_segs.ndim != 3 or pos_segs.shape[0] == 0:
        return None

    names_list = [str(x) for x in np.asarray(names).ravel()]
    if not names_list:
        return None
    raw_labels = [strip_cam_prefix(n) for n in names_list]
    display_labels = simple_beh_labels(raw_labels, preferred_keypoints)

    for kp in preferred_keypoints:
        try:
            dist = compute_normalized_distance_from_xy_3d(
                pos_segs,
                display_labels,
                beh_rel_t,
                keypoint=kp,
                ref_time_ms=ref_time_ms,
                mode=mode,
            )
            if dist is not None and np.any(np.isfinite(dist)):
                return dist
        except Exception:
            continue
    return None


def butterworth_filter(
    trace: np.ndarray,
    dt: float,
    cutoff_hz: float = 8.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth low-pass filter to a 1D kinematic trace.
    Handles NaN values via linear interpolation and restores them after filtering.

    Parameters
    ----------
    trace : np.ndarray
        1D signal array.
    dt : float
        Sampling interval in milliseconds (if dt < 0.9, automatically converted from seconds to ms).
    cutoff_hz : float, default 8.0
        Low-pass cutoff frequency in Hz.
    order : int, default 4
        Butterworth filter order.

    Returns
    -------
    np.ndarray
        Filtered 1D signal with identical shape.
    """
    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 1 or not np.any(np.isfinite(trace)):
        return trace

    if dt < 0.9:
        dt *= 1000.0  # convert seconds to ms

    fs = 1000.0 / dt
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        cutoff_hz = nyq * 0.9

    try:
        b, a = butter(order, cutoff_hz / nyq, btype="low")
    except Exception:
        return trace

    valid_mask = np.isfinite(trace)
    smoothed = trace.copy()

    if np.sum(valid_mask) < 3 * order:
        return smoothed

    try:
        if np.all(valid_mask):
            smoothed = filtfilt(b, a, trace)
        else:
            trace_interp = trace.copy()
            nans = ~valid_mask
            if np.any(nans) and np.any(valid_mask):
                trace_interp[nans] = np.interp(
                    np.flatnonzero(nans),
                    np.flatnonzero(valid_mask),
                    trace[valid_mask],
                )
            smoothed = filtfilt(b, a, trace_interp)
            smoothed[nans] = np.nan
    except Exception:
        smoothed = trace.copy()

    return smoothed


def extract_xy_coordinates(
    pos_segs: Optional[np.ndarray],
    names: Any,
    preferred_keypoints: Tuple[str, ...] = ("middle", "wrist", "hand", "index"),
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract X and Y coordinate arrays from a 3D behavior array (n_trials, n_keypoints, n_timepoints).
    Tries keypoints in preferred_keypoints in order until valid X and Y traces are found.

    Parameters
    ----------
    pos_segs : np.ndarray or None
        3D behavior position array of shape (n_trials, n_keypoints, n_timepoints).
    names : list or array of str
        Keypoint names corresponding to axis 1 of pos_segs.
    preferred_keypoints : tuple of str
        Preferred keypoint substrings to search for (e.g. ('middle', 'wrist')).

    Returns
    -------
    tuple of (x, y) ndarrays of shape (n_trials, n_timepoints), or None if not found.
    """
    if pos_segs is None or names is None:
        return None
    pos_segs = np.asarray(pos_segs, dtype=float)
    if pos_segs.ndim != 3 or pos_segs.shape[0] == 0:
        return None

    names_list = [str(x) for x in np.asarray(names).ravel()]
    if not names_list:
        return None

    raw_labels = [strip_cam_prefix(n) for n in names_list]
    display_labels = simple_beh_labels(raw_labels, preferred_keypoints)

    for kp in preferred_keypoints:
        x_idx, y_idx = find_xy_indices_for_keypoint(display_labels, kp)
        if x_idx is not None and y_idx is not None:
            x = pos_segs[:, x_idx, :]
            y = pos_segs[:, y_idx, :]
            if np.any(np.isfinite(x)) and np.any(np.isfinite(y)):
                return x, y
    return None


def compute_time_resolved_pairwise_variability(
    x: np.ndarray,
    y: np.ndarray,
    t_axis: Optional[np.ndarray] = None,
    smooth_butterworth: bool = True,
    cutoff_hz: float = 8.0,
    order: int = 4,
    agg: str = "median",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Computes time-resolved pairwise X-Y trajectory variability across trials at each sample,
    following plot_plateau_analysis.py.

    Parameters
    ----------
    x, y : np.ndarray
        Position arrays of shape (n_trials, n_samples).
    t_axis : np.ndarray, optional
        Time vector of length n_samples (used for Butterworth dt calculation).
    smooth_butterworth : bool, default True
        Whether to apply Butterworth low-pass filtering to X and Y per trial.
    cutoff_hz : float, default 8.0
        Cutoff frequency for Butterworth low-pass filter.
    order : int, default 4
        Order of Butterworth low-pass filter.
    agg : str, default 'median'
        Aggregation method across trial pairs ('median' or 'mean').

    Returns
    -------
    curve : np.ndarray of shape (n_samples,)
        Aggregate pairwise distance at each timepoint.
    lower : np.ndarray of shape (n_samples,)
        Lower error bound (25th percentile for median, curve - sem for mean).
    upper : np.ndarray of shape (n_samples,)
        Upper error bound (75th percentile for median, curve + sem for mean).
    """
    if x is None or y is None:
        return None, None, None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        return None, None, None

    n_trials, n_samples = x.shape
    if n_trials < 2:
        return None, None, None

    x_proc = x.copy()
    y_proc = y.copy()

    if smooth_butterworth and t_axis is not None and len(t_axis) > 1:
        dt = float(np.mean(np.diff(t_axis)))
        if dt < 0.9:
            dt *= 1000.0
        for tr in range(n_trials):
            x_proc[tr] = butterworth_filter(x_proc[tr], dt, cutoff_hz=cutoff_hz, order=order)
            y_proc[tr] = butterworth_filter(y_proc[tr], dt, cutoff_hz=cutoff_hz, order=order)

    pairwise = []
    for i in range(n_trials - 1):
        dx = x_proc[i + 1:] - x_proc[i]
        dy = y_proc[i + 1:] - y_proc[i]
        d = np.sqrt(dx ** 2 + dy ** 2)
        pairwise.append(d)

    if not pairwise:
        return None, None, None

    pairwise_dists = np.vstack(pairwise)
    if pairwise_dists.size == 0:
        return None, None, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if agg == "mean":
            curve = np.nanmean(pairwise_dists, axis=0)
            n_valid = np.sum(np.isfinite(pairwise_dists), axis=0)
            s = np.nanstd(pairwise_dists, axis=0)
            sem = np.where(n_valid > 0, s / np.sqrt(np.maximum(1, n_valid)), np.nan)
            lower = curve - sem
            upper = curve + sem
        else:  # 'median'
            curve = np.nanmedian(pairwise_dists, axis=0)
            lower = np.nanpercentile(pairwise_dists, 25, axis=0)
            upper = np.nanpercentile(pairwise_dists, 75, axis=0)

    return curve, lower, upper


def median_from_trial_traces(
    arr_nt: np.ndarray | None,
) -> np.ndarray | None:
    """
    Convert per-trial traces, shape (n_trials,T), to median plotting array,
    shape (1,T).
    """
    if arr_nt is None:
        return None

    arr_nt = np.asarray(arr_nt, dtype=float)

    if arr_nt.ndim != 2:
        raise ValueError(f"Expected 2-D array (n_trials,T), got shape {arr_nt.shape}")

    return np.nanmedian(arr_nt, axis=0, keepdims=True)

def mean_sd_from_trial_traces(
    arr_nt: np.ndarray | None,
    ddof: int = 0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Convert per-trial traces, shape (n_trials,T), to mean/std arrays,
    shape (1,T).
    """
    if arr_nt is None:
        return None, None

    arr_nt = np.asarray(arr_nt, dtype=float)

    if arr_nt.ndim != 2:
        raise ValueError(f"Expected 2-D array (n_trials,T), got shape {arr_nt.shape}")

    mean_1t = np.nanmean(arr_nt, axis=0, keepdims=True)
    sd_1t = np.nanstd(arr_nt, axis=0, ddof=ddof, keepdims=True)

    return mean_1t, sd_1t

def mean_ci95_from_trial_traces(
    arr_nt: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Convert per-trial traces, shape (n_trials,T), to mean and 95% CI half-width,
    each shape (1,T).

    Uses timepoint-specific n and t critical values.
    """
    if arr_nt is None:
        return None, None

    arr_nt = np.asarray(arr_nt, dtype=float)

    if arr_nt.ndim != 2 or arr_nt.shape[0] == 0:
        return None, None

    mean_1t = np.nanmean(arr_nt, axis=0, keepdims=True)

    n_1t = np.sum(np.isfinite(arr_nt), axis=0, keepdims=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        sd_1t = np.nanstd(arr_nt, axis=0, ddof=1, keepdims=True)
        sem_1t = sd_1t / np.sqrt(n_1t)

    ci95_1t = np.full_like(mean_1t, np.nan, dtype=float)

    valid = n_1t > 1

    if np.any(valid):
        tcrit = np.full_like(mean_1t, np.nan, dtype=float)
        tcrit[valid] = stats.t.ppf(0.975, df=n_1t[valid] - 1)
        ci95_1t[valid] = tcrit[valid] * sem_1t[valid]

    return mean_1t, ci95_1t

def mean_ci95_trace_summary(
    arr_nt: np.ndarray | None,
    time: np.ndarray | None = None,
) -> dict[str, np.ndarray] | None:
    """
    Compute mean trace and 95% CI bounds from trial x time data.

    Returns
    -------
    dict with:
        time
        mean
        ci95
        ci_lower
        ci_upper
        n_valid
    """
    if arr_nt is None:
        return None

    arr_nt = np.asarray(arr_nt, dtype=float)

    if arr_nt.ndim != 2:
        return None

    n_trials, n_time = arr_nt.shape

    if n_trials == 0 or n_time == 0:
        return None

    if time is None:
        time = np.arange(n_time, dtype=float)
    else:
        time = np.asarray(time, dtype=float)

    if time.ndim != 1 or time.size != n_time:
        time = np.arange(n_time, dtype=float)

    mean_1t, ci95_1t = mean_ci95_from_trial_traces(arr_nt)

    if mean_1t is None or ci95_1t is None:
        return None

    mean = mean_1t.ravel()
    ci95 = ci95_1t.ravel()
    n_valid = np.sum(np.isfinite(arr_nt), axis=0)

    return {
        "time": time,
        "mean": mean,
        "ci95": ci95,
        "ci_lower": mean - ci95,
        "ci_upper": mean + ci95,
        "n_valid": n_valid,
    }
