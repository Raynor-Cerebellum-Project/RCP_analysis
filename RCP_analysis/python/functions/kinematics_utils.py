from __future__ import annotations

import numpy as np
from scipy import stats

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
