#!/usr/bin/env python3
"""
summarize_stim_response_groups.py

Simple stimulation response grouping analysis.

This script replaces large per-channel raster grids / shuffle-heavy summaries with:

1. Channel/electrode classification:
    - unchanged
    - post_increase
    - post_decrease
    - mixed_post

2. Separate during-stimulation classification:
    - during_increase
    - no_during_increase

3. Two comparison modes:
    - "pre":     stim post/during compared to stim pre-stim baseline
    - "control": stim post/during compared to matched control trials

4. Simple group figures:
    - changed vs unchanged PSTH
    - response type PSTH
    - during increase vs no during increase PSTH
    - class count bar plot
    - Utah grid class map

No stim-rest comparison.
No shuffle analysis in this first draft.
"""

from pathlib import Path
import json
import warnings
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *


# =============================================================================
# Configuration
# =============================================================================

PROCESS_NPRW = True
PROCESS_UA = True

RUN_PRE_COMPARISON = True
RUN_CONTROL_COMPARISON = True

N_JOBS = 8
DPI_OUTPUT = 100

PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim"
FIG_ROOT = OUT_BASE / "figures" / "stim_response_groups"
RESULT_ROOT = OUT_BASE / "checkpoints" / "stim_response_groups"

METADATA_PATH = METADATA_CSV

WIN_PLOT_MS = (-500.0, 500.0)

# Simple analysis windows.
# These are defaults. stim duration is read from the data when possible.
PRE_WIN_MS = (-300.0, -50.0)
POST_EARLY_LEN_MS = 100.0
POST_LATE_LEN_MS = 200.0
POST_LATE_GAP_MS = 100.0

# Optional post-stim offset for avoiding artifacts.
NPRW_POST_OFFSET_MS = 20.0
UA_POST_OFFSET_MS = 0.0

# Classification thresholds.
Z_THRESH = 2.0
DELTA_HZ_THRESH = 5.0

# Classification logic:
# "either": abs(z) >= Z_THRESH OR abs(delta_hz) >= DELTA_HZ_THRESH
# "both":   abs(z) >= Z_THRESH AND abs(delta_hz) >= DELTA_HZ_THRESH
THRESHOLD_MODE = "either"

# Output style.
CLASS_COLORS = {
    "unchanged": "0.55",
    "post_increase": "#d62728",
    "post_decrease": "#1f77b4",
    "mixed_post": "#9467bd",
    "bad_or_missing": "black",
}

DURING_COLORS = {
    "during_increase": "#ff7f0e",
    "no_during_increase": "0.55",
}

GROUP_LINEWIDTH = 2.0
SEM_ALPHA = 0.25


# =============================================================================
# Existing-style metadata / mapping helpers
# =============================================================================

def get_electrode_mapping_csv():
    """
    Return monkey-specific electrode mapping CSV.

    Existing project convention:
        RCP_analysis/config/electrode_port_mapping_<monkey>.csv
    """
    monkey = getattr(PARAMS, "monkey", None)
    if monkey is None:
        monkey = PARAMS.preprocessing.get("monkey", None)

    if monkey not in ("Nike", "Ada"):
        raise ValueError(
            f"Could not determine monkey-specific mapping file. "
            f"Expected monkey to be 'Nike' or 'Ada', got {monkey!r}."
        )

    return Path(__file__).parent.parent / "config" / f"electrode_port_mapping_{monkey}.csv"


def load_utah_mapping(mapping_csv=None):
    """
    Load Utah mapping using package-level project utility.

    rcp.load_electrode_mapping returns:
        nsp_to_elec, region_grids, elec_to_region

    region_grids contains the grid layout used for Utah plotting.
    """
    if mapping_csv is None:
        mapping_csv = get_electrode_mapping_csv()

    if not Path(mapping_csv).exists():
        warnings.warn(f"Utah mapping CSV not found: {mapping_csv}")
        return {
            "mapping_csv": mapping_csv,
            "nsp_to_elec": {},
            "region_grids": {},
            "elec_to_region": {},
        }

    try:
        nsp_to_elec, region_grids, elec_to_region = rcp.load_electrode_mapping(mapping_csv)
    except Exception as exc:
        warnings.warn(f"Could not load Utah mapping with rcp.load_electrode_mapping: {exc}")
        return {
            "mapping_csv": mapping_csv,
            "nsp_to_elec": {},
            "region_grids": {},
            "elec_to_region": {},
        }

    return {
        "mapping_csv": mapping_csv,
        "nsp_to_elec": nsp_to_elec,
        "region_grids": region_grids,
        "elec_to_region": elec_to_region,
    }


def load_bad_channel_map():
    """
    Load bad-channel sets using project impedance utilities.

    Returns
    -------
    dict
        {'utah': set(...), 'nprw': set(...)}
    """
    try:
        bad_map = rcp.get_session_impedances(
            SESSION_LOC,
            utah_exclude_low=False,   # match intended prior behavior
        )
    except Exception as e:
        print(f"[WARN] Could not load bad channels from impedances: {e}")
        bad_map = {"utah": set(), "nprw": set()}

    bad_map.setdefault("utah", set())
    bad_map.setdefault("nprw", set())

    print(
        f"[INFO] Bad channels: "
        f"NPRW={len(bad_map['nprw'])}, UA={len(bad_map['utah'])}"
    )
    return bad_map

# =============================================================================
# File discovery and metadata
# =============================================================================

def find_peristim_files():
    files = sorted(PERI_ROOT.rglob("peristim__*.npz"))

    process_only = PARAMS.preprocessing.get("process_only", None)
    if process_only:
        process_only = {int(x) for x in process_only}

        kept = []
        for f in files:
            br_idx = parse_br_idx_from_name(f)
            if br_idx in process_only:
                kept.append(f)

        files = kept

    return files


def parse_br_idx_from_name(path):
    name = Path(path).name
    m = re.search(r"BR[_\-]?(\d+)", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

def load_metadata_table():
    if METADATA_PATH is None or not Path(METADATA_PATH).exists():
        warnings.warn(f"Metadata CSV not found: {METADATA_PATH}")
        return pd.DataFrame()

    return pd.read_csv(METADATA_PATH)


def safe_name(x):
    x = str(x)
    x = re.sub(r"[^\w\-\.]+", "_", x)
    return x.strip("_") or "unknown"


def extract_metadata_from_file(npz_path, data=None):
    """
    Extract metadata from peristim file and available embedded checkpoint fields.

    Important:
    Do not classify based on the substring 'peristim', because every file has it.
    """
    path = Path(npz_path)
    name = path.stem
    lower = name.lower()

    short = rcp.short_npz_name(path)
    short_lower = short.lower()

    meta = {
        "file": str(path),
        "stem": name,
        "short_name": short,
        "condition": "unknown",
        "target": "unknown",
        "br_idx": parse_br_idx_from_name(path),
        "cond_idx": None,
        "is_stim": False,
        "is_control": False,
    }

    # Embedded metadata/fields from npz, if available.
    if data is not None:
        for key in [
            "target",
            "condition",
            "cond_idx",
            "br_idx",
            "is_stim",
            "is_control",
            "task",
            "trial_type",
        ]:
            if key in data:
                try:
                    val = data[key]
                    if np.ndim(val) == 0:
                        val = val.item()
                    if isinstance(val, bytes):
                        val = val.decode()
                    meta[key] = val
                except Exception:
                    pass

    # Try condition index.
    cond_match = re.search(r"Cond[_\-]?(\d+)", name, flags=re.IGNORECASE)
    if cond_match:
        meta["cond_idx"] = int(cond_match.group(1))

    # Try target.
    target_match = re.search(r"Target[_\-]?([A-Za-z0-9]+)", name, flags=re.IGNORECASE)
    if target_match:
        meta["target"] = f"Target_{target_match.group(1)}"

    # Metadata CSV fallback.
    meta_table = load_metadata_table()
    if not meta_table.empty and meta["br_idx"] is not None and "BR_File" in meta_table.columns:
        br_numeric = pd.to_numeric(meta_table["BR_File"], errors="coerce")
        rows = meta_table.loc[br_numeric == int(meta["br_idx"])]

        if not rows.empty:
            row = rows.iloc[0]

            # Possible useful metadata columns.
            for out_key, possible_cols in {
                "target": ["Target", "target", "Reach_Target", "reach_target"],
                "condition": ["Condition", "condition", "Task", "task", "TrialType", "trial_type", "Type", "type"],
            }.items():
                for col in possible_cols:
                    if col in meta_table.columns and pd.notnull(row[col]):
                        meta[out_key] = row[col]
                        break

            text = " ".join(
                str(row[c]).lower()
                for c in meta_table.columns
                if pd.notnull(row[c])
            )

            if any(x in text for x in ["control", "ctrl", "sham", "no stim", "nostim", "no_stim"]):
                meta["is_control"] = True
                meta["is_stim"] = False
                meta["condition"] = "control"
            elif any(x in text for x in ["stim", "stimulation", "icms"]):
                meta["is_stim"] = True
                meta["is_control"] = False
                meta["condition"] = "stim"

    # Filename fallback.
    # Use short name so 'peristim' does not trigger stim.
    if not bool(meta.get("is_stim", False)) and not bool(meta.get("is_control", False)):
        if any(x in short_lower for x in ["control", "ctrl", "sham", "nostim", "no_stim", "no-stim"]):
            meta["is_control"] = True
            meta["condition"] = "control"
        elif any(x in short_lower for x in ["stim", "icms"]):
            meta["is_stim"] = True
            meta["condition"] = "stim"

    # Final fallback:
    # If no explicit control files are identifiable, treat non-control files as stim.
    # This preserves ability to run while avoiding the old peristim substring bug.
    if not bool(meta.get("is_control", False)) and not bool(meta.get("is_stim", False)):
        meta["is_stim"] = True
        meta["condition"] = "stim_unlabeled"

    return meta


def get_stim_duration_ms(data):
    """
    Get stimulation duration from checkpoint if present.
    Falls back to 100 ms.
    """
    candidate_keys = [
        "stim_duration_ms",
        "stim_dur_ms",
        "duration_ms",
        "stim_duration",
        "stim_dur",
    ]

    for key in candidate_keys:
        if key in data:
            val = data[key]
            try:
                if np.ndim(val) == 0:
                    return float(val.item())
                return float(np.asarray(val).ravel()[0])
            except Exception:
                pass

    return 100.0


def build_file_index(files):
    """
    Build simple stim/control index from available files.
    """
    index = []

    for f in files:
        try:
            with np.load(f, allow_pickle=True) as data:
                meta = extract_metadata_from_file(f, data)
        except Exception:
            meta = extract_metadata_from_file(f, None)

        index.append(meta)

    return pd.DataFrame(index)


def find_best_control_file(stim_meta, file_index):
    """
    Find matched control file for a stim file.

    This is deliberately simple:
    1. same target, control
    2. same BR index, control
    3. any control

    You can later replace this with your existing best-control logic if you have one.
    """
    controls = file_index[file_index["is_control"] == True].copy()

    if controls.empty:
        return None

    target = stim_meta.get("target", "unknown")
    br_idx = stim_meta.get("br_idx", None)

    same_target = controls[controls["target"].astype(str) == str(target)]
    if not same_target.empty:
        return Path(same_target.iloc[0]["file"])

    if br_idx is not None and "br_idx" in controls:
        same_br = controls[controls["br_idx"] == br_idx]
        if not same_br.empty:
            return Path(same_br.iloc[0]["file"])

    return Path(controls.iloc[0]["file"])


def nanmean_axis0(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.full(x.shape[1] if x.ndim > 1 else 1, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(x, axis=0)


def nanstd_axis0(x, ddof=1):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or x.shape[0] <= ddof:
        return np.full(x.shape[1] if x.ndim > 1 else 1, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(x, axis=0, ddof=ddof)

# =============================================================================
# Data extraction
# =============================================================================

def _as_float_array(x):
    return np.asarray(x, dtype=float)


def get_probe_arrays(data, probe):
    """
    Extract counts/time/channel IDs for NPRW or UA.

    Expected checkpoint keys:
        NPRW_counts, NPRW_rel_t
        UA_counts, UA_rel_t

    Optional:
        NPRW_channel_ids
        UA_electrode_ids
        UA_region
        UA_port
        UA_regions

    Returns
    -------
    counts : ndarray
        Trial x channel x time, or channel x time, normalized to trial x channel x time.
    rel_t : ndarray
        Time axis in ms.
    channel_ids : ndarray
        Channel/electrode IDs.
    extra : dict
    """
    if probe.upper() == "NPRW":
        count_keys = ["NPRW_counts", "nprw_counts"]
        time_keys = ["NPRW_rel_t", "nprw_rel_t", "NPRW_t", "nprw_t"]
        id_keys = ["NPRW_channel_ids", "nprw_channel_ids", "NPRW_channels", "nprw_channels"]
    elif probe.upper() == "UA":
        count_keys = ["UA_counts", "ua_counts"]
        time_keys = ["UA_rel_t", "ua_rel_t", "UA_t", "ua_t"]
        id_keys = ["UA_electrode_ids", "ua_electrode_ids", "UA_channels", "ua_channels"]
    else:
        raise ValueError(f"Unknown probe: {probe}")

    counts = None
    rel_t = None
    channel_ids = None

    for key in count_keys:
        if key in data:
            counts = _as_float_array(data[key])
            break

    for key in time_keys:
        if key in data:
            rel_t = _as_float_array(data[key]).ravel()
            break

    for key in id_keys:
        if key in data:
            channel_ids = np.asarray(data[key]).ravel()
            break

    if counts is None or rel_t is None:
        return None, None, None, {}

    # Normalize dimensions to trial x channel x time.
    if counts.ndim == 2:
        # channel x time -> one pseudo-trial
        counts = counts[None, :, :]
    elif counts.ndim == 3:
        # Assume already trial x channel x time.
        pass
    else:
        raise ValueError(f"{probe} counts must be 2D or 3D, got shape {counts.shape}")

    n_ch = counts.shape[1]

    if channel_ids is None:
        channel_ids = np.arange(1, n_ch + 1)

    if len(channel_ids) != n_ch:
        warnings.warn(
            f"{probe} channel ID length {len(channel_ids)} does not match data channels {n_ch}. "
            f"Using sequential IDs."
        )
        channel_ids = np.arange(n_ch)

    extra = {}

    for key in ["UA_regions", "ua_regions", "UA_region", "ua_region", "UA_port", "ua_port"]:
        if key in data:
            extra[key] = data[key]

    return counts, rel_t, channel_ids, extra


def infer_bin_ms(rel_t):
    rel_t = np.asarray(rel_t, dtype=float)
    if len(rel_t) < 2:
        return np.nan
    return float(np.nanmedian(np.diff(rel_t)))


def get_bins_in_window(rel_t, win_ms):
    rel_t = np.asarray(rel_t, dtype=float)
    lo, hi = win_ms
    return np.where((rel_t >= lo) & (rel_t < hi))[0]


def window_rate_hz(counts, rel_t, win_ms):
    """
    Compute trial-wise firing rate in Hz for a window.

    Parameters
    ----------
    counts : ndarray
        trial x channel x time_bin spike counts.
    rel_t : ndarray
        Time axis in ms.
    win_ms : tuple
        Window in ms.

    Returns
    -------
    rates : ndarray
        trial x channel firing rates in Hz.
    """
    bins = get_bins_in_window(rel_t, win_ms)

    if len(bins) == 0:
        return np.full(counts.shape[:2], np.nan)

    bin_ms = infer_bin_ms(rel_t)
    dur_s = len(bins) * bin_ms / 1000.0

    if dur_s <= 0 or not np.isfinite(dur_s):
        return np.full(counts.shape[:2], np.nan)

    spike_counts = np.nansum(counts[:, :, bins], axis=2)
    return spike_counts / dur_s


def psth_hz(counts, rel_t):
    """
    Convert trial x channel x time counts to trial-averaged PSTH in Hz.

    Returns
    -------
    psth : ndarray
        channel x time firing rate.
    """
    bin_ms = infer_bin_ms(rel_t)
    if bin_ms <= 0 or not np.isfinite(bin_ms):
        return np.nanmean(counts, axis=0)

    return np.nanmean(counts, axis=0) / (bin_ms / 1000.0)


# =============================================================================
# Metrics and classification
# =============================================================================

def threshold_pass(delta_hz, z):
    if THRESHOLD_MODE == "both":
        return (np.abs(delta_hz) >= DELTA_HZ_THRESH) & (np.abs(z) >= Z_THRESH)
    elif THRESHOLD_MODE == "either":
        return (np.abs(delta_hz) >= DELTA_HZ_THRESH) | (np.abs(z) >= Z_THRESH)
    else:
        raise ValueError(f"Unknown THRESHOLD_MODE: {THRESHOLD_MODE}")


def signed_threshold(delta_hz, z):
    """
    Return -1, 0, +1 based on thresholded signed effect.
    """
    passed = threshold_pass(delta_hz, z)

    sign = np.zeros_like(np.asarray(delta_hz), dtype=int)
    sign[passed & (delta_hz > 0)] = 1
    sign[passed & (delta_hz < 0)] = -1
    return sign


def compute_pre_comparison_metrics(
    counts,
    rel_t,
    stim_duration_ms,
    probe,
):
    """
    Compare stim trial windows to stim pre-stim window.

    Returns one row per channel.
    """
    post_offset = NPRW_POST_OFFSET_MS if probe.upper() == "NPRW" else UA_POST_OFFSET_MS

    pre_win = PRE_WIN_MS
    during_win = (0.0, stim_duration_ms)
    post_early_win = (
        stim_duration_ms + post_offset,
        stim_duration_ms + post_offset + POST_EARLY_LEN_MS,
    )
    post_late_win = (
        stim_duration_ms + post_offset + POST_LATE_GAP_MS,
        stim_duration_ms + post_offset + POST_LATE_GAP_MS + POST_LATE_LEN_MS,
    )

    pre = window_rate_hz(counts, rel_t, pre_win)
    during = window_rate_hz(counts, rel_t, during_win)
    post_early = window_rate_hz(counts, rel_t, post_early_win)
    post_late = window_rate_hz(counts, rel_t, post_late_win)

    pre_mean = nanmean_axis0(pre)
    pre_std = nanstd_axis0(pre, ddof=1)
    pre_std[pre_std == 0] = np.nan

    during_mean = nanmean_axis0(during)
    post_early_mean = nanmean_axis0(post_early)
    post_late_mean = nanmean_axis0(post_late)

    during_delta = during_mean - pre_mean
    post_early_delta = post_early_mean - pre_mean
    post_late_delta = post_late_mean - pre_mean

    during_z = during_delta / pre_std
    post_early_z = post_early_delta / pre_std
    post_late_z = post_late_delta / pre_std

    metrics = pd.DataFrame({
        "pre_rate_hz": pre_mean,
        "during_rate_hz": during_mean,
        "post_early_rate_hz": post_early_mean,
        "post_late_rate_hz": post_late_mean,
        "during_effect_hz": during_delta,
        "post_early_effect_hz": post_early_delta,
        "post_late_effect_hz": post_late_delta,
        "during_z": during_z,
        "post_early_z": post_early_z,
        "post_late_z": post_late_z,
    })

    metrics.attrs["windows"] = {
        "pre": pre_win,
        "during": during_win,
        "post_early": post_early_win,
        "post_late": post_late_win,
    }

    return metrics


def compute_control_comparison_metrics(
    stim_counts,
    stim_rel_t,
    ctrl_counts,
    ctrl_rel_t,
    stim_duration_ms,
    probe,
):
    """
    Compare stim windows to matched control windows at same event-relative times.
    """
    post_offset = NPRW_POST_OFFSET_MS if probe.upper() == "NPRW" else UA_POST_OFFSET_MS

    pre_win = PRE_WIN_MS
    during_win = (0.0, stim_duration_ms)
    post_early_win = (
        stim_duration_ms + post_offset,
        stim_duration_ms + post_offset + POST_EARLY_LEN_MS,
    )
    post_late_win = (
        stim_duration_ms + post_offset + POST_LATE_GAP_MS,
        stim_duration_ms + post_offset + POST_LATE_GAP_MS + POST_LATE_LEN_MS,
    )

    stim_pre = window_rate_hz(stim_counts, stim_rel_t, pre_win)
    stim_during = window_rate_hz(stim_counts, stim_rel_t, during_win)
    stim_post_early = window_rate_hz(stim_counts, stim_rel_t, post_early_win)
    stim_post_late = window_rate_hz(stim_counts, stim_rel_t, post_late_win)

    ctrl_pre = window_rate_hz(ctrl_counts, ctrl_rel_t, pre_win)
    ctrl_during = window_rate_hz(ctrl_counts, ctrl_rel_t, during_win)
    ctrl_post_early = window_rate_hz(ctrl_counts, ctrl_rel_t, post_early_win)
    ctrl_post_late = window_rate_hz(ctrl_counts, ctrl_rel_t, post_late_win)

    ctrl_pre_mean = nanmean_axis0(ctrl_pre)
    ctrl_during_mean = nanmean_axis0(ctrl_during)
    ctrl_post_early_mean = nanmean_axis0(ctrl_post_early)
    ctrl_post_late_mean = nanmean_axis0(ctrl_post_late)

    ctrl_during_std = nanstd_axis0(ctrl_during, ddof=1)
    ctrl_post_early_std = nanstd_axis0(ctrl_post_early, ddof=1)
    ctrl_post_late_std = nanstd_axis0(ctrl_post_late, ddof=1)

    ctrl_during_std[ctrl_during_std == 0] = np.nan
    ctrl_post_early_std[ctrl_post_early_std == 0] = np.nan
    ctrl_post_late_std[ctrl_post_late_std == 0] = np.nan

    stim_pre_mean = nanmean_axis0(stim_pre)
    stim_during_mean = nanmean_axis0(stim_during)
    stim_post_early_mean = nanmean_axis0(stim_post_early)
    stim_post_late_mean = nanmean_axis0(stim_post_late)

    during_delta = stim_during_mean - ctrl_during_mean
    post_early_delta = stim_post_early_mean - ctrl_post_early_mean
    post_late_delta = stim_post_late_mean - ctrl_post_late_mean

    during_z = during_delta / ctrl_during_std
    post_early_z = post_early_delta / ctrl_post_early_std
    post_late_z = post_late_delta / ctrl_post_late_std

    metrics = pd.DataFrame({
        "pre_rate_hz": stim_pre_mean,
        "during_rate_hz": stim_during_mean,
        "post_early_rate_hz": stim_post_early_mean,
        "post_late_rate_hz": stim_post_late_mean,

        "control_pre_rate_hz": ctrl_pre_mean,
        "control_during_rate_hz": ctrl_during_mean,
        "control_post_early_rate_hz": ctrl_post_early_mean,
        "control_post_late_rate_hz": ctrl_post_late_mean,

        "during_effect_hz": during_delta,
        "post_early_effect_hz": post_early_delta,
        "post_late_effect_hz": post_late_delta,

        "during_z": during_z,
        "post_early_z": post_early_z,
        "post_late_z": post_late_z,
    })

    metrics.attrs["windows"] = {
        "pre": pre_win,
        "during": during_win,
        "post_early": post_early_win,
        "post_late": post_late_win,
    }

    return metrics


def classify_channels(metrics):
    """
    Add response class labels to metrics table.

    Main post class:
        unchanged
        post_increase
        post_decrease
        mixed_post

    Separate during class:
        during_increase
        no_during_increase
    """
    early_sign = signed_threshold(
        metrics["post_early_effect_hz"].to_numpy(),
        metrics["post_early_z"].to_numpy(),
    )

    late_sign = signed_threshold(
        metrics["post_late_effect_hz"].to_numpy(),
        metrics["post_late_z"].to_numpy(),
    )

    during_sign = signed_threshold(
        metrics["during_effect_hz"].to_numpy(),
        metrics["during_z"].to_numpy(),
    )

    post_class = []

    for e, l in zip(early_sign, late_sign):
        if (e == 1 and l == -1) or (e == -1 and l == 1):
            post_class.append("mixed_post")
        elif e == 1 or l == 1:
            post_class.append("post_increase")
        elif e == -1 or l == -1:
            post_class.append("post_decrease")
        else:
            post_class.append("unchanged")

    during_class = np.where(
        during_sign == 1,
        "during_increase",
        "no_during_increase",
    )

    out = metrics.copy()
    out["post_class"] = post_class
    out["changed_post"] = out["post_class"] != "unchanged"
    out["during_class"] = during_class
    out["during_increase"] = out["during_class"] == "during_increase"

    return out


# =============================================================================
# Plotting
# =============================================================================

def set_common_style(ax):
    ax.axvline(0, color="k", lw=1.0, ls="--", alpha=0.8)
    ax.set_xlabel("Time from stimulation onset / event (ms)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_group_psth(
    rel_t,
    psth,
    classes,
    groups,
    colors,
    title,
    out_path,
    stim_duration_ms=None,
):
    """
    Plot channel-group mean PSTHs.

    Parameters
    ----------
    rel_t : ndarray
        Time axis.
    psth : ndarray
        channel x time.
    classes : array-like
        Class label per channel.
    groups : list
        Group labels to plot.
    colors : dict
        Map group label to color.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rel_t = np.asarray(rel_t)
    psth = np.asarray(psth)
    classes = np.asarray(classes)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=DPI_OUTPUT)

    if stim_duration_ms is not None:
        ax.axvspan(0, stim_duration_ms, color="gold", alpha=0.18, lw=0)

    for group in groups:
        idx = np.where(classes == group)[0]
        if len(idx) == 0:
            continue

        y = psth[idx, :]
        mean = np.nanmean(y, axis=0)
        sem = np.nanstd(y, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(y), axis=0))

        color = colors.get(group, None)

        ax.plot(rel_t, mean, lw=GROUP_LINEWIDTH, color=color, label=f"{group} (n={len(idx)})")
        ax.fill_between(rel_t, mean - sem, mean + sem, color=color, alpha=SEM_ALPHA, lw=0)

    ax.set_xlim(WIN_PLOT_MS)
    ax.set_title(title)
    set_common_style(ax)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_changed_vs_unchanged(rel_t, psth, metrics, title, out_path, stim_duration_ms=None):
    classes = np.where(metrics["changed_post"].to_numpy(), "changed", "unchanged")

    colors = {
        "changed": "#d62728",
        "unchanged": "0.55",
    }

    plot_group_psth(
        rel_t=rel_t,
        psth=psth,
        classes=classes,
        groups=["changed", "unchanged"],
        colors=colors,
        title=title,
        out_path=out_path,
        stim_duration_ms=stim_duration_ms,
    )


def plot_response_types(rel_t, psth, metrics, title, out_path, stim_duration_ms=None):
    plot_group_psth(
        rel_t=rel_t,
        psth=psth,
        classes=metrics["post_class"].to_numpy(),
        groups=["post_increase", "post_decrease", "mixed_post", "unchanged"],
        colors=CLASS_COLORS,
        title=title,
        out_path=out_path,
        stim_duration_ms=stim_duration_ms,
    )


def plot_during_groups(rel_t, psth, metrics, title, out_path, stim_duration_ms=None):
    plot_group_psth(
        rel_t=rel_t,
        psth=psth,
        classes=metrics["during_class"].to_numpy(),
        groups=["during_increase", "no_during_increase"],
        colors=DURING_COLORS,
        title=title,
        out_path=out_path,
        stim_duration_ms=stim_duration_ms,
    )


def plot_class_counts(metrics, title, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    class_order = ["unchanged", "post_increase", "post_decrease", "mixed_post"]
    counts = metrics["post_class"].value_counts().reindex(class_order).fillna(0)

    fig, ax = plt.subplots(figsize=(5, 4), dpi=DPI_OUTPUT)

    colors = [CLASS_COLORS[c] for c in class_order]
    ax.bar(class_order, counts.values, color=colors, edgecolor="k", linewidth=0.5)

    ax.set_ylabel("Number of channels")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ua_class_grid(metrics, mapping, region, title, out_path):
    """
    Plot Utah array class map using project region_grids.

    mapping should be the dict returned by load_utah_mapping():
        {
            "nsp_to_elec": ...,
            "region_grids": ...,
            "elec_to_region": ...
        }
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if mapping is None:
        return

    region_grids = mapping.get("region_grids", {})
    if not region_grids:
        warnings.warn("Could not plot UA grid because region_grids is empty.")
        return

    try:
        grid = rcp.get_region_grid(region, region_grids)
    except Exception:
        grid = region_grids.get(region, None)

    if grid is None:
        warnings.warn(f"No Utah grid found for region {region}")
        return

    grid = np.asarray(grid)

    class_by_elec = {
        int(row["channel_id"]): row["post_class"]
        for _, row in metrics.iterrows()
        if pd.notnull(row["channel_id"])
    }

    fig, ax = plt.subplots(figsize=(5, 5), dpi=DPI_OUTPUT)

    n_rows, n_cols = grid.shape

    for r in range(n_rows):
        for c in range(n_cols):
            try:
                elec = int(grid[r, c])
            except Exception:
                elec = None

            if elec is None or elec <= 0:
                cls = "bad_or_missing"
                label = ""
            else:
                cls = class_by_elec.get(elec, "bad_or_missing")
                label = str(elec)

            color = CLASS_COLORS.get(cls, "black")
            rect = plt.Rectangle(
                (c, r),
                1,
                1,
                facecolor=color,
                edgecolor="white",
                linewidth=0.7,
            )
            ax.add_patch(rect)

            if label:
                txt_color = "white" if cls in ["post_decrease", "mixed_post", "bad_or_missing"] else "black"
                ax.text(
                    c + 0.5,
                    r + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=txt_color,
                )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            color=CLASS_COLORS[k],
            label=k,
        )
        for k in [
            "unchanged",
            "post_increase",
            "post_decrease",
            "mixed_post",
            "bad_or_missing",
        ]
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# =============================================================================
# Probe processing
# =============================================================================

def add_common_columns(metrics, meta, probe, comparison_mode, channel_ids, region=None):
    metrics = metrics.copy()

    metrics.insert(0, "channel_id", channel_ids)
    metrics.insert(0, "region", region if region is not None else "")
    metrics.insert(0, "probe", probe)
    metrics.insert(0, "comparison_mode", comparison_mode)
    metrics.insert(0, "target", meta.get("target", "unknown"))
    metrics.insert(0, "condition", meta.get("condition", "unknown"))
    metrics.insert(0, "br_idx", meta.get("br_idx", None))
    metrics.insert(0, "source_file", meta.get("file", ""))

    return metrics


def filter_bad_channels(metrics, bad_channels):
    metrics = metrics.copy()
    metrics["is_bad_channel"] = metrics["channel_id"].apply(
        lambda x: int(x) in bad_channels if pd.notnull(x) else False
    )
    return metrics


def process_probe_pre(
    stim_data,
    stim_meta,
    probe,
    out_dir,
    region=None,
    mapping=None,
    bad_channels=None,
):
    counts, rel_t, channel_ids, extra = get_probe_arrays(stim_data, probe)
    if counts is None:
        return []

    stim_duration_ms = get_stim_duration_ms(stim_data)

    metrics = compute_pre_comparison_metrics(
        counts=counts,
        rel_t=rel_t,
        stim_duration_ms=stim_duration_ms,
        probe=probe,
    )
    metrics = classify_channels(metrics)
    metrics = add_common_columns(
        metrics,
        meta=stim_meta,
        probe=probe,
        comparison_mode="pre",
        channel_ids=channel_ids,
        region=region,
    )

    if bad_channels is not None:
        metrics = filter_bad_channels(metrics, bad_channels)

    psth = psth_hz(counts, rel_t)

    make_all_figures(
        rel_t=rel_t,
        psth=psth,
        metrics=metrics,
        out_dir=out_dir / "pre",
        title_prefix=f"{probe} {region or ''} pre comparison",
        stim_duration_ms=stim_duration_ms,
        mapping=mapping,
        region=region,
    )

    save_metrics(metrics, out_dir / "pre" / "channel_classes.csv")

    return [metrics]


def process_probe_control(
    stim_data,
    ctrl_data,
    stim_meta,
    probe,
    out_dir,
    region=None,
    mapping=None,
    bad_channels=None,
):
    stim_counts, stim_rel_t, stim_channel_ids, stim_extra = get_probe_arrays(stim_data, probe)
    ctrl_counts, ctrl_rel_t, ctrl_channel_ids, ctrl_extra = get_probe_arrays(ctrl_data, probe)

    if stim_counts is None or ctrl_counts is None:
        return []

    # Require same number of channels for first draft.
    # Later we can align explicitly by channel ID if needed.
    if stim_counts.shape[1] != ctrl_counts.shape[1]:
        warnings.warn(
            f"{probe}: stim/control channel count mismatch "
            f"{stim_counts.shape[1]} vs {ctrl_counts.shape[1]}; skipping control comparison."
        )
        return []

    stim_duration_ms = get_stim_duration_ms(stim_data)

    metrics = compute_control_comparison_metrics(
        stim_counts=stim_counts,
        stim_rel_t=stim_rel_t,
        ctrl_counts=ctrl_counts,
        ctrl_rel_t=ctrl_rel_t,
        stim_duration_ms=stim_duration_ms,
        probe=probe,
    )
    metrics = classify_channels(metrics)
    metrics = add_common_columns(
        metrics,
        meta=stim_meta,
        probe=probe,
        comparison_mode="control",
        channel_ids=stim_channel_ids,
        region=region,
    )

    if bad_channels is not None:
        metrics = filter_bad_channels(metrics, bad_channels)

    psth = psth_hz(stim_counts, stim_rel_t)

    make_all_figures(
        rel_t=stim_rel_t,
        psth=psth,
        metrics=metrics,
        out_dir=out_dir / "control",
        title_prefix=f"{probe} {region or ''} control comparison",
        stim_duration_ms=stim_duration_ms,
        mapping=mapping,
        region=region,
    )

    save_metrics(metrics, out_dir / "control" / "channel_classes.csv")

    return [metrics]


def make_all_figures(
    rel_t,
    psth,
    metrics,
    out_dir,
    title_prefix,
    stim_duration_ms,
    mapping=None,
    region=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exclude bad channels from group summaries if present.
    plot_metrics = metrics.copy()
    plot_psth = psth.copy()

    if "is_bad_channel" in plot_metrics.columns:
        keep = ~plot_metrics["is_bad_channel"].to_numpy(dtype=bool)
        plot_metrics = plot_metrics.loc[keep].reset_index(drop=True)
        plot_psth = plot_psth[keep, :]

    plot_changed_vs_unchanged(
        rel_t=rel_t,
        psth=plot_psth,
        metrics=plot_metrics,
        title=f"{title_prefix}: changed vs unchanged",
        out_path=out_dir / "group_psth_changed_vs_unchanged.png",
        stim_duration_ms=stim_duration_ms,
    )

    plot_response_types(
        rel_t=rel_t,
        psth=plot_psth,
        metrics=plot_metrics,
        title=f"{title_prefix}: response types",
        out_path=out_dir / "group_psth_response_types.png",
        stim_duration_ms=stim_duration_ms,
    )

    plot_during_groups(
        rel_t=rel_t,
        psth=plot_psth,
        metrics=plot_metrics,
        title=f"{title_prefix}: during-stim increase",
        out_path=out_dir / "group_psth_during_increase.png",
        stim_duration_ms=stim_duration_ms,
    )

    plot_class_counts(
        metrics=plot_metrics,
        title=f"{title_prefix}: channel classes",
        out_path=out_dir / "channel_class_counts.png",
    )

    if mapping is not None and region is not None:
        plot_ua_class_grid(
            metrics=plot_metrics,
            mapping=mapping,
            region=region,
            title=f"{title_prefix}: Utah class map",
            out_path=out_dir / f"ua_class_grid_{safe_name(region)}.png",
        )


def save_metrics(metrics, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_path, index=False)


# =============================================================================
# UA region handling
# =============================================================================

def get_ua_regions_from_mapping(mapping):
    """
    Return Utah region names from project mapping object.
    """
    if mapping is None:
        return []

    region_grids = mapping.get("region_grids", {})
    if region_grids:
        return sorted(region_grids.keys())

    elec_to_region = mapping.get("elec_to_region", {})
    if elec_to_region:
        return sorted(set(str(v) for v in elec_to_region.values()))

    return ["UA"]


def subset_ua_by_region(counts, channel_ids, mapping, region):
    """
    Subset UA counts/channel IDs to one mapped region.

    Uses elec_to_region returned by rcp.load_electrode_mapping.
    """
    if mapping is None:
        return counts, channel_ids

    elec_to_region = mapping.get("elec_to_region", {})
    if not elec_to_region:
        return counts, channel_ids

    keep_idx = []
    for i, ch in enumerate(channel_ids):
        try:
            elec = int(ch)
        except Exception:
            continue

        mapped_region = elec_to_region.get(elec, None)

        if mapped_region is None:
            # Some mappings may store keys as strings.
            mapped_region = elec_to_region.get(str(elec), None)

        if mapped_region is not None and str(mapped_region) == str(region):
            keep_idx.append(i)

    if len(keep_idx) == 0:
        return None, None

    keep_idx = np.asarray(keep_idx, dtype=int)
    return counts[:, keep_idx, :], np.asarray(channel_ids)[keep_idx]

def process_ua_region_pre(
    stim_data,
    stim_meta,
    out_dir,
    region,
    mapping,
    bad_channels,
):
    counts, rel_t, channel_ids, extra = get_probe_arrays(stim_data, "UA")
    if counts is None:
        return []

    sub_counts, sub_channel_ids = subset_ua_by_region(counts, channel_ids, mapping, region)
    if sub_counts is None:
        return []

    # Create lightweight data-like object by processing arrays directly.
    stim_duration_ms = get_stim_duration_ms(stim_data)

    metrics = compute_pre_comparison_metrics(
        counts=sub_counts,
        rel_t=rel_t,
        stim_duration_ms=stim_duration_ms,
        probe="UA",
    )
    metrics = classify_channels(metrics)
    metrics = add_common_columns(
        metrics,
        meta=stim_meta,
        probe="UA",
        comparison_mode="pre",
        channel_ids=sub_channel_ids,
        region=region,
    )
    metrics = filter_bad_channels(metrics, bad_channels)

    psth = psth_hz(sub_counts, rel_t)

    region_out = out_dir / "UA" / safe_name(region)

    make_all_figures(
        rel_t=rel_t,
        psth=psth,
        metrics=metrics,
        out_dir=region_out / "pre",
        title_prefix=f"UA {region} pre comparison",
        stim_duration_ms=stim_duration_ms,
        mapping=mapping,
        region=region,
    )

    save_metrics(metrics, region_out / "pre" / "channel_classes.csv")

    return [metrics]


def process_ua_region_control(
    stim_data,
    ctrl_data,
    stim_meta,
    out_dir,
    region,
    mapping,
    bad_channels,
):
    stim_counts, stim_rel_t, stim_channel_ids, _ = get_probe_arrays(stim_data, "UA")
    ctrl_counts, ctrl_rel_t, ctrl_channel_ids, _ = get_probe_arrays(ctrl_data, "UA")

    if stim_counts is None or ctrl_counts is None:
        return []

    stim_sub_counts, stim_sub_ids = subset_ua_by_region(
        stim_counts, stim_channel_ids, mapping, region
    )
    ctrl_sub_counts, ctrl_sub_ids = subset_ua_by_region(
        ctrl_counts, ctrl_channel_ids, mapping, region
    )

    if stim_sub_counts is None or ctrl_sub_counts is None:
        return []

    if stim_sub_counts.shape[1] != ctrl_sub_counts.shape[1]:
        warnings.warn(
            f"UA {region}: stim/control region channel count mismatch; "
            f"skipping control comparison."
        )
        return []

    stim_duration_ms = get_stim_duration_ms(stim_data)

    metrics = compute_control_comparison_metrics(
        stim_counts=stim_sub_counts,
        stim_rel_t=stim_rel_t,
        ctrl_counts=ctrl_sub_counts,
        ctrl_rel_t=ctrl_rel_t,
        stim_duration_ms=stim_duration_ms,
        probe="UA",
    )

    metrics = classify_channels(metrics)
    metrics = add_common_columns(
        metrics,
        meta=stim_meta,
        probe="UA",
        comparison_mode="control",
        channel_ids=stim_sub_ids,
        region=region,
    )
    metrics = filter_bad_channels(metrics, bad_channels)

    psth = psth_hz(stim_sub_counts, stim_rel_t)

    region_out = out_dir / "UA" / safe_name(region)

    make_all_figures(
        rel_t=stim_rel_t,
        psth=psth,
        metrics=metrics,
        out_dir=region_out / "control",
        title_prefix=f"UA {region} control comparison",
        stim_duration_ms=stim_duration_ms,
        mapping=mapping,
        region=region,
    )

    save_metrics(metrics, region_out / "control" / "channel_classes.csv")

    return [metrics]


# =============================================================================
# Main file processing
# =============================================================================

def process_stim_file(stim_file, file_index, mapping, bad_channel_map):
    all_metrics = []

    try:
        stim_data = np.load(stim_file, allow_pickle=True)
    except Exception as exc:
        warnings.warn(f"Could not load {stim_file}: {exc}")
        return []

    stim_meta = extract_metadata_from_file(stim_file, stim_data)

    # Only process stim files.
    if not stim_meta.get("is_stim", False):
        return []

    br = stim_meta.get("br_idx", "unknown")
    target = safe_name(stim_meta.get("target", "unknown"))
    stem = safe_name(Path(stim_file).stem)

    out_dir = FIG_ROOT / f"BR_{br}" / target / stem

    ctrl_file = None
    ctrl_data = None

    if RUN_CONTROL_COMPARISON:
        ctrl_file = find_best_control_file(stim_meta, file_index)
        if ctrl_file is not None:
            try:
                ctrl_data = np.load(ctrl_file, allow_pickle=True)
            except Exception as exc:
                warnings.warn(f"Could not load control file {ctrl_file}: {exc}")
                ctrl_data = None

    # -------------------------------------------------------------------------
    # NPRW
    # -------------------------------------------------------------------------
    if PROCESS_NPRW:
        nprw_out = out_dir / "NPRW"

        if RUN_PRE_COMPARISON:
            all_metrics.extend(
                process_probe_pre(
                    stim_data=stim_data,
                    stim_meta=stim_meta,
                    probe="NPRW",
                    out_dir=nprw_out,
                    region=None,
                    mapping=None,
                    bad_channels=bad_channel_map.get("nprw", set()),
                )
            )

        if RUN_CONTROL_COMPARISON and ctrl_data is not None:
            all_metrics.extend(
                process_probe_control(
                    stim_data=stim_data,
                    ctrl_data=ctrl_data,
                    stim_meta=stim_meta,
                    probe="NPRW",
                    out_dir=nprw_out,
                    region=None,
                    mapping=None,
                    bad_channels=bad_channel_map.get("nprw", set()),
                )
            )

    # -------------------------------------------------------------------------
    # Utah Array
    # -------------------------------------------------------------------------
    if PROCESS_UA:
        ua_regions = get_ua_regions_from_mapping(mapping)

        for region in ua_regions:
            if RUN_PRE_COMPARISON:
                all_metrics.extend(
                    process_ua_region_pre(
                        stim_data=stim_data,
                        stim_meta=stim_meta,
                        out_dir=out_dir,
                        region=region,
                        mapping=mapping,
                        bad_channels=bad_channel_map.get("utah", set()),
                    )
                )

            if RUN_CONTROL_COMPARISON and ctrl_data is not None:
                all_metrics.extend(
                    process_ua_region_control(
                        stim_data=stim_data,
                        ctrl_data=ctrl_data,
                        stim_meta=stim_meta,
                        out_dir=out_dir,
                        region=region,
                        mapping=mapping,
                        bad_channels=bad_channel_map.get("utah", set()),
                    )
                )

    try:
        stim_data.close()
    except Exception:
        pass

    if ctrl_data is not None:
        try:
            ctrl_data.close()
        except Exception:
            pass

    return all_metrics


def save_combined_results(all_metrics):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    if len(all_metrics) == 0:
        warnings.warn("No metrics to save.")
        return

    combined = pd.concat(all_metrics, ignore_index=True)

    combined_csv = RESULT_ROOT / "stim_response_channel_classes_all.csv"
    combined.to_csv(combined_csv, index=False)

    combined_npy = RESULT_ROOT / "stim_response_channel_classes_all.npy"
    np.save(combined_npy, combined.to_dict("records"), allow_pickle=True)

    # Small condition summary.
    group_cols = [
        "br_idx",
        "target",
        "probe",
        "region",
        "comparison_mode",
        "post_class",
        "during_class",
    ]

    summary = (
        combined
        .groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_channels")
    )

    summary_csv = RESULT_ROOT / "stim_response_group_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"Saved combined results:")
    print(f"  {combined_csv}")
    print(f"  {summary_csv}")


def main():
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Simple stimulation response grouping analysis")
    print("=" * 80)
    print(f"Session:       {getattr(PARAMS, 'session', 'unknown')}")
    print(f"Monkey:        {getattr(PARAMS, 'monkey', 'unknown')}")
    print(f"Peristim root: {PERI_ROOT}")
    print(f"Figure root:   {FIG_ROOT}")
    print(f"Result root:   {RESULT_ROOT}")
    print(f"Metadata CSV:  {METADATA_PATH}")
    print(f"Comparison modes:")
    print(f"  pre:     {RUN_PRE_COMPARISON}")
    print(f"  control: {RUN_CONTROL_COMPARISON}")
    print(f"Threshold mode: {THRESHOLD_MODE}")
    print(f"Z threshold:    {Z_THRESH}")
    print(f"Delta threshold:{DELTA_HZ_THRESH} Hz")
    print("=" * 80)

    files = find_peristim_files()

    if len(files) == 0:
        raise RuntimeError(f"No peristim files found under {PERI_ROOT}")

    print(f"Found {len(files)} peristim files.")

    file_index = build_file_index(files)

    n_stim = int(file_index["is_stim"].sum()) if "is_stim" in file_index else 0
    n_ctrl = int(file_index["is_control"].sum()) if "is_control" in file_index else 0
    print(f"Found {n_stim} stim files.")
    print(f"Found {n_ctrl} control files.")

    stim_files = [
        Path(row["file"])
        for _, row in file_index.iterrows()
        if bool(row.get("is_stim", False))
    ]

    mapping = load_utah_mapping()
    bad_channel_map = load_bad_channel_map()

    print(f"Loaded Utah mapping from: {mapping.get('mapping_csv', None)}")
    print(f"Loaded Utah regions: {list(mapping.get('region_grids', {}).keys())}")
    print(
        f"Loaded bad/excluded channels: "
        f"NPRW={len(bad_channel_map.get('nprw', set()))}, "
        f"UA={len(bad_channel_map.get('utah', set()))}"
    )

    if len(stim_files) == 0:
        warnings.warn("No stim files were identified. Nothing to process.")
        return

    if N_JOBS == 1:
        nested_results = [
            process_stim_file(f, file_index, mapping, bad_channel_map)
            for f in stim_files
        ]
    else:
        nested_results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(process_stim_file)(f, file_index, mapping, bad_channel_map)
            for f in stim_files
        )

    all_metrics = []
    for result in nested_results:
        for item in result:
            if isinstance(item, pd.DataFrame) and len(item) > 0:
                all_metrics.append(item)

    save_combined_results(all_metrics)

    print("Done.")

if __name__ == "__main__":
    main()