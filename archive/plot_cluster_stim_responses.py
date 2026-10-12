"""
plot_behavioral_stim_perturbation.py

Channel-Behavior Regression Analysis across Cerebellum (NPRW) and Motor Cortex (Utah Array).

Scientific Question:
    Which neural channels carry information relevant to the behavioral perturbation
    induced by stimulation on a trial-by-trial basis?

Key Methodology:
    1. Trial-Level Behavioral Perturbation Target:
       - Extracts multi-camera 3D kinematics (cam0 and cam1) using extract_xy_coordinates.
       - Concatenates camera axes into a unified 4D trajectory [x0, y0, x1, y1] per trial.
       - Computes each trial's mean Euclidean distance from the control-mean trajectory
         within a defined post-stimulus behavioral window (e.g. 100-400 ms).
       - Provides a scalar behavioral score y_i measuring trial-by-trial deviation/perturbation.
    2. Trial-Level Neural Feature Extraction:
       - Firing rate per channel averaged across an early post-stimulus neural window
         (e.g. 50-200 ms post-stimulus onset).
       - Stacks both STIM and CTRL trials together without condition-level subtraction.
    3. Ridge Regression with Cross-Validation:
       - RidgeCV with 5-fold cross-validation predicting perturbation from neural features.
       - Evaluates out-of-fold generalization (R² and Pearson r).
    4. Exact Vectorized Permutation Testing:
       - Shuffles behavioral targets across trials (B=1000 permutations) using precomputed
         projection matrices for instantaneous exact null distributions and FDR q-values.
    5. Modular & Consolidated Visualizations:
       - plot_channel_weights: Bar chart with significance stars (* p<0.05, ** p<0.01).
       - plot_prediction_scatter: Actual vs predicted perturbation scatter colored by STIM/CTRL.
       - plot_top_channels_traces: Time-resolved firing rate traces for top-weight channels.
       - plot_regression_summary_figure: Consolidated publication-grade 3-panel figure.
"""

from __future__ import annotations

import os
import re
import sys
import json
import math
import argparse
import warnings
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Sequence

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter1d
from scipy import stats

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict

# Suppress benign warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Degrees of freedom <= 0 for slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN slice encountered.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*tight_layout.*")

import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *
from RCP_analysis.python.functions.kinematics_utils import extract_xy_coordinates

# =============================================================================
# Configuration & Analysis Parameters
# =============================================================================

# Time Windows (ms relative to stimulation / IR beam onset)
WIN_PLOT_MS: Tuple[float, float] = (-800.0, 600.0)
DEFAULT_NEURAL_WINDOW_MS: Tuple[float, float] = (50.0, 200.0)
DEFAULT_BEHAVIOR_WINDOW_MS: Tuple[float, float] = (100.0, 400.0)

# Regression Settings
DEFAULT_CV_FOLDS: int = 5
DEFAULT_N_PERMUTATIONS: int = 1000
DEFAULT_ALPHAS: np.ndarray = np.logspace(-2, 4, 20)
FDR_ALPHA: float = 0.05
P_VAL_THRESH: float = 0.05

# Stimulation Artifact Blanking Parameters - derived from params.yaml
NPRW_BLANK_PRE_MS = float(PARAMS.NPRW_rate_est.get("remove_ms_before", 20.0)) if PARAMS else 20.0
NPRW_BLANK_POST_MS = float(PARAMS.NPRW_rate_est.get("remove_tail_ms_after", 20.0)) if PARAMS else 20.0
UA_BLANK_PRE_MS = float(PARAMS.UA_rate_est.get("remove_ms_before", 0.0)) if PARAMS else 0.0
UA_BLANK_POST_MS = float(PARAMS.UA_rate_est.get("remove_tail_ms_after", 0.0)) if PARAMS else 0.0
BLANK_UA_STIM_PERIOD = False

# Display Binning & Smoothing - derived from rate_est in params.yaml
NPRW_BIN_WIDTH_MS = float(PARAMS.NPRW_rate_est.get("bin_ms", 20.0)) if PARAMS else 20.0
NPRW_SMOOTHING_SIGMA_MS = float(PARAMS.NPRW_rate_est.get("sigma_ms", 20.0)) if PARAMS else 20.0
UA_BIN_WIDTH_MS = float(PARAMS.UA_rate_est.get("bin_ms", 20.0)) if PARAMS else 20.0
UA_SMOOTHING_SIGMA_MS = float(PARAMS.UA_rate_est.get("sigma_ms", 20.0)) if PARAMS else 20.0


def resolve_process_only() -> Optional[List[Any]]:
    env_val = os.environ.get("RCP_PROCESS_ONLY")
    if env_val:
        try:
            parsed = json.loads(env_val)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except Exception:
            pass
    if PARAMS is not None:
        po = getattr(PARAMS, "process_only", None)
        if isinstance(po, list) and len(po) > 0:
            return po
        if hasattr(PARAMS, "preprocessing"):
            po2 = PARAMS.preprocessing.get("process_only", None)
            if isinstance(po2, list) and len(po2) > 0:
                return po2
    return None


PROCESS_ONLY: Optional[List[Any]] = resolve_process_only()
ANALYZE_COMBINED_UTAH: bool = True
ANALYZE_INDIVIDUAL_UTAH_ARRAYS: bool = True

REGION_ORDER = ["SMA", "PMd", "M1i", "M1s"]

# File System Roots
PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim" if OUT_BASE is not None else Path(".")
REG_FIG_ROOT = OUT_BASE / "figures" / "channel_behavior_regression" if OUT_BASE is not None else Path(".")
CLUST_FIG_ROOT = OUT_BASE / "figures" / "stim_response_clustering" if OUT_BASE is not None else Path(".")
METADATA_PATH = METADATA_CSV if "METADATA_CSV" in globals() else None


# =============================================================================
# Helper Utilities
# =============================================================================

def _log(msg: str):
    print(f"[channel_behavior_regression] {msg}", flush=True)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_get_npz(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key in data.files:
        return data[key]
    return default


def scalarize(x, default=None):
    if x is None:
        return default
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return default
        return arr.item()
    except Exception:
        return x


def sanitize_name(s: Any) -> str:
    return re.sub(r"[^\w\.-]", "_", str(s))


def get_cond_type(npz_path: Path) -> str:
    path_str = str(npz_path).lower()
    if "control_reaches" in path_str or "target_control" in path_str:
        return "CTRL"
    if "stim_reaches" in path_str:
        return "STIM"
    if "at_rest" in path_str or "quiescent" in path_str or "no_reach" in path_str:
        return "REST"
    if "continuous_stim" in path_str:
        return "CONTSTIM"

    # Fallback checking file name without the 'peristim' prefix
    name = npz_path.name.lower().replace("peristim__", "").replace("peristim_", "")
    if "control" in name or "baseline" in name:
        return "CTRL"
    if "rest" in name:
        return "REST"
    if "stim" in name:
        return "STIM"
    return "OTHER"


def get_target_folder(npz_path: Path) -> str:
    path_str = str(npz_path).lower()
    if "target_a" in path_str or "_target_a" in path_str:
        return "Target_A"
    if "target_b" in path_str or "_target_b" in path_str:
        return "Target_B"
    if "target_control" in path_str or "_target_control" in path_str:
        return "Target_control"

    stem = npz_path.stem
    m = re.search(r"target_([A-Za-z0-9_-]+)", stem, re.IGNORECASE)
    if m:
        return f"Target_{m.group(1).upper()}"
    for p in npz_path.parents:
        m2 = re.search(r"target_([A-Za-z0-9_-]+)", p.name, re.IGNORECASE)
        if m2:
            return f"Target_{m2.group(1).upper()}"
    return ""


def get_stim_duration_ms(data: np.lib.npyio.NpzFile) -> float:
    for key in ("stim_dur_measured_ms", "stim_dur_nominal_ms", "stim_dur_ms"):
        val = safe_get_npz(data, key, None)
        if val is not None:
            arr = np.asarray(val, dtype=float).ravel()
            valid = arr[np.isfinite(arr) & (arr > 0)]
            if valid.size > 0:
                return float(np.median(valid))

    freq_val = safe_get_npz(data, "stim_freq_hz", None)
    pulse_val = safe_get_npz(data, "stim_pulse_counts", None)
    if freq_val is not None and pulse_val is not None:
        try:
            f = float(np.asarray(freq_val).ravel()[0])
            p_arr = np.asarray(pulse_val, dtype=float).ravel()
            p_arr = p_arr[np.isfinite(p_arr) & (p_arr > 0)]
            if f > 0 and p_arr.size > 0:
                p = float(np.median(p_arr))
                dur = (p / f) * 1000.0
                if dur > 0:
                    return dur
        except Exception:
            pass

    meta_raw = safe_get_npz(data, "align_meta_raw", None)
    if meta_raw is not None:
        try:
            m_dict = meta_raw.item() if hasattr(meta_raw, "item") else dict(meta_raw)
            if "stim_dur_nominal_ms" in m_dict:
                d = float(m_dict["stim_dur_nominal_ms"])
                if d > 0:
                    return d
        except Exception:
            pass

    return 0.0


def get_ua_port(data: np.lib.npyio.NpzFile) -> str:
    val = safe_get_npz(data, "ua_port", None)
    if val is not None:
        s = str(scalarize(val)).strip()
        if s:
            return s
    val = safe_get_npz(data, "port", None)
    if val is not None:
        s = str(scalarize(val)).strip()
        if s:
            return s
    return "A"


def compute_pulse_count(freq_hz: Optional[float], dur_ms: Optional[float]) -> Optional[float]:
    if freq_hz is None or dur_ms is None:
        return None
    try:
        f = float(freq_hz)
        d = float(dur_ms)
        if f > 0 and d > 0:
            return round((f * d) / 1000.0, 1)
    except Exception:
        pass
    return None


def load_reference_mapping(all_files: List[Path]) -> Dict[str, Any]:
    ctrl_map: Dict[Tuple[str, str], Path] = {}
    ctrl_by_target: Dict[str, Path] = {}
    ctrl_fallback: Optional[Path] = None
    rest_map: Dict[float, Path] = {}
    rest_fallback: Optional[Path] = None

    for f in all_files:
        ctype = get_cond_type(f)
        target = get_target_folder(f)
        try:
            with np.load(f, allow_pickle=True) as d:
                port = get_ua_port(d)
                fh = scalarize(safe_get_npz(d, "stim_freq_hz", None))
                dm = scalarize(safe_get_npz(d, "stim_dur_nominal_ms", None))
                pc = compute_pulse_count(fh, dm)
        except Exception:
            port = "A"
            pc = None

        if ctype == "CTRL":
            if (target, port) not in ctrl_map:
                ctrl_map[(target, port)] = f
            if target and target not in ctrl_by_target:
                ctrl_by_target[target] = f
            if ctrl_fallback is None:
                ctrl_fallback = f
        elif ctype == "REST":
            if pc is not None and pc not in rest_map:
                rest_map[pc] = f
            if rest_fallback is None:
                rest_fallback = f

    return {
        "ctrl": ctrl_map,
        "ctrl_by_target": ctrl_by_target,
        "ctrl_fallback": ctrl_fallback,
        "rest": rest_map,
        "rest_fallback": rest_fallback,
    }


def load_bad_channels(session_loc: Optional[Path] = None) -> Dict[str, set]:
    base_loc = session_loc or (SESSION_LOC if "SESSION_LOC" in globals() else None)
    out: Dict[str, set] = {"NPRW": set(), "UA": set()}
    if base_loc is None:
        return out

    # Intan impedance check
    intan_imp_files = list(base_loc.glob("**/Intan/**/*impedance*.csv")) + list(base_loc.glob("**/*impedance*intan*.csv"))
    for f in intan_imp_files:
        try:
            df = pd.read_csv(f)
            ch_col = next((c for c in df.columns if "channel" in c.lower()), None)
            val_col = next((c for c in df.columns if any(k in c.lower() for k in ("impedance", "magnitude", "value", "kohm", "ohm"))), None)
            if ch_col and val_col:
                for _, row in df.iterrows():
                    val = pd.to_numeric(row[val_col], errors="coerce")
                    if pd.notna(val) and (val > 2e6 or val < 1e3):
                        ch_str = str(row[ch_col])
                        digits = re.findall(r"\d+", ch_str)
                        if digits:
                            out["NPRW"].add(int(digits[-1]))
        except Exception:
            pass

    # Blackrock impedance check
    br_imp_files = list(base_loc.glob("**/Blackrock/**/*impedance*.txt")) + list(base_loc.glob("**/*impedance*.txt"))
    for f in br_imp_files:
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    m = re.search(r"chan(?:nel)?\s*(\d+)[:\s]+([\d\.]+)", line, re.IGNORECASE)
                    if m:
                        ch = int(m.group(1))
                        imp = float(m.group(2))
                        if imp > 2000 or imp < 10:
                            out["UA"].add(ch)
        except Exception:
            pass

    return out


@lru_cache(maxsize=4)
def get_electrode_mapping_csv(monkey: Optional[str] = None) -> Path:
    if monkey is None:
        monkey = PARAMS.monkey if PARAMS is not None else "Nike"
    repo_cfg = (REPO_ROOT / "config" / f"electrode_port_mapping_{monkey}.csv") if "REPO_ROOT" in globals() else None
    if repo_cfg and repo_cfg.exists():
        return repo_cfg
    local_cfg = Path(__file__).resolve().parent.parent / "config" / f"electrode_port_mapping_{monkey}.csv"
    if local_cfg.exists():
        return local_cfg
    raise FileNotFoundError(f"Electrode mapping CSV not found for monkey {monkey}")


@lru_cache(maxsize=4)
def load_electrode_mapping(monkey: Optional[str] = None) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    Load electrode mapping from CSV file.

    Returns:
        tuple: (elec_info, region_grids)
            - elec_info: dict mapping electrode_id -> {nsp_id, port, region, row, col}
            - region_grids: dict mapping region_name -> 2D array of electrode IDs (8x8)
    """
    try:
        csv_path = get_electrode_mapping_csv(monkey)
        df = pd.read_csv(csv_path)
        elec_info: Dict[int, Dict[str, Any]] = {}
        region_grids: Dict[str, np.ndarray] = {}
        for _, row in df.iterrows():
            elec_id = int(row["ElectrodeID"])
            nsp_id = int(row["NSP_ID"]) if "NSP_ID" in df.columns and pd.notna(row.get("NSP_ID")) else elec_id
            port_str = str(row["Port"]) if "Port" in df.columns else "A"
            port = "A" if "A" in port_str.upper() else "B"
            region = str(row["Array"]) if "Array" in df.columns else "Unknown"
            grid_row = int(row["GridRow"]) if "GridRow" in df.columns and pd.notna(row.get("GridRow")) else 0
            grid_col = int(row["GridCol"]) if "GridCol" in df.columns and pd.notna(row.get("GridCol")) else 0

            elec_info[elec_id] = {
                "nsp_id": nsp_id,
                "port": port,
                "region": region,
                "row": grid_row,
                "col": grid_col,
            }
            if region not in region_grids:
                region_grids[region] = np.zeros((8, 8), dtype=int)
            if 0 <= grid_row < 8 and 0 <= grid_col < 8:
                region_grids[region][grid_row, grid_col] = elec_id

        return elec_info, region_grids
    except Exception as e:
        _log(f"Warning: could not load electrode mapping CSV: {e}")
        return {}, {}


def extract_array_channel_indices(
    ua_ids: Optional[np.ndarray],
    elec_info: Dict[int, Dict[str, Any]],
    recording_port: str,
    target_region: str,
    ua_region: Optional[np.ndarray] = None,
    ua_region_names: Optional[np.ndarray] = None,
    max_channels: Optional[int] = None,
    bad_channels: Optional[set] = None,
) -> Tuple[List[int], List[int]]:
    if ua_ids is None:
        return [], []
    ua_ids_flat = np.asarray(ua_ids).ravel()
    n_ch = len(ua_ids_flat)
    if max_channels is not None:
        n_ch = min(n_ch, max_channels)

    bad = bad_channels or set()
    matched_indices = []
    matched_elecs = []

    # Strategy 1: Check embedded region annotations in peristim NPZ
    if ua_region is not None and ua_region_names is not None:
        try:
            reg_names_list = [str(x).strip().lower() for x in np.asarray(ua_region_names).ravel()]
            target_lc = target_region.strip().lower()
            target_code = next((idx for idx, name in enumerate(reg_names_list) if name == target_lc), None)
            if target_code is not None:
                codes_flat = np.asarray(ua_region).ravel()[:n_ch]
                for i, code in enumerate(codes_flat):
                    if code == target_code:
                        elec = int(ua_ids_flat[i])
                        if elec not in bad:
                            matched_indices.append(i)
                            matched_elecs.append(elec)
                if matched_indices:
                    return matched_indices, matched_elecs
        except Exception:
            pass

    # Strategy 2: Check probe electrode mapping table
    norm_port = (recording_port or "A").strip().upper()
    target_norm = target_region.strip().lower()

    for i in range(n_ch):
        elec = int(ua_ids_flat[i])
        if elec in bad:
            continue
        info = elec_info.get(elec, {})
        elec_port = str(info.get("port", "")).strip().upper()
        elec_reg = str(info.get("region", "")).strip().lower()

        if elec_port and elec_port != norm_port:
            continue
        if elec_reg == target_norm:
            matched_indices.append(i)
            matched_elecs.append(elec)

    return matched_indices, matched_elecs


def prepare_counts_3d(counts_raw: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Ensure neural count array is shaped (n_channels, n_trials, n_bins)."""
    if counts_raw is None:
        return None
    arr = np.asarray(counts_raw, dtype=float)
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim != 3:
        return None
    n0, n1, n2 = arr.shape
    if n1 in (32, 64, 96, 128, 256) and n0 not in (32, 64, 96, 128, 256):
        return np.transpose(arr, (1, 0, 2))
    elif n0 in (32, 64, 96, 128, 256) and n1 not in (32, 64, 96, 128, 256):
        return arr
    if n1 in (32, 64, 96, 128, 256):
        return np.transpose(arr, (1, 0, 2))
    return arr


def nan_gaussian_filter1d(arr_3d: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Apply 1D Gaussian filter along axis=2, ignoring NaNs to prevent artifact bleed."""
    if sigma_bins <= 0:
        return arr_3d

    val = arr_3d.copy()
    nan_mask = np.isnan(val)
    val[nan_mask] = 0.0
    weights = (~nan_mask).astype(float)

    smoothed_val = gaussian_filter1d(val, sigma=sigma_bins, axis=2, mode="nearest")
    smoothed_w = gaussian_filter1d(weights, sigma=sigma_bins, axis=2, mode="nearest")

    with np.errstate(divide="ignore", invalid="ignore"):
        out = smoothed_val / smoothed_w
        out[smoothed_w < 1e-5] = np.nan
    out[nan_mask] = np.nan
    return out


def rebin_and_smooth(
    counts_3d: np.ndarray,
    centers_ms: np.ndarray,
    orig_bin_w: Optional[float] = None,
    target_bin_w: float = 20.0,
    smooth_sigma_ms: float = 20.0,
    win_ms: Tuple[float, float] = WIN_PLOT_MS,
    blank_win: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rebin spike counts to target bin width, convert to firing rate (Hz), and smooth."""
    counts = np.asarray(counts_3d, dtype=float)
    centers = np.asarray(centers_ms, dtype=float)

    if orig_bin_w is None:
        if len(centers) > 1:
            orig_bin_w = float(np.median(np.diff(centers)))
        else:
            orig_bin_w = target_bin_w

    t_min, t_max = win_ms
    start_edge = np.floor(t_min / target_bin_w) * target_bin_w
    end_edge = np.ceil(t_max / target_bin_w) * target_bin_w
    out_edges = np.arange(start_edge, end_edge + target_bin_w * 0.5, target_bin_w)
    out_centers = 0.5 * (out_edges[:-1] + out_edges[1:])

    n_ch, n_trials, n_src = counts.shape
    n_out = len(out_centers)

    src_edges = np.concatenate([[centers[0] - 0.5 * orig_bin_w], centers + 0.5 * orig_bin_w])
    bin_idx = np.digitize(centers, out_edges) - 1

    rebinned = np.zeros((n_ch, n_trials, n_out), dtype=float)
    bin_counts_src = np.zeros(n_out, dtype=float)

    for b in range(n_out):
        mask = bin_idx == b
        if np.any(mask):
            rebinned[:, :, b] = np.sum(counts[:, :, mask], axis=2)
            bin_counts_src[b] = np.sum(mask)

    rate_scale = 1000.0 / target_bin_w
    rates_hz = rebinned * rate_scale

    if blank_win is not None:
        b0, b1 = blank_win
        blank_mask = (out_centers >= b0) & (out_centers <= b1)
        rates_hz[:, :, blank_mask] = np.nan

    sigma_bins = smooth_sigma_ms / target_bin_w
    rates_smoothed = nan_gaussian_filter1d(rates_hz, sigma_bins)

    win_mask = (out_centers >= t_min) & (out_centers <= t_max)
    return rates_smoothed[:, :, win_mask], out_centers[win_mask]


def compute_benjamini_hochberg_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction returning q-values in [0, 1]."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])
    valid_mask = np.isfinite(p)
    q = np.ones(n, dtype=float)
    if not np.any(valid_mask):
        return q

    p_valid = p[valid_mask]
    n_valid = len(p_valid)
    sort_idx = np.argsort(p_valid)
    sorted_p = p_valid[sort_idx]

    ranks = np.arange(1, n_valid + 1, dtype=float)
    sorted_q = sorted_p * float(n_valid) / ranks

    for i in range(n_valid - 2, -1, -1):
        if sorted_q[i] > sorted_q[i + 1]:
            sorted_q[i] = sorted_q[i + 1]

    sorted_q = np.clip(sorted_q, 0.0, 1.0)
    rev_idx = np.empty(n_valid, dtype=int)
    rev_idx[sort_idx] = np.arange(n_valid)

    q[valid_mask] = sorted_q[rev_idx]
    return q


# =============================================================================
# Behavioral Target Computation
# =============================================================================

def compute_behavioral_perturbation_scores(
    stim_cam0_segs: Optional[np.ndarray],
    stim_cam0_names: Any,
    stim_cam1_segs: Optional[np.ndarray],
    stim_cam1_names: Any,
    ctrl_cam0_segs: Optional[np.ndarray],
    ctrl_cam0_names: Any,
    ctrl_cam1_segs: Optional[np.ndarray],
    ctrl_cam1_names: Any,
    stim_beh_t: np.ndarray,
    ctrl_beh_t: np.ndarray,
    behavior_window_ms: Tuple[float, float] = DEFAULT_BEHAVIOR_WINDOW_MS,
    preferred_keypoints: Tuple[str, ...] = ("middle", "wrist", "hand", "index"),
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Computes per-trial behavioral perturbation magnitude (Euclidean distance to control-mean trajectory)
    concatenating cam0 and cam1 into a unified trajectory.

    Returns
    -------
    y_stim : (n_stim_trials,) scalar perturbation score per STIM trial
    y_ctrl : (n_ctrl_trials,) scalar deviation score per CTRL trial
    meta : dict with trajectory curves, coordinates, and diagnostic data
    """
    # 1. Extract XY coordinates from both cameras using kinematics_utils
    s_xy0 = rcp.extract_xy_coordinates(stim_cam0_segs, stim_cam0_names, preferred_keypoints)
    s_xy1 = rcp.extract_xy_coordinates(stim_cam1_segs, stim_cam1_names, preferred_keypoints)
    c_xy0 = rcp.extract_xy_coordinates(ctrl_cam0_segs, ctrl_cam0_names, preferred_keypoints)
    c_xy1 = rcp.extract_xy_coordinates(ctrl_cam1_segs, ctrl_cam1_names, preferred_keypoints)

    if c_xy0 is None and c_xy1 is None:
        _log("Missing control camera kinematics; cannot construct control-mean trajectory.")
        return None, None, {}

    if s_xy0 is None and s_xy1 is None:
        _log("Missing stim camera kinematics; cannot construct stim trajectory.")
        return None, None, {}

    # 2. Build multi-camera coordinate arrays: (n_trials, n_coords, n_timepoints)
    s_coord_list = []
    c_coord_list = []
    coord_names = []

    if s_xy0 is not None and c_xy0 is not None:
        s_coord_list.extend([s_xy0[0], s_xy0[1]])
        c_coord_list.extend([c_xy0[0], c_xy0[1]])
        coord_names.extend(["cam0_x", "cam0_y"])

    if s_xy1 is not None and c_xy1 is not None:
        s_coord_list.extend([s_xy1[0], s_xy1[1]])
        c_coord_list.extend([c_xy1[0], c_xy1[1]])
        coord_names.extend(["cam1_x", "cam1_y"])

    if not s_coord_list or not c_coord_list:
        _log("No overlapping camera coordinates available between STIM and CTRL.")
        return None, None, {}

    # Stack along coordinate axis (axis=1) -> shape: (trials, coords, time)
    s_coords = np.stack(s_coord_list, axis=1)
    c_coords = np.stack(c_coord_list, axis=1)

    t_beh = np.asarray(stim_beh_t, dtype=float)
    c_t = np.asarray(ctrl_beh_t, dtype=float)

    # 3. Control-mean trajectory across control trials: shape (n_coords, n_timepoints)
    ctrl_mean_traj = np.nanmean(c_coords, axis=0)

    # 4. Euclidean distance to control-mean trajectory per trial across time
    # Stim trials distance
    diff_s = s_coords - ctrl_mean_traj[None, :, :]
    all_nan_s = np.all(np.isnan(diff_s), axis=1)
    dist_s = np.sqrt(np.nansum(diff_s ** 2, axis=1))  # (n_stim_trials, n_timepoints)
    dist_s[all_nan_s] = np.nan

    # Control trials distance (deviation of each control trial from group control mean)
    diff_c = c_coords - ctrl_mean_traj[None, :, :]
    all_nan_c = np.all(np.isnan(diff_c), axis=1)
    dist_c = np.sqrt(np.nansum(diff_c ** 2, axis=1))  # (n_ctrl_trials, n_timepoints)
    dist_c[all_nan_c] = np.nan

    # 5. Scalar summary score: mean Euclidean distance in behavior window
    t0, t1 = behavior_window_ms
    win_mask = (t_beh >= t0) & (t_beh <= t1)
    if not np.any(win_mask):
        _log(f"Behavior window [{t0}, {t1}] ms out of range ({t_beh[0]:.1f} to {t_beh[-1]:.1f} ms); using all post-0 ms.")
        win_mask = (t_beh >= 0.0) & (t_beh <= 400.0)

    y_stim = np.nanmean(dist_s[:, win_mask], axis=1)
    y_ctrl = np.nanmean(dist_c[:, win_mask], axis=1)

    meta = {
        "coord_names": coord_names,
        "n_coords": len(coord_names),
        "dist_s_traces": dist_s,
        "dist_c_traces": dist_c,
        "ctrl_mean_traj": ctrl_mean_traj,
        "beh_t": t_beh,
        "win_mask": win_mask,
        "behavior_window_ms": behavior_window_ms,
    }

    return y_stim, y_ctrl, meta


# =============================================================================
# Data Preparation Function
# =============================================================================

def prepare_regression_data(
    session_data: Dict[str, Any],
    neural_window_ms: Tuple[float, float] = DEFAULT_NEURAL_WINDOW_MS,
    behavior_window_ms: Tuple[float, float] = DEFAULT_BEHAVIOR_WINDOW_MS,
    preferred_keypoints: Tuple[str, ...] = ("middle", "wrist", "hand", "index"),
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Prepares trial-level neural features and behavioral targets by stacking STIM and CTRL trials together.
    Do NOT subtract STIM minus CTRL; preserves single-trial observations.

    Parameters
    ----------
    session_data : dict
        Contains neural rates ('stim_rates', 'ctrl_rates', 'time_ms'),
        behavioral segments ('stim_cam0_segs', 'ctrl_cam0_segs', etc.),
        and channel identification ('channel_ids').
    neural_window_ms : tuple of float
        Time window (ms) to compute mean firing rate per channel.
    behavior_window_ms : tuple of float
        Time window (ms) to compute mean trajectory deviation from control mean.

    Returns
    -------
    X_neural : (n_trials, n_channels)
        Mean firing rate per channel in neural window across all trials.
    y_behavior : (n_trials,)
        Per-trial mean Euclidean distance from control-mean trajectory.
    condition_labels : (n_trials,)
        'stim' or 'control' condition labels.
    trial_ids : (n_trials,)
        Trial index tracking array.
    info : dict
        Metadata dictionary.
    """
    s_rates = session_data.get("stim_rates")  # (n_ch, n_stim_trials, n_bins)
    c_rates = session_data.get("ctrl_rates")  # (n_ch, n_ctrl_trials, n_bins)
    time_ms = session_data.get("time_ms")
    channel_ids = session_data.get("channel_ids", [])

    if s_rates is None or c_rates is None or time_ms is None:
        _log("Missing neural rate matrices for regression data preparation.")
        return None, None, None, None, {}

    n_ch, n_s_trials, _ = s_rates.shape
    _, n_c_trials, _ = c_rates.shape

    # 1. Behavioral Target Computation
    y_stim, y_ctrl, beh_meta = compute_behavioral_perturbation_scores(
        stim_cam0_segs=session_data.get("stim_cam0_segs"),
        stim_cam0_names=session_data.get("stim_cam0_names"),
        stim_cam1_segs=session_data.get("stim_cam1_segs"),
        stim_cam1_names=session_data.get("stim_cam1_names"),
        ctrl_cam0_segs=session_data.get("ctrl_cam0_segs"),
        ctrl_cam0_names=session_data.get("ctrl_cam0_names"),
        ctrl_cam1_segs=session_data.get("ctrl_cam1_segs"),
        ctrl_cam1_names=session_data.get("ctrl_cam1_names"),
        stim_beh_t=session_data.get("stim_beh_t"),
        ctrl_beh_t=session_data.get("ctrl_beh_t"),
        behavior_window_ms=behavior_window_ms,
        preferred_keypoints=preferred_keypoints,
    )

    if y_stim is None or y_ctrl is None:
        _log("Failed to compute behavioral perturbation targets.")
        return None, None, None, None, {}

    # Ensure trial counts align with behavioral perturbation vectors
    n_s = min(n_s_trials, len(y_stim))
    n_c = min(n_c_trials, len(y_ctrl))
    if n_s == 0 or n_c == 0:
        _log(f"Zero trials available after alignment (stim={n_s}, ctrl={n_c}).")
        return None, None, None, None, {}

    s_rates = s_rates[:, :n_s, :]
    c_rates = c_rates[:, :n_c, :]
    y_stim = y_stim[:n_s]
    y_ctrl = y_ctrl[:n_c]

    # 2. Neural Feature Extraction: Mean firing rate per channel in neural window
    n0, n1 = neural_window_ms
    n_mask = (time_ms >= n0) & (time_ms <= n1)
    if not np.any(n_mask):
        _log(f"Neural window [{n0}, {n1}] ms out of range ({time_ms[0]:.1f} to {time_ms[-1]:.1f} ms); fallback to [50, 150] ms.")
        n_mask = (time_ms >= 50.0) & (time_ms <= 150.0)

    # X_stim shape: (n_s, n_channels)
    X_stim = np.nanmean(s_rates[:, :, n_mask], axis=2).T
    # X_ctrl shape: (n_c, n_channels)
    X_ctrl = np.nanmean(c_rates[:, :, n_mask], axis=2).T

    # 3. Stack both STIM and CTRL trials together
    X_all = np.vstack([X_stim, X_ctrl])
    y_all = np.concatenate([y_stim, y_ctrl])

    stim_tids = session_data.get("stim_trial_ids", np.arange(n_s))[:n_s]
    ctrl_tids = session_data.get("ctrl_trial_ids", np.arange(n_c))[:n_c]
    trial_ids_all = np.concatenate([stim_tids, ctrl_tids])

    cond_labels_all = np.array(["stim"] * n_s + ["control"] * n_c)

    # Filter out channels that are >50% NaN across trials (e.g. inactive / blanked)
    ch_finite_frac = np.mean(np.isfinite(X_all), axis=0)
    valid_ch_mask = ch_finite_frac > 0.5
    if not np.any(valid_ch_mask):
        _log("No channels with sufficient finite firing rate data in neural window.")
        return None, None, None, None, {}

    X_all = X_all[:, valid_ch_mask]
    s_rates = s_rates[valid_ch_mask, :, :]
    c_rates = c_rates[valid_ch_mask, :, :]
    channel_ids = [ch for ch, keep in zip(channel_ids, valid_ch_mask) if keep]

    # Impute remaining sporadic NaNs with channel column mean
    col_means = np.nanmean(X_all, axis=0)
    for c in range(X_all.shape[1]):
        nan_c = np.isnan(X_all[:, c])
        if np.any(nan_c):
            X_all[nan_c, c] = col_means[c] if np.isfinite(col_means[c]) else 0.0

    # 4. Clean NaN trials
    valid_trials = np.isfinite(y_all) & np.all(np.isfinite(X_all), axis=1)
    n_valid = int(np.sum(valid_trials))
    if n_valid < 4:
        _log(f"Insufficient valid trials ({n_valid}) for regression.")
        return None, None, None, None, {}

    X_neural = X_all[valid_trials]
    y_behavior = y_all[valid_trials]
    condition_labels = cond_labels_all[valid_trials]
    trial_ids = trial_ids_all[valid_trials]

    info = {
        "neural_window_ms": neural_window_ms,
        "behavior_window_ms": behavior_window_ms,
        "channel_ids": channel_ids,
        "s_rates": s_rates,
        "c_rates": c_rates,
        "n_stim_valid": int(np.sum(condition_labels == "stim")),
        "n_ctrl_valid": int(np.sum(condition_labels == "control")),
        "beh_meta": beh_meta,
    }

    return X_neural, y_behavior, condition_labels, trial_ids, info


# =============================================================================
# Ridge Regression with Cross-Validation
# =============================================================================

def fit_channel_regression(
    X_neural: np.ndarray,
    y_behavior: np.ndarray,
    cv_folds: int = DEFAULT_CV_FOLDS,
    alphas: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, StandardScaler]:
    """
    Fit Ridge regression predicting behavior from neural activity.

    Returns
    -------
    channel_weights : (n_channels,) - regression coefficients per channel
    cv_r2_scores : (cv_folds,) cross-validated R² scores
    best_alpha : selected regularization strength
    y_pred_cv : (n_trials,) cross-validated predictions for every trial
    scaler : fitted StandardScaler
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_neural)

    # Replace any numerical 0-variance NaNs with 0.0
    if np.any(np.isnan(X_scaled)):
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    n_samples = len(y_behavior)
    n_splits = min(cv_folds, max(2, n_samples // 2)) if n_samples >= 4 else 2

    model = RidgeCV(alphas=alphas, cv=n_splits)
    model.fit(X_scaled, y_behavior)

    cv_scores = cross_val_score(model, X_scaled, y_behavior, cv=n_splits)
    best_alpha = float(model.alpha_)

    # Out-of-fold cross-validated prediction for every trial
    y_pred_cv = cross_val_predict(Ridge(alpha=best_alpha), X_scaled, y_behavior, cv=n_splits)

    return model.coef_, cv_scores, best_alpha, y_pred_cv, scaler


# =============================================================================
# Vectorized Permutation Testing for Significance
# =============================================================================

def permutation_test_weights(
    X_neural: np.ndarray,
    y_behavior: np.ndarray,
    best_alpha: Optional[float] = None,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle behavior labels to build null distribution for each channel weight.
    Uses precomputed Ridge projection matrix for instantaneous vectorized permutations.

    Returns
    -------
    p_values : (n_channels,) empirical two-sided permutation p-values
    q_values : (n_channels,) FDR Benjamini-Hochberg adjusted q-values
    null_weights : (n_channels, n_permutations) null coefficient matrix
    """
    n_trials, n_channels = X_neural.shape

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_neural)
    if np.any(np.isnan(X_scaled)):
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    if best_alpha is None:
        n_splits = min(5, max(2, n_trials // 2))
        rcv = RidgeCV(alphas=DEFAULT_ALPHAS, cv=n_splits)
        rcv.fit(X_scaled, y_behavior)
        best_alpha = float(rcv.alpha_)

    # Centered target
    y_centered = y_behavior - np.mean(y_behavior)

    # Ridge closed-form projection operator: K = (X^T X + α I)^(-1) X^T
    XtX_alpha = (X_scaled.T @ X_scaled) + best_alpha * np.eye(n_channels)
    try:
        K = np.linalg.solve(XtX_alpha, X_scaled.T)  # shape: (n_channels, n_trials)
    except np.linalg.LinAlgError:
        K = np.linalg.pinv(XtX_alpha) @ X_scaled.T

    # True weights on centered data
    w_true = K @ y_centered  # (n_channels,)

    # Vectorized permutations: generate shuffled target matrix (n_trials, n_permutations)
    rng = np.random.default_rng(rng_seed)
    perm_indices = np.column_stack([rng.permutation(n_trials) for _ in range(n_permutations)])
    Y_perm = y_centered[perm_indices]  # (n_trials, n_permutations)

    # Compute null weights via single matrix multiplication: shape (n_channels, n_permutations)
    null_weights = K @ Y_perm

    # Empirical two-sided p-values: |null| >= |true|
    abs_true = np.abs(w_true)[:, None]
    abs_null = np.abs(null_weights)
    exceedances = np.sum(abs_null >= abs_true, axis=1)
    p_values = (1.0 + exceedances.astype(float)) / (n_permutations + 1.0)

    # FDR correction across channels
    q_values = compute_benjamini_hochberg_fdr(p_values)

    return p_values, q_values, null_weights


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_channel_weights(
    channel_weights: np.ndarray,
    channel_ids: Sequence[Any],
    p_values: np.ndarray,
    q_values: Optional[np.ndarray] = None,
    cv_r2: Optional[float] = None,
    best_alpha: Optional[float] = None,
    array_name: str = "",
    condition_name: str = "",
    out_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (14.0, 5.5),
) -> plt.Figure:
    """
    Bar plot of regression weights across channels with statistical significance markers.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    ch_ids = np.asarray(channel_ids)
    n_ch = len(channel_weights)
    x_pos = np.arange(n_ch)

    # Distinct palettes: warm coral for positive weights, cool steel blue for negative weights
    colors = []
    alphas = []
    for w, p in zip(channel_weights, p_values):
        is_sig = p < P_VAL_THRESH
        if w >= 0:
            colors.append("#d9383a")  # Crimson / Coral
        else:
            colors.append("#2b7bba")  # Steel Blue
        alphas.append(0.95 if is_sig else 0.35)

    bars = ax.bar(x_pos, channel_weights, color=colors, alpha=0.85, edgecolor="none", width=0.8)

    # Adjust individual bar alpha
    for b, a in zip(bars, alphas):
        b.set_alpha(a)

    ax.axhline(0.0, color="#333333", linestyle="-", linewidth=1.0, alpha=0.7)

    # Add significance markers above/below bars
    y_max = np.nanmax(np.abs(channel_weights)) if n_ch > 0 else 1.0
    offset = max(y_max * 0.04, 0.005)

    n_sig_p = int(np.sum(p_values < P_VAL_THRESH))
    n_sig_q = int(np.sum(q_values < FDR_ALPHA)) if q_values is not None else 0

    for i, (w, p) in enumerate(zip(channel_weights, p_values)):
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            star = ""

        if star:
            y_annot = w + offset if w >= 0 else w - offset * 2.2
            va = "bottom" if w >= 0 else "top"
            ax.text(x_pos[i], y_annot, star, ha="center", va=va, fontsize=10, fontweight="bold", color="#111111")

    # Channel ID ticks
    step = max(1, n_ch // 32)
    ax.set_xticks(x_pos[::step])
    ax.set_xticklabels([str(ch_ids[i]) for i in range(0, n_ch, step)], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Electrode / Channel ID", fontsize=11, fontweight="medium")
    ax.set_ylabel("Standardized Weight (β)", fontsize=11, fontweight="medium")

    # Informative title banner
    r2_str = f" | Cross-Val R² = {cv_r2:.3f}" if cv_r2 is not None else ""
    alpha_str = f" | α = {best_alpha:.2g}" if best_alpha is not None else ""
    ax.set_title(
        f"{array_name} Channel Weights → Behavioral Perturbation\n"
        f"Significant channels: {n_sig_p}/{n_ch} (p < 0.05), {n_sig_q}/{n_ch} (FDR q < 0.05){r2_str}{alpha_str}",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    # Legend
    legend_elements = [
        Patch(facecolor="#d9383a", alpha=0.95, label="Positive Weight (p < 0.05)"),
        Patch(facecolor="#2b7bba", alpha=0.95, label="Negative Weight (p < 0.05)"),
        Patch(facecolor="#888888", alpha=0.35, label="Non-Significant (p ≥ 0.05)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=9)

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()

    if out_path:
        ensure_dir(out_path.parent)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")

    return fig


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    condition_labels: Sequence[str],
    cv_r2: Optional[float] = None,
    array_name: str = "",
    condition_name: str = "",
    out_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (7.5, 6.5),
) -> plt.Figure:
    """
    Scatter of predicted vs actual behavioral perturbation magnitude, colored by STIM/CTRL condition.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    cond_arr = np.asarray(condition_labels)
    is_stim = cond_arr == "stim"
    is_ctrl = cond_arr == "control"

    # Scatter points
    ax.scatter(
        y_true[is_ctrl], y_pred[is_ctrl],
        c="#2c3e50", s=45, alpha=0.80, edgecolors="#1a252f", linewidths=0.6,
        label=f"Control ({int(np.sum(is_ctrl))} trials)", zorder=3,
    )
    ax.scatter(
        y_true[is_stim], y_pred[is_stim],
        c="#e74c3c", s=50, alpha=0.85, edgecolors="#901a1c", linewidths=0.6,
        label=f"Stim ({int(np.sum(is_stim))} trials)", zorder=4,
    )

    # 45-degree Unity Reference Line
    min_val = min(float(np.nanmin(y_true)), float(np.nanmin(y_pred)))
    max_val = max(float(np.nanmax(y_true)), float(np.nanmax(y_pred)))
    pad = (max_val - min_val) * 0.08
    lims = (min_val - pad, max_val + pad)
    ax.plot(lims, lims, color="#7f8c8d", linestyle="--", linewidth=1.2, label="Identity (y = x)", zorder=1)

    # Empirical Regression Trendline
    if len(y_true) > 2 and np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred)):
        slope, intercept, r_val, p_val, _ = stats.linregress(y_true, y_pred)
        x_trend = np.linspace(min_val, max_val, 100)
        ax.plot(x_trend, slope * x_trend + intercept, color="#2980b9", linestyle="-", linewidth=1.8, label=f"Fit (r = {r_val:.2f})", zorder=2)
    else:
        r_val, p_val = np.nan, np.nan

    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Observed Perturbation Magnitude (mm)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Cross-Validated Predicted Perturbation (mm)", fontsize=11, fontweight="medium")

    # Stat annotation box
    r2_val = cv_r2 if cv_r2 is not None else (r_val ** 2 if np.isfinite(r_val) else np.nan)
    stat_text = (
        f"Cross-Val R² = {r2_val:.3f}\n"
        f"Pearson r = {r_val:.3f} (p = {p_val:.2e})\n"
        f"Total Trials = {len(y_true)}"
    )
    ax.text(
        0.05, 0.95, stat_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#bdc3c7", alpha=0.9),
    )

    ax.set_title(
        f"{array_name} Perturbation Prediction Performance\n{condition_name}",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()

    if out_path:
        ensure_dir(out_path.parent)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")

    return fig


def plot_top_channels_traces(
    s_rates: np.ndarray,
    c_rates: np.ndarray,
    time_ms: np.ndarray,
    channel_ids: Sequence[Any],
    channel_weights: np.ndarray,
    p_values: np.ndarray,
    neural_window_ms: Tuple[float, float] = DEFAULT_NEURAL_WINDOW_MS,
    top_n: int = 3,
    array_name: str = "",
    condition_name: str = "",
    out_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Time-resolved trial-averaged firing rate traces (mean ± SEM) for top positive and negative channels.
    """
    n_ch = len(channel_weights)
    if n_ch == 0:
        return None

    # Pick top positive and top negative weight channels
    pos_order = np.argsort(channel_weights)[::-1]
    neg_order = np.argsort(channel_weights)

    selected_indices = []
    # Add top positive
    for idx in pos_order:
        if len(selected_indices) >= top_n or channel_weights[idx] <= 0:
            break
        selected_indices.append(idx)
    # Add top negative
    for idx in neg_order:
        if len(selected_indices) >= top_n * 2 or channel_weights[idx] >= 0:
            break
        if idx not in selected_indices:
            selected_indices.append(idx)

    # Fallback to top absolute weights if one-sided
    if len(selected_indices) < min(4, n_ch):
        abs_order = np.argsort(np.abs(channel_weights))[::-1]
        for idx in abs_order:
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= min(6, n_ch):
                break

    n_plots = len(selected_indices)
    if n_plots == 0:
        return None

    n_cols = min(3, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.8 * n_rows), dpi=300, sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for plot_idx, ch_idx in enumerate(selected_indices):
        ax = axes_flat[plot_idx]
        ch_id = channel_ids[ch_idx]
        weight = channel_weights[ch_idx]
        pval = p_values[ch_idx]

        # STIM trace: mean ± SEM
        s_trace = s_rates[ch_idx, :, :]  # (trials, bins)
        s_mean = np.nanmean(s_trace, axis=0)
        s_sem = np.nanstd(s_trace, axis=0) / math.sqrt(max(1, s_trace.shape[0]))

        # CTRL trace: mean ± SEM
        c_trace = c_rates[ch_idx, :, :]  # (trials, bins)
        c_mean = np.nanmean(c_trace, axis=0)
        c_sem = np.nanstd(c_trace, axis=0) / math.sqrt(max(1, c_trace.shape[0]))

        # Plot shaded SEM and lines
        ax.fill_between(time_ms, c_mean - c_sem, c_mean + c_sem, color="#2c3e50", alpha=0.18, edgecolor="none")
        ax.plot(time_ms, c_mean, color="#2c3e50", linewidth=1.5, label="Control")

        ax.fill_between(time_ms, s_mean - s_sem, s_mean + s_sem, color="#e74c3c", alpha=0.22, edgecolor="none")
        ax.plot(time_ms, s_mean, color="#e74c3c", linewidth=1.8, label="Stim")

        # Vertical line at t=0 ms (stim onset)
        ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0, alpha=0.7)

        # Shaded neural feature window
        ax.axvspan(
            neural_window_ms[0], neural_window_ms[1],
            color="#f39c12", alpha=0.15, label="Neural Window" if plot_idx == 0 else None,
        )

        sig_star = " ***" if pval < 0.001 else (" **" if pval < 0.01 else (" *" if pval < 0.05 else ""))
        ax.set_title(f"Channel {ch_id} (β = {weight:+.2f}, p = {pval:.3f}{sig_star})", fontsize=10, fontweight="bold")
        ax.set_ylabel("Firing Rate (Hz)", fontsize=9)
        ax.grid(linestyle="--", alpha=0.25)

        if plot_idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Time from Stim Onset (ms)", fontsize=9)
        if plot_idx == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # Hide unused subplots
    for p_idx in range(n_plots, len(axes_flat)):
        axes_flat[p_idx].axis("off")

    fig.suptitle(f"{array_name} Top Channels Firing Rate Profiles: {condition_name}", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()

    if out_path:
        ensure_dir(out_path.parent)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")

    return fig


def plot_regression_summary_figure(
    channel_weights: np.ndarray,
    channel_ids: Sequence[Any],
    p_values: np.ndarray,
    q_values: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    condition_labels: Sequence[str],
    s_rates: np.ndarray,
    c_rates: np.ndarray,
    time_ms: np.ndarray,
    cv_r2: float,
    best_alpha: float,
    array_name: str,
    condition_name: str,
    neural_window_ms: Tuple[float, float] = DEFAULT_NEURAL_WINDOW_MS,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Consolidated 3-panel publication figure:
    Panel A: Channel weights bar chart with significance stars.
    Panel B: Actual vs predicted perturbation scatter.
    Panel C: Top 2 channel traces across time.
    """
    fig = plt.figure(figsize=(15.0, 10.0), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.25)

    # Panel A: Channel Weights
    ax_weights = fig.add_subplot(gs[0, :])
    ch_ids = np.asarray(channel_ids)
    n_ch = len(channel_weights)
    x_pos = np.arange(n_ch)

    colors = ["#d9383a" if w >= 0 else "#2b7bba" for w in channel_weights]
    bars = ax_weights.bar(x_pos, channel_weights, color=colors, alpha=0.85, width=0.8)
    for b, p in zip(bars, p_values):
        b.set_alpha(0.95 if p < P_VAL_THRESH else 0.35)

    ax_weights.axhline(0.0, color="#333333", linestyle="-", linewidth=0.9, alpha=0.7)

    y_max = np.nanmax(np.abs(channel_weights)) if n_ch > 0 else 1.0
    offset = max(y_max * 0.04, 0.005)
    for i, (w, p) in enumerate(zip(channel_weights, p_values)):
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        if star:
            y_annot = w + offset if w >= 0 else w - offset * 2.2
            va = "bottom" if w >= 0 else "top"
            ax_weights.text(x_pos[i], y_annot, star, ha="center", va=va, fontsize=9, fontweight="bold")

    step = max(1, n_ch // 28)
    ax_weights.set_xticks(x_pos[::step])
    ax_weights.set_xticklabels([str(ch_ids[i]) for i in range(0, n_ch, step)], rotation=45, ha="right", fontsize=8)
    ax_weights.set_xlabel("Electrode / Channel ID", fontsize=10)
    ax_weights.set_ylabel("Standardized Weight (β)", fontsize=10)
    n_sig_p = int(np.sum(p_values < P_VAL_THRESH))
    n_sig_q = int(np.sum(q_values < FDR_ALPHA))
    ax_weights.set_title(
        f"A. Channel Regression Weights ({array_name}) | Sig: {n_sig_p}/{n_ch} (p < 0.05), {n_sig_q}/{n_ch} (FDR q < 0.05)",
        fontsize=11, fontweight="bold", loc="left",
    )
    ax_weights.grid(axis="y", linestyle="--", alpha=0.3)

    # Panel B: Prediction Scatter
    ax_scatter = fig.add_subplot(gs[1, 0])
    cond_arr = np.asarray(condition_labels)
    is_stim = cond_arr == "stim"
    is_ctrl = cond_arr == "control"
    ax_scatter.scatter(y_true[is_ctrl], y_pred[is_ctrl], c="#2c3e50", s=40, alpha=0.8, edgecolors="none", label=f"Control (N={int(np.sum(is_ctrl))})")
    ax_scatter.scatter(y_true[is_stim], y_pred[is_stim], c="#e74c3c", s=45, alpha=0.85, edgecolors="none", label=f"Stim (N={int(np.sum(is_stim))})")

    min_val = min(float(np.nanmin(y_true)), float(np.nanmin(y_pred)))
    max_val = max(float(np.nanmax(y_true)), float(np.nanmax(y_pred)))
    pad = (max_val - min_val) * 0.08
    lims = (min_val - pad, max_val + pad)
    ax_scatter.plot(lims, lims, color="#7f8c8d", linestyle="--", linewidth=1.1, label="Unity (y=x)")

    if len(y_true) > 2:
        slope, intercept, r_val, p_val, _ = stats.linregress(y_true, y_pred)
        x_trend = np.linspace(min_val, max_val, 100)
        ax_scatter.plot(x_trend, slope * x_trend + intercept, color="#2980b9", linestyle="-", linewidth=1.6, label=f"Fit (r={r_val:.2f})")
    else:
        r_val, p_val = np.nan, np.nan

    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_xlabel("Observed Perturbation Magnitude (mm)", fontsize=10)
    ax_scatter.set_ylabel("Cross-Validated Predicted (mm)", fontsize=10)
    ax_scatter.set_title(f"B. Prediction Quality | CV R² = {cv_r2:.3f}, r = {r_val:.2f} (p = {p_val:.2e})", fontsize=11, fontweight="bold", loc="left")
    ax_scatter.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax_scatter.grid(linestyle="--", alpha=0.3)

    # Panel C: Top Channels Traces
    ax_traces = fig.add_subplot(gs[1, 1])
    top_pos_idx = int(np.argmax(channel_weights))
    top_neg_idx = int(np.argmin(channel_weights))

    # Show top positive channel
    s_m = np.nanmean(s_rates[top_pos_idx, :, :], axis=0)
    c_m = np.nanmean(c_rates[top_pos_idx, :, :], axis=0)
    s_se = np.nanstd(s_rates[top_pos_idx, :, :], axis=0) / math.sqrt(max(1, s_rates.shape[1]))
    c_se = np.nanstd(c_rates[top_pos_idx, :, :], axis=0) / math.sqrt(max(1, c_rates.shape[1]))

    ax_traces.fill_between(time_ms, c_m - c_se, c_m + c_se, color="#2c3e50", alpha=0.18)
    ax_traces.plot(time_ms, c_m, color="#2c3e50", linewidth=1.5, label="Control")
    ax_traces.fill_between(time_ms, s_m - s_se, s_m + s_se, color="#e74c3c", alpha=0.22)
    ax_traces.plot(time_ms, s_m, color="#e74c3c", linewidth=1.8, label="Stim")

    ax_traces.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_traces.axvspan(neural_window_ms[0], neural_window_ms[1], color="#f39c12", alpha=0.15, label="Neural Window")
    ax_traces.set_xlabel("Time from Stim Onset (ms)", fontsize=10)
    ax_traces.set_ylabel("Firing Rate (Hz)", fontsize=10)
    ax_traces.set_title(
        f"C. Top Predictive Channel {ch_ids[top_pos_idx]} (β = {channel_weights[top_pos_idx]:+.2f}, p = {p_values[top_pos_idx]:.3f})",
        fontsize=11, fontweight="bold", loc="left",
    )
    ax_traces.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax_traces.grid(linestyle="--", alpha=0.3)

    fig.suptitle(f"Channel-Behavior Regression: {condition_name} ({array_name})", fontsize=13, fontweight="bold", y=0.98)

    if out_path:
        ensure_dir(out_path.parent)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")

    return fig


# =============================================================================
# Single Condition Analysis Pipeline
# =============================================================================

def process_single_condition(
    stim_path: Path,
    ctrl_path: Optional[Path],
    bad_channels: Dict[str, set],
    out_dir: Path,
    neural_window_ms: Tuple[float, float] = DEFAULT_NEURAL_WINDOW_MS,
    behavior_window_ms: Tuple[float, float] = DEFAULT_BEHAVIOR_WINDOW_MS,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    file_idx: Optional[int] = None,
    total_files: Optional[int] = None,
) -> Dict[str, Any]:
    prefix = f"[{file_idx}/{total_files}] " if file_idx and total_files else ""
    _log(f"{prefix}Processing condition: {stim_path.name}")
    ensure_dir(out_dir)

    condition_name = stim_path.stem
    target = get_target_folder(stim_path)

    # 1. Load data from NPZ files
    with np.load(stim_path, allow_pickle=True) as stim_npz:
        stim_dur_ms = get_stim_duration_ms(stim_npz)
        port = get_ua_port(stim_npz)
        stim_freq_hz = scalarize(safe_get_npz(stim_npz, "stim_freq_hz", None))
        stim_dur_nominal_ms = scalarize(safe_get_npz(stim_npz, "stim_dur_nominal_ms", None))
        pulse_count = compute_pulse_count(stim_freq_hz, stim_dur_nominal_ms)

        nprw_counts = safe_get_npz(stim_npz, "NPRW_counts", None)
        nprw_t = safe_get_npz(stim_npz, "NPRW_rel_t", None)
        ua_counts = safe_get_npz(stim_npz, "UA_counts", None)
        ua_t = safe_get_npz(stim_npz, "UA_rel_t", None)
        ua_ids = safe_get_npz(stim_npz, "ua_ids_1based", None)
        ua_region = safe_get_npz(stim_npz, "ua_region", None)
        ua_region_names = safe_get_npz(stim_npz, "ua_region_names", None)

        stim_beh_t = safe_get_npz(stim_npz, "beh_rel_t", None)
        stim_cam0_segs = safe_get_npz(stim_npz, "beh_cam0_segs", None)
        stim_cam1_segs = safe_get_npz(stim_npz, "beh_cam1_segs", None)
        stim_cam0_names = safe_get_npz(stim_npz, "beh_cam0_names", None)
        stim_cam1_names = safe_get_npz(stim_npz, "beh_cam1_names", None)
        stim_trial_ids = safe_get_npz(stim_npz, "raw_trial_indices", None)

    ctrl_nprw_counts = None
    ctrl_nprw_t = None
    ctrl_ua_counts = None
    ctrl_ua_t = None
    ctrl_ua_ids = None
    ctrl_ua_reg = None
    ctrl_ua_names = None
    ctrl_port = None
    ctrl_beh_t = None
    ctrl_cam0_segs = None
    ctrl_cam1_segs = None
    ctrl_cam0_names = None
    ctrl_cam1_names = None
    ctrl_trial_ids = None

    if ctrl_path and ctrl_path.exists():
        _log(f"  Matched control: {ctrl_path.name}")
        with np.load(ctrl_path, allow_pickle=True) as ctrl_npz:
            ctrl_nprw_counts = safe_get_npz(ctrl_npz, "NPRW_counts", None)
            ctrl_nprw_t = safe_get_npz(ctrl_npz, "NPRW_rel_t", None)
            ctrl_ua_counts = safe_get_npz(ctrl_npz, "UA_counts", None)
            ctrl_ua_t = safe_get_npz(ctrl_npz, "UA_rel_t", None)
            ctrl_ua_ids = safe_get_npz(ctrl_npz, "ua_ids_1based", None)
            ctrl_ua_reg = safe_get_npz(ctrl_npz, "ua_region", None)
            ctrl_ua_names = safe_get_npz(ctrl_npz, "ua_region_names", None)
            ctrl_port = get_ua_port(ctrl_npz)

            ctrl_beh_t = safe_get_npz(ctrl_npz, "beh_rel_t", None)
            ctrl_cam0_segs = safe_get_npz(ctrl_npz, "beh_cam0_segs", None)
            ctrl_cam1_segs = safe_get_npz(ctrl_npz, "beh_cam1_segs", None)
            ctrl_cam0_names = safe_get_npz(ctrl_npz, "beh_cam0_names", None)
            ctrl_cam1_names = safe_get_npz(ctrl_npz, "beh_cam1_names", None)
            ctrl_trial_ids = safe_get_npz(ctrl_npz, "raw_trial_indices", None)
    else:
        _log(f"  [Warning] No matched control file found for {stim_path.name} (target={target})")

    elec_info, _ = load_electrode_mapping(PARAMS.monkey if PARAMS is not None else None)
    bad_nprw = bad_channels.get("NPRW", set()) if bad_channels else set()
    bad_ua = bad_channels.get("UA", set()) if bad_channels else set()

    # 2. Extract arrays
    array_data_list = []

    # Cerebellum (NPRW)
    if nprw_counts is not None and nprw_t is not None and ctrl_nprw_counts is not None:
        s_nprw_3d = prepare_counts_3d(nprw_counts)
        c_nprw_3d = prepare_counts_3d(ctrl_nprw_counts)
        if s_nprw_3d is not None and c_nprw_3d is not None:
            n_tot = s_nprw_3d.shape[0]
            valid_idxs = [i for i in range(n_tot) if (i + 1) not in bad_nprw and i not in bad_nprw]
            if valid_idxs:
                ch_ids = [i + 1 for i in valid_idxs]
                array_data_list.append(("NPRW", s_nprw_3d[valid_idxs, :, :], nprw_t, c_nprw_3d[valid_idxs, :, :], ctrl_nprw_t or nprw_t, ch_ids))

    # Utah Array (Motor Cortex)
    if ua_counts is not None and ua_t is not None and ctrl_ua_counts is not None:
        s_ua_3d = prepare_counts_3d(ua_counts)
        c_ua_3d = prepare_counts_3d(ctrl_ua_counts)
        if s_ua_3d is not None and c_ua_3d is not None:
            all_ua_ch = s_ua_3d.shape[0]
            ua_ids_arr = ua_ids.ravel() if ua_ids is not None else np.arange(1, all_ua_ch + 1)

            if ANALYZE_COMBINED_UTAH:
                valid_ua_idxs = [i for i in range(all_ua_ch) if int(ua_ids_arr[i]) not in bad_ua]
                if valid_ua_idxs:
                    all_elecs = [int(ua_ids_arr[i]) for i in valid_ua_idxs]
                    array_data_list.append(("UA_ALL", s_ua_3d[valid_ua_idxs, :, :], ua_t, c_ua_3d[valid_ua_idxs, :, :], ctrl_ua_t or ua_t, all_elecs))

            if ANALYZE_INDIVIDUAL_UTAH_ARRAYS:
                for reg in REGION_ORDER:
                    s_idxs, s_elecs = extract_array_channel_indices(
                        ua_ids, elec_info, port, reg,
                        ua_region=ua_region, ua_region_names=ua_region_names,
                        max_channels=all_ua_ch, bad_channels=bad_ua,
                    )
                    c_idxs, _ = extract_array_channel_indices(
                        ctrl_ua_ids if ctrl_ua_ids is not None else ua_ids,
                        elec_info, ctrl_port or port, reg,
                        ua_region=ctrl_ua_reg if ctrl_ua_reg is not None else ua_region,
                        ua_region_names=ctrl_ua_names if ctrl_ua_names is not None else ua_region_names,
                        max_channels=c_ua_3d.shape[0], bad_channels=bad_ua,
                    )
                    if s_idxs and c_idxs and len(s_idxs) == len(c_idxs):
                        array_data_list.append((f"UA_{reg}", s_ua_3d[s_idxs, :, :], ua_t, c_ua_3d[c_idxs, :, :], ctrl_ua_t or ua_t, s_elecs))

    summary_rows = []
    channel_rows = []

    # 3. Analyze each array modality
    for arr_name, s_3d, s_t, c_3d, c_t, ch_ids in array_data_list:
        _log(f"  Fitting Channel-Behavior Regression for {arr_name} ({len(ch_ids)} channels)")

        # Blanking parameters
        is_stim_cond = stim_dur_ms > 0
        blank_pre = NPRW_BLANK_PRE_MS if arr_name == "NPRW" else (UA_BLANK_PRE_MS if BLANK_UA_STIM_PERIOD else 0.0)
        blank_post = NPRW_BLANK_POST_MS if arr_name == "NPRW" else (UA_BLANK_POST_MS if BLANK_UA_STIM_PERIOD else 0.0)
        has_blanking = is_stim_cond and (blank_pre > 0 or blank_post > 0)
        blank_win = (-blank_pre, stim_dur_ms + blank_post) if has_blanking else None

        bin_w = NPRW_BIN_WIDTH_MS if arr_name == "NPRW" else UA_BIN_WIDTH_MS
        sigma = NPRW_SMOOTHING_SIGMA_MS if arr_name == "NPRW" else UA_SMOOTHING_SIGMA_MS

        s_rates, time_ms = rebin_and_smooth(s_3d, s_t, target_bin_w=bin_w, smooth_sigma_ms=sigma, win_ms=WIN_PLOT_MS, blank_win=blank_win)
        c_rates, _ = rebin_and_smooth(c_3d, c_t, target_bin_w=bin_w, smooth_sigma_ms=sigma, win_ms=WIN_PLOT_MS, blank_win=None)

        session_dict = {
            "stim_rates": s_rates,
            "ctrl_rates": c_rates,
            "time_ms": time_ms,
            "channel_ids": ch_ids,
            "stim_cam0_segs": stim_cam0_segs,
            "stim_cam0_names": stim_cam0_names,
            "stim_cam1_segs": stim_cam1_segs,
            "stim_cam1_names": stim_cam1_names,
            "ctrl_cam0_segs": ctrl_cam0_segs,
            "ctrl_cam0_names": ctrl_cam0_names,
            "ctrl_cam1_segs": ctrl_cam1_segs,
            "ctrl_cam1_names": ctrl_cam1_names,
            "stim_beh_t": stim_beh_t,
            "ctrl_beh_t": ctrl_beh_t,
            "stim_trial_ids": stim_trial_ids,
            "ctrl_trial_ids": ctrl_trial_ids,
        }

        # Step 1: Prepare trial-level regression data
        X_neural, y_behavior, condition_labels, trial_ids, prep_info = prepare_regression_data(
            session_dict,
            neural_window_ms=neural_window_ms,
            behavior_window_ms=behavior_window_ms,
        )

        if X_neural is None or len(y_behavior) < 4:
            _log(f"  [Skip] Inadequate observations for {arr_name}")
            continue

        ch_ids = prep_info["channel_ids"]
        s_rates = prep_info["s_rates"]
        c_rates = prep_info["c_rates"]

        # Step 2: Fit Ridge Regression with Cross-Validation
        channel_weights, cv_scores, best_alpha, y_pred_cv, scaler = fit_channel_regression(X_neural, y_behavior)
        cv_r2 = float(np.mean(cv_scores))

        # Step 3: Permutation Testing for Significance
        p_values, q_values, null_weights = permutation_test_weights(
            X_neural, y_behavior, best_alpha=best_alpha, n_permutations=n_permutations
        )

        n_sig_p = int(np.sum(p_values < P_VAL_THRESH))
        n_sig_q = int(np.sum(q_values < FDR_ALPHA))

        _log(f"    -> {arr_name}: CV R² = {cv_r2:.3f}, best α = {best_alpha:.2g}, sig channels: {n_sig_p}/{len(ch_ids)} (p<0.05), {n_sig_q}/{len(ch_ids)} (q<0.05)")

        base_name = f"{stim_path.stem}_{arr_name}"

        # Step 4: Visualizations
        # 1. Channel Weights Bar Plot
        fig_w = plot_channel_weights(
            channel_weights, ch_ids, p_values, q_values,
            cv_r2=cv_r2, best_alpha=best_alpha,
            array_name=arr_name, condition_name=condition_name,
            out_path=out_dir / f"{base_name}_channel_weights.png",
        )
        plt.close(fig_w)

        # 2. Prediction Scatter Plot
        fig_s = plot_prediction_scatter(
            y_behavior, y_pred_cv, condition_labels,
            cv_r2=cv_r2, array_name=arr_name, condition_name=condition_name,
            out_path=out_dir / f"{base_name}_prediction_scatter.png",
        )
        plt.close(fig_s)

        # 3. Top Channels Firing Rate Traces
        fig_t = plot_top_channels_traces(
            s_rates, c_rates, time_ms, ch_ids, channel_weights, p_values,
            neural_window_ms=neural_window_ms,
            array_name=arr_name, condition_name=condition_name,
            out_path=out_dir / f"{base_name}_top_channels_traces.png",
        )
        if fig_t is not None:
            plt.close(fig_t)

        # 4. Consolidated 3-Panel Figure
        fig_all = plot_regression_summary_figure(
            channel_weights, ch_ids, p_values, q_values,
            y_behavior, y_pred_cv, condition_labels,
            s_rates, c_rates, time_ms,
            cv_r2=cv_r2, best_alpha=best_alpha,
            array_name=arr_name, condition_name=condition_name,
            neural_window_ms=neural_window_ms,
            out_path=out_dir / f"{base_name}_regression_summary.png",
        )
        plt.close(fig_all)

        # Save per-channel results CSV
        df_chan = pd.DataFrame({
            "condition_name": condition_name,
            "target": target,
            "array_name": arr_name,
            "channel_id": ch_ids,
            "weight": channel_weights,
            "p_value": p_values,
            "q_value": q_values,
            "is_sig_p05": p_values < P_VAL_THRESH,
            "is_sig_fdr": q_values < FDR_ALPHA,
        })
        chan_csv_path = out_dir / f"{base_name}_regression_channels.csv"
        df_chan.to_csv(chan_csv_path, index=False)

        for _, row in df_chan.iterrows():
            channel_rows.append(row.to_dict())

        # Array summary record
        top_pos_ch = ch_ids[int(np.argmax(channel_weights))]
        top_neg_ch = ch_ids[int(np.argmin(channel_weights))]
        summary_rows.append({
            "condition_name": condition_name,
            "target": target,
            "array_name": arr_name,
            "stim_freq_hz": stim_freq_hz,
            "stim_dur_nominal_ms": stim_dur_nominal_ms,
            "pulse_count": pulse_count,
            "n_trials_stim": prep_info["n_stim_valid"],
            "n_trials_control": prep_info["n_ctrl_valid"],
            "n_channels": len(ch_ids),
            "cv_r2": cv_r2,
            "best_alpha": best_alpha,
            "n_sig_channels_p05": n_sig_p,
            "n_sig_channels_fdr": n_sig_q,
            "top_positive_channel": top_pos_ch,
            "top_positive_weight": float(np.max(channel_weights)),
            "top_negative_channel": top_neg_ch,
            "top_negative_weight": float(np.min(channel_weights)),
            "neural_window_start_ms": neural_window_ms[0],
            "neural_window_end_ms": neural_window_ms[1],
            "behavior_window_start_ms": behavior_window_ms[0],
            "behavior_window_end_ms": behavior_window_ms[1],
        })

    return {
        "summary_rows": summary_rows,
        "channel_rows": channel_rows,
    }


# =============================================================================
# Main Analysis Loop
# =============================================================================

def parse_condition_specifier(item: Any) -> Tuple[Optional[int], str]:
    s = str(item).strip()
    m_num = re.search(r"^\d+$", s)
    if m_num:
        return int(m_num.group(0)), s
    m_br = re.search(r"(?:br|cond)[_-]?(\d+)", s, re.IGNORECASE)
    if m_br:
        return int(m_br.group(1)), s
    return None, s


def is_stim_file_matching(stim_f: Path, targets: List[Any]) -> bool:
    if not targets:
        return True
    file_name = stim_f.name
    file_path_str = str(stim_f)
    br_match = re.search(r"BR_(\d+)", file_name, re.IGNORECASE)
    file_br = int(br_match.group(1)) if br_match else None

    for target in targets:
        target_int, str_val = parse_condition_specifier(target)
        if target_int is not None:
            if file_br is not None and target_int == file_br:
                return True
        else:
            str_lower = str_val.lower()
            if str_lower in file_name.lower() or str_lower in file_path_str.lower():
                return True
    return False


def run_channel_behavior_analysis(
    sessions: Optional[List[Path]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main analysis pipeline for Channel-Behavior Regression.

    Parameters
    ----------
    sessions : list of Path, optional
        List of peristim NPZ files to analyze. Defaults to all STIM files under PERI_ROOT.
    config : dict, optional
        Analysis configuration dictionary.
    """
    cfg = config or {}
    neural_win = cfg.get("neural_window_ms", DEFAULT_NEURAL_WINDOW_MS)
    behavior_win = cfg.get("behavior_window_ms", DEFAULT_BEHAVIOR_WINDOW_MS)
    n_perms = cfg.get("n_permutations", DEFAULT_N_PERMUTATIONS)
    out_root = Path(cfg.get("out_root", REG_FIG_ROOT))

    _log("Starting Channel-Behavior Regression Analysis")
    ensure_dir(out_root)

    bad_channels = load_bad_channels()
    all_files = sorted(PERI_ROOT.rglob("peristim__*.npz"))
    if not all_files:
        _log(f"No peristim files found under {PERI_ROOT}")
        return {}

    _log(f"Discovered {len(all_files)} total peristim files.")
    refs = load_reference_mapping(all_files)

    stim_files = [f for f in all_files if get_cond_type(f) == "STIM"]
    targets = cfg.get("targets", PROCESS_ONLY)
    if targets:
        stim_files = [f for f in stim_files if is_stim_file_matching(f, targets)]

    target_filter = cfg.get("target_filter", None)
    if target_filter:
        t_clean = target_filter.strip().lower()
        stim_files = [f for f in stim_files if t_clean in get_target_folder(f).lower() or t_clean in str(f).lower()]

    if sessions is not None:
        stim_files = [f for f in stim_files if f in sessions or any(s.name == f.name for s in sessions)]

    _log(f"Evaluating {len(stim_files)} STIM condition(s)...")

    all_summary_rows = []
    all_channel_rows = []

    for idx, stim_f in enumerate(stim_files):
        target = get_target_folder(stim_f)
        try:
            with np.load(stim_f, allow_pickle=True) as d:
                port = get_ua_port(d)
        except Exception:
            port = "A"

        matched_ctrl = (
            refs["ctrl"].get((target, port))
            or refs["ctrl"].get((target, "A"))
            or refs["ctrl_by_target"].get(target)
            or refs["ctrl_fallback"]
        )
        sub_dir = out_root / target if target else out_root / "other"

        try:
            res = process_single_condition(
                stim_path=stim_f,
                ctrl_path=matched_ctrl,
                bad_channels=bad_channels,
                out_dir=sub_dir,
                neural_window_ms=neural_win,
                behavior_window_ms=behavior_win,
                n_permutations=n_perms,
                file_idx=idx + 1,
                total_files=len(stim_files),
            )
            if res:
                all_summary_rows.extend(res.get("summary_rows", []))
                all_channel_rows.extend(res.get("channel_rows", []))
        except Exception as err:
            warnings.warn(f"Error processing {stim_f.name}: {err}\n{traceback.format_exc()}")

    # Save consolidated summaries
    df_summary = pd.DataFrame(all_summary_rows)
    df_channels = pd.DataFrame(all_channel_rows)

    if not df_summary.empty:
        summary_csv = out_root / "channel_behavior_regression_summary.csv"
        df_summary.to_csv(summary_csv, index=False)
        _log(f"Saved consolidated summary to: {summary_csv}")
        if CLUST_FIG_ROOT != out_root:
            try:
                ensure_dir(CLUST_FIG_ROOT)
                df_summary.to_csv(CLUST_FIG_ROOT / "channel_behavior_regression_summary.csv", index=False)
            except Exception:
                pass

    if not df_channels.empty:
        channels_csv = out_root / "channel_behavior_regression_weights.csv"
        df_channels.to_csv(channels_csv, index=False)
        _log(f"Saved consolidated weights to: {channels_csv}")
        if CLUST_FIG_ROOT != out_root:
            try:
                ensure_dir(CLUST_FIG_ROOT)
                df_channels.to_csv(CLUST_FIG_ROOT / "channel_behavior_regression_weights.csv", index=False)
            except Exception:
                pass

    _log("Channel-behavior regression analysis completed successfully.")
    return {
        "summary_df": df_summary,
        "channels_df": df_channels,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Trial-by-trial channel-behavior regression analysis predicting behavioral perturbation from neural activity."
    )
    parser.add_argument(
        "--process-only", "--condition", "--br", "-c",
        nargs="+", dest="process_only", default=None,
        help="Condition(s) or BR index(es) to process (e.g. 12, BR_012, Target_A).",
    )
    parser.add_argument(
        "--target", "-t", type=str, default=None,
        help="Optional target folder filter (e.g. Target_A, Target_B).",
    )
    parser.add_argument(
        "--neural-win", nargs=2, type=float, default=DEFAULT_NEURAL_WINDOW_MS,
        help="Neural feature window in ms post-stim (e.g. 50 200).",
    )
    parser.add_argument(
        "--behavior-win", nargs=2, type=float, default=DEFAULT_BEHAVIOR_WINDOW_MS,
        help="Behavioral perturbation window in ms post-stim (e.g. 100 400).",
    )
    parser.add_argument(
        "--permutations", "-p", type=int, default=DEFAULT_N_PERMUTATIONS,
        help="Number of permutations for significance testing (default: 1000).",
    )
    args, _ = parser.parse_known_args()

    cfg = {
        "targets": args.process_only or PROCESS_ONLY,
        "target_filter": args.target,
        "neural_window_ms": tuple(args.neural_win),
        "behavior_window_ms": tuple(args.behavior_win),
        "n_permutations": args.permutations,
        "out_root": REG_FIG_ROOT,
    }
    run_channel_behavior_analysis(config=cfg)


if __name__ == "__main__":
    main()
