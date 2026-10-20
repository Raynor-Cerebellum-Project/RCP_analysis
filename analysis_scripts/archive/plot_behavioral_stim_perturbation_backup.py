"""
plot_behavioral_stim_perturbation.py

Channel-Behavior Regression Analysis across Cerebellum (NPRW) and Motor Cortex (Utah Array).

Scientific Question:
    Which neural channels carry information relevant to the behavioral perturbation
    induced by stimulation on a trial-by-trial basis?

Key Methodological Design (Addressing Scientific Grounding & Rigor):
    1. True 3D Task-Frame Kinematics:
       - Extracts calibrated 3D task-frame reach coordinates [X_lateral, Y_forward, Z_height] (mm)
         using triangulated two-camera DeepLabCut (DLC_3D) via plot_3d_kinematics.
       - Falls back gracefully to multi-camera 2D coordinates if 3D DLC files are unavailable.
    2. Exact Trial Identity Matching:
       - Intersects valid trial indices (raw_trial_indices) between neural firing rate arrays
         and behavioral kinematics, eliminating array truncation drift and ensuring trial identity.
    3. Within-Condition Trial-to-Trial Variation:
       - Supports within-condition residual mode ('within_condition_residual', default), defining
         y_tilde_i = y_i - mean(y_condition). This prevents the regression decoder from trivially
         discriminating STIM vs CTRL, forcing it to decode true within-condition trial-to-trial variance.
       - Also provides unbiased leave-one-out (LOO) reference trajectories for control trials.
       - Supports multiple 3D metrics: trajectory_residual, trajectory_deviation, forward_extent,
         lateral_deviation, and peak_speed.
    4. Nested Cross-Validation with In-Fold Preprocessing:
       - Outer K-Fold CV (default 5-fold) where StandardScaler is fitted strictly on training folds,
         completely preventing data leakage.
       - Inner CV selects optimal regularization strength alpha per fold.
       - Tracks out-of-fold generalization (Pearson r and R²), mean channel weights (beta_bar),
         cross-fold standard deviation (SD), and cross-fold sign consistency.
    5. Performance-Based Vectorized Permutation Testing:
       - Condition-stratified permutation of behavioral targets across trials (B=1000).
       - Evaluates out-of-fold decoder performance (r_null) to establish empirical model significance
         p_model = (1 + sum(r_null >= r_obs)) / (1 + B).
       - Evaluates channel-level coefficient null distributions and FDR Benjamini-Hochberg q-values.
    6. Publication-Grade Modular & Consolidated Visualizations:
       - plot_channel_weights: Bar chart of beta_bar with +/- 1 SD error bars and significance stars.
       - plot_prediction_scatter: Observed vs out-of-fold predicted behavior, colored by condition.
       - plot_decoder_null_distribution: Null r distribution histogram vs observed r with p_model.
       - plot_top_channels_traces: Time-resolved firing rate PSTHs (mean +/- SEM) for top channels.
       - plot_regression_summary_figure: Consolidated 4-panel publication-grade figure (2x2 grid).
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
from sklearn.model_selection import KFold

# Suppress benign warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Degrees of freedom <= 0 for slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN slice encountered.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*tight_layout.*")

# Add analysis_scripts to sys.path to enable importing plot_3d_kinematics
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    import plot_3d_kinematics as p3d
    HAS_3D_KINEMATICS = True
except Exception as _p3d_err:
    HAS_3D_KINEMATICS = False
    p3d = None

import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *
from RCP_analysis.python.functions.kinematics_utils import extract_xy_coordinates

# =============================================================================
# Configuration & Analysis Parameters
# =============================================================================

# Time Windows (ms relative to stimulation / IR beam onset)
WIN_PLOT_MS: Tuple[float, float] = (-800.0, 600.0)

# Neural Feature Epochs:
#   'pre_stim' (Default): [-300, -50] ms. Artifact- & blanking-free window testing whether preparatory
#                         neural state predicts the magnitude and direction of the reach perturbation.
#   'early_post': [50, 150] ms. Early evoked neural response to stimulation.
#   'late_post': [150, 300] ms. Reach execution and late stabilization response.
#   'peri_stim': [50, 200] ms. Broad post-stimulus window.
NEURAL_EPOCHS: Dict[str, Tuple[float, float]] = {
    "pre_stim": (-300.0, -50.0),
    "early_post": (50.0, 150.0),
    "late_post": (150.0, 300.0),
    "peri_stim": (50.0, 200.0),
}
DEFAULT_NEURAL_EPOCH: str = "pre_stim"
DEFAULT_NEURAL_WINDOW_MS: Tuple[float, float] = NEURAL_EPOCHS[DEFAULT_NEURAL_EPOCH]
DEFAULT_BEHAVIOR_WINDOW_MS: Tuple[float, float] = (100.0, 500.0)

# Analysis Modes:
#   'within_condition_residual' (Default): Subtracts condition mean to isolate trial-to-trial variance.
#   'all_stacked_raw': Stacks raw behavioral scores without condition-mean subtraction.
#   'stim_only': Trains and evaluates decoder strictly on STIM trials.
#   'ctrl_only': Trains and evaluates decoder strictly on CTRL trials.
DEFAULT_ANALYSIS_MODE: str = "within_condition_residual"

# Behavioral Metrics:
#   'endpoint_delta_x' (Default): Signed lateral endpoint perturbation from target-matched control (mm; left/right veer).
#   'endpoint_delta_y': Signed forward endpoint perturbation from control (mm; undershoot/overshoot).
#   'endpoint_delta_z': Signed vertical endpoint perturbation from control (mm; height displacement).
#   'endpoint_error_3d': Scalar 3D Euclidean distance to matched control endpoint (mm).
#   'plateau_time': Reach termination / plateau onset time (ms).
#   'trajectory_deviation': Mean 3D distance to target-matched control trajectory (LOO for CTRL).
#   'trajectory_residual': Mean Euclidean deviation from condition's own trajectory (LOO for CTRL).
#   'forward_extent': Maximum forward reach extent (Y_task) in mm.
#   'lateral_deviation': Lateral displacement (X_task) at maximum forward extension.
#   'peak_speed': Maximum 3D velocity magnitude in mm/s.
DEFAULT_BEHAVIOR_METRIC: str = "endpoint_delta_x"
CONTROL_REF_STAT: str = "median"  # 'median' (recommended for reach kinematics) or 'mean'

# 3D Kinematics Settings
DEFAULT_KEYPOINT: str = "middle"
SQUARE_SIZE_MM: float = 1.0  # calibrated 1.0 mm/unit
ALIGN_TO_TASK_FRAME: bool = True

# Regression & Permutation Settings
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


def resolve_session_name_from_path(p: Path) -> str:
    m = re.search(r"(NRR_[A-Za-z0-9]+)", p.name, re.IGNORECASE)
    if m:
        return m.group(1)
    for parent in [p] + list(p.parents):
        m = re.search(r"(NRR_[A-Za-z0-9]+)", parent.name, re.IGNORECASE)
        if m:
            return m.group(1)
        m2 = re.search(r"(\d{8}_[A-Za-z0-9_-]+)", parent.name)
        if m2:
            return m2.group(1)
    if PARAMS and hasattr(PARAMS, "session") and PARAMS.session:
        return str(PARAMS.session)
    return "session"


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
    ctrl_candidates: Dict[Tuple[str, str], List[Tuple[int, Path]]] = {}
    ctrl_by_target_candidates: Dict[str, List[Tuple[int, Path]]] = {}
    ctrl_fallback_candidates: List[Tuple[int, Path]] = []
    rest_map: Dict[float, Path] = {}
    rest_fallback: Optional[Path] = None

    for f in all_files:
        ctype = get_cond_type(f)
        target = get_target_folder(f)
        n_trials = 0
        try:
            with np.load(f, allow_pickle=True) as d:
                port = get_ua_port(d)
                fh = scalarize(safe_get_npz(d, "stim_freq_hz", None))
                dm = scalarize(safe_get_npz(d, "stim_dur_nominal_ms", None))
                pc = compute_pulse_count(fh, dm)
                if "beh_cam0_segs" in d and hasattr(d["beh_cam0_segs"], "shape") and len(d["beh_cam0_segs"].shape) >= 1:
                    n_trials = int(d["beh_cam0_segs"].shape[0])
                elif "event_ms" in d:
                    n_trials = len(d["event_ms"])
        except Exception:
            port = "A"
            pc = None

        if ctype == "CTRL":
            key = (target, port)
            if key not in ctrl_candidates:
                ctrl_candidates[key] = []
            ctrl_candidates[key].append((n_trials, f))

            if target:
                if target not in ctrl_by_target_candidates:
                    ctrl_by_target_candidates[target] = []
                ctrl_by_target_candidates[target].append((n_trials, f))

            ctrl_fallback_candidates.append((n_trials, f))
        elif ctype == "REST":
            if pc is not None and pc not in rest_map:
                rest_map[pc] = f
            if rest_fallback is None:
                rest_fallback = f

    # Select the control condition with the most valid behavioral reaches
    ctrl_map: Dict[Tuple[str, str], Path] = {
        k: max(candidates, key=lambda x: x[0])[1] for k, candidates in ctrl_candidates.items() if candidates
    }
    ctrl_by_target: Dict[str, Path] = {
        k: max(candidates, key=lambda x: x[0])[1] for k, candidates in ctrl_by_target_candidates.items() if candidates
    }
    ctrl_fallback = max(ctrl_fallback_candidates, key=lambda x: x[0])[1] if ctrl_fallback_candidates else None

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
    if base_loc is None or not Path(base_loc).exists():
        return out

    base_p = Path(base_loc)
    intan_cands = [base_p / "Intan", base_p]
    for d in intan_cands:
        if d.exists():
            for f in d.glob("*impedance*.csv"):
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

    br_cands = [base_p / "Blackrock", base_p]
    for d in br_cands:
        if d.exists():
            for f in d.glob("*impedance*.txt"):
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
    if not monkey:
        monkey = (getattr(PARAMS, "monkey", None) if PARAMS is not None else None) or "Nike"
    repo_cfg = (REPO_ROOT / "config" / f"electrode_port_mapping_{monkey}.csv") if "REPO_ROOT" in globals() else None
    if repo_cfg and repo_cfg.exists():
        return repo_cfg
    local_cfg = Path(__file__).resolve().parent.parent / "config" / f"electrode_port_mapping_{monkey}.csv"
    if local_cfg.exists():
        return local_cfg
    raise FileNotFoundError(f"Electrode mapping CSV not found for monkey {monkey}")


@lru_cache(maxsize=4)
def load_electrode_mapping(monkey: Optional[str] = None) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, np.ndarray]]:
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


# =============================================================================
# Neural Signal Preprocessing
# =============================================================================

def prepare_counts_3d(counts_raw: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Ensure neural count array is shaped (n_channels, n_trials, n_bins)."""
    if counts_raw is None:
        return None
    arr = np.asarray(counts_raw, dtype=float)
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim != 3:
        return None
    n0, n1, _ = arr.shape
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

    n_ch, n_trials, _ = counts.shape
    n_out = len(out_centers)

    bin_idx = np.digitize(centers, out_edges) - 1
    rebinned = np.zeros((n_ch, n_trials, n_out), dtype=float)

    for b in range(n_out):
        mask = bin_idx == b
        if np.any(mask):
            rebinned[:, :, b] = np.sum(counts[:, :, mask], axis=2)

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
# 3D Task-Frame Kinematics Loading & Cutting
# =============================================================================

def load_condition_3d_task_kinematics(
    npz_path: Path,
    session: str,
    target: str = "A",
    keypoint: str = DEFAULT_KEYPOINT,
    square_size_mm: float = SQUARE_SIZE_MM,
    align_task_frame: bool = ALIGN_TO_TASK_FRAME,
    dlc_3d_dir: Optional[Path] = None,
    metadata_csv: Optional[Path] = None,
    aligned_ckpt_root: Optional[Path] = None,
    behv_ckpt_root: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load continuous triangulated 3D DLC kinematics, cut per-trial reaches around event_ms,
    and rotate into the monkey's calibrated 3D task frame [X_lateral, Y_forward, Z_height] in mm.

    Returns None if 3D DLC is unavailable, enabling graceful fallback to 2D camera coordinates.
    """
    if not HAS_3D_KINEMATICS or p3d is None:
        return None

    try:
        with np.load(npz_path, allow_pickle=True) as peri:
            br_idx = -1
            if "br_idx" in peri:
                try:
                    br_idx = int(peri["br_idx"])
                except Exception:
                    pass
            if br_idx < 0:
                m = re.search(r"BR_(\d+)", npz_path.name, re.IGNORECASE)
                if m:
                    br_idx = int(m.group(1))

            if br_idx < 0:
                return None

            event_ms = None
            for k in ["event_ms_sub", "event_ms"]:
                if k in peri:
                    event_ms = peri[k]
                    break
            if event_ms is None or len(event_ms) == 0:
                return None

            raw_trial_indices = None
            for k in ["raw_trial_indices", "raw_indices", "trial_indices", "raw_trial_idx"]:
                if k in peri:
                    raw_trial_indices = peri[k]
                    break

            labels = None
            for k in ["labels_sub", "labels", "trial_labels"]:
                if k in peri:
                    labels = peri[k]
                    break
            if labels is not None and np.size(labels) == np.size(event_ms) and target:
                labels_arr = np.asarray(labels).astype(str)
                t_mask = np.array([lbl.upper() == target.upper() for lbl in labels_arr])
                if np.any(t_mask):
                    event_ms = event_ms[t_mask]
                    if raw_trial_indices is not None and len(raw_trial_indices) == len(t_mask):
                        raw_trial_indices = raw_trial_indices[t_mask]

            peri_meta = {}
            if "meta" in peri and hasattr(peri["meta"], "item"):
                try:
                    peri_meta = peri["meta"].item()
                except Exception:
                    pass
            elif "align_meta_raw" in peri and hasattr(peri["align_meta_raw"], "item"):
                try:
                    raw_m = peri["align_meta_raw"].item()
                    peri_meta = json.loads(raw_m) if isinstance(raw_m, str) else raw_m
                except Exception:
                    pass

        sess_dir = p3d.resolve_session_dir(session=session, peri_root=npz_path.parent)
        if metadata_csv is None:
            metadata_csv = p3d.find_metadata_csv(session=session, peri_root=npz_path.parent)

        if aligned_ckpt_root is None and sess_dir:
            aligned_cands = [
                sess_dir / "results" / "checkpoints" / "Aligned",
                sess_dir / "results" / "checkpoints",
                npz_path.parent.parent / "Aligned",
                npz_path.parent,
            ]
            for ac in aligned_cands:
                if ac.exists():
                    aligned_ckpt_root = ac
                    break

        vid_idx = p3d.resolve_video_index_for_br(br_idx, metadata_csv, peri_meta)
        if vid_idx is None:
            vid_idx = br_idx

        # Resolve 3D DLC root
        if dlc_3d_dir is None:
            sess_dir = p3d.resolve_session_dir(session=session, peri_root=npz_path.parent)
            candidates = []
            if sess_dir:
                candidates.extend([
                    sess_dir / "Video" / "DLC_3D_new",
                    sess_dir / "Video" / "DLC_3D",
                    sess_dir / "DLC_3D_new",
                    sess_dir / "DLC_3D",
                ])
            if SESSION_LOC:
                candidates.extend([
                    Path(SESSION_LOC) / "Video" / "DLC_3D_new",
                    Path(SESSION_LOC) / "Video" / "DLC_3D",
                    Path(SESSION_LOC) / "DLC_3D_new",
                ])
            for cand in candidates:
                if cand.exists():
                    dlc_3d_dir = cand
                    break

        if dlc_3d_dir is None or not Path(dlc_3d_dir).exists():
            return None

        dlc_csv_path = p3d.find_3d_dlc_csv(Path(dlc_3d_dir), session, vid_idx, br_idx)
        if dlc_csv_path is None or not dlc_csv_path.exists():
            return None

        df_dlc, _ = p3d.load_dlc_3d_file(dlc_csv_path, square_size_mm=square_size_mm)
        if df_dlc is None:
            return None

        frame_t_ms = p3d.load_frame_timeline_ms(
            br_idx=br_idx,
            vid_idx=vid_idx,
            session=session,
            aligned_ckpt_root=aligned_ckpt_root,
            behv_ckpt_root=behv_ckpt_root,
        )
        if frame_t_ms is None:
            n_frames = len(df_dlc)
            frame_t_ms = np.arange(n_frames) * (1000.0 / 100.0)

        rel_t, x_all, y_all, z_all, valid_mask = p3d.segment_condition_3d_reaches(
            dlc_df=df_dlc,
            frame_t_ms=frame_t_ms,
            event_ms=event_ms,
            keypoint=keypoint,
            pre_zero_ms=p3d.PRE_ZERO_MS,
            post_zero_ms=p3d.POST_ZERO_MS,
            step_ms=p3d.STEP_MS,
        )

        if align_task_frame:
            session_landmarks = p3d.scan_session_landmarks(Path(dlc_3d_dir), square_size_mm=square_size_mm)
            R_task, p_start = p3d.compute_task_rotation_matrix(
                dlc_df=df_dlc,
                session_landmarks=session_landmarks,
                square_size_mm=square_size_mm,
            )
            pts_cam = np.stack([x_all, y_all, z_all], axis=0) - p_start.reshape(3, 1, 1)
            pts_task = np.einsum("ji,jkl->ikl", R_task, pts_cam)
            x_all, y_all, z_all = pts_task[0], pts_task[1], pts_task[2]

        coords_3d = np.stack([x_all, y_all, z_all], axis=1)  # (n_trials, 3, n_time)
        raw_tids = np.asarray(raw_trial_indices) if raw_trial_indices is not None else np.arange(len(event_ms))

        return {
            "beh_t": rel_t,
            "coords": coords_3d,
            "x_coords": x_all,
            "y_coords": y_all,
            "z_coords": z_all,
            "valid_mask": valid_mask,
            "raw_trial_indices": raw_tids,
            "is_3d": True,
            "coord_names": ["X_lateral", "Y_forward", "Z_height"],
        }
    except Exception as e:
        _log(f"3D kinematics loading skipped for {npz_path.name}: {e}")
        return None


def extract_br_index(path: Optional[Path]) -> int:
    """Extract numeric BR index from filename (e.g. BR_012 -> 12)."""
    if path is None:
        return -1
    m = re.search(r"BR_(\d+)", path.name, re.IGNORECASE)
    return int(m.group(1)) if m else -1


def load_plateau_analysis_table(
    session: str,
    peri_root: Optional[Path] = None,
    session_loc: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Load precomputed plateau detection results CSV if generated by plot_plateau_analysis.py."""
    candidates = []
    if peri_root is not None:
        try:
            candidates.append(Path(peri_root).parents[1] / "figures" / "plateau_analysis" / "tables" / "plateau_detection_results_all_conditions.csv")
            candidates.append(Path(peri_root).parent / "figures" / "plateau_analysis" / "tables" / "plateau_detection_results_all_conditions.csv")
        except Exception:
            pass
    if session_loc is not None:
        candidates.append(Path(session_loc) / "results" / "figures" / "plateau_analysis" / "tables" / "plateau_detection_results_all_conditions.csv")
    if "REG_FIG_ROOT" in globals() and REG_FIG_ROOT:
        candidates.append(Path(REG_FIG_ROOT).parents[0] / "plateau_analysis" / "tables" / "plateau_detection_results_all_conditions.csv")
    for cand in candidates:
        if cand.exists():
            try:
                df = pd.read_csv(cand)
                if "br_idx" in df.columns and "onset_ms" in df.columns:
                    _log(f"  Loaded plateau analysis table from: {cand.name}")
                    return df
            except Exception:
                pass
    return None


def extract_plateau_times_from_table(
    df_plat: Optional[pd.DataFrame],
    br_idx: int,
    target: str,
    n_trials: int,
) -> Optional[np.ndarray]:
    """Extract per-trial plateau onset times in ms from plot_plateau_analysis.py table."""
    if df_plat is None or df_plat.empty or br_idx < 0:
        return None
    try:
        t_clean = f"target{target.strip().lower()}"
        mask = (df_plat["br_idx"] == br_idx)
        if "condition" in df_plat.columns:
            mask = mask & df_plat["condition"].str.replace("_", "").str.lower().str.contains(t_clean)
        sub = df_plat[mask]
        if sub.empty:
            return None
        out = np.full(n_trials, np.nan, dtype=float)
        for _, row in sub.iterrows():
            t_idx = int(row["trial_idx"])
            if 0 <= t_idx < n_trials and bool(row.get("plateau_found", True)):
                out[t_idx] = float(row["onset_ms"])
        if np.any(np.isfinite(out)):
            return out
    except Exception:
        pass
    return None


# =============================================================================
# Behavioral Target & 3D Endpoint Computation
# =============================================================================

def compute_trial_endpoints_3d(
    coords: np.ndarray,
    beh_t: np.ndarray,
    search_window_ms: Tuple[float, float] = DEFAULT_BEHAVIOR_WINDOW_MS,
    speed_threshold_ratio: float = 0.20,
    min_stable_ms: float = 60.0,
    override_plateau_times: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes single-trial 3D reach endpoints and movement termination / plateau onset times.

    For calibrated 3D task-frame kinematics [X_lateral, Y_forward, Z_height]:
        1. If override_plateau_times is provided (from plot_plateau_analysis.py table), uses
           the detected plateau timestamps directly.
        2. Otherwise, calculates 3D velocity and speed magnitude v(t) = sqrt(vx^2 + vy^2 + vz^2).
        3. In search_window_ms, locates the peak forward reach index k_peak_y = argmax(Y(t)).
        4. Identifies the peak velocity preceding peak forward extent.
        5. Detects plateau onset where speed drops below threshold (movement deceleration into
           target plateau) or falls back to k_peak_y (reach apex).
        6. Extracts 3D endpoint coordinate [X, Y, Z] at plateau onset and timing t_plateau.

    Parameters
    ----------
    coords : np.ndarray, shape (n_trials, n_coords, n_time)
        Trial kinematics coordinates (mm).
    beh_t : np.ndarray, shape (n_time,)
        Relative time vector in ms.
    search_window_ms : Tuple[float, float]
        Time window to search for reach termination (default: 100 to 500 ms).
    speed_threshold_ratio : float
        Fraction of peak velocity defining stabilization (default: 0.20).
    min_stable_ms : float
        Minimum duration of stabilization in ms (default: 60.0 ms).
    override_plateau_times : Optional[np.ndarray]
        Precomputed plateau onset times from plot_plateau_analysis.py table.

    Returns
    -------
    endpoints : np.ndarray, shape (n_trials, n_coords)
        Coordinates at reach endpoint/plateau in mm.
    plateau_times : np.ndarray, shape (n_trials,)
        Plateau onset / reach termination time in ms.
    """
    n_trials, n_coords, n_time = coords.shape
    endpoints = np.full((n_trials, n_coords), np.nan, dtype=float)
    plateau_times = np.full(n_trials, np.nan, dtype=float)

    if n_time < 3 or len(beh_t) != n_time:
        return endpoints, plateau_times

    dt_s = (beh_t[1] - beh_t[0]) / 1000.0 if len(beh_t) > 1 else 0.01
    dt_ms = (beh_t[1] - beh_t[0]) if len(beh_t) > 1 else 10.0
    stable_samples = max(2, int(round(min_stable_ms / max(dt_ms, 1e-3))))

    t0, t1 = search_window_ms
    win_mask = (beh_t >= t0) & (beh_t <= t1)
    if not np.any(win_mask):
        win_mask = (beh_t >= 0.0) & (beh_t <= 500.0)
    win_indices = np.where(win_mask)[0]
    if len(win_indices) == 0:
        win_indices = np.arange(n_time)

    # Baseline speed before t=0 ms
    base_mask = (beh_t >= -400.0) & (beh_t <= -50.0)

    for i in range(n_trials):
        tr_coords = coords[i]  # (n_coords, n_time)
        if np.all(np.isnan(tr_coords)):
            continue

        # If override plateau times provided from plot_plateau_analysis.py table
        if override_plateau_times is not None and i < len(override_plateau_times) and np.isfinite(override_plateau_times[i]):
            t_plat = float(override_plateau_times[i])
            k_end = int(np.argmin(np.abs(beh_t - t_plat)))
            endpoints[i, :] = tr_coords[:, k_end]
            plateau_times[i] = t_plat
            continue

        vel = np.gradient(tr_coords, dt_s, axis=1)
        speed = np.sqrt(np.nansum(vel ** 2, axis=0))

        v_base = float(np.nanmedian(speed[base_mask])) if np.any(base_mask) else 20.0
        v_base = max(v_base, 10.0)

        fwd_axis = 1 if n_coords >= 2 else 0
        fwd_trace = tr_coords[fwd_axis]

        win_fwd = fwd_trace[win_indices]
        if np.all(np.isnan(win_fwd)):
            continue

        k_local_peak_y = int(np.nanargmax(win_fwd))
        k_peak_y = win_indices[k_local_peak_y]

        win_speed = speed[win_indices[: k_local_peak_y + 1]]
        v_max = float(np.nanmax(win_speed)) if len(win_speed) > 0 and np.any(np.isfinite(win_speed)) else v_base
        v_thresh = max(v_base * 1.5, v_max * speed_threshold_ratio)

        k_end = k_peak_y
        if len(win_speed) > 3 and v_max > 40.0:
            k_local_peak_v = int(np.nanargmax(win_speed))
            k_peak_v = win_indices[k_local_peak_v]

            for k in range(k_peak_v, min(k_peak_y + stable_samples, n_time - stable_samples)):
                window = speed[k : k + stable_samples]
                if np.all(np.isfinite(window)) and np.all(window <= v_thresh):
                    k_end = k
                    break

        endpoints[i, :] = tr_coords[:, k_end]
        plateau_times[i] = beh_t[k_end]

    return endpoints, plateau_times


def compute_behavioral_features(
    s_beh_data: Dict[str, Any],
    c_beh_data: Dict[str, Any],
    behavior_window_ms: Tuple[float, float] = DEFAULT_BEHAVIOR_WINDOW_MS,
    behavior_metric: str = DEFAULT_BEHAVIOR_METRIC,
    analysis_mode: str = DEFAULT_ANALYSIS_MODE,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Computes single-trial behavioral metrics in physical task units (mm, ms, or mm/s).

    Biologically Grounded Directional & Plateau Metrics:
        - 'endpoint_delta_x' (Default): Signed lateral reach perturbation from target-matched control
                                       endpoint (mm; left/right veer: ΔX = X_stim - X_ctrl).
        - 'endpoint_delta_y': Signed forward reach perturbation from control (mm; undershoot/overshoot).
        - 'endpoint_delta_z': Signed vertical reach perturbation from control (mm; height displacement).
        - 'endpoint_error_3d': Scalar 3D Euclidean distance to matched control endpoint (mm; ||Δe||).
        - 'plateau_time': Reach termination / plateau onset time (ms relative to stim onset).
        - 'trajectory_deviation': Mean 3D distance to target-matched control trajectory (LOO for CTRL).
        - 'trajectory_residual': Mean Euclidean deviation from condition's own trajectory (LOO for CTRL).
        - 'forward_extent': Maximum forward reach extent (Y_task) in mm.
        - 'lateral_deviation': Lateral coordinate (X_task) at maximum forward extension.
        - 'peak_speed': Maximum 3D velocity magnitude in mm/s.

    Condition Residualization:
        When analysis_mode == 'within_condition_residual', subtracts condition-mean behavioral
        target y_tilde = y - mean(y_cond) to isolate true within-condition trial-to-trial variance.
    """
    s_coords = s_beh_data.get("coords")
    c_coords = c_beh_data.get("coords")
    t_beh = s_beh_data.get("beh_t")
    is_3d = bool(s_beh_data.get("is_3d", False))
    coord_names = s_beh_data.get("coord_names", [])

    if s_coords is None or c_coords is None or t_beh is None:
        return None, None, {}

    t0, t1 = behavior_window_ms
    win_mask = (t_beh >= t0) & (t_beh <= t1)
    if not np.any(win_mask):
        win_mask = (t_beh >= 0.0) & (t_beh <= 500.0)

    n_s = s_coords.shape[0]
    n_c = c_coords.shape[0]

    s_override = s_beh_data.get("override_plateau_times")
    c_override = c_beh_data.get("override_plateau_times")

    # 1. Compute 3D reach endpoints and plateau onset times
    endpoints_s, plateau_t_s = compute_trial_endpoints_3d(s_coords, t_beh, behavior_window_ms, override_plateau_times=s_override)
    endpoints_c, plateau_t_c = compute_trial_endpoints_3d(c_coords, t_beh, behavior_window_ms, override_plateau_times=c_override)

    # Matched control endpoint reference (median or mean)
    c_median_endpoint = np.nanmedian(endpoints_c, axis=0)  # (n_coords,)
    s_median_endpoint = np.nanmedian(endpoints_s, axis=0)
    c_mean_endpoint = np.nanmean(endpoints_c, axis=0)
    s_mean_endpoint = np.nanmean(endpoints_s, axis=0)

    c_ref_endpoint = c_median_endpoint if CONTROL_REF_STAT == "median" else c_mean_endpoint
    s_ref_endpoint = s_median_endpoint if CONTROL_REF_STAT == "median" else s_mean_endpoint

    # Directional endpoint perturbation vectors relative to control reference
    if np.any(np.isnan(c_ref_endpoint)):
        endpoints_s_delta = endpoints_s - (s_ref_endpoint[None, :] if np.all(np.isfinite(s_ref_endpoint)) else 0.0)
    else:
        endpoints_s_delta = endpoints_s - c_ref_endpoint[None, :]

    # Unbiased Leave-One-Out (LOO) control endpoint perturbations
    n_c_valid = int(np.sum(np.all(np.isfinite(endpoints_c), axis=1)))
    endpoints_c_delta = np.zeros_like(endpoints_c)
    if n_c_valid > 1:
        sum_c = np.nansum(endpoints_c, axis=0)
        for i in range(n_c):
            if np.all(np.isfinite(endpoints_c[i])):
                if CONTROL_REF_STAT == "median":
                    mask_other = np.ones(n_c, dtype=bool)
                    mask_other[i] = False
                    mask_other = mask_other & np.all(np.isfinite(endpoints_c), axis=1)
                    loo_ref = np.nanmedian(endpoints_c[mask_other], axis=0) if np.any(mask_other) else endpoints_c[i]
                else:
                    loo_ref = (sum_c - endpoints_c[i]) / float(n_c_valid - 1)
                endpoints_c_delta[i] = endpoints_c[i] - loo_ref
            else:
                endpoints_c_delta[i] = np.nan
    else:
        endpoints_c_delta[:] = 0.0

    # Plateau time shifts
    c_mean_plateau = float(np.nanmean(plateau_t_c)) if np.any(np.isfinite(plateau_t_c)) else 0.0
    s_mean_plateau = float(np.nanmean(plateau_t_s)) if np.any(np.isfinite(plateau_t_s)) else 0.0

    # 2. Compute selected single-trial behavioral metric
    if behavior_metric == "endpoint_delta_x":
        # Signed lateral displacement relative to control (mm)
        y_s_raw = endpoints_s_delta[:, 0]
        y_c_raw = endpoints_c_delta[:, 0]
        unit_str = "mm"
        target_label = "Lateral Endpoint Shift ΔX (mm)"

    elif behavior_metric == "endpoint_delta_y":
        # Signed forward displacement relative to control (mm)
        axis_idx = 1 if s_coords.shape[1] >= 2 else 0
        y_s_raw = endpoints_s_delta[:, axis_idx]
        y_c_raw = endpoints_c_delta[:, axis_idx]
        unit_str = "mm"
        target_label = "Forward Endpoint Shift ΔY (mm)"

    elif behavior_metric == "endpoint_delta_z":
        # Signed vertical displacement relative to control (mm)
        axis_idx = 2 if s_coords.shape[1] >= 3 else 0
        y_s_raw = endpoints_s_delta[:, axis_idx]
        y_c_raw = endpoints_c_delta[:, axis_idx]
        unit_str = "mm"
        target_label = "Vertical Endpoint Shift ΔZ (mm)"

    elif behavior_metric == "endpoint_error_3d":
        # Scalar 3D endpoint error norm ||Δe|| (mm)
        y_s_raw = np.sqrt(np.nansum(endpoints_s_delta ** 2, axis=1))
        y_c_raw = np.sqrt(np.nansum(endpoints_c_delta ** 2, axis=1))
        unit_str = "mm"
        target_label = "3D Endpoint Error ||Δe|| (mm)"

    elif behavior_metric == "plateau_time":
        # Plateau onset / reach termination time (ms)
        y_s_raw = plateau_t_s
        y_c_raw = plateau_t_c
        unit_str = "ms"
        target_label = "Plateau Onset Time (ms)"

    elif behavior_metric == "forward_extent" and is_3d:
        y_s_raw = np.nanmax(s_coords[:, 1, win_mask], axis=1)
        y_c_raw = np.nanmax(c_coords[:, 1, win_mask], axis=1)
        unit_str = "mm"
        target_label = "Forward Reach Extent (mm)"

    elif behavior_metric == "lateral_deviation" and is_3d:
        y_s_raw = np.zeros(n_s, dtype=float)
        for i in range(n_s):
            win_y = s_coords[i, 1, win_mask]
            if np.all(np.isnan(win_y)):
                y_s_raw[i] = np.nan
            else:
                peak_idx = int(np.nanargmax(win_y))
                y_s_raw[i] = s_coords[i, 0, win_mask][peak_idx]

        y_c_raw = np.zeros(n_c, dtype=float)
        for i in range(n_c):
            win_y = c_coords[i, 1, win_mask]
            if np.all(np.isnan(win_y)):
                y_c_raw[i] = np.nan
            else:
                peak_idx = int(np.nanargmax(win_y))
                y_c_raw[i] = c_coords[i, 0, win_mask][peak_idx]
        unit_str = "mm"
        target_label = "Lateral Deviation at Apex (mm)"

    elif behavior_metric == "peak_speed":
        dt_s = (t_beh[1] - t_beh[0]) / 1000.0 if len(t_beh) > 1 else 0.01
        vel_s = np.gradient(s_coords, dt_s, axis=2)
        speed_s = np.sqrt(np.nansum(vel_s ** 2, axis=1))
        y_s_raw = np.nanmax(speed_s[:, win_mask], axis=1)

        vel_c = np.gradient(c_coords, dt_s, axis=2)
        speed_c = np.sqrt(np.nansum(vel_c ** 2, axis=1))
        y_c_raw = np.nanmax(speed_c[:, win_mask], axis=1)
        unit_str = "mm/s"
        target_label = "Peak 3D Speed (mm/s)"

    elif behavior_metric == "trajectory_deviation":
        ctrl_mean_traj = np.nanmean(c_coords, axis=0)
        diff_s = s_coords - ctrl_mean_traj[None, :, :]
        dist_s = np.sqrt(np.nansum(diff_s ** 2, axis=1))
        y_s_raw = np.nanmean(dist_s[:, win_mask], axis=1)

        dist_c = np.full((n_c, len(t_beh)), np.nan)
        if n_c > 1:
            for i in range(n_c):
                c_loo = (n_c * ctrl_mean_traj - c_coords[i]) / (n_c - 1)
                diff_ci = c_coords[i] - c_loo
                dist_c[i] = np.sqrt(np.nansum(diff_ci ** 2, axis=0))
        else:
            dist_c[:] = 0.0
        y_c_raw = np.nanmean(dist_c[:, win_mask], axis=1)
        unit_str = "mm"
        target_label = "Trajectory Deviation from Control (mm)"

    else:
        # 'trajectory_residual'
        s_mean_traj = np.nanmean(s_coords, axis=0)
        c_mean_traj = np.nanmean(c_coords, axis=0)

        diff_s = s_coords - s_mean_traj[None, :, :]
        dist_s = np.sqrt(np.nansum(diff_s ** 2, axis=1))
        y_s_raw = np.nanmean(dist_s[:, win_mask], axis=1)

        dist_c = np.full((n_c, len(t_beh)), np.nan)
        if n_c > 1:
            for i in range(n_c):
                c_loo = (n_c * c_mean_traj - c_coords[i]) / (n_c - 1)
                diff_ci = c_coords[i] - c_loo
                dist_c[i] = np.sqrt(np.nansum(diff_ci ** 2, axis=0))
        else:
            dist_c[:] = 0.0
        y_c_raw = np.nanmean(dist_c[:, win_mask], axis=1)
        unit_str = "mm"
        target_label = "Trajectory Residual (mm)"

    # 3. Behavioral Perturbation Statistics & Phenotype Classification
    mean_delta_e = s_mean_endpoint - c_mean_endpoint
    mean_delta_norm = float(np.linalg.norm(mean_delta_e)) if np.all(np.isfinite(mean_delta_e)) else 0.0

    # Directional t-tests comparing Stim vs Control endpoints
    t_x, p_val_dx = stats.ttest_ind(endpoints_s[:, 0], endpoints_c[:, 0], nan_policy="omit")
    t_y, p_val_dy = stats.ttest_ind(endpoints_s[:, 1], endpoints_c[:, 1], nan_policy="omit")
    t_z, p_val_dz = stats.ttest_ind(endpoints_s[:, 2], endpoints_c[:, 2], nan_policy="omit") if s_coords.shape[1] >= 3 else (0.0, 1.0)

    p_val_dx = float(p_val_dx) if np.isfinite(p_val_dx) else 1.0
    p_val_dy = float(p_val_dy) if np.isfinite(p_val_dy) else 1.0
    p_val_dz = float(p_val_dz) if np.isfinite(p_val_dz) else 1.0

    # 3D Endpoint Dispersion (RMS distance to condition's own endpoint mean)
    disp_s_trials = np.sqrt(np.nansum((endpoints_s - s_mean_endpoint[None, :]) ** 2, axis=1))
    disp_c_trials = np.sqrt(np.nansum((endpoints_c - c_mean_endpoint[None, :]) ** 2, axis=1))
    disp_s_rms = float(np.sqrt(np.nanmean(disp_s_trials ** 2))) if np.any(np.isfinite(disp_s_trials)) else 0.0
    disp_c_rms = float(np.sqrt(np.nanmean(disp_c_trials ** 2))) if np.any(np.isfinite(disp_c_trials)) else 0.0
    disp_ratio = disp_s_rms / max(disp_c_rms, 1e-6)

    # Levene's test for equality of endpoint variances
    valid_disp_s = disp_s_trials[np.isfinite(disp_s_trials)]
    valid_disp_c = disp_c_trials[np.isfinite(disp_c_trials)]
    if len(valid_disp_s) >= 2 and len(valid_disp_c) >= 2:
        try:
            _, p_disp = stats.levene(valid_disp_s, valid_disp_c)
            p_disp = float(p_disp)
        except Exception:
            p_disp = 1.0
    else:
        p_disp = 1.0

    # Plateau timing comparison
    delta_plateau_t = s_mean_plateau - c_mean_plateau
    try:
        _, p_plateau = stats.ttest_ind(plateau_t_s, plateau_t_c, nan_policy="omit")
        p_plateau = float(p_plateau)
    except Exception:
        p_plateau = 1.0

    # Phenotype Classification:
    #   Case A: Directional Shift (consistent mean displacement)
    #   Case B: Increased Variance / Degraded Precision (increased dispersion without shift)
    #   Case C: Both (directional shift AND increased dispersion)
    #   Case D: No Significant Effect
    any_dir_sig = (p_val_dx < 0.05) or (p_val_dy < 0.05) or (p_val_dz < 0.05)
    disp_sig = (p_disp < 0.05) and (disp_ratio > 1.0)

    if any_dir_sig and disp_sig:
        phenotype = "Both (Directional Shift + Increased Variance)"
    elif any_dir_sig:
        phenotype = "Directional Shift"
    elif disp_sig:
        phenotype = "Increased Variance"
    else:
        phenotype = "No Significant Effect"

    beh_stats = {
        "endpoints_s": endpoints_s,
        "endpoints_c": endpoints_c,
        "endpoints_s_delta": endpoints_s_delta,
        "endpoints_c_delta": endpoints_c_delta,
        "plateau_t_s": plateau_t_s,
        "plateau_t_c": plateau_t_c,
        "c_mean_endpoint": c_mean_endpoint,
        "s_mean_endpoint": s_mean_endpoint,
        "mean_delta_e": mean_delta_e,
        "mean_delta_x": float(mean_delta_e[0]) if len(mean_delta_e) > 0 else 0.0,
        "mean_delta_y": float(mean_delta_e[1]) if len(mean_delta_e) > 1 else 0.0,
        "mean_delta_z": float(mean_delta_e[2]) if len(mean_delta_e) > 2 else 0.0,
        "mean_delta_norm": mean_delta_norm,
        "p_val_dx": p_val_dx,
        "p_val_dy": p_val_dy,
        "p_val_dz": p_val_dz,
        "disp_s_rms": disp_s_rms,
        "disp_c_rms": disp_c_rms,
        "disp_ratio": disp_ratio,
        "p_disp": p_disp,
        "s_mean_plateau": s_mean_plateau,
        "c_mean_plateau": c_mean_plateau,
        "delta_plateau_t": delta_plateau_t,
        "p_plateau": p_plateau,
        "phenotype": phenotype,
    }

    # 4. Condition-level residualization
    stim_mean = float(np.nanmean(y_s_raw)) if len(y_s_raw) > 0 else 0.0
    ctrl_mean = float(np.nanmean(y_c_raw)) if len(y_c_raw) > 0 else 0.0

    if analysis_mode == "within_condition_residual":
        y_stim = y_s_raw - stim_mean
        y_ctrl = y_c_raw - ctrl_mean
        final_target_label = f"Residual {target_label}"
    else:
        y_stim = y_s_raw.copy()
        y_ctrl = y_c_raw.copy()
        final_target_label = target_label

    meta = {
        "y_stim_raw": y_s_raw,
        "y_ctrl_raw": y_c_raw,
        "stim_mean_raw": stim_mean,
        "ctrl_mean_raw": ctrl_mean,
        "behavior_metric": behavior_metric,
        "analysis_mode": analysis_mode,
        "is_3d": is_3d,
        "coord_names": coord_names,
        "beh_t": t_beh,
        "win_mask": win_mask,
        "unit_str": unit_str,
        "target_label": final_target_label,
        "raw_target_label": target_label,
        "beh_stats": beh_stats,
    }

    return y_stim, y_ctrl, meta


# =============================================================================
# Data Preparation Function with Exact Trial Identity Matching
# =============================================================================

def prepare_regression_data(
    session_data: Dict[str, Any],
    neural_window_ms: Tuple[float, float] = DEFAULT_NEURAL_WINDOW_MS,
    behavior_window_ms: Tuple[float, float] = DEFAULT_BEHAVIOR_WINDOW_MS,
    analysis_mode: str = DEFAULT_ANALYSIS_MODE,
    behavior_metric: str = DEFAULT_BEHAVIOR_METRIC,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Prepares trial-level neural features and behavioral targets by matching exact trial IDs
    (raw_trial_indices) between neural rate tensors and behavioral kinematics.
    """
    s_rates = session_data.get("stim_rates")  # (n_ch, n_stim_trials, n_bins)
    c_rates = session_data.get("ctrl_rates")  # (n_ch, n_ctrl_trials, n_bins)
    time_ms = session_data.get("time_ms")
    channel_ids = session_data.get("channel_ids", [])

    if s_rates is None or c_rates is None or time_ms is None:
        _log("Missing neural rate matrices for regression data preparation.")
        return None, None, None, None, {}

    stim_beh_data = session_data.get("stim_beh_data", {})
    ctrl_beh_data = session_data.get("ctrl_beh_data", {})

    # 1. Compute behavioral targets
    y_stim, y_ctrl, beh_meta = compute_behavioral_features(
        s_beh_data=stim_beh_data,
        c_beh_data=ctrl_beh_data,
        behavior_window_ms=behavior_window_ms,
        behavior_metric=behavior_metric,
        analysis_mode=analysis_mode,
    )

    if y_stim is None or y_ctrl is None:
        _log("Failed to compute behavioral features.")
        return None, None, None, None, {}

    # 2. Exact trial matching between neural rates and kinematics
    s_neural_tids = session_data.get("stim_trial_ids")
    c_neural_tids = session_data.get("ctrl_trial_ids")
    s_beh_tids = stim_beh_data.get("raw_trial_indices")
    c_beh_tids = ctrl_beh_data.get("raw_trial_indices")
    s_beh_valid = stim_beh_data.get("valid_mask", np.ones(len(y_stim), dtype=bool))
    c_beh_valid = ctrl_beh_data.get("valid_mask", np.ones(len(y_ctrl), dtype=bool))

    # STIM trial matching
    matched_s_neural_idx = []
    matched_s_beh_idx = []
    matched_s_tids = []

    if s_neural_tids is not None and s_beh_tids is not None:
        s_beh_lookup = {tid: idx for idx, tid in enumerate(s_beh_tids)}
        for n_idx, n_tid in enumerate(s_neural_tids):
            if n_tid in s_beh_lookup:
                b_idx = s_beh_lookup[n_tid]
                if b_idx < len(y_stim) and s_beh_valid[b_idx] and np.isfinite(y_stim[b_idx]):
                    matched_s_neural_idx.append(n_idx)
                    matched_s_beh_idx.append(b_idx)
                    matched_s_tids.append(n_tid)
    else:
        n_common_s = min(s_rates.shape[1], len(y_stim))
        for i in range(n_common_s):
            if s_beh_valid[i] and np.isfinite(y_stim[i]):
                matched_s_neural_idx.append(i)
                matched_s_beh_idx.append(i)
                matched_s_tids.append(i)

    # CTRL trial matching
    matched_c_neural_idx = []
    matched_c_beh_idx = []
    matched_c_tids = []

    if c_neural_tids is not None and c_beh_tids is not None:
        c_beh_lookup = {tid: idx for idx, tid in enumerate(c_beh_tids)}
        for n_idx, n_tid in enumerate(c_neural_tids):
            if n_tid in c_beh_lookup:
                b_idx = c_beh_lookup[n_tid]
                if b_idx < len(y_ctrl) and c_beh_valid[b_idx] and np.isfinite(y_ctrl[b_idx]):
                    matched_c_neural_idx.append(n_idx)
                    matched_c_beh_idx.append(b_idx)
                    matched_c_tids.append(n_tid)
    else:
        n_common_c = min(c_rates.shape[1], len(y_ctrl))
        for i in range(n_common_c):
            if c_beh_valid[i] and np.isfinite(y_ctrl[i]):
                matched_c_neural_idx.append(i)
                matched_c_beh_idx.append(i)
                matched_c_tids.append(i)

    # 3. Extract neural features (mean firing rate in neural window)
    n0, n1 = neural_window_ms
    n_mask = (time_ms >= n0) & (time_ms <= n1)
    if not np.any(n_mask):
        n_mask = (time_ms >= 50.0) & (time_ms <= 150.0)

    X_stim_matched = np.empty((0, len(channel_ids)))
    y_stim_matched = np.empty(0)
    stim_tids_arr = np.empty(0)
    if matched_s_neural_idx:
        s_rates_sub = s_rates[:, matched_s_neural_idx, :]
        X_stim_matched = np.nanmean(s_rates_sub[:, :, n_mask], axis=2).T
        y_stim_matched = y_stim[matched_s_beh_idx]
        stim_tids_arr = np.asarray(matched_s_tids)

    X_ctrl_matched = np.empty((0, len(channel_ids)))
    y_ctrl_matched = np.empty(0)
    ctrl_tids_arr = np.empty(0)
    if matched_c_neural_idx:
        c_rates_sub = c_rates[:, matched_c_neural_idx, :]
        X_ctrl_matched = np.nanmean(c_rates_sub[:, :, n_mask], axis=2).T
        y_ctrl_matched = y_ctrl[matched_c_beh_idx]
        ctrl_tids_arr = np.asarray(matched_c_tids)

    # 4. Filter trials by analysis mode
    if analysis_mode == "stim_only":
        X_all = X_stim_matched
        y_all = y_stim_matched
        cond_labels_all = np.array(["stim"] * len(y_stim_matched))
        trial_ids_all = stim_tids_arr
    elif analysis_mode == "ctrl_only":
        X_all = X_ctrl_matched
        y_all = y_ctrl_matched
        cond_labels_all = np.array(["control"] * len(y_ctrl_matched))
        trial_ids_all = ctrl_tids_arr
    else:
        # 'within_condition_residual' or 'all_stacked_raw'
        if len(y_stim_matched) == 0 or len(y_ctrl_matched) == 0:
            _log(f"Inadequate matched trials across conditions (stim={len(y_stim_matched)}, ctrl={len(y_ctrl_matched)})")
            return None, None, None, None, {}
        X_all = np.vstack([X_stim_matched, X_ctrl_matched])
        y_all = np.concatenate([y_stim_matched, y_ctrl_matched])
        cond_labels_all = np.array(["stim"] * len(y_stim_matched) + ["control"] * len(y_ctrl_matched))
        trial_ids_all = np.concatenate([stim_tids_arr, ctrl_tids_arr])

    if len(y_all) < 4:
        _log(f"Insufficient total valid trials ({len(y_all)}) for regression.")
        return None, None, None, None, {}

    # 5. Filter out low-information channels (>50% NaNs)
    ch_finite_frac = np.mean(np.isfinite(X_all), axis=0)
    valid_ch_mask = ch_finite_frac > 0.5
    if not np.any(valid_ch_mask):
        _log("No channels with sufficient finite firing rate observations.")
        return None, None, None, None, {}

    X_all = X_all[:, valid_ch_mask]
    channel_ids = [ch for ch, keep in zip(channel_ids, valid_ch_mask) if keep]

    # Impute remaining sporadic NaNs with column means
    col_means = np.nanmean(X_all, axis=0)
    for c in range(X_all.shape[1]):
        nan_c = np.isnan(X_all[:, c])
        if np.any(nan_c):
            X_all[nan_c, c] = col_means[c] if np.isfinite(col_means[c]) else 0.0

    valid_trials = np.isfinite(y_all) & np.all(np.isfinite(X_all), axis=1)
    if np.sum(valid_trials) < 4:
        return None, None, None, None, {}

    X_neural = X_all[valid_trials]
    y_behavior = y_all[valid_trials]
    condition_labels = cond_labels_all[valid_trials]
    trial_ids = trial_ids_all[valid_trials]

    info = {
        "neural_window_ms": neural_window_ms,
        "behavior_window_ms": behavior_window_ms,
        "channel_ids": channel_ids,
        "n_stim_valid": int(np.sum(condition_labels == "stim")),
        "n_ctrl_valid": int(np.sum(condition_labels == "control")),
        "analysis_mode": analysis_mode,
        "behavior_metric": behavior_metric,
        "beh_meta": beh_meta,
        "s_rates_matched": s_rates[:, matched_s_neural_idx, :] if matched_s_neural_idx else None,
        "c_rates_matched": c_rates[:, matched_c_neural_idx, :] if matched_c_neural_idx else None,
        "y_stim_matched": y_stim_matched,
        "y_ctrl_matched": y_ctrl_matched,
        "matched_s_beh_idx": matched_s_beh_idx,
        "matched_c_beh_idx": matched_c_beh_idx,
    }

    return X_neural, y_behavior, condition_labels, trial_ids, info


# =============================================================================
# Nested Cross-Validation (No Data Leakage)
# =============================================================================

def fit_channel_regression(
    X_neural: np.ndarray,
    y_behavior: np.ndarray,
    cv_folds: int = DEFAULT_CV_FOLDS,
    alphas: Optional[np.ndarray] = None,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """
    Fits Ridge regression predicting behavior from neural activity using strict outer K-Fold
    cross-validation with StandardScaler fitted strictly inside training folds (zero data leakage).

    Evaluates:
        - Out-of-fold predictions y_pred_cv
        - Cross-validated R² and Pearson r
        - Mean channel weights beta_bar, cross-fold SD, and cross-fold sign consistency.
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    n_samples, n_channels = X_neural.shape
    k_outer = min(cv_folds, max(2, n_samples // 2))

    kf = KFold(n_splits=k_outer, shuffle=True, random_state=rng_seed)
    y_pred_cv = np.zeros(n_samples, dtype=float)
    fold_weights = []
    fold_alphas = []
    fold_r2 = []
    kf_splits = list(kf.split(X_neural))

    for tr_idx, te_idx in kf_splits:
        X_tr, X_te = X_neural[tr_idx], X_neural[te_idx]
        y_tr, y_te = y_behavior[tr_idx], y_behavior[te_idx]

        scaler_k = StandardScaler()
        X_tr_sc = np.nan_to_num(scaler_k.fit_transform(X_tr), nan=0.0)
        X_te_sc = np.nan_to_num(scaler_k.transform(X_te), nan=0.0)

        inner_k = min(5, max(2, len(tr_idx) // 2))
        inner_model = RidgeCV(alphas=alphas, cv=inner_k)
        inner_model.fit(X_tr_sc, y_tr)

        best_a = float(inner_model.alpha_)
        pred_te = inner_model.predict(X_te_sc)
        y_pred_cv[te_idx] = pred_te

        fold_weights.append(inner_model.coef_)
        fold_alphas.append(best_a)
        fold_r2.append(float(inner_model.score(X_te_sc, y_te)))

    # Compute fold stability metrics
    fold_weights_arr = np.array(fold_weights)  # (k_outer, n_channels)
    mean_weights = np.mean(fold_weights_arr, axis=0)
    weight_sd = np.std(fold_weights_arr, axis=0, ddof=1) if k_outer > 1 else np.zeros(n_channels)
    sign_consistency = np.mean(np.sign(fold_weights_arr) == np.sign(mean_weights)[None, :], axis=0)

    # Out-of-fold generalization metrics
    ss_res = np.sum((y_behavior - y_pred_cv) ** 2)
    ss_tot = np.sum((y_behavior - np.mean(y_behavior)) ** 2)
    cv_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    if np.std(y_pred_cv) > 1e-12 and np.std(y_behavior) > 1e-12:
        r_cv, p_pearson = stats.pearsonr(y_behavior, y_pred_cv)
    else:
        r_cv, p_pearson = 0.0, 1.0

    best_alpha = float(np.median(fold_alphas))

    # Full data fit for overall scaler and reference model
    full_scaler = StandardScaler()
    X_full_sc = np.nan_to_num(full_scaler.fit_transform(X_neural), nan=0.0)
    full_model = RidgeCV(alphas=alphas, cv=k_outer).fit(X_full_sc, y_behavior)

    return {
        "channel_weights": mean_weights,
        "weight_sd": weight_sd,
        "sign_consistency": sign_consistency,
        "full_weights": full_model.coef_,
        "cv_r2": float(cv_r2),
        "cv_pearson_r": float(r_cv),
        "cv_pearson_p": float(p_pearson),
        "best_alpha": best_alpha,
        "fold_alphas": fold_alphas,
        "fold_r2": fold_r2,
        "y_pred_cv": y_pred_cv,
        "scaler": full_scaler,
        "kf_splits": kf_splits,
    }


# =============================================================================
# Performance-Based Permutation Testing
# =============================================================================

def permutation_test_model(
    X_neural: np.ndarray,
    y_behavior: np.ndarray,
    condition_labels: np.ndarray,
    best_alpha: float,
    observed_r: float,
    observed_weights: np.ndarray,
    kf_splits: List[Tuple[np.ndarray, np.ndarray]],
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates out-of-fold decoder performance under condition-stratified target permutations.
    Precomputes fold projection operators for exact, vectorized, instantaneous permutation testing.

    Returns:
        p_model : empirical permutation p-value for decoder correlation (r_null >= r_obs)
        r_null : (n_permutations,) out-of-fold correlation null distribution
        p_values : (n_channels,) channel-level permutation p-values
        q_values : (n_channels,) FDR Benjamini-Hochberg adjusted q-values
        null_weights : (n_channels, n_permutations) null coefficient matrix
    """
    n_samples, n_channels = X_neural.shape

    # Precompute per-fold projection and weight operators
    fold_ops = []
    for tr_idx, te_idx in kf_splits:
        scaler_k = StandardScaler()
        X_tr_sc = np.nan_to_num(scaler_k.fit_transform(X_neural[tr_idx]), nan=0.0)
        X_te_sc = np.nan_to_num(scaler_k.transform(X_neural[te_idx]), nan=0.0)

        XtX_a = (X_tr_sc.T @ X_tr_sc) + best_alpha * np.eye(n_channels)
        try:
            W_k = np.linalg.solve(XtX_a, X_tr_sc.T)  # (n_channels, N_tr)
        except np.linalg.LinAlgError:
            W_k = np.linalg.pinv(XtX_a) @ X_tr_sc.T

        P_k = X_te_sc @ W_k  # (N_te, N_tr)
        fold_ops.append((tr_idx, te_idx, P_k, W_k))

    # Condition-stratified permutations of y
    rng = np.random.default_rng(rng_seed)
    unique_conds = np.unique(condition_labels)
    cond_indices = {c: np.where(condition_labels == c)[0] for c in unique_conds}

    Y_perm = np.zeros((n_samples, n_permutations), dtype=float)
    for b in range(n_permutations):
        perm_idx = np.empty(n_samples, dtype=int)
        for _, idxs in cond_indices.items():
            perm_idx[idxs] = rng.permutation(idxs)
        Y_perm[:, b] = y_behavior[perm_idx]

    # Vectorized out-of-fold predictions across all permutations
    Y_pred_cv_perm = np.zeros((n_samples, n_permutations), dtype=float)
    null_weights_sum = np.zeros((n_channels, n_permutations), dtype=float)

    for tr_idx, te_idx, P_k, W_k in fold_ops:
        Y_tr = Y_perm[tr_idx, :]  # (N_tr, B)
        Y_tr_mean = np.mean(Y_tr, axis=0, keepdims=True)
        Y_tr_center = Y_tr - Y_tr_mean

        Y_pred_cv_perm[te_idx, :] = (P_k @ Y_tr_center) + Y_tr_mean
        null_weights_sum += (W_k @ Y_tr_center)

    null_weights = null_weights_sum / float(len(fold_ops))

    # Vectorized Pearson r across permutations
    Y_c = Y_perm - np.mean(Y_perm, axis=0, keepdims=True)
    Y_hat_c = Y_pred_cv_perm - np.mean(Y_pred_cv_perm, axis=0, keepdims=True)
    cov = np.sum(Y_c * Y_hat_c, axis=0)
    sd_y = np.sqrt(np.sum(Y_c ** 2, axis=0))
    sd_y_hat = np.sqrt(np.sum(Y_hat_c ** 2, axis=0))

    with np.errstate(divide="ignore", invalid="ignore"):
        r_null = np.where((sd_y > 1e-12) & (sd_y_hat > 1e-12), cov / (sd_y * sd_y_hat), 0.0)

    # Model-level significance
    p_model = float(1.0 + np.sum(r_null >= observed_r)) / float(n_permutations + 1.0)

    # Channel-level significance
    abs_obs = np.abs(observed_weights)[:, None]
    abs_null = np.abs(null_weights)
    exceedances = np.sum(abs_null >= abs_obs, axis=1)
    p_values = (1.0 + exceedances.astype(float)) / float(n_permutations + 1.0)
    q_values = compute_benjamini_hochberg_fdr(p_values)

    return {
        "p_model": p_model,
        "r_null": r_null,
        "p_values": p_values,
        "q_values": q_values,
        "null_weights": null_weights,
    }


# =============================================================================
# Modular Visualizations
# =============================================================================

def draw_confidence_ellipse(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    n_std: float = 2.0,
    facecolor: str = "none",
    **kwargs,
) -> None:
    """Draw a 2D covariance confidence ellipse (default ~95% confidence, n_std=2.0)."""
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return
    x_v = x[valid]
    y_v = y[valid]
    cov = np.cov(x_v, y_v)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    if not np.all(np.isfinite(vals)) or np.any(vals < 1e-4):
        return
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    if not np.isfinite(theta):
        return
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-6))
    ellipse = matplotlib.patches.Ellipse(
        xy=(float(np.mean(x_v)), float(np.mean(y_v))),
        width=width,
        height=height,
        angle=theta,
        facecolor=facecolor,
        **kwargs,
    )
    ax.add_patch(ellipse)


def plot_behavioral_perturbation_summary(
    beh_stats: Dict[str, Any],
    condition_name: str = "",
    target: str = "",
    out_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (15.0, 11.0),
) -> plt.Figure:
    """
    Consolidated 4-panel publication-grade behavioral perturbation summary:
        Panel A: Overhead (X-Y) reach endpoints & perturbation vector field (Control vs Stim).
        Panel B: Directional perturbation components (ΔX lateral, ΔY forward, ΔZ vertical) with p-values.
        Panel C: 3D endpoint dispersion comparison and stimulation phenotype classification.
        Panel D: Reach plateau timing / duration distribution (Control vs Stim).
    """
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.26)

    eps_s = beh_stats.get("endpoints_s", np.empty((0, 3)))
    eps_c = beh_stats.get("endpoints_c", np.empty((0, 3)))
    c_mean = beh_stats.get("c_mean_endpoint", np.zeros(3))
    s_mean = beh_stats.get("s_mean_endpoint", np.zeros(3))
    s_deltas = beh_stats.get("endpoints_s_delta", np.empty((0, 3)))
    plateau_t_s = beh_stats.get("plateau_t_s", np.empty(0))
    plateau_t_c = beh_stats.get("plateau_t_c", np.empty(0))

    mean_dx = beh_stats.get("mean_delta_x", 0.0)
    mean_dy = beh_stats.get("mean_delta_y", 0.0)
    mean_dz = beh_stats.get("mean_delta_z", 0.0)
    mean_dnorm = beh_stats.get("mean_delta_norm", 0.0)
    p_dx = beh_stats.get("p_val_dx", 1.0)
    p_dy = beh_stats.get("p_val_dy", 1.0)
    p_dz = beh_stats.get("p_val_dz", 1.0)

    disp_s = beh_stats.get("disp_s_rms", 0.0)
    disp_c = beh_stats.get("disp_c_rms", 0.0)
    disp_ratio = beh_stats.get("disp_ratio", 1.0)
    p_disp = beh_stats.get("p_disp", 1.0)
    phenotype = beh_stats.get("phenotype", "Unknown")

    # ---------------------------------------------------------
    # Panel A: Overhead (X-Y) Endpoint Map & Perturbation Vectors
    # ---------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])

    # Plot Control and Stim single-trial endpoints
    if len(eps_c) > 0:
        ax_a.scatter(
            eps_c[:, 0], eps_c[:, 1],
            c="#2c3e50", s=45, alpha=0.65, edgecolors="#1a252f", linewidths=0.6,
            label=f"Control Endpoints (N={len(eps_c)})", zorder=3,
        )
        draw_confidence_ellipse(ax_a, eps_c[:, 0], eps_c[:, 1], n_std=2.0, edgecolor="#2c3e50", linestyle="--", linewidth=1.2, alpha=0.7)

    if len(eps_s) > 0:
        ax_a.scatter(
            eps_s[:, 0], eps_s[:, 1],
            c="#e74c3c", s=55, alpha=0.75, edgecolors="#901a1c", linewidths=0.7,
            label=f"Stim Endpoints (N={len(eps_s)})", zorder=4,
        )
        draw_confidence_ellipse(ax_a, eps_s[:, 0], eps_s[:, 1], n_std=2.0, edgecolor="#e74c3c", linestyle="--", linewidth=1.4, alpha=0.8)

        # Draw subtle arrows from control mean to each stim endpoint
        for i in range(len(eps_s)):
            if np.all(np.isfinite(eps_s[i, :2])) and np.all(np.isfinite(c_mean[:2])):
                ax_a.annotate(
                    "", xy=(eps_s[i, 0], eps_s[i, 1]), xytext=(c_mean[0], c_mean[1]),
                    arrowprops=dict(arrowstyle="->", color="#bdc3c7", lw=0.6, alpha=0.45),
                    zorder=2,
                )

    # Plot Mean Endpoints and Prominent Mean Perturbation Vector
    if np.all(np.isfinite(c_mean[:2])):
        ax_a.scatter(
            c_mean[0], c_mean[1], marker="D", s=130, c="#2c3e50", edgecolors="#ffffff", linewidths=1.5,
            label="Control Mean", zorder=6,
        )
    if np.all(np.isfinite(s_mean[:2])):
        ax_a.scatter(
            s_mean[0], s_mean[1], marker="*", s=180, c="#e74c3c", edgecolors="#ffffff", linewidths=1.5,
            label="Stim Mean", zorder=7,
        )

    if np.all(np.isfinite(c_mean[:2])) and np.all(np.isfinite(s_mean[:2])):
        ax_a.annotate(
            "", xy=(s_mean[0], s_mean[1]), xytext=(c_mean[0], c_mean[1]),
            arrowprops=dict(arrowstyle="-|>", color="#8e44ad", lw=2.6, mutation_scale=18),
            zorder=8,
        )

    ax_a.set_xlabel("Lateral Position X (mm, left < 0 < right)", fontsize=10, fontweight="medium")
    ax_a.set_ylabel("Forward Position Y (mm, reach extension)", fontsize=10, fontweight="medium")
    ax_a.set_title("A. 3D Reach Endpoints (Overhead X-Y) & Perturbation Vectors", fontsize=11, fontweight="bold", loc="left")
    ax_a.grid(linestyle="--", alpha=0.3)
    ax_a.legend(loc="upper left", fontsize=8, framealpha=0.85)

    info_text = (
        f"Mean Shift: ||Δe|| = {mean_dnorm:.1f} mm\n"
        f"  ΔX = {mean_dx:+.1f} mm (p={p_dx:.3f})\n"
        f"  ΔY = {mean_dy:+.1f} mm (p={p_dy:.3f})"
    )
    ax_a.text(
        0.96, 0.04, info_text, transform=ax_a.transAxes, fontsize=8,
        ha="right", va="bottom", bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#bdc3c7", alpha=0.9),
    )

    # ---------------------------------------------------------
    # Panel B: Directional Perturbations (ΔX, ΔY, ΔZ)
    # ---------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    axes_labels = ["Lateral (ΔX)", "Forward (ΔY)", "Height (ΔZ)"]
    p_vals = [p_dx, p_dy, p_dz]
    means = [mean_dx, mean_dy, mean_dz]

    if len(s_deltas) > 0:
        rng = np.random.default_rng(42)
        bar_colors = ["#3498db" if m < 0 else "#e74c3c" for m in means]
        for col_idx in range(min(3, s_deltas.shape[1])):
            vals = s_deltas[:, col_idx]
            vals = vals[np.isfinite(vals)]
            x_pos = col_idx
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax_b.scatter(x_pos + jitter, vals, color=bar_colors[col_idx], alpha=0.6, s=35, edgecolors="#333333", linewidths=0.5)

            # Plot mean +/- SEM
            if len(vals) > 0:
                m_val = float(np.mean(vals))
                sem_val = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
                ax_b.errorbar(
                    x_pos, m_val, yerr=sem_val, fmt="D", color="#111111", ecolor="#111111",
                    elinewidth=1.8, capsize=4, capthick=1.8, markersize=7, zorder=5,
                )

            # Significance annotations
            pval = p_vals[col_idx]
            star = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s."))
            y_top = float(np.nanmax(s_deltas[:, col_idx])) if len(vals) > 0 else 10.0
            ax_b.text(x_pos, y_top + 4.0, f"{star}\n(p={pval:.3f})", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax_b.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=1.1, alpha=0.8)
    ax_b.set_xticks([0, 1, 2])
    ax_b.set_xticklabels(axes_labels, fontsize=9, fontweight="medium")
    ax_b.set_ylabel("Perturbation Displacement from Control (mm)", fontsize=10, fontweight="medium")
    ax_b.set_title("B. Directional Perturbation Components (mm)", fontsize=11, fontweight="bold", loc="left")
    ax_b.grid(axis="y", linestyle="--", alpha=0.3)

    # ---------------------------------------------------------
    # Panel C: Endpoint Dispersion & Phenotype Classification
    # ---------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    bars = ax_c.bar(
        [0, 1], [disp_c, disp_s],
        color=["#2c3e50", "#e74c3c"], alpha=0.80, width=0.55, edgecolor="#333333", linewidth=1.0,
    )
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels([f"Control (N={len(eps_c)})", f"Stim (N={len(eps_s)})"], fontsize=9, fontweight="medium")
    ax_c.set_ylabel("3D Endpoint Dispersion (RMS mm)", fontsize=10, fontweight="medium")
    ax_c.set_title("C. Endpoint Dispersion & Stimulation Phenotype", fontsize=11, fontweight="bold", loc="left")
    ax_c.grid(axis="y", linestyle="--", alpha=0.3)

    # Annotate dispersion values above bars
    ax_c.text(0, disp_c + 1.5, f"{disp_c:.1f} mm", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_c.text(1, disp_s + 1.5, f"{disp_s:.1f} mm", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Phenotype badge box
    phenotype_colors = {
        "Directional Shift": "#2980b9",
        "Increased Variance": "#d35400",
        "Both (Directional Shift + Increased Variance)": "#c0392b",
        "No Significant Effect": "#7f8c8d",
    }
    badge_col = phenotype_colors.get(phenotype, "#7f8c8d")
    banner_text = (
        f"Dispersion Ratio: {disp_ratio:.2f}x (Levene p = {p_disp:.3f})\n"
        f"STIMULATION PHENOTYPE:\n{phenotype.upper()}"
    )
    ax_c.text(
        0.5, 0.93, banner_text, transform=ax_c.transAxes, fontsize=9, fontweight="bold",
        color="#ffffff", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=badge_col, edgecolor="none", alpha=0.92),
    )

    # ---------------------------------------------------------
    # Panel D: Plateau Timing / Movement Duration
    # ---------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    val_pts_c = plateau_t_c[np.isfinite(plateau_t_c)]
    val_pts_s = plateau_t_s[np.isfinite(plateau_t_s)]

    rng = np.random.default_rng(42)
    m_c, se_c = 0.0, 0.0
    m_s, se_s = 0.0, 0.0

    if len(val_pts_c) > 0:
        j_c = rng.uniform(-0.12, 0.12, size=len(val_pts_c))
        ax_d.scatter(0 + j_c, val_pts_c, color="#2c3e50", alpha=0.65, s=38, edgecolors="#111111", linewidths=0.6, label=f"Control (N={len(val_pts_c)})")
        m_c = float(np.mean(val_pts_c))
        se_c = float(np.std(val_pts_c, ddof=1) / math.sqrt(len(val_pts_c))) if len(val_pts_c) > 1 else 0.0
        ax_d.errorbar(0, m_c, yerr=se_c, fmt="D", color="#111111", ecolor="#111111", elinewidth=2.0, capsize=5, capthick=2.0, markersize=8, zorder=6)
        ax_d.text(0, m_c - (se_c + 15), f"{m_c:.0f} ms", ha="center", va="top", fontsize=9, fontweight="bold", color="#2c3e50")

    if len(val_pts_s) > 0:
        j_s = rng.uniform(-0.12, 0.12, size=len(val_pts_s))
        ax_d.scatter(1 + j_s, val_pts_s, color="#e74c3c", alpha=0.70, s=42, edgecolors="#111111", linewidths=0.6, label=f"Stim (N={len(val_pts_s)})")
        m_s = float(np.mean(val_pts_s))
        se_s = float(np.std(val_pts_s, ddof=1) / math.sqrt(len(val_pts_s))) if len(val_pts_s) > 1 else 0.0
        ax_d.errorbar(1, m_s, yerr=se_s, fmt="D", color="#111111", ecolor="#111111", elinewidth=2.0, capsize=5, capthick=2.0, markersize=8, zorder=6)
        ax_d.text(1, m_s + (se_s + 15), f"{m_s:.0f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#e74c3c")

    p_plat = beh_stats.get("p_plateau", 1.0)
    delta_plat = beh_stats.get("delta_plateau_t", 0.0)
    star_plat = "***" if p_plat < 0.001 else ("**" if p_plat < 0.01 else ("*" if p_plat < 0.05 else "n.s."))

    if len(val_pts_c) > 0 and len(val_pts_s) > 0:
        y_max = max(float(np.max(val_pts_c)), float(np.max(val_pts_s)))
        bracket_y = y_max + 40.0
        ax_d.plot([0, 0, 1, 1], [bracket_y - 12, bracket_y, bracket_y, bracket_y - 12], color="#333333", linewidth=1.2)
        ax_d.text(0.5, bracket_y + 6, f"{star_plat} (Δt = {delta_plat:+.0f} ms, p = {p_plat:.3f})", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        y_min = min(float(np.min(val_pts_c)), float(np.min(val_pts_s)))
        ax_d.set_ylim(max(0.0, y_min - 40.0), bracket_y + 45.0)

    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels(["Control", "Stim"], fontsize=10, fontweight="medium")
    ax_d.set_ylabel("Plateau Onset Time from Stim (ms)", fontsize=10, fontweight="medium")
    ax_d.set_title(f"D. Plateau Timing | Δt = {delta_plat:+.0f} ms (p = {p_plat:.3f})", fontsize=11, fontweight="bold", loc="left")
    ax_d.grid(axis="y", linestyle="--", alpha=0.3)
    ax_d.legend(loc="lower right", fontsize=8, framealpha=0.85)

    fig.suptitle(f"Behavioral Perturbation Analysis: {condition_name} ({target})", fontsize=13, fontweight="bold", y=0.98)

    if out_path:
        ensure_dir(out_path.parent)
        try:
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
        except Exception:
            fig.savefig(out_path, dpi=300)
            fig.savefig(out_path.with_suffix(".svg"))

    return fig


def plot_channel_weights(
    channel_weights: np.ndarray,
    channel_ids: Sequence[Any],
    p_values: np.ndarray,
    q_values: Optional[np.ndarray] = None,
    weight_sd: Optional[np.ndarray] = None,
    cv_r2: Optional[float] = None,
    cv_r: Optional[float] = None,
    p_model: Optional[float] = None,
    best_alpha: Optional[float] = None,
    array_name: str = "",
    condition_name: str = "",
    out_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (14.0, 5.5),
) -> plt.Figure:
    """
    Bar plot of regression weights across channels with cross-fold error bars and significance stars.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    ch_ids = np.asarray(channel_ids)
    n_ch = len(channel_weights)
    x_pos = np.arange(n_ch)

    colors = []
    alphas = []
    for w, p in zip(channel_weights, p_values):
        is_sig = p < P_VAL_THRESH
        colors.append("#d9383a" if w >= 0 else "#2b7bba")
        alphas.append(0.95 if is_sig else 0.35)

    yerr = weight_sd if weight_sd is not None else None
    bars = ax.bar(
        x_pos, channel_weights,
        yerr=yerr,
        capsize=2.5,
        error_kw=dict(ecolor="#555555", lw=0.8, capthick=0.8),
        color=colors, alpha=0.85, width=0.8,
    )

    for b, a in zip(bars, alphas):
        b.set_alpha(a)

    ax.axhline(0.0, color="#333333", linestyle="-", linewidth=1.0, alpha=0.7)

    y_max = np.nanmax(np.abs(channel_weights)) if n_ch > 0 else 1.0
    if weight_sd is not None:
        y_max = max(y_max, float(np.nanmax(np.abs(channel_weights) + weight_sd)))
    offset = max(y_max * 0.05, 0.005)

    n_sig_p = int(np.sum(p_values < P_VAL_THRESH))
    n_sig_q = int(np.sum(q_values < FDR_ALPHA)) if q_values is not None else 0

    for i, (w, p) in enumerate(zip(channel_weights, p_values)):
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        if star:
            sd_i = weight_sd[i] if weight_sd is not None else 0.0
            y_annot = (w + sd_i + offset) if w >= 0 else (w - sd_i - offset * 2.2)
            va = "bottom" if w >= 0 else "top"
            ax.text(x_pos[i], y_annot, star, ha="center", va=va, fontsize=9, fontweight="bold", color="#111111")

    step = max(1, n_ch // 32)
    ax.set_xticks(x_pos[::step])
    ax.set_xticklabels([str(ch_ids[i]) for i in range(0, n_ch, step)], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Electrode / Channel ID", fontsize=11, fontweight="medium")
    ax.set_ylabel("Standardized Weight (β ± SD)", fontsize=11, fontweight="medium")

    stat_parts = []
    if cv_r is not None:
        stat_parts.append(f"CV r = {cv_r:.3f}")
    if cv_r2 is not None:
        stat_parts.append(f"CV R² = {cv_r2:.3f}")
    if p_model is not None:
        stat_parts.append(f"p_model = {p_model:.3f}")
    if best_alpha is not None:
        stat_parts.append(f"α = {best_alpha:.2g}")
    stat_str = " | ".join(stat_parts)

    ax.set_title(
        f"{array_name} Channel Weights → Behavior ({condition_name})\n"
        f"Sig: {n_sig_p}/{n_ch} (p < 0.05), {n_sig_q}/{n_ch} (FDR q < 0.05) | {stat_str}",
        fontsize=12, fontweight="bold", pad=12,
    )

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
    cv_r: Optional[float] = None,
    cv_r2: Optional[float] = None,
    p_model: Optional[float] = None,
    array_name: str = "",
    condition_name: str = "",
    target_label: str = "Perturbation Score",
    out_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (7.5, 6.5),
) -> plt.Figure:
    """
    Scatter plot of out-of-fold predicted vs observed behavioral perturbation, colored by condition.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    cond_arr = np.asarray(condition_labels)
    is_stim = cond_arr == "stim"
    is_ctrl = cond_arr == "control"

    if np.any(is_ctrl):
        ax.scatter(
            y_true[is_ctrl], y_pred[is_ctrl],
            c="#2c3e50", s=45, alpha=0.80, edgecolors="#1a252f", linewidths=0.6,
            label=f"Control ({int(np.sum(is_ctrl))} trials)", zorder=3,
        )
    if np.any(is_stim):
        ax.scatter(
            y_true[is_stim], y_pred[is_stim],
            c="#e74c3c", s=50, alpha=0.85, edgecolors="#901a1c", linewidths=0.6,
            label=f"Stim ({int(np.sum(is_stim))} trials)", zorder=4,
        )

    min_val = min(float(np.nanmin(y_true)), float(np.nanmin(y_pred)))
    max_val = max(float(np.nanmax(y_true)), float(np.nanmax(y_pred)))
    pad = max((max_val - min_val) * 0.08, 0.1)
    lims = (min_val - pad, max_val + pad)
    ax.plot(lims, lims, color="#7f8c8d", linestyle="--", linewidth=1.2, label="Identity (y = x)", zorder=1)

    r_val, p_val = np.nan, np.nan
    if len(y_true) > 2 and np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred)):
        if np.std(y_true) > 1e-6 and np.std(y_pred) > 1e-6:
            try:
                slope, intercept, r_val, p_val, _ = stats.linregress(y_true, y_pred)
                x_trend = np.linspace(min_val, max_val, 100)
                ax.plot(x_trend, slope * x_trend + intercept, color="#2980b9", linestyle="-", linewidth=1.8, label=f"Fit (r = {r_val:.2f})", zorder=2)
            except Exception:
                pass

    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel(f"Observed {target_label}", fontsize=11, fontweight="medium")
    ax.set_ylabel(f"Cross-Validated Predicted {target_label}", fontsize=11, fontweight="medium")

    r_disp = cv_r if cv_r is not None else r_val
    r2_disp = cv_r2 if cv_r2 is not None else (r_val ** 2 if np.isfinite(r_val) else np.nan)
    stat_lines = [
        f"Cross-Val R² = {r2_disp:.3f}",
        f"Pearson r = {r_disp:.3f}",
    ]
    if p_model is not None:
        stat_lines.append(f"Permutation p = {p_model:.3f}")
    stat_lines.append(f"Total Trials = {len(y_true)}")

    ax.text(
        0.05, 0.95, "\n".join(stat_lines),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#bdc3c7", alpha=0.9),
    )

    ax.set_title(
        f"{array_name} Perturbation Prediction Performance\n{condition_name}",
        fontsize=12, fontweight="bold", pad=10,
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


def plot_decoder_null_distribution(
    r_observed: float,
    r_null: np.ndarray,
    p_model: float,
    array_name: str = "",
    condition_name: str = "",
    out_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (7.0, 5.5),
) -> plt.Figure:
    """
    Histogram of out-of-fold Pearson r under the permutation null hypothesis vs observed decoder correlation.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Histogram of null distribution
    bins = np.linspace(min(-0.6, float(np.min(r_null))), max(0.8, float(np.max(r_null)), r_observed + 0.1), 35)
    ax.hist(r_null, bins=bins, color="#95a5a6", alpha=0.75, edgecolor="#7f8c8d", density=True, label=f"Null Distribution (B={len(r_null)})")

    # 95th percentile cutoff
    p95 = float(np.percentile(r_null, 95))
    ax.axvline(p95, color="#e67e22", linestyle="--", linewidth=1.5, label=f"Null 95% Cutoff ({p95:.2f})")

    # Observed correlation line
    obs_color = "#e74c3c" if p_model < 0.05 else "#2980b9"
    ax.axvline(r_observed, color=obs_color, linestyle="-", linewidth=2.2, label=f"Observed r = {r_observed:.3f}")

    ax.set_xlabel("Cross-Validated Pearson r", fontsize=11, fontweight="medium")
    ax.set_ylabel("Probability Density", fontsize=11, fontweight="medium")

    sig_status = "Significant (p < 0.05)" if p_model < 0.05 else "Not Significant (p ≥ 0.05)"
    box_text = (
        f"Observed r = {r_observed:.3f}\n"
        f"Null 95% limit = {p95:.3f}\n"
        f"Empirical p = {p_model:.3f}\n"
        f"Status: {sig_status}"
    )
    ax.text(
        0.05, 0.95, box_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#bdc3c7", alpha=0.9),
    )

    ax.set_title(
        f"{array_name} Decoder Permutation Test\n{condition_name}",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
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
    Time-resolved trial-averaged firing rate traces (mean +/- SEM) for top predictive channels.
    """
    n_ch = len(channel_weights)
    if n_ch == 0:
        return None

    pos_order = np.argsort(channel_weights)[::-1]
    neg_order = np.argsort(channel_weights)

    selected_indices = []
    for idx in pos_order:
        if len(selected_indices) >= top_n or channel_weights[idx] <= 0:
            break
        selected_indices.append(idx)
    for idx in neg_order:
        if len(selected_indices) >= top_n * 2 or channel_weights[idx] >= 0:
            break
        if idx not in selected_indices:
            selected_indices.append(idx)

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

        s_trace = s_rates[ch_idx, :, :]
        s_mean = np.nanmean(s_trace, axis=0)
        s_sem = np.nanstd(s_trace, axis=0) / math.sqrt(max(1, s_trace.shape[0]))

        c_trace = c_rates[ch_idx, :, :]
        c_mean = np.nanmean(c_trace, axis=0)
        c_sem = np.nanstd(c_trace, axis=0) / math.sqrt(max(1, c_trace.shape[0]))

        ax.fill_between(time_ms, c_mean - c_sem, c_mean + c_sem, color="#2c3e50", alpha=0.18, edgecolor="none")
        ax.plot(time_ms, c_mean, color="#2c3e50", linewidth=1.5, label="Control")

        ax.fill_between(time_ms, s_mean - s_sem, s_mean + s_sem, color="#e74c3c", alpha=0.22, edgecolor="none")
        ax.plot(time_ms, s_mean, color="#e74c3c", linewidth=1.8, label="Stim")

        ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0, alpha=0.7)
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
    weight_sd: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    condition_labels: Sequence[str],
    r_null: np.ndarray,
    p_model: float,
    s_rates: np.ndarray,
    c_rates: np.ndarray,
    time_ms: np.ndarray,
    cv_r2: float,
    cv_r: float,
    best_alpha: float,
    array_name: str,
    condition_name: str,
    neural_window_ms: Tuple[float, float] = DEFAULT_NEURAL_WINDOW_MS,
    target_label: str = "Perturbation Score",
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Consolidated 4-panel publication figure in a 2x2 grid:
    Panel A: Channel weights bar chart with cross-fold error bars and significance stars.
    Panel B: Observed vs cross-validated predicted behavior scatter plot.
    Panel C: Model-level permutation null distribution of r_null vs r_obs.
    Panel D: Top predictive channels firing rate PSTHs (Stim vs Control).
    """
    fig = plt.figure(figsize=(15.0, 11.0), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=0.32, wspace=0.25)

    ch_ids = np.asarray(channel_ids)
    n_ch = len(channel_weights)
    x_pos = np.arange(n_ch)

    # ---------------------------------------------------------
    # Panel A: Channel Weights
    # ---------------------------------------------------------
    ax_weights = fig.add_subplot(gs[0, 0])
    colors = ["#d9383a" if w >= 0 else "#2b7bba" for w in channel_weights]
    bars = ax_weights.bar(
        x_pos, channel_weights,
        yerr=weight_sd,
        capsize=2.0,
        error_kw=dict(ecolor="#555555", lw=0.7, capthick=0.7),
        color=colors, alpha=0.85, width=0.8,
    )
    for b, p in zip(bars, p_values):
        b.set_alpha(0.95 if p < P_VAL_THRESH else 0.35)

    ax_weights.axhline(0.0, color="#333333", linestyle="-", linewidth=0.9, alpha=0.7)

    y_max = np.nanmax(np.abs(channel_weights)) if n_ch > 0 else 1.0
    if weight_sd is not None:
        y_max = max(y_max, float(np.nanmax(np.abs(channel_weights) + weight_sd)))
    offset = max(y_max * 0.05, 0.005)

    for i, (w, p) in enumerate(zip(channel_weights, p_values)):
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        if star:
            sd_i = weight_sd[i] if weight_sd is not None else 0.0
            y_annot = (w + sd_i + offset) if w >= 0 else (w - sd_i - offset * 2.2)
            va = "bottom" if w >= 0 else "top"
            ax_weights.text(x_pos[i], y_annot, star, ha="center", va=va, fontsize=8, fontweight="bold")

    step = max(1, n_ch // 20)
    ax_weights.set_xticks(x_pos[::step])
    ax_weights.set_xticklabels([str(ch_ids[i]) for i in range(0, n_ch, step)], rotation=45, ha="right", fontsize=8)
    ax_weights.set_xlabel("Electrode / Channel ID", fontsize=10)
    ax_weights.set_ylabel("Standardized Weight (β ± SD)", fontsize=10)
    n_sig_p = int(np.sum(p_values < P_VAL_THRESH))
    n_sig_q = int(np.sum(q_values < FDR_ALPHA))
    ax_weights.set_title(
        f"A. Channel Weights | Sig: {n_sig_p}/{n_ch} (p<0.05), {n_sig_q}/{n_ch} (FDR q<0.05)",
        fontsize=11, fontweight="bold", loc="left",
    )
    ax_weights.grid(axis="y", linestyle="--", alpha=0.3)

    # ---------------------------------------------------------
    # Panel B: Prediction Scatter
    # ---------------------------------------------------------
    ax_scatter = fig.add_subplot(gs[0, 1])
    cond_arr = np.asarray(condition_labels)
    is_stim = cond_arr == "stim"
    is_ctrl = cond_arr == "control"
    if np.any(is_ctrl):
        ax_scatter.scatter(y_true[is_ctrl], y_pred[is_ctrl], c="#2c3e50", s=38, alpha=0.8, edgecolors="none", label=f"Control (N={int(np.sum(is_ctrl))})")
    if np.any(is_stim):
        ax_scatter.scatter(y_true[is_stim], y_pred[is_stim], c="#e74c3c", s=44, alpha=0.85, edgecolors="none", label=f"Stim (N={int(np.sum(is_stim))})")

    min_val = min(float(np.nanmin(y_true)), float(np.nanmin(y_pred)))
    max_val = max(float(np.nanmax(y_true)), float(np.nanmax(y_pred)))
    pad = max((max_val - min_val) * 0.08, 0.1)
    lims = (min_val - pad, max_val + pad)
    ax_scatter.plot(lims, lims, color="#7f8c8d", linestyle="--", linewidth=1.1, label="Unity (y=x)")

    if len(y_true) > 2 and np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred)):
        if np.std(y_true) > 1e-6 and np.std(y_pred) > 1e-6:
            try:
                slope, intercept, r_val, p_val, _ = stats.linregress(y_true, y_pred)
                x_trend = np.linspace(min_val, max_val, 100)
                ax_scatter.plot(x_trend, slope * x_trend + intercept, color="#2980b9", linestyle="-", linewidth=1.6, label=f"Fit (r={r_val:.2f})")
            except Exception:
                pass

    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)
    ax_scatter.set_xlabel(f"Observed {target_label}", fontsize=10)
    ax_scatter.set_ylabel(f"CV Predicted {target_label}", fontsize=10)
    ax_scatter.set_title(f"B. Prediction Quality | CV R² = {cv_r2:.3f}, r = {cv_r:.2f} (p_model = {p_model:.3f})", fontsize=11, fontweight="bold", loc="left")
    ax_scatter.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax_scatter.grid(linestyle="--", alpha=0.3)

    # ---------------------------------------------------------
    # Panel C: Model Permutation Null Distribution
    # ---------------------------------------------------------
    ax_null = fig.add_subplot(gs[1, 0])
    bins = np.linspace(min(-0.6, float(np.min(r_null))), max(0.8, float(np.max(r_null)), cv_r + 0.1), 30)
    ax_null.hist(r_null, bins=bins, color="#95a5a6", alpha=0.75, edgecolor="#7f8c8d", density=True, label="Null (B=1000)")
    p95 = float(np.percentile(r_null, 95))
    ax_null.axvline(p95, color="#e67e22", linestyle="--", linewidth=1.4, label=f"Null 95% Cutoff ({p95:.2f})")
    obs_col = "#e74c3c" if p_model < 0.05 else "#2980b9"
    ax_null.axvline(cv_r, color=obs_col, linestyle="-", linewidth=2.0, label=f"Observed r = {cv_r:.3f}")
    ax_null.set_xlabel("Cross-Validated Pearson r", fontsize=10)
    ax_null.set_ylabel("Probability Density", fontsize=10)
    sig_str = "Significant (p < 0.05)" if p_model < 0.05 else "Not Significant"
    ax_null.set_title(f"C. Model Permutation Test | p_model = {p_model:.3f} ({sig_str})", fontsize=11, fontweight="bold", loc="left")
    ax_null.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax_null.grid(linestyle="--", alpha=0.3)

    # ---------------------------------------------------------
    # Panel D: Top Channel PSTH Traces
    # ---------------------------------------------------------
    ax_traces = fig.add_subplot(gs[1, 1])
    top_pos_idx = int(np.argmax(channel_weights))

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
        f"D. Top Predictive Channel {ch_ids[top_pos_idx]} (β = {channel_weights[top_pos_idx]:+.2f}, p = {p_values[top_pos_idx]:.3f})",
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


def plot_spatiotemporal_projection_map(
    s_rates: Optional[np.ndarray],
    c_rates: Optional[np.ndarray],
    time_ms: Optional[np.ndarray],
    y_stim: Optional[np.ndarray],
    y_ctrl: Optional[np.ndarray],
    channel_ids: Sequence[Any],
    array_name: str = "NPRW",
    condition_name: str = "",
    target_label: str = "Behavioral Perturbation",
    stim_pulse_ms: float = 100.0,
    out_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (12.0, 8.0),
) -> Optional[plt.Figure]:
    """
    Spatiotemporal Neural-Behavior Projection Map (Channels x Time Bins).

    Scientific Grounding:
        Instead of collapsing neural activity into a single arbitrary scalar time window,
        this visualization computes the single-trial projection/correlation r(c, t) between
        firing rate R_i(c, t) and the single-trial reach perturbation metric y_i across every
        electrode channel c and every 20-ms time bin t.

    Layout:
        - Top Panel: Cross-Validated Decoding Timecourse r_cv(t) from Ridge regression.
        - Main Panel: 2D Heatmap (Channels x Time Bins) with diverging colormap (RdBu_r).
        - Right Panel: Channel Marginal Tuning Profile (Mean |r| and Peak Signed r along probe).
        - Overlay: Significance contour (p < 0.05) and stimulation pulse shading.
    """
    if s_rates is None or c_rates is None or time_ms is None or y_stim is None or y_ctrl is None:
        return None

    y_s = np.asarray(y_stim, dtype=float)
    y_c = np.asarray(y_ctrl, dtype=float)

    is_time_resolved = (y_s.ndim == 2)
    if is_time_resolved:
        valid_s = np.any(np.isfinite(y_s), axis=1)
        valid_c = np.any(np.isfinite(y_c), axis=1)
    else:
        valid_s = np.isfinite(y_s)
        valid_c = np.isfinite(y_c)

    if np.sum(valid_s) == 0 and np.sum(valid_c) == 0:
        return None

    # Slice rates to valid behavioral trials
    s_rates_sub = s_rates[:, valid_s, :] if np.any(valid_s) and s_rates.shape[1] >= len(y_s) else s_rates
    c_rates_sub = c_rates[:, valid_c, :] if np.any(valid_c) and c_rates.shape[1] >= len(y_c) else c_rates
    if is_time_resolved:
        y_s_sub = y_s[valid_s, :] if np.any(valid_s) and len(y_s) == s_rates.shape[1] else y_s
        y_c_sub = y_c[valid_c, :] if np.any(valid_c) and len(y_c) == c_rates.shape[1] else y_c
    else:
        y_s_sub = y_s[valid_s] if np.any(valid_s) and len(y_s) == s_rates.shape[1] else y_s
        y_c_sub = y_c[valid_c] if np.any(valid_c) and len(y_c) == c_rates.shape[1] else y_c

    min_t = min(s_rates_sub.shape[2], c_rates_sub.shape[2], len(time_ms))
    if is_time_resolved:
        min_t = min(min_t, y_s_sub.shape[1], y_c_sub.shape[1])
    if min_t < 3:
        return None

    rates_s_cut = s_rates_sub[:, :, :min_t]
    rates_c_cut = c_rates_sub[:, :, :min_t]
    time_vec = time_ms[:min_t]

    rates_all = np.concatenate([rates_s_cut, rates_c_cut], axis=1)  # (n_ch, n_trials, n_bins)
    if is_time_resolved:
        y_s_cut = y_s_sub[:, :min_t]
        y_c_cut = y_c_sub[:, :min_t]
        y_all = np.concatenate([y_s_cut, y_c_cut], axis=0)  # (n_trials, n_bins)
    else:
        y_all = np.concatenate([y_s_sub, y_c_sub])  # (n_trials,)
    n_ch, n_trials, n_bins = rates_all.shape

    if n_trials < 4 or n_ch == 0:
        return None

    # Compute Channel x Time Pearson correlation & p-value map
    corr_map = np.zeros((n_ch, n_bins), dtype=float)
    pval_map = np.ones((n_ch, n_bins), dtype=float)

    for c in range(n_ch):
        for t in range(n_bins):
            r_tr = rates_all[c, :, t]
            y_tr = y_all[:, t] if is_time_resolved else y_all
            valid_pts = np.isfinite(r_tr) & np.isfinite(y_tr)
            if np.sum(valid_pts) >= 4 and np.std(r_tr[valid_pts]) > 1e-6 and np.std(y_tr[valid_pts]) > 1e-6:
                r_val, p_val = stats.pearsonr(r_tr[valid_pts], y_tr[valid_pts])
                corr_map[c, t] = r_val if np.isfinite(r_val) else 0.0
                pval_map[c, t] = p_val if np.isfinite(p_val) else 1.0

    # Compute Time-Resolved Ridge CV Decoder Performance r_cv(t)
    cv_r_by_time = np.zeros(n_bins, dtype=float)
    for t in range(n_bins):
        X_t = rates_all[:, :, t].T  # (n_trials, n_ch)
        y_t = y_all[:, t] if is_time_resolved else y_all
        valid_y = np.isfinite(y_t)
        if np.sum(valid_y) < 4 or np.std(y_t[valid_y]) < 1e-6:
            continue

        valid_idx = np.where(valid_y)[0]
        col_m = np.nanmean(X_t, axis=0)
        for ch in range(n_ch):
            nan_m = np.isnan(X_t[:, ch])
            if np.any(nan_m):
                X_t[nan_m, ch] = col_m[ch] if np.isfinite(col_m[ch]) else 0.0

        y_pred_t = np.zeros(len(valid_idx), dtype=float)
        kf_t = KFold(n_splits=min(5, max(2, len(valid_idx))), shuffle=True, random_state=42)
        X_sub = X_t[valid_idx]
        y_sub = y_t[valid_idx]
        for tr_sub, val_sub in kf_t.split(X_sub):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_sub[tr_sub])
            X_val = scaler.transform(X_sub[val_sub])
            ridge = RidgeCV(alphas=np.logspace(-2, 4, 15))
            ridge.fit(X_tr, y_sub[tr_sub])
            y_pred_t[val_sub] = ridge.predict(X_val)

        if np.std(y_pred_t) > 1e-6 and np.std(y_sub) > 1e-6:
            r_cv, _ = stats.pearsonr(y_sub, y_pred_t)
            cv_r_by_time[t] = r_cv if np.isfinite(r_cv) else 0.0

    # Build Publication Figure
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[5.2, 1.1], height_ratios=[1.2, 4.0], hspace=0.08, wspace=0.06)

    # 1. Top Panel: Timecourse of Decoding Correlation r_cv(t)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.plot(time_vec, cv_r_by_time, color="#2980b9", linewidth=2.0, label="Cross-Validated r(t)")
    ax_top.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_top.axvline(0.0, color="#222222", linestyle="--", linewidth=1.2, alpha=0.8)
    pulse_w = max(10.0, float(stim_pulse_ms))
    ax_top.axvspan(0.0, pulse_w, color="#e74c3c", alpha=0.15, label=f"Stim Window ({pulse_w:.0f} ms)")

    best_t_idx = int(np.argmax(cv_r_by_time))
    best_t = time_vec[best_t_idx]
    best_r = cv_r_by_time[best_t_idx]
    ax_top.plot(best_t, best_r, "o", color="#e74c3c", markersize=6)
    annot_y_offset = 0.06 if best_r < 0.6 else -0.10
    ax_top.annotate(
        f"Peak r = {best_r:+.2f} ({best_t:+.0f} ms)",
        xy=(best_t, best_r), xytext=(best_t + 35, best_r + annot_y_offset),
        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2),
        fontsize=9, fontweight="bold", color="#c0392b",
    )

    ax_top.set_xlim(time_vec[0], time_vec[-1])
    y_min_top = min(-0.6, float(np.min(cv_r_by_time)) - 0.1)
    y_max_top = max(0.4, float(np.max(cv_r_by_time)) + 0.15)
    ax_top.set_ylim(y_min_top, y_max_top)
    ax_top.set_ylabel("CV Pearson r", fontsize=10, fontweight="medium")
    ax_top.set_title(f"A. Time-Resolved Behavioral Decoding Performance ({array_name} {n_ch} Channels)", fontsize=11, fontweight="bold", loc="left")
    ax_top.set_xticklabels([])
    ax_top.grid(linestyle="--", alpha=0.3)
    ax_top.legend(loc="upper left", fontsize=8, framealpha=0.85)

    # 2. Main Heatmap: Channels x Time Bins
    ax_main = fig.add_subplot(gs[1, 0])
    vmax = max(0.35, float(np.percentile(np.abs(corr_map), 99)))
    im = ax_main.imshow(
        corr_map, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        extent=[time_vec[0], time_vec[-1], n_ch + 0.5, 0.5], interpolation="nearest"
    )

    # Significance contours (p < 0.05)
    sig_mask = pval_map < 0.05
    if np.any(sig_mask):
        try:
            y_grid = np.arange(1, n_ch + 1)
            ax_main.contour(time_vec, y_grid, pval_map, levels=[0.05], colors="#2c3e50", linewidths=0.7, linestyles="-")
        except Exception:
            pass

    ax_main.axvline(0.0, color="#222222", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_main.axvspan(0.0, pulse_w, color="#e74c3c", alpha=0.15)
    ax_main.set_xlabel("Time from Stim Onset (ms)", fontsize=11, fontweight="medium")
    ax_main.set_ylabel(f"{array_name} Electrode / Channel ID (Tip to Base)", fontsize=11, fontweight="medium")

    y_step = max(1, n_ch // 8)
    y_ticks = np.arange(1, n_ch + 1, y_step)
    ax_main.set_yticks(y_ticks)
    ch_labels = [str(channel_ids[i - 1]) if (i - 1) < len(channel_ids) else str(i) for i in y_ticks]
    ax_main.set_yticklabels(ch_labels, fontsize=9)

    # 3. Right Marginal Panel: Channel Tuning Magnitude Profile
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    mean_abs_r = np.nanmean(np.abs(corr_map), axis=1)
    peak_r = np.array([corr_map[c, int(np.argmax(np.abs(corr_map[c])))] for c in range(n_ch)])
    y_ch_idx = np.arange(1, n_ch + 1)
    ax_right.plot(mean_abs_r, y_ch_idx, color="#2c3e50", linewidth=1.3, label="Mean |r|")
    ax_right.plot(peak_r, y_ch_idx, color="#e74c3c", linewidth=1.0, alpha=0.7, label="Peak Signed r")
    ax_right.axvline(0.0, color="#7f8c8d", linestyle="--", linewidth=0.8)
    ax_right.set_xlabel("Tuning Magnitude", fontsize=9, fontweight="medium")
    ax_right.set_xlim(-0.6, 0.6)
    ax_right.grid(linestyle="--", alpha=0.3)
    ax_right.legend(loc="lower right", fontsize=7.5, framealpha=0.85)

    # 4. Colorbar in the top-right slot
    ax_cb = fig.add_subplot(gs[0, 1])
    ax_cb.axis("off")
    cbar = fig.colorbar(im, ax=ax_cb, orientation="horizontal", fraction=0.7, pad=0.15)
    cbar.set_label(f"Correlation with {target_label}", fontsize=8.5, fontweight="medium")

    fig.suptitle(f"Spatiotemporal Neural-Behavior Projection Map: {condition_name} ({array_name})", fontsize=13, fontweight="bold", y=0.98)

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
    analysis_mode: str = DEFAULT_ANALYSIS_MODE,
    behavior_metric: str = DEFAULT_BEHAVIOR_METRIC,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    keypoint: str = DEFAULT_KEYPOINT,
    square_size_mm: float = SQUARE_SIZE_MM,
    align_task_frame: bool = ALIGN_TO_TASK_FRAME,
    dlc_3d_dir: Optional[Path] = None,
    file_idx: Optional[int] = None,
    total_files: Optional[int] = None,
) -> Dict[str, Any]:
    prefix = f"[{file_idx}/{total_files}] " if file_idx and total_files else ""
    _log(f"{prefix}Processing condition: {stim_path.name}")
    ensure_dir(out_dir)

    condition_name = stim_path.stem
    target = get_target_folder(stim_path)
    session_name = resolve_session_name_from_path(stim_path)

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

    # 2. Attempt loading calibrated 3D task-frame kinematics
    stim_beh_3d = load_condition_3d_task_kinematics(
        npz_path=stim_path,
        session=session_name,
        target="B" if "target_b" in target.lower() else "A",
        keypoint=keypoint,
        square_size_mm=square_size_mm,
        align_task_frame=align_task_frame,
        dlc_3d_dir=dlc_3d_dir,
    )
    ctrl_beh_3d = None
    if ctrl_path and ctrl_path.exists():
        ctrl_beh_3d = load_condition_3d_task_kinematics(
            npz_path=ctrl_path,
            session=session_name,
            target="B" if "target_b" in target.lower() else "A",
            keypoint=keypoint,
            square_size_mm=square_size_mm,
            align_task_frame=align_task_frame,
            dlc_3d_dir=dlc_3d_dir,
        )

    if stim_beh_3d is not None and ctrl_beh_3d is not None:
        kinematics_source = "3D_DLC_task_frame"
        stim_beh_dict = stim_beh_3d
        ctrl_beh_dict = ctrl_beh_3d
        _log(f"  Using 3D DLC task-frame kinematics ({keypoint}, {square_size_mm:.1f} mm/unit)")
    else:
        kinematics_source = "2D_camera_pixel"
        _log("  Falling back to 2D camera kinematics (cam0 + cam1)")
        # Construct 2D coordinate dictionary
        s_xy0 = extract_xy_coordinates(stim_cam0_segs, stim_cam0_names, ("middle", "wrist", "hand", "index"))
        s_xy1 = extract_xy_coordinates(stim_cam1_segs, stim_cam1_names, ("middle", "wrist", "hand", "index"))
        c_xy0 = extract_xy_coordinates(ctrl_cam0_segs, ctrl_cam0_names, ("middle", "wrist", "hand", "index"))
        c_xy1 = extract_xy_coordinates(ctrl_cam1_segs, ctrl_cam1_names, ("middle", "wrist", "hand", "index"))

        s_coords_list = []
        c_coords_list = []
        c_names = []
        if s_xy0 is not None and c_xy0 is not None:
            s_coords_list.extend([s_xy0[0], s_xy0[1]])
            c_coords_list.extend([c_xy0[0], c_xy0[1]])
            c_names.extend(["cam0_x", "cam0_y"])
        if s_xy1 is not None and c_xy1 is not None:
            s_coords_list.extend([s_xy1[0], s_xy1[1]])
            c_coords_list.extend([c_xy1[0], c_xy1[1]])
            c_names.extend(["cam1_x", "cam1_y"])

        if s_coords_list and c_coords_list:
            stim_beh_dict = {
                "beh_t": stim_beh_t,
                "coords": np.stack(s_coords_list, axis=1),
                "valid_mask": np.ones(s_coords_list[0].shape[0], dtype=bool),
                "raw_trial_indices": stim_trial_ids,
                "is_3d": False,
                "coord_names": c_names,
            }
            ctrl_beh_dict = {
                "beh_t": ctrl_beh_t if ctrl_beh_t is not None else stim_beh_t,
                "coords": np.stack(c_coords_list, axis=1),
                "valid_mask": np.ones(c_coords_list[0].shape[0], dtype=bool),
                "raw_trial_indices": ctrl_trial_ids,
                "is_3d": False,
                "coord_names": c_names,
            }
        else:
            stim_beh_dict = {}
            ctrl_beh_dict = {}

    # Ingest precomputed plateau detection table from plot_plateau_analysis.py if available
    df_plat = load_plateau_analysis_table(session=session_name, peri_root=stim_path.parent)
    if df_plat is not None and stim_beh_dict and ctrl_beh_dict:
        stim_br = extract_br_index(stim_path)
        ctrl_br = extract_br_index(ctrl_path) if ctrl_path else -1
        target_str = "B" if "target_b" in target.lower() else "A"
        n_s_trials = stim_beh_dict["coords"].shape[0] if "coords" in stim_beh_dict else 0
        n_c_trials = ctrl_beh_dict["coords"].shape[0] if "coords" in ctrl_beh_dict else 0
        s_plat_override = extract_plateau_times_from_table(df_plat, stim_br, target_str, n_s_trials)
        c_plat_override = extract_plateau_times_from_table(df_plat, ctrl_br, target_str, n_c_trials)
        if s_plat_override is not None:
            stim_beh_dict["override_plateau_times"] = s_plat_override
        if c_plat_override is not None:
            ctrl_beh_dict["override_plateau_times"] = c_plat_override

    elec_info, _ = load_electrode_mapping(PARAMS.monkey if PARAMS is not None else None)
    bad_nprw = bad_channels.get("NPRW", set()) if bad_channels else set()
    bad_ua = bad_channels.get("UA", set()) if bad_channels else set()

    # 3. Extract array modalities
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
                array_data_list.append(("NPRW", s_nprw_3d[valid_idxs, :, :], nprw_t, c_nprw_3d[valid_idxs, :, :], ctrl_nprw_t if ctrl_nprw_t is not None else nprw_t, ch_ids))

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
                    array_data_list.append(("UA_ALL", s_ua_3d[valid_ua_idxs, :, :], ua_t, c_ua_3d[valid_ua_idxs, :, :], ctrl_ua_t if ctrl_ua_t is not None else ua_t, all_elecs))

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
                        array_data_list.append((f"UA_{reg}", s_ua_3d[s_idxs, :, :], ua_t, c_ua_3d[c_idxs, :, :], ctrl_ua_t if ctrl_ua_t is not None else ua_t, s_elecs))

    # 4. Generate dedicated behavioral perturbation summary figure for this condition
    y_stim_init, y_ctrl_init, beh_meta_init = compute_behavioral_features(
        s_beh_data=stim_beh_dict,
        c_beh_data=ctrl_beh_dict,
        behavior_window_ms=behavior_window_ms,
        behavior_metric=behavior_metric,
        analysis_mode=analysis_mode,
    )
    beh_stats_cond = beh_meta_init.get("beh_stats", {})
    if beh_stats_cond:
        beh_fig_path = out_dir / f"{stim_path.stem}_behavioral_perturbation.png"
        fig_beh = plot_behavioral_perturbation_summary(
            beh_stats=beh_stats_cond,
            condition_name=condition_name,
            target=target,
            out_path=beh_fig_path,
        )
        if fig_beh is not None:
            plt.close(fig_beh)

    summary_rows = []
    channel_rows = []

    # 5. Analyze each array modality
    for arr_name, s_3d, s_t, c_3d, c_t, ch_ids in array_data_list:
        _log(f"  Fitting Regression for {arr_name} ({len(ch_ids)} channels) | Mode: {analysis_mode} | Metric: {behavior_metric}")

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
            "stim_trial_ids": stim_trial_ids,
            "ctrl_trial_ids": ctrl_trial_ids,
            "stim_beh_data": stim_beh_dict,
            "ctrl_beh_data": ctrl_beh_dict,
        }

        # Step 1: Prepare trial-matched regression data
        X_neural, y_behavior, condition_labels, trial_ids, prep_info = prepare_regression_data(
            session_dict,
            neural_window_ms=neural_window_ms,
            behavior_window_ms=behavior_window_ms,
            analysis_mode=analysis_mode,
            behavior_metric=behavior_metric,
        )

        if X_neural is None or len(y_behavior) < 4:
            _log(f"  [Skip] Inadequate observations for {arr_name}")
            continue

        ch_ids = prep_info["channel_ids"]
        target_label = prep_info["beh_meta"].get("target_label", "Perturbation Score")

        # Step 2: Fit Ridge Regression with Nested Cross-Validation (No Data Leakage)
        reg_res = fit_channel_regression(X_neural, y_behavior)
        channel_weights = reg_res["channel_weights"]
        weight_sd = reg_res["weight_sd"]
        sign_consistency = reg_res["sign_consistency"]
        cv_r2 = reg_res["cv_r2"]
        cv_r = reg_res["cv_pearson_r"]
        best_alpha = reg_res["best_alpha"]
        y_pred_cv = reg_res["y_pred_cv"]
        kf_splits = reg_res["kf_splits"]

        # Step 3: Performance-Based Permutation Testing
        perm_res = permutation_test_model(
            X_neural=X_neural,
            y_behavior=y_behavior,
            condition_labels=condition_labels,
            best_alpha=best_alpha,
            observed_r=cv_r,
            observed_weights=channel_weights,
            kf_splits=kf_splits,
            n_permutations=n_permutations,
        )
        p_model = perm_res["p_model"]
        r_null = perm_res["r_null"]
        p_values = perm_res["p_values"]
        q_values = perm_res["q_values"]

        n_sig_p = int(np.sum(p_values < P_VAL_THRESH))
        n_sig_q = int(np.sum(q_values < FDR_ALPHA))

        _log(
            f"    -> {arr_name}: CV r = {cv_r:.3f}, CV R² = {cv_r2:.3f}, p_model = {p_model:.3f}, "
            f"sig channels: {n_sig_p}/{len(ch_ids)} (p<0.05), {n_sig_q}/{len(ch_ids)} (FDR q<0.05)"
        )

        base_name = f"{stim_path.stem}_{arr_name}"

        # Step 4: Visualizations
        # 1. Channel Weights Bar Plot with Error Bars
        fig_w = plot_channel_weights(
            channel_weights=channel_weights,
            channel_ids=ch_ids,
            p_values=p_values,
            q_values=q_values,
            weight_sd=weight_sd,
            cv_r2=cv_r2,
            cv_r=cv_r,
            p_model=p_model,
            best_alpha=best_alpha,
            array_name=arr_name,
            condition_name=condition_name,
            out_path=out_dir / f"{base_name}_channel_weights.png",
        )
        plt.close(fig_w)

        # 2. Prediction Scatter Plot
        fig_s = plot_prediction_scatter(
            y_true=y_behavior,
            y_pred=y_pred_cv,
            condition_labels=condition_labels,
            cv_r=cv_r,
            cv_r2=cv_r2,
            p_model=p_model,
            array_name=arr_name,
            condition_name=condition_name,
            target_label=target_label,
            out_path=out_dir / f"{base_name}_prediction_scatter.png",
        )
        plt.close(fig_s)

        # 3. Null Distribution Histogram
        fig_null = plot_decoder_null_distribution(
            r_observed=cv_r,
            r_null=r_null,
            p_model=p_model,
            array_name=arr_name,
            condition_name=condition_name,
            out_path=out_dir / f"{base_name}_decoder_null_distribution.png",
        )
        plt.close(fig_null)

        # 4. Top Channels Firing Rate Traces
        fig_t = plot_top_channels_traces(
            s_rates=s_rates,
            c_rates=c_rates,
            time_ms=time_ms,
            channel_ids=ch_ids,
            channel_weights=channel_weights,
            p_values=p_values,
            neural_window_ms=neural_window_ms,
            array_name=arr_name,
            condition_name=condition_name,
            out_path=out_dir / f"{base_name}_top_channels_traces.png",
        )
        if fig_t is not None:
            plt.close(fig_t)

        # 5. Consolidated 4-Panel Publication Figure
        fig_all = plot_regression_summary_figure(
            channel_weights=channel_weights,
            channel_ids=ch_ids,
            p_values=p_values,
            q_values=q_values,
            weight_sd=weight_sd,
            y_true=y_behavior,
            y_pred=y_pred_cv,
            condition_labels=condition_labels,
            r_null=r_null,
            p_model=p_model,
            s_rates=s_rates,
            c_rates=c_rates,
            time_ms=time_ms,
            cv_r2=cv_r2,
            cv_r=cv_r,
            best_alpha=best_alpha,
            array_name=arr_name,
            condition_name=condition_name,
            neural_window_ms=neural_window_ms,
            target_label=target_label,
            out_path=out_dir / f"{base_name}_regression_summary.png",
        )
        plt.close(fig_all)

        # 6. Spatiotemporal Neural-Behavior Projection Maps (Channels x Time Bins)
        # Three separate plots, each for the behavioral perturbation component at each time point:
        #   1) Lateral Perturbation Component ΔX(t)
        #   2) Forward Perturbation Component ΔY(t)
        #   3) Height Perturbation Component ΔZ(t)
        s_beh_coords = stim_beh_dict.get("coords")  # (n_s_trials, n_coords, n_beh_time)
        c_beh_coords = ctrl_beh_dict.get("coords")  # (n_c_trials, n_coords, n_beh_time)
        beh_t_vec = stim_beh_dict.get("beh_t")

        if s_beh_coords is not None and c_beh_coords is not None and beh_t_vec is not None and len(time_ms) > 0:
            n_s_tr, n_coords, _ = s_beh_coords.shape
            n_c_tr = c_beh_coords.shape[0]
            n_tbins = len(time_ms)

            # Rebin / interpolate kinematics onto neural time bins
            s_coords_rebinned = np.full((n_s_tr, n_coords, n_tbins), np.nan, dtype=float)
            for i in range(n_s_tr):
                for d in range(n_coords):
                    val_mask = np.isfinite(s_beh_coords[i, d])
                    if np.sum(val_mask) > 1:
                        s_coords_rebinned[i, d] = np.interp(time_ms, beh_t_vec[val_mask], s_beh_coords[i, d, val_mask], left=np.nan, right=np.nan)

            c_coords_rebinned = np.full((n_c_tr, n_coords, n_tbins), np.nan, dtype=float)
            for i in range(n_c_tr):
                for d in range(n_coords):
                    val_mask = np.isfinite(c_beh_coords[i, d])
                    if np.sum(val_mask) > 1:
                        c_coords_rebinned[i, d] = np.interp(time_ms, beh_t_vec[val_mask], c_beh_coords[i, d, val_mask], left=np.nan, right=np.nan)

            # Control reference trajectory across time bins (median)
            ctrl_ref_traj = np.nanmedian(c_coords_rebinned, axis=0)  # (n_coords, n_tbins)

            # Subselect to trial-matched indices from prepare_regression_data
            s_m_idx = prep_info.get("matched_s_beh_idx")
            c_m_idx = prep_info.get("matched_c_beh_idx")
            s_traj_matched = s_coords_rebinned[s_m_idx] if s_m_idx else s_coords_rebinned
            c_traj_matched = c_coords_rebinned[c_m_idx] if c_m_idx else c_coords_rebinned

            beh_components = [
                ("lateral_dx", "Lateral Perturbation ΔX(t) (mm)", 0),
                ("forward_dy", "Forward Perturbation ΔY(t) (mm)", 1),
                ("height_dz", "Height Perturbation ΔZ(t) (mm)", 2),
            ]

            for comp_key, comp_title, axis_idx in beh_components:
                if axis_idx < n_coords:
                    y_s_traj = s_traj_matched[:, axis_idx, :] - ctrl_ref_traj[axis_idx, None, :]
                    y_c_traj = c_traj_matched[:, axis_idx, :] - ctrl_ref_traj[axis_idx, None, :]

                    fig_spatio = plot_spatiotemporal_projection_map(
                        s_rates=prep_info.get("s_rates_matched", s_rates),
                        c_rates=prep_info.get("c_rates_matched", c_rates),
                        time_ms=time_ms,
                        y_stim=y_s_traj,
                        y_ctrl=y_c_traj,
                        channel_ids=ch_ids,
                        array_name=arr_name,
                        condition_name=condition_name,
                        target_label=comp_title,
                        stim_pulse_ms=stim_dur_nominal_ms or 100.0,
                        out_path=out_dir / f"{base_name}_spatiotemporal_{comp_key}.png",
                    )
                    if fig_spatio is not None:
                        plt.close(fig_spatio)

        # Per-channel detailed results CSV
        df_chan = pd.DataFrame({
            "condition_name": condition_name,
            "target": target,
            "array_name": arr_name,
            "channel_id": ch_ids,
            "weight_mean": channel_weights,
            "weight_sd": weight_sd,
            "sign_consistency": sign_consistency,
            "p_value": p_values,
            "q_value": q_values,
            "is_sig_p05": p_values < P_VAL_THRESH,
            "is_sig_fdr": q_values < FDR_ALPHA,
            "analysis_mode": analysis_mode,
            "behavior_metric": behavior_metric,
            "kinematics_source": kinematics_source,
            "stim_phenotype": beh_stats_cond.get("phenotype", "Unknown"),
        })
        chan_csv_path = out_dir / f"{base_name}_regression_channels.csv"
        df_chan.to_csv(chan_csv_path, index=False)

        for _, row in df_chan.iterrows():
            channel_rows.append(row.to_dict())

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
            "cv_pearson_r": cv_r,
            "p_model": p_model,
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
            "analysis_mode": analysis_mode,
            "behavior_metric": behavior_metric,
            "kinematics_source": kinematics_source,
            # Behavioral perturbation metrics
            "mean_delta_x_mm": beh_stats_cond.get("mean_delta_x", np.nan),
            "mean_delta_y_mm": beh_stats_cond.get("mean_delta_y", np.nan),
            "mean_delta_z_mm": beh_stats_cond.get("mean_delta_z", np.nan),
            "mean_delta_norm_mm": beh_stats_cond.get("mean_delta_norm", np.nan),
            "p_val_delta_x": beh_stats_cond.get("p_val_dx", np.nan),
            "p_val_delta_y": beh_stats_cond.get("p_val_dy", np.nan),
            "p_val_delta_z": beh_stats_cond.get("p_val_dz", np.nan),
            "dispersion_stim_mm": beh_stats_cond.get("disp_s_rms", np.nan),
            "dispersion_ctrl_mm": beh_stats_cond.get("disp_c_rms", np.nan),
            "dispersion_ratio": beh_stats_cond.get("disp_ratio", np.nan),
            "p_val_dispersion": beh_stats_cond.get("p_disp", np.nan),
            "plateau_time_stim_ms": beh_stats_cond.get("s_mean_plateau", np.nan),
            "plateau_time_ctrl_ms": beh_stats_cond.get("c_mean_plateau", np.nan),
            "delta_plateau_time_ms": beh_stats_cond.get("delta_plateau_t", np.nan),
            "p_val_plateau_time": beh_stats_cond.get("p_plateau", np.nan),
            "stim_phenotype": beh_stats_cond.get("phenotype", "Unknown"),
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
    """
    cfg = config or {}
    neural_epoch = cfg.get("neural_epoch", DEFAULT_NEURAL_EPOCH)
    neural_win = cfg.get("neural_window_ms")
    if neural_win is None:
        neural_win = NEURAL_EPOCHS.get(neural_epoch, DEFAULT_NEURAL_WINDOW_MS)
    behavior_win = cfg.get("behavior_window_ms", DEFAULT_BEHAVIOR_WINDOW_MS)
    n_perms = cfg.get("n_permutations", DEFAULT_N_PERMUTATIONS)
    analysis_mode = cfg.get("analysis_mode", DEFAULT_ANALYSIS_MODE)
    behavior_metric = cfg.get("behavior_metric", DEFAULT_BEHAVIOR_METRIC)
    keypoint = cfg.get("keypoint", DEFAULT_KEYPOINT)
    square_size_mm = cfg.get("square_size_mm", SQUARE_SIZE_MM)
    align_task_frame = cfg.get("align_task_frame", ALIGN_TO_TASK_FRAME)
    dlc_3d_dir = cfg.get("dlc_3d_dir", None)
    session = cfg.get("session") or (PARAMS.session if PARAMS and hasattr(PARAMS, "session") and PARAMS.session else "NRR_RW022")
    sess_dir = p3d.resolve_session_dir(session=session) if (HAS_3D_KINEMATICS and p3d) else None

    # Resolve peri_root
    peri_root_arg = Path(cfg["peri_root"]) if cfg.get("peri_root") else None
    if peri_root_arg and peri_root_arg.exists():
        peri_root = peri_root_arg
    elif sess_dir and (sess_dir / "results" / "checkpoints" / "PeriStim").exists():
        peri_root = sess_dir / "results" / "checkpoints" / "PeriStim"
    elif PERI_ROOT is not None and Path(PERI_ROOT).exists():
        peri_root = Path(PERI_ROOT)
    else:
        peri_root = Path(rf"Y:\Current Project Databases - NHP\2025 Cerebellum prosthesis\Nike\20260311_{session}\results\checkpoints\PeriStim")

    # Resolve out_root
    if cfg.get("out_root") and str(cfg["out_root"]) != str(REG_FIG_ROOT):
        out_root = Path(cfg["out_root"])
    elif sess_dir and (sess_dir / "results" / "figures" / "channel_behavior_regression").exists() or (sess_dir and (sess_dir / "results").exists()):
        out_root = sess_dir / "results" / "figures" / "channel_behavior_regression"
    elif OUT_BASE is not None:
        out_root = OUT_BASE / "figures" / "channel_behavior_regression"
    else:
        out_root = Path.cwd() / "results" / "channel_behavior_regression"

    # Resolve dlc_3d_dir if not specified
    if dlc_3d_dir is None:
        if sess_dir and (sess_dir / "Video" / "DLC_3D_new").exists():
            dlc_3d_dir = sess_dir / "Video" / "DLC_3D_new"
        elif sess_dir and (sess_dir / "Video" / "DLC_3D").exists():
            dlc_3d_dir = sess_dir / "Video" / "DLC_3D"

    _log("Starting Channel-Behavior Regression Analysis")
    _log(f"  Session        : {session}")
    _log(f"  PeriStim Root  : {peri_root}")
    _log(f"  Output Root    : {out_root}")
    _log(f"  Mode           : {analysis_mode}")
    _log(f"  Behavior Metric: {behavior_metric}")
    _log(f"  Neural Epoch   : {neural_epoch} ({neural_win[0]:.0f} to {neural_win[1]:.0f} ms)")
    _log(f"  3D Keypoint    : {keypoint} (scale: {square_size_mm:.1f} mm/unit, align_task_frame: {align_task_frame})")
    ensure_dir(out_root)

    bad_channels = load_bad_channels(sess_dir or (SESSION_LOC if "SESSION_LOC" in globals() else None))
    all_files: List[Path] = []
    if peri_root.exists():
        search_dirs = [
            peri_root / "stim_reaches",
            peri_root / "control_reaches",
            peri_root / "Target_A",
            peri_root / "Target_B",
            peri_root,
        ]
        seen_paths = set()
        for sdir in search_dirs:
            if sdir.exists():
                for f in sdir.glob("*.npz"):
                    if not f.name.startswith("._") and f not in seen_paths:
                        all_files.append(f)
                        seen_paths.add(f)
                for child in sdir.iterdir():
                    if child.is_dir() and not child.name.startswith("."):
                        for f in child.glob("*.npz"):
                            if not f.name.startswith("._") and f not in seen_paths:
                                all_files.append(f)
                                seen_paths.add(f)
        if not all_files:
            all_files = sorted([f for f in peri_root.rglob("peristim__*.npz") if not f.name.startswith("._")])
    else:
        all_files = []
    if not all_files:
        _log(f"No peristim files found under {peri_root}")
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
                analysis_mode=analysis_mode,
                behavior_metric=behavior_metric,
                n_permutations=n_perms,
                keypoint=keypoint,
                square_size_mm=square_size_mm,
                align_task_frame=align_task_frame,
                dlc_3d_dir=dlc_3d_dir,
                file_idx=idx + 1,
                total_files=len(stim_files),
            )
            if res:
                all_summary_rows.extend(res.get("summary_rows", []))
                all_channel_rows.extend(res.get("channel_rows", []))
        except Exception as err:
            warnings.warn(f"Error processing {stim_f.name}: {err}\n{traceback.format_exc()}")

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
        "--session", "-s", type=str, default=None,
        help="Session name (e.g. NRR_RW022).",
    )
    parser.add_argument(
        "--peri-root", type=str, default=None,
        help="Custom directory containing PeriStim NPZ checkpoints.",
    )
    parser.add_argument(
        "--save-dir", "--out-dir", type=str, default=None,
        help="Output figure directory.",
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
        "--analysis-mode", type=str, default=DEFAULT_ANALYSIS_MODE,
        choices=["within_condition_residual", "all_stacked_raw", "stim_only", "ctrl_only"],
        help="Analysis mode: 'within_condition_residual' (default), 'all_stacked_raw', 'stim_only', 'ctrl_only'.",
    )
    parser.add_argument(
        "--behavior-metric", type=str, default=DEFAULT_BEHAVIOR_METRIC,
        choices=[
            "endpoint_delta_x", "endpoint_delta_y", "endpoint_delta_z",
            "endpoint_error_3d", "plateau_time", "trajectory_deviation",
            "trajectory_residual", "forward_extent", "lateral_deviation", "peak_speed",
        ],
        help="Behavioral metric to predict: 'endpoint_delta_x' (default, signed lateral shift in mm), 'endpoint_delta_y' (forward shift), 'endpoint_delta_z' (height shift), 'endpoint_error_3d', 'plateau_time', 'trajectory_deviation', 'trajectory_residual', 'forward_extent', 'lateral_deviation', 'peak_speed'.",
    )
    parser.add_argument(
        "--neural-epoch", type=str, default=DEFAULT_NEURAL_EPOCH,
        choices=list(NEURAL_EPOCHS.keys()),
        help="Neural epoch preset: 'pre_stim' ([-300, -50] ms, default, artifact-free!), 'early_post' ([50, 150] ms), 'late_post' ([150, 300] ms), 'peri_stim' ([50, 200] ms).",
    )
    parser.add_argument(
        "--neural-win", nargs=2, type=float, default=None,
        help="Custom neural feature window in ms (e.g. -300 -50). Overrides --neural-epoch if provided.",
    )
    parser.add_argument(
        "--behavior-win", nargs=2, type=float, default=DEFAULT_BEHAVIOR_WINDOW_MS,
        help="Behavioral perturbation window in ms post-stim (e.g. 100 500).",
    )
    parser.add_argument(
        "--permutations", "-p", type=int, default=DEFAULT_N_PERMUTATIONS,
        help="Number of permutations for significance testing (default: 1000).",
    )
    parser.add_argument(
        "--keypoint", type=str, default=DEFAULT_KEYPOINT,
        help="3D DLC keypoint to analyze (default: 'middle').",
    )
    parser.add_argument(
        "--square-size-mm", type=float, default=SQUARE_SIZE_MM,
        help="Checkerboard square size in mm (default: 1.0).",
    )
    parser.add_argument(
        "--no-align-task-frame", action="store_false", dest="align_task_frame", default=True,
        help="Disable automatic rotation to monkey task frame.",
    )
    parser.add_argument(
        "--dlc-3d-dir", type=Path, default=None,
        help="Custom root directory containing continuous DLC 3D CSV/H5 files.",
    )
    args, _ = parser.parse_known_args()

    neural_win = tuple(args.neural_win) if args.neural_win is not None else NEURAL_EPOCHS.get(args.neural_epoch, DEFAULT_NEURAL_WINDOW_MS)

    cfg = {
        "session": args.session,
        "peri_root": args.peri_root,
        "out_root": args.save_dir or REG_FIG_ROOT,
        "targets": args.process_only or PROCESS_ONLY,
        "target_filter": args.target,
        "analysis_mode": args.analysis_mode,
        "behavior_metric": args.behavior_metric,
        "neural_epoch": args.neural_epoch,
        "neural_window_ms": neural_win,
        "behavior_window_ms": tuple(args.behavior_win),
        "n_permutations": args.permutations,
        "keypoint": args.keypoint,
        "square_size_mm": args.square_size_mm,
        "align_task_frame": args.align_task_frame,
        "dlc_3d_dir": args.dlc_3d_dir,
    }
    run_channel_behavior_analysis(config=cfg)


if __name__ == "__main__":
    main()

