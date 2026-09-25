"""
plot_cluster_stim_responses.py

Stimulation response clustering, functional phenotyping, and neural state-space
trajectory dynamics across NPRW (cerebellum) and Utah Array (motor cortex) populations.

Key Scientific Modules:
  1. Trial-Level Paired / Stratified Contrast:
       Δr_{c, i}(t) = r_{STIM, c, i}(t) - r_{CTRL, c, π(i)}(t)
       Preserves trial-level variance to isolate stimulation effect from kinematic variability.
  2. Statistical Responsiveness Pre-Gating:
       Two-sample Welch t-test on trial-level baseline-corrected rate changes (STIM vs CTRL)
       with Benjamini-Hochberg FDR (q < 0.05).
       Non-responsive channels are isolated as a distinct physiological group before clustering.
  3. Dual-Normalization Workflow:
       - Unit-norm trace ||Δr_c(t)||_2 for shape-based clustering (unbiased by high-rate units).
       - Baseline-Z and raw Hz for physiological effect sizes and gain reporting.
  4. Empirical Cluster Selection & Data-Driven Phenotypes:
       Hierarchical Agglomerative Clustering (Ward's linkage) across K in [2, 8],
       validated via Silhouette Score and Calinski-Harabasz Index.
       Empirical phenotype labels derived from reconstructed cluster means.
  5. Two-Window Early vs Late Metric Quadrant Map:
       E_c (early mean) vs L_c (late mean) 4-quadrant Cartesian representation.
  6. Context Interaction (Reaching vs At-Rest):
       Δ_interaction(t) = [r_STIM - r_CTRL]_reach - [r_STIM - r_baseline]_rest
       Distinguishes context-robust responses from behaviorally gated responses.
"""

from __future__ import annotations

import re
import sys
import math
import argparse
import warnings
import traceback

# Suppress benign RuntimeWarnings resulting from all-NaN slices in deliberate artifact blanking windows
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Degrees of freedom <= 0 for slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN slice encountered.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*tight_layout.*")
warnings.filterwarnings("ignore", message=".*Axes that are not compatible with tight_layout.*")
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from scipy.stats import ttest_ind, t

try:
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
except ImportError:
    silhouette_score = None
    calinski_harabasz_score = None

import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

# =============================================================================
# Configuration & Analysis Parameters
# =============================================================================

# Temporal Windows (ms relative to stimulation / event onset)
WIN_PLOT_MS = (-500.0, 500.0)
BASELINE_WIN_MS = (-600.0, -400.0)

# Early response window (defaults to rsa_params.poststim_win_ms if specified in params.yaml)
EARLY_WIN_MS = (0.0, 50.0)
LATE_WIN_MS = (50.0, 150.0)
FULL_RESPONSE_WIN_MS = (0.0, 350.0)

# Stimulation Artifact Blanking Parameters - derived from params.yaml
# Zeroes out interpolated firing rates across [-remove_ms_before, stim_dur + remove_tail_ms_after]
NPRW_BLANK_PRE_MS = float(PARAMS.NPRW_rate_est.get("remove_ms_before", 20.0))
NPRW_BLANK_POST_MS = float(PARAMS.NPRW_rate_est.get("remove_tail_ms_after", 20.0))
UA_BLANK_PRE_MS = float(PARAMS.UA_rate_est.get("remove_ms_before", 0.0))
UA_BLANK_POST_MS = float(PARAMS.UA_rate_est.get("remove_tail_ms_after", 0.0))
BLANK_UA_STIM_PERIOD = False  # By default, motor cortex (UA) records real spikes during cerebellar stim

# Display Binning & Smoothing - derived from rate_est in params.yaml
NPRW_BIN_WIDTH_MS = float(PARAMS.NPRW_rate_est.get("bin_ms", 20.0))
NPRW_SMOOTHING_SIGMA_MS = float(PARAMS.NPRW_rate_est.get("sigma_ms", 20.0))
UA_BIN_WIDTH_MS = float(PARAMS.UA_rate_est.get("bin_ms", 20.0))
UA_SMOOTHING_SIGMA_MS = float(PARAMS.UA_rate_est.get("sigma_ms", 20.0))

# General fallback defaults
BIN_WIDTH_MS = NPRW_BIN_WIDTH_MS
SMOOTHING_SIGMA_MS = NPRW_SMOOTHING_SIGMA_MS

# Responsiveness Test Modes & Gating
RESPONSIVENESS_TEST_MODE = "welch_baseline_corrected"     # "welch_baseline_corrected", "paired_signflip", "unpaired_label_shuffle", or "both"
RESPONSIVENESS_BOTH_COMBINATION = "intersection" # "intersection", "union", "paired_only", "unpaired_only"
RESPONSIVENESS_CRITERION = "fdr"           # "fdr" (BH FDR q < FDR_ALPHA), "p_val" (permutation p < P_VAL_THRESH & |Δr| >= MIN_DELTA_HZ), or "hybrid"
CONTRAST_SOURCE_FOR_PLOTS = "unpaired"          # "paired" (mean paired contrast) or "unpaired" (mean STIM - mean CTRL)
N_PERMUTATIONS = 3000
FDR_ALPHA = 0.05
P_VAL_THRESH = 0.05
MIN_DELTA_HZ = 3.0                            # Minimum post-stim mean absolute contrast (Hz) when using p_val criterion
MIN_SPIKE_COUNT_THRESH = 5                    # Exclude completely silent channels

# Clustering Hyperparameters
CLUSTER_RANGE_K = range(2, 9)
DEFAULT_K = 4
LINKAGE_METHOD = "ward"

# Trial Matching Method:
# "chronological": (Recommended for pre-curated trials) Pairs trials 1-to-1 in chronological session order up to min(n_stim, n_ctrl).
# "kinematic": Pairs trials 1-to-1 by closest peak reaching velocity without replacement up to min(n_stim, n_ctrl).
TRIAL_MATCHING_MODE = "chronological"

# Utah Array Separation Options
ANALYZE_INDIVIDUAL_UTAH_ARRAYS = True  # Separates Utah channels into individual arrays (SMA, PMd, M1i, M1s)
ANALYZE_COMBINED_UTAH = False          # Set to True if pooled UA analysis is also desired alongside individual arrays

# Processing & Session-level Filtering from params.yaml
PROCESS_ONLY = PARAMS.preprocessing.get("process_only")
HAS_BR = bool(PARAMS.preprocessing.get("has_BR", True))
HAS_KINEMATICS = bool(PARAMS.preprocessing.get("has_kinematics", True))

# Reaching Kinematics Parameters (consistent with plot_firing_rates.py and plot_plateau_analysis.py)
KIN_KEYPOINTS = ("middle", "wrist", "hand", "index")
KIN_REF_TIME_MS = -600.0
KIN_NORM_MODE = "max_abs"
TRAJECTORY_VARIABILITY_AGG = "median"  # "median" (matching plot_plateau_analysis.py) or "mean"

# Probe channel counts from params.yaml
NPRW_N_CHANNELS = int(PARAMS.probes.get("NPRW", {}).get("n_channels", 128))
UA_N_CHANNELS = int(PARAMS.probes.get("UA", {}).get("n_channels", 256))

# Geometry path from params.yaml
GEOM_PATH = (
    Path(PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None) and str(PARAMS.geom_mat_rel).startswith("/")
    else (REPO_ROOT / PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None)
    else rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
) if (PARAMS is not None and getattr(PARAMS, "geom_mat_rel", None)) else None

# Plotting Aesthetics & Colors (consistent with plot_stim_response_overlays.py)
DPI_OUTPUT = 150
SKIP_EXISTING = False
VERBOSE = True

CURRENT_COLOR = "tab:grey"
COLOR_STIM = "tab:grey"
COLOR_CTRL = "tab:orange"
COLOR_REST = "tab:green"
STIM_REGION_COLOR = "gold"
STIM_REGION_ALPHA = 0.40
BAD_CH_COLOR = "0.85"

CLUSTER_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]

# Roots
PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim" if OUT_BASE is not None else Path(".")
CLUST_FIG_ROOT = OUT_BASE / "figures" / "stim_response_clustering" if OUT_BASE is not None else Path(".")
METADATA_PATH = METADATA_CSV if "METADATA_CSV" in globals() else None

REGION_ORDER = ["SMA", "PMd", "M1i", "M1s"]


# =============================================================================
# Helper Utilities
# =============================================================================

def _log(msg: str):
    if VERBOSE:
        print(f"[cluster_stim] {msg}", flush=True)

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
        if arr.shape == ():
            return arr.item()
        if arr.size == 1:
            return arr.reshape(-1)[0].item()
    except Exception:
        pass
    return x

def sanitize_name(s: Any) -> str:
    s = str(s).replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)

def get_cond_type(npz_path: Path) -> str:
    path_str = str(npz_path).lower()
    if "control_reaches" in path_str:
        return "CTRL"
    if "stim_reaches" in path_str:
        return "STIM"
    if "continuous_stim" in path_str:
        return "CONTSTIM"
    if "at_rest" in path_str:
        return "REST"
    return "OTHER"

def get_target_folder(npz_path: Path) -> str:
    name = npz_path.name.lower()
    path_str = str(npz_path).lower()
    if "_target_a" in name or "target_a" in path_str:
        return "Target_A"
    if "_target_b" in name or "target_b" in path_str:
        return "Target_B"
    if "_target_control" in name or "target_control" in path_str:
        return "Target_control"
    return ""

def get_stim_duration_ms(data: np.lib.npyio.NpzFile) -> float:
    """
    Get stimulation train duration in ms, based on:
      1. Explicit pulse count and frequency (n_pulses / freq_hz * 1000 ms)
      2. Nominal duration from metadata (stim_dur_nominal_ms)
      3. Measured hardware bounds (stim_dur_measured_ms)
      4. Fallback default (100 ms)
    """
    # 1. Pulse count and frequency
    freq = scalarize(safe_get_npz(data, "stim_freq_hz", None))
    pulses = safe_get_npz(data, "stim_pulse_counts", None)
    if freq is not None and pulses is not None:
        try:
            fh = float(freq)
            arr_p = np.asarray(pulses, dtype=float).ravel()
            arr_p = arr_p[np.isfinite(arr_p) & (arr_p > 0)]
            if fh > 0 and arr_p.size > 0:
                p_med = float(np.nanmedian(arr_p))
                return float(p_med / fh * 1000.0)
        except Exception:
            pass

    # 2. Nominal duration
    dur_nom = safe_get_npz(data, "stim_dur_nominal_ms", None)
    if dur_nom is not None:
        try:
            val = float(scalarize(dur_nom))
            if np.isfinite(val) and val > 0:
                return val
        except Exception:
            pass

    # 3. Measured duration from hardware pulse bounds
    dur_meas = safe_get_npz(data, "stim_dur_measured_ms", None)
    if dur_meas is not None:
        try:
            arr = np.asarray(dur_meas, dtype=float).ravel()
            arr = arr[np.isfinite(arr) & (arr > 0)]
            if arr.size > 0:
                return float(np.nanmedian(arr))
        except Exception:
            pass

    return 100.0  # Fallback standard duration


# =============================================================================
# Reference Matching & Metadata
# =============================================================================

def get_ua_port(data: np.lib.npyio.NpzFile) -> str:
    port = safe_get_npz(data, "ua_port", None)
    if port is not None:
        p = str(scalarize(port)).strip().upper()
        if p in ("A", "B"):
            return p
    return "A"

def compute_pulse_count(freq_hz: Optional[float], dur_ms: Optional[float]) -> Optional[float]:
    if freq_hz is None or dur_ms is None:
        return None
    try:
        fh = float(freq_hz)
        dm = float(dur_ms)
        if np.isfinite(fh) and np.isfinite(dm) and fh > 0 and dm > 0:
            return round(fh * dm / 1000.0, 4)
    except Exception:
        pass
    return None

def load_reference_mapping(all_files: List[Path]) -> Dict[str, Any]:
    """Find best matched control (by target & port) and rest files (by pulse count)."""
    refs = {
        "ctrl_by_target_port": {},
        "rest_by_pulse": {},
    }
    for f in all_files:
        c_type = get_cond_type(f)
        if c_type == "CTRL":
            t = get_target_folder(f)
            if not t:
                continue
            try:
                with np.load(f, allow_pickle=True) as d:
                    port = get_ua_port(d)
                    n_trials = safe_get_npz(d, "n_trials", 0)
                    key = (t, port)
                    old_f = refs["ctrl_by_target_port"].get(key)
                    if old_f is None:
                        refs["ctrl_by_target_port"][key] = (f, int(scalarize(n_trials, 0)))
                    else:
                        if int(scalarize(n_trials, 0)) > old_f[1]:
                            refs["ctrl_by_target_port"][key] = (f, int(scalarize(n_trials, 0)))
            except Exception:
                pass
        elif c_type == "REST":
            try:
                with np.load(f, allow_pickle=True) as d:
                    fh = scalarize(safe_get_npz(d, "stim_freq_hz", None))
                    dm = scalarize(safe_get_npz(d, "stim_dur_nominal_ms", None))
                    pc = compute_pulse_count(fh, dm)
                    if pc is not None:
                        n_trials = safe_get_npz(d, "n_trials", 0)
                        old_f = refs["rest_by_pulse"].get(pc)
                        if old_f is None:
                            refs["rest_by_pulse"][pc] = (f, int(scalarize(n_trials, 0)))
                        else:
                            if int(scalarize(n_trials, 0)) > old_f[1]:
                                refs["rest_by_pulse"][pc] = (f, int(scalarize(n_trials, 0)))
            except Exception:
                pass

    return {
        "ctrl": {k: v[0] for k, v in refs["ctrl_by_target_port"].items()},
        "rest": {k: v[0] for k, v in refs["rest_by_pulse"].items()},
    }


# =============================================================================
# Bad Channels & Utah Array Mapping
# =============================================================================

def load_bad_channels(session_loc: Optional[Path] = None) -> Dict[str, set]:
    bad = {"NPRW": set(), "UA": set()}
    try:
        if session_loc is None:
            session_loc = globals().get("SESSION_LOC") or getattr(PARAMS, "session_loc", None) or getattr(PARAMS, "session_path", None)
        if session_loc is None and "OUT_BASE" in globals() and OUT_BASE is not None:
            out_base = Path(OUT_BASE)
            parts = [p.lower() for p in out_base.parts]
            if "results" in parts:
                idx = parts.index("results")
                session_loc = Path(*out_base.parts[:idx])
            else:
                session_loc = out_base.parent

        if hasattr(rcp, "get_session_impedances"):
            imp = rcp.get_session_impedances(session_loc)
            if isinstance(imp, dict):
                bad["NPRW"] = set(imp.get("nprw", []))
                bad["UA"] = set(imp.get("utah", []))
        _log(f"Impedance check for session ({session_loc}): NPRW bad={len(bad['NPRW'])}, UA bad={len(bad['UA'])}")
    except Exception as e:
        warnings.warn(f"Could not load impedances: {e}")
    return bad

@lru_cache(maxsize=4)
def get_electrode_mapping_csv(monkey: str = None) -> Path:
    """Get path to electrode mapping CSV for the specified monkey."""
    if monkey is None:
        monkey = PARAMS.monkey if PARAMS is not None else "Nike"
    csv_path = REPO_ROOT / "config" / f"electrode_port_mapping_{monkey}.csv"
    if not csv_path.exists():
        fallback = Path(__file__).parent.parent / "config" / f"electrode_port_mapping_{monkey}.csv"
        if fallback.exists():
            return fallback
    return csv_path

@lru_cache(maxsize=4)
def load_electrode_mapping(monkey: str = None) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    Load electrode mapping from CSV file.
    
    Returns:
        tuple: (elec_info, region_grids)
            - elec_info: dict mapping electrode_id -> {nsp_id, port, region, row, col}
            - region_grids: dict mapping region_name -> 2D array of electrode IDs (8x8)
    """
    if monkey is None:
        monkey = PARAMS.monkey if PARAMS is not None else "Nike"
    
    csv_path = get_electrode_mapping_csv(monkey)
    elec_info = {}
    region_grids = {r: np.zeros((8, 8), dtype=int) for r in REGION_ORDER}

    if not csv_path.exists():
        warnings.warn(f"Electrode mapping CSV not found: {csv_path}")
        return elec_info, region_grids

    df = pd.read_csv(csv_path)
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
        else:
            warnings.warn(
                f"Electrode {elec_id} has out-of-range grid position "
                f"({grid_row}, {grid_col}) for 8x8 grid."
            )

    return elec_info, region_grids

def build_elec_to_channel_idx(ua_ids_1based, elec_info, recording_port='A'):
    """
    Build electrode_id -> channel index mapping.
    For Port A: local data channels correspond to NSP 1-128.
    For Port B: local data channels correspond to NSP 129-256.
    """
    elec_to_idx = {}
    recording_port = str(recording_port).strip().upper() if recording_port else "A"
    if recording_port not in ("A", "B"):
        recording_port = "A"

    port_elecs = {
        elec_id
        for elec_id, info in elec_info.items()
        if str(info.get("port", "")).upper() == recording_port
    }

    ua_ids_1based = None if ua_ids_1based is None else np.asarray(ua_ids_1based).ravel()

    is_elec_ids = False
    if ua_ids_1based is not None and len(ua_ids_1based) > 0 and len(port_elecs) > 0:
        valid_elecs_count = sum(1 for x in ua_ids_1based if int(x) in port_elecs)
        is_elec_ids = valid_elecs_count > len(ua_ids_1based) * 0.5

    if is_elec_ids:
        for ch_idx, elec_id_raw in enumerate(ua_ids_1based):
            try:
                elec_id = int(elec_id_raw)
            except Exception:
                continue
            elec_to_idx[elec_id] = ch_idx
        return elec_to_idx

    nsp_to_elec = {}
    for elec_id, info in elec_info.items():
        try:
            nsp_to_elec[int(info["nsp_id"])] = elec_id
        except Exception:
            pass

    nsp_offset = 0 if recording_port == "A" else 128

    if ua_ids_1based is not None and len(ua_ids_1based) > 0:
        for ch_idx, nsp_id_raw in enumerate(ua_ids_1based):
            try:
                nsp_id = int(nsp_id_raw)
            except Exception:
                continue
            if 1 <= nsp_id <= 128:
                global_nsp_id = nsp_id + nsp_offset
            else:
                global_nsp_id = nsp_id
            elec_id = nsp_to_elec.get(global_nsp_id)
            if elec_id is not None:
                elec_to_idx[elec_id] = ch_idx
    else:
        for ch_idx in range(128):
            global_nsp_id = ch_idx + 1 + nsp_offset
            elec_id = nsp_to_elec.get(global_nsp_id)
            if elec_id is not None:
                elec_to_idx[elec_id] = ch_idx

    return elec_to_idx

def extract_array_channel_indices(
    ua_ids_1based: Optional[np.ndarray],
    elec_info: Dict[int, Dict[str, Any]],
    port: str,
    region: str,
    ua_region: Optional[np.ndarray] = None,
    ua_region_names: Optional[np.ndarray] = None,
    max_channels: int = 128,
    bad_channels: Optional[set] = None,
) -> Tuple[List[int], List[int]]:
    """
    Extract channel indices and electrode IDs corresponding to a specific Utah Array region (SMA, PMd, M1i, M1s).
    Prioritizes ua_region stored directly with the data in peristim npz files, with fallback to CSV electrode port mapping.
    Filters out bad channels based on impedance if bad_channels set is provided.
    """
    idxs = []
    elecs = []

    # 1. Primary: Match directly via ua_region and ua_region_names from npz
    if ua_region is not None and ua_region_names is not None:
        region_lower = region.lower()
        reg_code = None
        for code_idx, name in enumerate(ua_region_names):
            n_str = str(name).lower()
            if region_lower == "sma" and "sma" in n_str:
                reg_code = code_idx
                break
            elif region_lower == "pmd" and ("pmd" in n_str or "premotor" in n_str):
                reg_code = code_idx
                break
            elif region_lower == "m1i" and ("m1i" in n_str or "inferior" in n_str):
                reg_code = code_idx
                break
            elif region_lower == "m1s" and ("m1s" in n_str or "superior" in n_str):
                reg_code = code_idx
                break

        if reg_code is not None:
            raw_idxs = np.where(np.asarray(ua_region).ravel() == reg_code)[0].tolist()
            raw_idxs = [int(i) for i in raw_idxs if int(i) < max_channels]
            if raw_idxs:
                raw_elecs = [int(ua_ids_1based[i]) for i in raw_idxs] if ua_ids_1based is not None and len(ua_ids_1based) > max(raw_idxs) else raw_idxs
                idxs = raw_idxs
                elecs = raw_elecs

    # 2. Fallback: Match via elec_info region field and elec_to_idx from CSV
    if not idxs:
        elec_to_idx = build_elec_to_channel_idx(ua_ids_1based, elec_info, port)
        reg_clean = region.strip().upper()
        elecs_in_reg = [
            e for e, info in elec_info.items()
            if str(info.get("region", "")).strip().upper() == reg_clean
        ]
        matched_idxs = []
        matched_elecs = []
        for e in elecs_in_reg:
            idx = elec_to_idx.get(e)
            if idx is not None and 0 <= idx < max_channels:
                matched_idxs.append(idx)
                matched_elecs.append(e)

        if matched_idxs:
            pairs = sorted(zip(matched_idxs, matched_elecs), key=lambda p: p[0])
            idxs = [p[0] for p in pairs]
            elecs = [p[1] for p in pairs]

    # Filter bad channels based on impedance if bad_channels provided
    if bad_channels and idxs:
        filtered = [
            (i, e) for i, e in zip(idxs, elecs)
            if e not in bad_channels and i not in bad_channels and (i + 1) not in bad_channels
        ]
        idxs = [p[0] for p in filtered]
        elecs = [p[1] for p in filtered]

    return idxs, elecs



# =============================================================================
# Core Signal Processing & Trial Pairing
# =============================================================================

def prepare_counts_3d(counts_raw: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Ensure array is shaped (n_channels, n_trials, n_bins)."""
    if counts_raw is None:
        return None
    arr = np.asarray(counts_raw, dtype=float)
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim != 3:
        return None
    n0, n1, n2 = arr.shape
    # If axis 1 looks like channels and axis 0 is trials (standard peristim npz format):
    if n1 in (32, 64, 96, 128, 256) and n0 not in (32, 64, 96, 128, 256):
        return np.transpose(arr, (1, 0, 2))
    elif n0 in (32, 64, 96, 128, 256):
        return arr
    if n1 > n0 and n1 in (16, 24, 32, 48, 64, 96, 128, 256):
        return np.transpose(arr, (1, 0, 2))
    return arr

def nan_gaussian_filter1d(arr_3d: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Apply 1D Gaussian filter along axis=2, ignoring NaNs to prevent 0-count artifact bleed into pre-stim baseline."""
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
    target_bin_w: float = BIN_WIDTH_MS,
    smooth_sigma_ms: float = SMOOTHING_SIGMA_MS,
    win_ms: Tuple[float, float] = WIN_PLOT_MS,
    blank_win: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rebin counts to target_bin_w, convert to rates (Hz), mask blanking window, and smooth along time axis.
    Returns: rates_3d (channels, trials, bins), bin_centers_ms.
    """
    if counts_3d is None or centers_ms is None:
        return np.zeros((0, 0, 0)), np.array([])

    arr = np.asarray(counts_3d, dtype=float)
    centers = np.asarray(centers_ms, dtype=float).ravel()

    if arr.ndim != 3 or centers.size == 0:
        return np.zeros((0, 0, 0)), np.array([])

    n_ch, n_trials, n_bins_orig = arr.shape
    m_bins = min(n_bins_orig, centers.size)
    arr = arr[:, :, :m_bins]
    centers = centers[:m_bins]

    t_start, t_stop = win_ms

    edges = np.arange(t_start, t_stop + 0.5 * target_bin_w, target_bin_w, dtype=float)
    n_target_bins = len(edges) - 1
    new_centers = 0.5 * (edges[:-1] + edges[1:])

    # Rebin
    bin_idx = np.digitize(centers, edges) - 1
    rebinned = np.zeros((n_ch, n_trials, n_target_bins), dtype=float)
    for b in range(n_target_bins):
        mask = bin_idx == b
        if np.any(mask):
            rebinned[:, :, b] = np.nansum(arr[:, :, mask], axis=2)

    # Convert counts to instantaneous firing rate (Hz)
    rates_hz = rebinned / (target_bin_w / 1000.0)

    # Apply artifact blanking mask BEFORE Gaussian smoothing to prevent 0-count leakage into baseline
    if blank_win is not None:
        blank_mask = (new_centers >= blank_win[0]) & (new_centers <= blank_win[1])
        rates_hz[:, :, blank_mask] = np.nan

    # Gaussian smoothing
    if smooth_sigma_ms > 0:
        sigma_bins = smooth_sigma_ms / target_bin_w
        if np.any(np.isnan(rates_hz)):
            rates_hz = nan_gaussian_filter1d(rates_hz, sigma_bins)
        else:
            rates_hz = gaussian_filter1d(rates_hz, sigma=sigma_bins, axis=2, mode="nearest")

    return rates_hz, new_centers

def _extract_trial_peaks(vel_array: Any, n_trials: int) -> Optional[np.ndarray]:
    """
    Extract a 1D array of scalar peak reach speeds for each trial.
    Safely reduces across all non-trial dimensions (e.g. keypoints and timepoints).
    """
    if vel_array is None:
        return None
    try:
        arr = np.asarray(vel_array, dtype=float)
        if arr.size == 0 or arr.shape[0] != n_trials:
            return None
        if arr.ndim > 1:
            axes = tuple(range(1, arr.ndim))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                peaks = np.nanmax(np.abs(arr), axis=axes)
        else:
            peaks = np.abs(arr)
        peaks = np.nan_to_num(peaks, nan=0.0)
        return peaks.astype(float)
    except Exception:
        return None


def format_trial_info(trial_counts: Optional[Dict[str, Any]]) -> str:
    """Format trial counts dictionary into a clean string for plot headers/legends."""
    if not trial_counts:
        return ""
    parts = []
    if trial_counts.get("stim") is not None:
        parts.append(f"Stim={trial_counts['stim']}")
    if trial_counts.get("ctrl") is not None:
        parts.append(f"Ctrl={trial_counts['ctrl']}")
    if trial_counts.get("matched") is not None and trial_counts["matched"] != trial_counts.get("stim"):
        parts.append(f"Matched={trial_counts['matched']}")
    if trial_counts.get("rest") is not None and trial_counts["rest"] > 0:
        parts.append(f"Rest={trial_counts['rest']}")
    return f"Trials: {', '.join(parts)}" if parts else ""


def match_trials_stratified(
    stim_rates: np.ndarray,
    ctrl_rates: np.ndarray,
    stim_vel: Optional[np.ndarray] = None,
    ctrl_vel: Optional[np.ndarray] = None,
    mode: str = TRIAL_MATCHING_MODE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform strict 1-to-1 trial matching between STIM and CTRL conditions.
    Matches n_match = min(n_stim, n_ctrl) unique trials without replacement.
    Ensures strict statistical exchangeability and independence for paired permutation tests (Point 5).

    Modes:
      - "chronological": (Recommended for pre-curated trials) Pairs trials 1-to-1 in natural session order up to n_match.
      - "kinematic": Pairs trials 1-to-1 by closest peak reaching velocity without replacement up to n_match.
    """
    n_ch, n_stim, n_bins = stim_rates.shape
    _, n_ctrl, _ = ctrl_rates.shape

    if n_stim == 0 or n_ctrl == 0:
        return np.zeros((n_ch, 0, n_bins)), np.zeros((n_ch, 0, n_bins))

    n_match = min(n_stim, n_ctrl)

    if mode == "kinematic" and stim_vel is not None and ctrl_vel is not None:
        stim_peaks = _extract_trial_peaks(stim_vel, n_stim)
        ctrl_peaks = _extract_trial_peaks(ctrl_vel, n_ctrl)
        if stim_peaks is not None and ctrl_peaks is not None and (np.any(stim_peaks > 0) or np.any(ctrl_peaks > 0)):
            if n_stim <= n_ctrl:
                matched_s = list(range(n_stim))
                matched_c = []
                used_ctrl = set()
                for i in range(n_stim):
                    avail = [c for c in range(n_ctrl) if c not in used_ctrl]
                    best_c = min(avail, key=lambda c: abs(float(stim_peaks[i]) - float(ctrl_peaks[c])))
                    matched_c.append(best_c)
                    used_ctrl.add(best_c)
            else:
                matched_c = list(range(n_ctrl))
                matched_s = []
                used_stim = set()
                for c in range(n_ctrl):
                    avail = [s for s in range(n_stim) if s not in used_stim]
                    best_s = min(avail, key=lambda s: abs(float(stim_peaks[s]) - float(ctrl_peaks[c])))
                    matched_s.append(best_s)
                    used_stim.add(best_s)
            paired_stim = stim_rates[:, matched_s, :]
            paired_ctrl = ctrl_rates[:, matched_c, :]
            return paired_stim, paired_ctrl

    # Default / Chronological: Pair trials 1-to-1 in natural session order up to n_match without reuse
    paired_stim = stim_rates[:, :n_match, :]
    paired_ctrl = ctrl_rates[:, :n_match, :]
    return paired_stim, paired_ctrl


# =============================================================================
# =============================================================================
# Statistical Responsiveness Testing (Welch Baseline-Corrected, Paired Sign-Flip & Unpaired Label-Shuffle)
# =============================================================================

def apply_responsiveness_criterion(
    p_values: np.ndarray,
    q_values: np.ndarray,
    mean_abs_delta_hz: np.ndarray,
    criterion: str = RESPONSIVENESS_CRITERION,
    alpha: float = FDR_ALPHA,
    p_thresh: float = P_VAL_THRESH,
    min_delta_hz: float = MIN_DELTA_HZ,
) -> np.ndarray:
    """
    Apply statistical and effect-size gating criteria to p-values and q-values.
    
    Modes:
      - "fdr": q <= alpha (effect-size threshold not required by default when using FDR)
      - "fdr_min_delta" / "fdr_effect_size": (q <= alpha) & (mean_abs_delta_hz >= min_delta_hz)
      - "p_val": (p <= p_thresh) & (mean_abs_delta_hz >= min_delta_hz)
      - "hybrid": (q <= alpha) | ((p <= p_thresh) & (mean_abs_delta_hz >= min_delta_hz))
    """
    crit = str(criterion).strip().lower()
    if crit == "p_val":
        return (p_values <= p_thresh) & (mean_abs_delta_hz >= min_delta_hz)
    elif crit in ("fdr_min_delta", "fdr_effect_size", "fdr_delta"):
        return (q_values <= alpha) & (mean_abs_delta_hz >= min_delta_hz)
    elif crit == "hybrid":
        return (q_values <= alpha) | ((p_values <= p_thresh) & (mean_abs_delta_hz >= min_delta_hz))
    else:  # "fdr"
        return q_values <= alpha


def compute_benjamini_hochberg_fdr(p_values: np.ndarray) -> np.ndarray:
    """
    Compute Benjamini-Hochberg FDR-adjusted q-values across an array of p-values.
    Handles NaNs by treating them as p = 1.0 (non-significant).
    """
    p = np.array(p_values, dtype=float, copy=True)
    p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
    p = np.clip(p, 0.0, 1.0)
    n = len(p)
    if n == 0:
        return np.empty(0, dtype=float)

    sorted_indices = np.argsort(p)
    q_values = np.ones(n, dtype=float)
    cumulative_min = 1.0
    for rank in range(n - 1, -1, -1):
        idx = sorted_indices[rank]
        q_val = p[idx] * n / (rank + 1)
        cumulative_min = min(cumulative_min, q_val)
        q_values[idx] = min(cumulative_min, 1.0)
    return q_values


def compute_welch_baseline_corrected_responsiveness(
    stim_rates: np.ndarray,
    ctrl_rates: np.ndarray,
    time_ms: np.ndarray,
    baseline_win_ms: Tuple[float, float] = BASELINE_WIN_MS,
    response_win_ms: Tuple[float, float] = FULL_RESPONSE_WIN_MS,
    blank_mask: Optional[np.ndarray] = None,
    alpha: float = FDR_ALPHA,
    criterion: str = RESPONSIVENESS_CRITERION,
    p_thresh: float = P_VAL_THRESH,
    min_delta_hz: float = MIN_DELTA_HZ,
    contrast_trace: Optional[np.ndarray] = None,
    channel_ids: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Trial-level baseline-corrected Welch two-sample t-test for responsive-channel selection.

    For each channel and each trial:
      1. Baseline firing rate:
           B_i = mean(r_i(t), t in [-500, -300] ms)
      2. Response-window firing rate:
           P_i = mean(r_i(t), t in resp_mask)
         - For NPRW: post-artifact response window (resp_win)
         - For UA: response window, including during-stimulation bins when UA blanking is disabled
      3. Baseline-corrected rate change:
           Δ_i = P_i - B_i
      4. Compare STIM trial changes vs CTRL trial changes using a two-sided Welch two-sample t-test:
           scipy.stats.ttest_ind(stim_changes, ctrl_changes, equal_var=False, nan_policy="omit")
      5. Apply Benjamini-Hochberg FDR correction across channels within each analyzed array/condition:
           q_value <= FDR_ALPHA
         (MIN_DELTA_HZ threshold available as optional additional effect-size gate, but not required by default).
      6. Report and return per-channel metrics:
           - raw Welch t-test p-value
           - BH-FDR q-value
           - Welch t statistic
           - STIM mean baseline-corrected change (Hz)
           - CTRL mean baseline-corrected change (Hz)
           - stimulation-specific effect size: Δ_effect = mean(Δ_STIM) - mean(Δ_CTRL)
           - 95% confidence interval for effect size [ci_low, ci_high]
           - responsive boolean

    Returns:
      is_responsive: (n_channels,) boolean array
      p_values: (n_channels,) raw Welch t-test p-values
      q_values: (n_channels,) BH-FDR adjusted q-values
      mean_abs_delta_hz: (n_channels,) absolute effect size |Δ_effect| in Hz
      observed_contrast_mean: (n_channels, n_bins) time-resolved contrast trace
      stats_dict: dict containing full per-channel metrics and DataFrame
    """
    n_ch, n_stim, n_bins = stim_rates.shape
    n_ctrl = ctrl_rates.shape[1]

    # Baseline mask [-500, -300] ms
    base_mask = (time_ms >= baseline_win_ms[0]) & (time_ms <= baseline_win_ms[1])

    # Response window mask
    resp_mask = (time_ms >= response_win_ms[0]) & (time_ms <= response_win_ms[1])
    if blank_mask is not None:
        resp_mask = resp_mask & (~blank_mask)

    # Time-resolved contrast trace for downstream clustering/plots
    if contrast_trace is not None:
        observed_contrast_mean = contrast_trace
    else:
        mean_s = np.nanmean(stim_rates, axis=1)
        mean_c = np.nanmean(ctrl_rates, axis=1)
        observed_contrast_mean = mean_s - mean_c

    # Check adequate trials and valid masks
    if n_stim < 2 or n_ctrl < 2 or not np.any(resp_mask) or not np.any(base_mask):
        _log(f"  [welch_baseline_corrected] Inadequate trials (stim={n_stim}, ctrl={n_ctrl}, require >= 2 each) or empty windows. Marking non-responsive.")
        is_resp = np.zeros(n_ch, dtype=bool)
        p_vals = np.ones(n_ch, dtype=float)
        q_vals = np.ones(n_ch, dtype=float)
        t_stats = np.zeros(n_ch, dtype=float)
        stim_mean_delta = np.zeros(n_ch, dtype=float)
        ctrl_mean_delta = np.zeros(n_ch, dtype=float)
        delta_effect = np.zeros(n_ch, dtype=float)
        ci_95_low = np.full(n_ch, np.nan)
        ci_95_high = np.full(n_ch, np.nan)
        mean_abs_delta = np.zeros(n_ch, dtype=float)

        df_metrics = pd.DataFrame({
            "channel_idx": np.arange(n_ch),
            "channel_id": channel_ids if channel_ids is not None else np.arange(1, n_ch + 1),
            "is_responsive": is_resp,
            "welch_p_value": p_vals,
            "bh_fdr_q_value": q_vals,
            "welch_t_stat": t_stats,
            "stim_mean_delta_hz": stim_mean_delta,
            "ctrl_mean_delta_hz": ctrl_mean_delta,
            "delta_effect_hz": delta_effect,
            "ci_95_low_hz": ci_95_low,
            "ci_95_high_hz": ci_95_high,
            "n_stim_trials": np.full(n_ch, n_stim),
            "n_ctrl_trials": np.full(n_ch, n_ctrl),
        })
        stats_dict = {
            "is_responsive": is_resp,
            "p_values": p_vals,
            "q_values": q_vals,
            "t_stats": t_stats,
            "stim_mean_delta_hz": stim_mean_delta,
            "ctrl_mean_delta_hz": ctrl_mean_delta,
            "delta_effect_hz": delta_effect,
            "ci_95_low_hz": ci_95_low,
            "ci_95_high_hz": ci_95_high,
            "mean_abs_delta_hz": mean_abs_delta,
            "contrast_mean": observed_contrast_mean,
            "metrics_df": df_metrics,
            "n_resp": 0,
            "n_sig_p": 0,
            "n_sig_q": 0,
        }
        return is_resp, p_vals, q_vals, mean_abs_delta, observed_contrast_mean, stats_dict

    # 1. Compute trial-level baseline firing rates: B_i = mean(r_i(t), t in [-500, -300] ms)
    stim_base = np.nanmean(stim_rates[:, :, base_mask], axis=2)  # (n_ch, n_stim)
    ctrl_base = np.nanmean(ctrl_rates[:, :, base_mask], axis=2)  # (n_ch, n_ctrl)

    # 2. Compute trial-level response-window firing rates: P_i = mean(r_i(t), t in resp_mask)
    stim_resp = np.nanmean(stim_rates[:, :, resp_mask], axis=2)  # (n_ch, n_stim)
    ctrl_resp = np.nanmean(ctrl_rates[:, :, resp_mask], axis=2)  # (n_ch, n_ctrl)

    # 3. Compute each trial's baseline-corrected rate change: Δ_i = P_i - B_i
    stim_changes = stim_resp - stim_base  # (n_ch, n_stim)
    ctrl_changes = ctrl_resp - ctrl_base  # (n_ch, n_ctrl)

    # Preallocate metric arrays
    raw_p_values = np.ones(n_ch, dtype=float)
    welch_t_stats = np.zeros(n_ch, dtype=float)
    stim_mean_delta_hz = np.zeros(n_ch, dtype=float)
    ctrl_mean_delta_hz = np.zeros(n_ch, dtype=float)
    delta_effect_hz = np.zeros(n_ch, dtype=float)
    ci_95_low_hz = np.full(n_ch, np.nan, dtype=float)
    ci_95_high_hz = np.full(n_ch, np.nan, dtype=float)
    n_stim_trials_arr = np.zeros(n_ch, dtype=int)
    n_ctrl_trials_arr = np.zeros(n_ch, dtype=int)

    # 4. Compare STIM trial changes vs CTRL trial changes using two-sided Welch t-test
    for c in range(n_ch):
        s_c = stim_changes[c, :]
        c_c = ctrl_changes[c, :]

        s_clean = s_c[np.isfinite(s_c)]
        c_clean = c_c[np.isfinite(c_c)]

        n_s = len(s_clean)
        n_c = len(c_clean)
        n_stim_trials_arr[c] = n_s
        n_ctrl_trials_arr[c] = n_c

        if n_s == 0 or n_c == 0:
            continue

        mean_s = float(np.mean(s_clean))
        mean_c = float(np.mean(c_clean))
        stim_mean_delta_hz[c] = mean_s
        ctrl_mean_delta_hz[c] = mean_c

        diff_mean = mean_s - mean_c
        delta_effect_hz[c] = diff_mean

        if n_s < 2 or n_c < 2:
            raw_p_values[c] = 1.0
            welch_t_stats[c] = np.nan
            continue

        var_s = float(np.var(s_clean, ddof=1))
        var_c = float(np.var(c_clean, ddof=1))

        if var_s == 0.0 and var_c == 0.0:
            if np.isclose(mean_s, mean_c):
                welch_t_stats[c] = 0.0
                raw_p_values[c] = 1.0
                ci_95_low_hz[c] = diff_mean
                ci_95_high_hz[c] = diff_mean
            else:
                welch_t_stats[c] = np.nan
                raw_p_values[c] = 1.0
            continue

        # Two-sided Welch t-test
        res = stats.ttest_ind(s_clean, c_clean, equal_var=False, nan_policy="omit")
        t_val = float(res.statistic) if np.isfinite(res.statistic) else 0.0
        p_val = float(res.pvalue) if np.isfinite(res.pvalue) else 1.0
        welch_t_stats[c] = t_val
        raw_p_values[c] = p_val

        # 95% Confidence Interval for effect size via Welch-Satterthwaite approximation
        se_diff = math.sqrt(var_s / n_s + var_c / n_c)
        v_s = var_s / n_s
        v_c = var_c / n_c
        denom = ((v_s**2) / (n_s - 1)) + ((v_c**2) / (n_c - 1))
        if denom > 0 and se_diff > 0:
            df = ((v_s + v_c)**2) / denom
            t_crit = float(stats.t.ppf(0.975, df))
            ci_95_low_hz[c] = diff_mean - t_crit * se_diff
            ci_95_high_hz[c] = diff_mean + t_crit * se_diff

    # 5. Apply Benjamini-Hochberg FDR correction across channels
    bh_fdr_q_values = compute_benjamini_hochberg_fdr(raw_p_values)

    # Effect size magnitude
    mean_abs_delta_hz = np.abs(delta_effect_hz)

    # Mark responsive channels
    is_responsive = apply_responsiveness_criterion(
        raw_p_values,
        bh_fdr_q_values,
        mean_abs_delta_hz,
        criterion=criterion,
        alpha=alpha,
        p_thresh=p_thresh,
        min_delta_hz=min_delta_hz,
    )

    # Build per-channel report DataFrame
    ch_ids = channel_ids if channel_ids is not None else np.arange(1, n_ch + 1)
    df_metrics = pd.DataFrame({
        "channel_idx": np.arange(n_ch),
        "channel_id": ch_ids,
        "is_responsive": is_responsive,
        "welch_p_value": raw_p_values,
        "bh_fdr_q_value": bh_fdr_q_values,
        "welch_t_stat": welch_t_stats,
        "stim_mean_delta_hz": stim_mean_delta_hz,
        "ctrl_mean_delta_hz": ctrl_mean_delta_hz,
        "delta_effect_hz": delta_effect_hz,
        "ci_95_low_hz": ci_95_low_hz,
        "ci_95_high_hz": ci_95_high_hz,
        "n_stim_trials": n_stim_trials_arr,
        "n_ctrl_trials": n_ctrl_trials_arr,
    })

    stats_dict = {
        "is_responsive": is_responsive,
        "p_values": raw_p_values,
        "q_values": bh_fdr_q_values,
        "t_stats": welch_t_stats,
        "stim_mean_delta_hz": stim_mean_delta_hz,
        "ctrl_mean_delta_hz": ctrl_mean_delta_hz,
        "delta_effect_hz": delta_effect_hz,
        "ci_95_low_hz": ci_95_low_hz,
        "ci_95_high_hz": ci_95_high_hz,
        "mean_abs_delta_hz": mean_abs_delta_hz,
        "contrast_mean": observed_contrast_mean,
        "metrics_df": df_metrics,
        "n_resp": int(np.sum(is_responsive)),
        "n_sig_p": int(np.sum(raw_p_values <= p_thresh)),
        "n_sig_q": int(np.sum(bh_fdr_q_values <= alpha)),
    }

    return is_responsive, raw_p_values, bh_fdr_q_values, mean_abs_delta_hz, observed_contrast_mean, stats_dict


def compute_paired_signflip_responsiveness(
    paired_stim: np.ndarray,
    paired_ctrl: np.ndarray,
    time_ms: np.ndarray,
    response_win_ms: Tuple[float, float] = FULL_RESPONSE_WIN_MS,
    blank_mask: Optional[np.ndarray] = None,
    n_perms: int = N_PERMUTATIONS,
    alpha: float = FDR_ALPHA,
    criterion: str = RESPONSIVENESS_CRITERION,
    p_thresh: float = P_VAL_THRESH,
    min_delta_hz: float = MIN_DELTA_HZ,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Paired sign-flip permutation test on 1-to-1 matched trials (STIM vs CTRL label flip).
    
    Null hypothesis:
      Conditional on matched trial pairs, the sign of the difference Δr_{c, i}(t) is
      exchangeable (±1 with equal probability). There is no condition effect beyond
      trial-to-trial variance.

    Returns:
      is_responsive: (n_channels,) boolean array
      p_values: (n_channels,) raw permutation p-values
      q_values: (n_channels,) FDR-adjusted q-values
      mean_abs_delta_hz: (n_channels,) response window mean absolute contrast in Hz
      observed_contrast_mean: (n_channels, n_bins) trial-averaged contrast trace
    """
    n_ch, n_trials, n_bins = paired_stim.shape
    resp_mask = (time_ms >= response_win_ms[0]) & (time_ms <= response_win_ms[1])
    if blank_mask is not None:
        resp_mask = resp_mask & (~blank_mask)

    diff_observed = paired_stim - paired_ctrl  # (ch, trials, bins)
    observed_contrast_mean = np.nanmean(diff_observed, axis=1)  # (ch, bins)

    if n_trials < 3 or not np.any(resp_mask):
        _log(f"  [paired_signflip] Inadequate trials ({n_trials} < 3) or empty response window. Marking non-responsive.")
        return (
            np.zeros(n_ch, dtype=bool),
            np.ones(n_ch, dtype=float),
            np.ones(n_ch, dtype=float),
            np.zeros(n_ch, dtype=float),
            observed_contrast_mean,
        )

    # Observed absolute contrast integral across response window
    obs_integrals = np.nansum(np.abs(observed_contrast_mean[:, resp_mask]), axis=1)
    n_resp_bins = max(1, int(np.sum(resp_mask)))
    mean_abs_delta_hz = obs_integrals / n_resp_bins

    # Vectorized paired sign-flip permutations: multiply paired differences by ±1
    diff_resp = np.nan_to_num(diff_observed[:, :, resp_mask], nan=0.0)  # (ch, trials, resp_bins)
    rng = np.random.default_rng(rng_seed)
    signs = rng.choice([-1.0, 1.0], size=(n_trials, n_perms))
    perm_mean = np.tensordot(diff_resp, signs, axes=([1], [0])) / max(1, n_trials)
    perm_integrals = np.nansum(np.abs(perm_mean), axis=1)  # (ch, perms)

    # Compute empirical p-values
    p_values = np.zeros(n_ch, dtype=float)
    for c in range(n_ch):
        p_values[c] = (1.0 + float(np.sum(perm_integrals[c, :] >= obs_integrals[c]))) / (n_perms + 1.0)

    # Benjamini-Hochberg FDR correction
    q_values = compute_benjamini_hochberg_fdr(p_values)

    is_responsive = apply_responsiveness_criterion(
        p_values, q_values, mean_abs_delta_hz,
        criterion=criterion, alpha=alpha, p_thresh=p_thresh, min_delta_hz=min_delta_hz
    )

    return is_responsive, p_values, q_values, mean_abs_delta_hz, observed_contrast_mean


def compute_unpaired_label_shuffle_responsiveness(
    stim_rates: np.ndarray,
    ctrl_rates: np.ndarray,
    time_ms: np.ndarray,
    response_win_ms: Tuple[float, float] = FULL_RESPONSE_WIN_MS,
    blank_mask: Optional[np.ndarray] = None,
    n_perms: int = N_PERMUTATIONS,
    alpha: float = FDR_ALPHA,
    criterion: str = RESPONSIVENESS_CRITERION,
    p_thresh: float = P_VAL_THRESH,
    min_delta_hz: float = MIN_DELTA_HZ,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpaired STIM-vs-CTRL condition-label permutation responsiveness test.
    Uses all available STIM and CTRL trials without truncation when trial counts differ.

    Null hypothesis:
      Conditional on the observed set of trials, STIM and CTRL labels are exchangeable:
      firing-rate traces are drawn from the same condition-independent distribution.
      In practical terms, there is no condition-associated STIM-vs-CTRL firing-rate
      difference in the response window.

    Returns:
      is_responsive: (n_channels,) boolean array
      p_values: (n_channels,) raw permutation p-values
      q_values: (n_channels,) FDR-adjusted q-values
      mean_abs_delta_hz: (n_channels,) response window mean absolute contrast in Hz
      observed_contrast_mean: (n_channels, n_bins) unpaired contrast trace mean(STIM) - mean(CTRL)
    """
    n_ch, n_stim, n_bins = stim_rates.shape
    n_ctrl = ctrl_rates.shape[1]

    # Observed unpaired contrast: mean(STIM) - mean(CTRL)
    mean_stim_obs = np.nanmean(stim_rates, axis=1)  # (ch, bins)
    mean_ctrl_obs = np.nanmean(ctrl_rates, axis=1)  # (ch, bins)
    observed_contrast_mean = mean_stim_obs - mean_ctrl_obs  # (ch, bins)

    resp_mask = (time_ms >= response_win_ms[0]) & (time_ms <= response_win_ms[1])
    if blank_mask is not None:
        resp_mask = resp_mask & (~blank_mask)

    if n_stim < 2 or n_ctrl < 2 or not np.any(resp_mask):
        _log(f"  [unpaired_shuffle] Inadequate trials (stim={n_stim}, ctrl={n_ctrl}, require >= 2 each) or empty response window. Marking non-responsive.")
        return (
            np.zeros(n_ch, dtype=bool),
            np.ones(n_ch, dtype=float),
            np.ones(n_ch, dtype=float),
            np.zeros(n_ch, dtype=float),
            observed_contrast_mean,
        )

    # Two-sided response-magnitude statistic: T_obs = mean_t_in_response_window(|delta_obs|)
    obs_integrals = np.nansum(np.abs(observed_contrast_mean[:, resp_mask]), axis=1)
    n_resp_bins = max(1, int(np.sum(resp_mask)))
    mean_abs_delta_hz = obs_integrals / n_resp_bins

    # Pool STIM and CTRL trials across response window: shape (n_ch, n_tot, n_resp_bins)
    n_tot = n_stim + n_ctrl
    stim_resp = np.nan_to_num(stim_rates[:, :, resp_mask], nan=0.0)
    ctrl_resp = np.nan_to_num(ctrl_rates[:, :, resp_mask], nan=0.0)
    pooled_resp = np.concatenate([stim_resp, ctrl_resp], axis=1)  # (ch, n_tot, resp_bins)

    rng = np.random.default_rng(rng_seed)
    counts = np.zeros(n_ch, dtype=float)

    # Chunked permutation execution for memory efficiency and vectorized matrix multiplication
    chunk_size = min(500, n_perms)
    n_processed = 0

    while n_processed < n_perms:
        b_size = min(chunk_size, n_perms - n_processed)
        # Construct weight matrix for this chunk: shape (n_tot, b_size)
        weights = np.zeros((n_tot, b_size), dtype=float)
        for b in range(b_size):
            perm = rng.permutation(n_tot)
            weights[perm[:n_stim], b] = 1.0 / n_stim
            weights[perm[n_stim:], b] = -1.0 / n_ctrl

        # delta_perm has shape (n_ch, n_resp_bins, b_size)
        delta_perm = np.tensordot(pooled_resp, weights, axes=([1], [0]))
        # T_perm has shape (n_ch, b_size)
        T_perm = np.nanmean(np.abs(delta_perm), axis=1)

        counts += np.sum(T_perm >= mean_abs_delta_hz[:, None], axis=1)
        n_processed += b_size

    # Empirical p-values: (1 + count(T_perm >= T_obs)) / (1 + n_perms)
    p_values = (1.0 + counts) / (n_perms + 1.0)

    # Benjamini-Hochberg FDR correction
    q_values = compute_benjamini_hochberg_fdr(p_values)

    is_responsive = apply_responsiveness_criterion(
        p_values, q_values, mean_abs_delta_hz,
        criterion=criterion, alpha=alpha, p_thresh=p_thresh, min_delta_hz=min_delta_hz
    )

    return is_responsive, p_values, q_values, mean_abs_delta_hz, observed_contrast_mean


def compute_responsiveness_gating(
    paired_stim: Optional[np.ndarray],
    paired_ctrl: Optional[np.ndarray],
    stim_rates: Optional[np.ndarray],
    ctrl_rates: Optional[np.ndarray],
    time_ms: np.ndarray,
    baseline_win_ms: Tuple[float, float] = BASELINE_WIN_MS,
    response_win_ms: Tuple[float, float] = FULL_RESPONSE_WIN_MS,
    blank_mask: Optional[np.ndarray] = None,
    test_mode: str = RESPONSIVENESS_TEST_MODE,
    both_combination: str = RESPONSIVENESS_BOTH_COMBINATION,
    n_perms: int = N_PERMUTATIONS,
    alpha: float = FDR_ALPHA,
    criterion: str = RESPONSIVENESS_CRITERION,
    p_thresh: float = P_VAL_THRESH,
    min_delta_hz: float = MIN_DELTA_HZ,
    rng_seed: int = 42,
    channel_ids: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Unified responsiveness gating orchestrator.
    Supports:
      - "welch_baseline_corrected": trial-level baseline-corrected Welch two-sample t-test with BH-FDR (default).
      - "paired_signflip": paired sign-flip permutation test.
      - "unpaired_label_shuffle": unpaired condition-label shuffle permutation test.
      - "both": runs paired and unpaired permutation tests and combines them via both_combination ('intersection', 'union', 'paired_only', 'unpaired_only').

    Note on contrast_mean:
      The returned contrast_mean strictly remains the existing time-resolved STIM-minus-CTRL trace
      (paired or unpaired) so that downstream clustering, quadrant maps, and phenotype profiles
      continue to operate on time-resolved physiological waveforms regardless of the statistical gate.
    """
    mode = str(test_mode).strip().lower()
    comb = str(both_combination).strip().lower()
    crit = str(criterion).strip().lower()

    # Precompute existing time-resolved contrast trace to preserve for downstream clustering / plots
    paired_contrast = None
    if paired_stim is not None and paired_ctrl is not None:
        paired_contrast = np.nanmean(paired_stim - paired_ctrl, axis=1)

    unpaired_contrast = None
    if stim_rates is not None and ctrl_rates is not None:
        unpaired_contrast = np.nanmean(stim_rates, axis=1) - np.nanmean(ctrl_rates, axis=1)

    if CONTRAST_SOURCE_FOR_PLOTS == "unpaired" and unpaired_contrast is not None:
        time_resolved_contrast = unpaired_contrast
    elif paired_contrast is not None:
        time_resolved_contrast = paired_contrast
    elif unpaired_contrast is not None:
        time_resolved_contrast = unpaired_contrast
    else:
        n_ch_fallback = paired_stim.shape[0] if paired_stim is not None else (stim_rates.shape[0] if stim_rates is not None else 0)
        time_resolved_contrast = np.zeros((n_ch_fallback, len(time_ms)))

    n_ch = paired_stim.shape[0] if paired_stim is not None else (stim_rates.shape[0] if stim_rates is not None else len(time_resolved_contrast))

    run_welch = mode == "welch_baseline_corrected"
    run_paired = mode in ("paired_signflip", "both")
    run_unpaired = mode in ("unpaired_label_shuffle", "both")

    welch_res = None
    if run_welch:
        s_in = stim_rates if stim_rates is not None else paired_stim
        c_in = ctrl_rates if ctrl_rates is not None else paired_ctrl
        if s_in is not None and c_in is not None:
            welch_res = compute_welch_baseline_corrected_responsiveness(
                s_in, c_in, time_ms,
                baseline_win_ms=baseline_win_ms,
                response_win_ms=response_win_ms,
                blank_mask=blank_mask,
                alpha=alpha,
                criterion=criterion,
                p_thresh=p_thresh,
                min_delta_hz=min_delta_hz,
                contrast_trace=time_resolved_contrast,
                channel_ids=channel_ids,
            )

    paired_res = None
    if run_paired:
        if paired_stim is not None and paired_ctrl is not None:
            paired_res = compute_paired_signflip_responsiveness(
                paired_stim, paired_ctrl, time_ms,
                response_win_ms=response_win_ms, blank_mask=blank_mask,
                n_perms=n_perms, alpha=alpha, criterion=criterion,
                p_thresh=p_thresh, min_delta_hz=min_delta_hz, rng_seed=rng_seed,
            )

    unpaired_res = None
    if run_unpaired:
        if stim_rates is not None and ctrl_rates is not None:
            unpaired_res = compute_unpaired_label_shuffle_responsiveness(
                stim_rates, ctrl_rates, time_ms,
                response_win_ms=response_win_ms, blank_mask=blank_mask,
                n_perms=n_perms, alpha=alpha, criterion=criterion,
                p_thresh=p_thresh, min_delta_hz=min_delta_hz, rng_seed=rng_seed,
            )

    # Determine active gate and descriptions
    if mode == "welch_baseline_corrected":
        if welch_res is not None:
            is_responsive, p_vals, q_vals, mean_delta, contrast, welch_info = welch_res
        else:
            is_responsive = np.zeros(n_ch, dtype=bool)
            p_vals, q_vals = np.ones(n_ch), np.ones(n_ch)
            mean_delta = np.zeros(n_ch)
            contrast = time_resolved_contrast
            welch_info = None

        gate_desc = "Welch t-test on trial-level baseline-corrected STIM vs CTRL changes, BH-FDR q < 0.05"
        if alpha != 0.05:
            gate_desc = f"Welch t-test on trial-level baseline-corrected STIM vs CTRL changes, BH-FDR q < {alpha:.2f}"
        if crit in ("fdr_min_delta", "fdr_effect_size", "p_val"):
            gate_desc += f" & |Δ| >= {min_delta_hz:.1f} Hz"

    elif mode == "paired_signflip":
        if paired_res is not None:
            is_responsive, p_vals, q_vals, mean_delta, contrast = paired_res
        else:
            is_responsive = np.zeros(n_ch, dtype=bool)
            p_vals, q_vals = np.ones(n_ch), np.ones(n_ch)
            mean_delta = np.zeros(n_ch)
            contrast = time_resolved_contrast
        gate_desc = f"paired ({criterion})"
        welch_info = None

    elif mode == "unpaired_label_shuffle":
        if unpaired_res is not None:
            is_responsive, p_vals, q_vals, mean_delta, contrast = unpaired_res
        else:
            is_responsive = np.zeros(n_ch, dtype=bool)
            p_vals, q_vals = np.ones(n_ch), np.ones(n_ch)
            mean_delta = np.zeros(n_ch)
            contrast = time_resolved_contrast
        gate_desc = f"unpaired ({criterion})"
        welch_info = None

    else:  # "both"
        p_resp = paired_res[0] if paired_res is not None else np.zeros(n_ch, dtype=bool)
        u_resp = unpaired_res[0] if unpaired_res is not None else np.zeros(n_ch, dtype=bool)

        if comb == "intersection":
            is_responsive = p_resp & u_resp
            gate_desc = f"both/intersection ({criterion})"
        elif comb == "union":
            is_responsive = p_resp | u_resp
            gate_desc = f"both/union ({criterion})"
        elif comb == "unpaired_only":
            is_responsive = u_resp
            gate_desc = f"unpaired_only ({criterion})"
        else:  # "paired_only"
            is_responsive = p_resp
            gate_desc = f"paired_only ({criterion})"

        p_vals = paired_res[1] if paired_res is not None else (unpaired_res[1] if unpaired_res is not None else np.ones(n_ch))
        q_vals = paired_res[2] if paired_res is not None else (unpaired_res[2] if unpaired_res is not None else np.ones(n_ch))
        mean_delta = paired_res[3] if paired_res is not None else (unpaired_res[3] if unpaired_res is not None else np.zeros(n_ch))
        contrast = paired_res[4] if paired_res is not None else (unpaired_res[4] if unpaired_res is not None else time_resolved_contrast)
        welch_info = None

    gate_info = {
        "mode": mode,
        "combination": comb,
        "criterion": criterion,
        "gate_desc": gate_desc,
        "is_responsive": is_responsive,
        "contrast_mean": contrast,
        "p_values": p_vals,
        "q_values": q_vals,
        "mean_abs_delta_hz": mean_delta,
        "welch": welch_info,
        "paired": {
            "is_responsive": paired_res[0] if paired_res is not None else None,
            "p_values": paired_res[1] if paired_res is not None else None,
            "q_values": paired_res[2] if paired_res is not None else None,
            "mean_abs_delta_hz": paired_res[3] if paired_res is not None else None,
            "contrast_mean": paired_res[4] if paired_res is not None else None,
            "n_resp": int(np.sum(paired_res[0])) if paired_res is not None else 0,
            "n_sig_p": int(np.sum(paired_res[1] <= p_thresh)) if paired_res is not None else 0,
            "n_sig_q": int(np.sum(paired_res[2] <= alpha)) if paired_res is not None else 0,
        },
        "unpaired": {
            "is_responsive": unpaired_res[0] if unpaired_res is not None else None,
            "p_values": unpaired_res[1] if unpaired_res is not None else None,
            "q_values": unpaired_res[2] if unpaired_res is not None else None,
            "mean_abs_delta_hz": unpaired_res[3] if unpaired_res is not None else None,
            "contrast_mean": unpaired_res[4] if unpaired_res is not None else None,
            "n_resp": int(np.sum(unpaired_res[0])) if unpaired_res is not None else 0,
            "n_sig_p": int(np.sum(unpaired_res[1] <= p_thresh)) if unpaired_res is not None else 0,
            "n_sig_q": int(np.sum(unpaired_res[2] <= alpha)) if unpaired_res is not None else 0,
        },
    }

    return is_responsive, p_vals, q_vals, gate_info


# =============================================================================
# Dual-Normalization & Empirical Clustering
# =============================================================================

def dual_normalize_contrasts(
    contrast_mean: np.ndarray,
    diff_trials: np.ndarray,
    time_ms: np.ndarray,
    baseline_win_ms: Tuple[float, float] = BASELINE_WIN_MS,
    full_win_ms: Tuple[float, float] = FULL_RESPONSE_WIN_MS,
    blank_mask: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """
    Computes baseline-centered, unit-norm shape, and baseline-Z traces for each channel.
    The baseline Z-score is derived from trial-level pre-stimulus noise variability across trials and baseline time (Point 4).
    """
    n_ch, n_bins = contrast_mean.shape
    base_mask = (time_ms >= baseline_win_ms[0]) & (time_ms <= baseline_win_ms[1])
    resp_mask = (time_ms >= full_win_ms[0]) & (time_ms <= full_win_ms[1])
    if blank_mask is not None:
        base_mask = base_mask & (~blank_mask)
        resp_mask = resp_mask & (~blank_mask)

    # 1. Baseline centering (subtract mean baseline from contrast trace)
    base_means = np.nanmean(contrast_mean[:, base_mask], axis=1, keepdims=True)
    centered = contrast_mean - base_means

    # 2. Trial-level baseline standard deviation for Z-scoring (Point 4)
    # diff_trials has shape (n_ch, n_trials, n_bins)
    baseline_trials = diff_trials[:, :, base_mask]
    base_stds = np.nanstd(baseline_trials.reshape(n_ch, -1), axis=1, keepdims=True)
    baseline_z = centered / (base_stds + eps)

    # 3. Unit-norm shape normalization (temporal shape across response window)
    resp_traces = centered[:, resp_mask]
    norms = np.sqrt(np.nansum(resp_traces**2, axis=1, keepdims=True))
    unit_norm = centered / (norms + eps)

    return {
        "raw_centered": centered,
        "baseline_z": baseline_z,
        "unit_norm": unit_norm,
        "base_stds": base_stds,
        "resp_mask": resp_mask,
    }

def fit_empirical_clustering(
    unit_norm_traces: np.ndarray,
    k_range: range = CLUSTER_RANGE_K,
    method: str = LINKAGE_METHOD,
) -> Dict[str, Any]:
    """
    Performs Hierarchical Agglomerative Clustering (Ward's linkage).
    Evaluates Silhouette Score and Calinski-Harabasz Index across K in k_range.
    """
    n_samples = unit_norm_traces.shape[0]
    if n_samples < 4:
        # Fallback for very few responsive channels
        return {
            "linkage_matrix": None,
            "best_k": 1,
            "cluster_labels": np.zeros(n_samples, dtype=int),
            "silhouette_scores": {},
            "ch_scores": {},
        }

    # Ensure finite values - replace any residual NaNs or Infs with 0.0
    unit_norm_clean = np.nan_to_num(unit_norm_traces, nan=0.0, posinf=0.0, neginf=0.0)

    # Ward's linkage with defensive exception handling
    try:
        z = linkage(unit_norm_clean, method=method)
    except Exception as err:
        warnings.warn(f"Hierarchical clustering linkage failed: {err}. Falling back to single cluster.")
        return {
            "linkage_matrix": None,
            "best_k": 1,
            "cluster_labels": np.zeros(n_samples, dtype=int),
            "silhouette_scores": {},
            "ch_scores": {},
        }

    sil_scores = {}
    ch_scores = {}
    valid_k = [k for k in k_range if 2 <= k < n_samples]

    for k in valid_k:
        labels = fcluster(z, t=k, criterion="maxclust") - 1
        n_unique = len(np.unique(labels))
        if 1 < n_unique < n_samples:
            if silhouette_score is not None:
                try:
                    sil = float(silhouette_score(unit_norm_clean, labels))
                    sil_scores[k] = sil
                except Exception:
                    pass
            if calinski_harabasz_score is not None:
                try:
                    ch = float(calinski_harabasz_score(unit_norm_clean, labels))
                    ch_scores[k] = ch
                except Exception:
                    pass

    best_k = max(sil_scores, key=sil_scores.get) if sil_scores else min(DEFAULT_K, n_samples - 1)
    best_labels = fcluster(z, t=best_k, criterion="maxclust") - 1

    return {
        "linkage_matrix": z,
        "best_k": best_k,
        "cluster_labels": best_labels,
        "silhouette_scores": sil_scores,
        "ch_scores": ch_scores,
    }

def assign_phenotype_labels(
    cluster_means_raw: Dict[int, np.ndarray],
    time_ms: np.ndarray,
    early_win: Tuple[float, float] = EARLY_WIN_MS,
    late_win: Tuple[float, float] = LATE_WIN_MS,
    blank_mask: Optional[np.ndarray] = None,
) -> Dict[int, str]:
    """
    Assign interpretable physiological phenotype names to clusters based on mean temporal profile.
    """
    phenotypes = {}
    early_mask = (time_ms >= early_win[0]) & (time_ms <= early_win[1])
    late_mask = (time_ms >= late_win[0]) & (time_ms <= late_win[1])
    if blank_mask is not None:
        early_mask = early_mask & (~blank_mask)
        late_mask = late_mask & (~blank_mask)

    for k, trace in cluster_means_raw.items():
        e_val = np.nanmean(trace[early_mask]) if np.any(early_mask) else 0.0
        l_val = np.nanmean(trace[late_mask]) if np.any(late_mask) else 0.0
        e_val = 0.0 if not np.isfinite(e_val) else float(e_val)
        l_val = 0.0 if not np.isfinite(l_val) else float(l_val)
        peak_idx = np.nanargmax(np.abs(trace))
        peak_time = time_ms[peak_idx]

        if e_val > 5.0 and l_val > 5.0:
            name = "Sustained Excitation"
        elif e_val < -5.0 and l_val > 5.0:
            name = "Suppression to Rebound"
        elif e_val < -5.0 and l_val < -5.0:
            name = "Sustained Suppression"
        elif e_val > 5.0 and l_val < -2.0:
            name = "Transient Excitation"
        elif peak_time > 100.0 and trace[peak_idx] > 0:
            name = "Delayed Rebound"
        elif e_val > 0:
            name = "Weak Facilitation"
        else:
            name = "Weak Suppression"
        phenotypes[k] = f"Cluster {k+1}: {name}"

    return phenotypes


# =============================================================================
# Two-Window Early vs Late Metric Quadrant Map
# =============================================================================

def compute_early_late_quadrants(
    contrast_mean: np.ndarray,
    time_ms: np.ndarray,
    early_win: Tuple[float, float] = EARLY_WIN_MS,
    late_win: Tuple[float, float] = LATE_WIN_MS,
    blank_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute E_c and L_c window integrals per channel and assign quadrant ID (1..4).
    """
    early_mask = (time_ms >= early_win[0]) & (time_ms <= early_win[1])
    late_mask = (time_ms >= late_win[0]) & (time_ms <= late_win[1])
    if blank_mask is not None:
        early_mask = early_mask & (~blank_mask)
        late_mask = late_mask & (~blank_mask)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        e_vals = np.nanmean(contrast_mean[:, early_mask], axis=1) if np.any(early_mask) else np.zeros(contrast_mean.shape[0])
        l_vals = np.nanmean(contrast_mean[:, late_mask], axis=1) if np.any(late_mask) else np.zeros(contrast_mean.shape[0])
    e_vals = np.nan_to_num(e_vals, nan=0.0)
    l_vals = np.nan_to_num(l_vals, nan=0.0)

    # Quadrant:
    # Q1: (+, +) Sustained Excitation
    # Q2: (-, +) Suppression to Rebound
    # Q3: (-, -) Sustained Suppression
    # Q4: (+, -) Transient Excitation
    quads = np.zeros(len(e_vals), dtype=int)
    for i, (e, l) in enumerate(zip(e_vals, l_vals)):
        if e >= 0 and l >= 0:
            quads[i] = 1
        elif e < 0 and l >= 0:
            quads[i] = 2
        elif e < 0 and l < 0:
            quads[i] = 3
        else:
            quads[i] = 4

    return e_vals, l_vals, quads


# =============================================================================
# Plotting Implementations (Figures 1 - 5)
# =============================================================================

def plot_fig1_dendrogram_and_metrics(
    linkage_matrix: np.ndarray,
    sil_scores: Dict[int, float],
    ch_scores: Dict[int, float],
    best_k: int,
    out_path: Path,
    title_suffix: str = "",
):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    
    # 1. Dendrogram with branches colored by the chosen cut threshold
    axes[0].set_title("Response Profile Dendrogram", fontsize=11, fontweight="bold")
    thresh = 0.0
    if linkage_matrix is not None and len(linkage_matrix) >= best_k > 1:
        d_merge = float(linkage_matrix[-best_k + 1, 2])
        d_prev = float(linkage_matrix[-best_k, 2]) if len(linkage_matrix) > best_k else 0.0
        thresh = 0.5 * (d_merge + d_prev)
    dendrogram(linkage_matrix, ax=axes[0], no_labels=True, color_threshold=thresh, above_threshold_color="#888888")
    if thresh > 0:
        axes[0].axhline(thresh, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8, label=f"Cut: K={best_k}")
        axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[0].set_xlabel("Responsive Channels")
    axes[0].set_ylabel("Ward Distance")

    # 2. Silhouette Score
    axes[1].set_title("Silhouette Score vs K", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Mean Silhouette Score")
    if sil_scores:
        k_vals = sorted(sil_scores.keys())
        s_vals = [sil_scores[k] for k in k_vals]
        axes[1].plot(k_vals, s_vals, "o-", color="#1f77b4", linewidth=2.0, markersize=6)
        axes[1].axvline(best_k, color="#d62728", linestyle="--", linewidth=1.2, label=f"Optimal K={best_k}")
        axes[1].set_xticks(k_vals)
        axes[1].legend(loc="best", frameon=False, fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "Silhouette score unavailable", ha="center", va="center", transform=axes[1].transAxes, color="0.5")

    # 3. Calinski-Harabasz Index
    axes[2].set_title("Calinski-Harabasz Index vs K", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Number of Clusters (K)")
    axes[2].set_ylabel("CH Index")
    if ch_scores:
        k_vals = sorted(ch_scores.keys())
        c_vals = [ch_scores[k] for k in k_vals]
        axes[2].plot(k_vals, c_vals, "s-", color="#2ca02c", linewidth=2.0, markersize=6)
        axes[2].axvline(best_k, color="#d62728", linestyle="--", linewidth=1.2, label=f"Optimal K={best_k}")
        axes[2].set_xticks(k_vals)
        axes[2].legend(loc="best", frameon=False, fontsize=9)
    else:
        axes[2].text(0.5, 0.5, "CH index unavailable", ha="center", va="center", transform=axes[2].transAxes, color="0.5")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, linestyle=":")

    fig.suptitle(f"Empirical Cluster Selection {title_suffix}", fontsize=13, fontweight="bold", y=1.02)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
    fig.savefig(out_path, dpi=DPI_OUTPUT, bbox_inches="tight")
    plt.close(fig)

def plot_fig2_cluster_phenotype_profiles(
    time_ms: np.ndarray,
    unit_norm_traces: np.ndarray,
    raw_hz_traces: np.ndarray,
    cluster_labels: np.ndarray,
    phenotype_names: Dict[int, str],
    stim_dur_ms: float,
    out_path: Path,
    title_suffix: str = "",
    blank_win: Optional[Tuple[float, float]] = None,
    kinematics_data: Optional[Dict[str, Any]] = None,
    stim_rates: Optional[np.ndarray] = None,
    ctrl_rates: Optional[np.ndarray] = None,
    rest_rates: Optional[np.ndarray] = None,
    trial_counts: Optional[Dict[str, Any]] = None,
):
    k_unique = sorted(np.unique(cluster_labels))
    n_k = len(k_unique)

    has_variability = False
    has_dist = False
    cam_label = "Camera 1"
    stim_dist = None
    ctrl_dist = None
    if kinematics_data is not None:
        stim_var_curve = kinematics_data.get("stim_variability_curve")
        ctrl_var_curve = kinematics_data.get("ctrl_variability_curve")
        stim_dist = kinematics_data.get("stim_cam1_dist")
        ctrl_dist = kinematics_data.get("ctrl_cam1_dist")
        cam_label = kinematics_data.get("cam_label", "Camera 1")

        has_variability = (stim_var_curve is not None and np.any(np.isfinite(stim_var_curve))) or \
                          (ctrl_var_curve is not None and np.any(np.isfinite(ctrl_var_curve)))
        has_dist = (stim_dist is not None and np.any(np.isfinite(stim_dist))) or \
                   (ctrl_dist is not None and np.any(np.isfinite(ctrl_dist)))

    has_kin = has_variability or has_dist
    has_overlay = (stim_rates is not None and ctrl_rates is not None)
    n_cols = 3 if has_overlay else 2
    n_rows = n_k + (1 if has_kin else 0)

    fig = plt.figure(figsize=(18 if has_overlay else 13, 2.8 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.25)

    ax_ref = None

    for i, k in enumerate(k_unique):
        mask = cluster_labels == k
        n_members = np.sum(mask)

        # 1. Unit-norm shape profile
        ax_shape = fig.add_subplot(gs[i, 0], sharex=ax_ref)
        if ax_ref is None:
            ax_ref = ax_shape

        k_shape = unit_norm_traces[mask, :]
        mean_shape = np.nanmean(k_shape, axis=0)
        sem_shape = np.nanstd(k_shape, axis=0) / math.sqrt(max(1, n_members))

        ax_shape.plot(time_ms, mean_shape, color=CLUSTER_PALETTE[k % len(CLUSTER_PALETTE)], linewidth=2.0)
        ax_shape.fill_between(time_ms, mean_shape - sem_shape, mean_shape + sem_shape,
                              where=np.isfinite(mean_shape),
                              color=CLUSTER_PALETTE[k % len(CLUSTER_PALETTE)], alpha=0.20)
        if blank_win is not None:
            ax_shape.axvspan(blank_win[0], blank_win[1], color="0.90", alpha=0.55, label="Artifact Blanking")
        else:
            ax_shape.axvspan(0, stim_dur_ms, color=STIM_REGION_COLOR, alpha=STIM_REGION_ALPHA)
        ax_shape.axhline(0, color="#666666", linestyle=":", alpha=0.7, linewidth=1.0)
        ax_shape.set_ylabel("Shape (Norm)")
        ax_shape.set_title(f"{phenotype_names.get(k, f'Cluster {k+1}')} (n={n_members})", fontsize=10, fontweight="bold")

        # 2. Raw Hz Effect Size
        ax_hz = fig.add_subplot(gs[i, 1], sharex=ax_ref)
        k_hz = raw_hz_traces[mask, :]
        mean_hz = np.nanmean(k_hz, axis=0)
        sem_hz = np.nanstd(k_hz, axis=0) / math.sqrt(max(1, n_members))

        ax_hz.plot(time_ms, mean_hz, color=CLUSTER_PALETTE[k % len(CLUSTER_PALETTE)], linewidth=2.0)
        ax_hz.fill_between(time_ms, mean_hz - sem_hz, mean_hz + sem_hz,
                           where=np.isfinite(mean_hz),
                           color=CLUSTER_PALETTE[k % len(CLUSTER_PALETTE)], alpha=0.20)
        if blank_win is not None:
            ax_hz.axvspan(blank_win[0], blank_win[1], color="0.90", alpha=0.55, label="Artifact Blanking")
        else:
            ax_hz.axvspan(0, stim_dur_ms, color=STIM_REGION_COLOR, alpha=STIM_REGION_ALPHA)
        ax_hz.axhline(0, color="#666666", linestyle=":", alpha=0.7, linewidth=1.0)
        ax_hz.set_ylabel("Δ Rate (Hz)")
        ax_hz.set_title("Raw Effect Size (Mean ± SEM Hz)", fontsize=10)

        # 3. Firing Rate Overlay (Stim vs Control Pooling Cluster Channels)
        if has_overlay:
            ax_overlay = fig.add_subplot(gs[i, 2], sharex=ax_ref)
            k_stim = stim_rates[mask, :]
            k_ctrl = ctrl_rates[mask, :]
            stim_pool_m = np.nanmean(k_stim, axis=0)
            stim_pool_s = np.nanstd(k_stim, axis=0) / math.sqrt(max(1, n_members))
            ctrl_pool_m = np.nanmean(k_ctrl, axis=0)
            ctrl_pool_s = np.nanstd(k_ctrl, axis=0) / math.sqrt(max(1, n_members))

            s_tr = trial_counts.get("stim") if trial_counts else None
            c_tr = trial_counts.get("ctrl") if trial_counts else None
            r_tr = trial_counts.get("rest") if trial_counts else None

            s_lbl = f"Stim (n={n_members} ch, {s_tr} tr)" if s_tr is not None else f"Stim (n={n_members} ch)"
            c_lbl = f"Control ({c_tr} tr)" if c_tr is not None else "Control"
            r_lbl = f"Rest ({r_tr} tr)" if r_tr is not None else "Rest"

            ax_overlay.plot(time_ms, stim_pool_m, color="#1f77b4", linewidth=2.0, label=s_lbl)
            ax_overlay.fill_between(time_ms, stim_pool_m - stim_pool_s, stim_pool_m + stim_pool_s,
                                    where=np.isfinite(stim_pool_m),
                                    color="#1f77b4", alpha=0.20)

            ax_overlay.plot(time_ms, ctrl_pool_m, color="#ff7f0e", linewidth=2.0, linestyle="--", label=c_lbl)
            ax_overlay.fill_between(time_ms, ctrl_pool_m - ctrl_pool_s, ctrl_pool_m + ctrl_pool_s,
                                    where=np.isfinite(ctrl_pool_m),
                                    color="#ff7f0e", alpha=0.18)

            if rest_rates is not None:
                k_rest = rest_rates[mask, :]
                rest_pool_m = np.nanmean(k_rest, axis=0)
                rest_pool_s = np.nanstd(k_rest, axis=0) / math.sqrt(max(1, n_members))
                if np.any(np.isfinite(rest_pool_m)):
                    ax_overlay.plot(time_ms, rest_pool_m, color="#2ca02c", linewidth=1.6, linestyle=":", label=r_lbl)
                    ax_overlay.fill_between(time_ms, rest_pool_m - rest_pool_s, rest_pool_m + rest_pool_s,
                                            where=np.isfinite(rest_pool_m),
                                            color="#2ca02c", alpha=0.15)

            if blank_win is not None:
                ax_overlay.axvspan(blank_win[0], blank_win[1], color="0.90", alpha=0.55, label="_nolegend_")
            else:
                ax_overlay.axvspan(0, stim_dur_ms, color=STIM_REGION_COLOR, alpha=STIM_REGION_ALPHA, label="_nolegend_")

            ax_overlay.set_ylabel("Rate (Hz)")
            ax_overlay.set_title("Firing Rate Overlay (Mean ± SEM Hz)", fontsize=10)
            ax_overlay.set_ylim(bottom=0)
            ax_overlay.legend(loc="upper right", frameon=False, fontsize=8)

        cur_axes = [ax_shape, ax_hz] + ([ax_overlay] if has_overlay else [])
        for a in cur_axes:
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            a.grid(True, alpha=0.15, linestyle=":")
            if i < n_rows - 1:
                a.tick_params(labelbottom=False)

    # 4. Dedicated Bottom Kinematics Panel (Row n_k)
    if has_kin and kinematics_data is not None:
        stim_t = kinematics_data.get("stim_beh_t")
        ctrl_t = kinematics_data.get("ctrl_beh_t")
        if ctrl_t is None:
            ctrl_t = stim_t

        s_tr = trial_counts.get("stim") if trial_counts else None
        c_tr = trial_counts.get("ctrl") if trial_counts else None

        def _calc_mean_sem(dist_arr):
            if dist_arr is None or getattr(dist_arr, "size", 0) == 0:
                return None, None
            arr = np.asarray(dist_arr, dtype=float)
            if arr.ndim == 1:
                arr = arr[None, :]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                m = np.nanmean(arr, axis=0)
                n = np.sum(np.isfinite(arr), axis=0)
                s = np.nanstd(arr, axis=0)
                sem = np.where(n > 0, s / np.sqrt(np.maximum(1, n)), np.nan)
            return m, sem

        def _plot_var_ax(ax):
            stim_var_curve = kinematics_data.get("stim_variability_curve")
            stim_var_lower = kinematics_data.get("stim_variability_lower")
            stim_var_upper = kinematics_data.get("stim_variability_upper")

            ctrl_var_curve = kinematics_data.get("ctrl_variability_curve")
            ctrl_var_lower = kinematics_data.get("ctrl_variability_lower")
            ctrl_var_upper = kinematics_data.get("ctrl_variability_upper")

            s_n_tr = kinematics_data.get("stim_n_trials", s_tr)
            c_n_tr = kinematics_data.get("ctrl_n_trials", c_tr)

            s_lbl = f"Stim Reach ({s_n_tr} tr)" if s_n_tr is not None else "Stim Reach"
            c_lbl = f"Control Reach ({c_n_tr} tr)" if c_n_tr is not None else "Control Reach"

            if stim_var_curve is not None and stim_t is not None:
                ax.plot(stim_t, stim_var_curve, color="#1f77b4", linewidth=2.0, label=s_lbl)
                if stim_var_lower is not None and stim_var_upper is not None:
                    valid_s = np.isfinite(stim_var_lower) & np.isfinite(stim_var_upper)
                    ax.fill_between(stim_t, stim_var_lower, stim_var_upper, where=valid_s, color="#1f77b4", alpha=0.20)

            if ctrl_var_curve is not None and ctrl_t is not None:
                ax.plot(ctrl_t, ctrl_var_curve, color="#ff7f0e", linewidth=2.0, linestyle="--", label=c_lbl)
                if ctrl_var_lower is not None and ctrl_var_upper is not None:
                    valid_c = np.isfinite(ctrl_var_lower) & np.isfinite(ctrl_var_upper)
                    ax.fill_between(ctrl_t, ctrl_var_lower, ctrl_var_upper, where=valid_c, color="#ff7f0e", alpha=0.18)

            if blank_win is not None:
                ax.axvspan(blank_win[0], blank_win[1], color="0.90", alpha=0.55, label="Artifact Blanking")
            else:
                ax.axvspan(0, stim_dur_ms, color=STIM_REGION_COLOR, alpha=STIM_REGION_ALPHA)

            ax.set_ylim(bottom=0)
            ax.set_ylabel("Pairwise Distance")
            ax.set_title(f"Kinematics: Pairwise X-Y Variability ({cam_label})", fontsize=10, fontweight="bold")
            ax.set_xlabel("Time from Stim Onset (ms)")
            ax.legend(loc="upper left", frameon=False, fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.15, linestyle=":")

        def _plot_dist_ax(ax):
            s_mean, s_sem = _calc_mean_sem(stim_dist)
            c_mean, c_sem = _calc_mean_sem(ctrl_dist)

            n_s_kin = stim_dist.shape[0] if (stim_dist is not None and getattr(stim_dist, "ndim", 0) >= 2) else s_tr
            n_c_kin = ctrl_dist.shape[0] if (ctrl_dist is not None and getattr(ctrl_dist, "ndim", 0) >= 2) else c_tr

            s_kin_lbl = f"Stim Reach ({n_s_kin} tr)" if n_s_kin is not None else "Stim Reach"
            c_kin_lbl = f"Control Reach ({n_c_kin} tr)" if n_c_kin is not None else "Control Reach"

            if s_mean is not None and stim_t is not None:
                ax.plot(stim_t, s_mean, color="#1f77b4", linewidth=2.0, label=s_kin_lbl)
                valid_s = np.isfinite(s_mean) & np.isfinite(s_sem)
                ax.fill_between(stim_t, s_mean - s_sem, s_mean + s_sem, where=valid_s, color="#1f77b4", alpha=0.25)

            if c_mean is not None and ctrl_t is not None:
                ax.plot(ctrl_t, c_mean, color="#ff7f0e", linewidth=2.0, linestyle="--", label=c_kin_lbl)
                valid_c = np.isfinite(c_mean) & np.isfinite(c_sem)
                ax.fill_between(ctrl_t, c_mean - c_sem, c_mean + c_sem, where=valid_c, color="#ff7f0e", alpha=0.20)

            if blank_win is not None:
                ax.axvspan(blank_win[0], blank_win[1], color="0.90", alpha=0.55, label="Artifact Blanking")
            else:
                ax.axvspan(0, stim_dur_ms, color=STIM_REGION_COLOR, alpha=STIM_REGION_ALPHA)

            ax.axhline(0, color="#666666", linestyle=":", alpha=0.7, linewidth=1.0)
            ax.axhline(1, color="#888888", linestyle=":", alpha=0.5, linewidth=0.8)
            ax.set_ylim(-0.1, 1.15)
            ax.set_ylabel("Norm. Distance")
            ax.set_title(f"Kinematics: Norm. Distance ({cam_label})", fontsize=10, fontweight="bold")
            ax.set_xlabel("Time from Stim Onset (ms)")
            ax.legend(loc="upper left", frameon=False, fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.15, linestyle=":")

        if has_variability and has_dist:
            gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[n_k, :], wspace=0.22)
            ax_var = fig.add_subplot(gs_bottom[0, 0], sharex=ax_ref)
            _plot_var_ax(ax_var)

            ax_dist = fig.add_subplot(gs_bottom[0, 1], sharex=ax_ref)
            _plot_dist_ax(ax_dist)
        elif has_variability:
            ax_var = fig.add_subplot(gs[n_k, :], sharex=ax_ref)
            _plot_var_ax(ax_var)
        elif has_dist:
            ax_dist = fig.add_subplot(gs[n_k, :], sharex=ax_ref)
            _plot_dist_ax(ax_dist)
    else:
        # No kinematics: label x-axis on bottom neural row
        if ax_ref is not None:
            ax_shape.set_xlabel("Time from Stim Onset (ms)")
            ax_hz.set_xlabel("Time from Stim Onset (ms)")
            if has_overlay:
                ax_overlay.set_xlabel("Time from Stim Onset (ms)")

    if ax_ref is not None:
        ax_ref.set_xlim(time_ms[0], time_ms[-1])

    trial_str = format_trial_info(trial_counts)
    title_main = (
        "Response Phenotype Profiles, Overlays & Kinematics"
        if (has_kin and has_overlay)
        else (
            "Response Phenotype Profiles & Overlays"
            if has_overlay
            else ("Response Phenotype Profiles & Kinematics" if has_kin else "Response Phenotype Profiles")
        )
    )
    full_title = f"{title_main} {title_suffix}" + (f" | {trial_str}" if trial_str else "")
    fig.suptitle(full_title, fontsize=13, fontweight="bold", y=1.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
    fig.savefig(out_path, dpi=DPI_OUTPUT, bbox_inches="tight")
    plt.close(fig)

def plot_fig3_cluster_sorted_heatmaps(
    time_ms: np.ndarray,
    raw_hz_traces: np.ndarray,
    cluster_labels: np.ndarray,
    nonresp_hz_traces: Optional[np.ndarray],
    stim_dur_ms: float,
    out_path: Path,
    title_suffix: str = "",
    blank_win: Optional[Tuple[float, float]] = None,
    post_stim_start: Optional[float] = None,
    nonresp_label: Optional[str] = None,
):
    # Sort responsive channels by cluster, then by peak response latency in post-stim window
    p_start = post_stim_start if post_stim_start is not None else (blank_win[1] if blank_win is not None else 0.0)
    resp_mask_time = (time_ms >= p_start) & (time_ms <= p_start + 300.0)
    if len(cluster_labels) > 0 and len(raw_hz_traces) > 0:
        sort_keys = []
        for i, k in enumerate(cluster_labels):
            tr = raw_hz_traces[i]
            sub_tr = np.abs(tr[resp_mask_time])
            if np.any(np.isfinite(sub_tr)):
                peak_lat = time_ms[resp_mask_time][np.nanargmax(sub_tr)]
            else:
                peak_lat = p_start
            sort_keys.append((k, peak_lat))

        sorted_indices = sorted(range(len(cluster_labels)), key=lambda x: sort_keys[x])
        sorted_resp_traces = raw_hz_traces[sorted_indices]
    else:
        sorted_resp_traces = np.empty((0, len(time_ms)))

    # Combine with non-responsive channels at the bottom
    if nonresp_hz_traces is not None and len(nonresp_hz_traces) > 0:
        if len(sorted_resp_traces) > 0:
            all_traces = np.vstack([sorted_resp_traces, nonresp_hz_traces])
            divider_row = len(sorted_resp_traces)
        else:
            all_traces = nonresp_hz_traces
            divider_row = None
    elif len(sorted_resp_traces) > 0:
        all_traces = sorted_resp_traces
        divider_row = None
    else:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    vmax = np.nanpercentile(np.abs(all_traces), 98)
    vmin = -vmax if vmax > 0 else -10
    vmax = vmax if vmax > 0 else 10

    cmap = plt.colormaps.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#e0e0e0")  # Explicit light gray for blanked artifact window

    im = ax.imshow(all_traces, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[time_ms[0], time_ms[-1], all_traces.shape[0], 0])
    if blank_win is not None:
        ax.axvspan(blank_win[0], blank_win[1], color="0.85", alpha=0.45, hatch="//", label="Artifact Blanking")
    else:
        ax.axvspan(0, stim_dur_ms, color=STIM_REGION_COLOR, alpha=0.25)
    ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
    
    lbl = nonresp_label if nonresp_label is not None else "Non-responsive"
    if divider_row is not None:
        ax.axhline(divider_row, color="#333333", linewidth=1.5, linestyle="--")
        ax.text(time_ms[0] + 15, divider_row + 4, lbl, color="#444444", fontsize=9, fontstyle="italic")
    elif len(sorted_resp_traces) == 0:
        ax.text(time_ms[0] + 15, max(1, all_traces.shape[0] - 2), f"All Channels {lbl}", color="#444444", fontsize=9, fontstyle="italic")

    ax.set_xlabel("Time from Stim Onset (ms)")
    ax.set_ylabel("Sorted Channel Index")
    ax.set_title(f"Stimulation Contrast Heatmap {title_suffix}", fontsize=12, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Contrast Δr(t) (Hz)")
    cbar.outline.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
    fig.savefig(out_path, dpi=DPI_OUTPUT, bbox_inches="tight")
    plt.close(fig)

def plot_fig4_early_late_quadrant_scatter(
    e_vals: np.ndarray,
    l_vals: np.ndarray,
    cluster_labels: np.ndarray,
    nonresp_mask: np.ndarray,
    phenotype_names: Dict[int, str],
    out_path: Path,
    title_suffix: str = "",
    early_win: Tuple[float, float] = EARLY_WIN_MS,
    late_win: Tuple[float, float] = LATE_WIN_MS,
    nonresp_label: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    lbl = nonresp_label if nonresp_label is not None else "Non-responsive"
    # Plot Non-responsive
    if np.any(nonresp_mask):
        ax.scatter(e_vals[nonresp_mask], l_vals[nonresp_mask], color="#c0c0c0", alpha=0.5, s=28, edgecolors="none", label=lbl)

    # Plot Clusters
    resp_indices = np.where(~nonresp_mask)[0]
    for k in sorted(np.unique(cluster_labels)):
        members = [idx for i, idx in enumerate(resp_indices) if cluster_labels[i] == k]
        if members:
            ax.scatter(
                e_vals[members], l_vals[members],
                color=CLUSTER_PALETTE[k % len(CLUSTER_PALETTE)],
                s=55, edgecolors="white", linewidth=0.8,
                label=f"{phenotype_names.get(k, f'Cluster {k+1}')} (n={len(members)})",
                zorder=4,
            )

    # Clean, subtle Quadrant Dividers
    ax.axhline(0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.7, zorder=2)
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.7, zorder=2)

    # Subtle corner quadrant annotations in margins (clean and not intrusive)
    ax.text(0.98, 0.98, "Q1: Excitation (+, +)", transform=ax.transAxes, ha="right", va="top", fontsize=9, color="0.45", fontstyle="italic")
    ax.text(0.02, 0.98, "Q2: Rebound (-, +)", transform=ax.transAxes, ha="left", va="top", fontsize=9, color="0.45", fontstyle="italic")
    ax.text(0.02, 0.02, "Q3: Suppression (-, -)", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color="0.45", fontstyle="italic")
    ax.text(0.98, 0.02, "Q4: Transient (+, -)", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="0.45", fontstyle="italic")

    ax.set_xlabel(f"Early Window Contrast $E_c$ (Hz) [{early_win[0]:.0f} - {early_win[1]:.0f} ms]", fontsize=11)
    ax.set_ylabel(f"Late Window Contrast $L_c$ (Hz) [{late_win[0]:.0f} - {late_win[1]:.0f} ms]", fontsize=11)
    ax.set_title(f"Early vs Late Modulation Quadrants {title_suffix}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.15, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, framealpha=0.9, edgecolor="0.85", fontsize=9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
    fig.savefig(out_path, dpi=DPI_OUTPUT, bbox_inches="tight")
    plt.close(fig)

def plot_fig5_context_interaction(
    time_ms: np.ndarray,
    interaction_traces: np.ndarray,
    reach_contrasts: np.ndarray,
    rest_contrasts: np.ndarray,
    stim_dur_ms: float,
    out_path: Path,
    title_suffix: str = "",
    blank_win: Optional[Tuple[float, float]] = None,
    post_stim_start: Optional[float] = None,
):
    """
    Fig 5: Reach vs Rest context interaction.
    Δ_interaction(t) = [r_STIM - r_CTRL]_reach - [r_STIM - r_baseline]_rest
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Interaction contrast mean ± SEM
    mean_int = np.nanmean(interaction_traces, axis=0)
    sem_int = np.nanstd(interaction_traces, axis=0) / math.sqrt(max(1, len(interaction_traces)))

    axes[0].plot(time_ms, mean_int, color="darkslateblue", linewidth=2.0, label="Δ_interaction(t)")
    axes[0].fill_between(time_ms, mean_int - sem_int, mean_int + sem_int,
                         where=np.isfinite(mean_int),
                         color="darkslateblue", alpha=0.20)
    if blank_win is not None:
        axes[0].axvspan(blank_win[0], blank_win[1], color="0.9", alpha=0.55, label="Artifact Blanking")
    else:
        axes[0].axvspan(0, stim_dur_ms, color=STIM_REGION_COLOR, alpha=STIM_REGION_ALPHA)
    axes[0].axhline(0, color="#666666", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Time from Stim Onset (ms)")
    axes[0].set_ylabel("Interaction Rate Difference (Hz)")
    axes[0].set_title("Context Interaction Trace: Reach vs Rest", fontsize=11, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=9)

    # 2. Scatter: Peak Reach Modulation vs Peak Rest Modulation
    p_start = post_stim_start if post_stim_start is not None else (blank_win[1] if blank_win is not None else 0.0)
    post_mask = (time_ms >= p_start) & (time_ms <= p_start + 250.0) & np.isfinite(mean_int)
    peak_reach = np.nanmax(np.abs(reach_contrasts[:, post_mask]), axis=1) if np.any(post_mask) else np.zeros(len(reach_contrasts))
    peak_rest = np.nanmax(np.abs(rest_contrasts[:, post_mask]), axis=1) if np.any(post_mask) else np.zeros(len(rest_contrasts))

    axes[1].scatter(peak_rest, peak_reach, color="#2b7b8b", s=45, edgecolors="white", linewidth=0.8, alpha=0.8)
    lim_max = max(np.nanmax(peak_reach), np.nanmax(peak_rest)) * 1.1 if len(peak_reach) > 0 and len(peak_rest) > 0 else 10.0
    axes[1].plot([0, lim_max], [0, lim_max], "r--", linewidth=1.0, label="Unity (Context-Robust)")
    axes[1].set_xlabel("Peak Modulation at Rest (Hz)", fontsize=11)
    axes[1].set_ylabel("Peak Modulation during Reach (Hz)", fontsize=11)
    axes[1].set_title("Behavioral Gating: Reach vs Rest Gain", fontsize=11, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=9)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(True, alpha=0.15, linestyle=":")

    fig.suptitle(f"Context Robustness vs Behavioral Gating {title_suffix}", fontsize=13, fontweight="bold", y=1.02)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
    fig.savefig(out_path, dpi=DPI_OUTPUT, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# High-Level Session Pipeline Execution
# =============================================================================

def process_single_condition(
    stim_path: Path,
    ctrl_path: Optional[Path],
    rest_path: Optional[Path],
    bad_channels: Dict[str, set],
    out_dir: Path,
    file_idx: Optional[int] = None,
    total_files: Optional[int] = None,
):
    if file_idx is not None and total_files is not None:
        width = len(str(total_files))
        prefix = f"[{file_idx:>{width}}/{total_files}] "
    else:
        prefix = ""
    _log(f"{prefix}Processing condition: {stim_path.name}")
    ensure_dir(out_dir)

    with np.load(stim_path, allow_pickle=True) as stim_npz:
        stim_dur_ms = get_stim_duration_ms(stim_npz)
        port = get_ua_port(stim_npz)
        elec_info, region_grids = load_electrode_mapping(PARAMS.monkey if PARAMS is not None else None)

        # Load time vectors and raw counts
        nprw_counts = safe_get_npz(stim_npz, "NPRW_counts", None)
        nprw_t = safe_get_npz(stim_npz, "NPRW_rel_t", None)
        ua_counts = safe_get_npz(stim_npz, "UA_counts", None)
        ua_t = safe_get_npz(stim_npz, "UA_rel_t", None)
        ua_ids = safe_get_npz(stim_npz, "ua_ids_1based", None)
        ua_region = safe_get_npz(stim_npz, "ua_region", None)
        ua_region_names = safe_get_npz(stim_npz, "ua_region_names", None)
        stim_vel = safe_get_npz(stim_npz, "beh_cam0_vel_segs", None)
        stim_beh_t = safe_get_npz(stim_npz, "beh_rel_t", None)
        stim_cam0_segs = safe_get_npz(stim_npz, "beh_cam0_segs", None)
        stim_cam1_segs = safe_get_npz(stim_npz, "beh_cam1_segs", None)
        stim_cam0_names = safe_get_npz(stim_npz, "beh_cam0_names", None)
        stim_cam1_names = safe_get_npz(stim_npz, "beh_cam1_names", None)

    ctrl_nprw_counts = None
    ctrl_nprw_t = None
    ctrl_ua_counts = None
    ctrl_ua_t = None
    ctrl_ua_ids = None
    ctrl_ua_reg = None
    ctrl_ua_names = None
    ctrl_port = None
    ctrl_vel = None
    ctrl_beh_t = None
    ctrl_cam0_segs = None
    ctrl_cam1_segs = None
    ctrl_cam0_names = None
    ctrl_cam1_names = None
    if ctrl_path and ctrl_path.exists():
        with np.load(ctrl_path, allow_pickle=True) as ctrl_npz:
            ctrl_nprw_counts = safe_get_npz(ctrl_npz, "NPRW_counts", None)
            ctrl_nprw_t = safe_get_npz(ctrl_npz, "NPRW_rel_t", None)
            ctrl_ua_counts = safe_get_npz(ctrl_npz, "UA_counts", None)
            ctrl_ua_t = safe_get_npz(ctrl_npz, "UA_rel_t", None)
            ctrl_ua_ids = safe_get_npz(ctrl_npz, "ua_ids_1based", None)
            ctrl_ua_reg = safe_get_npz(ctrl_npz, "ua_region", None)
            ctrl_ua_names = safe_get_npz(ctrl_npz, "ua_region_names", None)
            ctrl_port = get_ua_port(ctrl_npz)
            ctrl_vel = safe_get_npz(ctrl_npz, "beh_cam0_vel_segs", None)
            ctrl_beh_t = safe_get_npz(ctrl_npz, "beh_rel_t", None)
            ctrl_cam0_segs = safe_get_npz(ctrl_npz, "beh_cam0_segs", None)
            ctrl_cam1_segs = safe_get_npz(ctrl_npz, "beh_cam1_segs", None)
            ctrl_cam0_names = safe_get_npz(ctrl_npz, "beh_cam0_names", None)
            ctrl_cam1_names = safe_get_npz(ctrl_npz, "beh_cam1_names", None)

    rest_nprw_counts = None
    rest_nprw_t = None
    rest_ua_counts = None
    rest_ua_t = None
    rest_ua_ids = None
    rest_ua_reg = None
    rest_ua_names = None
    rest_port = None
    if rest_path and rest_path.exists():
        with np.load(rest_path, allow_pickle=True) as rest_npz:
            rest_nprw_counts = safe_get_npz(rest_npz, "NPRW_counts", None)
            rest_nprw_t = safe_get_npz(rest_npz, "NPRW_rel_t", None)
            rest_ua_counts = safe_get_npz(rest_npz, "UA_counts", None)
            rest_ua_t = safe_get_npz(rest_npz, "UA_rel_t", None)
            rest_ua_ids = safe_get_npz(rest_npz, "ua_ids_1based", None)
            rest_ua_reg = safe_get_npz(rest_npz, "ua_region", None)
            rest_ua_names = safe_get_npz(rest_npz, "ua_region_names", None)
            rest_port = get_ua_port(rest_npz)

    # Kinematics extraction: Focus on Camera 1 (as requested), replacing Camera 0 with Pairwise X-Y Variability
    primary_cam_segs = stim_cam1_segs if (stim_cam1_segs is not None and getattr(stim_cam1_segs, "ndim", 0) == 3 and stim_cam1_segs.shape[0] > 0) else stim_cam0_segs
    primary_cam_names = stim_cam1_names if primary_cam_segs is stim_cam1_segs else stim_cam0_names
    cam_label = "Camera 1" if primary_cam_segs is stim_cam1_segs else "Camera 0"

    primary_ctrl_segs = ctrl_cam1_segs if cam_label == "Camera 1" else ctrl_cam0_segs
    primary_ctrl_names = ctrl_cam1_names if cam_label == "Camera 1" else ctrl_cam0_names

    # 1. Normalized reaching distance trajectories for primary camera
    stim_dist = rcp.extract_normalized_distance(primary_cam_segs, primary_cam_names, stim_beh_t)
    ctrl_dist = rcp.extract_normalized_distance(primary_ctrl_segs, primary_ctrl_names, ctrl_beh_t)

    # 2. Time-resolved pairwise X-Y trajectory variability (from plot_plateau_analysis.py)
    stim_xy = rcp.extract_xy_coordinates(primary_cam_segs, primary_cam_names)
    ctrl_xy = rcp.extract_xy_coordinates(primary_ctrl_segs, primary_ctrl_names)

    stim_var_curve, stim_var_lower, stim_var_upper = None, None, None
    ctrl_var_curve, ctrl_var_lower, ctrl_var_upper = None, None, None

    if stim_xy is not None:
        sx, sy = stim_xy
        stim_var_curve, stim_var_lower, stim_var_upper = rcp.compute_time_resolved_pairwise_variability(
            sx, sy, t_axis=stim_beh_t, smooth_butterworth=True, agg=TRAJECTORY_VARIABILITY_AGG
        )

    if ctrl_xy is not None:
        cx, cy = ctrl_xy
        ctrl_var_curve, ctrl_var_lower, ctrl_var_upper = rcp.compute_time_resolved_pairwise_variability(
            cx, cy, t_axis=ctrl_beh_t, smooth_butterworth=True, agg=TRAJECTORY_VARIABILITY_AGG
        )

    kinematics_data = {
        "cam_label": cam_label,
        "stim_beh_t": stim_beh_t,
        "ctrl_beh_t": ctrl_beh_t,
        "stim_cam1_dist": stim_dist,
        "ctrl_cam1_dist": ctrl_dist,
        "stim_variability_curve": stim_var_curve,
        "stim_variability_lower": stim_var_lower,
        "stim_variability_upper": stim_var_upper,
        "ctrl_variability_curve": ctrl_var_curve,
        "ctrl_variability_lower": ctrl_var_lower,
        "ctrl_variability_upper": ctrl_var_upper,
        "stim_n_trials": stim_dist.shape[0] if (stim_dist is not None and getattr(stim_dist, "ndim", 0) >= 2) else (stim_xy[0].shape[0] if stim_xy is not None else None),
        "ctrl_n_trials": ctrl_dist.shape[0] if (ctrl_dist is not None and getattr(ctrl_dist, "ndim", 0) >= 2) else (ctrl_xy[0].shape[0] if ctrl_xy is not None else None),
    }

    bad_nprw = bad_channels.get("NPRW", set()) if bad_channels else set()
    bad_ua = bad_channels.get("UA", set()) if bad_channels else set()

    # Process Arrays: We analyze NPRW and Utah Arrays (individual arrays and/or combined)
    array_data_list = []
    if nprw_counts is not None and nprw_t is not None:
        s_nprw_3d = prepare_counts_3d(nprw_counts)
        c_nprw_3d = prepare_counts_3d(ctrl_nprw_counts) if ctrl_nprw_counts is not None else None
        r_nprw_3d = prepare_counts_3d(rest_nprw_counts) if rest_nprw_counts is not None else None

        if s_nprw_3d is not None and s_nprw_3d.shape[0] > 0:
            n_nprw_total = s_nprw_3d.shape[0]
            # Exclude bad channels based on impedance (checking both 0-based and 1-based channel IDs)
            valid_nprw_idxs = [
                ch for ch in range(n_nprw_total)
                if ch not in bad_nprw and (ch + 1) not in bad_nprw
            ]
            n_nprw_bad = n_nprw_total - len(valid_nprw_idxs)
            if n_nprw_bad > 0:
                _log(f"  [NPRW] Excluded {n_nprw_bad}/{n_nprw_total} bad impedance channels; analyzing {len(valid_nprw_idxs)} valid channels.")

            if valid_nprw_idxs:
                s_nprw_valid = s_nprw_3d[valid_nprw_idxs, :, :]
                c_nprw_valid = c_nprw_3d[valid_nprw_idxs, :, :] if (c_nprw_3d is not None and c_nprw_3d.shape[0] == n_nprw_total) else c_nprw_3d
                r_nprw_valid = r_nprw_3d[valid_nprw_idxs, :, :] if (r_nprw_3d is not None and r_nprw_3d.shape[0] == n_nprw_total) else r_nprw_3d
                nprw_ch_ids = [ch + 1 for ch in valid_nprw_idxs]
                array_data_list.append((
                    "NPRW",
                    s_nprw_valid, nprw_t,
                    c_nprw_valid, ctrl_nprw_t if ctrl_nprw_t is not None else nprw_t,
                    r_nprw_valid, rest_nprw_t if rest_nprw_t is not None else nprw_t,
                    nprw_ch_ids,
                ))
            else:
                _log("  [NPRW] No valid channels remaining after impedance exclusion, skipping.")

    if ua_counts is not None and ua_t is not None:
        s_ua_3d = prepare_counts_3d(ua_counts)
        c_ua_3d = prepare_counts_3d(ctrl_ua_counts) if ctrl_ua_counts is not None else None
        r_ua_3d = prepare_counts_3d(rest_ua_counts) if rest_ua_counts is not None else None

        if ANALYZE_COMBINED_UTAH and s_ua_3d is not None and s_ua_3d.shape[0] > 0:
            all_ua_ch = s_ua_3d.shape[0]
            ua_ids_arr = ua_ids.ravel() if ua_ids is not None else np.arange(1, all_ua_ch + 1)
            valid_ua_idxs = [
                i for i in range(all_ua_ch)
                if int(ua_ids_arr[i]) not in bad_ua and i not in bad_ua and (i + 1) not in bad_ua
            ]
            n_ua_bad = all_ua_ch - len(valid_ua_idxs)
            if n_ua_bad > 0:
                _log(f"  [UA_ALL] Excluded {n_ua_bad}/{all_ua_ch} bad impedance channels; analyzing {len(valid_ua_idxs)} valid channels.")

            if valid_ua_idxs:
                s_ua_valid = s_ua_3d[valid_ua_idxs, :, :]
                c_ua_valid = c_ua_3d[valid_ua_idxs, :, :] if (c_ua_3d is not None and c_ua_3d.shape[0] == all_ua_ch) else c_ua_3d
                r_ua_valid = r_ua_3d[valid_ua_idxs, :, :] if (r_ua_3d is not None and r_ua_3d.shape[0] == all_ua_ch) else r_ua_3d
                all_elecs = [int(ua_ids_arr[i]) for i in valid_ua_idxs]
                array_data_list.append((
                    "UA_ALL",
                    s_ua_valid, ua_t,
                    c_ua_valid, ctrl_ua_t if ctrl_ua_t is not None else ua_t,
                    r_ua_valid, rest_ua_t if rest_ua_t is not None else ua_t,
                    all_elecs,
                ))

        if ANALYZE_INDIVIDUAL_UTAH_ARRAYS and s_ua_3d is not None:
            for reg in REGION_ORDER:
                s_idxs, s_elecs = extract_array_channel_indices(
                    ua_ids, elec_info, port, reg,
                    ua_region=ua_region,
                    ua_region_names=ua_region_names,
                    max_channels=s_ua_3d.shape[0],
                    bad_channels=bad_ua,
                )
                if not s_idxs:
                    _log(f"  [UA_{reg}] No valid channels found for region (all excluded or missing), skipping.")
                    continue

                s_sub = s_ua_3d[s_idxs, :, :]

                c_sub = None
                if c_ua_3d is not None:
                    c_idxs, _ = extract_array_channel_indices(
                        ctrl_ua_ids if ctrl_ua_ids is not None else ua_ids,
                        elec_info,
                        ctrl_port if ctrl_port else port,
                        reg,
                        ua_region=ctrl_ua_reg if ctrl_ua_reg is not None else ua_region,
                        ua_region_names=ctrl_ua_names if ctrl_ua_names is not None else ua_region_names,
                        max_channels=c_ua_3d.shape[0],
                        bad_channels=bad_ua,
                    )
                    if c_idxs and len(c_idxs) == len(s_idxs):
                        c_sub = c_ua_3d[c_idxs, :, :]
                    elif len(s_idxs) <= c_ua_3d.shape[0]:
                        c_sub = c_ua_3d[s_idxs, :, :]

                r_sub = None
                if r_ua_3d is not None:
                    r_idxs, _ = extract_array_channel_indices(
                        rest_ua_ids if rest_ua_ids is not None else ua_ids,
                        elec_info,
                        rest_port if rest_port else port,
                        reg,
                        ua_region=rest_ua_reg if rest_ua_reg is not None else ua_region,
                        ua_region_names=rest_ua_names if rest_ua_names is not None else ua_region_names,
                        max_channels=r_ua_3d.shape[0],
                        bad_channels=bad_ua,
                    )
                    if r_idxs and len(r_idxs) == len(s_idxs):
                        r_sub = r_ua_3d[r_idxs, :, :]
                    elif len(s_idxs) <= r_ua_3d.shape[0]:
                        r_sub = r_ua_3d[s_idxs, :, :]

                array_data_list.append((
                    f"UA_{reg}",
                    s_sub, ua_t,
                    c_sub, ctrl_ua_t if ctrl_ua_t is not None else ua_t,
                    r_sub, rest_ua_t if rest_ua_t is not None else ua_t,
                    s_elecs,
                ))

    cluster_assignments = {}
    phenotypes_by_array = {}

    for item in array_data_list:
        if len(item) == 8:
            arr_name, s_3d, s_t, c_3d, c_t, r_3d, r_t, arr_ch_ids = item
        else:
            arr_name, s_3d, s_t, c_3d, c_t, r_3d, r_t = item
            arr_ch_ids = np.arange(1, s_3d.shape[0] + 1)
        _log(f"Analyzing array modality: {arr_name}")

        if s_3d is None or c_3d is None or s_3d.shape[0] == 0 or c_3d.shape[0] == 0:
            _log(f"Missing STIM or CTRL counts for {arr_name}, skipping.")
            continue

        # Determine blanking parameters based on array modality and stim duration
        is_stim_cond = stim_dur_ms > 0
        blank_pre = NPRW_BLANK_PRE_MS if arr_name == "NPRW" else (UA_BLANK_PRE_MS if BLANK_UA_STIM_PERIOD else 0.0)
        blank_post = NPRW_BLANK_POST_MS if arr_name == "NPRW" else (UA_BLANK_POST_MS if BLANK_UA_STIM_PERIOD else 0.0)

        has_blanking = is_stim_cond and (blank_pre > 0 or blank_post > 0)
        blank_win = (-blank_pre, stim_dur_ms + blank_post) if has_blanking else None

        # Rebin and smooth rates (Hz) using parameters from params.yaml with NaN-aware blanking
        bin_w = NPRW_BIN_WIDTH_MS if arr_name == "NPRW" else UA_BIN_WIDTH_MS
        sigma = NPRW_SMOOTHING_SIGMA_MS if arr_name == "NPRW" else UA_SMOOTHING_SIGMA_MS
        s_rates, time_ms = rebin_and_smooth(s_3d, s_t, target_bin_w=bin_w, smooth_sigma_ms=sigma, win_ms=WIN_PLOT_MS, blank_win=blank_win)
        c_rates, _ = rebin_and_smooth(c_3d, c_t, target_bin_w=bin_w, smooth_sigma_ms=sigma, win_ms=WIN_PLOT_MS, blank_win=None)

        if has_blanking:
            blank_mask = (time_ms >= blank_win[0]) & (time_ms <= blank_win[1])
            _log(f"  [{arr_name}] Applying artifact blanking window: [{blank_win[0]:.1f}, {blank_win[1]:.1f}] ms")
        else:
            blank_mask = np.zeros(len(time_ms), dtype=bool)

        # 1. Stratified trial pairing
        paired_s, paired_c = match_trials_stratified(s_rates, c_rates, stim_vel, ctrl_vel)
        if paired_s.shape[1] == 0:
            continue

        # Trial-level paired contrast Δr_{c, i}(t)
        diff_trials = paired_s - paired_c  # (ch, trials, bins)
        paired_contrast_mean = np.nanmean(diff_trials, axis=1)  # (ch, bins)
        contrast_sem = np.nanstd(diff_trials, axis=1) / math.sqrt(max(1, paired_s.shape[1]))

        # Define dynamic post-artifact response windows
        post_stim_start = (stim_dur_ms + blank_post) if has_blanking else 0.0
        resp_win = (post_stim_start, min(post_stim_start + 300.0, float(WIN_PLOT_MS[1])))
        early_win = (post_stim_start, min(post_stim_start + 50.0, float(WIN_PLOT_MS[1])))
        late_win = (post_stim_start + 50.0, min(post_stim_start + 200.0, float(WIN_PLOT_MS[1])))

        # 2. Statistical Responsiveness Gating (Welch baseline-corrected, Paired, Unpaired, or Both)
        is_responsive, p_vals, q_vals, gate_info = compute_responsiveness_gating(
            paired_stim=paired_s,
            paired_ctrl=paired_c,
            stim_rates=s_rates,
            ctrl_rates=c_rates,
            time_ms=time_ms,
            baseline_win_ms=BASELINE_WIN_MS,
            response_win_ms=resp_win,
            blank_mask=blank_mask,
            test_mode=RESPONSIVENESS_TEST_MODE,
            both_combination=RESPONSIVENESS_BOTH_COMBINATION,
            n_perms=N_PERMUTATIONS,
            alpha=FDR_ALPHA,
            criterion=RESPONSIVENESS_CRITERION,
            p_thresh=P_VAL_THRESH,
            min_delta_hz=MIN_DELTA_HZ,
            channel_ids=arr_ch_ids,
        )
        n_resp = int(np.sum(is_responsive))
        n_ch_total = len(is_responsive)
        p_info = gate_info.get("paired", {})
        u_info = gate_info.get("unpaired", {})
        w_info = gate_info.get("welch", {})

        # Structured diagnostic logging for condition
        _log(f"       -> {arr_name:<6}:")
        _log(f"            trials: stim={s_rates.shape[1]}, ctrl={c_rates.shape[1]}, paired={paired_s.shape[1]}")
        if w_info and w_info.get("is_responsive") is not None:
            _log(f"            welch:    responsive={w_info['n_resp']}/{n_ch_total}, p<{P_VAL_THRESH}={w_info['n_sig_p']}, q<{FDR_ALPHA}={w_info['n_sig_q']}")
        if p_info and p_info.get("is_responsive") is not None:
            _log(f"            paired:   responsive={p_info['n_resp']}/{n_ch_total}, p<{P_VAL_THRESH}={p_info['n_sig_p']}, q<{FDR_ALPHA}={p_info['n_sig_q']}")
        if u_info and u_info.get("is_responsive") is not None:
            _log(f"            unpaired: responsive={u_info['n_resp']}/{n_ch_total}, p<{P_VAL_THRESH}={u_info['n_sig_p']}, q<{FDR_ALPHA}={u_info['n_sig_q']}")
        _log(f"            active gate: {gate_info['gate_desc']} => {n_resp}/{n_ch_total}")

        base_name = f"{stim_path.stem}_{arr_name}"

        # Save per-channel responsiveness metrics if available
        if w_info and w_info.get("metrics_df") is not None:
            welch_df = w_info["metrics_df"].copy()
            welch_df.insert(0, "array", arr_name)
            metrics_csv_path = out_dir / f"{base_name}_responsiveness_metrics.csv"
            welch_df.to_csv(metrics_csv_path, index=False)
            welch_csv_path = out_dir / f"{base_name}_responsiveness_welch.csv"
            welch_df.to_csv(welch_csv_path, index=False)
            _log(f"            saved metrics: {metrics_csv_path.name}")

        # Determine contrast mean source for downstream clustering and plots
        if CONTRAST_SOURCE_FOR_PLOTS == "unpaired" and u_info is not None and u_info.get("contrast_mean") is not None:
            contrast_mean = u_info["contrast_mean"]
            contrast_source_desc = "Unpaired mean contrast"
        elif "contrast_mean" in gate_info and gate_info["contrast_mean"] is not None:
            contrast_mean = gate_info["contrast_mean"]
            contrast_source_desc = f"{CONTRAST_SOURCE_FOR_PLOTS.capitalize()} mean contrast"
        else:
            contrast_mean = paired_contrast_mean
            contrast_source_desc = "Paired mean contrast"

        # Rest rates (if present)
        resp_rest_rates = None
        rest_pop = None
        n_rest_tr = None
        if r_3d is not None and r_t is not None:
            r_rates, _ = rebin_and_smooth(r_3d, r_t, target_bin_w=bin_w, smooth_sigma_ms=sigma, win_ms=WIN_PLOT_MS)
            if has_blanking:
                r_rates[:, :, blank_mask] = np.nan
            rest_pop = np.nanmean(r_rates, axis=1)
            n_rest_tr = r_rates.shape[1]

        # Trial count tracking across conditions
        trial_counts = {
            "stim": int(s_rates.shape[1]),
            "ctrl": int(c_rates.shape[1]),
            "matched": int(paired_s.shape[1]),
            "rest": int(n_rest_tr) if n_rest_tr is not None else None,
        }
        trial_info_str = format_trial_info(trial_counts)
        nonresp_label = f"Not retained ({gate_info['gate_desc']})"

        # Channel-level trial-averaged firing rates (Hz) for phenotype overlays
        stim_ch_mean = np.nanmean(s_rates, axis=1)
        ctrl_ch_mean = np.nanmean(c_rates, axis=1)

        # Early vs Late Quadrants
        e_vals, l_vals, quads = compute_early_late_quadrants(
            contrast_mean, time_ms, early_win=early_win, late_win=late_win, blank_mask=blank_mask
        )

        # 3. Dual Normalization & Figure Generation
        if n_resp >= 2:
            resp_mean = contrast_mean[is_responsive, :]
            resp_diff_trials = diff_trials[is_responsive, :, :]
            norm_dict = dual_normalize_contrasts(
                resp_mean, resp_diff_trials, time_ms,
                baseline_win_ms=BASELINE_WIN_MS,
                full_win_ms=resp_win,
                blank_mask=blank_mask,
            )

            # 4. Empirical Clustering (Ward's Linkage)
            resp_window_traces = norm_dict["unit_norm"][:, norm_dict["resp_mask"]]
            resp_window_traces = np.nan_to_num(resp_window_traces, nan=0.0)
            clust_results = fit_empirical_clustering(resp_window_traces)
            k_labels = clust_results["cluster_labels"]

            # Map cluster labels back to full array (-1 for non-responsive)
            full_labels = np.full(len(is_responsive), -1, dtype=int)
            full_labels[is_responsive] = k_labels
            cluster_assignments[arr_name] = full_labels

            # Cluster phenotypes
            mean_by_cluster = {}
            for k in np.unique(k_labels):
                mean_by_cluster[k] = np.nanmean(norm_dict["raw_centered"][k_labels == k], axis=0)
            phenotypes = assign_phenotype_labels(
                mean_by_cluster, time_ms, early_win=early_win, late_win=late_win, blank_mask=blank_mask
            )
            phenotypes_by_array[arr_name] = phenotypes

            # Non-responsive traces and channel means
            nonresp_traces = contrast_mean[~is_responsive, :] if np.any(~is_responsive) else None
            resp_stim_rates = stim_ch_mean[is_responsive, :]
            resp_ctrl_rates = ctrl_ch_mean[is_responsive, :]
            resp_rest_rates = rest_pop[is_responsive, :] if rest_pop is not None else None

            # 6. Plot Figures 1 to 5
            if clust_results["linkage_matrix"] is not None:
                plot_fig1_dendrogram_and_metrics(
                    clust_results["linkage_matrix"],
                    clust_results["silhouette_scores"],
                    clust_results["ch_scores"],
                    clust_results["best_k"],
                    out_dir / f"{base_name}_fig1_dendrogram_metrics.png",
                    title_suffix=f"({arr_name}) | {trial_info_str}",
                )

            plot_fig2_cluster_phenotype_profiles(
                time_ms,
                norm_dict["unit_norm"],
                norm_dict["raw_centered"],
                k_labels,
                phenotypes,
                stim_dur_ms,
                out_dir / f"{base_name}_fig2_phenotypes.png",
                title_suffix=f"({arr_name}) | {gate_info['gate_desc']}",
                blank_win=blank_win,
                kinematics_data=kinematics_data,
                stim_rates=resp_stim_rates,
                ctrl_rates=resp_ctrl_rates,
                rest_rates=resp_rest_rates,
                trial_counts=trial_counts,
            )

            plot_fig3_cluster_sorted_heatmaps(
                time_ms,
                norm_dict["raw_centered"],
                k_labels,
                nonresp_traces,
                stim_dur_ms,
                out_dir / f"{base_name}_fig3_heatmaps.png",
                title_suffix=f"({arr_name}) | {gate_info['gate_desc']} | {trial_info_str}",
                blank_win=blank_win,
                post_stim_start=post_stim_start,
                nonresp_label=nonresp_label,
            )

            plot_fig4_early_late_quadrant_scatter(
                e_vals,
                l_vals,
                k_labels,
                ~is_responsive,
                phenotypes,
                out_dir / f"{base_name}_fig4_quadrant_scatter.png",
                title_suffix=f"({arr_name}) | {gate_info['gate_desc']} | {trial_info_str}",
                early_win=early_win,
                late_win=late_win,
                nonresp_label=nonresp_label,
            )

            # Context Interaction (Reach vs Rest)
            if rest_pop is not None:
                rest_base = np.nanmean(rest_pop[:, (time_ms >= BASELINE_WIN_MS[0]) & (time_ms <= BASELINE_WIN_MS[1])], axis=1, keepdims=True)
                rest_contrast = rest_pop - rest_base

                interaction = contrast_mean - rest_contrast
                plot_fig5_context_interaction(
                    time_ms,
                    interaction,
                    contrast_mean,
                    rest_contrast,
                    stim_dur_ms,
                    out_dir / f"{base_name}_fig5_reach_vs_rest.png",
                    title_suffix=f"({arr_name}) | {trial_info_str}",
                    blank_win=blank_win,
                    post_stim_start=post_stim_start,
                )
        else:
            _log(f"       -> {arr_name:<6}: Insufficient responsive channels ({n_resp}) for clustering. Rendering Population Fallback...")
            # Normalize all channels across array for consistent population representation
            norm_dict_all = dual_normalize_contrasts(
                contrast_mean, diff_trials, time_ms,
                baseline_win_ms=BASELINE_WIN_MS,
                full_win_ms=resp_win,
                blank_mask=blank_mask,
            )

            # Assign single population cluster (or split if 1 responsive channel)
            if n_resp == 1:
                resp_idx = int(np.where(is_responsive)[0][0])
                k_labels = np.where(is_responsive, 0, 1)
                phenotypes = {
                    0: f"Responsive Unit (Ch {resp_idx + 1}, n=1)",
                    1: f"Non-responsive Baseline (n={len(contrast_mean) - 1})",
                }
                fig2_title_suffix = f"({arr_name}) | 1 Responsive Channel ({gate_info['gate_desc']}) | Population Overview"
            else:
                k_labels = np.zeros(len(contrast_mean), dtype=int)
                phenotypes = {0: f"Array Population (n={len(contrast_mean)})"}
                fig2_title_suffix = f"({arr_name}) | 0 Responsive Channels ({gate_info['gate_desc']}) | Population Overview"

            cluster_assignments[arr_name] = np.where(is_responsive, 0, -1)
            phenotypes_by_array[arr_name] = phenotypes

            # 1. Figure 2: Population Overview & Kinematics
            plot_fig2_cluster_phenotype_profiles(
                time_ms,
                norm_dict_all["unit_norm"],
                norm_dict_all["raw_centered"],
                k_labels,
                phenotypes,
                stim_dur_ms,
                out_dir / f"{base_name}_fig2_phenotypes.png",
                title_suffix=fig2_title_suffix,
                blank_win=blank_win,
                kinematics_data=kinematics_data,
                stim_rates=stim_ch_mean,
                ctrl_rates=ctrl_ch_mean,
                rest_rates=rest_pop,
                trial_counts=trial_counts,
            )

            # 2. Figure 3: Heatmap (All channels showing non-responsive baseline)
            if n_resp == 1:
                fig3_resp_traces = norm_dict_all["raw_centered"][[resp_idx], :]
                fig3_labels = np.array([0])
                fig3_nonresp = norm_dict_all["raw_centered"][~is_responsive, :]
            else:
                fig3_resp_traces = np.empty((0, len(time_ms)))
                fig3_labels = np.empty(0, dtype=int)
                fig3_nonresp = norm_dict_all["raw_centered"]

            plot_fig3_cluster_sorted_heatmaps(
                time_ms,
                fig3_resp_traces,
                fig3_labels,
                fig3_nonresp,
                stim_dur_ms,
                out_dir / f"{base_name}_fig3_heatmaps.png",
                title_suffix=f"({arr_name}) | {n_resp} Responsive | {gate_info['gate_desc']} | {trial_info_str}",
                blank_win=blank_win,
                post_stim_start=post_stim_start,
                nonresp_label=nonresp_label,
            )

            # 3. Figure 4: Quadrant Scatter (All channels in grey centered near 0)
            if n_resp == 1:
                fig4_labels = np.array([0])
            else:
                fig4_labels = np.empty(0, dtype=int)

            plot_fig4_early_late_quadrant_scatter(
                e_vals,
                l_vals,
                fig4_labels,
                ~is_responsive,
                phenotypes if n_resp == 1 else {},
                out_dir / f"{base_name}_fig4_quadrant_scatter.png",
                title_suffix=f"({arr_name}) | {n_resp} Responsive | {gate_info['gate_desc']} | {trial_info_str}",
                early_win=early_win,
                late_win=late_win,
                nonresp_label=nonresp_label,
            )

            # 4. Figure 5: Reach vs Rest (Population interaction if rest exists)
            if rest_pop is not None:
                rest_base = np.nanmean(rest_pop[:, (time_ms >= BASELINE_WIN_MS[0]) & (time_ms <= BASELINE_WIN_MS[1])], axis=1, keepdims=True)
                rest_contrast = rest_pop - rest_base
                interaction = contrast_mean - rest_contrast
                plot_fig5_context_interaction(
                    time_ms,
                    interaction,
                    contrast_mean,
                    rest_contrast,
                    stim_dur_ms,
                    out_dir / f"{base_name}_fig5_reach_vs_rest.png",
                    title_suffix=f"({arr_name}) | Population Interaction | {trial_info_str}",
                    blank_win=blank_win,
                    post_stim_start=post_stim_start,
                )

    _log(f"{prefix}Completed analysis for {stim_path.name}\n")


def parse_condition_specifier(item: Any) -> Tuple[Optional[int], str]:
    """Parse a condition specifier item into (br_int, str_repr)."""
    if isinstance(item, int):
        return item, str(item)
    s = str(item).strip()
    try:
        return int(s), s
    except ValueError:
        pass
    m = re.search(r"(?:BR|cond)[_]?(\d+)", s, re.IGNORECASE)
    if m:
        return int(m.group(1)), s
    return None, s


def is_stim_file_matching(stim_f: Path, targets: List[Any]) -> bool:
    """Check if a stimulation NPZ file matches any specified condition targets."""
    if not targets:
        return True

    file_name = stim_f.name
    file_path_str = str(stim_f).lower()

    m_br = re.search(r"BR_(\d+)", file_name, re.IGNORECASE)
    file_br_idx = int(m_br.group(1)) if m_br else None

    for target_item in targets:
        br_int, str_val = parse_condition_specifier(target_item)

        if br_int is not None:
            # When an explicit BR index was identified (e.g. 7, "7", "BR_007", "cond7"),
            # match STRICTLY against the file's BR index.
            if file_br_idx is not None and file_br_idx == br_int:
                return True
        else:
            # Non-numeric target specifier (e.g. "Target_A", "NRR_RW012"): substring match
            str_lower = str_val.lower()
            if str_lower in file_name.lower() or str_lower in file_path_str:
                return True

    return False


def main(process_only: Optional[List[Any]] = None):
    cli_targets = None
    target_filter = None

    if process_only is None and len(sys.argv) > 1:
        try:
            parser = argparse.ArgumentParser(
                description="Stimulation response clustering, functional phenotyping, and neural state-space dynamics."
            )
            parser.add_argument(
                "--process-only", "--condition", "--br", "-c",
                nargs="+",
                dest="process_only",
                default=None,
                help="Condition(s) or BR index(es) to process (e.g. 12, BR_012, cond012, Target_A). Defaults to PARAMS.preprocessing['process_only']."
            )
            parser.add_argument(
                "--target", "-t",
                type=str,
                default=None,
                help="Optional target folder filter (e.g. Target_A, Target_B, Target_control)."
            )
            parser.add_argument(
                "--test-mode", "--mode",
                type=str,
                default=RESPONSIVENESS_TEST_MODE,
                choices=["welch_baseline_corrected", "paired_signflip", "unpaired_label_shuffle", "both"],
                help="Responsiveness test mode (default: welch_baseline_corrected).",
            )
            parsed_args, _ = parser.parse_known_args()
            if parsed_args.process_only:
                cli_targets = parsed_args.process_only
            if parsed_args.target:
                target_filter = parsed_args.target
            if parsed_args.test_mode:
                RESPONSIVENESS_TEST_MODE = parsed_args.test_mode
        except Exception as e:
            _log(f"CLI parsing warning: {e}")

    targets = process_only if process_only is not None else (cli_targets if cli_targets is not None else PROCESS_ONLY)

    _log("Starting Stimulation Response Clustering & Neural Dynamics Analysis")
    ensure_dir(CLUST_FIG_ROOT)

    bad_channels = load_bad_channels()
    all_files = sorted(PERI_ROOT.rglob("peristim__*.npz"))
    if not all_files:
        _log(f"No peristim files found under {PERI_ROOT}")
        return

    _log(f"Discovered {len(all_files)} total peristim files.")
    refs = load_reference_mapping(all_files)

    stim_files = [f for f in all_files if get_cond_type(f) == "STIM"]
    if targets:
        stim_files = [f for f in stim_files if is_stim_file_matching(f, targets)]

    if target_filter:
        t_clean = target_filter.strip().lower()
        stim_files = [f for f in stim_files if t_clean in get_target_folder(f).lower() or t_clean in str(f).lower()]

    _log(f"Found {len(stim_files)} STIM condition(s) to evaluate matching criteria: {targets or 'All'}")

    for idx, stim_f in enumerate(stim_files):
        target = get_target_folder(stim_f)
        try:
            with np.load(stim_f, allow_pickle=True) as d:
                port = get_ua_port(d)
                fh = scalarize(safe_get_npz(d, "stim_freq_hz", None))
                dm = scalarize(safe_get_npz(d, "stim_dur_nominal_ms", None))
                pc = compute_pulse_count(fh, dm)
        except Exception:
            port = "A"
            pc = None

        matched_ctrl = refs["ctrl"].get((target, port), None)
        matched_rest = refs["rest"].get(pc, None) if pc is not None else None

        # Output folder grouped by condition & target
        sub_dir = CLUST_FIG_ROOT / target if target else CLUST_FIG_ROOT / "other"

        try:
            process_single_condition(
                stim_path=stim_f,
                ctrl_path=matched_ctrl,
                rest_path=matched_rest,
                bad_channels=bad_channels,
                out_dir=sub_dir,
                file_idx=idx + 1,
                total_files=len(stim_files),
            )
        except Exception as err:
            warnings.warn(f"Error processing {stim_f.name}: {err}\n{traceback.format_exc()}")

    _log("All stimulation response clustering figures generated successfully.")


if __name__ == "__main__":
    main()
