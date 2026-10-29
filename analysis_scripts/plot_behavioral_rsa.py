"""
plot_behavioral_rsa.py
======================

3D Behavioral Representational Similarity Analysis (Kinematic RSA), Time-Resolved
Trajectory Variability, and Multi-Condition Directional Perturbation Analysis.

Scientific Questions Addressed:
    1. Behavioral Representational Geometry (Kinematic RSA):
       - How similar or distinct are single-reach 3D trajectories across stimulation
         conditions compared to control reaches?
       - Does stimulation preserve the natural geometrical separation between reach
         targets (Target A vs Target B), or does it degrade target discriminability?
       - Are within-condition reaches consistent and stereotyped, or highly dispersed?
    2. Time-Resolved Trajectory Variability (Stim vs Control Across Time):
       - When during the reach does stimulation inject motor variability?
       - Does stimulation cause a transient burst of trajectory jitter during the
         stimulus delivery window, or does elevated variability persist until target touch?
    3. Multi-Condition Directional Perturbation Components:
       - How do lateral (Delta X, left/right veer), forward (Delta Y, undershoot/overshoot),
         and vertical (Delta Z, height lift/dip) deviations evolve across time?
       - How do different stimulation parameters (frequency, pulse count, duration)
         steer reaching kinematics relative to unperturbed control reaches?

Key Visualizations Generated:
    1. Dual-Window Behavioral RSA Figure: 4-panel figure comparing Early (0–75 ms) and
       Late (75–350 ms) trial-by-trial kinematic dissimilarity matrices (RDMs) and condition
       separation (pairwise silhouette matrices) with permutation testing (p-values & FDR stars).
    2. Time-Resolved Trajectory Variability Comparison: Within-condition 3D variability V(t)
       and net 3D deflection from control reference D_to_ctrl(t) across reach time with
       condition-staggered stimulation duration ladders.
    3. Multi-Condition Directional Perturbation Traces: Time-resolved Lateral Veer Delta X(t),
       Vertical Lift/Dip Delta Z(t), and Forward Extension Delta Y(t).
    4. Touchscreen Endpoint Perturbations & 3D Spatial Trajectories: 3D spatial reach trajectories
       with an overlaid 2D touchscreen endpoint difference panel, 2-SE dispersion ellipses,
       perturbation vector arrow, target hit rate, and delta badge.
    5. Progress-Resampled Trajectories & Progress Profiles: 3D trajectories and deviation
       profiles resampled onto a normalized reach progress axis (Forward Y or arc length)
       to decouple reach speed from spatial path geometry.
    6. Forward-Y Normalization Methodological Validation: 6-panel diagnostic validating operational
       maximum forward Y endpoint assumptions, arrival time distributions, retention criteria, and
       physical landmark congruence (Section 3.6).
"""

from __future__ import annotations

import os
import re
import sys
import json
import math
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Sequence, Union

import colorsys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.patches import Patch, Ellipse, Circle
from matplotlib.colors import LinearSegmentedColormap, to_hex
from scipy import stats

# Suppress benign numerical runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Degrees of freedom <= 0 for slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*All-NaN slice encountered.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Add analysis_scripts and repository root to sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import 3D kinematics batch utilities
try:
    import plot_3d_kinematics as p3d
    HAS_3D_KINEMATICS = True
except Exception as _err:
    HAS_3D_KINEMATICS = False
    p3d = None

import RCP_analysis as rcp
try:
    from RCP_analysis.python.functions.config_loading import *
    CONFIG_LOADED = True
except Exception:
    CONFIG_LOADED = False
    PARAMS = None
    OUT_BASE = None
    PERI_ROOT = None
    VIDEO_ROOT = None
    SESSION_LOC = None
    METADATA_CSV = None

try:
    from RCP_analysis.python.functions.kinematics_utils import (
        butterworth_filter,
        mean_ci95_trace_summary,
    )
    from RCP_analysis.python.functions.rsa_utils import (
        _two_group_silhouette,
        _fdr_adjust_pairs,
    )
except Exception:
    pass

# =============================================================================
# Global Analysis Configuration & Defaults
# =============================================================================

DEFAULT_KEYPOINT: str = "middle"
SQUARE_SIZE_MM: float = 1.0
ALIGN_TO_TASK_FRAME: bool = True

# Global plotting time window (ms relative to stimulation onset / IR beam crossing)
# Extended to 450 ms so prolonged/delayed reaches (e.g. 40 pulses, 100 ms duration) complete touchscreen touch
DEFAULT_PLOT_WINDOW_MS: Tuple[float, float] = (-75.0, 500.0)

# Dual-window Behavioral RSA analysis windows:
# Region 1 (Early Window): Pre-divergence / expected control-like phase
RSA_EARLY_WINDOW_MS: Tuple[float, float] = (0.0, 75.0)
# Region 2 (Perturbation Window): Active stimulation deflection & divergence phase
RSA_LATE_WINDOW_MS: Tuple[float, float] = (75.0, 350.0)

# Dual-Window Progress Epochs (Normalized reach depth fraction p in [0.0, 1.0]):
RSA_EARLY_PROGRESS_RANGE: Tuple[float, float] = (0.0, 0.60)   # Early reach progress: 0–25% forward depth
RSA_LATE_PROGRESS_RANGE: Tuple[float, float] = (0.60, 1.00)   # Late reach progress: 25–100% forward depth


MIN_REACHES_PER_CONDITION: int = 3
N_PERMUTATIONS: int = 1000
FDR_ALPHA: float = 0.05
FIG_FORMAT: str = "png"  # Preselected figure format: "png" or "svg"
FIG_DPI: int = 300

# Color definitions
COLOR_CTRL: str = "#d32f2f"         # Crimson red for unperturbed control reference
COLOR_CTRL_LIGHT: str = "#ef9a9a"   # Soft light red
COLOR_STIM_DEFAULT: str = "#1f77b4" # Default steel blue for stimulation conditions

# Condition Label Display Options
SHOW_CONDITION_INDICES: bool = False  # If False, omit condition index prefixes (e.g. 'BR 1', 'BR 17')
SHOW_TRIAL_COUNTS: bool = False       # If False, omit trial count annotations (e.g. 'n=16')

# Figure Output & Display Options
VARIABILITY_PLOT_MODE: str = "both"         # Options: "both" (default), "variability", or "deflection"
VARIABILITY_LEGEND_ORIENTATION: str = "column"  # Options: "column" (default) or "row"
SPATIAL_PLOT_MODE: str = "overlay"          # Options: "overlay" (default: 3D Trajectory with 2D Touchscreen Inset) or "two_panel"
TRAJECTORY_STAT: str = "median"             # Options: "median" (default, robust central tendency) or "mean"
ELLIPSE_SCALE: str = "2se"                  # Options: "2se" (default, 2 standard errors of the mean), "1se", or "1sd"
DISSIMILARITY_STAT: str = "median"          # Options: "median" (default, robust to reach speed/duration jitter) or "mean"

# Path-Geometry RSA & Progress Resampling Options
RSA_ALIGN_BY_PROGRESS: bool = True          # If True (default), resample reaches onto a normalized 0-100% path progress axis to decouple speed from spatial geometry
RSA_PROGRESS_MODE: str = "forward_y"        # Options: "forward_y" (default, normalize along physical reaching axis Forward Y) or "arclength" (3D path distance)
RSA_N_PROGRESS_SAMPLES: int = 100           # Number of normalized spatial progress samples along reach path
RSA_DISTANCE_AGGREGATION: str = "median"    # Options: "median" (default, typical path deviation), "p90" (captures brief transient deflections), "mean", or "max"

# Target Size & Anchor Options
TARGET_RADIUS_MM: float = 5.0                # Target anchor circle radius around unperturbed control median center (5 mm radius = 10 mm diameter)
TARGET_DIAMETER_MM: float = 2.0 * TARGET_RADIUS_MM  # Nominal target diameter (10 mm)
TARGET_STABILITY_THRESHOLD_MM: float = 8.0   # Stability threshold (mm)




# =============================================================================
# Logging & Helper Utilities
# =============================================================================

def _adjust_color_lightness(color_val: Any, factor: float = 0.72) -> Any:
    """Adjust color luminance while keeping the hue and chroma in the same family."""
    try:
        rgb = mcolors.to_rgb(color_val)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        new_l = max(0.0, min(1.0, l * factor))
        return colorsys.hls_to_rgb(h, new_l, s)
    except Exception:
        return color_val


def _log(msg: str) -> None:
    print(f"[behavioral_rsa] {msg}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_process_only() -> Optional[List[int]]:
    """Resolve active conditions to process from environment (VELES) or PARAMS."""
    env_val = os.environ.get("RCP_PROCESS_ONLY")
    if env_val:
        try:
            parsed = json.loads(env_val)
            if isinstance(parsed, list) and len(parsed) > 0:
                return [int(x) for x in parsed if str(x).lstrip("-").isdigit()]
        except Exception:
            pass
    if PARAMS is not None:
        po = getattr(PARAMS, "process_only", None)
        if isinstance(po, list) and len(po) > 0:
            return [int(x) for x in po if str(x).lstrip("-").isdigit()]
        if hasattr(PARAMS, "preprocessing"):
            po2 = PARAMS.preprocessing.get("process_only", None)
            if isinstance(po2, list) and len(po2) > 0:
                return [int(x) for x in po2 if str(x).lstrip("-").isdigit()]
    return None


def p_stars_label(p_val: float) -> str:
    """Return significance stars string for p-values."""
    if not np.isfinite(p_val):
        return ""
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    return "n.s."


# =============================================================================
# Data Extraction & Condition Organization
# =============================================================================

def extract_stim_channels_info(
    peri: Any,
    csv_meta: Optional[Dict[str, Any]] = None,
    notes: str = "",
    cond_str: str = "",
) -> Optional[str]:
    """
    Extract stimulation channel specification (e.g. '1-32', '1-8', '16-24', '17-24')
    from peristim npz metadata, session metadata CSV row, notes, or title string.
    """
    stim_list = []
    if hasattr(peri, "__getitem__"):
        for meta_k in ("nprw_meta", "meta"):
            if meta_k in peri:
                try:
                    obj = peri[meta_k]
                    m_dict = obj.item() if hasattr(obj, "item") else obj
                    if isinstance(m_dict, dict):
                        for ch_k in ("stim_channels", "active_channels", "stim_channel", "channels"):
                            if ch_k in m_dict and m_dict[ch_k] is not None:
                                stim_list = m_dict[ch_k]
                                if np.size(stim_list) > 0:
                                    break
                except Exception:
                    pass
            if np.size(stim_list) > 0:
                break

        if np.size(stim_list) == 0:
            for direct_k in ("stim_channels", "stim_channels_0based", "active_channels"):
                if direct_k in peri:
                    stim_list = peri[direct_k]
                    break

    if np.size(stim_list) > 0:
        try:
            sl = [int(x) for x in np.asarray(stim_list).ravel() if np.isfinite(x)]
            if len(sl) > 0:
                chs = sorted(list(set(sl)))
                ranges = []
                start = chs[0]
                end = chs[0]
                for ch in chs[1:]:
                    if ch == end + 1:
                        end = ch
                    else:
                        ranges.append(f"{start}-{end}" if end > start else f"{start}")
                        start = ch
                        end = ch
                ranges.append(f"{start}-{end}" if end > start else f"{start}")
                return ", ".join(ranges)
        except Exception:
            pass

    if csv_meta:
        for ch_k in ("stim_channels", "stim_chs", "channels", "channel", "stim_ch", "active_channels", "probe_channels"):
            if ch_k in csv_meta and pd.notna(csv_meta[ch_k]):
                val_str = str(csv_meta[ch_k]).strip()
                if val_str and val_str.lower() not in ("nan", "none", ""):
                    m_range = re.findall(r"\d+", val_str)
                    if len(m_range) >= 2:
                        return f"{m_range[0]}-{m_range[-1]}"
                    elif len(m_range) == 1:
                        return m_range[0]
                    return val_str

    for text_cand in (notes, cond_str):
        if not text_cand:
            continue
        m_explicit = re.search(r"(?:chs?|channels?)\.?\s*[:=]?\s*([0-9]+\s*[-–]\s*[0-9]+)", text_cand, re.IGNORECASE)
        if m_explicit:
            return m_explicit.group(1).replace(" ", "").replace("–", "-")
        m_dash = re.search(r"\b([0-9]+\s*[-–]\s*[0-9]+)\b", text_cand)
        if m_dash:
            return m_dash.group(1).replace(" ", "").replace("–", "-")
        m_single = re.search(r"(?:chs?|channels?)\.?\s*[:=]?\s*([0-9]+)\b", text_cand, re.IGNORECASE)
        if m_single:
            return m_single.group(1)

    return None


class ReachCondition:
    """Container holding calibrated 3D reaches and metadata for one experimental condition."""
    def __init__(
        self,
        cond_num: int,
        br_idx: int,
        label: str,
        target: str,
        is_control: bool,
        time: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        raw_trial_indices: Optional[np.ndarray] = None,
        pulse_count: Optional[int] = None,
        stim_freq_hz: Optional[float] = None,
        stim_dur_ms: Optional[float] = None,
        notes: str = "",
        base_label: Optional[str] = None,
        stim_channels: Optional[str] = None,
        short_label: Optional[str] = None,
        fixed_params_summary: str = "",
        combined_br_indices: Optional[List[int]] = None,
        R_task: Optional[np.ndarray] = None,
        p_start: Optional[np.ndarray] = None,
        dlc_3d_dir: Optional[Path] = None,
        session_landmarks: Optional[Dict[str, np.ndarray]] = None,
        square_size_mm: float = SQUARE_SIZE_MM,
    ):
        self.cond_num = cond_num
        self.br_idx = br_idx
        self.label = label
        self.base_label = base_label or label
        self.target = target.upper()
        self.is_control = is_control
        self.time = time  # (T,)
        self.x = x        # (N, T)
        self.y = y        # (N, T)
        self.z = z        # (N, T)
        self.raw_trial_indices = raw_trial_indices
        self.pulse_count = pulse_count
        self.stim_freq_hz = stim_freq_hz
        self.stim_dur_ms = stim_dur_ms if stim_dur_ms is not None else 0.0
        self.notes = notes
        self.stim_channels = stim_channels
        self.short_label = short_label or label
        self.fixed_params_summary = fixed_params_summary
        self.combined_br_indices = combined_br_indices
        self.R_task = R_task
        self.p_start = p_start
        self.dlc_3d_dir = dlc_3d_dir
        self.session_landmarks = session_landmarks
        self.square_size_mm = square_size_mm

    @property
    def n_trials(self) -> int:
        return self.x.shape[0]

    @property
    def n_time(self) -> int:
        return len(self.time)

    def mean_trajectory(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute condition mean 3D trajectory (X, Y, Z)."""
        mx = np.nanmean(self.x, axis=0)
        my = np.nanmean(self.y, axis=0)
        mz = np.nanmean(self.z, axis=0)
        return mx, my, mz

    def median_trajectory(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute condition median 3D trajectory (X, Y, Z)."""
        medx = np.nanmedian(self.x, axis=0)
        medy = np.nanmedian(self.y, axis=0)
        medz = np.nanmedian(self.z, axis=0)
        return medx, medy, medz

    def slice_to_window(self, win: Tuple[float, float]) -> "ReachCondition":
        """Return a new ReachCondition with reaches sliced to the specified [start_ms, end_ms] window."""
        mask = (self.time >= win[0]) & (self.time <= win[1])
        if not np.any(mask):
            mask = np.ones(len(self.time), dtype=bool)
        return ReachCondition(
            cond_num=self.cond_num,
            br_idx=self.br_idx,
            label=self.label,
            target=self.target,
            is_control=self.is_control,
            time=self.time[mask],
            x=self.x[:, mask],
            y=self.y[:, mask],
            z=self.z[:, mask],
            raw_trial_indices=self.raw_trial_indices,
            pulse_count=self.pulse_count,
            stim_freq_hz=self.stim_freq_hz,
            stim_dur_ms=self.stim_dur_ms,
            notes=self.notes,
            base_label=self.base_label,
            stim_channels=self.stim_channels,
            short_label=self.short_label,
            fixed_params_summary=self.fixed_params_summary,
            combined_br_indices=self.combined_br_indices,
            R_task=self.R_task,
            p_start=self.p_start,
            dlc_3d_dir=self.dlc_3d_dir,
            session_landmarks=self.session_landmarks,
            square_size_mm=self.square_size_mm,
        )

    def slice_to_progress(
        self,
        p_range: Tuple[float, float],
        n_progress: int = 100,
        mode: str = "forward_y",
    ) -> "ReachCondition":
        """
        Return a new ReachCondition with completed reaches normalized by Forward-Y (or arc-length)
        and sliced to the progress fraction range [p_start, p_end] (e.g. (0.0, 0.25) or (0.25, 1.0)).
        """
        rx, ry, rz = resample_reach_trajectory(self.x, self.y, self.z, time=self.time, n_progress=n_progress, mode=mode)
        valid = np.all(np.isfinite(rx) & np.isfinite(ry) & np.isfinite(rz), axis=1)
        if not np.any(valid):
            valid = np.ones(self.n_trials, dtype=bool)
            rx, ry, rz = self.x, self.y, self.z

        progress_grid = np.linspace(0.0, 1.0, n_progress)
        p_mask = (progress_grid >= p_range[0] - 1e-5) & (progress_grid <= p_range[1] + 1e-5)
        if not np.any(p_mask):
            p_mask = np.ones(n_progress, dtype=bool)

        rx_sub = rx[valid][:, p_mask]
        ry_sub = ry[valid][:, p_mask]
        rz_sub = rz[valid][:, p_mask]
        p_sub = progress_grid[p_mask] * 100.0

        raw_idx = self.raw_trial_indices[valid] if self.raw_trial_indices is not None else None

        new_n = rx_sub.shape[0]
        new_label = re.sub(r"\(n=\d+\)", f"(n={new_n})", self.label)
        new_short_label = self.short_label
        if new_short_label:
            new_short_label = re.sub(r"\(n=\d+\)", f"(n={new_n})", new_short_label)

        return ReachCondition(
            cond_num=self.cond_num,
            br_idx=self.br_idx,
            label=new_label,
            target=self.target,
            is_control=self.is_control,
            time=p_sub,
            x=rx_sub,
            y=ry_sub,
            z=rz_sub,
            raw_trial_indices=raw_idx,
            pulse_count=self.pulse_count,
            stim_freq_hz=self.stim_freq_hz,
            stim_dur_ms=self.stim_dur_ms,
            notes=self.notes,
            base_label=self.base_label,
            stim_channels=self.stim_channels,
            short_label=new_short_label,
            fixed_params_summary=self.fixed_params_summary,
            combined_br_indices=self.combined_br_indices,
            R_task=self.R_task,
            p_start=self.p_start,
            dlc_3d_dir=self.dlc_3d_dir,
            session_landmarks=self.session_landmarks,
            square_size_mm=self.square_size_mm,
        )


def _get_cond_attr(c: Any, key: str, default: Any = None) -> Any:
    """Helper to retrieve attributes from either ReachCondition objects or dictionaries."""
    if isinstance(c, dict):
        return c.get(key, default)
    return getattr(c, key, default)


def build_adaptive_condition_labels(
    conditions: Sequence[Any],
    show_indices: Optional[bool] = None,
    show_trial_counts: Optional[bool] = None,
) -> Tuple[Dict[Any, str], Dict[Any, str], str]:
    """
    Construct publication-quality condition labels displaying only what is unique across conditions.

    Rules:
      1. If all stim conditions have the same stimulation channels, omit channels from condition labels.
      2. If all stim conditions have the same stimulation frequency, omit frequency from condition labels.
      3. If all stim conditions have the same pulse count, omit pulse count from condition labels.
      4. Place identical/fixed parameters in a summary string for figure-level display (e.g. 'Fixed parameters: 400 Hz, ch 1-32').
      5. Configurable options:
         - show_indices (default False): Toggle condition index prefix (e.g. 'BR 1', 'BR 17').
         - show_trial_counts (default False): Toggle trial counts (e.g. 'n=16').

    Returns:
      full_labels: Dict mapping cond_num -> full descriptive label (e.g. '40 pulses')
      short_labels: Dict mapping cond_num -> abbreviated label for RDM matrix ticks (e.g. '40p')
      fixed_summary: String summarizing identical parameters (e.g. 'Fixed parameters: 400 Hz, ch 1-32')
    """
    if show_indices is None:
        show_indices = SHOW_CONDITION_INDICES
    if show_trial_counts is None:
        show_trial_counts = SHOW_TRIAL_COUNTS

    ctrl_conds = [c for c in conditions if _get_cond_attr(c, "is_control", False)]
    stim_conds = [c for c in conditions if not _get_cond_attr(c, "is_control", False)]

    # Extract stimulation parameters across all stim conditions
    pulses = [_get_cond_attr(c, "pulse_count") for c in stim_conds]
    valid_pulses = [p for p in pulses if p is not None]
    unique_pulses = sorted(list(set(valid_pulses)))

    freqs = [_get_cond_attr(c, "stim_freq_hz") for c in stim_conds]
    valid_freqs = [f for f in freqs if f is not None and not np.isnan(f)]
    unique_freqs = sorted(list(set(valid_freqs)))

    channels = [_get_cond_attr(c, "stim_channels") for c in stim_conds]
    valid_channels = [str(ch).strip() for ch in channels if ch and str(ch).strip() and str(ch).lower() not in ("none", "nan")]
    unique_channels = sorted(list(set(valid_channels)))

    # Check for continuous stimulation
    cont_flags = [
        bool(
            any(kw in f"{_get_cond_attr(c, 'notes', '')} {_get_cond_attr(c, 'label', '')}".lower() for kw in ("cont", "continuous"))
            or (_get_cond_attr(c, "stim_freq_hz") is not None and _get_cond_attr(c, "stim_freq_hz") <= 2.5)
        )
        for c in stim_conds
    ]
    any_cont = any(cont_flags)
    all_cont = all(cont_flags) and len(cont_flags) > 0

    # Check for current steering
    curr_steer_flags = [
        bool("curr steer" in f"{_get_cond_attr(c, 'notes', '')} {_get_cond_attr(c, 'label', '')}".lower() or
             "current steer" in f"{_get_cond_attr(c, 'notes', '')} {_get_cond_attr(c, 'label', '')}".lower())
        for c in stim_conds
    ]
    vary_steer = any(curr_steer_flags) and not all(curr_steer_flags)

    # Determine what varies vs what is identical across stim conditions
    if len(stim_conds) <= 1:
        # Only 1 stim condition (or none): display its parameters fully, no fixed parameter stripping needed
        vary_pulses = True
        vary_freqs = True
        vary_channels = True
        constant_pulses = False
        constant_freqs = False
        constant_channels = False
    else:
        vary_pulses = (len(unique_pulses) > 1) or (any_cont and not all_cont)
        vary_freqs = (len(unique_freqs) > 1)
        vary_channels = (len(unique_channels) > 1)

        constant_pulses = (len(unique_pulses) == 1 and len(valid_pulses) == len(stim_conds) and not any_cont)
        constant_freqs = (len(unique_freqs) == 1 and len(valid_freqs) == len(stim_conds))
        constant_channels = (len(unique_channels) == 1 and len(valid_channels) == len(stim_conds))

    # Guard: If nothing varies among stim conditions, don't strip everything into blank labels
    if len(stim_conds) > 1 and not (vary_pulses or vary_freqs or vary_channels or vary_steer):
        vary_pulses = True
        vary_freqs = True
        vary_channels = True

    # Build fixed parameters summary string
    fixed_parts = []
    if constant_freqs and not vary_freqs and len(unique_freqs) == 1:
        fv = unique_freqs[0]
        fixed_parts.append(f"{int(fv)} Hz" if float(fv).is_integer() else f"{fv:.1f} Hz")
    if constant_pulses and not vary_pulses and len(unique_pulses) == 1:
        pv = unique_pulses[0]
        fixed_parts.append(f"{pv} pulse" if pv == 1 else f"{pv} pulses")
    if constant_channels and not vary_channels and len(unique_channels) == 1:
        fixed_parts.append(f"ch {unique_channels[0]}")

    if len(fixed_parts) > 1:
        fixed_summary = f"Fixed parameters: {', '.join(fixed_parts)}"
    elif len(fixed_parts) == 1:
        fixed_summary = f"Fixed parameter: {fixed_parts[0]}"
    else:
        fixed_summary = ""

    # Build condition-specific labels
    full_labels: Dict[Any, str] = {}
    short_labels: Dict[Any, str] = {}

    for c in conditions:
        cond_num = _get_cond_attr(c, "cond_num", _get_cond_attr(c, "br_idx", 0))
        is_ctrl = _get_cond_attr(c, "is_control", False)
        n_tri = _get_cond_attr(c, "n_trials", 0)

        if is_ctrl:
            base_full = "Control"
            base_short = "Ctrl"
        else:
            p_full: List[str] = []
            p_short: List[str] = []

            # Check continuous
            is_c_cont = (
                any(kw in f"{_get_cond_attr(c, 'notes', '')} {_get_cond_attr(c, 'label', '')}".lower() for kw in ("cont", "continuous"))
                or (_get_cond_attr(c, "stim_freq_hz") is not None and _get_cond_attr(c, "stim_freq_hz") <= 2.5)
            )

            if is_c_cont:
                p_full.append("2Hz-cont")
                p_short.append("2Hz-cont")
            elif vary_pulses:
                p_val = _get_cond_attr(c, "pulse_count")
                if p_val is not None:
                    p_full.append(f"{p_val} pulse" if p_val == 1 else f"{p_val} pulses")
                    p_short.append(f"{p_val}p")

            if vary_freqs and not is_c_cont:
                f_val = _get_cond_attr(c, "stim_freq_hz")
                if f_val is not None and not np.isnan(f_val):
                    f_txt = f"{int(f_val)} Hz" if float(f_val).is_integer() else f"{f_val:.1f} Hz"
                    f_short = f"{int(f_val)}Hz" if float(f_val).is_integer() else f"{f_val:.1f}Hz"
                    p_full.append(f_txt)
                    p_short.append(f_short)

            if vary_channels:
                ch_val = _get_cond_attr(c, "stim_channels")
                if ch_val and str(ch_val).strip() and str(ch_val).lower() not in ("none", "nan"):
                    p_full.append(f"ch {str(ch_val).strip()}")
                    p_short.append(f"ch {str(ch_val).strip()}")

            # Current steering
            is_cs = (
                "curr steer" in f"{_get_cond_attr(c, 'notes', '')} {_get_cond_attr(c, 'label', '')}".lower()
                or "current steer" in f"{_get_cond_attr(c, 'notes', '')} {_get_cond_attr(c, 'label', '')}".lower()
            )
            if is_cs:
                p_full.append("Current Steering")
                p_short.append("Curr Steer")

            # Additional notes if unique
            notes_str = str(_get_cond_attr(c, "notes", "")).strip()
            if notes_str and notes_str.lower() not in ("baseline", "control", "ctrl", "nan", "none", "") and "curr steer" not in notes_str.lower():
                p_full.append(notes_str)
                p_short.append(notes_str)

            if not p_full:
                p_full.append("Stim")
                p_short.append("Stim")

            base_full = ", ".join(p_full)
            base_short = ", ".join(p_short)

        # Apply SHOW_CONDITION_INDICES
        if show_indices:
            comb_brs = _get_cond_attr(c, "combined_br_indices", None)
            if comb_brs and len(comb_brs) > 1:
                br_tag = f"BR {','.join(map(str, comb_brs))}"
            else:
                br_tag = f"BR {cond_num}"
            full_lbl = f"{br_tag}: {base_full}"
            short_lbl = f"{br_tag} ({base_short})"
        else:
            full_lbl = base_full
            short_lbl = base_short

        # Apply SHOW_TRIAL_COUNTS
        if show_trial_counts and n_tri > 0:
            full_lbl = f"{full_lbl} (n={n_tri})"
            short_lbl = f"{short_lbl} (n={n_tri})"

        full_labels[cond_num] = full_lbl
        short_labels[cond_num] = short_lbl

    return full_labels, short_labels, fixed_summary


def apply_adaptive_condition_labels(
    conditions: List[ReachCondition],
    show_indices: Optional[bool] = None,
    show_trial_counts: Optional[bool] = None,
) -> str:
    """Apply adaptive labels directly to ReachCondition objects in place and return fixed_summary."""
    if not conditions:
        return ""
    full_labels, short_labels, fixed_summary = build_adaptive_condition_labels(
        conditions, show_indices=show_indices, show_trial_counts=show_trial_counts
    )
    for c in conditions:
        c.label = full_labels.get(c.cond_num, c.label)
        c.base_label = full_labels.get(c.cond_num, c.base_label)
        c.short_label = short_labels.get(c.cond_num, c.label)
        c.fixed_params_summary = fixed_summary
    return fixed_summary


def load_conditions_for_target(
    session: str,
    target: str,
    peri_root: Path,
    dlc_3d_dir: Optional[Path] = None,
    compare_conditions: Optional[Sequence[int]] = None,
    control_br_indices: Optional[Sequence[int]] = None,
    keypoint: str = DEFAULT_KEYPOINT,
    time_window: Tuple[float, float] = DEFAULT_PLOT_WINDOW_MS,
    align_task_frame: bool = ALIGN_TO_TASK_FRAME,
    square_size_mm: float = SQUARE_SIZE_MM,
) -> List[ReachCondition]:
    """
    Load continuous triangulated 3D DLC kinematics, segment reaches around event times,
    rotate into calibrated task frame, and return condition objects.
    """
    if not HAS_3D_KINEMATICS or p3d is None:
        _log("Error: plot_3d_kinematics module is unavailable.")
        return []

    target_str = target.upper()
    _log(f"Discovering 3D reaches for Session {session}, Target {target_str}...")

    sess_dir = p3d.resolve_session_dir(session=session, peri_root=peri_root)
    metadata_csv = p3d.find_metadata_csv(session=session, peri_root=peri_root)

    if dlc_3d_dir is None:
        if sess_dir and (sess_dir / "Video" / "DLC_3D_new").exists():
            dlc_3d_dir = sess_dir / "Video" / "DLC_3D_new"
        elif sess_dir and (sess_dir / "Video" / "DLC_3D").exists():
            dlc_3d_dir = sess_dir / "Video" / "DLC_3D"
        elif VIDEO_ROOT is not None and (VIDEO_ROOT / "DLC_3D_new").exists():
            dlc_3d_dir = VIDEO_ROOT / "DLC_3D_new"
        else:
            dlc_3d_dir = peri_root.parent / "Video" / "DLC_3D_new"

    aligned_ckpt = (sess_dir / "results" / "checkpoints" / "Aligned") if sess_dir else ALIGNED_CKPT_ROOT
    behv_ckpt = (sess_dir / "results" / "checkpoints" / "Behavior") if sess_dir else BEHV_CKPT_ROOT

    session_meta: Dict[int, Dict[str, Any]] = {}
    if metadata_csv and metadata_csv.exists():
        try:
            df_meta = pd.read_csv(metadata_csv)
            for _, row in df_meta.iterrows():
                br_val = row.get("BR_File") or row.get("br_idx") or row.get("Condition")
                if pd.notna(br_val):
                    try:
                        session_meta[int(br_val)] = row.to_dict()
                    except Exception:
                        pass
        except Exception as e:
            _log(f"Warning: Could not parse metadata CSV: {e}")

    # Discover PeriStim checkpoint files
    peri_candidates = [
        peri_root / "stim_reaches" / f"target_{target_str}" / "*.npz",
        peri_root / "stim_reaches" / f"target_{target.lower()}" / "*.npz",
        peri_root / "control_reaches" / f"target_{target_str}" / "*.npz",
        peri_root / "control_reaches" / f"target_{target.lower()}" / "*.npz",
        peri_root / f"Target_{target_str}" / "*.npz",
        peri_root / f"Target_{target.lower()}" / "*.npz",
        peri_root / "*.npz",
    ]

    discovered_files: List[Path] = []
    for pat in peri_candidates:
        if pat.parent.exists():
            for f in sorted(list(pat.parent.glob(pat.name))):
                if f not in discovered_files and not f.name.startswith("._"):
                    discovered_files.append(f)

    session_landmarks = None
    if dlc_3d_dir and dlc_3d_dir.exists():
        session_landmarks = p3d.scan_session_landmarks(dlc_3d_dir, square_size_mm=square_size_mm)

    dlc_cache: Dict[Path, Tuple[pd.DataFrame, List[str]]] = {}
    conditions: List[ReachCondition] = []

    # Pass 1: Parse candidate files, target events, and control status
    parsed_candidates: List[Dict[str, Any]] = []
    for peri_path in discovered_files:
        path_str = str(peri_path).replace("\\", "/").lower()
        other_tgt = "b" if target_str == "A" else "a"
        if f"target_{other_tgt}" in path_str or f"target{other_tgt}" in path_str:
            if f"target_{target.lower()}" not in path_str and f"target{target.lower()}" not in path_str:
                continue

        try:
            peri = np.load(peri_path, allow_pickle=True)
        except Exception:
            continue

        br_idx = -1
        if "br_idx" in peri:
            try:
                br_idx = int(peri["br_idx"])
            except Exception:
                pass
        if br_idx < 0:
            m = re.search(r"BR_(\d+)", peri_path.name, re.IGNORECASE)
            if m:
                br_idx = int(m.group(1))

        if br_idx < 0:
            continue

        csv_meta = session_meta.get(br_idx, {})
        notes = str(csv_meta.get("notes", "")).strip()

        cond_str = str(peri.get("overall_title", f"BR_{br_idx}"))
        pulse_count = int(peri["stim_pulse_counts"].flat[0]) if "stim_pulse_counts" in peri and peri["stim_pulse_counts"].size else None
        stim_freq_hz = float(peri["stim_freq_hz"]) if "stim_freq_hz" in peri else csv_meta.get("stim_freq_hz")
        stim_dur_nominal_ms = float(peri["stim_dur_nominal_ms"]) if "stim_dur_nominal_ms" in peri else None

        if pulse_count is None:
            for k in ["pulse_count", "pulses", "n_pulses", "Pulse_Count", "Pulses"]:
                if k in csv_meta and pd.notna(csv_meta[k]):
                    try:
                        pulse_count = int(csv_meta[k])
                        break
                    except Exception:
                        pass

        if stim_freq_hz is None:
            for k in ["stim_freq_hz", "frequency", "Frequency", "freq_hz", "Stim_Freq_Hz"]:
                if k in csv_meta and pd.notna(csv_meta[k]):
                    try:
                        stim_freq_hz = float(csv_meta[k])
                        break
                    except Exception:
                        pass

        if stim_dur_nominal_ms is None or stim_dur_nominal_ms <= 0:
            for k in ["stim_dur_nominal_ms", "stim_dur_ms", "duration_ms", "Duration", "stim_duration_ms", "stim_dur"]:
                if k in csv_meta and pd.notna(csv_meta[k]):
                    try:
                        stim_dur_nominal_ms = float(csv_meta[k])
                        break
                    except Exception:
                        pass

        if (stim_dur_nominal_ms is None or stim_dur_nominal_ms <= 0) and pulse_count and stim_freq_hz and stim_freq_hz > 0:
            stim_dur_nominal_ms = float(pulse_count * 1000.0 / stim_freq_hz)

        stim_channels = extract_stim_channels_info(
            peri=peri,
            csv_meta=csv_meta,
            notes=notes,
            cond_str=cond_str,
        )

        notes_clean = notes.lower() if notes else ""
        cond_clean = cond_str.lower() if cond_str else ""
        stem_clean = peri_path.stem.lower()
        path_clean = path_str

        is_control = bool(
            (control_br_indices and br_idx in control_br_indices)
            or ("control_reaches" in path_clean)
            or ("target_control" in path_clean)
            or any(kw in notes_clean for kw in ("control", "baseline", "ctrl", "no stim", "nostim", "no_stim"))
            or any(kw in cond_clean for kw in ("control", "baseline", "ctrl", "no stim", "nostim", "no_stim"))
            or any(kw in stem_clean for kw in ("control", "baseline", "ctrl"))
            or (pulse_count is not None and pulse_count == 0)
            or (stim_freq_hz is not None and stim_freq_hz == 0.0)
            or (stim_dur_nominal_ms is not None and stim_dur_nominal_ms == 0.0)
        )

        event_ms = None
        for k in ["event_ms_sub", "event_ms"]:
            if k in peri:
                event_ms = peri[k]
                break

        if event_ms is None or len(event_ms) == 0:
            continue

        raw_trial_indices = None
        for k in ["raw_trial_indices", "raw_indices", "trial_indices"]:
            if k in peri:
                raw_trial_indices = peri[k]
                break

        labels = None
        for k in ["labels_sub", "labels", "trial_labels"]:
            if k in peri:
                labels = peri[k]
                break

        if labels is not None and np.size(labels) == np.size(event_ms):
            labels_arr = np.asarray(labels).astype(str)
            t_mask = np.array([lbl.upper() == target_str for lbl in labels_arr])
            if not np.any(t_mask):
                continue
            event_ms = event_ms[t_mask]
            if raw_trial_indices is not None and len(raw_trial_indices) == len(t_mask):
                raw_trial_indices = raw_trial_indices[t_mask]

        if len(event_ms) == 0:
            continue

        peri_meta = {}
        if "meta" in peri and hasattr(peri["meta"], "item"):
            try:
                peri_meta = peri["meta"].item()
            except Exception:
                pass

        parsed_candidates.append({
            "peri_path": peri_path,
            "br_idx": br_idx,
            "is_control": is_control,
            "event_ms": event_ms,
            "raw_trial_indices": raw_trial_indices,
            "peri_meta": peri_meta,
            "csv_meta": csv_meta,
            "notes": notes,
            "cond_str": cond_str,
            "pulse_count": pulse_count,
            "stim_freq_hz": stim_freq_hz,
            "stim_dur_nominal_ms": stim_dur_nominal_ms,
            "stim_channels": stim_channels,
            "n_events": len(event_ms),
        })

    # Pass 2: Select conditions based on compare_conditions / process_only rule
    if compare_conditions:
        comp_set = set(compare_conditions)
        req_matched = [c for c in parsed_candidates if c["br_idx"] in comp_set]
        ctrl_in_req = [c for c in req_matched if c["is_control"] and c["n_events"] > 0]

        if ctrl_in_req:
            # Control is provided in process_only and has valid reaches for this target!
            # Use requested control condition(s) (will be combined into 1 unified Control group if multiple).
            _log(f"Target {target_str}: Control condition(s) {sorted(set(c['br_idx'] for c in ctrl_in_req))} provided in process_only with reaches for Target {target_str}. (Will be combined into 1 group).")
            selected_candidates = req_matched
        else:
            # Control NOT provided in process_only, or target is not available in the process_only control session.
            # Look for all available control sessions with reaches for this target to combine!
            avail_ctrls = [c for c in parsed_candidates if c["is_control"] and c["n_events"] > 0]
            if avail_ctrls:
                ctrl_idxs = sorted(set(c['br_idx'] for c in avail_ctrls))
                _log(f"Target {target_str}: Control not in process_only; including all {len(avail_ctrls)} available control condition(s) (BR {ctrl_idxs}) to combine for Target {target_str}.")
                selected_candidates = req_matched + avail_ctrls
            else:
                _log(f"Target {target_str}: No control conditions available with reaches for Target {target_str}; proceeding with requested conditions.")
                selected_candidates = req_matched
    else:
        # No process_only filtering: take all available control conditions to combine + all stim conditions
        avail_ctrls = [c for c in parsed_candidates if c["is_control"] and c["n_events"] > 0]
        if avail_ctrls:
            ctrl_idxs = sorted(set(c['br_idx'] for c in avail_ctrls))
            _log(f"Target {target_str}: Including all {len(avail_ctrls)} available control condition(s) (BR {ctrl_idxs}) to combine for Target {target_str}.")
            selected_candidates = avail_ctrls + [c for c in parsed_candidates if not c["is_control"]]
        else:
            selected_candidates = parsed_candidates

    # Deduplicate selected candidates by br_idx (keeping candidate with most events)
    dedup_dict: Dict[int, Dict[str, Any]] = {}
    for c in selected_candidates:
        b = c["br_idx"]
        if b not in dedup_dict or c["n_events"] > dedup_dict[b]["n_events"]:
            dedup_dict[b] = c
    selected_candidates = list(dedup_dict.values())

    # Pass 3: Load DLC kinematics and cut reaches for selected candidates
    for cand in selected_candidates:
        peri_path = cand["peri_path"]
        br_idx = cand["br_idx"]
        is_control = cand["is_control"]
        event_ms = cand["event_ms"]
        raw_trial_indices = cand["raw_trial_indices"]
        peri_meta = cand["peri_meta"]
        csv_meta = cand["csv_meta"]
        notes = cand["notes"]
        cond_str = cand["cond_str"]
        pulse_count = cand["pulse_count"]
        stim_freq_hz = cand["stim_freq_hz"]
        stim_dur_nominal_ms = cand["stim_dur_nominal_ms"]
        stim_channels = cand.get("stim_channels")

        vid_idx = p3d.resolve_video_index_for_br(br_idx, metadata_csv, peri_meta)
        if vid_idx is None:
            vid_idx = br_idx

        dlc_csv_path = p3d.find_3d_dlc_csv(dlc_3d_dir, session, vid_idx, br_idx)
        if dlc_csv_path is None or not dlc_csv_path.exists():
            continue

        if dlc_csv_path not in dlc_cache:
            df_dlc, bodyparts = p3d.load_dlc_3d_file(dlc_csv_path, square_size_mm=square_size_mm)
            if df_dlc is None:
                continue
            dlc_cache[dlc_csv_path] = (df_dlc, bodyparts)
        else:
            df_dlc, bodyparts = dlc_cache[dlc_csv_path]

        frame_t_ms = p3d.load_frame_timeline_ms(
            br_idx=br_idx,
            vid_idx=vid_idx,
            session=session,
            aligned_ckpt_root=aligned_ckpt,
            behv_ckpt_root=behv_ckpt,
        )

        if frame_t_ms is None:
            frame_t_ms = np.arange(len(df_dlc)) * 10.0

        try:
            rel_t, x_all, y_all, z_all, valid_mask = p3d.segment_condition_3d_reaches(
                dlc_df=df_dlc,
                frame_t_ms=frame_t_ms,
                event_ms=event_ms,
                keypoint=keypoint,
                pre_zero_ms=p3d.PRE_ZERO_MS,
                post_zero_ms=p3d.POST_ZERO_MS,
                step_ms=p3d.STEP_MS,
            )
        except Exception:
            continue

        R_task_cond = None
        p_start_cond = None
        if align_task_frame:
            try:
                R_task_cond, p_start_cond = p3d.compute_task_rotation_matrix(
                    dlc_df=df_dlc,
                    session_landmarks=session_landmarks,
                    square_size_mm=square_size_mm,
                )
                pts_cam = np.stack([x_all, y_all, z_all], axis=0) - p_start_cond.reshape(3, 1, 1)
                pts_task = np.einsum('ji,jkl->ikl', R_task_cond, pts_cam)
                x_all, y_all, z_all = pts_task[0], pts_task[1], pts_task[2]
            except Exception:
                pass

        n_valid = int(np.sum(valid_mask))
        if n_valid < MIN_REACHES_PER_CONDITION:
            continue

        # Extract only valid reach trials
        x_val = x_all[valid_mask]
        y_val = y_all[valid_mask]
        z_val = z_all[valid_mask]
        raw_idx_val = raw_trial_indices[valid_mask] if raw_trial_indices is not None else None

        # Slice to requested time window
        t_start, t_end = time_window
        win_mask = (rel_t >= t_start) & (rel_t <= t_end)
        if not np.any(win_mask):
            win_mask = (rel_t >= 0.0) & (rel_t <= 450.0)

        t_sub = rel_t[win_mask]
        x_sub = x_val[:, win_mask]
        y_sub = y_val[:, win_mask]
        z_sub = z_val[:, win_mask]

        cond_info_dict = {
            "is_control": is_control,
            "pulse_count": pulse_count,
            "stim_freq_hz": stim_freq_hz,
            "notes": notes,
            "stim_channels": stim_channels,
            "cond_str": cond_str,
        }
        base_label = p3d.format_condition_label(
            cond_num=br_idx,
            cond_info=cond_info_dict,
            control_br_indices=control_br_indices,
            include_target=False,
        )

        if not is_control:
            # Check for continuous stimulation
            is_cont = any(kw in f"{notes} {cond_str}".lower() for kw in ("cont", "continuous")) or (stim_freq_hz is not None and stim_freq_hz <= 2.5)
            if is_cont and "cont" not in base_label.lower():
                f_txt = f"{int(stim_freq_hz)} Hz-cont" if stim_freq_hz else "2Hz-cont"
                if "pulse" in base_label.lower():
                    base_label = re.sub(r"\d+\s*pulses?(?:,\s*[\d.]+\s*Hz)?", f"{f_txt}", base_label, flags=re.IGNORECASE)
                elif "hz" in base_label.lower():
                    base_label = re.sub(r"[\d.]+\s*Hz", f"{f_txt}", base_label, flags=re.IGNORECASE)
                else:
                    base_label = f"{base_label}, {f_txt}"

            if stim_channels:
                ch_tag = f"ch {stim_channels}"
                if ch_tag.lower() not in base_label.lower() and f"ch{stim_channels}".lower() not in base_label.lower():
                    base_label = f"{base_label}, {ch_tag}"

            if not is_cont and stim_dur_nominal_ms is not None and stim_dur_nominal_ms > 0:
                dur_txt = f"{stim_dur_nominal_ms:.1f} ms" if stim_dur_nominal_ms < 10 and stim_dur_nominal_ms % 1 != 0 else f"{int(stim_dur_nominal_ms)} ms"
                if "ms" not in base_label.lower():
                    base_label = f"{base_label} ({dur_txt})"

        # Label including trial count
        label = f"{base_label} (n={n_valid})"

        rc = ReachCondition(
            cond_num=br_idx,
            br_idx=br_idx,
            label=label,
            base_label=base_label,
            target=target_str,
            is_control=is_control,
            time=t_sub,
            x=x_sub,
            y=y_sub,
            z=z_sub,
            raw_trial_indices=raw_idx_val,
            pulse_count=pulse_count,
            stim_freq_hz=stim_freq_hz,
            stim_dur_ms=stim_dur_nominal_ms,
            notes=notes,
            stim_channels=stim_channels,
            R_task=R_task_cond,
            p_start=p_start_cond,
            dlc_3d_dir=dlc_3d_dir,
            session_landmarks=session_landmarks,
            square_size_mm=square_size_mm,
        )
        conditions.append(rc)

    # Combine multiple control conditions for this target into a single unified Control group
    ctrl_conds = [c for c in conditions if c.is_control]
    stim_conds = [c for c in conditions if not c.is_control]

    if len(ctrl_conds) > 1:
        min_t = min(c.x.shape[1] for c in ctrl_conds)
        x_comb = np.vstack([c.x[:, :min_t] for c in ctrl_conds])
        y_comb = np.vstack([c.y[:, :min_t] for c in ctrl_conds])
        z_comb = np.vstack([c.z[:, :min_t] for c in ctrl_conds])

        raw_idx_list = [c.raw_trial_indices for c in ctrl_conds if c.raw_trial_indices is not None]
        raw_idx_comb = np.concatenate(raw_idx_list) if len(raw_idx_list) == len(ctrl_conds) else None

        br_list = sorted(list(set(c.br_idx for c in ctrl_conds)))
        primary_br = br_list[0]
        n_tot = x_comb.shape[0]

        comb_notes = f"Combined Control (BR {', '.join(map(str, br_list))})"
        comb_base = "Control"
        comb_label = f"Control (n={n_tot})"

        combined_ctrl = ReachCondition(
            cond_num=primary_br,
            br_idx=primary_br,
            label=comb_label,
            base_label=comb_base,
            target=target_str,
            is_control=True,
            time=ctrl_conds[0].time[:min_t],
            x=x_comb,
            y=y_comb,
            z=z_comb,
            raw_trial_indices=raw_idx_comb,
            pulse_count=0,
            stim_freq_hz=0.0,
            stim_dur_ms=0.0,
            notes=comb_notes,
            stim_channels=None,
            combined_br_indices=br_list,
            R_task=ctrl_conds[0].R_task,
            p_start=ctrl_conds[0].p_start,
            dlc_3d_dir=ctrl_conds[0].dlc_3d_dir,
            session_landmarks=ctrl_conds[0].session_landmarks,
            square_size_mm=square_size_mm,
        )
        _log(f"Target {target_str}: Combined {len(ctrl_conds)} control conditions (BR {br_list}) into 1 unified Control group ({n_tot} total reaches).")
        conditions = [combined_ctrl] + stim_conds

    # Sort conditions: Control conditions first, followed by stim conditions sorted by BR index
    conditions.sort(key=lambda c: (0 if c.is_control else 1, c.br_idx))
    fixed_summary = apply_adaptive_condition_labels(conditions)
    _log(f"Loaded {len(conditions)} valid conditions for Target {target_str} (Control count: {sum(c.is_control for c in conditions)}):")
    if fixed_summary:
        _log(f"   [{fixed_summary}]")
    for c in conditions:
        ctrl_mark = " [CONTROL]" if c.is_control else ""
        _log(f"   • {c.label}{ctrl_mark}")
    return conditions


# =============================================================================
# Behavioral Representational Similarity Analysis (Kinematic RDM / RSA)
# =============================================================================

def resample_trajectory_by_arclength(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    time: Optional[np.ndarray] = None,
    n_progress: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample each 3D reach trajectory onto a common 0.0 to 1.0 arc-length progress axis,
    defined strictly starting at zero time (t = 0.0 ms, stimulation onset / IR cross)
    forward until the reach endpoint at physical touchscreen contact.

    Parameters:
        x, y, z: Arrays of shape (N_trials, T) in physical task frame (mm).
        time: Optional array of time points (T,) in ms. If provided, progress 0% begins at t >= 0.0 ms.
        n_progress: Number of normalized path-progress samples (e.g. 100 samples).

    Returns:
        x_norm, y_norm, z_norm: Arrays of shape (N_trials, n_progress) in mm.
    """
    n_trials, n_t = x.shape
    progress_grid = np.linspace(0.0, 1.0, n_progress)

    x_out = np.full((n_trials, n_progress), np.nan)
    y_out = np.full((n_trials, n_progress), np.nan)
    z_out = np.full((n_trials, n_progress), np.nan)

    t_arr = np.asarray(time, dtype=float) if time is not None and len(time) == n_t else None

    for i in range(n_trials):
        pts = np.column_stack([x[i], y[i], z[i]])
        valid = np.all(np.isfinite(pts), axis=1)

        # Restrict strictly to zero time forward (t >= 0.0 ms)
        if t_arr is not None:
            valid = valid & (t_arr >= 0.0)
            t_sub = t_arr[valid]
        else:
            t_sub = None

        if valid.sum() < 3:
            continue

        pts_valid = pts[valid]
        y_vals = pts_valid[:, 1]

        # Truncate at touchscreen endpoint (at or after 75 ms if available)
        if t_sub is not None and np.any(t_sub >= 75.0):
            cand_mask = (t_sub >= 75.0)
            cand_indices = np.where(cand_mask)[0]
            y_max_idx = int(cand_indices[np.nanargmax(y_vals[cand_indices])])
        else:
            y_max_idx = int(np.nanargmax(y_vals))

        if y_max_idx < 1:
            continue

        pts_fwd = pts_valid[:y_max_idx + 1]

        step_dist = np.sqrt(np.sum(np.diff(pts_fwd, axis=0) ** 2, axis=1))
        cumulative_dist = np.r_[0.0, np.cumsum(step_dist)]

        total_length = cumulative_dist[-1]
        if not np.isfinite(total_length) or total_length <= 1e-6:
            continue

        progress = cumulative_dist / total_length

        # Remove repeated progress values, e.g. pauses / zero displacement samples
        keep = np.r_[True, np.diff(progress) > 1e-8]
        progress = progress[keep]
        pts_fwd = pts_fwd[keep]

        if len(progress) < 2:
            continue

        progress[0] = 0.0
        progress[-1] = 1.0

        x_out[i] = np.interp(progress_grid, progress, pts_fwd[:, 0])
        y_out[i] = np.interp(progress_grid, progress, pts_fwd[:, 1])
        z_out[i] = np.interp(progress_grid, progress, pts_fwd[:, 2])

    return x_out, y_out, z_out


def resample_trajectory_by_forward_y(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    time: Optional[np.ndarray] = None,
    n_progress: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample each 3D reach trajectory onto a common 0.0 to 1.0 Forward Y progress axis,
    defined strictly starting at zero time (t = 0.0 ms, stimulation onset / IR cross)
    forward until the reach endpoint at physical touchscreen contact (t_peak >= 75 ms).

    Parameters:
        x, y, z: Arrays of shape (N_trials, T) in physical task frame (mm).
        time: Optional array of time points (T,) in ms. If provided, progress 0% begins at t >= 0.0 ms.
        n_progress: Number of normalized Forward Y progress samples (default: 100).

    Returns:
        x_norm, y_norm, z_norm: Arrays of shape (N_trials, n_progress) in mm.
    """
    n_trials, n_t = x.shape
    progress_grid = np.linspace(0.0, 1.0, n_progress)

    x_out = np.full((n_trials, n_progress), np.nan)
    y_out = np.full((n_trials, n_progress), np.nan)
    z_out = np.full((n_trials, n_progress), np.nan)

    t_arr = np.asarray(time, dtype=float) if time is not None and len(time) == n_t else None

    for i in range(n_trials):
        pts = np.column_stack([x[i], y[i], z[i]])
        valid = np.all(np.isfinite(pts), axis=1)

        # Restrict strictly to zero time forward (t >= 0.0 ms)
        if t_arr is not None:
            valid = valid & (t_arr >= 0.0)
            t_sub = t_arr[valid]
        else:
            t_sub = None

        if valid.sum() < 3:
            continue

        pts_valid = pts[valid]
        y_vals = pts_valid[:, 1]

        # Find touchscreen arrival endpoint (peak forward extension Y at or after 75 ms if available)
        if t_sub is not None and np.any(t_sub >= 75.0):
            cand_mask = (t_sub >= 75.0)
            cand_indices = np.where(cand_mask)[0]
            y_max_idx = int(cand_indices[np.nanargmax(y_vals[cand_indices])])
        else:
            y_max_idx = int(np.nanargmax(y_vals))

        y_start = y_vals[0]
        y_max = y_vals[y_max_idx]
        delta_y = y_max - y_start

        # Require outbound progress from t = 0 to endpoint (no physical mm requirement)
        if y_max_idx < 1 or delta_y <= 1e-4:
            continue

        # Truncate trajectory from t = 0.0 ms up to touchscreen endpoint arrival
        pts_fwd = pts_valid[:y_max_idx + 1]
        y_fwd = y_vals[:y_max_idx + 1]

        # Enforce monotonic forward progress to eliminate tracking noise
        y_mono = np.maximum.accumulate(y_fwd)
        progress = (y_mono - y_start) / delta_y
        progress = np.clip(progress, 0.0, 1.0)

        # Keep strictly increasing points for 1D interpolation
        keep = np.r_[True, np.diff(progress) > 1e-6]
        progress_clean = progress[keep]
        pts_clean = pts_fwd[keep]

        if len(progress_clean) < 2:
            continue

        progress_clean[0] = 0.0
        progress_clean[-1] = 1.0

        x_out[i] = np.interp(progress_grid, progress_clean, pts_clean[:, 0])
        y_out[i] = np.interp(progress_grid, progress_clean, pts_clean[:, 1])
        z_out[i] = np.interp(progress_grid, progress_clean, pts_clean[:, 2])

    return x_out, y_out, z_out


def resample_reach_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    time: Optional[np.ndarray] = None,
    n_progress: int = 100,
    mode: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample reaches along either 'forward_y' (default, physical reach depth axis)
    or 'arclength' (3D path distance), defined strictly starting at zero time (t = 0.0 ms)
    forward until the reach endpoint at physical touchscreen contact.
    """
    m = str(mode or RSA_PROGRESS_MODE).lower().strip()
    if m in ("forward_y", "y", "depth", "forward"):
        return resample_trajectory_by_forward_y(x, y, z, time=time, n_progress=n_progress)
    else:
        return resample_trajectory_by_arclength(x, y, z, time=time, n_progress=n_progress)



def compute_pairwise_reach_distances(
    x1: np.ndarray, y1: np.ndarray, z1: np.ndarray,
    x2: np.ndarray, y2: np.ndarray, z2: np.ndarray,
    aggregation: Optional[str] = None,
    stat: Optional[str] = None,
    axis: Optional[str] = None,
) -> np.ndarray:
    """
    Compute pairwise reach distances between reaches in group 1 (N1 trials)
    and group 2 (N2 trials).

    Args:
        aggregation: 'median' (default: typical spatial path deviation),
                     'p90' / 'q90' (90th percentile: captures brief transient deflections),
                     'mean' (integrated average deviation),
                     'max' / 'peak' (maximum deflection).
        stat: Backward-compatibility alias for aggregation.
        axis: Dimension along which to measure dissimilarity:
              None or '3d': Isotropic 3D Euclidean distance sqrt(dx^2 + dy^2 + dz^2)
              'x' or 'lateral': Pure lateral dissimilarity |dx| (steering / veer)
              'z' or 'vertical': Pure vertical dissimilarity |dz| (elevation / dip)
              'y' or 'forward': Pure forward depth dissimilarity |dy|

    Returns:
        dist_mat: (N1, N2) array of distances in mm.
    """
    agg = aggregation if aggregation is not None else (stat if stat is not None else RSA_DISTANCE_AGGREGATION)
    agg_clean = str(agg).lower().strip()

    # x1: (N1, P), x2: (N2, P) where P is either time samples or progress samples
    n1, p1 = x1.shape
    n2, p2 = x2.shape
    assert p1 == p2, "Sample lengths must match."

    # Pairwise coordinate deltas at each sample
    dx = x1[:, None, :] - x2[None, :, :]  # (N1, N2, P)
    dy = y1[:, None, :] - y2[None, :, :]
    dz = z1[:, None, :] - z2[None, :, :]

    ax_clean = str(axis).lower().strip() if axis is not None else "3d"
    if ax_clean in ("x", "lateral", "lat"):
        point_dist = np.abs(dx)
    elif ax_clean in ("z", "vertical", "vert"):
        point_dist = np.abs(dz)
    elif ax_clean in ("y", "forward", "depth"):
        point_dist = np.abs(dy)
    else:
        point_dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)  # (N1, N2, P)

    if agg_clean in ("median", "med"):
        return np.nanmedian(point_dist, axis=2)        # (N1, N2)
    elif agg_clean in ("p90", "q90", "90th", "percentile90"):
        return np.nanpercentile(point_dist, 90, axis=2) # (N1, N2)
    elif agg_clean in ("max", "peak"):
        return np.nanmax(point_dist, axis=2)           # (N1, N2)
    else:
        return np.nanmean(point_dist, axis=2)          # (N1, N2)


def build_behavioral_rdm(
    conditions: List[ReachCondition],
    metric: str = "euclidean",
    align_by_progress: Optional[bool] = None,
    n_progress_samples: int = RSA_N_PROGRESS_SAMPLES,
    aggregation: Optional[str] = None,
    stat: Optional[str] = None,
    progress_mode: Optional[str] = None,
    axis: Optional[str] = None,
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Construct full trial-by-trial Kinematic Dissimilarity Matrix (RDM / RSM).

    Args:
        conditions: List of ReachCondition objects.
        metric: 'euclidean' (distance in mm) or 'correlation' (1 - Pearson r).
        align_by_progress: If True (default), resample each reach by spatial progress
                           (Forward Y or arc-length) to decouple movement speed from path geometry.
        n_progress_samples: Number of normalized path samples (default: 100).
        aggregation: 'median' (default), 'p90', 'mean', or 'max'.
        stat: Backward-compatibility alias for aggregation.
        progress_mode: 'forward_y' (default: normalized along physical reach depth Y) or 'arclength'.
        axis: Dimension along which to measure dissimilarity:
              None or '3d': Isotropic 3D Euclidean distance
              'x' or 'lateral': Pure lateral dissimilarity |dx|
              'z' or 'vertical': Pure vertical dissimilarity |dz|
              'y' or 'forward': Pure forward depth dissimilarity |dy|

    Returns:
        RDM: (N_total, N_total) dissimilarity matrix in mm.
        block_sizes: Number of trials in each condition block.
        block_labels: Formatted label for each condition block.
    """
    if align_by_progress is None:
        align_by_progress = RSA_ALIGN_BY_PROGRESS
    agg = aggregation if aggregation is not None else (stat if stat is not None else RSA_DISTANCE_AGGREGATION)
    pm = progress_mode or RSA_PROGRESS_MODE

    block_sizes = [c.n_trials for c in conditions]
    block_labels = [c.label for c in conditions]
    n_total = sum(block_sizes)

    RDM = np.zeros((n_total, n_total), dtype=float)

    # Flatten feature vectors for correlation metric
    flattened_features: List[np.ndarray] = []
    if metric == "correlation":
        ax_clean = str(axis).lower().strip() if axis is not None else "3d"
        for c in conditions:
            if ax_clean in ("x", "lateral", "lat"):
                feat = c.x.copy()
            elif ax_clean in ("z", "vertical", "vert"):
                feat = c.z.copy()
            elif ax_clean in ("y", "forward", "depth"):
                feat = c.y.copy()
            else:
                # Flatten [X, Y, Z] for each trial: shape (N, 3 * T)
                feat = np.hstack([c.x, c.y, c.z])
            # Center each trial
            feat = feat - np.nanmean(feat, axis=1, keepdims=True)
            norm = np.linalg.norm(feat, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            flattened_features.append(feat / norm)
        all_features = np.vstack(flattened_features)
        # Pearson correlation matrix
        corr = np.dot(all_features, all_features.T)
        corr = np.clip(corr, -1.0, 1.0)
        # Dissimilarity: Angular distance or 1 - r
        RDM = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))
        np.fill_diagonal(RDM, 0.0)
        return RDM, block_sizes, block_labels

    # Prepare trajectory data (progress-normalized or raw time)
    trajectory_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for c in conditions:
        if align_by_progress:
            x_use, y_use, z_use = resample_reach_trajectory(
                c.x, c.y, c.z, time=c.time, n_progress=n_progress_samples, mode=pm,
            )
        else:
            x_use, y_use, z_use = c.x, c.y, c.z
        trajectory_data.append((x_use, y_use, z_use))

    # Euclidean distance metric (mm)
    starts = np.r_[0, np.cumsum(block_sizes)[:-1]]
    for i in range(len(conditions)):
        s1, e1 = starts[i], starts[i] + block_sizes[i]
        x1, y1, z1 = trajectory_data[i]
        for j in range(len(conditions)):
            s2, e2 = starts[j], starts[j] + block_sizes[j]
            if i <= j:
                x2, y2, z2 = trajectory_data[j]
                d_sub = compute_pairwise_reach_distances(
                    x1, y1, z1, x2, y2, z2, aggregation=agg, axis=axis,
                )
                if i == j:
                    np.fill_diagonal(d_sub, 0.0)
                RDM[s1:e1, s2:e2] = d_sub
                RDM[s2:e2, s1:e1] = d_sub.T

    return RDM, block_sizes, block_labels


def compute_two_group_silhouette(
    D: np.ndarray,
    in_group_b: np.ndarray,
    stat: Optional[str] = None,
) -> float:
    """
    Fast two-group silhouette score for precomputed distance matrix D.
    Supports robust median aggregation (default) or mean aggregation.
    """
    if "_two_group_silhouette" in globals() and callable(_two_group_silhouette):
        return _two_group_silhouette(D, in_group_b)
    if stat is None:
        stat = DISSIMILARITY_STAT
    use_median = str(stat).lower().strip() in ("median", "med")

    gb = np.asarray(in_group_b, dtype=bool)
    ga = ~gb
    nb, na = int(gb.sum()), int(ga.sum())
    if nb < 2 or na < 2:
        return np.nan

    n_tot = len(in_group_b)
    if use_median:
        a = np.zeros(n_tot, dtype=float)
        b = np.zeros(n_tot, dtype=float)
        for i in range(n_tot):
            if gb[i]:
                own = gb.copy()
                own[i] = False
                a[i] = float(np.nanmedian(D[i, own]))
                b[i] = float(np.nanmedian(D[i, ga]))
            else:
                own = ga.copy()
                own[i] = False
                a[i] = float(np.nanmedian(D[i, own]))
                b[i] = float(np.nanmedian(D[i, gb]))
        denom = np.maximum(a, b)
        s = np.divide(b - a, denom, out=np.zeros_like(denom), where=denom > 0)
        return float(np.nanmedian(s))
    else:
        sum_b = D[:, gb].sum(axis=1)
        sum_a = D[:, ga].sum(axis=1)
        a = np.where(gb, sum_b / (nb - 1), sum_a / (na - 1))
        b = np.where(gb, sum_a / na, sum_b / nb)
        denom = np.maximum(a, b)
        s = np.divide(b - a, denom, out=np.zeros_like(denom), where=denom > 0)
        return float(np.mean(s))


def compute_pairwise_silhouette_matrix(
    D: np.ndarray,
    block_sizes: List[int],
    n_perm: int = N_PERMUTATIONS,
    rng: Optional[np.random.Generator] = None,
    stat: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute condition-level pairwise silhouette matrix with permutation testing.

    Returns:
        S: (K, K) pairwise silhouette matrix.
        P: (K, K) permutation p-values.
        Q: (K, K) FDR-corrected q-values.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    K = len(block_sizes)
    S = np.full((K, K), np.nan)
    P = np.full((K, K), np.nan)

    starts = np.r_[0, np.cumsum(block_sizes)[:-1]]

    for i in range(K):
        ni = block_sizes[i]
        si, ei = starts[i], starts[i] + ni
        for j in range(i + 1, K):
            nj = block_sizes[j]
            sj, ej = starts[j], starts[j] + nj

            # Extract submatrix for pair (i, j)
            idx = np.r_[np.arange(si, ei), np.arange(sj, ej)]
            D_sub = D[np.ix_(idx, idx)]
            in_b = np.r_[np.zeros(ni, dtype=bool), np.ones(nj, dtype=bool)]

            s_obs = compute_two_group_silhouette(D_sub, in_b, stat=stat)
            S[i, j] = s_obs
            S[j, i] = s_obs

            # Permutation test
            if n_perm > 0 and np.isfinite(s_obs):
                null_scores = np.zeros(n_perm)
                for p in range(n_perm):
                    shuffled_b = rng.permutation(in_b)
                    null_scores[p] = compute_two_group_silhouette(D_sub, shuffled_b, stat=stat)
                p_val = (1.0 + np.sum(null_scores >= s_obs)) / (n_perm + 1.0)
                P[i, j] = p_val
                P[j, i] = p_val

    # Benjamini-Hochberg FDR correction across unique pairs
    if "_fdr_adjust_pairs" in globals() and callable(_fdr_adjust_pairs):
        Q = _fdr_adjust_pairs(P)
    else:
        Q = np.full_like(P, np.nan)
        iu = np.triu_indices(K, k=1)
        p_vals = P[iu]
        valid = np.isfinite(p_vals)
        if np.any(valid):
            adj = stats.false_discovery_control(p_vals[valid], method="bh")
            Q[iu[0][valid], iu[1][valid]] = adj
            Q[iu[1][valid], iu[0][valid]] = adj

    return S, P, Q


def fdr_adjust_1d(p_values: Sequence[float], method: str = "bh") -> np.ndarray:
    """Adjust 1D p-values across multiple comparisons using Benjamini-Hochberg FDR correction."""
    p_arr = np.asarray(p_values, dtype=float)
    if len(p_arr) <= 1:
        return p_arr.copy()
    valid = np.isfinite(p_arr)
    q_arr = np.ones_like(p_arr)
    if np.any(valid):
        try:
            q_arr[valid] = stats.false_discovery_control(p_arr[valid], method=method)
        except Exception:
            p_val = p_arr[valid]
            n = len(p_val)
            order = np.argsort(p_val)
            ranked_p = p_val[order]
            q = np.empty(n, dtype=float)
            cummin = 1.0
            for i in range(n - 1, -1, -1):
                cummin = min(cummin, ranked_p[i] * n / (i + 1))
                q[order[i]] = cummin
            q_arr[valid] = np.clip(q, 0.0, 1.0)
    return q_arr


def compute_within_condition_stereotypy(
    D: np.ndarray,
    block_sizes: List[int],
    stat: Optional[str] = None,
) -> np.ndarray:
    """Compute within-condition pairwise distance (stereotypy in mm) along diagonal."""
    if stat is None:
        stat = DISSIMILARITY_STAT
    use_median = str(stat).lower().strip() in ("median", "med")

    K = len(block_sizes)
    within_scores = np.zeros(K)
    starts = np.r_[0, np.cumsum(block_sizes)[:-1]]

    for i in range(K):
        ni = block_sizes[i]
        si, ei = starts[i], starts[i] + ni
        D_within = D[si:ei, si:ei]
        if ni > 1:
            iu = np.triu_indices(ni, k=1)
            pair_vals = D_within[iu]
            within_scores[i] = float(np.nanmedian(pair_vals) if use_median else np.nanmean(pair_vals))
        else:
            within_scores[i] = 0.0

    return within_scores


# =============================================================================
# Time-Resolved Trajectory Variability Computation
# =============================================================================

def compute_time_resolved_variability(
    conditions: List[ReachCondition],
) -> Dict[int, Dict[str, Any]]:
    """
    Compute within-condition 3D trajectory variability at every time sample:
        V_c(t) = median_{i < j} ||r_i(t) - r_j(t)|| in mm.
    Also computes excess variability ratio and divergence from control.
    """
    results: Dict[int, Dict[str, Any]] = {}

    ctrl_candidates = [c for c in conditions if c.is_control]
    ctrl_cond = max(ctrl_candidates, key=lambda c: c.n_trials) if ctrl_candidates else None
    ctrl_mean_traj = ctrl_cond.mean_trajectory() if ctrl_cond else None

    for c in conditions:
        n_trials = c.n_trials
        n_t = c.n_time
        time = c.time

        if n_trials < 2:
            pairwise_time = np.zeros((1, n_t))
        else:
            # Compute pairwise distance across all unique trial pairs at each time sample
            pair_list = []
            for i in range(n_trials - 1):
                dx = c.x[i + 1:] - c.x[i]
                dy = c.y[i + 1:] - c.y[i]
                dz = c.z[i + 1:] - c.z[i]
                d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                pair_list.append(d)
            pairwise_time = np.vstack(pair_list)

        curve_median = np.nanmedian(pairwise_time, axis=0) if pairwise_time.size > 0 else np.zeros(n_t)
        curve_mean = np.nanmean(pairwise_time, axis=0) if pairwise_time.size > 0 else np.zeros(n_t)
        curve_std = np.nanstd(pairwise_time, axis=0) if pairwise_time.size > 0 else np.zeros(n_t)
        curve_sem = curve_std / math.sqrt(pairwise_time.shape[0]) if pairwise_time.shape[0] > 1 else np.zeros(n_t)

        # 3D divergence from control mean trajectory
        if ctrl_mean_traj is not None:
            mx_c, my_c, mz_c = ctrl_mean_traj
            dx_to_ctrl = c.x - mx_c[None, :]
            dy_to_ctrl = c.y - my_c[None, :]
            dz_to_ctrl = c.z - mz_c[None, :]
            dist_to_ctrl_all = np.sqrt(dx_to_ctrl ** 2 + dy_to_ctrl ** 2 + dz_to_ctrl ** 2)
            divergence_mean = np.nanmean(dist_to_ctrl_all, axis=0)
            divergence_sem = np.nanstd(dist_to_ctrl_all, axis=0) / math.sqrt(n_trials) if n_trials > 1 else np.zeros(n_t)
        else:
            divergence_mean = np.zeros(n_t)
            divergence_sem = np.zeros(n_t)

        results[c.cond_num] = {
            "cond_num": c.cond_num,
            "label": c.label,
            "base_label": getattr(c, "base_label", c.label),
            "is_control": c.is_control,
            "time": time,
            "var_median": curve_median,
            "var_mean": curve_mean,
            "var_sem": curve_sem,
            "divergence_mean": divergence_mean,
            "divergence_sem": divergence_sem,
            "stim_dur_ms": c.stim_dur_ms,
            "n_trials": n_trials,
        }

    return results


# =============================================================================
# Multi-Condition Directional Perturbation & Target Anchor Analysis
# =============================================================================

def resolve_target_spatial_anchor(
    target: str = "A",
    conditions: Optional[Sequence[ReachCondition]] = None,
    target_radius_mm: float = 1.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Resolve target spatial anchor as a 1 mm radius circle around the unperturbed
    control reach median endpoint. Fully decouples analysis from camera-based target tracking.
    """
    tgt_clean = str(target).upper().replace("TARGET_", "").strip()
    target_diameter_mm = 2.0 * target_radius_mm

    ctrl_end = None
    if conditions:
        ctrl_cond = next((c for c in conditions if c.is_control), None)
        if ctrl_cond is not None and ctrl_cond.n_trials > 0:
            c_xs, c_ys, c_zs = [], [], []
            for i in range(ctrl_cond.n_trials):
                y_tr = ctrl_cond.y[i]
                valid_i = np.isfinite(y_tr)
                if not np.any(valid_i):
                    continue
                k_max = int(np.nanargmax(y_tr))
                y_peak = float(y_tr[k_max])
                delta_y = y_peak - float(y_tr[valid_i][0])
                if delta_y > 20.0 and k_max >= 2:
                    k_lo = max(0, k_max - 1)
                    k_hi = min(ctrl_cond.n_time, k_max + 2)
                    c_xs.append(np.nanmean(ctrl_cond.x[i, k_lo:k_hi]))
                    c_ys.append(y_peak)
                    c_zs.append(np.nanmean(ctrl_cond.z[i, k_lo:k_hi]))
            if c_xs:
                c_end_x = float(np.nanmedian(c_xs))
                c_end_y = float(np.nanmedian(c_ys))
                c_end_z = float(np.nanmedian(c_zs))
                ctrl_end = (c_end_x, c_end_y, c_end_z)

    if ctrl_end is not None:
        chosen_x, chosen_y, chosen_z = ctrl_end
        _log(f"[Target Anchor] Target {tgt_clean}: 1 mm circle centered at unperturbed control median endpoint: [{chosen_x:.1f}, {chosen_y:.1f}, {chosen_z:.1f}] mm.")
    else:
        chosen_x, chosen_y, chosen_z = 0.0, 0.0, 0.0
        _log(f"[Target Anchor] Target {tgt_clean}: Control condition unavailable; defaulting to origin.")

    return {
        "target_name": f"Target {tgt_clean}",
        "target_x": chosen_x,
        "target_y": chosen_y,
        "target_z": chosen_z,
        "target_radius_mm": target_radius_mm,
        "target_diameter_mm": target_diameter_mm,
        "is_stable": True,
        "dispersion_3d_mm": 0.0,
        "source": "Control Reach Median Center (1 mm)",
        "dlc_pos_task": None,
        "ctrl_pos_task": ctrl_end,
        "offset_dlc_to_ctrl_mm": 0.0,
    }


def compute_directional_perturbations(
    conditions: List[ReachCondition],
    target: Optional[str] = None,
    session_landmarks: Optional[Dict[str, np.ndarray]] = None,
    dlc_3d_dir: Optional[Path] = None,
    square_size_mm: float = SQUARE_SIZE_MM,
    target_diameter_mm: float = TARGET_DIAMETER_MM,
    stability_threshold_mm: float = TARGET_STABILITY_THRESHOLD_MM,
) -> Dict[int, Dict[str, Any]]:
    """
    Decompose reach trajectories into directional perturbation components
    relative to matched control reference:
        Delta X(t) = Lateral displacement (mm; left < 0 < right)
        Delta Y(t) = Forward displacement (mm; undershoot < 0 < overshoot)
        Delta Z(t) = Vertical displacement (mm; dip < 0 < lift)
        ||Delta r(t)|| = 3D vector norm (mm)

    Also evaluates:
      - 1 mm Target Hit Rate (% of reach endpoints landing within 1 mm of control anchor)
      - Relative endpoint displacement (% of nominal target diameter)
      - Continuous trajectory mid-reach peak deflection vs final endpoint
    """
    ctrl_candidates = [c for c in conditions if c.is_control]
    ctrl_cond = max(ctrl_candidates, key=lambda c: c.n_trials) if ctrl_candidates else None
    if ctrl_cond is None:
        _log("Warning: No control condition found; cannot compute directional perturbations.")
        return {}

    tgt_str = target or (conditions[0].target if conditions else "A")
    target_info = resolve_target_spatial_anchor(
        target=tgt_str,
        conditions=conditions,
        target_radius_mm=TARGET_RADIUS_MM,
    )

    tgt_x = target_info["target_x"]
    tgt_z = target_info["target_z"]
    tgt_r = target_info["target_radius_mm"]
    tgt_y = target_info.get("target_y")

    mx_c, my_c, mz_c = ctrl_cond.mean_trajectory()
    medx_c, medy_c, medz_c = ctrl_cond.median_trajectory()

    # Extract control endpoints at physical touchscreen contact (peak forward excursion Y)
    ctrl_n_t = ctrl_cond.n_time
    ctrl_abs_end_x = np.full(ctrl_cond.n_trials, np.nan)
    ctrl_abs_end_y = np.full(ctrl_cond.n_trials, np.nan)
    ctrl_abs_end_z = np.full(ctrl_cond.n_trials, np.nan)
    ctrl_t_end = np.full(ctrl_cond.n_trials, np.nan)
    ctrl_completed = np.zeros(ctrl_cond.n_trials, dtype=bool)

    for i in range(ctrl_cond.n_trials):
        y_tr = ctrl_cond.y[i]
        valid_i = np.isfinite(y_tr)
        if not np.any(valid_i):
            continue
        # Touch arrival cannot happen before 75 ms: find peak forward excursion Y at or after 75 ms
        t_valid = (ctrl_cond.time >= 75.0) & np.isfinite(y_tr)
        if np.any(t_valid):
            k_cand = np.where(t_valid)[0]
            k_max = int(k_cand[np.nanargmax(y_tr[k_cand])])
        else:
            k_max = int(np.nanargmax(y_tr))

        k_lo = max(0, k_max - 1)
        k_hi = min(ctrl_n_t, k_max + 2)
        ctrl_abs_end_x[i] = np.nanmean(ctrl_cond.x[i, k_lo:k_hi])
        ctrl_abs_end_y[i] = float(np.nanmax(y_tr[k_lo:k_hi]))
        ctrl_abs_end_z[i] = np.nanmean(ctrl_cond.z[i, k_lo:k_hi])
        t_peak = float(ctrl_cond.time[k_max])
        ctrl_t_end[i] = t_peak
        ctrl_completed[i] = bool(t_peak >= 75.0 and k_max >= 2)

    if not np.any(ctrl_completed):
        ctrl_completed = np.array([bool(np.isfinite(ctrl_t_end[i]) and ctrl_t_end[i] >= 75.0) for i in range(ctrl_cond.n_trials)], dtype=bool)

    ctrl_use = ctrl_completed if np.any(ctrl_completed) else np.isfinite(ctrl_abs_end_x)
    ctrl_mean_end_x = float(np.nanmean(ctrl_abs_end_x[ctrl_use]))
    ctrl_mean_end_y = float(np.nanmean(ctrl_abs_end_y[ctrl_use]))
    ctrl_mean_end_z = float(np.nanmean(ctrl_abs_end_z[ctrl_use]))
    ctrl_median_end_x = float(np.nanmedian(ctrl_abs_end_x[ctrl_use]))
    ctrl_median_end_y = float(np.nanmedian(ctrl_abs_end_y[ctrl_use]))
    ctrl_median_end_z = float(np.nanmedian(ctrl_abs_end_z[ctrl_use]))

    # Control hit rate against the 10 mm target (completed reaches only)
    ctrl_d_to_tgt = np.sqrt((ctrl_abs_end_x[ctrl_use] - tgt_x)**2 + (ctrl_abs_end_z[ctrl_use] - tgt_z)**2)
    ctrl_is_hit = ctrl_d_to_tgt <= tgt_r
    ctrl_hit_count = int(np.sum(ctrl_is_hit))
    ctrl_hit_rate_pct = float(100.0 * ctrl_hit_count / len(ctrl_is_hit)) if len(ctrl_is_hit) > 0 else 0.0

    results: Dict[int, Dict[str, Any]] = {}

    for c in conditions:
        n_trials = c.n_trials
        n_t = c.n_time
        time = c.time

        if c.is_control:
            # Leave-one-out control deviation
            if n_trials > 1:
                dx_trials = np.zeros((n_trials, n_t))
                dy_trials = np.zeros((n_trials, n_t))
                dz_trials = np.zeros((n_trials, n_t))
                for i in range(n_trials):
                    loo_x = (n_trials * mx_c - c.x[i]) / (n_trials - 1)
                    loo_y = (n_trials * my_c - c.y[i]) / (n_trials - 1)
                    loo_z = (n_trials * mz_c - c.z[i]) / (n_trials - 1)
                    dx_trials[i] = c.x[i] - loo_x
                    dy_trials[i] = c.y[i] - loo_y
                    dz_trials[i] = c.z[i] - loo_z
            else:
                dx_trials = np.zeros((1, n_t))
                dy_trials = np.zeros((1, n_t))
                dz_trials = np.zeros((1, n_t))
        else:
            dx_trials = c.x - mx_c[None, :]
            dy_trials = c.y - my_c[None, :]
            dz_trials = c.z - mz_c[None, :]

        dnorm_trials = np.sqrt(dx_trials ** 2 + dy_trials ** 2 + dz_trials ** 2)

        mean_dx = np.nanmean(dx_trials, axis=0)
        sem_dx = np.nanstd(dx_trials, axis=0) / math.sqrt(n_trials) if n_trials > 1 else np.zeros(n_t)

        mean_dy = np.nanmean(dy_trials, axis=0)
        sem_dy = np.nanstd(dy_trials, axis=0) / math.sqrt(n_trials) if n_trials > 1 else np.zeros(n_t)

        mean_dz = np.nanmean(dz_trials, axis=0)
        sem_dz = np.nanstd(dz_trials, axis=0) / math.sqrt(n_trials) if n_trials > 1 else np.zeros(n_t)

        mean_dnorm = np.nanmean(dnorm_trials, axis=0)
        sem_dnorm = np.nanstd(dnorm_trials, axis=0) / math.sqrt(n_trials) if n_trials > 1 else np.zeros(n_t)

        # Extract per-trial physical touchscreen endpoints at true peak forward excursion Y (touchscreen contact)
        abs_end_x = np.full(n_trials, np.nan)
        abs_end_y = np.full(n_trials, np.nan)
        abs_end_z = np.full(n_trials, np.nan)
        trial_t_end = np.full(n_trials, np.nan)
        is_completed_reach = np.zeros(n_trials, dtype=bool)

        tgt_ref_y = tgt_y if (tgt_y is not None and np.isfinite(tgt_y)) else ctrl_median_end_y
        for i in range(n_trials):
            y_tr = c.y[i]
            valid_i = np.isfinite(y_tr)
            if not np.any(valid_i):
                continue
            # Touch arrival cannot happen before 75 ms: find peak forward excursion Y at or after 75 ms
            t_valid = (time >= 75.0) & np.isfinite(y_tr)
            if np.any(t_valid):
                k_cand = np.where(t_valid)[0]
                k_max = int(k_cand[np.nanargmax(y_tr[k_cand])])
            else:
                k_max = int(np.nanargmax(y_tr))
            y_peak = float(y_tr[k_max])

            k_lo = max(0, k_max - 1)
            k_hi = min(n_t, k_max + 2)
            abs_end_x[i] = np.nanmean(c.x[i, k_lo:k_hi])
            abs_end_y[i] = y_peak
            abs_end_z[i] = np.nanmean(c.z[i, k_lo:k_hi])
            t_peak = float(time[k_max])
            trial_t_end[i] = t_peak

            # Reach completion criteria: touch arrival cannot happen before 75 ms
            near_screen = bool(y_peak >= (tgt_ref_y - 15.0))
            not_truncated = bool((k_max < n_t - 1) or near_screen)
            is_completed_reach[i] = bool((t_peak >= 75.0) and not_truncated and (k_max >= 2))

        if not np.any(is_completed_reach):
            # Fallback: enforce t_peak >= 75 ms
            for i in range(n_trials):
                if np.isfinite(trial_t_end[i]) and trial_t_end[i] >= 75.0:
                    is_completed_reach[i] = True

        # Endpoint shifts relative to control reference touch center
        end_dx = abs_end_x - ctrl_median_end_x
        end_dy = abs_end_y - ctrl_median_end_y
        end_dz = abs_end_z - ctrl_median_end_z
        end_norm = np.sqrt(end_dx ** 2 + end_dy ** 2 + end_dz ** 2)

        # Central tendencies (calculated over completed reaches)
        use_mask = is_completed_reach if np.any(is_completed_reach) else np.isfinite(abs_end_x)
        mean_abs_end_x = float(np.nanmean(abs_end_x[use_mask]))
        mean_abs_end_y = float(np.nanmean(abs_end_y[use_mask]))
        mean_abs_end_z = float(np.nanmean(abs_end_z[use_mask]))
        median_abs_end_x = float(np.nanmedian(abs_end_x[use_mask]))
        median_abs_end_y = float(np.nanmedian(abs_end_y[use_mask]))
        median_abs_end_z = float(np.nanmedian(abs_end_z[use_mask]))

        # Target hit rate & distance to target for completed reaches
        d_to_tgt = np.sqrt((abs_end_x - tgt_x)**2 + (abs_end_z - tgt_z)**2)
        valid_hits = use_mask & np.isfinite(d_to_tgt)
        if np.any(valid_hits):
            is_hit = d_to_tgt[valid_hits] <= tgt_r
            hit_count = int(np.sum(is_hit))
            hit_rate_pct = float(100.0 * hit_count / len(is_hit))
        else:
            hit_rate_pct = 0.0

        # Endpoint displacement relative to target diameter
        end_shift_2d = float(np.sqrt((median_abs_end_x - ctrl_median_end_x)**2 + (median_abs_end_z - ctrl_median_end_z)**2))
        rel_disp_pct = float((end_shift_2d / target_info["target_diameter_mm"]) * 100.0)

        # Trajectory mid-reach peak deflection (evaluated strictly during forward reach time >= 0 ms)
        reach_mask = (time >= 0.0) & (time <= 350.0) & np.isfinite(mean_dnorm)
        if not np.any(reach_mask):
            reach_mask = (time >= 0.0) & np.isfinite(mean_dnorm)
        if np.any(reach_mask):
            sub_dnorm = mean_dnorm[reach_mask]
            sub_time = time[reach_mask]
            peak_mid_deflection = float(np.nanmax(sub_dnorm))
            peak_mid_t_ms = float(sub_time[np.nanargmax(sub_dnorm)])
        else:
            peak_mid_deflection = 0.0
            peak_mid_t_ms = 0.0
        rel_peak_pct = float((peak_mid_deflection / target_info["target_diameter_mm"]) * 100.0)

        mean_x, mean_y, mean_z = c.mean_trajectory()
        median_x, median_y, median_z = c.median_trajectory()

        # Statistical comparisons against unperturbed control endpoints (2-sample tests)
        s_x_val = abs_end_x[use_mask]
        s_y_val = abs_end_y[use_mask]
        s_z_val = abs_end_z[use_mask]
        c_x_val = ctrl_abs_end_x[ctrl_use]
        c_y_val = ctrl_abs_end_y[ctrl_use]
        c_z_val = ctrl_abs_end_z[ctrl_use]

        if not c.is_control and len(s_x_val) >= 2 and len(c_x_val) >= 2:
            p_dx = float(stats.ttest_ind(s_x_val, c_x_val, equal_var=False).pvalue)
            p_dy = float(stats.ttest_ind(s_y_val, c_y_val, equal_var=False).pvalue)
            p_dz = float(stats.ttest_ind(s_z_val, c_z_val, equal_var=False).pvalue)

            # 2D Touchscreen Face Displacement Permutation Test vs Control
            d_obs = math.sqrt((median_abs_end_x - ctrl_median_end_x)**2 + (median_abs_end_z - ctrl_median_end_z)**2)
            pool_x = np.concatenate([s_x_val, c_x_val])
            pool_z = np.concatenate([s_z_val, c_z_val])
            n_s = len(s_x_val)
            rng_disp = np.random.default_rng(42)
            n_perms = 2000
            perm_dists = np.empty(n_perms)
            for p_idx in range(n_perms):
                shuffled = rng_disp.permutation(len(pool_x))
                shuf_sx = pool_x[shuffled[:n_s]]
                shuf_sz = pool_z[shuffled[:n_s]]
                shuf_cx = pool_x[shuffled[n_s:]]
                shuf_cz = pool_z[shuffled[n_s:]]
                perm_dists[p_idx] = math.sqrt(
                    (np.median(shuf_sx) - np.median(shuf_cx))**2 +
                    (np.median(shuf_sz) - np.median(shuf_cz))**2
                )
            p_norm = float((1.0 + np.sum(perm_dists >= d_obs)) / (n_perms + 1.0))
        else:
            p_dx, p_dy, p_dz, p_norm = 1.0, 1.0, 1.0, 1.0

        results[c.cond_num] = {
            "cond_num": c.cond_num,
            "label": c.label,
            "base_label": getattr(c, "base_label", c.label),
            "target": c.target,
            "is_control": c.is_control,
            "time": time,
            "mean_dx": mean_dx, "sem_dx": sem_dx,
            "mean_dy": mean_dy, "sem_dy": sem_dy,
            "mean_dz": mean_dz, "sem_dz": sem_dz,
            "mean_dnorm": mean_dnorm, "sem_dnorm": sem_dnorm,
            "end_dx": end_dx, "end_dy": end_dy, "end_dz": end_dz, "end_norm": end_norm,
            "abs_end_x": abs_end_x, "abs_end_y": abs_end_y, "abs_end_z": abs_end_z,
            "mean_abs_end_x": mean_abs_end_x,
            "mean_abs_end_y": mean_abs_end_y,
            "mean_abs_end_z": mean_abs_end_z,
            "median_abs_end_x": median_abs_end_x,
            "median_abs_end_y": median_abs_end_y,
            "median_abs_end_z": median_abs_end_z,
            "mean_abs_end": (mean_abs_end_x, mean_abs_end_y, mean_abs_end_z),
            "median_abs_end": (median_abs_end_x, median_abs_end_y, median_abs_end_z),
            "ctrl_abs_end_x": ctrl_abs_end_x,
            "ctrl_abs_end_y": ctrl_abs_end_y,
            "ctrl_abs_end_z": ctrl_abs_end_z,
            "ctrl_mean_end": (ctrl_mean_end_x, ctrl_mean_end_y, ctrl_mean_end_z),
            "ctrl_median_end": (ctrl_median_end_x, ctrl_median_end_y, ctrl_median_end_z),
            "ctrl_mean_traj": (mx_c, my_c, mz_c),
            "ctrl_median_traj": (medx_c, medy_c, medz_c),
            "mean_traj": (mean_x, mean_y, mean_z),
            "median_traj": (median_x, median_y, median_z),
            "trials_x": c.x,
            "trials_y": c.y,
            "trials_z": c.z,
            "ctrl_trials_x": ctrl_cond.x,
            "ctrl_trials_y": ctrl_cond.y,
            "ctrl_trials_z": ctrl_cond.z,
            "target_info": target_info,
            "hit_rate_pct": hit_rate_pct,
            "hit_count": hit_count,
            "is_hit": is_hit,
            "d_to_tgt": d_to_tgt,
            "ctrl_hit_rate_pct": ctrl_hit_rate_pct,
            "ctrl_hit_count": ctrl_hit_count,
            "ctrl_is_hit": ctrl_is_hit,
            "ctrl_d_to_tgt": ctrl_d_to_tgt,
            "rel_disp_pct": rel_disp_pct,
            "peak_mid_deflection": peak_mid_deflection,
            "peak_mid_t_ms": peak_mid_t_ms,
            "rel_peak_pct": rel_peak_pct,
            "p_dx": p_dx, "p_dy": p_dy, "p_dz": p_dz,
            "p_norm": p_norm, "q_norm": p_norm,
            "stim_dur_ms": c.stim_dur_ms,
            "n_trials": n_trials,
            "is_completed_reach": is_completed_reach,
            "trial_t_end": trial_t_end,
            "median_t_end": float(np.nanmedian(trial_t_end[use_mask])) if np.any(use_mask) else 0.0,
            "n_completed": int(np.sum(is_completed_reach)),
            "n_incomplete": int(np.sum(~is_completed_reach)),
        }

    # Benjamini-Hochberg FDR correction across stimulation conditions for endpoint perturbation
    stim_cnums = [c.cond_num for c in conditions if not c.is_control and c.cond_num in results]
    if stim_cnums:
        raw_p_norms = [results[cn]["p_norm"] for cn in stim_cnums]
        q_norms = fdr_adjust_1d(raw_p_norms, method="bh")
        for cn, q_val in zip(stim_cnums, q_norms):
            results[cn]["q_norm"] = float(q_val)
    for c in conditions:
        if c.is_control and c.cond_num in results:
            results[c.cond_num]["q_norm"] = 1.0

    return results


# =============================================================================
# Visualizations & Plotting Functions
# =============================================================================

def _safe_color_hex(col: Any) -> str:
    """Safely convert any color representation (string, tuple, numpy array) to a hex string."""
    try:
        return to_hex(col)
    except Exception:
        if isinstance(col, (np.ndarray, list, tuple)):
            return str(tuple(col))
        return str(col)


def draw_staggered_stim_block(
    ax: plt.Axes,
    stim_conditions: List[Dict[str, Any]],
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    ladder_height_ratio: Optional[float] = None,
    show_ladder: bool = True,
    show_stepped_shading: bool = True,
    show_drop_lines: bool = True,
) -> None:
    """
    Draw a condition-dependent 'staggered' stimulation window block:
      1. Stepped background shading: Cumulative soft shading stepping down as each condition ends.
      2. Condition-specific drop-lines: Dotted vertical lines at each condition's exact stimulus termination.
      3. Top-margin duration ladder: Staggered horizontal bars colored by condition with ms duration callouts.
    """
    valid_stims = [
        c for c in stim_conditions
        if not c.get("is_control", False) and (c.get("stim_dur_ms") or 0.0) > 0
    ]
    if not valid_stims:
        return

    # Sort conditions by duration (shortest to longest)
    sorted_stims = sorted(valid_stims, key=lambda c: (float(c.get("stim_dur_ms", 0.0)), int(c.get("cond_num", 0))))
    durations = [float(c["stim_dur_ms"]) for c in sorted_stims]
    unique_durs = sorted(list(set(durations)))

    # 1. Stepped background shading
    if show_stepped_shading:
        prev_d = 0.0
        for d in unique_durs:
            n_active = sum(1 for c in sorted_stims if float(c.get("stim_dur_ms", 0.0)) >= d)
            alpha_step = max(0.02, min(0.18, 0.025 * n_active))
            ax.axvspan(prev_d, d, color="#f9e79f", alpha=alpha_step, zorder=0)
            prev_d = d

    # 2. Condition-specific drop-lines at stimulus termination (deduplicated by duration)
    if show_drop_lines:
        for dur in unique_durs:
            matching_c = [c for c in sorted_stims if abs(float(c.get("stim_dur_ms", 0.0)) - dur) < 1e-3]
            unique_colors = []
            seen_hex = set()
            for c in matching_c:
                raw_col = c.get("color", "#e74c3c")
                hex_col = _safe_color_hex(raw_col)
                if hex_col not in seen_hex:
                    seen_hex.add(hex_col)
                    unique_colors.append(raw_col)
            color = unique_colors[0] if len(unique_colors) == 1 else "#7f8c8d"
            ax.axvline(dur, color=color, linestyle=":", linewidth=1.1, alpha=0.55, zorder=1)

    # 3. Top-margin duration ladder (deduplicated: exactly one bar per unique stim duration)
    if show_ladder:
        cur_ylim = ax.get_ylim()
        ymin = y_min if y_min is not None else cur_ylim[0]
        ymax = y_max if y_max is not None else cur_ylim[1]
        yrange = ymax - ymin
        if yrange <= 0:
            yrange = 1.0

        n_bars = len(unique_durs)
        ladder_ratio = max(0.18, min(0.35, 0.05 * n_bars)) if ladder_height_ratio is None else ladder_height_ratio
        bar_height_total = yrange * ladder_ratio
        bar_step = bar_height_total / max(1, n_bars)
        y_top = ymax - (yrange * 0.035)

        for i, dur in enumerate(reversed(unique_durs)):
            matching_c = [c for c in sorted_stims if abs(float(c.get("stim_dur_ms", 0.0)) - dur) < 1e-3]
            unique_colors = []
            seen_hex = set()
            for c in matching_c:
                raw_col = c.get("color", "#e74c3c")
                hex_col = _safe_color_hex(raw_col)
                if hex_col not in seen_hex:
                    seen_hex.add(hex_col)
                    unique_colors.append(raw_col)
            color = unique_colors[0] if len(unique_colors) == 1 else "#4a6572"
            y_bar = y_top - (i * bar_step)
            # Horizontal bar
            ax.plot([0.0, dur], [y_bar, y_bar], color=color, linewidth=2.4, solid_capstyle="butt", zorder=7)
            # End tick
            tick_dy = bar_step * 0.35
            ax.plot([dur, dur], [y_bar - tick_dy, y_bar + tick_dy], color=color, linewidth=1.8, zorder=7)
            # Duration callout text
            dur_str = f"{dur:.1f}ms" if dur < 10 and dur % 1 != 0 else f"{int(dur)}ms"
            ax.text(dur + 2.0, y_bar, dur_str, color=color, fontsize=7.0, va="center", ha="left", fontweight="bold", zorder=8)


def draw_confidence_ellipse(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    n_scale: float = 1.0,
    scale_mode: str = "2se",
    edgecolor: str = "black",
    facecolor: str = "none",
    center: Optional[Tuple[float, float]] = None,
    n_std: Optional[float] = None,
    **kwargs,
) -> Optional[Ellipse]:
    """Draw a covariance confidence ellipse representing 2D trajectory dispersion (1/2 SD) or standard error of the mean (1/2 SE)."""
    valid = np.isfinite(x) & np.isfinite(y)
    n_valid = int(np.sum(valid))
    if n_valid < 3:
        return None
    x_val = x[valid]
    y_val = y[valid]
    cov = np.cov(x_val, y_val)
    if np.all(np.isfinite(cov)) and cov.ndim == 2:
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        
        mode_str = str(scale_mode).lower().strip()
        if mode_str in ("2se", "2_se"):
            default_mult = 2.0
            div = np.sqrt(n_valid)
        elif mode_str in ("1se", "se", "sem"):
            default_mult = 1.0
            div = np.sqrt(n_valid)
        elif mode_str in ("2sd", "2_sd"):
            default_mult = 2.0
            div = 1.0
        elif mode_str in ("1sd", "sd", "std"):
            default_mult = 1.0
            div = 1.0
        else:
            default_mult = 2.0 if "2" in mode_str else 1.0
            div = np.sqrt(n_valid) if "se" in mode_str else 1.0

        if n_std is not None:
            scale_mult = float(n_std)
        elif n_scale != 1.0:
            scale_mult = float(n_scale)
        else:
            scale_mult = default_mult

        width = 2.0 * scale_mult * np.sqrt(np.maximum(vals[0], 1e-6)) / div
        height = 2.0 * scale_mult * np.sqrt(np.maximum(vals[1], 1e-6)) / div
        c_x, c_y = center if center is not None else (float(np.mean(x_val)), float(np.mean(y_val)))
        ellipse = Ellipse(
            xy=(c_x, c_y),
            width=width,
            height=height,
            angle=theta,
            edgecolor=edgecolor,
            facecolor=facecolor,
            **kwargs,
        )
        ax.add_patch(ellipse)
        return ellipse
    return None


def get_rdm_colormap() -> LinearSegmentedColormap:
    """Sequential colormap for RDM: Pure White (0 mm) -> Warm Gold -> Coral Red -> Dark Mahogany."""
    cmap = LinearSegmentedColormap.from_list(
        "rdm_white_gold_crimson",
        [
            (0.00, "#ffffff"),  # 0 mm: Pure White (diagonal / identical reaches)
            (0.08, "#fffde7"),  # Pale cream
            (0.20, "#fee391"),  # Light warm gold (10-14 mm)
            (0.36, "#fec44f"),  # Golden amber (15-18 mm)
            (0.52, "#fe9929"),  # Rich orange (19-22 mm)
            (0.68, "#e31a1c"),  # Bright coral red (23-26 mm)
            (0.84, "#bd0026"),  # Deep ruby crimson (27-30 mm)
            (1.00, "#4a0008"),  # Dark mahogany (> 30 mm)
        ],
        N=256,
    )
    cmap.set_bad("#ffffff")
    return cmap


def to_short_condition_label(lbl: str, n_tri: Optional[int] = None) -> str:
    """
    Shorten condition label for matrix tick display respecting SHOW_CONDITION_INDICES
    and SHOW_TRIAL_COUNTS.
    """
    m = re.match(r"(BR\s*\d+)", lbl, re.IGNORECASE)
    prefix = m.group(1) if m else ""

    # Extract trial count if not provided
    if n_tri is None:
        m_n = re.search(r"\(n=(\d+)\)", lbl)
        if m_n:
            n_tri = int(m_n.group(1))
    n_str = f"n={n_tri}" if (SHOW_TRIAL_COUNTS and n_tri is not None) else ""

    if "control" in lbl.lower() or "ctrl" in lbl.lower():
        parts = ["Ctrl"]
        if n_str:
            parts.append(n_str)
        if SHOW_CONDITION_INDICES and prefix:
            return f"{prefix} ({', '.join(parts)})"
        return ", ".join(parts)

    parts = []

    # 1. Continuous / Pulse count
    is_cont = any(kw in lbl.lower() for kw in ("cont", "continuous", "2hz-cont"))

    # Check frequency first
    f_match = re.search(r"(\d+(?:\.\d+)?)\s*Hz(?:-cont)?", lbl, re.IGNORECASE)
    freq_str = None
    if f_match:
        val_f = float(f_match.group(1))
        if is_cont or val_f <= 2.5:
            freq_str = f"{int(val_f) if val_f.is_integer() else val_f}Hz-cont"
            is_cont = True
        else:
            freq_str = f"{int(val_f) if val_f.is_integer() else val_f}Hz"
    elif is_cont:
        freq_str = "2Hz-cont"

    if is_cont:
        parts.append(freq_str if freq_str else "2Hz-cont")
    else:
        p_match = re.search(r"(\d+)\s*pulse", lbl, re.IGNORECASE)
        if not p_match:
            p_match = re.search(r"\b(\d+)p\b", lbl, re.IGNORECASE)
        if p_match:
            parts.append(f"{p_match.group(1)}p")
        if freq_str:
            parts.append(freq_str)

    # 2. Channels (e.g. ch 1-8, ch 1-32, ch 16-24)
    ch_match = re.search(r"(?:ch(?:annels?)?\.?\s*[:=]?\s*)([0-9]+(?:\s*[-–]\s*[0-9]+)?)", lbl, re.IGNORECASE)
    if ch_match:
        ch_clean = ch_match.group(1).replace(" ", "").replace("–", "-")
        parts.append(f"ch {ch_clean}")
    else:
        m_dash = re.search(r",\s*([0-9]+[-–][0-9]+)\b", lbl)
        if m_dash:
            ch_clean = m_dash.group(1).replace(" ", "").replace("–", "-")
            parts.append(f"ch {ch_clean}")

    if n_str:
        parts.append(n_str)

    if not parts:
        clean_lbl = re.sub(r"^BR\s*\d+:\s*", "", lbl, flags=re.IGNORECASE).strip()
        if not SHOW_TRIAL_COUNTS:
            clean_lbl = re.sub(r"\s*\(n=\d+\)", "", clean_lbl).strip()
        if SHOW_CONDITION_INDICES and prefix:
            return f"{prefix} ({clean_lbl})" if clean_lbl else prefix
        return clean_lbl

    if SHOW_CONDITION_INDICES and prefix:
        return f"{prefix} ({', '.join(parts)})"
    return ", ".join(parts)


def plot_dual_window_rsa_comparison_figure(
    RDM_early: np.ndarray,
    S_early: np.ndarray,
    Q_early: np.ndarray,
    within_stereotypy_early: np.ndarray,
    RDM_late: np.ndarray,
    S_late: np.ndarray,
    Q_late: np.ndarray,
    within_stereotypy_late: np.ndarray,
    block_sizes: List[int],
    block_labels: List[str],
    target: str = "A",
    window_early: Tuple[float, float] = RSA_EARLY_WINDOW_MS,
    window_late: Tuple[float, float] = RSA_LATE_WINDOW_MS,
    is_progress_mode: bool = False,
    early_tag: Optional[str] = None,
    late_tag: Optional[str] = None,
    metric_label: str = "3D Distance (mm)",
    block_short_labels: Optional[List[str]] = None,
    fixed_params_summary: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Consolidated 4-panel Dual-Window Behavioral RSA Figure:
        Top Row: Early Epoch (Kinematic RDM & Condition Separation)
        Bottom Row: Late Epoch (Kinematic RDM & Condition Separation)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.5), dpi=FIG_DPI)
    ax_a = axes[0, 0]
    ax_b = axes[0, 1]
    ax_c = axes[1, 0]
    ax_d = axes[1, 1]

    # Shared RDM scaling for direct visual comparability across epochs
    vmax_early = np.nanpercentile(RDM_early[RDM_early > 0], 98) if np.any(RDM_early > 0) else 40.0
    vmax_late = np.nanpercentile(RDM_late[RDM_late > 0], 98) if np.any(RDM_late > 0) else 40.0
    vmax_rdm = max(vmax_early, vmax_late, 30.0)

    cmap_rdm = get_rdm_colormap()

    cum = np.cumsum(block_sizes)
    starts = np.r_[0, cum[:-1]]
    boundaries = cum[:-1] - 0.5
    centers = starts + (np.asarray(block_sizes) / 2.0) - 0.5
    short_labels = block_short_labels if block_short_labels is not None else [to_short_condition_label(l, sz) for l, sz in zip(block_labels, block_sizes)]

    if early_tag is None:
        early_tag = f"{int(round(RSA_EARLY_PROGRESS_RANGE[0] * 100))}–{int(round(RSA_EARLY_PROGRESS_RANGE[1] * 100))}% Depth" if is_progress_mode else f"{int(window_early[0])}–{int(window_early[1])} ms"
    if late_tag is None:
        late_tag = f"{int(round(RSA_LATE_PROGRESS_RANGE[0] * 100))}–{int(round(RSA_LATE_PROGRESS_RANGE[1] * 100))}% Depth" if is_progress_mode else f"{int(window_late[0])}–{int(window_late[1])} ms"
    mode_prefix = "Progress " if is_progress_mode else ""

    # -------------------------------------------------------------
    # Panel A: Early RDM
    # -------------------------------------------------------------
    im_a = ax_a.imshow(
        RDM_early, aspect="equal", origin="upper", interpolation="nearest",
        cmap=cmap_rdm, vmin=0.0, vmax=vmax_rdm,
    )
    ax_a.set_title(
        f"A. Early {mode_prefix}Kinematic RDM ({early_tag})",
        fontsize=10.5, fontweight="bold", loc="left",
    )
    ax_a.set_xticks(boundaries, minor=True)
    ax_a.set_yticks(boundaries, minor=True)
    ax_a.grid(which="minor", color="#2c3e50", linestyle="-", linewidth=1.0, alpha=0.6)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for c, lab in zip(centers, short_labels):
        ax_a.text(c, -0.02, lab, transform=ax_a.get_xaxis_transform(),
                  ha="right", va="top", rotation=35, fontsize=7.5)
        ax_a.text(-0.02, c, lab, transform=ax_a.get_yaxis_transform(),
                  ha="right", va="center", fontsize=7.5)
    cb_a = fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cb_a.set_label(f"Dissimilarity: {metric_label}", fontsize=8.5)

    # -------------------------------------------------------------
    # Panel B: Early Silhouette Matrix
    # -------------------------------------------------------------
    K = len(block_labels)
    finite_s_early = S_early[np.isfinite(S_early)]
    finite_s_late = S_late[np.isfinite(S_late)]
    max_s_early = float(np.max(np.abs(finite_s_early))) if len(finite_s_early) > 0 else 0.3
    max_s_late = float(np.max(np.abs(finite_s_late))) if len(finite_s_late) > 0 else 0.3
    vmax_s = max(max_s_early, max_s_late, 0.25)

    cmap_b = plt.colormaps["RdBu_r"].copy()
    cmap_b.set_bad("#eaecee")

    im_b = ax_b.imshow(
        np.ma.masked_invalid(S_early), cmap=cmap_b, vmin=-vmax_s, vmax=vmax_s,
        origin="upper", interpolation="nearest", aspect="equal",
    )
    ax_b.set_title(
        f"B. Early {mode_prefix}Condition Separation ({early_tag})",
        fontsize=10.5, fontweight="bold", loc="left",
    )
    ax_b.set_xticks(range(K))
    ax_b.set_yticks(range(K))
    ax_b.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=7.5)
    ax_b.set_yticklabels(short_labels, fontsize=7.5)
    ax_b.set_xticks(np.arange(K + 1) - 0.5, minor=True)
    ax_b.set_yticks(np.arange(K + 1) - 0.5, minor=True)
    ax_b.grid(which="minor", color="white", linewidth=1.0)
    ax_b.tick_params(which="minor", length=0)

    for i in range(K):
        for j in range(K):
            if i == j:
                score_str = f"[{within_stereotypy_early[i]:.1f}]"
                ax_b.text(j, i, score_str, ha="center", va="center", fontsize=7.0, color="#1b2631", fontweight="bold")
            elif np.isfinite(S_early[i, j]):
                val = S_early[i, j]
                q_val = Q_early[i, j]
                star = p_stars_label(q_val)
                color = "white" if abs(val) > 0.55 * vmax_s else "black"
                val_str = f"{val:+.2f}"
                if star and star != "n.s.":
                    num_font = max(5.8, min(7.5, 8.5 - 0.15 * K))
                    star_font = max(10.0, min(13.5, 15.5 - 0.25 * K))
                    ax_b.text(j, i - 0.13, val_str, ha="center", va="center", fontsize=num_font, color=color)
                    ax_b.text(j, i + 0.16, star, ha="center", va="center", fontsize=star_font, fontweight="bold", color=color)
                else:
                    num_font = max(6.0, min(8.0, 9.0 - 0.15 * K))
                    ax_b.text(j, i, val_str, ha="center", va="center", fontsize=num_font, color=color)

    cb_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cb_b.set_label("Pairwise Silhouette S", fontsize=8.5)

    # -------------------------------------------------------------
    # Panel C: Late Kinematic RDM
    # -------------------------------------------------------------
    im_c = ax_c.imshow(
        RDM_late, aspect="equal", origin="upper", interpolation="nearest",
        cmap=cmap_rdm, vmin=0.0, vmax=vmax_rdm,
    )
    ax_c.set_title(
        f"C. Late {mode_prefix}Kinematic RDM ({late_tag})",
        fontsize=10.5, fontweight="bold", loc="left",
    )
    ax_c.set_xticks(boundaries, minor=True)
    ax_c.set_yticks(boundaries, minor=True)
    ax_c.grid(which="minor", color="#2c3e50", linestyle="-", linewidth=1.0, alpha=0.6)
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    for c, lab in zip(centers, short_labels):
        ax_c.text(c, -0.02, lab, transform=ax_c.get_xaxis_transform(),
                  ha="right", va="top", rotation=35, fontsize=7.5)
        ax_c.text(-0.02, c, lab, transform=ax_c.get_yaxis_transform(),
                  ha="right", va="center", fontsize=7.5)
    cb_c = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
    cb_c.set_label(f"Dissimilarity: {metric_label}", fontsize=8.5)

    # -------------------------------------------------------------
    # Panel D: Late Condition Separation
    # -------------------------------------------------------------
    im_d = ax_d.imshow(
        np.ma.masked_invalid(S_late), cmap=cmap_b, vmin=-vmax_s, vmax=vmax_s,
        origin="upper", interpolation="nearest", aspect="equal",
    )
    ax_d.set_title(
        f"D. Late {mode_prefix}Condition Separation ({late_tag})",
        fontsize=10.5, fontweight="bold", loc="left",
    )
    ax_d.set_xticks(range(K))
    ax_d.set_yticks(range(K))
    ax_d.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=7.5)
    ax_d.set_yticklabels(short_labels, fontsize=7.5)
    ax_d.set_xticks(np.arange(K + 1) - 0.5, minor=True)
    ax_d.set_yticks(np.arange(K + 1) - 0.5, minor=True)
    ax_d.grid(which="minor", color="white", linewidth=1.0)
    ax_d.tick_params(which="minor", length=0)

    for i in range(K):
        for j in range(K):
            if i == j:
                score_str = f"[{within_stereotypy_late[i]:.1f}]"
                ax_d.text(j, i, score_str, ha="center", va="center", fontsize=7.0, color="#1b2631", fontweight="bold")
            elif np.isfinite(S_late[i, j]):
                val = S_late[i, j]
                q_val = Q_late[i, j]
                star = p_stars_label(q_val)
                color = "white" if abs(val) > 0.55 * vmax_s else "black"
                val_str = f"{val:+.2f}"
                if star and star != "n.s.":
                    num_font = max(5.8, min(7.5, 8.5 - 0.15 * K))
                    star_font = max(10.0, min(13.5, 15.5 - 0.25 * K))
                    ax_d.text(j, i - 0.13, val_str, ha="center", va="center", fontsize=num_font, color=color)
                    ax_d.text(j, i + 0.16, star, ha="center", va="center", fontsize=star_font, fontweight="bold", color=color)
                else:
                    num_font = max(6.0, min(8.0, 9.0 - 0.15 * K))
                    ax_d.text(j, i, val_str, ha="center", va="center", fontsize=num_font, color=color)

    cb_d = fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
    cb_d.set_label("Pairwise Silhouette S", fontsize=8.5)

    # Full condition details legend line placed outside the plots at bottom of figure
    if len(block_labels) > 8:
        chunk_size = 4 if len(block_labels) > 10 else int(np.ceil(len(block_labels) / 2))
        lines = []
        for i in range(0, len(block_labels), chunk_size):
            lines.append("   |   ".join(block_labels[i:i + chunk_size]))
        full_legend_text = "\n".join(lines)
        fig.text(0.5, 0.045, full_legend_text, ha="center", va="top", fontsize=6.8, color="#2c3e50", fontweight="medium")
        fig.tight_layout(rect=[0, 0.09, 1, 0.93])
    elif len(block_labels) > 4:
        mid = int(np.ceil(len(block_labels) / 2))
        line1 = "   |   ".join(block_labels[:mid])
        line2 = "   |   ".join(block_labels[mid:])
        full_legend_text = f"{line1}\n{line2}"
        fig.text(0.5, 0.045, full_legend_text, ha="center", va="top", fontsize=7.2, color="#2c3e50", fontweight="medium")
        fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    else:
        full_legend_text = "   |   ".join(block_labels)
        fig.text(0.5, 0.04, full_legend_text, ha="center", va="top", fontsize=7.5, color="#2c3e50", fontweight="medium")
        fig.tight_layout(rect=[0, 0.06, 1, 0.93])

    epoch_type = "Forward-Y Progress Epoch" if is_progress_mode else "Temporal Epoch"
    suptitle_text = f"Behavioral RSA {epoch_type} Comparison: Early ({early_tag}) vs. Late ({late_tag}) — Target {target}"
    if fixed_params_summary:
        suptitle_text += f"\n[{fixed_params_summary}]"
    fig.suptitle(
        suptitle_text,
        fontsize=12.5, fontweight="bold", y=0.985,
    )

    fig.text(
        0.98, 0.01,
        "Off-diagonal: Pairwise Silhouette S (FDR * q<0.05, ** q<0.01, *** q<0.001). Diagonal: [Within-condition mean pairwise distance mm].",
        ha="right", va="bottom", fontsize=7.2, color="#5d6d7e", style="italic",
    )

    if save_path:
        out_file = save_path.with_suffix(f".{FIG_FORMAT.lower()}")
        fig.savefig(out_file, format=FIG_FORMAT.lower(), dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Dual-Window Behavioral RSA comparison figure: {out_file.name}")
        plt.close(fig)
    return fig


def plot_time_resolved_variability_figure(
    var_results: Dict[int, Dict[str, Any]],
    target: str = "A",
    fixed_params_summary: str = "",
    save_path: Optional[Path] = None,
    mode: Optional[str] = None,
    legend_orientation: Optional[str] = None,
) -> plt.Figure:
    """
    Consolidated Trajectory Variability & Control Deflection Figure:
        Panel A: Within-Condition 3D Trajectory Variability V(t) in mm.
        Panel B: Net 3D Trajectory Deflection from Control Reference D_to_ctrl(t) in mm.

    Options:
        mode: "both" (default, 2 panels), "variability" (Panel A only), or "deflection" (Panel B only).
        legend_orientation: "column" (default, vertical list on the right), or "row" (horizontal banner across top).
    """
    if mode is None:
        mode = VARIABILITY_PLOT_MODE
    mode_clean = str(mode).lower().strip()
    if mode_clean in ("var", "variability"):
        plot_var = True
        plot_def = False
    elif mode_clean in ("def", "deflection", "dev", "deviation"):
        plot_var = False
        plot_def = True
    else:  # "both"
        plot_var = True
        plot_def = True

    if legend_orientation is None:
        legend_orientation = VARIABILITY_LEGEND_ORIENTATION
    legend_orientation = str(legend_orientation).lower().strip()

    # Figure & subplot geometry
    if plot_var and plot_def:
        fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(11.5, 7.8), sharex=True, dpi=FIG_DPI)
    elif plot_var:
        fig, ax_a = plt.subplots(1, 1, figsize=(11.5, 5.0), dpi=FIG_DPI)
        ax_b = None
    else:
        fig, ax_b = plt.subplots(1, 1, figsize=(11.5, 5.0), dpi=FIG_DPI)
        ax_a = None

    # Assign distinct colors
    cond_keys = sorted(var_results.keys(), key=lambda k: (0 if var_results[k]["is_control"] else 1, k))
    n_stim = sum(not var_results[k]["is_control"] for k in cond_keys)
    palette = plt.cm.tab10(np.linspace(0.0, 1.0, max(10, n_stim + 1)))

    # Build condition colors and metadata for staggered stim window block
    cond_colors = {}
    stim_info = []
    stim_idx = 0
    for k in cond_keys:
        d = var_results[k]
        if d["is_control"]:
            cond_colors[k] = COLOR_CTRL
        else:
            col = palette[stim_idx % len(palette)]
            cond_colors[k] = col
            stim_idx += 1
            stim_info.append({
                "cond_num": k,
                "label": d["label"],
                "color": col,
                "stim_dur_ms": d.get("stim_dur_ms", 0.0) or 0.0,
                "is_control": False,
            })

    for k in cond_keys:
        d = var_results[k]
        t = d["time"]
        v_med = d["var_median"]
        v_sem = d["var_sem"]
        div_m = d["divergence_mean"]
        div_sem = d["divergence_sem"]
        is_ctrl = d["is_control"]
        label = d["label"]
        color = cond_colors[k]

        if is_ctrl:
            ls = "--"
            lw = 2.2
            zorder = 5
        else:
            ls = "-"
            lw = 1.8
            zorder = 3

        # Variability plot (ax_a)
        if ax_a is not None:
            ax_a.plot(t, v_med, color=color, linestyle=ls, linewidth=lw, label=label, zorder=zorder)
            ax_a.fill_between(t, v_med - v_sem, v_med + v_sem, color=color, alpha=0.15, zorder=zorder - 1)

        # Deflection / Divergence from Control plot (ax_b)
        # Note: Control is omitted from deflection panel because deflection of control from control reference is 0.
        if ax_b is not None and not is_ctrl:
            ax_b.plot(t, div_m, color=color, linestyle="-", linewidth=lw, label=label, zorder=zorder)
            ax_b.fill_between(t, div_m - div_sem, div_m + div_sem, color=color, alpha=0.15, zorder=zorder - 1)

    # Style axes: Despine, zero line, and remove grid completely
    for ax in [ax_a, ax_b]:
        if ax is None:
            continue
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.2, zorder=1)
        ax.grid(False)

    # Stim duration ladders & shading:
    # If both are plotted: ladder bar ONLY on top plot (ax_a), still shade and draw drop lines on both.
    # If only one is plotted: ladder on the active plot.
    n_durs = len(set(float(c.get("stim_dur_ms", 0.0)) for c in stim_info if (c.get("stim_dur_ms") or 0.0) > 0))
    ladder_headroom = max(0.28, 0.055 * n_durs + 0.12)
    ladder_ratio = max(0.18, min(0.35, 0.05 * n_durs))

    if plot_var and plot_def:
        # Top plot: ax_a (Variability)
        c_ymin_a, c_ymax_a = ax_a.get_ylim()
        n_ymax_a = c_ymax_a + (c_ymax_a - c_ymin_a) * ladder_headroom
        ax_a.set_ylim(c_ymin_a, n_ymax_a)
        draw_staggered_stim_block(
            ax_a, stim_info, y_min=c_ymin_a, y_max=n_ymax_a,
            ladder_height_ratio=ladder_ratio,
            show_ladder=True, show_stepped_shading=True, show_drop_lines=True,
        )

        # Bottom plot: ax_b (Deflection) — no ladder bar, but stepped shading and drop lines preserved
        c_ymin_b, c_ymax_b = ax_b.get_ylim()
        n_ymax_b = c_ymax_b + (c_ymax_b - c_ymin_b) * 0.06
        ax_b.set_ylim(c_ymin_b, n_ymax_b)
        draw_staggered_stim_block(
            ax_b, stim_info, y_min=c_ymin_b, y_max=n_ymax_b,
            show_ladder=False, show_stepped_shading=True, show_drop_lines=True,
        )
    elif plot_var:
        c_ymin_a, c_ymax_a = ax_a.get_ylim()
        n_ymax_a = c_ymax_a + (c_ymax_a - c_ymin_a) * ladder_headroom
        ax_a.set_ylim(c_ymin_a, n_ymax_a)
        draw_staggered_stim_block(
            ax_a, stim_info, y_min=c_ymin_a, y_max=n_ymax_a,
            ladder_height_ratio=ladder_ratio,
            show_ladder=True, show_stepped_shading=True, show_drop_lines=True,
        )
    else:  # plot_def only
        c_ymin_b, c_ymax_b = ax_b.get_ylim()
        n_ymax_b = c_ymax_b + (c_ymax_b - c_ymin_b) * ladder_headroom
        ax_b.set_ylim(c_ymin_b, n_ymax_b)
        draw_staggered_stim_block(
            ax_b, stim_info, y_min=c_ymin_b, y_max=n_ymax_b,
            ladder_height_ratio=ladder_ratio,
            show_ladder=True, show_stepped_shading=True, show_drop_lines=True,
        )

    # Titles and axis labels
    # Note: "Across Time" removed from variability title; X-axis says "Time (ms)" only.
    if ax_a is not None:
        ax_a.set_ylabel("3D Variability (mm)", fontsize=9.5, fontweight="medium")
        if plot_def:
            ax_a.set_title("A. Within-Condition Trajectory Variability", fontsize=10.5, fontweight="bold", loc="left")
        else:
            ax_a.set_title("Within-Condition Trajectory Variability", fontsize=10.5, fontweight="bold", loc="left")
            ax_a.set_xlabel("Time (ms)", fontsize=10, fontweight="medium")

    if ax_b is not None:
        ax_b.set_ylabel("3D Deflection from Ctrl (mm)", fontsize=9.5, fontweight="medium")
        ax_b.set_xlabel("Time (ms)", fontsize=10, fontweight="medium")
        if plot_var:
            ax_b.set_title("B. Net 3D Trajectory Deflection from Control Reference", fontsize=10.5, fontweight="bold", loc="left")
        else:
            ax_b.set_title("Net 3D Trajectory Deflection from Control Reference", fontsize=10.5, fontweight="bold", loc="left")

    # Unified Figure Legend: Consolidated in a vertical column by default
    active_axes = [ax for ax in [ax_a, ax_b] if ax is not None]
    by_label = {}
    for ax in active_axes:
        h_list, l_list = ax.get_legend_handles_labels()
        for h, l in zip(h_list, l_list):
            if l not in by_label:
                by_label[l] = h
    handles = list(by_label.values())
    labels = list(by_label.keys())

    if legend_orientation == "column":
        ncol = 2 if len(labels) > 9 else 1
        right_margin = 0.79 if ncol == 1 else 0.73
        fig.legend(
            handles, labels,
            loc="center left",
            bbox_to_anchor=(0.81 if ncol == 1 else 0.75, 0.5),
            ncol=ncol,
            frameon=True, framealpha=0.95, fontsize=8.5,
        )
        if plot_var and plot_def:
            fig.subplots_adjust(left=0.08, right=right_margin, top=0.90 if fixed_params_summary else 0.92, bottom=0.09, hspace=0.24)
        else:
            fig.subplots_adjust(left=0.08, right=right_margin, top=0.88 if fixed_params_summary else 0.90, bottom=0.13)
    else:  # row
        ncol = min(7, int(np.ceil(len(labels) / 2))) if len(labels) > 7 else len(labels)
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94 if (len(labels) > 7 or fixed_params_summary) else 0.95),
            ncol=ncol,
            frameon=True, framealpha=0.95, fontsize=8,
        )
        if plot_var and plot_def:
            fig.subplots_adjust(top=0.81 if (len(labels) > 7 or fixed_params_summary) else 0.85, hspace=0.24, left=0.08, right=0.96, bottom=0.09)
        else:
            fig.subplots_adjust(top=0.79 if (len(labels) > 7 or fixed_params_summary) else 0.83, left=0.08, right=0.96, bottom=0.12)

    # Suptitle
    if plot_var and plot_def:
        suptitle_text = f"3D Trajectory Variability & Control Deflection — Target {target}"
    elif plot_var:
        suptitle_text = f"3D Trajectory Variability — Target {target}"
    else:
        suptitle_text = f"3D Control Deflection — Target {target}"

    if fixed_params_summary:
        suptitle_text += f"\n[{fixed_params_summary}]"
    fig.suptitle(suptitle_text, fontsize=11.5, fontweight="bold", y=0.995)

    if save_path:
        out_file = save_path.with_suffix(f".{FIG_FORMAT.lower()}")
        fig.savefig(out_file, format=FIG_FORMAT.lower(), dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Time-Resolved figure: {out_file.name}")
        plt.close(fig)
    return fig



def plot_target_directional_deviation_traces(
    dir_results: Dict[int, Dict[str, Any]],
    target: str = "A",
    fixed_params_summary: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Time-Resolved 3D Directional Deviation Profiles for a Single Target:
        Panel A: Lateral Veer Delta X(t) across time (Left < 0 < Right).
        Panel B: Vertical Lift/Dip Delta Z(t) across time (Dip < 0 < Lift).
        Panel C: Forward Extension Delta Y(t) across time (Undershoot < 0 < Overshoot).
    """
    fig, (ax_x, ax_z, ax_y) = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=FIG_DPI)

    cond_keys = sorted(dir_results.keys(), key=lambda k: (0 if dir_results[k]["is_control"] else 1, k))
    n_stim = sum(not dir_results[k]["is_control"] for k in cond_keys)
    palette = plt.cm.tab10(np.linspace(0.0, 1.0, max(10, n_stim + 1)))

    # Assign consistent condition colors and collect metadata for staggered stim window block
    cond_colors = {}
    stim_info = []
    stim_idx = 0
    for k in cond_keys:
        d = dir_results[k]
        if d["is_control"]:
            cond_colors[k] = COLOR_CTRL
        else:
            col = palette[stim_idx % len(palette)]
            cond_colors[k] = col
            stim_idx += 1
            stim_info.append({
                "cond_num": k,
                "label": d["label"],
                "color": col,
                "stim_dur_ms": d.get("stim_dur_ms", 0.0) or 0.0,
                "is_control": False,
            })

    for k in cond_keys:
        d = dir_results[k]
        t = d["time"]
        is_ctrl = d["is_control"]
        label = d["label"]
        color = cond_colors[k]

        if is_ctrl:
            ls = "--"
            lw = 2.0
            zorder = 5
        else:
            ls = "-"
            lw = 1.8
            zorder = 3

        # Panel A: Delta X (Lateral)
        ax_x.plot(t, d["mean_dx"], color=color, linestyle=ls, linewidth=lw, label=label, zorder=zorder)
        ax_x.fill_between(t, d["mean_dx"] - d["sem_dx"], d["mean_dx"] + d["sem_dx"], color=color, alpha=0.15)

        # Panel B: Delta Z (Vertical)
        ax_z.plot(t, d["mean_dz"], color=color, linestyle=ls, linewidth=lw, label=label, zorder=zorder)
        ax_z.fill_between(t, d["mean_dz"] - d["sem_dz"], d["mean_dz"] + d["sem_dz"], color=color, alpha=0.15)

        # Panel C: Forward (Delta Y)
        ax_y.plot(t, d["mean_dy"], color=color, linestyle=ls, linewidth=lw, label=label, zorder=zorder)
        ax_y.fill_between(t, d["mean_dy"] - d["sem_dy"], d["mean_dy"] + d["sem_dy"], color=color, alpha=0.15)

    for ax, title, ylab in [
        (ax_x, "A. Lateral Deviation ΔX(t) (Left < 0 < Right)", "Lateral Shift ΔX (mm)"),
        (ax_z, "B. Vertical Deviation ΔZ(t) (Dip < 0 < Lift)", "Vertical Shift ΔZ (mm)"),
        (ax_y, "C. Forward Deviation ΔY(t) (Undershoot < 0 < Overshoot)", "Forward Shift ΔY (mm)"),
    ]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.axvline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.1)
        ax.set_title(title, fontsize=10.0, fontweight="bold", loc="left")
        ax.set_ylabel(ylab, fontsize=9.0)
        ax.set_xlabel("Time (ms)", fontsize=9.0)
        ax.grid(True, linestyle="--", alpha=0.35)

        # Expand Y-max dynamically and draw staggered stim window block
        n_durs = len(set(float(c.get("stim_dur_ms", 0.0)) for c in stim_info if (c.get("stim_dur_ms") or 0.0) > 0))
        ladder_headroom = max(0.25, 0.045 * n_durs + 0.10)
        ladder_ratio = max(0.18, min(0.32, 0.045 * n_durs))
        c_ymin, c_ymax = ax.get_ylim()
        n_ymax = c_ymax + (c_ymax - c_ymin) * ladder_headroom
        ax.set_ylim(c_ymin, n_ymax)
        draw_staggered_stim_block(
            ax, stim_info, y_min=c_ymin, y_max=n_ymax,
            ladder_height_ratio=ladder_ratio,
        )

    # Single unified figure legend placed cleanly OUTSIDE the subplots at top
    handles, labels = ax_x.get_legend_handles_labels()
    ncol = min(7, int(np.ceil(len(labels) / 2))) if len(labels) > 7 else len(labels)
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.985),
        ncol=ncol,
        frameon=True, framealpha=0.95, fontsize=8,
    )
    suptitle_text = f"3D Directional Perturbation Time Profiles — Target {target}"
    if fixed_params_summary:
        suptitle_text += f"\n[{fixed_params_summary}]"
    fig.suptitle(suptitle_text, fontsize=11.5, fontweight="bold", y=1.03 if len(labels) > 7 else 1.02)
    fig.subplots_adjust(top=0.82 if len(labels) > 7 else 0.86, left=0.06, right=0.96, bottom=0.12, wspace=0.24)

    if save_path:
        out_file = save_path.with_suffix(f".{FIG_FORMAT.lower()}")
        fig.savefig(out_file, format=FIG_FORMAT.lower(), dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Directional Deviation figure: {out_file.name}")
        plt.close(fig)
    return fig


def plot_combined_spatial_perturbations_figure(
    dir_results_by_target: Dict[str, Dict[int, Dict[str, Any]]],
    fixed_params_summary: str = "",
    save_path: Optional[Path] = None,
    mode: Optional[str] = None,
    trajectory_stat: Optional[str] = None,
    ellipse_scale: Optional[str] = None,
    time_window: Tuple[float, float] = DEFAULT_PLOT_WINDOW_MS,
) -> plt.Figure:
    """
    Spatial Perturbation & Reach Trajectory Figure:
        Overlay Mode (Default): Full 3D spatial reach trajectories (from lift-off to touchscreen)
                 with an overlaid 2D touchscreen endpoint difference panel in the bottom-left with a
                 crisp solid black border, 2-SE dispersion ellipses, perturbation vector arrow, and delta badge.
        Two-Panel Mode: Panel A (Endpoint Perturbations) and Panel B (3D Trajectories) side-by-side.

    Options:
        mode: "overlay" (default: 3D plot with 2D endpoint inset) or "two_panel".
        trajectory_stat: "median" (default: robust to outliers) or "mean".
        ellipse_scale: "2se" (default: 2 standard errors of the mean), "1se", or "1sd" (standard deviation).
        time_window: Reach time window (ms), e.g. (-100.0, 400.0).
    """
    if mode is None:
        mode = SPATIAL_PLOT_MODE
    mode_clean = str(mode).lower().strip()
    is_overlay = mode_clean in ("overlay", "inset", "inset_overlay", "default", "auto")

    if trajectory_stat is None:
        trajectory_stat = TRAJECTORY_STAT
    use_median = str(trajectory_stat).lower().strip() in ("median", "med")
    stat_label = "Median" if use_median else "Mean"

    if ellipse_scale is None:
        ellipse_scale = ELLIPSE_SCALE
    scale_clean = str(ellipse_scale).lower().strip()
    if scale_clean in ("2se", "2_se"):
        ellipse_label = "2-SE"
    elif scale_clean in ("1se", "se", "sem"):
        ellipse_label = "1-SE"
    elif scale_clean in ("2sd", "2_sd"):
        ellipse_label = "2-SD"
    else:
        ellipse_label = "1-SD"

    if is_overlay:
        fig = plt.figure(figsize=(11.0, 10.5), dpi=FIG_DPI)
        ax_3d = fig.add_subplot(1, 1, 1, projection="3d")
        # Inset 2D axes positioned in the bottom-left quadrant of the 3D canvas
        # [left, bottom, width, height] in normalized figure coordinates
        ax_face = fig.add_axes([0.06, 0.10, 0.38, 0.38], zorder=10)
    else:
        fig = plt.figure(figsize=(15.5, 6.8), dpi=FIG_DPI)
        # Equal width ratios so 3D plot visual width matches left plot
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.18)
        ax_face = fig.add_subplot(gs[0, 0])
        ax_3d = fig.add_subplot(gs[0, 1], projection="3d")

    # Extract all unique stimulation conditions across targets for consistent color mapping
    all_stim_conds = sorted(list(set(
        k for res in dir_results_by_target.values() for k, v in res.items() if not v["is_control"]
    )))
    # Non-red color palette for stimulation conditions (Control is uniquely red)
    stim_color_pool = [
        "#1f77b4",  # Steel blue
        "#ff7f0e",  # Amber orange
        "#2ca02c",  # Emerald green
        "#9467bd",  # Purple
        "#17becf",  # Teal/Cyan
        "#e377c2",  # Magenta/Pink
        "#bcbd22",  # Olive
        "#8c564b",  # Brown
        "#34495e",  # Slate navy
    ]
    cond_colors = {
        cond: stim_color_pool[i % len(stim_color_pool)]
        for i, cond in enumerate(all_stim_conds)
    }

    # Track global stim duration across targets
    stim_dur_max = 0.0
    for d_tgt in dir_results_by_target.values():
        for v in d_tgt.values():
            stim_dur_max = max(stim_dur_max, v.get("stim_dur_ms", 0.0) or 0.0)

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # Panel A: Touchscreen Face (Endpoint Perturbations)
    # -------------------------------------------------------------------------
    targets_present = sorted(dir_results_by_target.keys())
    delta_badges: List[Tuple[int, str, Any]] = []
    all_face_x: List[float] = []
    all_face_z: List[float] = []
    target_info_by_tgt: Dict[str, Any] = {}

    for tgt in targets_present:
        d_tgt = dir_results_by_target[tgt]
        ctrl_d = next((v for v in d_tgt.values() if v["is_control"]), None)

        # Extract target_info if present in any condition
        tgt_info = None
        for v in d_tgt.values():
            if "target_info" in v and v["target_info"]:
                tgt_info = v["target_info"]
                break
        if tgt_info is not None:
            target_info_by_tgt[tgt] = tgt_info
            tgt_x = tgt_info["target_x"]
            tgt_z = tgt_info["target_z"]
            tgt_r = tgt_info["target_radius_mm"]
            tgt_d = tgt_info["target_diameter_mm"]

            # 1 mm target circle around control center with subtle neutral tone and dashed boundary
            target_patch = Circle(
                (tgt_x, tgt_z), radius=tgt_r,
                facecolor="#eaeded", edgecolor="#5d6d7e",
                linestyle="--", linewidth=1.4, alpha=0.60, zorder=1,
            )
            ax_face.add_patch(target_patch)

            # Target center crosshair (+)
            ax_face.plot([tgt_x - 1.0, tgt_x + 1.0], [tgt_z, tgt_z], color="#5d6d7e", linewidth=1.1, zorder=2)
            ax_face.plot([tgt_x, tgt_x], [tgt_z - 1.0, tgt_z + 1.0], color="#5d6d7e", linewidth=1.1, zorder=2)

            # Subtle label on target perimeter
            tgt_lbl = f"{int(round(tgt_d))} mm Target" if tgt_d >= 2.0 else f"{tgt_d:.1f} mm Target"
            ax_face.text(
                tgt_x, tgt_z + tgt_r + 0.5, tgt_lbl,
                fontsize=7.5, color="#566573", ha="center", va="bottom",
                fontweight="medium", zorder=3,
            )
            all_face_x.extend([tgt_x - tgt_r - 2.0, tgt_x + tgt_r + 2.0])
            all_face_z.extend([tgt_z - tgt_r - 2.0, tgt_z + tgt_r + 2.0])

        if ctrl_d:
            ctrl_end = ctrl_d.get("ctrl_median_end") if use_median else ctrl_d.get("ctrl_mean_end")
            if ctrl_end is None:
                ctrl_end = ctrl_d["ctrl_mean_end"]
            ctrl_x, ctrl_y, ctrl_z = ctrl_end

            # Control endpoint cluster with completed reaches
            c_comp = ctrl_d.get("is_completed_reach", ctrl_d.get("ctrl_completed"))
            c_use = c_comp if (c_comp is not None and np.any(c_comp)) else np.isfinite(ctrl_d["abs_end_x"])
            c_px = ctrl_d["abs_end_x"][c_use]
            c_pz = ctrl_d["abs_end_z"][c_use]

            # Control endpoint cluster with lighter, softer opacity
            ax_face.scatter(
                c_px, c_pz,
                color=COLOR_CTRL, alpha=0.22, s=24, edgecolors="none", zorder=4,
            )
            # Dispersion ellipse with soft background tint centered at the control center
            draw_confidence_ellipse(
                ax_face, c_px, c_pz,
                n_scale=1.0, scale_mode=ellipse_scale,
                edgecolor=COLOR_CTRL, facecolor=COLOR_CTRL, alpha=0.09,
                center=(ctrl_x, ctrl_z),
                linestyle=":", linewidth=1.3, zorder=3,
            )
            # Subtle reference crosshair lines centered at Control center
            ax_face.axhline(ctrl_z, color=COLOR_CTRL, linestyle=":", linewidth=0.9, alpha=0.30, zorder=2)
            ax_face.axvline(ctrl_x, color=COLOR_CTRL, linestyle=":", linewidth=0.9, alpha=0.30, zorder=2)

            # Control center marker
            ax_face.scatter(
                ctrl_x, ctrl_z, marker="D", s=115, color=COLOR_CTRL,
                edgecolors="#ffffff", linewidths=1.8, zorder=8,
            )
            all_face_x.extend([ctrl_x, float(np.nanmin(c_px)), float(np.nanmax(c_px))])
            all_face_z.extend([ctrl_z, float(np.nanmin(c_pz)), float(np.nanmax(c_pz))])

        # Plot each stimulation condition
        for cond_num, v in d_tgt.items():
            if v["is_control"]:
                continue
            color = cond_colors.get(cond_num, COLOR_STIM_DEFAULT)
            stim_end = v.get("median_abs_end") if use_median else v.get("mean_abs_end")
            if stim_end is None:
                stim_end = (v["mean_abs_end_x"], v["mean_abs_end_y"], v["mean_abs_end_z"])
            mx, my, mz = stim_end
            pts_x = v["abs_end_x"]
            pts_z = v["abs_end_z"]

            # Individual reach trial endpoints (completed reaches only)
            is_comp = v.get("is_completed_reach")
            s_use = is_comp if (is_comp is not None and np.any(is_comp)) else np.isfinite(pts_x)
            plot_px = pts_x[s_use]
            plot_pz = pts_z[s_use]

            ax_face.scatter(plot_px, plot_pz, color=color, alpha=0.28, s=26, edgecolors="none", zorder=5)
            # Dispersion ellipse with soft fill centered at condition center
            draw_confidence_ellipse(
                ax_face, plot_px, plot_pz,
                n_scale=1.0, scale_mode=ellipse_scale,
                edgecolor=color, facecolor=color, alpha=0.10,
                center=(mx, mz),
                linestyle="--", linewidth=1.3, zorder=3,
            )
            all_face_x.extend([mx, float(np.nanmin(plot_px)), float(np.nanmax(plot_px))])
            all_face_z.extend([mz, float(np.nanmin(plot_pz)), float(np.nanmax(plot_pz))])

            # Perturbation vector arrow from control to stim center
            if ctrl_d:
                ax_face.annotate(
                    "", xy=(mx, mz), xytext=(ctrl_x, ctrl_z),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.6, mutation_scale=14),
                    zorder=7,
                )
                dx = mx - ctrl_x
                dz = mz - ctrl_z
                d2d = float(np.sqrt(dx**2 + dz**2))
                p_val = v.get("p_norm", 1.0)
                q_val = v.get("q_norm", p_val)
                star_str = f" ({p_stars_label(q_val)})" if q_val < 0.05 else ""

                rel_disp = v.get("rel_disp_pct", (d2d / 10.0) * 100.0)
                hit_rate = v.get("hit_rate_pct", 0.0)
                ctrl_hit = v.get("ctrl_hit_rate_pct", ctrl_d.get("hit_rate_pct", 0.0) if ctrl_d else 0.0)
                med_t_touch = v.get("median_t_end", 0.0)

                tgt_r_val = v.get("target_info", {}).get("target_radius_mm", TARGET_RADIUS_MM) if v.get("target_info") else TARGET_RADIUS_MM
                if len(all_stim_conds) == 1:
                    touch_str = f" (Touch: ~{med_t_touch:.0f} ms)" if med_t_touch >= 75.0 else ""
                    badge_lines = [
                        f"Endpoint Deflection: Δ = {d2d:.1f} mm{star_str}{touch_str}",
                        f"  [ΔX: {dx:+.1f}, ΔZ: {dz:+.1f} mm]",
                        f"Hit Rate (≤{int(round(tgt_r_val))} mm): Stim {hit_rate:.0f}% vs Ctrl {ctrl_hit:.0f}%",
                    ]
                    badge_text = "\n".join(badge_lines)
                else:
                    c_lbl = v.get("base_label", f"BR {cond_num}")
                    c_lbl = re.sub(r"\s*\(n=\d+\)", "", c_lbl).strip()
                    touch_info = f" [Touch ~{med_t_touch:.0f} ms]" if med_t_touch >= 75.0 else ""
                    badge_text = (
                        f"• {c_lbl}: Endpoint Δ = {d2d:.1f} mm{star_str}{touch_info} | "
                        f"Hit (≤{int(round(tgt_r_val))} mm): {hit_rate:.0f}% (Ctrl {ctrl_hit:.0f}%)"
                    )
                delta_badges.append((cond_num, badge_text, color))

            # Stim center endpoint marker
            ax_face.scatter(
                mx, mz, color=color, s=95, edgecolors="#1b2631", linewidths=1.8,
                zorder=9,
            )

    # Place delta callout box separately outside the plots so no data is obscured
    if delta_badges:
        combined_badge_text = "\n".join(b[1] for b in delta_badges)
        badge_ec = delta_badges[0][2] if len(delta_badges) == 1 else "#bdc3c7"
        if is_overlay:
            # Positioned in the open upper-left quadrant of the canvas,
            # comfortably above the 2D touchscreen inset and to the left of 3D trajectories
            header = "Endpoint Perturbations & Hit Rates:" if len(all_stim_conds) > 1 else ""
            full_text = f"{header}\n{combined_badge_text}" if header else combined_badge_text
            fig.text(
                0.06, 0.85, full_text,
                fontsize=7.8, fontweight="bold", color="#1c2833",
                ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec=badge_ec, lw=1.2, alpha=0.96),
                zorder=12,
            )
        else:
            # Two-panel mode: Place inside ax_face without covering the center
            ax_face.text(
                0.03, 0.04, combined_badge_text,
                transform=ax_face.transAxes,
                fontsize=7.5, fontweight="bold", color="#1c2833",
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.35", fc="#ffffff", ec=badge_ec, lw=1.1, alpha=0.95),
                zorder=12,
            )

    # Format touchscreen face
    tgt_d_first = next((info.get("target_diameter_mm", TARGET_DIAMETER_MM) for info in target_info_by_tgt.values()), TARGET_DIAMETER_MM)
    title_face = f"Endpoint Perturbations ({ellipse_label}) | Target: {int(round(tgt_d_first))} mm Control Anchor"

    if is_overlay:
        # Give the 2D inset an opaque white background and a crisp solid black border
        ax_face.set_facecolor("#ffffff")
        ax_face.patch.set_alpha(1.0)
        for spine in ax_face.spines.values():
            spine.set_color("#000000")
            spine.set_linewidth(1.3)
            spine.set_visible(True)
        ax_face.tick_params(axis="both", labelsize=8.0, length=3.5, width=1.0)
        ax_face.set_xlabel("Lateral X (mm)", fontsize=8.5, fontweight="medium", labelpad=3)
        ax_face.set_ylabel("Vertical Z (mm)", fontsize=8.5, fontweight="medium", labelpad=3)
        ax_face.set_title(title_face, fontsize=9.0, fontweight="bold", loc="left", pad=4)
        ax_face.grid(False)
    else:
        ax_face.spines["top"].set_visible(False)
        ax_face.spines["right"].set_visible(False)
        ax_face.set_xlabel("Lateral X (mm)", fontsize=9.5, fontweight="medium")
        ax_face.set_ylabel("Vertical Z (mm)", fontsize=9.5, fontweight="medium")
        ax_face.set_title(title_face, fontsize=10.5, fontweight="bold", loc="left")
        ax_face.grid(False)

    if all_face_x and all_face_z:
        x_min, x_max = float(np.nanmin(all_face_x)), float(np.nanmax(all_face_x))
        z_min, z_max = float(np.nanmin(all_face_z)), float(np.nanmax(all_face_z))
        x_span = max(14.0, x_max - x_min)
        z_span = max(14.0, z_max - z_min)
        max_span = max(x_span, z_span)
        cx = (x_min + x_max) / 2.0
        cz = (z_min + z_max) / 2.0
        half = (max_span / 2.0) * 1.18
        ax_face.set_xlim(cx - half, cx + half)
        ax_face.set_ylim(cz - half, cz + half)
    ax_face.set_aspect("equal", adjustable="datalim")

    # -------------------------------------------------------------
    # Panel B: True 3D Spatial Trajectories
    # -------------------------------------------------------------
    all_3d_x, all_3d_y, all_3d_z = [], [], []

    for tgt in targets_present:
        d_tgt = dir_results_by_target[tgt]
        ctrl_d = next((v for v in d_tgt.values() if v["is_control"]), None)

        if ctrl_d:
            # Control trial trajectory bundle (individual trials a bit lighter)
            c_tx = ctrl_d.get("trials_x")
            c_ty = ctrl_d.get("trials_y")
            c_tz = ctrl_d.get("trials_z")
            c_dark_col = _adjust_color_lightness(COLOR_CTRL, factor=0.85)
            if c_tx is not None and getattr(c_tx, "ndim", 0) == 2:
                n_draw = min(len(c_tx), 35)
                step = max(1, len(c_tx) // n_draw)
                for i in range(0, len(c_tx), step):
                    ax_3d.plot(c_tx[i], c_ty[i], c_tz[i], color=c_dark_col, alpha=0.22, linewidth=0.75, zorder=2)

            ctrl_traj = ctrl_d.get("ctrl_median_traj") if use_median else ctrl_d.get("ctrl_mean_traj")
            if ctrl_traj is None:
                ctrl_traj = ctrl_d["ctrl_mean_traj"]
            cmx, cmy, cmz = ctrl_traj
            ax_3d.plot(
                cmx, cmy, cmz, color=COLOR_CTRL, linestyle="-", linewidth=2.8, alpha=0.98,
                zorder=7,
            )
            ctrl_end = ctrl_d.get("ctrl_median_end") if use_median else ctrl_d.get("ctrl_mean_end")
            if ctrl_end is None:
                ctrl_end = ctrl_d["ctrl_mean_end"]
            ctrl_x, ctrl_y, ctrl_z = ctrl_end
            ax_3d.scatter([ctrl_x], [ctrl_y], [ctrl_z], marker="D", s=85, color=COLOR_CTRL, edgecolors="#ffffff", linewidths=1.5, zorder=9)

            # Red dot at t = 0 ms (stim onset / IR beam crossing) on control median trajectory
            ctrl_time = ctrl_d.get("time")
            if ctrl_time is not None and len(ctrl_time) > 0:
                t0_idx = int(np.argmin(np.abs(ctrl_time - 0.0)))
                if abs(ctrl_time[t0_idx] - 0.0) <= 30.0 and t0_idx < len(cmx):
                    ax_3d.scatter([cmx[t0_idx]], [cmy[t0_idx]], [cmz[t0_idx]], color="#e74c3c", s=70, edgecolors="#ffffff", linewidths=1.6, zorder=14)

            all_3d_x.extend([float(np.nanmin(cmx)), float(np.nanmax(cmx)), ctrl_x])
            all_3d_y.extend([float(np.nanmin(cmy)), float(np.nanmax(cmy)), ctrl_y])
            all_3d_z.extend([float(np.nanmin(cmz)), float(np.nanmax(cmz)), ctrl_z])

            # Draw 10 mm physical target perimeter in 3D at touchscreen plane
            if tgt in target_info_by_tgt:
                t_inf = target_info_by_tgt[tgt]
                t_x = t_inf["target_x"]
                t_y = t_inf["target_y"] if t_inf.get("target_y") is not None and abs(t_inf.get("target_y")) > 10 else ctrl_y
                t_z = t_inf["target_z"]
                t_r = t_inf["target_radius_mm"]
                th = np.linspace(0, 2*np.pi, 50)
                xc = t_x + t_r * np.cos(th)
                zc = t_z + t_r * np.sin(th)
                yc = np.full_like(xc, t_y)
                ax_3d.plot(xc, yc, zc, color="#5d6d7e", linestyle="--", linewidth=1.5, alpha=0.75, zorder=6)
                ax_3d.scatter([t_x], [t_y], [t_z], color="#5d6d7e", s=35, marker="+", linewidths=1.2, zorder=7)
                all_3d_x.extend([t_x - t_r, t_x + t_r])
                all_3d_y.extend([t_y])
                all_3d_z.extend([t_z - t_r, t_z + t_r])

        for cond_num, v in d_tgt.items():
            if v["is_control"]:
                continue
            color = cond_colors.get(cond_num, COLOR_STIM_DEFAULT)

            # Stim trial trajectory bundle (individual trials a bit lighter)
            s_tx = v.get("trials_x")
            s_ty = v.get("trials_y")
            s_tz = v.get("trials_z")
            s_dark_col = _adjust_color_lightness(color, factor=0.85)
            if s_tx is not None and getattr(s_tx, "ndim", 0) == 2:
                n_draw = min(len(s_tx), 35)
                step = max(1, len(s_tx) // n_draw)
                for i in range(0, len(s_tx), step):
                    ax_3d.plot(s_tx[i], s_ty[i], s_tz[i], color=s_dark_col, alpha=0.22, linewidth=0.75, zorder=3)

            stim_traj = v.get("median_traj") if use_median else v.get("mean_traj")
            if stim_traj is None:
                stim_traj = v["mean_traj"]
            smx, smy, smz = stim_traj
            ax_3d.plot(
                smx, smy, smz, color=color, linestyle="-", linewidth=2.4, alpha=0.98,
                zorder=8,
            )
            stim_end = v.get("median_abs_end") if use_median else v.get("mean_abs_end")
            if stim_end is None:
                stim_end = (v["mean_abs_end_x"], v["mean_abs_end_y"], v["mean_abs_end_z"])
            mx, my, mz = stim_end
            ax_3d.scatter([mx], [my], [mz], color=color, s=75, edgecolors="#1b2631", linewidths=1.5, zorder=11)

            # Red dot at t = 0 ms (stim onset / IR beam crossing) on stim median trajectory
            s_time = v.get("time")
            if s_time is not None and len(s_time) > 0:
                t0_idx = int(np.argmin(np.abs(s_time - 0.0)))
                if abs(s_time[t0_idx] - 0.0) <= 30.0 and t0_idx < len(smx):
                    ax_3d.scatter([smx[t0_idx]], [smy[t0_idx]], [smz[t0_idx]], color="#e74c3c", s=70, edgecolors="#ffffff", linewidths=1.6, zorder=14)

            all_3d_x.extend([float(np.nanmin(smx)), float(np.nanmax(smx)), mx])
            all_3d_y.extend([float(np.nanmin(smy)), float(np.nanmax(smy)), my])
            all_3d_z.extend([float(np.nanmin(smz)), float(np.nanmax(smz)), mz])

    ax_3d.set_xlabel("Lateral X (mm)", fontsize=8.5, labelpad=6)
    ax_3d.set_ylabel("Forward Y (mm)", fontsize=8.5, labelpad=6)
    ax_3d.set_zlabel("Vertical Z (mm)", fontsize=8.5, labelpad=6)
    if not is_overlay:
        ax_3d.set_title("3D Trajectories", fontsize=11.0, fontweight="bold", loc="left")

    # Spaced-out 3D ticks (at most 4 bins per axis to avoid crowded numbers)
    ax_3d.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax_3d.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax_3d.zaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    # Clean 3D white panes and subtle grid lines
    ax_3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    for ax_info in (ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis):
        if hasattr(ax_info, "_axinfo") and "grid" in ax_info._axinfo:
            ax_info._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 0.5)
            ax_info._axinfo["grid"]["linestyle"] = ":"

    # Aspect ratio
    if all_3d_x and all_3d_y and all_3d_z:
        span_x = max(10.0, float(np.nanmax(all_3d_x) - np.nanmin(all_3d_x)))
        span_y = max(10.0, float(np.nanmax(all_3d_y) - np.nanmin(all_3d_y)))
        span_z = max(10.0, float(np.nanmax(all_3d_z) - np.nanmin(all_3d_z)))
        ax_3d.set_box_aspect([span_x, span_y, span_z])

    # Viewpoint & camera distance zoom to ensure 3D plot visual width matches
    ax_3d.view_init(elev=20, azim=-78)
    if hasattr(ax_3d, "dist"):
        ax_3d.dist = 8.8 if is_overlay else 8.5



    # Build clean unified figure legend placed outside the plots at top
    legend_handles = []
    legend_labels = []

    ctrl_a = next((v for v in dir_results_by_target.get("A", {}).values() if v["is_control"]), None)
    ctrl_b = next((v for v in dir_results_by_target.get("B", {}).values() if v["is_control"]), None)
    if SHOW_TRIAL_COUNTS:
        if ctrl_a and ctrl_b:
            ctrl_lbl = f"Control ({stat_label}, Tgt A n={ctrl_a['n_trials']}, Tgt B n={ctrl_b['n_trials']})"
        elif ctrl_a:
            ctrl_lbl = f"Control ({stat_label}, n={ctrl_a['n_trials']})"
        elif ctrl_b:
            ctrl_lbl = f"Control ({stat_label}, n={ctrl_b['n_trials']})"
        else:
            ctrl_lbl = f"Control ({stat_label})"
    else:
        ctrl_lbl = f"Control ({stat_label})"

    ctrl_handle = plt.Line2D([0], [0], color=COLOR_CTRL, linestyle="-", linewidth=2.8, marker="D", markersize=6.5)
    legend_handles.append(ctrl_handle)
    legend_labels.append(ctrl_lbl)

    for c_num in all_stim_conds:
        v_a = dir_results_by_target.get("A", {}).get(c_num)
        v_b = dir_results_by_target.get("B", {}).get(c_num)
        v_sample = v_a or v_b
        base_lbl = v_sample.get("base_label") if v_sample else f"BR {c_num}"
        base_lbl = re.sub(r"\s*\(n=\d+\)", "", base_lbl).strip()

        if SHOW_TRIAL_COUNTS:
            if v_a is not None and v_b is not None:
                n_str = f"n_A={v_a['n_trials']}, n_B={v_b['n_trials']}"
            elif v_a is not None:
                n_str = f"n={v_a['n_trials']}"
            elif v_b is not None:
                n_str = f"n={v_b['n_trials']}"
            else:
                n_str = ""
            lbl = f"{base_lbl} ({stat_label}, {n_str})" if n_str else f"{base_lbl} ({stat_label})"
        else:
            lbl = f"{base_lbl} ({stat_label})"

        col = cond_colors.get(c_num, COLOR_STIM_DEFAULT)
        h = plt.Line2D([0], [0], color=col, linestyle="-", linewidth=2.2, marker="o", markersize=6)
        legend_handles.append(h)
        legend_labels.append(lbl)

    # Red dot legend handle for t = 0 ms (stim onset / IR crossing)
    t0_handle = plt.Line2D([0], [0], color="#e74c3c", linestyle="none", marker="o", markersize=6.5, markeredgecolor="#ffffff", markeredgewidth=1.0)
    legend_handles.append(t0_handle)
    legend_labels.append("t = 0 ms (Stim / IR Cross)")

    ncol = min(7, int(np.ceil(len(legend_labels) / 2))) if len(legend_labels) > 7 else len(legend_labels)
    fig.legend(
        legend_handles, legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.940 if is_overlay else 0.965),
        ncol=ncol,
        frameon=True, framealpha=0.95, fontsize=8.5,
    )

    time_str = f"{int(time_window[0])}–{int(time_window[1])} ms"
    targets_str = " & ".join(f"Target {t}" for t in targets_present)
    suptitle_text = f"Touchscreen Endpoint Perturbations & 3D Spatial Trajectories ({time_str}) — {targets_str} ({stat_label})"
    if fixed_params_summary:
        suptitle_text += f"\n[{fixed_params_summary}]"
    fig.suptitle(
        suptitle_text,
        fontsize=11.5, fontweight="bold", y=0.992 if is_overlay else 0.995,
    )

    if is_overlay:
        fig.subplots_adjust(top=0.86 if len(legend_labels) > 7 else 0.88, left=0.04, right=0.96, bottom=0.06)
    else:
        fig.subplots_adjust(top=0.86 if len(legend_labels) > 7 else 0.88, left=0.06, right=0.97, bottom=0.10, wspace=0.18)

    if save_path:
        out_file = save_path.with_suffix(f".{FIG_FORMAT.lower()}")
        fig.savefig(out_file, format=FIG_FORMAT.lower(), dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Spatial Perturbation figure: {out_file.name}")
        plt.close(fig)
    return fig


def plot_progress_resampled_trajectories_figure(
    conditions: Sequence[ReachCondition],
    target: str = "A",
    fixed_params_summary: str = "",
    save_path: Optional[Path] = None,
    resample_window: Optional[Tuple[float, float]] = None,
    n_progress_samples: int = RSA_N_PROGRESS_SAMPLES,
    trajectory_stat: str = TRAJECTORY_STAT,
    progress_mode: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Diagnostic & Verification Figure for Normalized Reaching Trajectories:
        Panel A: 3D Trajectories in Task Space resampled to 100 uniform progress points (0% to 100%).
        Panel B: 3D Spatial Path Deviation from Control vs. Reach Progress (0% to 100%).
        Panel C: Lateral X (Veer) Coordinate vs. Progress (0% to 100%).
        Panel D: Forward Y (Reach Depth) Coordinate vs. Progress (0% to 100%).
        Panel E: Vertical Z (Lift/Dip) Coordinate vs. Progress (0% to 100%).
    """
    if not conditions:
        return None

    if resample_window is None:
        resample_window = (0.0, 350.0)

    pm = progress_mode or RSA_PROGRESS_MODE
    is_y_mode = str(pm).lower().strip() in ("forward_y", "y", "depth", "forward")
    prog_axis_name = "Forward Reach Depth Progress (%)" if is_y_mode else "Reach Path Progress (%)"

    use_median = (str(trajectory_stat).lower() in ("median", "med"))
    stat_label = "Median" if use_median else "Mean"

    cond_list = sorted(conditions, key=lambda c: (0 if c.is_control else 1, c.cond_num))
    stim_color_pool = [
        "#1f77b4",  # Steel blue (1p)
        "#ff7f0e",  # Amber orange (2p)
        "#2ca02c",  # Emerald green (4p)
        "#9467bd",  # Purple (8p) - distinct from crimson control
        "#17becf",  # Teal/Cyan (20p)
        "#8c564b",  # Brown (40p)
        "#e377c2",  # Magenta/Pink
        "#bcbd22",  # Olive
        "#34495e",  # Slate navy
    ]

    cond_colors = {}
    stim_idx = 0
    for c in cond_list:
        if c.is_control:
            cond_colors[c.cond_num] = COLOR_CTRL
        else:
            cond_colors[c.cond_num] = stim_color_pool[stim_idx % len(stim_color_pool)]
            stim_idx += 1

    progress_pct = np.linspace(0.0, 100.0, n_progress_samples)

    resampled_items: List[Dict[str, Any]] = []
    ctrl_item: Optional[Dict[str, Any]] = None

    for c in cond_list:
        rx, ry, rz = resample_reach_trajectory(
            c.x, c.y, c.z, time=c.time, n_progress=n_progress_samples, mode=pm
        )

        valid_mask = np.all(np.isfinite(rx) & np.isfinite(ry) & np.isfinite(rz), axis=1)
        rx_val = rx[valid_mask]
        ry_val = ry[valid_mask]
        rz_val = rz[valid_mask]

        if len(rx_val) == 0:
            continue

        if use_median:
            mx = np.nanmedian(rx_val, axis=0)
            my = np.nanmedian(ry_val, axis=0)
            mz = np.nanmedian(rz_val, axis=0)
        else:
            mx = np.nanmean(rx_val, axis=0)
            my = np.nanmean(ry_val, axis=0)
            mz = np.nanmean(rz_val, axis=0)

        base_lbl = getattr(c, "short_label", None) or re.sub(r"\s*\(n=\d+\)", "", c.label).strip()
        disp_lbl = f"{base_lbl} (n={len(rx_val)})" if SHOW_TRIAL_COUNTS else base_lbl

        item = {
            "cond": c,
            "cond_num": c.cond_num,
            "label": disp_lbl,
            "base_label": base_lbl,
            "is_control": c.is_control,
            "color": cond_colors[c.cond_num],
            "rx": rx_val,
            "ry": ry_val,
            "rz": rz_val,
            "mx": mx,
            "my": my,
            "mz": mz,
            "n_valid": len(rx_val),
        }
        resampled_items.append(item)
        if c.is_control and ctrl_item is None:
            ctrl_item = item

    if not resampled_items:
        return None

    # Calculate within-condition variability and deviations relative to control along normalized progress
    for it in resampled_items:
        n_tr = len(it["rx"])
        if n_tr > 1:
            p_list = []
            for i in range(n_tr - 1):
                p_dx = it["rx"][i + 1:] - it["rx"][i]
                p_dy = it["ry"][i + 1:] - it["ry"][i]
                p_dz = it["rz"][i + 1:] - it["rz"][i]
                p_dist = np.sqrt(p_dx**2 + p_dy**2 + p_dz**2)
                p_list.append(p_dist)
            p_pairs = np.vstack(p_list)
            it["var_m"] = np.nanmedian(p_pairs, axis=0) if use_median else np.nanmean(p_pairs, axis=0)
            it["var_sem"] = np.nanstd(p_pairs, axis=0) / math.sqrt(p_pairs.shape[0]) if p_pairs.shape[0] > 1 else np.zeros(n_progress_samples)
        else:
            it["var_m"] = np.zeros(n_progress_samples)
            it["var_sem"] = np.zeros(n_progress_samples)

    if ctrl_item is not None:
        cmx, cmy, cmz = ctrl_item["mx"], ctrl_item["my"], ctrl_item["mz"]
        for it in resampled_items:
            it["dx"] = it["mx"] - cmx
            it["dy"] = it["my"] - cmy
            it["dz"] = it["mz"] - cmz
            it["dev_3d"] = np.sqrt(it["dx"]**2 + it["dy"]**2 + it["dz"]**2)

            t_dx = it["rx"] - cmx[None, :]
            t_dy = it["ry"] - cmy[None, :]
            t_dz = it["rz"] - cmz[None, :]
            it["trial_dev_3d"] = np.sqrt(t_dx**2 + t_dy**2 + t_dz**2)
            n_tr = len(it["rx"])
            it["sem_dev_3d"] = np.nanstd(it["trial_dev_3d"], axis=0) / math.sqrt(n_tr) if n_tr > 1 else np.zeros(n_progress_samples)
            it["sem_dx"] = np.nanstd(t_dx, axis=0) / math.sqrt(n_tr) if n_tr > 1 else np.zeros(n_progress_samples)
            it["sem_dy"] = np.nanstd(t_dy, axis=0) / math.sqrt(n_tr) if n_tr > 1 else np.zeros(n_progress_samples)
            it["sem_dz"] = np.nanstd(t_dz, axis=0) / math.sqrt(n_tr) if n_tr > 1 else np.zeros(n_progress_samples)

    # Build 6-panel unified figure (2 rows x 3 columns)
    fig = plt.figure(figsize=(16.5, 9.8), dpi=FIG_DPI)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.10, 0.90], hspace=0.35, wspace=0.28)

    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_def = fig.add_subplot(gs[0, 1])
    ax_var = fig.add_subplot(gs[0, 2])
    ax_x = fig.add_subplot(gs[1, 0])
    ax_z = fig.add_subplot(gs[1, 1])
    ax_y = fig.add_subplot(gs[1, 2])

    legend_handles = []
    legend_labels = []

    # Panel A: 3D Resampled Spatial Paths
    for it in resampled_items:
        col = it["color"]
        is_ctrl = it["is_control"]
        light_col = COLOR_CTRL_LIGHT if is_ctrl else _adjust_color_lightness(col, factor=0.85)

        # Plot individual resampled trials
        n_draw = min(len(it["rx"]), 35)
        step = max(1, len(it["rx"]) // n_draw)
        for i in range(0, len(it["rx"]), step):
            ax_3d.plot(it["rx"][i], it["ry"][i], it["rz"][i], color=light_col, alpha=0.18, linewidth=0.75, zorder=2)

        # Plot central tendency
        lw = 2.8 if is_ctrl else 2.3
        ls = "-"
        ax_3d.plot(it["mx"], it["my"], it["mz"], color=col, linestyle=ls, linewidth=lw, alpha=0.98, zorder=8)

        # Start (0%) and End (100%) markers
        ax_3d.scatter([it["mx"][0]], [it["my"][0]], [it["mz"][0]], color="#2ecc71" if is_ctrl else col, marker="o", s=40, edgecolors="#ffffff", linewidths=1.2, zorder=12)
        ax_3d.scatter([it["mx"][-1]], [it["my"][-1]], [it["mz"][-1]], color=col, marker="D" if is_ctrl else "o", s=70, edgecolors="#1b2631", linewidths=1.3, zorder=13)

        h = plt.Line2D([0], [0], color=col, linestyle="-", linewidth=2.4, marker="D" if is_ctrl else "o", markersize=6)
        legend_handles.append(h)
        legend_labels.append(it["label"])

    ax_3d.set_xlabel("Lateral X (mm)", fontsize=8.5, labelpad=6)
    ax_3d.set_ylabel("Forward Y (mm)", fontsize=8.5, labelpad=6)
    ax_3d.set_zlabel("Vertical Z (mm)", fontsize=8.5, labelpad=6)
    title_a = "A. 3D Forward-Y Paths (0% to 100% Depth)" if is_y_mode else "A. 3D Resampled Trajectories (0% to 100% Progress)"
    ax_3d.set_title(title_a, fontsize=10.0, fontweight="bold", loc="left")
    ax_3d.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax_3d.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax_3d.zaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax_3d.view_init(elev=20, azim=-78)

    # Panel B: Net 3D Deflection from Control Reference
    ax_def.axhline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.2, alpha=0.8)
    ax_def.axhline(TARGET_DIAMETER_MM, color="#e67e22", linestyle="--", linewidth=1.2, alpha=0.75, label=f"Target Width ({int(TARGET_DIAMETER_MM)} mm)")
    ax_def.axhline(TARGET_DIAMETER_MM / 2.0, color="#f39c12", linestyle=":", linewidth=1.0, alpha=0.65, label=f"Target Radius ({int(TARGET_DIAMETER_MM/2.0)} mm)")
    if ctrl_item is not None:
        for it in resampled_items:
            if it["is_control"] or "dev_3d" not in it:
                continue
            col = it["color"]
            ax_def.plot(progress_pct, it["dev_3d"], color=col, linewidth=2.2, label=it["label"], zorder=5)
            ax_def.fill_between(progress_pct, it["dev_3d"] - it["sem_dev_3d"], it["dev_3d"] + it["sem_dev_3d"], color=col, alpha=0.15, zorder=3)

            peak_idx = int(np.nanargmax(it["dev_3d"]))
            peak_p = progress_pct[peak_idx]
            peak_val = it["dev_3d"][peak_idx]
            ax_def.scatter([peak_p], [peak_val], color=col, s=42, edgecolors="#1b2631", linewidths=1.2, zorder=7)

    ax_def.set_xlabel(prog_axis_name, fontsize=8.5, labelpad=5)
    ax_def.set_ylabel("3D Deflection from Ctrl (mm)", fontsize=8.5, labelpad=5)
    title_b = "B. Net 3D Deflection from Control vs. Forward Depth" if is_y_mode else "B. Net 3D Deflection vs. Progress"
    ax_def.set_title(title_b, fontsize=10.0, fontweight="bold", loc="left")
    ax_def.set_xlim(-2.0, 102.0)
    ax_def.set_ylim(bottom=-0.5)
    ax_def.spines["top"].set_visible(False)
    ax_def.spines["right"].set_visible(False)
    ax_def.grid(True, linestyle=":", alpha=0.45, color="#b0bec5")

    # Panel C: Within-Condition 3D Trajectory Variability
    for it in resampled_items:
        col = it["color"]
        is_ctrl = it["is_control"]
        ls = "--" if is_ctrl else "-"
        lw = 2.4 if is_ctrl else 1.9
        ax_var.plot(progress_pct, it["var_m"], color=col, linestyle=ls, linewidth=lw, label=it["label"], zorder=5)
        ax_var.fill_between(progress_pct, it["var_m"] - it["var_sem"], it["var_m"] + it["var_sem"], color=col, alpha=0.15, zorder=3)

    ax_var.set_xlabel(prog_axis_name, fontsize=8.5, labelpad=5)
    ax_var.set_ylabel("3D Spatial Variability (mm)", fontsize=8.5, labelpad=5)
    title_c = "C. Within-Condition Variability vs. Forward Depth" if is_y_mode else "C. Within-Condition Variability vs. Progress"
    ax_var.set_title(title_c, fontsize=10.0, fontweight="bold", loc="left")
    ax_var.set_xlim(-2.0, 102.0)
    ax_var.set_ylim(bottom=-0.2)
    ax_var.spines["top"].set_visible(False)
    ax_var.spines["right"].set_visible(False)
    ax_var.grid(True, linestyle=":", alpha=0.45, color="#b0bec5")

    # Bottom Row: Directional Perturbations (D: Lateral X, E: Vertical Z, F: Forward Y)
    for ax in [ax_x, ax_z, ax_y]:
        ax.axhline(0.0, color=COLOR_CTRL, linestyle="--", linewidth=1.5, alpha=0.85, zorder=4)
        ax.set_xlim(-2.0, 102.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle=":", alpha=0.45, color="#b0bec5")

    if ctrl_item is not None:
        for it in resampled_items:
            if it["is_control"] or "dx" not in it:
                continue
            col = it["color"]
            # Panel D: Lateral X
            ax_x.plot(progress_pct, it["dx"], color=col, linewidth=2.0, zorder=5)
            ax_x.fill_between(progress_pct, it["dx"] - it["sem_dx"], it["dx"] + it["sem_dx"], color=col, alpha=0.15, zorder=3)

            # Panel E: Vertical Z
            ax_z.plot(progress_pct, it["dz"], color=col, linewidth=2.0, zorder=5)
            ax_z.fill_between(progress_pct, it["dz"] - it["sem_dz"], it["dz"] + it["sem_dz"], color=col, alpha=0.15, zorder=3)

            # Panel F: Forward Y
            ax_y.plot(progress_pct, it["dy"], color=col, linewidth=2.0, zorder=5)
            ax_y.fill_between(progress_pct, it["dy"] - it["sem_dy"], it["dy"] + it["sem_dy"], color=col, alpha=0.15, zorder=3)

    ax_x.set_xlabel(prog_axis_name, fontsize=8.5, labelpad=5)
    ax_x.set_ylabel("Lateral Shift ΔX (mm)", fontsize=8.5, labelpad=5)
    title_d = "D. Lateral Deviation ΔX (Left < 0 < Right)"
    ax_x.set_title(title_d, fontsize=10.0, fontweight="bold", loc="left")

    ax_z.set_xlabel(prog_axis_name, fontsize=8.5, labelpad=5)
    ax_z.set_ylabel("Vertical Shift ΔZ (mm)", fontsize=8.5, labelpad=5)
    title_e = "E. Vertical Deviation ΔZ (Dip < 0 < Lift)"
    ax_z.set_title(title_e, fontsize=10.0, fontweight="bold", loc="left")

    ax_y.set_xlabel(prog_axis_name, fontsize=8.5, labelpad=5)
    ax_y.set_ylabel("Forward Shift ΔY (mm)", fontsize=8.5, labelpad=5)
    title_f = "F. Forward Deviation ΔY (Lag < 0 < Lead)"
    ax_y.set_title(title_f, fontsize=10.0, fontweight="bold", loc="left")

    # Legend markers for Start (0%) and End (100%)
    start_handle = plt.Line2D([0], [0], color="#2ecc71", linestyle="none", marker="o", markersize=6.5, markeredgecolor="#ffffff", markeredgewidth=1.0)
    end_handle = plt.Line2D([0], [0], color="#34495e", linestyle="none", marker="D", markersize=6.5, markeredgecolor="#ffffff", markeredgewidth=1.0)
    legend_handles.extend([start_handle, end_handle])
    start_lbl = "Stim Onset / t = 0 ms (0% Depth)" if is_y_mode else "Stim Onset / t = 0 ms (0% Progress)"
    end_lbl = "Touchscreen Endpoint (100% Depth)" if is_y_mode else "Touchscreen Endpoint (100% Progress)"
    legend_labels.extend([start_lbl, end_lbl])

    ncol = min(8, int(np.ceil(len(legend_labels) / 2))) if len(legend_labels) > 8 else len(legend_labels)
    fig.legend(
        legend_handles, legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.968),
        ncol=ncol, frameon=True, framealpha=0.95, fontsize=8.5,
    )

    norm_type_str = "Forward-Y Resampled" if is_y_mode else "Arc-Length Resampled"
    suptitle_text = f"{norm_type_str} Kinematics, Perturbations & Spatial Variability (t = 0 ms to Endpoint) — Target {target} ({stat_label})"
    if fixed_params_summary:
        suptitle_text += f"\n[{fixed_params_summary}]"
    fig.suptitle(suptitle_text, fontsize=11.5, fontweight="bold", y=0.995)

    fig.subplots_adjust(top=0.86, bottom=0.07, left=0.06, right=0.96)

    if save_path:
        out_file = save_path.with_suffix(f".{FIG_FORMAT.lower()}")
        fig.savefig(out_file, format=FIG_FORMAT.lower(), dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved {norm_type_str} Trajectories figure: {out_file.name}")
        plt.close(fig)

    return fig


def plot_forward_y_normalization_validation_figure(
    conditions: Sequence[ReachCondition],
    target: str = "A",
    target_info: Optional[Dict[str, Any]] = None,
    fixed_params_summary: str = "",
    save_path: Optional[Path] = None,
    analysis_window: Optional[Tuple[float, float]] = None,
) -> Optional[plt.Figure]:
    """
    Diagnostic & Validation Figure for Forward-Y Maximum Displacement Normalization:
    Directly tests and verifies the operational assumptions and recommendations of Section 3.6:
        Panel A: Individual Y(t) trajectories with marked peak extension samples (k_max),
                 separating outbound forward reach (solid) from post-touch dwell/retraction (dotted),
                 relative to the physical touchscreen target position (Y_target).
        Panel B: Distribution of peak arrival times t(k_max) relative to the analysis window boundary,
                 confirming peak forward extension is achieved well before the window cutoff.
        Panel C: Distribution of maximum forward excursions Delta Y_max = Y_max - Y_start and
                 verification of the Delta Y_max > 10 mm and k_max >= 3 retention criteria.
        Panel D: Pre- vs. Post-normalization comparison: Raw Y(t) showing speed/duration dispersion
                 vs. Normalized Y(p) on the matched 0% to 100% progress axis.
        Panel E: Landmark Congruence: Distribution of peak forward Y extension (Y_max) compared
                 against the calibrated physical touchscreen target plane.
        Panel F: Scientific Utility: Lateral Veer over elapsed time X(t) vs. Matched-depth X(p),
                 showing how progress normalization reveals reach geometry independent of speed.
    """
    if not conditions:
        return None

    cond_list = sorted(conditions, key=lambda c: (0 if c.is_control else 1, c.cond_num))
    stim_color_pool = [
        "#1f77b4",  # Steel blue (1p)
        "#ff7f0e",  # Amber orange (2p)
        "#2ca02c",  # Emerald green (4p)
        "#9467bd",  # Purple (8p) - distinct from crimson control
        "#17becf",  # Teal/Cyan (20p)
        "#8c564b",  # Brown (40p)
        "#e377c2",  # Magenta/Pink
        "#bcbd22",  # Olive
        "#34495e",  # Slate navy
    ]

    cond_colors = {}
    stim_idx = 0
    for c in cond_list:
        if c.is_control:
            cond_colors[c.cond_num] = COLOR_CTRL
        else:
            cond_colors[c.cond_num] = stim_color_pool[stim_idx % len(stim_color_pool)]
            stim_idx += 1

    # Extract per-trial trajectory parameters
    trial_data = []
    for c in cond_list:
        n_trials = c.n_trials
        col = cond_colors[c.cond_num]
        base_lbl = getattr(c, "short_label", None) or re.sub(r"\s*\(n=\d+\)", "", c.label).strip()

        for i in range(n_trials):
            pts = np.column_stack([c.x[i], c.y[i], c.z[i]])
            valid = np.all(np.isfinite(pts), axis=1)
            if valid.sum() < 3:
                continue

            t_valid = c.time[valid]
            y_valid = pts[valid, 1]
            x_valid = pts[valid, 0]
            z_valid = pts[valid, 2]

            y_start = float(y_valid[0])
            # Physical touchscreen arrival cannot occur before 75 ms
            t_candidates = (t_valid >= 75.0)
            if np.any(t_candidates):
                k_sub = np.where(t_candidates)[0]
                k_max = int(k_sub[np.argmax(y_valid[k_sub])])
            else:
                k_max = int(np.argmax(y_valid))
            y_max = float(y_valid[k_max])
            t_max = float(t_valid[k_max])
            delta_y = y_max - y_start
            is_retained = bool((t_max >= 75.0) and (k_max >= 2))

            trial_data.append({
                "cond_num": c.cond_num,
                "is_control": c.is_control,
                "label": base_lbl,
                "color": col,
                "t": t_valid,
                "x": x_valid,
                "y": y_valid,
                "z": z_valid,
                "y_start": y_start,
                "k_max": k_max,
                "y_max": y_max,
                "t_max": t_max,
                "delta_y": delta_y,
                "is_retained": is_retained,
            })

    if not trial_data:
        return None

    # Calculate resampled trajectories for all conditions
    resampled_by_cond = {}
    for c in cond_list:
        rx, ry, rz = resample_trajectory_by_forward_y(c.x, c.y, c.z, time=c.time, n_progress=100)
        valid_m = np.all(np.isfinite(rx) & np.isfinite(ry) & np.isfinite(rz), axis=1)
        resampled_by_cond[c.cond_num] = {
            "rx": rx[valid_m],
            "ry": ry[valid_m],
            "rz": rz[valid_m],
            "mx": np.nanmedian(rx[valid_m], axis=0) if np.any(valid_m) else None,
            "my": np.nanmedian(ry[valid_m], axis=0) if np.any(valid_m) else None,
            "mz": np.nanmedian(rz[valid_m], axis=0) if np.any(valid_m) else None,
        }

    # Target info resolution
    target_y = None
    target_d = TARGET_DIAMETER_MM
    if target_info and "target_y" in target_info and np.isfinite(target_info["target_y"]):
        target_y = float(target_info["target_y"])
        target_d = float(target_info.get("target_diameter_mm", TARGET_DIAMETER_MM))
    else:
        # Fall back to median control y_max as operational screen estimate
        ctrl_ymaxs = [td["y_max"] for td in trial_data if td["is_control"] and td["is_retained"]]
        if ctrl_ymaxs:
            target_y = float(np.median(ctrl_ymaxs))

    all_t_max = [td["t_max"] for td in trial_data if td["is_retained"]]
    all_delta_y = [td["delta_y"] for td in trial_data if td["is_retained"]]
    all_y_max = [td["y_max"] for td in trial_data if td["is_retained"]]
    all_y_start = [td["y_start"] for td in trial_data if td["is_retained"]]
    median_y_start = float(np.median(all_y_start)) if all_y_start else 45.0

    t_end_window = float(conditions[0].time[-1]) if len(conditions[0].time) > 0 else 350.0
    if analysis_window is not None:
        t_end_window = float(analysis_window[1])

    # Figure creation
    fig = plt.figure(figsize=(16.8, 10.5), dpi=FIG_DPI)
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    legend_handles = []
    legend_labels = []

    # -------------------------------------------------------------------------
    # Panel A: Individual Forward Y(t) & Maximum-Y Endpoints
    # -------------------------------------------------------------------------
    for c in cond_list:
        c_trials = [td for td in trial_data if td["cond_num"] == c.cond_num and td["is_retained"]]
        if not c_trials:
            continue
        col = cond_colors[c.cond_num]
        base_lbl = c_trials[0]["label"]

        # Plot individual reach traces
        n_show = min(len(c_trials), 30)
        step = max(1, len(c_trials) // n_show)
        for idx in range(0, len(c_trials), step):
            td = c_trials[idx]
            km = td["k_max"]
            # Outbound forward reach (solid)
            ax_a.plot(td["t"][:km + 1], td["y"][:km + 1], color=col, alpha=0.28, linewidth=0.9, zorder=3)
            # Post-peak retraction/dwell (dotted, faded)
            if km < len(td["t"]) - 1:
                ax_a.plot(td["t"][km:], td["y"][km:], color=col, alpha=0.12, linewidth=0.7, linestyle=":", zorder=2)
            # Peak dot
            ax_a.scatter([td["t_max"]], [td["y_max"]], color=col, s=26, alpha=0.65, edgecolors="none", zorder=5)

        h = plt.Line2D([0], [0], color=col, linestyle="-", linewidth=2.4, marker="o", markersize=6)
        legend_handles.append(h)
        n_str = f" (n={len(c_trials)})" if SHOW_TRIAL_COUNTS else ""
        legend_labels.append(f"{base_lbl}{n_str}")

    if target_y is not None:
        ax_a.axhline(target_y, color="#2c3e50", linestyle="--", linewidth=1.5, alpha=0.85, zorder=6)
        ax_a.text(ax_a.get_xlim()[0] + 5, target_y + 1.5, f"Touchscreen Plane ({target_y:.1f} mm)",
                  fontsize=8.0, color="#2c3e50", fontweight="bold", zorder=7)

    ax_a.axhline(median_y_start, color="#7f8c8d", linestyle=":", linewidth=1.2, alpha=0.75, zorder=2)
    ax_a.axvline(0.0, color="#95a5a6", linestyle=":", linewidth=1.1, zorder=2)
    ax_a.set_title("A. Forward Y(t) & Maximum-Y Endpoints", fontsize=10.5, fontweight="bold", loc="left")
    ax_a.set_xlabel("Time relative to IR cross (ms)", fontsize=9.0)
    ax_a.set_ylabel("Forward Position Y (mm)", fontsize=9.0)
    ax_a.grid(True, linestyle="--", alpha=0.35)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel B: Touchscreen Arrival Time vs. Condition (Time vs. Pulse)
    # -------------------------------------------------------------------------
    cond_positions_b = []
    cond_labels_b = []
    ctrl_t_vals = [td["t_max"] for td in trial_data if td["is_control"] and td["is_retained"]]
    ctrl_med_t = float(np.median(ctrl_t_vals)) if ctrl_t_vals else None

    for idx_c, c in enumerate(cond_list):
        c_t = [td["t_max"] for td in trial_data if td["cond_num"] == c.cond_num and td["is_retained"]]
        if not c_t:
            continue
        col = cond_colors[c.cond_num]
        x_pos = idx_c + 1
        cond_positions_b.append(x_pos)
        base_lbl = getattr(c, "short_label", None) or re.sub(r"\s*\(n=\d+\)", "", c.label).strip()
        cond_labels_b.append(base_lbl)

        # Condition Boxplot
        ax_b.boxplot(
            c_t, positions=[x_pos], widths=0.42, patch_artist=True,
            boxprops=dict(facecolor=col, alpha=0.35, color=col, linewidth=1.3),
            medianprops=dict(color="#1b2631", linewidth=2.0),
            whiskerprops=dict(color=col, linewidth=1.2),
            capprops=dict(color=col, linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, color=col, alpha=0.5),
        )
        jitter = np.random.normal(0, 0.05, len(c_t))
        ax_b.scatter(x_pos + jitter, c_t, color=col, s=24, alpha=0.7, edgecolors="#1b2631", linewidths=0.5, zorder=5)

    if ctrl_med_t is not None:
        ax_b.axhline(ctrl_med_t, color="#2c3e50", linestyle="--", linewidth=1.4, alpha=0.85, zorder=2)
        ax_b.text(0.5, ctrl_med_t + 5.0, f"Ctrl Median ({ctrl_med_t:.0f} ms)", fontsize=7.5, color="#2c3e50", fontweight="bold")
    ax_b.axhline(t_end_window, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.85, zorder=2)
    ax_b.text(0.5, t_end_window - 14.0, f"Window Cutoff ({int(t_end_window)} ms)", fontsize=7.5, color="#c0392b", fontweight="bold")

    med_t = float(np.median(all_t_max)) if all_t_max else 0.0
    p5_t = float(np.percentile(all_t_max, 5)) if all_t_max else 0.0
    p95_t = float(np.percentile(all_t_max, 95)) if all_t_max else 0.0
    badge_text_b = (
        f"Time vs. Pulse Metric\n"
        f"Overall Median: {med_t:.0f} ms\n"
        f"90% Range: [{p5_t:.0f}, {p95_t:.0f}] ms"
    )
    ax_b.text(0.04, 0.95, badge_text_b, transform=ax_b.transAxes, fontsize=8.0, va="top",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9f9", edgecolor="#bdc3c7", alpha=0.92))

    ax_b.set_xticks(cond_positions_b)
    ax_b.set_xticklabels(cond_labels_b, rotation=25, ha="right", fontsize=8.0)
    ax_b.set_title("B. Touchscreen Arrival Time vs. Condition (Time vs. Pulse)", fontsize=10.5, fontweight="bold", loc="left")
    ax_b.set_xlabel("Condition / Pulse Count", fontsize=9.0)
    ax_b.set_ylabel("Arrival Time t_max (ms)", fontsize=9.0)
    ax_b.grid(True, linestyle="--", alpha=0.35)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel C: Maximum Forward Excursion Delta Y_max & Retention Validation
    # -------------------------------------------------------------------------
    n_total = len(trial_data)
    n_retained = sum(td["is_retained"] for td in trial_data)
    n_rejected = n_total - n_retained
    pct_retained = 100.0 * n_retained / n_total if n_total > 0 else 0.0
    mean_delta = float(np.mean(all_delta_y)) if all_delta_y else 0.0
    std_delta = float(np.std(all_delta_y)) if all_delta_y else 0.0

    bins_c = np.linspace(0.0, max(all_delta_y + [200.0]), 25)
    for c in cond_list:
        c_dy = [td["delta_y"] for td in trial_data if td["cond_num"] == c.cond_num and td["is_retained"]]
        if c_dy:
            col = cond_colors[c.cond_num]
            ax_c.hist(c_dy, bins=bins_c, color=col, alpha=0.45, edgecolor=col, linewidth=1.0)
            jitter = np.random.normal(0, 0.25, len(c_dy))
            ax_c.scatter(c_dy, -0.8 + jitter, color=col, s=18, alpha=0.6, edgecolors="none")

    badge_text_c = (
        f"Analyzed Reaches: {n_total}\n"
        f"Retained (t_peak >= 75ms & k>=2): {n_retained} ({pct_retained:.1f}%)\n"
        f"Excluded / Incomplete: {n_rejected}\n"
        f"Mean Forward Excursion: {mean_delta:.1f} ± {std_delta:.1f} mm"
    )
    ax_c.text(0.04, 0.95, badge_text_c, transform=ax_c.transAxes, fontsize=8.0, va="top",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9f9", edgecolor="#bdc3c7", alpha=0.92))

    ax_c.set_title("C. Excursion ΔY_max & Retention Criteria", fontsize=10.5, fontweight="bold", loc="left")
    ax_c.set_xlabel("Maximum Forward Excursion ΔY_max (mm)", fontsize=9.0)
    ax_c.set_ylabel("Number of Reaches", fontsize=9.0)
    ax_c.grid(True, linestyle="--", alpha=0.35)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel D: Normalized Forward Y Profiles: Y(p) vs. Progress (0% to 100%)
    # -------------------------------------------------------------------------
    progress_grid = np.linspace(0.0, 100.0, 100)
    for c in cond_list:
        res = resampled_by_cond.get(c.cond_num)
        if not res or res["my"] is None:
            continue
        col = cond_colors[c.cond_num]
        is_ctrl = c.is_control
        ry = res["ry"]

        # Individual resampled reach profiles
        n_show = min(len(ry), 30)
        step = max(1, len(ry) // n_show)
        for i in range(0, len(ry), step):
            ax_d.plot(progress_grid, ry[i], color=col, alpha=0.18, linewidth=0.7, zorder=2)

        # Median trajectory
        lw = 2.8 if is_ctrl else 2.2
        ax_d.plot(progress_grid, res["my"], color=col, linestyle="-", linewidth=lw, label=c.label, zorder=5)

    if target_y is not None:
        ax_d.axhline(target_y, color="#2c3e50", linestyle="--", linewidth=1.4, alpha=0.8, zorder=6)
        ax_d.text(5.0, target_y + 1.2, f"Touchscreen Plane ({target_y:.1f} mm)", fontsize=7.5, color="#2c3e50", fontweight="bold")

    ax_d.set_title("D. Normalized Forward Y Profile: Y(p) vs. Progress", fontsize=10.5, fontweight="bold", loc="left")
    ax_d.set_xlabel("Normalized Forward Reach Progress p (%)", fontsize=9.0)
    ax_d.set_ylabel("Forward Position Y (mm)", fontsize=9.0)
    ax_d.set_xlim(-2.0, 102.0)
    ax_d.grid(True, linestyle="--", alpha=0.35)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel E: Physical Landmark Congruence: Y_max vs. Touchscreen Target
    # -------------------------------------------------------------------------
    cond_positions = []
    cond_labels_e = []
    for idx_c, c in enumerate(cond_list):
        c_ymax = [td["y_max"] for td in trial_data if td["cond_num"] == c.cond_num and td["is_retained"]]
        if not c_ymax:
            continue
        col = cond_colors[c.cond_num]
        x_pos = idx_c + 1
        cond_positions.append(x_pos)
        base_lbl = getattr(c, "short_label", None) or re.sub(r"\s*\(n=\d+\)", "", c.label).strip()
        cond_labels_e.append(base_lbl)

        # Boxplot
        ax_e.boxplot(
            c_ymax, positions=[x_pos], widths=0.42, patch_artist=True,
            boxprops=dict(facecolor=col, alpha=0.35, color=col, linewidth=1.3),
            medianprops=dict(color="#1b2631", linewidth=2.0),
            whiskerprops=dict(color=col, linewidth=1.2),
            capprops=dict(color=col, linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, color=col, alpha=0.5),
        )
        jitter = np.random.normal(0, 0.05, len(c_ymax))
        ax_e.scatter(x_pos + jitter, c_ymax, color=col, s=24, alpha=0.7, edgecolors="#1b2631", linewidths=0.5, zorder=5)

    if target_y is not None:
        tgt_rad = target_d / 2.0
        ax_e.axhspan(target_y - tgt_rad, target_y + tgt_rad, color="#eaeded", alpha=0.75, zorder=1)
        ax_e.axhline(target_y, color="#2c3e50", linestyle="--", linewidth=1.5, alpha=0.85, zorder=2)
        ax_e.text(0.5, target_y + tgt_rad + 0.8, f"Target Span ({target_d:.0f} mm)", fontsize=7.5, color="#5d6d7e", ha="left")

        all_resids = [ym - target_y for ym in all_y_max]
        mean_res = float(np.mean(all_resids)) if all_resids else 0.0
        std_res = float(np.std(all_resids)) if all_resids else 0.0
        badge_text_e = f"Touchscreen Y: {target_y:.1f} mm\nMean Residual (Y_max - Y_tgt): {mean_res:+.1f} ± {std_res:.1f} mm"
        ax_e.text(0.04, 0.06, badge_text_e, transform=ax_e.transAxes, fontsize=8.0, va="bottom",
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9f9", edgecolor="#bdc3c7", alpha=0.92))

    ax_e.set_xticks(cond_positions)
    ax_e.set_xticklabels(cond_labels_e, rotation=25, ha="right", fontsize=8.0)
    ax_e.set_title("E. Physical Congruence: Y_max vs. Target Plane", fontsize=10.5, fontweight="bold", loc="left")
    ax_e.set_ylabel("Peak Forward Coordinate Y_max (mm)", fontsize=9.0)
    ax_e.grid(True, linestyle="--", alpha=0.35)
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel F: Scientific Utility: Lateral Veer X(p) at Matched Forward Depth
    # -------------------------------------------------------------------------
    for c in cond_list:
        res = resampled_by_cond.get(c.cond_num)
        if not res or res["mx"] is None:
            continue
        col = cond_colors[c.cond_num]
        is_ctrl = c.is_control
        rx = res["rx"]

        # Traces
        n_show = min(len(rx), 25)
        step = max(1, len(rx) // n_show)
        for i in range(0, len(rx), step):
            ax_f.plot(progress_grid, rx[i], color=col, alpha=0.14, linewidth=0.6, zorder=2)

        lw = 2.8 if is_ctrl else 2.2
        ax_f.plot(progress_grid, res["mx"], color=col, linestyle="-", linewidth=lw, label=c.label, zorder=5)

    ax_f.set_title("F. Lateral Veer X(p) at Matched Depth", fontsize=10.5, fontweight="bold", loc="left")
    ax_f.set_xlabel("Normalized Forward Reach Progress p (%)", fontsize=9.0)
    ax_f.set_ylabel("Lateral Position X (mm)", fontsize=9.0)
    ax_f.set_xlim(-2.0, 102.0)
    ax_f.grid(True, linestyle="--", alpha=0.35)
    ax_f.spines["top"].set_visible(False)
    ax_f.spines["right"].set_visible(False)

    # Unified Header Legend
    ncol = min(8, int(np.ceil(len(legend_labels) / 2))) if len(legend_labels) > 8 else len(legend_labels)
    fig.legend(
        legend_handles, legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.962),
        ncol=ncol, frameon=True, framealpha=0.95, fontsize=8.5,
    )

    time_str = f"{int(conditions[0].time[0])}–{int(conditions[0].time[-1])} ms"
    suptitle_text = f"Forward-Y Normalization Methodological Validation — Target {target} ({time_str})"
    if fixed_params_summary:
        suptitle_text += f"\n[{fixed_params_summary}]"
    fig.suptitle(suptitle_text, fontsize=12.0, fontweight="bold", y=0.993)

    fig.subplots_adjust(top=0.90 if len(legend_labels) > 8 else 0.92, bottom=0.08, left=0.06, right=0.96)

    if save_path:
        out_file = save_path.with_suffix(f".{FIG_FORMAT.lower()}")
        fig.savefig(out_file, format=FIG_FORMAT.lower(), dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Forward-Y Validation figure: {out_file.name}")
        plt.close(fig)

    return fig


def plot_kinematic_dose_response_figure(
    conditions: List[ReachCondition],
    dir_results: Dict[int, Dict[str, Any]],
    target: str = "A",
    fixed_params_summary: str = "",
    save_path: Optional[Path] = None,
    time_window: Tuple[float, float] = (0.0, 500.0),
) -> Optional[plt.Figure]:
    """
    Kinematic Dose-Response & Directional RSA Decomposition Figure (4 Panels):
        Panel A: Touchscreen Arrival Time vs. Pulse Count (Time vs. Pulse)
        Panel B: Lateral Steering Deviation (ΔX) vs. Pulse Count [Left < 0 < Right]
        Panel C: Vertical Deflection (ΔZ) vs. Pulse Count [Dip < 0 < Lift]
        Panel D: Directional RSA Separation (Lateral X, Vertical Z, 3D) & Net 3D Deflection
    """
    if not conditions:
        return None

    # Order conditions: Control first, then stimulation conditions in order
    ctrl_conds = [c for c in conditions if c.is_control]
    stim_conds = [c for c in conditions if not c.is_control]
    cond_list = ctrl_conds + stim_conds

    stim_color_pool = [
        "#1f77b4",  # Steel blue (1p)
        "#ff7f0e",  # Amber orange (2p)
        "#2ca02c",  # Emerald green (4p)
        "#9467bd",  # Purple (8p) - distinct from crimson control
        "#17becf",  # Teal/Cyan (20p)
        "#8c564b",  # Brown (40p)
        "#e377c2",  # Magenta/Pink
        "#bcbd22",  # Olive
        "#34495e",  # Slate navy
    ]
    cond_colors = {}
    stim_idx = 0
    for c in cond_list:
        if c.is_control:
            cond_colors[c.cond_num] = COLOR_CTRL if "COLOR_CTRL" in globals() else "#d32f2f"
        else:
            cond_colors[c.cond_num] = stim_color_pool[stim_idx % len(stim_color_pool)]
            stim_idx += 1

    fig = plt.figure(figsize=(16.5, 12.0), dpi=FIG_DPI)
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    legend_handles = []
    legend_labels = []

    # -------------------------------------------------------------------------
    # Panel A: Touchscreen Arrival Time vs. Pulse Count (Time vs. Pulse)
    # -------------------------------------------------------------------------
    cond_positions = []
    cond_labels = []
    cond_medians_t = []
    ctrl_t_arr = None

    for idx_c, c in enumerate(cond_list):
        res = dir_results.get(c.cond_num, {})
        use_mask = res.get("is_completed_reach", np.ones(c.n_trials, dtype=bool))
        raw_t = res.get("trial_t_end", np.full(c.n_trials, np.nan))
        valid_t = raw_t[use_mask] if np.any(use_mask) else raw_t
        valid_t = valid_t[np.isfinite(valid_t) & (valid_t >= 75.0)]

        if len(valid_t) == 0:
            valid_t = np.array([c.time[-1]])

        if c.is_control:
            ctrl_t_arr = valid_t

        x_pos = idx_c + 1
        cond_positions.append(x_pos)
        base_lbl = getattr(c, "short_label", None) or re.sub(r"\s*\(n=\d+\)", "", c.label).strip()
        cond_labels.append(base_lbl)
        col = cond_colors[c.cond_num]

        # Top unified legend entry
        n_str = f" (n={c.n_trials})" if SHOW_TRIAL_COUNTS else ""
        h = plt.Line2D([0], [0], color=col, linestyle="-", linewidth=2.4, marker="o", markersize=6.5)
        legend_handles.append(h)
        legend_labels.append(f"{base_lbl}{n_str}")

        # Boxplot with semi-transparent fill
        ax_a.boxplot(
            valid_t, positions=[x_pos], widths=0.42, patch_artist=True,
            boxprops=dict(facecolor=col, alpha=0.32, color=col, linewidth=1.3),
            medianprops=dict(color="#1b2631", linewidth=2.2),
            whiskerprops=dict(color=col, linewidth=1.2),
            capprops=dict(color=col, linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, color=col, alpha=0.5),
        )
        jitter = np.random.normal(0, 0.05, len(valid_t))
        ax_a.scatter(x_pos + jitter, valid_t, color=col, s=26, alpha=0.72, edgecolors="#1b2631", linewidths=0.5, zorder=5)

        med_val = float(np.median(valid_t))
        cond_medians_t.append(med_val)

        # Statistical comparison vs Control (Mann-Whitney U test)
        if not c.is_control and ctrl_t_arr is not None and len(ctrl_t_arr) >= 2 and len(valid_t) >= 2:
            try:
                p_mw = stats.mannwhitneyu(valid_t, ctrl_t_arr, alternative="two-sided").pvalue
                if p_mw < 0.001:
                    star = "***"
                elif p_mw < 0.01:
                    star = "**"
                elif p_mw < 0.05:
                    star = "*"
                else:
                    star = "ns"
                if star != "ns":
                    y_star = float(np.max(valid_t)) + 12.0
                    ax_a.text(x_pos, y_star, star, ha="center", va="bottom", fontsize=10.0, fontweight="bold", color="#c0392b")
            except Exception:
                pass

    # Median trend line
    ax_a.plot(cond_positions, cond_medians_t, color="#2c3e50", linestyle="--", linewidth=1.6, alpha=0.80, zorder=4)

    t_end = time_window[1] if time_window else 500.0
    if ctrl_t_arr is not None and len(ctrl_t_arr) > 0:
        c_med = float(np.median(ctrl_t_arr))
        c_q25 = float(np.percentile(ctrl_t_arr, 25))
        c_q75 = float(np.percentile(ctrl_t_arr, 75))
        ax_a.axhspan(c_q25, c_q75, color="#f2f4f4", alpha=0.55, zorder=1)
        ax_a.axhline(c_med, color="#2c3e50", linestyle="--", linewidth=1.3, alpha=0.85, zorder=2)
        ax_a.text(0.55, c_med + 6.0, f"Ctrl Baseline ({c_med:.0f} ms)", fontsize=8.0, color="#2c3e50", fontweight="bold")

    ax_a.axhline(t_end, color="#e74c3c", linestyle="--", linewidth=1.4, alpha=0.85, zorder=2)
    ax_a.text(len(cond_list) + 0.35, t_end, f"Window Cutoff ({int(t_end)} ms)", fontsize=7.5, color="#c0392b", fontweight="bold", va="center", ha="right")

    # Time vs Pulse dose-response summary badge (positioned cleanly below top boundary)
    if len(cond_medians_t) > 1:
        c_t0 = cond_medians_t[0]
        max_t = max(cond_medians_t)
        delta_max_t = max_t - c_t0
        badge_a = f"Time vs. Pulse Metric:\n• Ctrl Arrival: {c_t0:.0f} ms\n• Max Delay: {delta_max_t:+.0f} ms\n• Dose Prolongation: Yes"
        ax_a.text(0.04, 0.88, badge_a, transform=ax_a.transAxes, fontsize=8.0, va="top",
                  bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8f9fa", edgecolor="#cfd8dc", alpha=0.95), zorder=8)

    # Ample headroom so cutoff line and dots do not touch top edge
    min_arr = min(min([np.min(td) for td in [res.get("trial_t_end", [140]) for res in dir_results.values()] if len(td) > 0 and np.any(np.isfinite(td))], default=120), 120)
    ax_a.set_ylim(min(120.0, float(min_arr) - 20.0), t_end + 45.0)

    ax_a.set_xticks(cond_positions)
    ax_a.set_xticklabels(cond_labels, rotation=22, ha="right", fontsize=8.5)
    ax_a.set_title("A. Touchscreen Arrival Time vs. Pulse Count (Time vs. Pulse)", fontsize=10.5, fontweight="bold", loc="left")
    ax_a.set_ylabel("Arrival Time t_max (ms)", fontsize=9.0)
    ax_a.set_xlabel("Condition / Pulse Count", fontsize=9.0)
    ax_a.grid(True, linestyle="--", alpha=0.35)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel B: Lateral Steering Deviation (ΔX) vs. Pulse Count
    # -------------------------------------------------------------------------
    cond_medians_dx = []
    inflight_dx_list = []
    for idx_c, c in enumerate(cond_list):
        res = dir_results.get(c.cond_num, {})
        use_mask = res.get("is_completed_reach", np.ones(c.n_trials, dtype=bool))
        raw_dx = res.get("end_dx", np.full(c.n_trials, np.nan))
        valid_dx = raw_dx[use_mask] if np.any(use_mask) else raw_dx
        valid_dx = valid_dx[np.isfinite(valid_dx)]
        if len(valid_dx) == 0:
            valid_dx = np.array([0.0])

        x_pos = idx_c + 1
        col = cond_colors[c.cond_num]

        ax_b.boxplot(
            valid_dx, positions=[x_pos], widths=0.42, patch_artist=True,
            boxprops=dict(facecolor=col, alpha=0.32, color=col, linewidth=1.3),
            medianprops=dict(color="#1b2631", linewidth=2.2),
            whiskerprops=dict(color=col, linewidth=1.2),
            capprops=dict(color=col, linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, color=col, alpha=0.5),
        )
        jitter = np.random.normal(0, 0.05, len(valid_dx))
        ax_b.scatter(x_pos + jitter, valid_dx, color=col, s=26, alpha=0.72, edgecolors="#1b2631", linewidths=0.5, zorder=5)

        cond_medians_dx.append(float(np.median(valid_dx)))

        # In-flight lateral mean shift (t >= 75 ms)
        mean_dx_t = res.get("mean_dx", None)
        t_vec = res.get("time", c.time)
        if mean_dx_t is not None and len(mean_dx_t) > 0:
            t_mask = (t_vec >= 75.0) & np.isfinite(mean_dx_t)
            inf_val = float(np.nanmean(mean_dx_t[t_mask])) if np.any(t_mask) else float(np.nanmean(mean_dx_t))
        else:
            inf_val = float(np.median(valid_dx))
        inflight_dx_list.append(inf_val)
        # Prominent distinct diamond marker with white outline
        ax_b.scatter([x_pos], [inf_val], marker="D", s=68, color=col, edgecolors="#1b2631", linewidths=1.4, zorder=7)

        # Significance star vs Control
        p_dx = res.get("p_dx", 1.0)
        if not c.is_control and p_dx < 0.05:
            star = "***" if p_dx < 0.001 else ("**" if p_dx < 0.01 else "*")
            y_s = float(np.max(valid_dx)) + 2.8 if np.median(valid_dx) >= 0 else float(np.min(valid_dx)) - 3.8
            ax_b.text(x_pos, y_s, star, ha="center", va="center", fontsize=10.0, fontweight="bold", color="#1f77b4")

    ax_b.plot(cond_positions, cond_medians_dx, color="#2c3e50", linestyle="--", linewidth=1.6, alpha=0.80, zorder=4)
    ax_b.axhline(0.0, color="#2c3e50", linestyle="-", linewidth=1.2, alpha=0.7, zorder=2)
    ax_b.axhspan(-TARGET_DIAMETER_MM / 2.0, TARGET_DIAMETER_MM / 2.0, color="#eaeded", alpha=0.55, zorder=1)
    ax_b.axhline(TARGET_DIAMETER_MM / 2.0, color="#b0bec5", linestyle=":", linewidth=1.1, alpha=0.8, zorder=2)
    ax_b.axhline(-TARGET_DIAMETER_MM / 2.0, color="#b0bec5", linestyle=":", linewidth=1.1, alpha=0.8, zorder=2)

    # Direction annotations as sleek pill badges
    y_lims_b = ax_b.get_ylim()
    span_b = max(abs(y_lims_b[0]), abs(y_lims_b[1]), 14.0) + 4.0
    ax_b.set_ylim(-span_b, span_b)
    ax_b.text(0.96, 0.93, "Rightward Veer (ΔX > 0) →", transform=ax_b.transAxes, ha="right", va="top",
              fontsize=7.8, color="#1f77b4", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="#ebf5fb", edgecolor="#aed6f1", alpha=0.92))
    ax_b.text(0.96, 0.07, "← Leftward Veer (ΔX < 0)", transform=ax_b.transAxes, ha="right", va="bottom",
              fontsize=7.8, color="#1f77b4", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="#ebf5fb", edgecolor="#aed6f1", alpha=0.92))

    ax_b.set_xticks(cond_positions)
    ax_b.set_xticklabels(cond_labels, rotation=22, ha="right", fontsize=8.5)
    ax_b.set_title("B. Lateral Steering Deviation ΔX vs. Pulse Count", fontsize=10.5, fontweight="bold", loc="left")
    ax_b.set_ylabel("Lateral Deviation ΔX (mm) [Left < 0 < Right]", fontsize=9.0)
    ax_b.set_xlabel("Condition / Pulse Count", fontsize=9.0)
    ax_b.grid(True, linestyle="--", alpha=0.35)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel C: Vertical Deflection (ΔZ) vs. Pulse Count
    # -------------------------------------------------------------------------
    cond_medians_dz = []
    inflight_dz_list = []
    for idx_c, c in enumerate(cond_list):
        res = dir_results.get(c.cond_num, {})
        use_mask = res.get("is_completed_reach", np.ones(c.n_trials, dtype=bool))
        raw_dz = res.get("end_dz", np.full(c.n_trials, np.nan))
        valid_dz = raw_dz[use_mask] if np.any(use_mask) else raw_dz
        valid_dz = valid_dz[np.isfinite(valid_dz)]
        if len(valid_dz) == 0:
            valid_dz = np.array([0.0])

        x_pos = idx_c + 1
        col = cond_colors[c.cond_num]

        ax_c.boxplot(
            valid_dz, positions=[x_pos], widths=0.42, patch_artist=True,
            boxprops=dict(facecolor=col, alpha=0.32, color=col, linewidth=1.3),
            medianprops=dict(color="#1b2631", linewidth=2.2),
            whiskerprops=dict(color=col, linewidth=1.2),
            capprops=dict(color=col, linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, color=col, alpha=0.5),
        )
        jitter = np.random.normal(0, 0.05, len(valid_dz))
        ax_c.scatter(x_pos + jitter, valid_dz, color=col, s=26, alpha=0.72, edgecolors="#1b2631", linewidths=0.5, zorder=5)

        cond_medians_dz.append(float(np.median(valid_dz)))

        # In-flight vertical mean shift (t >= 75 ms)
        mean_dz_t = res.get("mean_dz", None)
        t_vec = res.get("time", c.time)
        if mean_dz_t is not None and len(mean_dz_t) > 0:
            t_mask = (t_vec >= 75.0) & np.isfinite(mean_dz_t)
            inf_val = float(np.nanmean(mean_dz_t[t_mask])) if np.any(t_mask) else float(np.nanmean(mean_dz_t))
        else:
            inf_val = float(np.median(valid_dz))
        inflight_dz_list.append(inf_val)
        # Prominent distinct diamond marker with white outline
        ax_c.scatter([x_pos], [inf_val], marker="D", s=68, color=col, edgecolors="#1b2631", linewidths=1.4, zorder=7)

        # Significance star vs Control
        p_dz = res.get("p_dz", 1.0)
        if not c.is_control and p_dz < 0.05:
            star = "***" if p_dz < 0.001 else ("**" if p_dz < 0.01 else "*")
            y_s = float(np.max(valid_dz)) + 2.8 if np.median(valid_dz) >= 0 else float(np.min(valid_dz)) - 3.8
            ax_c.text(x_pos, y_s, star, ha="center", va="center", fontsize=10.0, fontweight="bold", color="#2ca02c")

    ax_c.plot(cond_positions, cond_medians_dz, color="#2c3e50", linestyle="--", linewidth=1.6, alpha=0.80, zorder=4)
    ax_c.axhline(0.0, color="#2c3e50", linestyle="-", linewidth=1.2, alpha=0.7, zorder=2)
    ax_c.axhspan(-TARGET_DIAMETER_MM / 2.0, TARGET_DIAMETER_MM / 2.0, color="#eaeded", alpha=0.55, zorder=1)
    ax_c.axhline(TARGET_DIAMETER_MM / 2.0, color="#b0bec5", linestyle=":", linewidth=1.1, alpha=0.8, zorder=2)
    ax_c.axhline(-TARGET_DIAMETER_MM / 2.0, color="#b0bec5", linestyle=":", linewidth=1.1, alpha=0.8, zorder=2)

    y_lims_c = ax_c.get_ylim()
    span_c = max(abs(y_lims_c[0]), abs(y_lims_c[1]), 14.0) + 4.0
    ax_c.set_ylim(-span_c, span_c)
    ax_c.text(0.96, 0.93, "Upward Lift (ΔZ > 0) ↑", transform=ax_c.transAxes, ha="right", va="top",
              fontsize=7.8, color="#2ca02c", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="#eafaf1", edgecolor="#a9dfbf", alpha=0.92))
    ax_c.text(0.96, 0.07, "↓ Downward Dip (ΔZ < 0)", transform=ax_c.transAxes, ha="right", va="bottom",
              fontsize=7.8, color="#2ca02c", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.35", facecolor="#eafaf1", edgecolor="#a9dfbf", alpha=0.92))

    ax_c.set_xticks(cond_positions)
    ax_c.set_xticklabels(cond_labels, rotation=22, ha="right", fontsize=8.5)
    ax_c.set_title("C. Vertical Deflection ΔZ vs. Pulse Count", fontsize=10.5, fontweight="bold", loc="left")
    ax_c.set_ylabel("Vertical Deviation ΔZ (mm) [Dip < 0 < Lift]", fontsize=9.0)
    ax_c.set_xlabel("Condition / Pulse Count", fontsize=9.0)
    ax_c.grid(True, linestyle="--", alpha=0.35)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # -------------------------------------------------------------------------
    # Panel D: Directional RSA Separation (S_X, S_Z, S_3D) & Net 3D Deflection
    # -------------------------------------------------------------------------
    try:
        conds_prog = [
            c.slice_to_progress(RSA_LATE_PROGRESS_RANGE, n_progress=RSA_N_PROGRESS_SAMPLES, mode=RSA_PROGRESS_MODE)
            for c in conditions
        ]
        RDM_x, sz_x, _ = build_behavioral_rdm(conds_prog, metric="euclidean", axis="x", align_by_progress=False, aggregation=RSA_DISTANCE_AGGREGATION)
        S_x, _, _ = compute_pairwise_silhouette_matrix(RDM_x, sz_x, n_perm=0)

        RDM_z, sz_z, _ = build_behavioral_rdm(conds_prog, metric="euclidean", axis="z", align_by_progress=False, aggregation=RSA_DISTANCE_AGGREGATION)
        S_z, _, _ = compute_pairwise_silhouette_matrix(RDM_z, sz_z, n_perm=0)

        RDM_3d, sz_3d, _ = build_behavioral_rdm(conds_prog, metric="euclidean", axis="3d", align_by_progress=False, aggregation=RSA_DISTANCE_AGGREGATION)
        S_3d, _, _ = compute_pairwise_silhouette_matrix(RDM_3d, sz_3d, n_perm=0)
    except Exception as e:
        _log(f"Directional RSA computation note: {e}")
        S_x = np.zeros((len(conditions), len(conditions)))
        S_z = np.zeros((len(conditions), len(conditions)))
        S_3d = np.zeros((len(conditions), len(conditions)))

    # Primary axis: Silhouette separation vs Control
    stim_positions = cond_positions[1:] if len(cond_positions) > 1 else cond_positions
    stim_labels = cond_labels[1:] if len(cond_labels) > 1 else cond_labels
    s_x_vals = [float(S_x[0, k]) if k < S_x.shape[0] else 0.0 for k in range(1, len(cond_list))]
    s_z_vals = [float(S_z[0, k]) if k < S_z.shape[0] else 0.0 for k in range(1, len(cond_list))]
    s_3d_vals = [float(S_3d[0, k]) if k < S_3d.shape[0] else 0.0 for k in range(1, len(cond_list))]

    if not s_x_vals:
        s_x_vals = [0.0]
        s_z_vals = [0.0]
        s_3d_vals = [0.0]

    # Twin axis for Net 3D Deflection (mm)
    ax_d_twin = ax_d.twinx()

    net_dnorm_vals = []
    for k in range(1, len(cond_list)):
        c = cond_list[k]
        res = dir_results.get(c.cond_num, {})
        use_mask = res.get("is_completed_reach", np.ones(c.n_trials, dtype=bool))
        raw_norm = res.get("end_norm", np.full(c.n_trials, np.nan))
        valid_norm = raw_norm[use_mask] if np.any(use_mask) else raw_norm
        valid_norm = valid_norm[np.isfinite(valid_norm)]
        net_dnorm_vals.append(float(np.median(valid_norm)) if len(valid_norm) > 0 else 0.0)

    if not net_dnorm_vals:
        net_dnorm_vals = [0.0]

    # Plot Net 3D bars in background without confusing cross-plot lines
    bar_width = 0.38
    b_bars = ax_d_twin.bar(stim_positions, net_dnorm_vals, width=bar_width, color="#bdc3c7", alpha=0.35,
                           edgecolor="#95a5a6", linewidth=0.8, label="Net 3D Deflection ||Δr|| (mm)", zorder=1)
    # Shaded band on twin axis for target radius (0 to 5 mm)
    ax_d_twin.axhspan(0.0, TARGET_DIAMETER_MM / 2.0, color="#eaeded", alpha=0.45, zorder=0)
    ax_d_twin.set_ylabel("Net 3D Deflection ||Δr|| (mm)", fontsize=9.0, color="#7f8c8d")
    ax_d_twin.tick_params(axis="y", labelcolor="#7f8c8d")
    max_dnorm = max(net_dnorm_vals + [TARGET_DIAMETER_MM * 1.5])
    ax_d_twin.set_ylim(0.0, max(max_dnorm, 20.0))

    # Chance level band on Silhouette primary axis (S <= 0)
    ax_d.axhspan(-0.25, 0.0, color="#f8f9fa", alpha=0.65, zorder=1)
    ax_d.axhline(0.0, color="#7f8c8d", linestyle="-", linewidth=1.2, alpha=0.75, zorder=2)

    # Plot Directional RSA curves on primary axis
    h_x, = ax_d.plot(stim_positions, s_x_vals, marker="o", markersize=7.5, linewidth=2.3, color="#1f77b4",
                     label="Lateral-X Silhouette (S_X)", zorder=5)
    h_z, = ax_d.plot(stim_positions, s_z_vals, marker="^", markersize=8.0, linewidth=2.3, color="#2ca02c",
                     label="Vertical-Z Silhouette (S_Z)", zorder=5)
    h_3d, = ax_d.plot(stim_positions, s_3d_vals, marker="s", markersize=6.8, linewidth=1.9, color="#34495e",
                      linestyle="--", label="Isotropic 3D Silhouette (S_3D)", zorder=4)

    # Determine winning axis and annotate peak
    max_sx = max(s_x_vals) if s_x_vals else 0.0
    max_sz = max(s_z_vals) if s_z_vals else 0.0
    max_s3d = max(s_3d_vals) if s_3d_vals else 0.0

    if max_sz > max_sx and max_sz >= 0.15:
        p_idx = int(np.argmax(s_z_vals))
        ax_d.scatter([stim_positions[p_idx]], [s_z_vals[p_idx]], s=120, facecolor="none", edgecolor="#2ca02c", linewidth=2.2, zorder=7)
        ax_d.annotate(f"Peak S_Z = {s_z_vals[p_idx]:+.3f}", xy=(stim_positions[p_idx], s_z_vals[p_idx]),
                      xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8.0, fontweight="bold", color="#2ca02c")
        win_title = "Primary Steering: Vertical Dip"
    elif max_sx > max_sz and max_sx >= 0.15:
        p_idx = int(np.argmax(s_x_vals))
        ax_d.scatter([stim_positions[p_idx]], [s_x_vals[p_idx]], s=120, facecolor="none", edgecolor="#1f77b4", linewidth=2.2, zorder=7)
        ax_d.annotate(f"Peak S_X = {s_x_vals[p_idx]:+.3f}", xy=(stim_positions[p_idx], s_x_vals[p_idx]),
                      xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8.0, fontweight="bold", color="#1f77b4")
        win_title = "Primary Steering: Lateral Veer"
    else:
        win_title = "Directional RSA Decomposition"

    ax_d.set_ylabel("Condition Separation vs. Control (Silhouette S)", fontsize=9.0, color="#1b2631")
    ax_d.set_xlabel("Condition / Pulse Count", fontsize=9.0)
    ax_d.set_xticks(stim_positions)
    ax_d.set_xticklabels(stim_labels, rotation=22, ha="right", fontsize=8.5)

    all_s = s_x_vals + s_z_vals + s_3d_vals
    min_s = min(all_s + [-0.1])
    max_s = max(all_s + [0.3])
    ax_d.set_ylim(min_s - 0.05, max(max_s + 0.12, 0.45))
    ax_d.set_title(f"D. Directional RSA Decomposition ({win_title})", fontsize=10.5, fontweight="bold", loc="left")
    ax_d.grid(True, linestyle="--", alpha=0.35)
    ax_d.spines["top"].set_visible(False)

    badge_d = (
        f"{win_title}:\n"
        f"• Lateral-X Separation: S_X = {max_sx:+.3f}\n"
        f"• Vertical-Z Separation: S_Z = {max_sz:+.3f}\n"
        f"• Isotropic 3D Euclidean: S_3D = {max_s3d:+.3f}\n"
        f"Isolates steering axis from 3D speed/depth noise"
    )
    ax_d.text(0.04, 0.95, badge_d, transform=ax_d.transAxes, fontsize=8.0, va="top",
              bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8f9fa", edgecolor="#cfd8dc", alpha=0.95), zorder=8)

    lines_d = [h_x, h_z, h_3d, b_bars]
    labels_d = ["Lateral-X Silhouette (S_X)", "Vertical-Z Silhouette (S_Z)", "Isotropic 3D Silhouette (S_3D)", "Net 3D Deflection ||Δr||"]
    ax_d.legend(lines_d, labels_d, loc="lower right", framealpha=0.92, fontsize=7.8)

    # Unified Header Legend at Top
    ncol = min(8, len(legend_labels))
    fig.legend(
        legend_handles, legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.965),
        ncol=ncol, frameon=True, framealpha=0.95, fontsize=8.5,
    )

    # Figure Suptitle & Subtitle
    time_str = f"{int(conditions[0].time[0])}–{int(conditions[0].time[-1])} ms"
    suptitle_text = f"Kinematic Dose-Response & Directional Perturbations — Target {target} ({time_str})"
    if fixed_params_summary:
        suptitle_text += f"\n[{fixed_params_summary}]"
    fig.suptitle(suptitle_text, fontsize=12.2, fontweight="bold", y=0.995)

    # Footer annotation
    fig.text(
        0.5, 0.02,
        "◆ Diamond marker = In-flight trajectory centroid (t ≥ 75 ms)   |   Boxplot/Dots = Physical touchscreen arrival endpoints   |   * p<0.05, ** p<0.01, *** p<0.001 vs. Control (Mann-Whitney U)",
        ha="center", va="center", fontsize=8.0, color="#5d6d7e", style="italic",
    )

    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.07, right=0.93)

    if save_path:
        out_file = save_path.with_suffix(f".{FIG_FORMAT.lower()}")
        fig.savefig(out_file, format=FIG_FORMAT.lower(), dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Kinematic Dose-Response figure: {out_file.name}")
        plt.close(fig)

    return fig


# =============================================================================
# Main Execution Pipeline
# =============================================================================

def run_behavioral_rsa_for_target(
    session: str,
    target: str,
    peri_root: Path,
    dlc_3d_dir: Optional[Path] = None,
    save_dir: Optional[Path] = None,
    compare_conditions: Optional[Sequence[int]] = None,
    control_br_indices: Optional[Sequence[int]] = None,
    time_window: Tuple[float, float] = DEFAULT_PLOT_WINDOW_MS,
    rsa_early_window: Tuple[float, float] = RSA_EARLY_WINDOW_MS,
    rsa_late_window: Tuple[float, float] = RSA_LATE_WINDOW_MS,
    rsa_early_progress: Tuple[float, float] = RSA_EARLY_PROGRESS_RANGE,
    rsa_late_progress: Tuple[float, float] = RSA_LATE_PROGRESS_RANGE,
    keypoint: str = DEFAULT_KEYPOINT,
    align_task_frame: bool = ALIGN_TO_TASK_FRAME,
) -> Dict[str, Any]:
    """Execute complete Behavioral RSA, Variability, and Directional analysis for one target."""
    target_str = target.upper()
    _log(f"\n{'='*70}\n[run] Session={session} | Target={target_str} | Trajectory Window={time_window[0]}-{time_window[1]} ms\n{'='*70}")

    if save_dir is None:
        save_dir = OUT_BASE / "figures" / "behavioral_rsa" if OUT_BASE else Path.cwd() / "results" / "behavioral_rsa"
    ensure_dir(save_dir)

    # 1. Load conditions across the full reach plotting window
    conditions = load_conditions_for_target(
        session=session,
        target=target_str,
        peri_root=peri_root,
        dlc_3d_dir=dlc_3d_dir,
        compare_conditions=compare_conditions,
        control_br_indices=control_br_indices,
        keypoint=keypoint,
        time_window=time_window,
        align_task_frame=align_task_frame,
    )

    if len(conditions) < 2:
        _log(f"Insufficient conditions ({len(conditions)}) for Target {target_str}; skipping.")
        return {"conditions": conditions}

    cond_nums_all = []
    for c in conditions:
        comb_brs = getattr(c, "combined_br_indices", None)
        if comb_brs:
            cond_nums_all.extend(comb_brs)
        else:
            cond_nums_all.append(c.cond_num)
    cond_tag = "_".join(map(str, sorted(set(cond_nums_all))))
    fixed_summary = getattr(conditions[0], "fixed_params_summary", "") if conditions else ""

    agg_disp = "P90" if str(RSA_DISTANCE_AGGREGATION).lower() in ("p90", "q90") else str(RSA_DISTANCE_AGGREGATION).capitalize()

    if RSA_ALIGN_BY_PROGRESS:
        conds_early = [
            c.slice_to_progress(rsa_early_progress, n_progress=RSA_N_PROGRESS_SAMPLES, mode=RSA_PROGRESS_MODE)
            for c in conditions
        ]
        conds_late = [
            c.slice_to_progress(rsa_late_progress, n_progress=RSA_N_PROGRESS_SAMPLES, mode=RSA_PROGRESS_MODE)
            for c in conditions
        ]
        is_prog_mode = True
        early_tag = f"{int(round(rsa_early_progress[0] * 100))}–{int(round(rsa_early_progress[1] * 100))}% Depth"
        late_tag = f"{int(round(rsa_late_progress[0] * 100))}–{int(round(rsa_late_progress[1] * 100))}% Depth"
        rdm_metric_lbl = f"Forward-Y 3D Distance ({agg_disp}) (mm)"
    else:
        conds_early = [c.slice_to_window(rsa_early_window) for c in conditions]
        conds_late = [c.slice_to_window(rsa_late_window) for c in conditions]
        is_prog_mode = False
        early_tag = f"{int(rsa_early_window[0])}–{int(rsa_early_window[1])} ms"
        late_tag = f"{int(rsa_late_window[0])}–{int(rsa_late_window[1])} ms"
        rdm_metric_lbl = f"3D Distance ({agg_disp}) (mm)"

    # -------------------------------------------------------------------------
    # 2a. Region 1: Early Window RSA (0-25% Depth or 0-75 ms) — Expected Control-Like
    # -------------------------------------------------------------------------
    RDM_early, block_sizes_early, block_labels_early = build_behavioral_rdm(
        conds_early,
        metric="euclidean",
        align_by_progress=False,
        n_progress_samples=RSA_N_PROGRESS_SAMPLES,
        aggregation=RSA_DISTANCE_AGGREGATION,
    )
    block_short_labels_early = [getattr(c, "short_label", to_short_condition_label(c.label)) for c in conds_early]
    S_early, P_early, Q_early = compute_pairwise_silhouette_matrix(RDM_early, block_sizes_early, n_perm=N_PERMUTATIONS)
    stereotypy_early = compute_within_condition_stereotypy(RDM_early, block_sizes_early)

    # -------------------------------------------------------------------------
    # 2b. Region 2: Late Window RSA (25-100% Depth or 75-350 ms) — Active Perturbation Phase
    # -------------------------------------------------------------------------
    RDM_late, block_sizes_late, block_labels_late = build_behavioral_rdm(
        conds_late,
        metric="euclidean",
        align_by_progress=False,
        n_progress_samples=RSA_N_PROGRESS_SAMPLES,
        aggregation=RSA_DISTANCE_AGGREGATION,
    )
    block_short_labels_late = [getattr(c, "short_label", to_short_condition_label(c.label)) for c in conds_late]
    S_late, P_late, Q_late = compute_pairwise_silhouette_matrix(RDM_late, block_sizes_late, n_perm=N_PERMUTATIONS)
    stereotypy_late = compute_within_condition_stereotypy(RDM_late, block_sizes_late)

    # Directional late epoch separation along lateral X and vertical Z
    try:
        RDM_late_x, _, _ = build_behavioral_rdm(conds_late, metric="euclidean", axis="x", align_by_progress=False, aggregation=RSA_DISTANCE_AGGREGATION)
        S_late_x, _, _ = compute_pairwise_silhouette_matrix(RDM_late_x, block_sizes_late, n_perm=0)
        RDM_late_z, _, _ = build_behavioral_rdm(conds_late, metric="euclidean", axis="z", align_by_progress=False, aggregation=RSA_DISTANCE_AGGREGATION)
        S_late_z, _, _ = compute_pairwise_silhouette_matrix(RDM_late_z, block_sizes_late, n_perm=0)

        _log(f"[behavioral_rsa] Target {target_str} Directional RSA Silhouette vs Control ({late_tag}):")
        for k in range(1, len(conditions)):
            lbl_k = getattr(conditions[k], "short_label", None) or conditions[k].label
            _log(f"   • {lbl_k}: Lateral-X S={S_late_x[0, k]:+.3f} | Vertical-Z S={S_late_z[0, k]:+.3f} | Isotropic-3D S={S_late[0, k]:+.3f}")
    except Exception as e:
        _log(f"[behavioral_rsa] Directional RSA logging note: {e}")
        S_late_x = np.zeros((len(conditions), len(conditions)))
        S_late_z = np.zeros((len(conditions), len(conditions)))

    # -------------------------------------------------------------------------
    # 2c. Consolidated Dual-Window Comparison Figure
    # -------------------------------------------------------------------------
    out_rsa_dual = save_dir / f"behavioral_rsa_dual_window_comparison_Target_{target_str}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
    plot_dual_window_rsa_comparison_figure(
        RDM_early=RDM_early,
        S_early=S_early,
        Q_early=Q_early,
        within_stereotypy_early=stereotypy_early,
        RDM_late=RDM_late,
        S_late=S_late,
        Q_late=Q_late,
        within_stereotypy_late=stereotypy_late,
        block_sizes=block_sizes_early,
        block_labels=block_labels_early,
        target=target_str,
        window_early=rsa_early_window,
        window_late=rsa_late_window,
        is_progress_mode=is_prog_mode,
        early_tag=early_tag,
        late_tag=late_tag,
        metric_label=rdm_metric_lbl,
        block_short_labels=block_short_labels_early,
        fixed_params_summary=fixed_summary,
        save_path=out_rsa_dual,
    )

    # -------------------------------------------------------------------------
    # 3. Time-Resolved Trajectory Variability & Control Deflection (0-500 ms)
    # -------------------------------------------------------------------------
    var_results = compute_time_resolved_variability(conditions)
    var_clean_mode = VARIABILITY_PLOT_MODE.lower().strip()
    if var_clean_mode in ("var", "variability"):
        out_var = save_dir / f"trajectory_variability_Target_{target_str}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
    elif var_clean_mode in ("def", "deflection", "dev", "deviation"):
        out_var = save_dir / f"trajectory_deflection_Target_{target_str}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
    else:
        out_var = save_dir / f"trajectory_variability_time_resolved_Target_{target_str}_conds_{cond_tag}.{FIG_FORMAT.lower()}"

    plot_time_resolved_variability_figure(
        var_results=var_results,
        target=target_str,
        fixed_params_summary=fixed_summary,
        save_path=out_var,
        mode=VARIABILITY_PLOT_MODE,
        legend_orientation=VARIABILITY_LEGEND_ORIENTATION,
    )

    # -------------------------------------------------------------------------
    # 4. Multi-Condition Directional Perturbations (Across full movement window)
    # -------------------------------------------------------------------------
    session_landmarks = getattr(conditions[0], "session_landmarks", None) if conditions else None
    dir_results = compute_directional_perturbations(
        conditions=conditions,
        target=target_str,
        session_landmarks=session_landmarks,
        dlc_3d_dir=dlc_3d_dir,
        square_size_mm=SQUARE_SIZE_MM,
        target_diameter_mm=TARGET_DIAMETER_MM,
        stability_threshold_mm=TARGET_STABILITY_THRESHOLD_MM,
    )
    out_dir = save_dir / f"directional_deviations_Target_{target_str}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
    plot_target_directional_deviation_traces(
        dir_results=dir_results,
        target=target_str,
        fixed_params_summary=fixed_summary,
        save_path=out_dir,
    )

    # 4b. Dedicated Touchscreen-Face Endpoint Perturbation & 3D Trajectory Figure
    out_touchscreen = save_dir / f"touchscreen_endpoint_and_trajectory_Target_{target_str}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
    plot_combined_spatial_perturbations_figure(
        dir_results_by_target={target_str: dir_results},
        fixed_params_summary=fixed_summary,
        save_path=out_touchscreen,
        mode=SPATIAL_PLOT_MODE,
        trajectory_stat=TRAJECTORY_STAT,
        ellipse_scale=ELLIPSE_SCALE,
        time_window=time_window,
    )

    # 4c. Dedicated Forward-Y / Arc-Length Resampled Trajectories & Progress Profiles Figure
    is_y_mode = str(RSA_PROGRESS_MODE).lower().strip() in ("forward_y", "y", "depth", "forward")
    mode_prefix = "forward_y" if is_y_mode else "arclength"
    out_resampled = save_dir / f"{mode_prefix}_resampled_trajectories_Target_{target_str}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
    plot_progress_resampled_trajectories_figure(
        conditions=conditions,
        target=target_str,
        fixed_params_summary=fixed_summary,
        save_path=out_resampled,
        resample_window=time_window,
        n_progress_samples=RSA_N_PROGRESS_SAMPLES,
        trajectory_stat=TRAJECTORY_STAT,
        progress_mode=RSA_PROGRESS_MODE,
    )
    # 4d. Dedicated Forward-Y Normalization Methodological Validation Figure (Section 3.6)
    if is_y_mode:
        out_validation = save_dir / f"forward_y_validation_Target_{target_str}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
        tgt_anchor = dir_results.get(list(dir_results.keys())[0], {}).get("target_info") if dir_results else None
        plot_forward_y_normalization_validation_figure(
            conditions=conditions,
            target=target_str,
            target_info=tgt_anchor,
            fixed_params_summary=fixed_summary,
            save_path=out_validation,
            analysis_window=time_window,
        )

    # 4e. Dedicated Kinematic Dose-Response & Directional RSA Decomposition Figure (Section 3.7)
    out_dose = save_dir / f"kinematic_dose_response_Target_{target_str}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
    plot_kinematic_dose_response_figure(
        conditions=conditions,
        dir_results=dir_results,
        target=target_str,
        fixed_params_summary=fixed_summary,
        save_path=out_dose,
        time_window=time_window,
    )

    # Remove opposite stale mode file if present so only current mode remains
    alt_prefix = "arclength" if is_y_mode else "forward_y"
    stale_alt = save_dir / f"{alt_prefix}_resampled_trajectories_Target_{target_str}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
    if stale_alt.exists():
        try:
            stale_alt.unlink()
        except Exception:
            pass

    return {
        "conditions": conditions,
        "RDM_early": RDM_early,
        "S_early": S_early,
        "Q_early": Q_early,
        "stereotypy_early": stereotypy_early,
        "RDM_late": RDM_late,
        "S_late": S_late,
        "Q_late": Q_late,
        "stereotypy_late": stereotypy_late,
        "S_late_x": S_late_x,
        "S_late_z": S_late_z,
        "RDM": RDM_late,
        "S": S_late,
        "Q": Q_late,
        "within_stereotypy": stereotypy_late,
        "var_results": var_results,
        "dir_results": dir_results,
        "fixed_params_summary": fixed_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D Behavioral Representational Similarity Analysis and Trajectory Variability Analysis."
    )
    parser.add_argument("--session", type=str, default=None, help="Session name (e.g. NRR_RW022).")
    parser.add_argument("--peri-root", type=str, default=None, help="PeriStim checkpoint root directory.")
    parser.add_argument("--dlc-3d-dir", type=str, default=None, help="Directory containing triangulated 3D DLC CSV files.")
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        choices=["auto", "A", "B", "both", "all"],
        help="Target button ('auto' analyzes all available targets).",
    )
    parser.add_argument("--keypoint", type=str, default=DEFAULT_KEYPOINT, help="DLC keypoint (middle, index, wrist).")
    parser.add_argument("--compare-conditions", nargs="+", type=int, default=None, help="Conditions / BR files to compare.")
    parser.add_argument("--control-br", nargs="+", type=int, default=None, help="Control condition BR indices.")
    parser.add_argument("--save-dir", type=str, default=None, help="Output figure directory.")
    parser.add_argument(
        "--window",
        nargs=2,
        type=float,
        default=None,
        metavar=("START_MS", "END_MS"),
        help="Global time window (ms) for reaching trajectories (default: -75 450).",
    )
    parser.add_argument(
        "--rsa-early-window",
        nargs=2,
        type=float,
        default=None,
        metavar=("START_MS", "END_MS"),
        help="Early time window (ms) for RSA analysis (default: 0 75).",
    )
    parser.add_argument(
        "--rsa-late-window",
        nargs=2,
        type=float,
        default=None,
        metavar=("START_MS", "END_MS"),
        help="Late/Perturbation time window (ms) for RSA analysis (default: 75 400).",
    )
    parser.add_argument(
        "--rsa-early-progress",
        nargs=2,
        type=float,
        default=None,
        metavar=("P_START", "P_END"),
        help="Early reach progress range fraction (default: 0.0 0.25 for 0–25% forward depth).",
    )
    parser.add_argument(
        "--rsa-late-progress",
        nargs=2,
        type=float,
        default=None,
        metavar=("P_START", "P_END"),
        help="Late reach progress range fraction (default: 0.25 1.0 for 25–100% forward depth).",
    )
    parser.add_argument(
        "--rsa-no-progress-align",
        dest="rsa_align_by_progress",
        action="store_false",
        default=True,
        help="Disable spatial progress (arc-length) resampling for RSA and use raw time matching instead.",
    )
    parser.add_argument(
        "--rsa-aggregation",
        type=str,
        default=None,
        choices=["median", "p90", "mean", "max"],
        help="Temporal/progress aggregation statistic for pairwise reach distances: 'median' (default), 'p90', 'mean', or 'max'.",
    )
    parser.add_argument(
        "--rsa-progress-samples",
        type=int,
        default=None,
        help="Number of normalized path samples for progress resampling (default: 100).",
    )
    parser.add_argument(
        "--rsa-progress-mode",
        type=str,
        default=None,
        choices=["forward_y", "y", "arclength", "arc"],
        help="Progress normalization mode: 'forward_y' (default: normalize along physical reaching axis Forward Y) or 'arclength' (3D path distance).",
    )
    parser.add_argument(
        "--no-task-align",
        dest="align_task_frame",
        action="store_false",
        default=True,
        help="Disable rotation into calibrated monkey task frame.",
    )
    parser.add_argument(
        "--show-indices",
        action="store_true",
        default=False,
        help="Include condition BR index prefix in labels (default: False).",
    )
    parser.add_argument(
        "--show-trial-counts",
        action="store_true",
        default=False,
        help="Include trial counts in condition labels (default: False).",
    )
    parser.add_argument(
        "--var-mode",
        type=str,
        default=None,
        choices=["both", "variability", "var", "deflection", "deviation", "dev"],
        help="Panels to plot in variability/deflection figure: 'both' (default), 'variability', or 'deflection'.",
    )
    parser.add_argument(
        "--var-legend-col",
        dest="legend_orientation",
        type=str,
        default=None,
        choices=["column", "row"],
        help="Legend layout for variability/deflection figure: 'column' (default) or 'row'.",
    )
    parser.add_argument(
        "--spatial-mode",
        type=str,
        default=None,
        choices=["overlay", "inset", "two_panel", "2", "endpoints_and_3d", "side_by_side"],
        help="Layout for spatial perturbation figure: 'overlay' (default: 3D trajectories with 2D touchscreen endpoint inset) or 'two_panel'.",
    )
    parser.add_argument(
        "--traj-stat",
        type=str,
        default=None,
        choices=["median", "mean", "med", "avg"],
        help="Central tendency statistic for reach trajectories and endpoints: 'median' (default) or 'mean'.",
    )
    parser.add_argument(
        "--ellipse-scale",
        type=str,
        default=None,
        choices=["2se", "1se", "2sd", "1sd", "se", "sd"],
        help="Dispersion ellipse scaling: '2se' (default, 2 standard errors of the mean), '1se', or '1sd' (sample standard deviation).",
    )
    parser.add_argument(
        "--target-diameter",
        type=float,
        default=None,
        help="Physical touchscreen target diameter/width in mm (default: 10.0 mm).",
    )
    parser.add_argument(
        "--target-stability-thresh",
        type=float,
        default=None,
        help="Maximum allowed 3D jitter (std in mm) across video detections before landmark is deemed unstable (default: 8.0 mm).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global SHOW_CONDITION_INDICES, SHOW_TRIAL_COUNTS, VARIABILITY_PLOT_MODE, VARIABILITY_LEGEND_ORIENTATION, SPATIAL_PLOT_MODE, TRAJECTORY_STAT, ELLIPSE_SCALE, RSA_ALIGN_BY_PROGRESS, RSA_DISTANCE_AGGREGATION, RSA_N_PROGRESS_SAMPLES, RSA_PROGRESS_MODE, TARGET_DIAMETER_MM, TARGET_STABILITY_THRESHOLD_MM
    if args.target_diameter is not None and args.target_diameter > 0:
        TARGET_DIAMETER_MM = float(args.target_diameter)
    if args.target_stability_thresh is not None and args.target_stability_thresh > 0:
        TARGET_STABILITY_THRESHOLD_MM = float(args.target_stability_thresh)
    if args.show_indices:
        SHOW_CONDITION_INDICES = True
    if args.show_trial_counts:
        SHOW_TRIAL_COUNTS = True
    if args.var_mode:
        VARIABILITY_PLOT_MODE = args.var_mode
    if args.legend_orientation:
        VARIABILITY_LEGEND_ORIENTATION = args.legend_orientation
    if args.spatial_mode:
        SPATIAL_PLOT_MODE = args.spatial_mode
    if args.traj_stat:
        TRAJECTORY_STAT = args.traj_stat
    if args.ellipse_scale:
        ELLIPSE_SCALE = args.ellipse_scale
    if args.rsa_align_by_progress is not None:
        RSA_ALIGN_BY_PROGRESS = args.rsa_align_by_progress
    if args.rsa_progress_mode:
        RSA_PROGRESS_MODE = args.rsa_progress_mode
    if args.rsa_aggregation:
        RSA_DISTANCE_AGGREGATION = args.rsa_aggregation
    if args.rsa_progress_samples:
        RSA_N_PROGRESS_SAMPLES = args.rsa_progress_samples

    # Session resolution
    env_sess = os.environ.get("RCP_SESSION")
    session = args.session or env_sess or (PARAMS.session if PARAMS and hasattr(PARAMS, "session") else "NRR_RW022")

    peri_root_arg = Path(args.peri_root) if args.peri_root else (PERI_ROOT if PERI_ROOT else Path("."))
    sess_dir = p3d.resolve_session_dir(session=session, peri_root=peri_root_arg) if (HAS_3D_KINEMATICS and p3d) else None

    # Resolve PeriStim root
    if peri_root_arg and peri_root_arg.exists():
        peri_root = peri_root_arg
    elif sess_dir and (sess_dir / "results" / "checkpoints" / "PeriStim").exists():
        peri_root = sess_dir / "results" / "checkpoints" / "PeriStim"
    elif OUT_BASE is not None and (OUT_BASE / "checkpoints" / "PeriStim").exists():
        peri_root = OUT_BASE / "checkpoints" / "PeriStim"
    else:
        peri_root = Path.cwd() / "results" / "checkpoints" / "PeriStim"

    # Resolve 3D DLC directory
    if args.dlc_3d_dir:
        dlc_3d_dir = Path(args.dlc_3d_dir)
    elif sess_dir and (sess_dir / "Video" / "DLC_3D_new").exists():
        dlc_3d_dir = sess_dir / "Video" / "DLC_3D_new"
    elif sess_dir and (sess_dir / "Video" / "DLC_3D").exists():
        dlc_3d_dir = sess_dir / "Video" / "DLC_3D"
    elif VIDEO_ROOT is not None and (VIDEO_ROOT / "DLC_3D_new").exists():
        dlc_3d_dir = VIDEO_ROOT / "DLC_3D_new"
    else:
        dlc_3d_dir = None

    # Resolve Save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    elif sess_dir and (sess_dir / "results").exists():
        save_dir = sess_dir / "results" / "figures" / "behavioral_rsa"
    elif OUT_BASE is not None:
        save_dir = OUT_BASE / "figures" / "behavioral_rsa"
    else:
        save_dir = Path.cwd() / "results" / "figures" / "behavioral_rsa"
    ensure_dir(save_dir)

    # Active conditions
    active_process_only = resolve_process_only()
    compare_conds = args.compare_conditions if args.compare_conditions is not None else active_process_only
    control_brs = args.control_br

    time_win = (float(args.window[0]), float(args.window[1])) if args.window else DEFAULT_PLOT_WINDOW_MS
    early_win = (float(args.rsa_early_window[0]), float(args.rsa_early_window[1])) if args.rsa_early_window else RSA_EARLY_WINDOW_MS
    late_win = (float(args.rsa_late_window[0]), float(args.rsa_late_window[1])) if args.rsa_late_window else RSA_LATE_WINDOW_MS
    early_prog = (float(args.rsa_early_progress[0]), float(args.rsa_early_progress[1])) if args.rsa_early_progress else RSA_EARLY_PROGRESS_RANGE
    late_prog = (float(args.rsa_late_progress[0]), float(args.rsa_late_progress[1])) if args.rsa_late_progress else RSA_LATE_PROGRESS_RANGE

    # Target resolution
    if args.target.lower() in ("auto", "both", "all"):
        targets = ["A", "B"]
    else:
        targets = [args.target.upper()]

    results_all: Dict[str, Dict[str, Any]] = {}
    for tgt in targets:
        res = run_behavioral_rsa_for_target(
            session=session,
            target=tgt,
            peri_root=peri_root,
            dlc_3d_dir=dlc_3d_dir,
            save_dir=save_dir,
            compare_conditions=compare_conds,
            control_br_indices=control_brs,
            time_window=time_win,
            rsa_early_window=early_win,
            rsa_late_window=late_win,
            rsa_early_progress=early_prog,
            rsa_late_progress=late_prog,
            keypoint=args.keypoint,
            align_task_frame=args.align_task_frame,
        )
        if res.get("conditions"):
            results_all[tgt] = res

    # Extract global fixed parameter summary if any target had fixed parameters
    global_fixed_summary = ""
    for res in results_all.values():
        if res.get("fixed_params_summary"):
            global_fixed_summary = res["fixed_params_summary"]
            break

    # Consolidated Multi-Target Spatial Perturbations Figure (Screen-Face, 3D Vectors, 3D Magnitudes)
    dir_results_by_target = {
        tgt: res["dir_results"]
        for tgt, res in results_all.items()
        if "dir_results" in res and res["dir_results"]
    }
    # Only generated if more than one target is present (avoids duplicate single-target figure)
    if dir_results_by_target and len(dir_results_by_target) > 1:
        tgt_keys = sorted(dir_results_by_target.keys())
        tgt_tag = "_and_".join(f"Target_{t}" for t in tgt_keys)
        all_conds = sorted(list(set(c for d in dir_results_by_target.values() for c in d.keys())))
        cond_tag = "_".join(map(str, all_conds))
        out_spatial = save_dir / f"combined_spatial_perturbations_{tgt_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
        plot_combined_spatial_perturbations_figure(
            dir_results_by_target=dir_results_by_target,
            fixed_params_summary=global_fixed_summary,
            save_path=out_spatial,
            mode=SPATIAL_PLOT_MODE,
            trajectory_stat=TRAJECTORY_STAT,
            ellipse_scale=ELLIPSE_SCALE,
            time_window=time_win,
        )
    elif dir_results_by_target and len(dir_results_by_target) == 1:
        # Clean up any leftover duplicate from earlier versions if it exists
        tgt_key = list(dir_results_by_target.keys())[0]
        all_conds = sorted(list(set(c for d in dir_results_by_target.values() for c in d.keys())))
        cond_tag = "_".join(map(str, all_conds))
        stale_file = save_dir / f"combined_spatial_perturbations_Target_{tgt_key}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
        if stale_file.exists():
            try:
                stale_file.unlink()
                _log(f"Removed redundant single-target duplicate: {stale_file.name}")
            except Exception:
                pass

    _log("Behavioral RSA & Trajectory Variability Pipeline Complete.")


if __name__ == "__main__":
    main()
