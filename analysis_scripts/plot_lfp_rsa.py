"""
plot_lfp_rsa.py
===============

LFP Representational Similarity Analysis (LFP RSA) for Utah-array Recordings
using GPU/CPU-Accelerated DPSS Multitaper Spectral Estimation.

Compares stimulation conditions (e.g., 40p@400Hz vs 13p@130Hz; pulse counts 1p, 2p, 4p, 8p, 20p, 40p;
stim+reach vs reach-only control) within a single session across cortical arrays:
M1i, M1s, PMd, SMA.

Key Methodological & Physical Principles:
    1. Information Criterion:
       - Feature vector per trial: Spatial pattern across valid electrodes of
         baseline-normalized, control-subtracted band power (dB).
       - Spectral Estimation: DPSS Multitaper (Slepian tapers).
         * Suppresses high-frequency spectral variance by 1/K (e.g., 2x for K=2).
         * Temporal smearing strictly bounded to +/- Tw / 2 (constant across all frequencies).
       - Pre-stimulus Baseline Window: [-950, -650] ms.
         * Strictly before reach onset (~ -200 ms for reaching tasks).
       - Separate Physiological Analysis Windows:
         * [-200, 0] ms:   Pre-stimulus reach initiation / motor preparation phase.
         * [0, 150] ms:    Early stimulus delivery phase (stim pulses + immediate evoked potentials).
         * [0, 350] ms:    Full reach perturbation phase (movement initiation through peak divergence).
         * [150, 350] ms:  Post-stimulus clean reach execution / recovery phase (100% artifact-free).
       - Matched Reach-Only Control Subtraction:
         * Because reach motor initiation begins at ~ -200 ms, post-zero power
           changes reflect both reach execution and stimulation perturbation.
         * Subtracting matched reach-only control isolates the STIMULATION-INDUCED NEURAL PERTURBATION:
             Delta_Power = Power_dB(stim+reach) - mean(Power_dB(control_reach))
       - Frequency Bands (avoiding line noise 55-65 Hz, 115-125 Hz):
         * Beta: (13–25 Hz)
         * Low Gamma: (25–55 Hz)
         * High Gamma: (65–115 Hz)
         * Alpha: (8–12 Hz)
         * Theta: (4–8 Hz)

    2. Combined Visualizations:
       - Combined 4-Array Silhouette Figure: Displays M1i, M1s, PMd, and SMA on the
         SAME 2x2 multi-panel plot with unified colormap limits, FDR permutation stars,
         and within-condition stereotypy.
       - Subtraction Analysis Figure: Stim-minus-Control excess perturbation (Delta_dB)
         with pooled 95% CI error bars and parametric dose-response trend line for all 4 arrays.
       - 4-Panel Response Magnitude / Dose-Response Figure: Baseline-normalized power (dB).
       - Trial-by-Trial RDM Heatmaps per array and frequency band.
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

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal, stats
from scipy.spatial.distance import cdist

# Try importing torch for GPU-accelerated multitaper computation
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

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

import RCP_analysis as rcp
try:
    from RCP_analysis.python.functions.config_loading import *
    CONFIG_LOADED = True
except Exception:
    CONFIG_LOADED = False
    PARAMS = None
    OUT_BASE = None
    SESSION_LOC = None
    METADATA_CSV = None
    METADATA_ROOT = None
    PERI_ROOT = None

try:
    from RCP_analysis.python.functions.rsa_utils import (
        _two_group_silhouette,
        _fdr_adjust_pairs,
        _p_stars,
    )
except Exception:
    _two_group_silhouette = None
    _fdr_adjust_pairs = None
    _p_stars = None

try:
    from RCP_analysis.python.functions.br_preproc import (
        load_electrode_mapping,
        get_region_grid,
        build_elec_to_data_idx,
    )
except Exception:
    load_electrode_mapping = None
    get_region_grid = None
    build_elec_to_data_idx = None


# =============================================================================
# Global Analysis Configuration & Defaults
# =============================================================================

# Pre-reach baseline: strictly before reach movement onset (~ -200 ms)
DEFAULT_BASELINE_WINDOW_MS: Tuple[float, float] = (-650.0, -450.0)

# Physiological Analysis Windows (Stim potentiation vs. Pre-reach baseline)
RSA_ANALYSIS_WINDOWS: Dict[str, Tuple[float, float]] = {
    "0_to_75ms": (0.0, 75.0),       # Early stimulation delivery / potentiation phase
    "75_to_150ms": (75.0, 175.0),   # Intermediate cortical response (40p offset ~100ms)
    "250_to_350ms": (175.0, 300.0), # Late reach recovery phase
    "75_to_350ms": (300.0, 450.0),   # Late reach execution / recovery phase
    # "0_to_20ms": (0.0, 20.0),       # Direct pulse train / stimulation delivery (8p offset ~20ms)
    # "0_to_50ms": (0.0, 50.0),       # for 20p
    # "0_to_100ms": (0.0, 100.0),     # for 40p
    # "150_to_250ms": (150.0, 250.0), # Movement-related reach execution phase
}
DEFAULT_ACTIVE_WINDOWS: List[str] = list(RSA_ANALYSIS_WINDOWS.keys())
DEFAULT_RESPONSE_WINDOW_MS: Tuple[float, float] = (0.0, 75.0)

# Multitaper parameters (Slepian DPSS tapers)
DEFAULT_MULTITAPER_WIN_MS: float = 75.0   # Window duration in ms
DEFAULT_MULTITAPER_NW: float = 2.0         # Time-halfbandwidth product NW
DEFAULT_MULTITAPER_K: int = 2              # Number of tapers (variance reduction = 1/2)

# Primary frequency bands (avoiding 55-65 Hz and 115-125 Hz line noise notch regions)
DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "Low Frequency": (2.0, 12.0),
    "Beta": (13.0, 25.0),           # Aligned with motor cortical beta (13-25 Hz)
    "LowGamma": (25.0, 55.0),       # Safely below 55-65 Hz line notch
    "HighGamma": (70.0, 110.0),     # Safely above 55-65 Hz line notch and below 115-125 Hz
}
NOTCH_REGIONS: List[Tuple[float, float]] = [(55.0, 65.0), (115.0, 125.0)]

DEFAULT_PRIMARY_BAND: str = "Beta"
DEFAULT_METRIC: str = "euclidean"
# 2x2 Layout Order:
# Top-Left: M1s (lower limb), Top-Right: SMA
# Bottom-Left: M1i (upper limb), Bottom-Right: PMd (Premotor dorsal)
DEFAULT_ARRAYS: List[str] = ["M1s", "SMA", "M1i", "PMd"]

MIN_TRIALS_PER_CONDITION: int = 3
N_PERMUTATIONS: int = 1000
FDR_ALPHA: float = 0.05
FIG_DPI: int = 300
FIG_FORMAT: str = "png"

# Condition Label & Plot Display Options matching plot_behavioral_rsa / plot_plateau_analysis
SHOW_CONDITION_INDICES: bool = False  # If False, omit condition index prefixes (e.g. 'BR 11', 'BR 18')
SHOW_TRIAL_COUNTS: bool = False       # If False, omit trial count in labels (e.g. 'n=15')

# Title & Header Display Options
SHOW_TITLES: bool = True              # Master switch: set to False to fully turn off all title label generation (both suptitle and subplot titles)
SHOW_SUPTITLE: bool = False            # If False, omit the main multi-line diagnostic suptitle at top of figures
SHOW_SUBPLOT_TITLES: bool = True      # If False, omit individual subplot/panel titles (e.g., array names)

# =============================================================================
# Plot Generation Settings (Choose which figures to generate)
# =============================================================================
GENERATE_SILHOUETTE_FIGURES: bool = False       # Combined 4-array condition silhouette separation matrices (per window)
GENERATE_TRIAL_RDM_FIGURES: bool = False        # Individual trial-by-trial RDM heatmaps per array
GENERATE_COMBINED_4ARRAY_RDM: bool = False      # Combined 2x2 trial-by-trial RDM heatmap for all 4 Utah arrays (per window)
GENERATE_SINGLE_ARRAY_WINDOWS_FIGURE: bool = False  # Compare a single array across active temporal analysis windows (e.g. M1i across 4 windows)
SINGLE_ARRAY_WINDOWS_TARGET: str = "M1i"       # Array to plot across windows: 'M1i' (M1 inferior), 'M1s', 'PMd', 'SMA', or 'all'
GENERATE_SUBTRACTION_FIGURES: bool = False     # Stim-minus-Control excess perturbation bar profiles
GENERATE_RESPONSE_MAGNITUDE_FIGURES: bool = False  # 4-panel array response magnitude dose-response bars
GENERATE_DISAMBIGUATION_FIGURE: bool = False   # High-temporal-precision sliding multitaper artifact disambiguation

# Time-Binned Spectral Dynamics Heatmaps (Condition vs 20 ms Time Bins)
GENERATE_CONDITION_TIME_HEATMAPS: bool = True       # Condition vs 20ms time-bin heatmaps (Within-condition & vs Control)
DYNAMICS_TARGET_ARRAYS: Union[str, List[str]] = "M1i" # Array(s) to plot: 'M1i' (M1 inferior), 'M1s', 'PMd', 'SMA', or 'all'
DYNAMICS_TIME_RANGE_MS: Tuple[float, float] = (-460.0, 460.0) # 20 ms bins spanning [-460, +460] ms aligned to 0 ms
DYNAMICS_BIN_WIDTH_MS: float = 20.0                 # Temporal bin width (ms)
DYNAMICS_MULTITAPER_WIN_MS: float = 50.0            # Multitaper sliding analysis window duration (ms)
DYNAMICS_SOFT_MASK: bool = False                    # True: soft mask (faded non-sig bins), False: full mask (fully masked out)
DYNAMICS_SHOW_STARS: bool = False                   # True: draw significance stars (*, **, ***), False: omit stars
DYNAMICS_MASK_COLOR: str = "#ffffff"                # Color for masked non-significant bins ('#ffffff' solid white)
DYNAMICS_USE_FDR: bool = True                       # Benjamini-Hochberg FDR correction across time bins
DYNAMICS_ALPHA: float = 0.05                        # Statistical significance threshold (alpha or FDR q)
DYNAMICS_BANDS: Optional[List[str]] = None          # Frequency bands to run (None runs all active bands in DEFAULT_BANDS)

# Anatomical display titles for clean publication labels
REGION_DISPLAY_NAMES: Dict[str, str] = {
    "M1s": "M1 (lower limb)",
    "M1 Sup": "M1 (lower limb)",
    "M1 (lower limb)": "M1 (lower limb)",
    "M1i": "M1 (upper limb)",
    "M1 Inf": "M1 (upper limb)",
    "M1 (upper limb)": "M1 (upper limb)",
    "PMd": "Premotor (dorsal)",
    "Premotor (dorsal)": "Premotor (dorsal)",
    "SMA": "SMA",
}

# Color definitions matching project standard
REGION_COLORS: Dict[str, str] = {
    "M1i": "#2ca02c",               # Green (Primary Motor Deep / Upper Limb)
    "M1 (upper limb)": "#2ca02c",   # Green
    "M1 Inf": "#2ca02c",            # Green
    "M1s": "#d62728",               # Red (Primary Motor Superficial / Lower Limb)
    "M1 (lower limb)": "#d62728",   # Red
    "M1 Sup": "#d62728",            # Red
    "PMd": "#ff7f0e",               # Orange (Dorsal Premotor)
    "Premotor (dorsal)": "#ff7f0e", # Orange
    "SMA": "#1f77b4",               # Blue (Supplementary Motor Area)
}
COLOR_CTRL: str = "#2c3e50"
COLOR_STIM_DEFAULT: str = "#e74c3c"


def get_region_color(region_name: str) -> str:
    """Return project-standard color for a cortical array region."""
    rn = str(region_name).lower()
    if "upper" in rn or "m1i" in rn or "inf" in rn:
        return "#2ca02c"
    if "lower" in rn or "m1s" in rn or "sup" in rn:
        return "#d62728"
    if "premotor" in rn or "pmd" in rn:
        return "#ff7f0e"
    if "sma" in rn:
        return "#1f77b4"
    return REGION_COLORS.get(region_name, "#2c3e50")


def get_region_display_name(region_name: str) -> str:
    """Return anatomical display title for a cortical array region."""
    return REGION_DISPLAY_NAMES.get(region_name, region_name)


def resolve_canonical_array(name: str) -> str:
    """Normalize user array name string (e.g. 'M1inferior', 'M1 (upper limb)') to canonical array identifier."""
    if not name:
        return "M1i"
    n = str(name).lower().replace(" ", "").replace("_", "").replace("-", "")
    if any(k in n for k in ["m1i", "m1inf", "inferior", "upper", "deep"]):
        return "M1i"
    if any(k in n for k in ["m1s", "m1sup", "superior", "lower", "superficial"]):
        return "M1s"
    if any(k in n for k in ["pmd", "premotor", "dorsal"]):
        return "PMd"
    if "sma" in n:
        return "SMA"
    for reg in DEFAULT_ARRAYS:
        if reg.lower() == n:
            return reg
    return str(name)


def resolve_target_arrays_for_windows(target_setting: Any) -> List[str]:
    """Resolve target array(s) setting to a list of canonical array names."""
    if not target_setting:
        return ["M1i"]
    if isinstance(target_setting, (list, tuple, set)):
        resolved = []
        for item in target_setting:
            canon = resolve_canonical_array(str(item))
            if canon not in resolved:
                resolved.append(canon)
        return resolved if resolved else ["M1i"]

    t_str = str(target_setting).strip()
    if t_str.lower() in ("all", "both", "*"):
        return list(DEFAULT_ARRAYS)

    return [resolve_canonical_array(t_str)]


# =============================================================================
# Logging & Helper Utilities
# =============================================================================

def _log(msg: str) -> None:
    print(f"[lfp_rsa] {msg}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _next_power_of_two(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


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


def _get_cond_attr(c: Any, key: str, default: Any = None) -> Any:
    """Helper to retrieve attributes from either LfpCondition objects or dictionaries."""
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
         - show_indices (default False): Toggle condition index prefix (e.g. 'BR 11:').
         - show_trial_counts (default False): Toggle trial counts (e.g. 'n=15').

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
            base_short = "Control"
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
    conditions: List[LfpCondition],
    show_indices: Optional[bool] = None,
    show_trial_counts: Optional[bool] = None,
) -> str:
    """Apply adaptive labels directly to LfpCondition objects in place and return fixed_summary."""
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


# =============================================================================
# Data Structures & Condition Organization
# =============================================================================

class LfpCondition:
    """Container holding broadband LFP trial recordings and metadata for one condition."""
    def __init__(
        self,
        cond_num: int,
        br_idx: int,
        label: str,
        base_label: str,
        target: str,
        is_control: bool,
        time_ms: np.ndarray,
        broadband: np.ndarray,  # (N_trials, N_channels, N_time)
        fs: float,
        ua_ids: Optional[np.ndarray] = None,
        pulse_count: Optional[int] = None,
        stim_freq_hz: Optional[float] = None,
        stim_dur_ms: Optional[float] = None,
        notes: str = "",
        stim_channels: Optional[str] = None,
        short_label: Optional[str] = None,
        fixed_params_summary: str = "",
    ):
        self.cond_num = cond_num
        self.br_idx = br_idx
        self.label = label
        self.base_label = base_label
        self.target = target.upper()
        self.is_control = is_control
        self.time_ms = time_ms      # (T,)
        self.broadband = broadband  # (N, C, T)
        self.fs = fs
        self.ua_ids = ua_ids
        self.pulse_count = pulse_count
        self.stim_freq_hz = stim_freq_hz
        self.stim_dur_ms = stim_dur_ms if stim_dur_ms is not None else 0.0
        self.notes = notes
        self.stim_channels = stim_channels
        self.short_label = short_label
        self.fixed_params_summary = fixed_params_summary

    @property
    def n_trials(self) -> int:
        return self.broadband.shape[0]

    @property
    def n_channels(self) -> int:
        return self.broadband.shape[1]

    @property
    def n_time(self) -> int:
        return len(self.time_ms)

    @property
    def clean_label(self) -> str:
        raw = self.short_label or self.label or f"BR_{self.br_idx}"
        return re.sub(r'\s*\(n=\d+\)', '', raw) if not SHOW_TRIAL_COUNTS else raw


# =============================================================================
# Utah Array Channel Mapping
# =============================================================================

def build_region_channel_indices(
    ua_ids: np.ndarray,
    monkey: Optional[str] = None,
) -> Dict[str, List[int]]:
    """
    Map channel indices in data array to anatomical Utah array regions:
    M1i, M1s, PMd, SMA.
    """
    if monkey is None:
        monkey = str(getattr(PARAMS, "monkey", "Nike")) if PARAMS else "Nike"

    mapping_csv = _REPO_ROOT / "config" / f"electrode_port_mapping_{monkey}.csv"
    region_map: Dict[str, List[int]] = {"M1i": [], "M1s": [], "PMd": [], "SMA": []}

    if load_electrode_mapping is not None and mapping_csv.exists():
        try:
            nsp_to_elec, _, elec_to_region = load_electrode_mapping(mapping_csv)
            for ch_idx, nsp_val in enumerate(ua_ids):
                nsp_id = int(nsp_val)
                elec_id = nsp_to_elec.get(nsp_id, -1)
                if elec_id < 1:
                    continue
                region = elec_to_region.get(elec_id, "")
                if region == "SMA":
                    region_map["SMA"].append(ch_idx)
                elif region == "PMd":
                    region_map["PMd"].append(ch_idx)
                elif region in ("M1i", "M1 Inf"):
                    region_map["M1i"].append(ch_idx)
                elif region in ("M1s", "M1 Sup"):
                    region_map["M1s"].append(ch_idx)
            return region_map
        except Exception as e:
            _log(f"Warning: Failed to parse electrode mapping: {e}")

    # Fallback: evenly divide channels into 4 pseudo-regions if mapping unavailable
    n_ch = len(ua_ids)
    quarter = max(1, n_ch // 4)
    region_map["M1i"] = list(range(0, quarter))
    region_map["M1s"] = list(range(quarter, 2 * quarter))
    region_map["PMd"] = list(range(2 * quarter, 3 * quarter))
    region_map["SMA"] = list(range(3 * quarter, n_ch))
    return region_map


# =============================================================================
# LFP Data Discovery & Loading
# =============================================================================

def _find_peri_data_for_br(br_num: int, target_str: str) -> Optional[Any]:
    """Helper to discover PeriStim npz files for a given BR index to extract exact stim pulse counts."""
    if PERI_ROOT is None or not Path(PERI_ROOT).exists():
        return None
    pat = f"*BR_{br_num:03d}*.npz"
    for sub in [
        Path(PERI_ROOT) / "stim_reaches" / f"target_{target_str}",
        Path(PERI_ROOT) / "stim_reaches" / f"target_{target_str.lower()}",
        Path(PERI_ROOT) / "control_reaches" / f"target_{target_str}",
        Path(PERI_ROOT) / "control_reaches" / f"target_{target_str.lower()}",
        Path(PERI_ROOT) / "stim_reaches",
        Path(PERI_ROOT) / "control_reaches",
        Path(PERI_ROOT),
    ]:
        if sub.exists():
            cands = list(sub.rglob(pat))
            for c in cands:
                if not c.name.startswith("._"):
                    try:
                        return np.load(c, allow_pickle=True)
                    except Exception:
                        pass
    return None


def load_lfp_conditions_for_target(
    session: str,
    target: str,
    lfp_dir: Path,
    compare_conditions: Optional[Sequence[int]] = None,
    control_br_indices: Optional[Sequence[int]] = None,
    metadata_csv: Optional[Path] = None,
) -> List[LfpCondition]:
    """
    Discover and load aligned LFP .npz files for a given session and target.
    Selects control and stimulation conditions based on process_only / compare_conditions.
    Robustly resolves exact pulse counts, durations, and frequencies from metadata CSV
    and PeriStim checkpoints to eliminate false 40p defaults.
    """
    target_str = target.upper()
    _log(f"Discovering LFP conditions for Session '{session}', Target {target_str} in {lfp_dir}...")

    if not lfp_dir.exists():
        _log(f"Error: LFP directory does not exist: {lfp_dir}")
        return []

    npz_candidates = sorted(list(lfp_dir.rglob("aligned_lfp__*.npz")))
    if not npz_candidates:
        _log(f"No aligned LFP npz files found in {lfp_dir}")
        return []

    # Parse metadata CSV if available with normalized column names
    session_meta: Dict[int, Dict[str, Any]] = {}
    if metadata_csv and Path(metadata_csv).exists():
        try:
            df_meta = None
            for enc in ("utf-8-sig", "latin-1", "cp1252", "utf-8"):
                try:
                    df_meta = pd.read_csv(metadata_csv, encoding=enc)
                    break
                except Exception:
                    pass
            if df_meta is not None:
                # Normalize column names: strip, lowercase, replace spaces/dashes with underscores
                df_meta.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df_meta.columns]
                br_col = None
                for candidate in ["br_file", "br_idx", "br", "condition"]:
                    if candidate in df_meta.columns:
                        br_col = candidate
                        break
                if br_col:
                    for _, row in df_meta.iterrows():
                        val = row.get(br_col)
                        if pd.notna(val):
                            try:
                                session_meta[int(float(val))] = row.to_dict()
                            except Exception:
                                pass
        except Exception as e:
            _log(f"Warning: Could not parse metadata CSV: {e}")

    parsed_candidates: List[Dict[str, Any]] = []

    for npz_path in npz_candidates:
        sess_id = npz_path.stem.replace("aligned_lfp__", "")

        # Target matching: filename or path
        tgt_match = re.search(r'target_([AB])', sess_id, re.IGNORECASE)
        tgt_val = tgt_match.group(1).upper() if tgt_match else None
        if tgt_val is None:
            if f"target_{target_str.lower()}" in str(npz_path).lower():
                tgt_val = target_str
            else:
                other_tgt = "B" if target_str == "A" else "A"
                if f"target_{other_tgt.lower()}" in str(npz_path).lower():
                    continue

        try:
            data = dict(np.load(npz_path, allow_pickle=True))
        except Exception:
            continue

        if "broadband_full" not in data:
            continue

        if tgt_val is None and "target" in data:
            try:
                t_key = str(data["target"]).strip().upper()
                if t_key in ("A", "B"):
                    tgt_val = t_key
            except Exception:
                pass

        if tgt_val is not None and tgt_val != target_str:
            continue

        bb = data["broadband_full"]
        if bb.ndim != 3 or bb.shape[0] < MIN_TRIALS_PER_CONDITION:
            continue

        # Extract Blackrock index
        br_idx = -1
        if "br_idx" in data:
            try:
                br_idx = int(data["br_idx"])
            except Exception:
                pass
        sess_clean = sess_id.lower()
        if br_idx < 0:
            m = re.search(r'_(\d{3})(?:_|$|\.)', sess_id)
            if m:
                br_idx = int(m.group(1))

        if br_idx < 0:
            if "baseline" in sess_clean or "control" in sess_clean:
                br_idx = 18  # BR 18 is Reach Baseline for NRR_RW022
            else:
                continue

        csv_meta = session_meta.get(br_idx, {})
        notes = str(csv_meta.get("notes", "")).strip()
        notes_clean = notes.lower() if notes else ""

        # Check PeriStim checkpoint file for ground truth stim pulse counts
        peri_data = _find_peri_data_for_br(br_idx, target_str)
        peri_pulses = None
        peri_freq = None
        peri_dur = None
        if peri_data is not None:
            if "stim_pulse_counts" in peri_data and peri_data["stim_pulse_counts"].size > 0:
                try:
                    peri_pulses = int(peri_data["stim_pulse_counts"].flat[0])
                except Exception:
                    pass
            if "stim_freq_hz" in peri_data:
                try:
                    peri_freq = float(peri_data["stim_freq_hz"])
                except Exception:
                    pass
            if "stim_dur_nominal_ms" in peri_data:
                try:
                    peri_dur = float(peri_data["stim_dur_nominal_ms"])
                except Exception:
                    pass

        # Check CSV metadata
        csv_pulses = None
        for pk in ["pulse_count", "pulses", "n_pulses", "pulse", "pulses_count", "stim_pulses"]:
            if pk in csv_meta and pd.notna(csv_meta[pk]):
                try:
                    csv_pulses = int(float(csv_meta[pk]))
                    break
                except Exception:
                    pass

        csv_freq = None
        for fk in ["stim_frequency_hz", "stim_freq_hz", "frequency", "freq_hz", "freq"]:
            if fk in csv_meta and pd.notna(csv_meta[fk]):
                try:
                    csv_freq = float(csv_meta[fk])
                    break
                except Exception:
                    pass

        csv_dur = None
        for dk in ["stim_duration_nominal_ms", "stim_duration_ms", "stim_dur_nominal_ms", "stim_dur_ms", "duration_ms", "duration"]:
            if dk in csv_meta and pd.notna(csv_meta[dk]):
                try:
                    csv_dur = float(csv_meta[dk])
                    break
                except Exception:
                    pass

        # Regex pulses from notes or sess_id
        regex_pulses = None
        for s_src in [notes, sess_id]:
            m = re.search(r'(\d+)\s*p(?:ulse)?s?', s_src, re.IGNORECASE)
            if m:
                try:
                    regex_pulses = int(m.group(1))
                    break
                except Exception:
                    pass

        # Check Control condition status (continuous stim is NEVER a control condition)
        is_continuous = ("continuous" in sess_clean or "continuous_stim" in sess_clean or "continuous" in notes_clean)
        is_control = bool(
            not is_continuous
            and (
                (control_br_indices and br_idx in control_br_indices)
                or (br_idx in (1, 18) and ("baseline" in sess_clean or "baseline" in notes_clean))
                or bool(data.get("is_control", False))
                or (peri_pulses is not None and peri_pulses == 0)
                or (csv_pulses is not None and csv_pulses == 0)
                or any(kw in sess_clean for kw in ("control", "baseline", "0p", "0pulse", "nostim", "no_stim"))
                or any(kw in notes_clean for kw in ("control", "baseline", "0p", "0pulse", "no stim", "nostim", "no_stim"))
            )
        )

        if is_control:
            n_pulses = 0
            stim_dur_ms = 0.0
            stim_freq_hz = 0.0
        else:
            # Resolve n_pulses with priority: PeriStim ground-truth > CSV meta > Regex > NPZ data
            if peri_pulses is not None and peri_pulses > 0:
                n_pulses = peri_pulses
            elif csv_pulses is not None and csv_pulses > 0:
                n_pulses = csv_pulses
            elif regex_pulses is not None and regex_pulses > 0:
                n_pulses = regex_pulses
            else:
                n_pulses = int(data.get("n_pulses", 40))

            # Resolve stim_freq_hz:
            if peri_freq is not None and peri_freq > 0:
                stim_freq_hz = peri_freq
            elif csv_freq is not None and csv_freq > 0:
                stim_freq_hz = csv_freq
            else:
                stim_freq_hz = float(data.get("stim_freq_hz", 400.0))

            # Resolve stim_dur_ms:
            if peri_dur is not None and peri_dur > 0:
                stim_dur_ms = peri_dur
            elif csv_dur is not None and csv_dur > 0:
                stim_dur_ms = csv_dur
            elif n_pulses and stim_freq_hz > 0:
                stim_dur_ms = float(n_pulses * 1000.0 / stim_freq_hz)
            else:
                stim_dur_ms = float(data.get("stim_dur_ms", 100.0))

        # Time axis
        t_ms = None
        for tk in ("t_full_ms", "rel_time_full", "t_ms"):
            if tk in data:
                t_ms = np.asarray(data[tk], dtype=float)
                break
        if t_ms is None or len(t_ms) != bb.shape[2]:
            fs_val = float(data.get("fs_lfp", 1000.0))
            t_ms = np.linspace(-1000.0, 1000.0, bb.shape[2])
        else:
            fs_val = float(data.get("fs_lfp", 1000.0))

        ua_ids = data.get("ua_ids_1based", None)
        if ua_ids is None:
            ua_ids = np.arange(1, bb.shape[1] + 1)

        stim_channels = extract_stim_channels_info(
            peri=peri_data,
            csv_meta=csv_meta,
            notes=notes,
            cond_str=sess_id,
        )

        parsed_candidates.append({
            "npz_path": npz_path,
            "br_idx": br_idx,
            "is_control": is_control,
            "n_trials": bb.shape[0],
            "broadband": bb,
            "t_ms": t_ms,
            "fs": fs_val,
            "ua_ids": ua_ids,
            "n_pulses": n_pulses,
            "stim_freq_hz": stim_freq_hz,
            "stim_dur_ms": stim_dur_ms,
            "notes": notes,
            "sess_id": sess_id,
            "stim_channels": stim_channels,
        })

    if not parsed_candidates:
        _log(f"No valid LFP files found for Target {target_str}.")
        return []

    # Condition filtering: process_only / compare_conditions
    if compare_conditions:
        comp_set = set(compare_conditions)
        req_matched = [c for c in parsed_candidates if c["br_idx"] in comp_set]
        ctrl_in_req = [c for c in req_matched if c["is_control"]]

        if ctrl_in_req:
            _log(f"Target {target_str}: Using control condition(s) {sorted(set(c['br_idx'] for c in ctrl_in_req))} from process_only.")
            selected = req_matched
        else:
            avail_ctrls = [c for c in parsed_candidates if c["is_control"]]
            if avail_ctrls:
                best_ctrl = max(avail_ctrls, key=lambda c: (10 if c["br_idx"] in (18, 1) else 1, c["n_trials"]))
                _log(f"Target {target_str}: Control not in process_only; selected BR {best_ctrl['br_idx']} ({best_ctrl['n_trials']} trials).")
                selected = req_matched + [best_ctrl]
            else:
                selected = req_matched
    else:
        avail_ctrls = [c for c in parsed_candidates if c["is_control"]]
        if avail_ctrls:
            best_ctrl = max(avail_ctrls, key=lambda c: (10 if c["br_idx"] in (18, 1) else 1, c["n_trials"]))
            _log(f"Target {target_str}: Auto-selected BR {best_ctrl['br_idx']} as control ({best_ctrl['n_trials']} trials).")
            selected = [best_ctrl] + [c for c in parsed_candidates if not c["is_control"]]
        else:
            selected = parsed_candidates

    # Deduplicate by br_idx (keep candidate with most trials)
    dedup: Dict[int, Dict[str, Any]] = {}
    for c in selected:
        b = c["br_idx"]
        if b not in dedup or c["n_trials"] > dedup[b]["n_trials"]:
            dedup[b] = c
    selected = list(dedup.values())

    conditions: List[LfpCondition] = []
    for cand in selected:
        br = cand["br_idx"]
        is_ctrl = cand["is_control"]
        npul = cand["n_pulses"]
        sfreq = cand["stim_freq_hz"]
        sdur = cand["stim_dur_ms"]
        ntri = cand["n_trials"]

        if is_ctrl:
            base_label = f"BR_{br} Reach Control" if br in (18, 1) or "baseline" in cand["sess_id"].lower() else f"BR_{br} Control"
        else:
            dur_str = f"{sdur:.1f}ms" if (sdur and not sdur.is_integer()) else (f"{int(sdur)}ms" if sdur else "")
            pulse_str = f"{npul}p@{int(sfreq)}Hz" if npul else f"BR_{br}"
            base_label = f"BR_{br} {pulse_str} ({dur_str})" if dur_str else f"BR_{br} {pulse_str}"

        label = f"{base_label} (n={ntri})"

        cond = LfpCondition(
            cond_num=br,
            br_idx=br,
            label=label,
            base_label=base_label,
            target=target_str,
            is_control=is_ctrl,
            time_ms=cand["t_ms"],
            broadband=cand["broadband"],
            fs=cand["fs"],
            ua_ids=cand["ua_ids"],
            pulse_count=npul,
            stim_freq_hz=sfreq,
            stim_dur_ms=sdur,
            notes=cand["notes"],
            stim_channels=cand.get("stim_channels", None),
        )
        conditions.append(cond)

    # Sort conditions: Control first, then stim conditions sorted by pulse count / BR index
    conditions.sort(key=lambda c: (0 if c.is_control else 1, c.pulse_count or 0, c.br_idx))

    # Apply adaptive publication condition labels (strip identical parameters across conditions)
    fixed_summary = apply_adaptive_condition_labels(
        conditions,
        show_indices=SHOW_CONDITION_INDICES,
        show_trial_counts=SHOW_TRIAL_COUNTS,
    )
    if fixed_summary:
        _log(f"   [{fixed_summary}]")

    _log(f"Loaded {len(conditions)} conditions for Target {target_str}:")
    for c in conditions:
        _log(f"   BR {c.br_idx:03d}: '{c.label}' (short='{c.short_label}', ctrl={c.is_control}, pulses={c.pulse_count}, dur={c.stim_dur_ms:.1f}ms)")

    return conditions


# =============================================================================
# DPSS Multitaper Spectral Power Estimation
# =============================================================================

def compute_multitaper_window_psd(
    traces_2d: np.ndarray,      # (N_signals, N_samples)
    fs: float,
    win_ms: float = DEFAULT_MULTITAPER_WIN_MS,
    nw: float = DEFAULT_MULTITAPER_NW,
    k: int = DEFAULT_MULTITAPER_K,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute DPSS Multitaper Power Spectral Density (PSD) for a batch of 1D traces.
    
    If segment duration exceeds win_ms, slides overlapping DPSS windows (50% step)
    across the entire interval and averages the PSDs across windows to cover the full
    temporal duration without discarding signal (e.g. for long intervals like 75-350 ms).
    If the segment is shorter than win_ms, adapts window length to segment length.
    
    Returns:
        psd: (N_signals, N_rfft_bins)
        freqs: (N_rfft_bins,) in Hz.
    """
    n_signals, n_samples = traces_2d.shape
    Nw = int(round(win_ms / 1000.0 * fs))
    if Nw % 2 == 0:
        Nw += 1
    Nw = max(7, min(Nw, n_samples))

    # Adaptive taper parameters if segment is smaller than requested window
    nw_eff = min(nw, max(1.0, (Nw - 1) / 4.0)) if Nw < int(round(win_ms / 1000.0 * fs)) else nw
    k_eff = max(1, min(k, int(2 * nw_eff) - 1)) if Nw < int(round(win_ms / 1000.0 * fs)) else k

    # Compute DPSS tapers (k_eff, Nw)
    tapers, _ = signal.windows.dpss(Nw, nw_eff, k_eff, return_ratios=True)
    tapers = np.ascontiguousarray(tapers, dtype=np.float32)

    # Slice segment across entire duration if n_samples > Nw
    if n_samples > Nw:
        step = max(1, Nw // 2)
        starts = list(range(0, n_samples - Nw + 1, step))
        if starts[-1] + Nw < n_samples:
            starts.append(n_samples - Nw)
        slice_list = [traces_2d[:, s:s + Nw] for s in starts]
        seg_batch = np.concatenate(slice_list, axis=0)  # (n_slices * N_signals, Nw)
        n_slices = len(starts)
    else:
        seg_batch = traces_2d[:, :Nw]
        n_slices = 1

    n_fft = max(1024, _next_power_of_two(Nw * 2))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    # GPU acceleration if torch available
    if use_gpu and TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            t_torch = torch.tensor(tapers, device=device, dtype=torch.float32)  # (K, Nw)
            x_torch = torch.tensor(np.ascontiguousarray(seg_batch), device=device, dtype=torch.float32)  # (n_slices*N, Nw)

            tapered = x_torch.unsqueeze(1) * t_torch.unsqueeze(0)
            fft_res = torch.fft.rfft(tapered, n=n_fft, dim=-1)
            power = torch.abs(fft_res) ** 2
            psd_torch = power.mean(dim=1)  # average across K tapers -> (n_slices*N, N_rfft_bins)
            psd_all = psd_torch.detach().cpu().numpy()
            del t_torch, x_torch, tapered, fft_res, power, psd_torch
            torch.cuda.empty_cache()

            if n_slices > 1:
                psd_all = psd_all.reshape(n_slices, n_signals, -1).mean(axis=0)
            return psd_all, freqs
        except Exception:
            pass  # Fall back to CPU

    # CPU vectorized computation
    seg_clean = np.nan_to_num(seg_batch, nan=0.0)
    tapered = seg_clean[:, None, :] * tapers[None, :, :]
    fft_res = np.fft.rfft(tapered, n=n_fft, axis=-1)
    power = np.abs(fft_res) ** 2
    psd_all = power.mean(axis=1)  # (n_slices*N, N_rfft_bins)
    if n_slices > 1:
        psd_all = psd_all.reshape(n_slices, n_signals, -1).mean(axis=0)
    return psd_all, freqs


def extract_trial_multitaper_band_power(
    broadband: np.ndarray,      # (N_trials, N_ch, N_time)
    time_ms: np.ndarray,        # (N_time,)
    fs: float,
    ch_indices: Sequence[int],
    band: Tuple[float, float],
    baseline_win: Tuple[float, float] = DEFAULT_BASELINE_WINDOW_MS,
    resp_win: Tuple[float, float] = DEFAULT_RESPONSE_WINDOW_MS,
    win_ms: float = DEFAULT_MULTITAPER_WIN_MS,
    nw: float = DEFAULT_MULTITAPER_NW,
    k: int = DEFAULT_MULTITAPER_K,
) -> np.ndarray:
    """
    Extract baseline-normalized band power (in dB) using DPSS Multitaper estimation.

    Steps:
      1. Extract pre-reach baseline window segment (strictly before reach initiation).
      2. Extract active response window segment.
      3. Compute DPSS multitaper PSD across the entire duration of both segments.
      4. Integrate PSD across band frequencies [f_lo, f_hi].
      5. Compute dB power change (potentiation): 10 * log10(P_resp / P_base).
      6. Handle invalid channel entries using condition channel medians rather than blind 0.0.

    Returns:
      feat: (N_trials, len(ch_indices)) array of baseline-normalized dB power.
    """
    n_trials = broadband.shape[0]
    n_sel_ch = len(ch_indices)

    sub_data = broadband[:, ch_indices, :].astype(np.float32)
    n_signals = n_trials * n_sel_ch
    signals_flat = sub_data.reshape(n_signals, -1)

    base_mask = (time_ms >= baseline_win[0]) & (time_ms <= baseline_win[1])
    resp_mask = (time_ms >= resp_win[0]) & (time_ms <= resp_win[1])

    if not np.any(base_mask):
        base_mask = (time_ms >= DEFAULT_BASELINE_WINDOW_MS[0]) & (time_ms <= DEFAULT_BASELINE_WINDOW_MS[1])
        if not np.any(base_mask):
            base_mask = (time_ms >= time_ms[0]) & (time_ms <= time_ms[0] + 200.0)
    if not np.any(resp_mask):
        resp_mask = (time_ms >= resp_win[0] - 50.0) & (time_ms <= resp_win[1] + 50.0)

    base_seg = signals_flat[:, base_mask]
    resp_seg = signals_flat[:, resp_mask]

    # Compute Multitaper PSDs
    psd_base, freqs = compute_multitaper_window_psd(base_seg, fs=fs, win_ms=win_ms, nw=nw, k=k)
    psd_resp, _ = compute_multitaper_window_psd(resp_seg, fs=fs, win_ms=win_ms, nw=nw, k=k)

    # Band frequency mask
    f_lo, f_hi = band
    band_mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band_mask):
        band_mask = (freqs >= 13.0) & (freqs <= 25.0)

    # Average PSD over band frequencies
    p_base = np.nanmean(psd_base[:, band_mask], axis=-1)  # (N_signals,)
    p_resp = np.nanmean(psd_resp[:, band_mask], axis=-1)  # (N_signals,)

    eps = 1e-12
    p_base = np.maximum(p_base, eps)
    p_resp = np.maximum(p_resp, eps)
    db_power = 10.0 * np.log10(p_resp / p_base)
    db_power = db_power.reshape(n_trials, n_sel_ch)

    # Robust handling: impute isolated non-finite values with condition channel median
    # rather than setting blindly to 0 dB (which falsely distorts distance metrics).
    invalid_mask = ~np.isfinite(db_power)
    if np.any(invalid_mask):
        for ch_col in range(n_sel_ch):
            col_vals = db_power[:, ch_col]
            valid_vals = col_vals[np.isfinite(col_vals)]
            med_val = float(np.median(valid_vals)) if len(valid_vals) > 0 else 0.0
            db_power[~np.isfinite(col_vals), ch_col] = med_val

    return db_power


# =============================================================================
# Representational Similarity Analysis (LFP RDM & Separation)
# =============================================================================

def build_lfp_rdm(
    conditions: List[LfpCondition],
    ch_indices: Sequence[int],
    band: Tuple[float, float],
    metric: str = "euclidean",
    baseline_win: Tuple[float, float] = DEFAULT_BASELINE_WINDOW_MS,
    resp_win: Tuple[float, float] = DEFAULT_RESPONSE_WINDOW_MS,
    win_ms: float = DEFAULT_MULTITAPER_WIN_MS,
    nw: float = DEFAULT_MULTITAPER_NW,
    k: int = DEFAULT_MULTITAPER_K,
    subtract_control: bool = False,
    zscore_features: Optional[bool] = None,
) -> Tuple[np.ndarray, List[int], List[str], List[np.ndarray], Optional[np.ndarray]]:
    """
    Construct trial-by-trial LFP Representational Dissimilarity Matrix (RDM).

    Evaluates stimulation potentiation of neural activity relative to pre-reach baseline
    within each condition and tests dissimilarity between conditions.
    
    Distance Metrics:
      - 'euclidean': Preserves trial power magnitude / gain increase across electrodes (dose-response effect)
        as well as spatial gradient. Raw dB features are preserved (zscore_features=False by default).
      - 'correlation': Standardizes features and normalizes each trial to unit length (direction only;
        removes overall amplitude / power magnitude differences).
      - 'cosine': Normalized dot product (direction only).
    """
    cond_features: List[np.ndarray] = []
    block_sizes = [c.n_trials for c in conditions]
    block_labels = [c.short_label if getattr(c, "short_label", None) else c.label for c in conditions]

    for c in conditions:
        feat = extract_trial_multitaper_band_power(
            broadband=c.broadband,
            time_ms=c.time_ms,
            fs=c.fs,
            ch_indices=ch_indices,
            band=band,
            baseline_win=baseline_win,
            resp_win=resp_win,
            win_ms=win_ms,
            nw=nw,
            k=k,
        )
        cond_features.append(feat)

    all_features = np.vstack(cond_features)  # (N_total, K_ch)

    # Matched reach-only control subtraction: Stim+Reach - ReachOnlyControl
    ctrl_mean = None
    if subtract_control:
        ctrl_indices = [idx for idx, c in enumerate(conditions) if c.is_control]
        if ctrl_indices:
            ctrl_feats = np.vstack([cond_features[i] for i in ctrl_indices])
            ctrl_mean = np.nanmean(ctrl_feats, axis=0, keepdims=True)  # (1, K_ch)
            all_features = all_features - ctrl_mean

    # Feature standardization:
    # For Euclidean distance, preserving raw trial-by-trial channel dB potentiation (zscore=False)
    # allows both power magnitude (dose-response gain) and spatial gradient to directly drive distances.
    # For correlation distance, cross-trial channel standardization (zscore=True) is standard.
    if zscore_features is None:
        zscore_features = (metric == "correlation")

    if zscore_features and all_features.shape[0] > 1:
        mu = np.nanmean(all_features, axis=0, keepdims=True)
        sd = np.nanstd(all_features, axis=0, keepdims=True)
        sd[sd < 1e-8] = 1.0
        all_features = (all_features - mu) / sd

    # Safety check against non-finite values before distance computation
    if np.any(~np.isfinite(all_features)):
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)

    if metric == "correlation":
        feat_c = all_features - np.nanmean(all_features, axis=1, keepdims=True)
        norms = np.linalg.norm(feat_c, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        feat_norm = feat_c / norms
        corr = np.dot(feat_norm, feat_norm.T)
        corr = np.clip(corr, -1.0, 1.0)
        RDM = np.clip(1.0 - corr, 0.0, 2.0)
        np.fill_diagonal(RDM, 0.0)
    elif metric == "cosine":
        norms = np.linalg.norm(all_features, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        feat_norm = all_features / norms
        cos_sim = np.dot(feat_norm, feat_norm.T)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        RDM = np.clip(1.0 - cos_sim, 0.0, 2.0)
        np.fill_diagonal(RDM, 0.0)
    else:
        # Euclidean distance on dB features (magnitude + spatial pattern)
        RDM = cdist(all_features, all_features, metric="euclidean")
        np.fill_diagonal(RDM, 0.0)

    return RDM, block_sizes, block_labels, cond_features, ctrl_mean


def compute_two_group_silhouette_score(D: np.ndarray, in_group_b: np.ndarray) -> float:
    """Fast two-group silhouette score for precomputed distance matrix D."""
    if _two_group_silhouette is not None:
        try:
            return float(_two_group_silhouette(D, in_group_b))
        except Exception:
            pass

    gb = np.asarray(in_group_b, dtype=bool)
    ga = ~gb
    nb, na = int(gb.sum()), int(ga.sum())
    if nb < 2 or na < 2:
        return np.nan
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pairwise condition silhouette score matrix S and permutation p/q values."""
    if rng is None:
        rng = np.random.default_rng(42)

    K = len(block_sizes)
    cum = np.cumsum(block_sizes)
    starts = np.r_[0, cum[:-1]]
    ends = cum

    S = np.zeros((K, K), dtype=float)
    P = np.ones((K, K), dtype=float)

    pairs: List[Tuple[int, int]] = []
    p_vals: List[float] = []

    for i in range(K):
        for j in range(i + 1, K):
            idx_i = np.arange(starts[i], ends[i])
            idx_j = np.arange(starts[j], ends[j])
            sub_idx = np.concatenate([idx_i, idx_j])
            n_sub = len(sub_idx)

            D_sub = D[np.ix_(sub_idx, sub_idx)]
            in_b = np.zeros(n_sub, dtype=bool)
            in_b[len(idx_i):] = True

            s_obs = compute_two_group_silhouette_score(D_sub, in_b)
            S[i, j] = s_obs
            S[j, i] = s_obs

            # Permutation test
            shuff_ge = 0
            for _ in range(n_perm):
                perm_b = rng.permutation(in_b)
                s_perm = compute_two_group_silhouette_score(D_sub, perm_b)
                if s_perm >= s_obs:
                    shuff_ge += 1

            p_val = (shuff_ge + 1) / (n_perm + 1)
            P[i, j] = p_val
            P[j, i] = p_val
            pairs.append((i, j))
            p_vals.append(p_val)

    # Benjamini-Hochberg FDR correction across all unique pairwise comparisons
    Q = np.ones((K, K), dtype=float)
    if p_vals:
        p_arr = np.asarray(p_vals, dtype=float)
        try:
            q_arr = stats.false_discovery_control(p_arr, method="bh")
        except Exception:
            # Vectorized fallback Benjamini-Hochberg
            order = np.argsort(p_arr)
            ranks = np.arange(1, len(p_arr) + 1)
            q_arr = np.minimum.accumulate((p_arr[order] * len(p_arr) / ranks)[::-1])[::-1]
            q_arr = np.clip(q_arr, 0.0, 1.0)
            q_orig = np.empty_like(q_arr)
            q_orig[order] = q_arr
            q_arr = q_orig

        for (i, j), q_val in zip(pairs, q_arr):
            Q[i, j] = q_val
            Q[j, i] = q_val

    return S, P, Q


def compute_within_condition_consistency(
    D: np.ndarray,
    block_sizes: List[int],
) -> np.ndarray:
    """Compute average pairwise trial distance within each condition block (stereotypy)."""
    K = len(block_sizes)
    cum = np.cumsum(block_sizes)
    starts = np.r_[0, cum[:-1]]
    ends = cum
    within = np.zeros(K, dtype=float)

    for i in range(K):
        idx = np.arange(starts[i], ends[i])
        if len(idx) < 2:
            within[i] = 0.0
            continue
        sub = D[np.ix_(idx, idx)]
        triu_idx = np.triu_indices(len(idx), k=1)
        within[i] = float(np.mean(sub[triu_idx]))
    return within


# =============================================================================
# Plotting & Visualization Functions
# =============================================================================

def plot_lfp_rdm_figure(
    RDM: np.ndarray,
    block_sizes: List[int],
    block_labels: List[str],
    array_name: str,
    band_name: str,
    target: str,
    window_name: str,
    resp_win: Tuple[float, float],
    baseline_win: Tuple[float, float] = DEFAULT_BASELINE_WINDOW_MS,
    metric_label: str = "Distance (Euclidean dB)",
    multitaper_info: str = "DPSS (win=75ms, NW=2.0, K=2)",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate publication-ready trial-by-trial LFP RDM heatmap with condition blocks."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=FIG_DPI)

    cmap = "viridis" if "1 -" in metric_label or "Distance" in metric_label or "euclidean" in metric_label.lower() else "RdBu_r"
    vmax = np.nanpercentile(RDM[RDM > 0], 98) if np.any(RDM > 0) else 1.0

    im = ax.imshow(
        RDM,
        origin="upper",
        aspect="equal",
        cmap=cmap,
        vmin=0.0,
        vmax=max(vmax, 0.1),
        interpolation="nearest",
    )

    cum = np.cumsum(block_sizes)
    starts = np.r_[0, cum[:-1]]
    boundaries = cum[:-1] - 0.5
    centers = starts + (np.asarray(block_sizes) / 2.0) - 0.5

    # Draw block boundary lines
    for b in boundaries:
        ax.axvline(b, color="white", linewidth=1.2, linestyle="-")
        ax.axhline(b, color="white", linewidth=1.2, linestyle="-")

    clean_labels = [re.sub(r'\s*\(n=\d+\)', '', lab) for lab in block_labels] if not SHOW_TRIAL_COUNTS else list(block_labels)

    ax.set_xticks(centers)
    ax.set_xticklabels(clean_labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(centers)
    ax.set_yticklabels(clean_labels, fontsize=8.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_label, fontsize=10)

    if SHOW_TITLES and SHOW_SUBPLOT_TITLES:
        color_reg = REGION_COLORS.get(array_name, "#2c3e50")
        ax.set_title(
            f"LFP Multitaper RDM Heatmap | Array: {array_name} | Band: {band_name}\n"
            f"Target {target} | Window: [{int(resp_win[0])}, {int(resp_win[1])}] ms ({window_name}) vs Pre-Reach [{int(baseline_win[0])}, {int(baseline_win[1])}] ms | Method: {multitaper_info} (N={RDM.shape[0]} trials)",
            fontsize=10.5,
            fontweight="bold",
            pad=12,
            color=color_reg,
        )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved RDM figure -> {save_path.name}")
    return fig


def plot_combined_4array_rdm_figure(
    results_by_array: Dict[str, Dict[str, Any]],
    band_name: str,
    band_range: Tuple[float, float],
    target: str,
    window_name: str,
    resp_win: Tuple[float, float],
    baseline_win: Tuple[float, float] = DEFAULT_BASELINE_WINDOW_MS,
    metric_label: str = "Euclidean Distance (dB)",
    multitaper_info: str = "DPSS (win=75ms, NW=2.0, K=2)",
    subtract_control: bool = False,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Combined 4-panel figure displaying trial-by-trial LFP RDM heatmaps
    for all 4 Utah arrays (M1i, M1s, PMd, SMA) on the SAME plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 14.0), dpi=FIG_DPI)

    # Compute global 98th percentile for RDM across all arrays for consistent scaling
    all_rdm_vals = []
    for res in results_by_array.values():
        if "RDM" in res and res["RDM"] is not None:
            r = res["RDM"]
            if np.any(r > 0):
                all_rdm_vals.append(r[r > 0])
    vmax = float(np.nanpercentile(np.concatenate(all_rdm_vals), 98)) if all_rdm_vals else 1.0
    vmax = max(vmax, 0.1)

    cmap = "viridis" if "1 -" in metric_label or "Distance" in metric_label or "euclidean" in metric_label.lower() else "RdBu_r"
    im_ref = None

    for idx, array_name in enumerate(DEFAULT_ARRAYS):
        r, c = idx // 2, idx % 2
        ax = axes[r, c]
        color_reg = REGION_COLORS.get(array_name, "#2c3e50")

        if array_name not in results_by_array:
            ax.axis("off")
            continue

        res = results_by_array[array_name]
        RDM = res["RDM"]
        block_sizes = res["block_sizes"]
        block_labels = res["block_labels"]
        clean_labels = [re.sub(r'\s*\(n=\d+\)', '', lab) for lab in block_labels] if not SHOW_TRIAL_COUNTS else list(block_labels)
        n_ch = len(res.get("ch_indices", []))

        im = ax.imshow(
            RDM,
            origin="upper",
            aspect="equal",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        im_ref = im

        cum = np.cumsum(block_sizes)
        starts = np.r_[0, cum[:-1]]
        boundaries = cum[:-1] - 0.5
        centers = starts + (np.asarray(block_sizes) / 2.0) - 0.5

        for b in boundaries:
            ax.axvline(b, color="white", linewidth=1.0, linestyle="-")
            ax.axhline(b, color="white", linewidth=1.0, linestyle="-")

        ax.set_xticks(centers)
        ax.set_xticklabels(clean_labels, rotation=45, ha="right", fontsize=8.0)
        ax.set_yticks(centers)
        ax.set_yticklabels(clean_labels, fontsize=8.0)
        disp_title = get_region_display_name(array_name)
        color_reg = get_region_color(array_name)
        if SHOW_TITLES and SHOW_SUBPLOT_TITLES:
            ax.set_title(disp_title, fontsize=12, fontweight="bold", color=color_reg, pad=10)

    # Shared vertical colorbar on right side
    if im_ref is not None:
        cbar_ax = fig.add_axes([0.90, 0.22, 0.020, 0.56])
        cb = fig.colorbar(im_ref, cax=cbar_ax, orientation="vertical")
        cb.set_label(f"Trial Dissimilarity [{metric_label}]", fontsize=10.5, fontweight="bold")
        cb.ax.tick_params(labelsize=9)

    fixed_params_summary = ""
    for res in results_by_array.values():
        if res.get("fixed_params_summary"):
            fixed_params_summary = res["fixed_params_summary"]
            break

    fixed_note = f"\n[{fixed_params_summary}]" if fixed_params_summary else ""
    feat_desc = (
        "Stim-minus-Control Excess Perturbation (ΔdB)"
        if subtract_control else
        f"Stim Potentiation vs Pre-Reach (dB re: [{int(baseline_win[0])}, {int(baseline_win[1])}] ms)"
    )
    if SHOW_TITLES and SHOW_SUPTITLE:
        fig.suptitle(
            f"LFP Trial-by-Trial RDM Heatmaps across Utah Arrays\n"
            f"Band: {band_name} ({int(band_range[0])}–{int(band_range[1])} Hz) | Window: [{int(resp_win[0])}, {int(resp_win[1])}] ms ({window_name}) vs Pre-Reach [{int(baseline_win[0])}, {int(baseline_win[1])}] ms | Target {target}\n"
            f"Features: {feat_desc} | Method: {multitaper_info}{fixed_note}",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )
        top_margin = 0.86 if fixed_note else 0.88
    else:
        top_margin = 0.95
    fig.subplots_adjust(top=top_margin, bottom=0.08, left=0.07, right=0.88, hspace=0.30, wspace=0.22)

    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Combined 4-Array RDM figure -> {save_path.name}")
    return fig


def plot_combined_4array_silhouette_figure(
    results_by_array: Dict[str, Dict[str, Any]],
    band_name: str,
    band_range: Tuple[float, float],
    target: str,
    window_name: str,
    resp_win: Tuple[float, float],
    multitaper_info: str = "DPSS (win=75ms, NW=2.0, K=2)",
    subtract_control: bool = False,
    baseline_win: Tuple[float, float] = DEFAULT_BASELINE_WINDOW_MS,
    metric: str = "euclidean",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Combined 4-panel figure displaying the condition separation silhouette matrices
    for all 4 Utah arrays (M1i, M1s, PMd, SMA) on the SAME plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 13.5), dpi=FIG_DPI)

    # Compute global symmetric color limit across all 4 arrays for direct comparability
    all_s_vals = []
    for res in results_by_array.values():
        if "S" in res and res["S"] is not None:
            all_s_vals.append(res["S"])
    if all_s_vals:
        v_abs = max(0.40, float(np.nanmax([np.nanmax(np.abs(s)) for s in all_s_vals])))
    else:
        v_abs = 0.50

    im_ref = None

    for idx, array_name in enumerate(DEFAULT_ARRAYS):
        r, c = idx // 2, idx % 2
        ax = axes[r, c]
        color_reg = REGION_COLORS.get(array_name, "#2c3e50")

        if array_name not in results_by_array:
            ax.axis("off")
            continue

        res = results_by_array[array_name]
        S = res["S"]
        Q = res["Q"]
        within_dist = res.get("within_dist", np.zeros(S.shape[0]))
        block_labels = res["block_labels"]
        clean_labels = [re.sub(r'\s*\(n=\d+\)', '', lab) for lab in block_labels] if not SHOW_TRIAL_COUNTS else list(block_labels)
        K = len(clean_labels)
        n_ch = len(res.get("ch_indices", []))

        im = ax.imshow(
            S,
            origin="upper",
            aspect="equal",
            cmap="RdBu_r",
            vmin=-v_abs,
            vmax=v_abs,
            interpolation="nearest",
        )
        im_ref = im

        for i in range(K):
            for j in range(K):
                if i == j:
                    val = within_dist[i]
                    txt = f"within\n{val:.2f}" if np.isfinite(val) else "—"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="#555555", style="italic")
                else:
                    s_val = S[i, j]
                    q_val = Q[i, j]
                    star = p_stars_label(q_val) if s_val > 0 else ""
                    col = "white" if abs(s_val) > v_abs * 0.55 else "black"
                    val_str = f"{s_val:+.2f}" if np.isfinite(s_val) else "—"
                    if star and star != "n.s.":
                        num_font = max(6.0, min(8.0, 9.0 - 0.15 * K))
                        star_font = max(10.5, min(14.0, 16.0 - 0.25 * K))
                        ax.text(j, i - 0.13, val_str, ha="center", va="center", fontsize=num_font, fontweight="bold", color=col)
                        ax.text(j, i + 0.16, star, ha="center", va="center", fontsize=star_font, fontweight="bold", color=col)
                    else:
                        num_font = max(6.5, min(8.5, 9.5 - 0.15 * K))
                        ax.text(j, i, val_str, ha="center", va="center", fontsize=num_font, fontweight="bold", color=col)

        ax.set_xticks(np.arange(K))
        ax.set_xticklabels(clean_labels, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks(np.arange(K))
        ax.set_yticklabels(clean_labels, fontsize=8.5)
        disp_title = get_region_display_name(array_name)
        color_reg = get_region_color(array_name)
        if SHOW_TITLES and SHOW_SUBPLOT_TITLES:
            ax.set_title(disp_title, fontsize=12, fontweight="bold", color=color_reg, pad=10)

    # Shared vertical colorbar on right side
    if im_ref is not None:
        cbar_ax = fig.add_axes([0.90, 0.22, 0.020, 0.56])
        cb = fig.colorbar(im_ref, cax=cbar_ax, orientation="vertical")
        cb.set_label("Pairwise Condition Silhouette Score (S)  [Separation Metric]", fontsize=10.5, fontweight="bold")
        cb.ax.tick_params(labelsize=9)

    fixed_params_summary = ""
    for res in results_by_array.values():
        if res.get("fixed_params_summary"):
            fixed_params_summary = res["fixed_params_summary"]
            break

    fixed_note = f"\n[{fixed_params_summary}]" if fixed_params_summary else ""
    feat_desc = (
        "Stim-minus-Control Excess Perturbation (ΔdB)"
        if subtract_control else
        f"Stim Potentiation vs Pre-Reach (dB re: [{int(baseline_win[0])}, {int(baseline_win[1])}] ms)"
    )
    metric_desc = "Euclidean Distance" if metric == "euclidean" else "Correlation Distance (1 - r)"
    if SHOW_TITLES and SHOW_SUPTITLE:
        fig.suptitle(
            f"LFP Condition Separation Matrix across Utah Arrays (Silhouette Score)\n"
            f"Band: {band_name} ({int(band_range[0])}–{int(band_range[1])} Hz) | Window: [{int(resp_win[0])}, {int(resp_win[1])}] ms ({window_name}) vs Pre-Reach [{int(baseline_win[0])}, {int(baseline_win[1])}] ms | Target {target}\n"
            f"Metric: {metric_desc} | Features: {feat_desc} | Method: {multitaper_info} | * q<0.05, ** q<0.01, *** q<0.001 (FDR BH){fixed_note}",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )
        top_margin = 0.86 if fixed_note else 0.88
    else:
        top_margin = 0.95
    fig.subplots_adjust(top=top_margin, bottom=0.08, left=0.07, right=0.88, hspace=0.30, wspace=0.22)

    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Combined 4-Array Silhouette figure -> {save_path.name}")
    return fig


def plot_single_array_windows_silhouette_figure(
    windows_dict: Dict[str, Dict[str, Any]],
    array_name: str,
    band_name: str,
    band_range: Tuple[float, float],
    target: str,
    active_windows: Dict[str, Tuple[float, float]],
    baseline_win: Tuple[float, float] = DEFAULT_BASELINE_WINDOW_MS,
    multitaper_info: str = "DPSS (win=75ms, NW=2.0, K=2)",
    subtract_control: bool = False,
    metric: str = "euclidean",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Combined figure comparing a single cortical array across multiple temporal
    analysis windows (e.g. 4 windows in a 2x2 grid) with condition silhouette matrices.
    """
    ordered_win_tags = [w for w in active_windows.keys() if w in windows_dict]
    n_wins = len(ordered_win_tags)
    if n_wins == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No window results available", ha="center", va="center")
        return fig

    if n_wins <= 1:
        nrows, ncols = 1, 1
        figsize = (8.5, 7.5)
    elif n_wins == 2:
        nrows, ncols = 1, 2
        figsize = (15.0, 7.0)
    elif n_wins <= 4:
        nrows, ncols = 2, 2
        figsize = (15.0, 14.0)
    elif n_wins <= 6:
        nrows, ncols = 2, 3
        figsize = (21.0, 14.0)
    else:
        nrows, ncols = (n_wins + 3) // 4, 4
        figsize = (28.0, 7.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=FIG_DPI)
    axes = np.atleast_1d(axes).flatten()

    all_s = [windows_dict[w]["S"] for w in ordered_win_tags if "S" in windows_dict[w]]
    v_abs = max(float(np.nanmax([np.nanmax(np.abs(s)) for s in all_s])), 0.2) if all_s else 0.5

    im_ref = None
    color_reg = get_region_color(array_name)
    disp_name = get_region_display_name(array_name)

    for idx, win_tag in enumerate(ordered_win_tags):
        ax = axes[idx]
        res = windows_dict[win_tag]
        resp_win = active_windows[win_tag]

        S = res["S"]
        Q = res["Q"]
        within_dist = res.get("within_dist", np.zeros(S.shape[0]))
        block_labels = res["block_labels"]
        clean_labels = [re.sub(r'\s*\(n=\d+\)', '', lab) for lab in block_labels] if not SHOW_TRIAL_COUNTS else list(block_labels)
        K = len(clean_labels)

        im = ax.imshow(
            S,
            origin="upper",
            aspect="equal",
            cmap="RdBu_r",
            vmin=-v_abs,
            vmax=v_abs,
            interpolation="nearest",
        )
        im_ref = im

        for i in range(K):
            for j in range(K):
                if i == j:
                    val = within_dist[i]
                    txt = f"within\n{val:.2f}" if np.isfinite(val) else "—"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="#555555", style="italic")
                else:
                    s_val = S[i, j]
                    q_val = Q[i, j]
                    star = p_stars_label(q_val) if s_val > 0 else ""
                    col = "white" if abs(s_val) > v_abs * 0.55 else "black"
                    val_str = f"{s_val:+.2f}" if np.isfinite(s_val) else "—"
                    if star and star != "n.s.":
                        num_font = max(6.0, min(8.0, 9.0 - 0.15 * K))
                        star_font = max(10.5, min(14.0, 16.0 - 0.25 * K))
                        ax.text(j, i - 0.13, val_str, ha="center", va="center", fontsize=num_font, fontweight="bold", color=col)
                        ax.text(j, i + 0.16, star, ha="center", va="center", fontsize=star_font, fontweight="bold", color=col)
                    else:
                        num_font = max(6.5, min(8.5, 9.5 - 0.15 * K))
                        ax.text(j, i, val_str, ha="center", va="center", fontsize=num_font, fontweight="bold", color=col)

        ax.set_xticks(np.arange(K))
        ax.set_xticklabels(clean_labels, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks(np.arange(K))
        ax.set_yticklabels(clean_labels, fontsize=8.5)

        if SHOW_TITLES and SHOW_SUBPLOT_TITLES:
            disp_win = f"[{int(resp_win[0])}, {int(resp_win[1])}] ms"
            ax.set_title(disp_win, fontsize=12, fontweight="bold", color=color_reg, pad=10)

    for idx in range(n_wins, len(axes)):
        axes[idx].axis("off")

    if im_ref is not None:
        cbar_ax = fig.add_axes([0.90, 0.22, 0.020, 0.56])
        cb = fig.colorbar(im_ref, cax=cbar_ax, orientation="vertical")
        cb.set_label("Pairwise Condition Silhouette Score (S)  [Separation Metric]", fontsize=10.5, fontweight="bold")
        cb.ax.tick_params(labelsize=9)

    fixed_params_summary = ""
    for res in windows_dict.values():
        if res.get("fixed_params_summary"):
            fixed_params_summary = res["fixed_params_summary"]
            break

    fixed_note = f"\n[{fixed_params_summary}]" if fixed_params_summary else ""
    feat_desc = (
        "Stim-minus-Control Excess Perturbation (ΔdB)"
        if subtract_control else
        f"Stim Potentiation vs Pre-Reach (dB re: [{int(baseline_win[0])}, {int(baseline_win[1])}] ms)"
    )
    metric_desc = "Euclidean Distance" if metric == "euclidean" else "Correlation Distance (1 - r)"

    if SHOW_TITLES and SHOW_SUPTITLE:
        fig.suptitle(
            f"LFP Condition Separation Matrix across Windows | {disp_name} ({array_name})\n"
            f"Band: {band_name} ({int(band_range[0])}–{int(band_range[1])} Hz) | Pre-Reach [{int(baseline_win[0])}, {int(baseline_win[1])}] ms | Target {target}\n"
            f"Metric: {metric_desc} | Features: {feat_desc} | Method: {multitaper_info} | * q<0.05, ** q<0.01, *** q<0.001 (FDR BH){fixed_note}",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )
        top_margin = 0.86 if fixed_note else 0.88
    else:
        top_margin = 0.95
    fig.subplots_adjust(top=top_margin, bottom=0.08, left=0.07, right=0.88, hspace=0.30, wspace=0.22)

    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Single-Array Windows Silhouette figure -> {save_path.name}")
    return fig


def plot_single_array_windows_rdm_figure(
    windows_dict: Dict[str, Dict[str, Any]],
    array_name: str,
    band_name: str,
    band_range: Tuple[float, float],
    target: str,
    active_windows: Dict[str, Tuple[float, float]],
    baseline_win: Tuple[float, float] = DEFAULT_BASELINE_WINDOW_MS,
    metric_label: str = "Euclidean Distance (dB)",
    multitaper_info: str = "DPSS (win=75ms, NW=2.0, K=2)",
    subtract_control: bool = False,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Combined figure comparing a single cortical array across multiple temporal
    analysis windows (e.g. 4 windows in a 2x2 grid) with trial-by-trial RDM heatmaps.
    """
    ordered_win_tags = [w for w in active_windows.keys() if w in windows_dict]
    n_wins = len(ordered_win_tags)
    if n_wins == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No window results available", ha="center", va="center")
        return fig

    if n_wins <= 1:
        nrows, ncols = 1, 1
        figsize = (8.5, 7.5)
    elif n_wins == 2:
        nrows, ncols = 1, 2
        figsize = (15.0, 7.0)
    elif n_wins <= 4:
        nrows, ncols = 2, 2
        figsize = (15.0, 14.0)
    elif n_wins <= 6:
        nrows, ncols = 2, 3
        figsize = (21.0, 14.0)
    else:
        nrows, ncols = (n_wins + 3) // 4, 4
        figsize = (28.0, 7.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=FIG_DPI)
    axes = np.atleast_1d(axes).flatten()

    all_rdm_vals = []
    for res in windows_dict.values():
        if "RDM" in res and res["RDM"] is not None:
            r = res["RDM"]
            if np.any(r > 0):
                all_rdm_vals.append(r[r > 0])
    vmax = float(np.nanpercentile(np.concatenate(all_rdm_vals), 98)) if all_rdm_vals else 1.0
    vmax = max(vmax, 0.1)

    cmap = "viridis" if "1 -" in metric_label or "Distance" in metric_label or "euclidean" in metric_label.lower() else "RdBu_r"
    im_ref = None
    color_reg = get_region_color(array_name)
    disp_name = get_region_display_name(array_name)

    for idx, win_tag in enumerate(ordered_win_tags):
        ax = axes[idx]
        res = windows_dict[win_tag]
        resp_win = active_windows[win_tag]

        RDM = res["RDM"]
        block_sizes = res["block_sizes"]
        block_labels = res["block_labels"]
        clean_labels = [re.sub(r'\s*\(n=\d+\)', '', lab) for lab in block_labels] if not SHOW_TRIAL_COUNTS else list(block_labels)

        im = ax.imshow(
            RDM,
            origin="upper",
            aspect="equal",
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        im_ref = im

        cum = np.cumsum(block_sizes)
        starts = np.r_[0, cum[:-1]]
        boundaries = cum[:-1] - 0.5
        centers = starts + (np.asarray(block_sizes) / 2.0) - 0.5

        for b in boundaries:
            ax.axvline(b, color="white", linewidth=1.0, linestyle="-")
            ax.axhline(b, color="white", linewidth=1.0, linestyle="-")

        ax.set_xticks(centers)
        ax.set_xticklabels(clean_labels, rotation=45, ha="right", fontsize=8.0)
        ax.set_yticks(centers)
        ax.set_yticklabels(clean_labels, fontsize=8.0)

        if SHOW_TITLES and SHOW_SUBPLOT_TITLES:
            disp_win = f"[{int(resp_win[0])}, {int(resp_win[1])}] ms"
            ax.set_title(disp_win, fontsize=12, fontweight="bold", color=color_reg, pad=10)

    for idx in range(n_wins, len(axes)):
        axes[idx].axis("off")

    if im_ref is not None:
        cbar_ax = fig.add_axes([0.90, 0.22, 0.020, 0.56])
        cb = fig.colorbar(im_ref, cax=cbar_ax, orientation="vertical")
        cb.set_label(f"Trial Dissimilarity [{metric_label}]", fontsize=10.5, fontweight="bold")
        cb.ax.tick_params(labelsize=9)

    fixed_params_summary = ""
    for res in windows_dict.values():
        if res.get("fixed_params_summary"):
            fixed_params_summary = res["fixed_params_summary"]
            break

    fixed_note = f"\n[{fixed_params_summary}]" if fixed_params_summary else ""
    feat_desc = (
        "Stim-minus-Control Excess Perturbation (ΔdB)"
        if subtract_control else
        f"Stim Potentiation vs Pre-Reach (dB re: [{int(baseline_win[0])}, {int(baseline_win[1])}] ms)"
    )

    if SHOW_TITLES and SHOW_SUPTITLE:
        fig.suptitle(
            f"LFP Trial-by-Trial RDM Heatmaps across Windows | {disp_name} ({array_name})\n"
            f"Band: {band_name} ({int(band_range[0])}–{int(band_range[1])} Hz) | Pre-Reach [{int(baseline_win[0])}, {int(baseline_win[1])}] ms | Target {target}\n"
            f"Features: {feat_desc} | Method: {multitaper_info}{fixed_note}",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )
        top_margin = 0.86 if fixed_note else 0.88
    else:
        top_margin = 0.95
    fig.subplots_adjust(top=top_margin, bottom=0.08, left=0.07, right=0.88, hspace=0.30, wspace=0.22)

    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Single-Array Windows RDM figure -> {save_path.name}")
    return fig


def plot_subtraction_analysis_figure(
    conditions: List[LfpCondition],
    region_map: Dict[str, List[int]],
    band: Tuple[float, float],
    band_name: str,
    target: str,
    window_name: str,
    resp_win: Tuple[float, float],
    baseline_win: Tuple[float, float] = DEFAULT_BASELINE_WINDOW_MS,
    win_ms: float = DEFAULT_MULTITAPER_WIN_MS,
    nw: float = DEFAULT_MULTITAPER_NW,
    k: int = DEFAULT_MULTITAPER_K,
    save_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Generate 4-panel Subtraction Analysis figure:
    M1i, M1s, PMd, SMA showing Stim minus Control excess perturbation (ΔdB)
    with pooled 95% CI error bars and parametric dose-response trend line.
    """
    ctrl_conds = [c for c in conditions if c.is_control]
    stim_conds = [c for c in conditions if not c.is_control]

    if not ctrl_conds or not stim_conds:
        _log("Subtraction analysis requires at least one control and one stim condition.")
        return None

    stim_conds_sorted = sorted(stim_conds, key=lambda c: (c.pulse_count or 0, c.stim_dur_ms))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), dpi=FIG_DPI, sharey=True)
    axes_flat = axes.flatten()

    for idx, array_name in enumerate(DEFAULT_ARRAYS):
        ax = axes_flat[idx]
        ch_idx = region_map.get(array_name, [])
        color_reg = REGION_COLORS.get(array_name, "#34495e")

        if not ch_idx:
            ax.text(0.5, 0.5, f"No channels mapped for {array_name}", ha="center", va="center")
            continue

        # Extract control trial power (average across channels -> 1D array of trial dB)
        ctrl_trial_list = []
        for c in ctrl_conds:
            p_db = extract_trial_multitaper_band_power(
                broadband=c.broadband,
                time_ms=c.time_ms,
                fs=c.fs,
                ch_indices=ch_idx,
                band=band,
                baseline_win=baseline_win,
                resp_win=resp_win,
                win_ms=win_ms,
                nw=nw,
                k=k,
            )
            ctrl_trial_list.append(np.nanmean(p_db, axis=1))
        ctrl_trials = np.concatenate(ctrl_trial_list)
        mu_ctrl = float(np.nanmean(ctrl_trials))
        std_ctrl = float(np.nanstd(ctrl_trials))
        n_ctrl = len(ctrl_trials)

        diff_means: List[float] = []
        diff_cis: List[float] = []
        labels: List[str] = []

        for c in stim_conds_sorted:
            p_db = extract_trial_multitaper_band_power(
                broadband=c.broadband,
                time_ms=c.time_ms,
                fs=c.fs,
                ch_indices=ch_idx,
                band=band,
                baseline_win=baseline_win,
                resp_win=resp_win,
                win_ms=win_ms,
                nw=nw,
                k=k,
            )
            tr_mean = np.nanmean(p_db, axis=1)
            mu_stim = float(np.nanmean(tr_mean))
            std_stim = float(np.nanstd(tr_mean))
            n_stim = len(tr_mean)

            diff = mu_stim - mu_ctrl
            sem_diff = float(np.sqrt((std_stim**2 / max(1, n_stim)) + (std_ctrl**2 / max(1, n_ctrl))) * 1.96)

            diff_means.append(diff)
            diff_cis.append(sem_diff)

            pul = c.pulse_count or 0
            dur = c.stim_dur_ms
            dur_str = f"{dur:.1f}ms" if (dur and not dur.is_integer()) else (f"{int(dur)}ms" if dur else "")
            labels.append(f"{pul}p ({dur_str})" if dur_str else f"{pul}p")

        x = np.arange(len(stim_conds_sorted))
        bar_colors = ["#e74c3c" if d > 0 else "#2980b9" for d in diff_means]

        # Bar plot
        bars = ax.bar(
            x,
            diff_means,
            yerr=diff_cis,
            color=bar_colors,
            alpha=0.82,
            capsize=4,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

        # Overlay dose-response trend line
        ax.plot(x, diff_means, color="#2c3e50", linestyle="--", linewidth=1.5, marker="o", markersize=5, zorder=4, alpha=0.7)

        # Baseline control reference line at 0 dB
        ax.axhline(0.0, color="black", linestyle="-", linewidth=1.2, alpha=0.85, zorder=2)

        # Annotate values on bars
        for xi, dm, ci in zip(x, diff_means, diff_cis):
            y_offset = (ci + 0.15) if dm >= 0 else (-ci - 0.35)
            va = "bottom" if dm >= 0 else "top"
            ax.text(xi, dm + y_offset, f"{dm:+.2f}", ha="center", va=va, fontsize=7.5, fontweight="bold", color="#2c3e50")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
        disp_title = get_region_display_name(array_name)
        color_reg = get_region_color(array_name)
        if SHOW_TITLES and SHOW_SUBPLOT_TITLES:
            ax.set_title(disp_title, fontsize=11.5, fontweight="bold", color=color_reg)
        if idx % 2 == 0:
            ax.set_ylabel("Stim − Control ΔPower (dB)", fontsize=9.5, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=1)

    if SHOW_TITLES and SHOW_SUPTITLE:
        fig.suptitle(
            f"LFP Stimulation-minus-Control Excess Perturbation (ΔdB re Matched Reach-Only Control)\n"
            f"Cortical Utah Arrays: M1i, M1s, PMd, SMA | Band: {band_name} ({int(band[0])}–{int(band[1])} Hz)\n"
            f"Window: [{int(resp_win[0])}, {int(resp_win[1])}] ms ({window_name}) vs Pre-Reach [{int(baseline_win[0])}, {int(baseline_win[1])}] ms | Target {target} (Mean ± 95% CI)",
            fontsize=11.5,
            fontweight="bold",
            y=0.99,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Subtraction Analysis figure -> {save_path.name}")
    return fig


def plot_array_response_magnitude_figure(
    conditions: List[LfpCondition],
    region_map: Dict[str, List[int]],
    band: Tuple[float, float],
    band_name: str,
    target: str,
    window_name: str,
    resp_win: Tuple[float, float],
    baseline_win: Tuple[float, float] = DEFAULT_BASELINE_WINDOW_MS,
    win_ms: float = DEFAULT_MULTITAPER_WIN_MS,
    nw: float = DEFAULT_MULTITAPER_NW,
    k: int = DEFAULT_MULTITAPER_K,
    subtract_control: bool = False,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Generate 4-panel dose-response / array response magnitude figure:
    M1i, M1s, PMd, SMA showing stimulation potentiation relative to pre-reach baseline (or control-subtracted).
    """
    clean_labels = [re.sub(r'\s*\(n=\d+\)', '', c.label) for c in conditions]
    n_conds = len(conditions)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=FIG_DPI, sharey=True)
    axes_flat = axes.flatten()

    for idx, array_name in enumerate(DEFAULT_ARRAYS):
        ax = axes_flat[idx]
        ch_idx = region_map.get(array_name, [])
        color_reg = REGION_COLORS.get(array_name, "#34495e")

        if not ch_idx:
            ax.text(0.5, 0.5, f"No channels mapped for {array_name}", ha="center", va="center")
            continue

        cond_trial_means: List[np.ndarray] = []
        for c in conditions:
            db_pow = extract_trial_multitaper_band_power(
                broadband=c.broadband,
                time_ms=c.time_ms,
                fs=c.fs,
                ch_indices=ch_idx,
                band=band,
                baseline_win=baseline_win,
                resp_win=resp_win,
                win_ms=win_ms,
                nw=nw,
                k=k,
            )
            tr_mean = np.nanmean(db_pow, axis=1)
            cond_trial_means.append(tr_mean)

        ctrl_offset = 0.0
        if subtract_control:
            ctrl_indices = [i for i, c in enumerate(conditions) if c.is_control]
            if ctrl_indices:
                ctrl_pool = np.concatenate([cond_trial_means[i] for i in ctrl_indices])
                ctrl_offset = float(np.nanmean(ctrl_pool))

        means: List[float] = []
        ci_lows: List[float] = []
        ci_highs: List[float] = []

        for tr_m in cond_trial_means:
            adj_m = tr_m - ctrl_offset
            m = float(np.nanmean(adj_m))
            sem = float(np.nanstd(adj_m) / np.sqrt(max(1, len(adj_m))))
            means.append(m)
            ci_lows.append(m - 1.96 * sem)
            ci_highs.append(m + 1.96 * sem)

        x = np.arange(n_conds)
        colors = [COLOR_CTRL if c.is_control else color_reg for c in conditions]

        ax.bar(x, means, yerr=[np.array(means) - np.array(ci_lows), np.array(ci_highs) - np.array(means)],
               color=colors, alpha=0.85, capsize=4, edgecolor="black", linewidth=0.8)

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(clean_labels, rotation=45, ha="right", fontsize=8.5)
        disp_title = get_region_display_name(array_name)
        color_reg = get_region_color(array_name)
        if SHOW_TITLES and SHOW_SUBPLOT_TITLES:
            ax.set_title(disp_title, fontsize=11, fontweight="bold", color=color_reg)
        y_lbl = "Excess Power relative to Control (dB)" if subtract_control else f"Power Potentiation vs Pre-Reach [{int(baseline_win[0])}, {int(baseline_win[1])}] ms (dB)"
        ax.set_ylabel(y_lbl, fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    if SHOW_TITLES and SHOW_SUPTITLE:
        sub_title = "Control-Subtracted Stim Perturbation" if subtract_control else f"Stim Potentiation vs Pre-Reach [{int(baseline_win[0])}, {int(baseline_win[1])}] ms"
        fig.suptitle(
            f"LFP Multitaper Band Power Response across Utah Arrays ({sub_title})\n"
            f"Band: {band_name} ({int(band[0])}–{int(band[1])} Hz) | Window: [{int(resp_win[0])}, {int(resp_win[1])}] ms ({window_name}) vs Pre-Reach [{int(baseline_win[0])}, {int(baseline_win[1])}] ms | Target {target} (Mean ± 95% CI)",
            fontsize=11.5,
            fontweight="bold",
            y=0.99,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Response Magnitude figure -> {save_path.name}")
    return fig


def plot_time_resolved_stim_artifact_disambiguation(
    conditions: List[LfpCondition],
    region_map: Dict[str, List[int]],
    target: str,
    save_dir: Path,
    win_ms: float = 40.0,
    step_ms: float = 10.0,
    time_range_ms: Tuple[float, float] = (-100.0, 400.0),
    baseline_win: Tuple[float, float] = DEFAULT_BASELINE_WINDOW_MS,
    nw: float = DEFAULT_MULTITAPER_NW,
    k: int = DEFAULT_MULTITAPER_K,
    save_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    High-Temporal-Precision Stimulation Artifact Disambiguation Analysis.

    Evaluates graded pulse count conditions (e.g., 1p, 2p, 4p, 8p, 20p, 40p at 400 Hz)
    against matched Reach Control across Utah arrays (M1i, M1s, PMd, SMA).
    
    Computes time-resolved sliding-window multitaper excess power:
        ΔdB(t) = dB_stim(t) - dB_control(t)
    with narrow sliding windows (win=40ms, step=10ms) across High Gamma (65-115 Hz) vs Beta (13-25 Hz).
    
    Temporal Disambiguation Principle:
    - Electrical Stim Artifact & Filter Ringing:
      Dominates high frequencies (>65 Hz). An artifact MUST terminate immediately (within ~15-20 ms)
      after pulse train cessation (1p: 2.5ms, 2p: 5ms, 4p: 10ms, 8p: 20ms, 20p: 50ms, 40p: 100ms).
      In High Gamma, excess power collapses back to baseline right after stim offset.
    - Genuine Neurophysiological Cortical Modulation:
      In Beta (13-25 Hz), modulation outlasts stimulation cessation by hundreds of milliseconds,
      showing sustained excess perturbation (+3 to +6 dB) across 150 to 350 ms.
    """
    ctrl_conds = [c for c in conditions if c.is_control]
    if not ctrl_conds:
        _log("   [Disambiguation] No control condition found. Skipping.")
        return None
    ctrl_c = ctrl_conds[0]

    # Filter stim conditions with known pulse counts (e.g. 1, 2, 4, 8, 20, 40)
    stim_conds = [c for c in conditions if not c.is_control and c.pulse_count is not None and c.pulse_count > 0]
    if not stim_conds:
        _log("   [Disambiguation] No stim pulse conditions found. Skipping.")
        return None
    stim_conds.sort(key=lambda c: (c.pulse_count or 0, c.stim_dur_ms or 0))

    # Condition styles (palette, duration in ms, label)
    palette_map = {
        1: ("#8e44ad", 2.5, "1p (2.5ms)"),
        2: ("#2980b9", 5.0, "2p (5.0ms)"),
        4: ("#16a085", 10.0, "4p (10.0ms)"),
        8: ("#27ae60", 20.0, "8p (20.0ms)"),
        20: ("#e67e22", 50.0, "20p (50.0ms)"),
        40: ("#c0392b", 100.0, "40p (100.0ms)"),
    }

    half_w = win_ms / 2.0
    t_centers = np.arange(time_range_ms[0] + half_w, time_range_ms[1] - half_w + 1e-5, step_ms)
    n_times = len(t_centers)

    bands_eval = {
        "HighGamma (70–110 Hz)": (70.0, 110.0),
        "Beta (13–25 Hz)": (13.0, 25.0),
    }

    fig, axes = plt.subplots(len(DEFAULT_ARRAYS), 2, figsize=(15, 12), dpi=FIG_DPI, sharex=True)
    if len(DEFAULT_ARRAYS) == 1:
        axes = np.array([axes])

    for row_idx, array_name in enumerate(DEFAULT_ARRAYS):
        ch_idx = region_map.get(array_name, [])
        if not ch_idx:
            for col_idx in range(2):
                axes[row_idx, col_idx].text(0.5, 0.5, f"No channels for {array_name}", ha="center", va="center")
            continue

        for col_idx, (b_title, band_range) in enumerate(bands_eval.items()):
            ax = axes[row_idx, col_idx]

            # Compute control power curve across sliding windows
            ctrl_time_pow = np.zeros(n_times)
            for t_i, t_c in enumerate(t_centers):
                w_resp = (t_c - half_w, t_c + half_w)
                db_ctrl = extract_trial_multitaper_band_power(
                    broadband=ctrl_c.broadband,
                    time_ms=ctrl_c.time_ms,
                    fs=ctrl_c.fs,
                    ch_indices=ch_idx,
                    band=band_range,
                    baseline_win=baseline_win,
                    resp_win=w_resp,
                    win_ms=win_ms,
                    nw=nw,
                    k=k,
                )
                ctrl_time_pow[t_i] = float(np.nanmean(db_ctrl))

            # Compute and plot stim condition curves
            for sc in stim_conds:
                p_cnt = sc.pulse_count or 1
                color, dur_ms, dur_lbl = palette_map.get(p_cnt, ("#555555", sc.stim_dur_ms or 10.0, f"{p_cnt}p"))
                lbl = f"BR_{sc.br_idx} {dur_lbl}"

                stim_time_pow = np.zeros(n_times)
                for t_i, t_c in enumerate(t_centers):
                    w_resp = (t_c - half_w, t_c + half_w)
                    db_stim = extract_trial_multitaper_band_power(
                        broadband=sc.broadband,
                        time_ms=sc.time_ms,
                        fs=sc.fs,
                        ch_indices=ch_idx,
                        band=band_range,
                        baseline_win=baseline_win,
                        resp_win=w_resp,
                        win_ms=win_ms,
                        nw=nw,
                        k=k,
                    )
                    stim_time_pow[t_i] = float(np.nanmean(db_stim))

                delta_dB = stim_time_pow - ctrl_time_pow

                ax.plot(t_centers, delta_dB, color=color, linewidth=2.0, label=lbl, alpha=0.9)
                # Dotted line marking stimulation train offset
                ax.axvline(dur_ms, color=color, linestyle=":", linewidth=1.2, alpha=0.6)

            ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.axvline(0.0, color="black", linestyle="-", linewidth=1.2, alpha=0.9)

            # Shaded regions
            if col_idx == 0:
                ax.axvspan(0.0, 100.0, color="red", alpha=0.06, label="Stim Delivery (0-100ms)")
            else:
                ax.axvspan(150.0, 350.0, color="green", alpha=0.07, label="Clean Post-Stim Window (150-350ms)")

            ax.set_ylabel(f"{array_name}\nExcess Power (ΔdB)", fontsize=9.5, fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.35)
            ax.set_xlim(time_range_ms[0], time_range_ms[1])

            if row_idx == 0:
                if SHOW_TITLES and SHOW_SUBPLOT_TITLES:
                    sub_header = (
                        "Electrical Artifact & Filter Ringing (Collapses Post-Stim)"
                        if col_idx == 0 else
                        "Genuine Motor Cortical Perturbation (Sustained Modulation)"
                    )
                    ax.set_title(f"{b_title}\n{sub_header}", fontsize=11, fontweight="bold")

            if row_idx == len(DEFAULT_ARRAYS) - 1:
                ax.set_xlabel("Time from Stimulus Onset (ms)", fontsize=10, fontweight="bold")

            if row_idx == 0 and col_idx == 1:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    if SHOW_TITLES and SHOW_SUPTITLE:
        fig.suptitle(
            f"High-Temporal-Precision Stimulation Artifact Disambiguation across Utah Arrays\n"
            f"Session Target {target} | Sliding Multitaper (win={int(win_ms)}ms, step={int(step_ms)}ms, NW={nw}, K={k})\n"
            f"ΔdB = dB(Stim) - dB(Matched Reach Control) | Vertical dotted lines = Condition stim train offsets",
            fontsize=12,
            fontweight="bold",
            y=0.99,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Stim Artifact Disambiguation figure -> {save_path.name}")
    return fig


# =============================================================================
# Time-Binned Spectral Dynamics (Condition vs 20 ms Time Bins)
# =============================================================================

def compute_time_binned_condition_dynamics(
    conditions: List[LfpCondition],
    ch_indices: List[int],
    band: Tuple[float, float],
    baseline_win: Tuple[float, float] = DEFAULT_BASELINE_WINDOW_MS,
    time_range_ms: Tuple[float, float] = DYNAMICS_TIME_RANGE_MS,
    bin_width_ms: float = DYNAMICS_BIN_WIDTH_MS,
    win_ms: float = DYNAMICS_MULTITAPER_WIN_MS,
    nw: float = DEFAULT_MULTITAPER_NW,
    k: int = DEFAULT_MULTITAPER_K,
    use_fdr: bool = DYNAMICS_USE_FDR,
    alpha: float = DYNAMICS_ALPHA,
) -> Dict[str, Any]:
    """
    Compute time-binned spectral power dynamics across conditions for a single cortical array and band.
    
    Extracts baseline-normalized power (dB) across consecutive 20 ms bins aligned to 0 ms,
    averages across the electrodes of the array for each trial, and computes:
      1. Within-condition potentiation vs baseline (1-sample t-test testing whether trial dB != 0)
      2. Stimulation condition vs matched control difference (Welch's two-sample t-test)
      3. Benjamini-Hochberg FDR correction across time bins.
    """
    t_min, t_max = time_range_ms
    # Align bin edges to 0.0 ms exactly
    neg_edges = np.arange(0.0, t_min - 1e-6, -bin_width_ms)[::-1]
    pos_edges = np.arange(0.0, t_max + 1e-6, bin_width_ms)
    bin_edges = np.unique(np.concatenate([neg_edges, pos_edges]))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins = len(bin_centers)

    eff_win_ms = max(win_ms, bin_width_ms)
    half_win = eff_win_ms / 2.0

    cond_trials_db: Dict[int, np.ndarray] = {}

    for c in conditions:
        sub_data = c.broadband[:, ch_indices, :].astype(np.float32)
        n_trials, n_ch, n_time = sub_data.shape
        signals_flat = sub_data.reshape(n_trials * n_ch, -1)

        # 1. Baseline PSD
        base_mask = (c.time_ms >= baseline_win[0]) & (c.time_ms <= baseline_win[1])
        if not np.any(base_mask):
            base_mask = (c.time_ms >= DEFAULT_BASELINE_WINDOW_MS[0]) & (c.time_ms <= DEFAULT_BASELINE_WINDOW_MS[1])
        if not np.any(base_mask):
            base_mask = (c.time_ms >= c.time_ms[0]) & (c.time_ms <= c.time_ms[0] + 200.0)

        base_seg = signals_flat[:, base_mask]
        psd_base, freqs = compute_multitaper_window_psd(
            base_seg, fs=c.fs, win_ms=DEFAULT_MULTITAPER_WIN_MS, nw=nw, k=k
        )

        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        if not np.any(band_mask):
            band_mask = (freqs >= 13.0) & (freqs <= 25.0)

        p_base = np.nanmean(psd_base[:, band_mask], axis=-1)
        p_base = np.maximum(p_base, 1e-12).reshape(n_trials, n_ch)

        # 2. Time-binned response PSDs
        trial_bin_db = np.zeros((n_trials, n_bins), dtype=np.float32)

        for b_idx, t_c in enumerate(bin_centers):
            resp_mask = (c.time_ms >= (t_c - half_win)) & (c.time_ms <= (t_c + half_win))
            if not np.any(resp_mask):
                nearest_idx = np.argmin(np.abs(c.time_ms - t_c))
                resp_mask = slice(max(0, nearest_idx - 10), min(len(c.time_ms), nearest_idx + 10))

            resp_seg = signals_flat[:, resp_mask]
            psd_resp, _ = compute_multitaper_window_psd(
                resp_seg, fs=c.fs, win_ms=eff_win_ms, nw=nw, k=k
            )
            p_resp = np.nanmean(psd_resp[:, band_mask], axis=-1)
            p_resp = np.maximum(p_resp, 1e-12).reshape(n_trials, n_ch)

            ch_db = 10.0 * np.log10(p_resp / p_base)

            # Impute isolated non-finite channels with median across trials for that channel
            bad = ~np.isfinite(ch_db)
            if np.any(bad):
                for col in range(n_ch):
                    vals = ch_db[:, col]
                    v_fin = vals[np.isfinite(vals)]
                    med = float(np.median(v_fin)) if len(v_fin) > 0 else 0.0
                    ch_db[~np.isfinite(vals), col] = med

            # Average across array electrodes for each trial
            trial_bin_db[:, b_idx] = np.nanmean(ch_db, axis=1)

        cond_trials_db[c.br_idx] = trial_bin_db

    # 3. Within-condition statistics (vs Baseline, 0 dB null)
    n_conds = len(conditions)
    within_matrix = np.zeros((n_conds, n_bins), dtype=float)
    within_p_matrix = np.ones((n_conds, n_bins), dtype=float)
    within_q_matrix = np.ones((n_conds, n_bins), dtype=float)

    for r_idx, c in enumerate(conditions):
        t_db = cond_trials_db[c.br_idx]
        within_matrix[r_idx, :] = np.nanmean(t_db, axis=0)

        p_row = np.ones(n_bins, dtype=float)
        for b_idx in range(n_bins):
            v = t_db[:, b_idx]
            v_fin = v[np.isfinite(v)]
            if len(v_fin) >= 2:
                try:
                    _, p_val = stats.ttest_1samp(v_fin, popmean=0.0)
                    p_row[b_idx] = p_val if np.isfinite(p_val) else 1.0
                except Exception:
                    p_row[b_idx] = 1.0

        within_p_matrix[r_idx, :] = p_row

        if use_fdr:
            try:
                q_row = stats.false_discovery_control(p_row, method="bh")
            except Exception:
                order = np.argsort(p_row)
                ranks = np.arange(1, len(p_row) + 1)
                q_row = np.minimum.accumulate((p_row[order] * len(p_row) / ranks)[::-1])[::-1]
                q_row = np.clip(q_row, 0.0, 1.0)
                q_orig = np.empty_like(q_row)
                q_orig[order] = q_row
                q_row = q_orig
            within_q_matrix[r_idx, :] = q_row
        else:
            within_q_matrix[r_idx, :] = p_row

    within_sig_matrix = within_q_matrix < alpha if use_fdr else within_p_matrix < alpha

    # 4. Condition vs Matched Control statistics (Stim - Control ΔdB)
    ctrl_conds = [c for c in conditions if c.is_control]
    stim_conds = [c for c in conditions if not c.is_control]
    has_ctrl = len(ctrl_conds) > 0 and len(stim_conds) > 0

    vs_ctrl_matrix = None
    vs_ctrl_p_matrix = None
    vs_ctrl_q_matrix = None
    vs_ctrl_sig_matrix = None
    stim_labels = []

    if has_ctrl:
        ctrl_c = max(ctrl_conds, key=lambda c: (10 if c.br_idx in (18, 1) else 1, c.n_trials))
        ctrl_t_db = cond_trials_db[ctrl_c.br_idx]
        ctrl_mean_curve = np.nanmean(ctrl_t_db, axis=0)

        n_stim = len(stim_conds)
        vs_ctrl_matrix = np.zeros((n_stim, n_bins), dtype=float)
        vs_ctrl_p_matrix = np.ones((n_stim, n_bins), dtype=float)
        vs_ctrl_q_matrix = np.ones((n_stim, n_bins), dtype=float)
        stim_labels = [sc.clean_label for sc in stim_conds]

        for r_idx, sc in enumerate(stim_conds):
            stim_t_db = cond_trials_db[sc.br_idx]
            vs_ctrl_matrix[r_idx, :] = np.nanmean(stim_t_db, axis=0) - ctrl_mean_curve

            p_row = np.ones(n_bins, dtype=float)
            for b_idx in range(n_bins):
                v_stim = stim_t_db[:, b_idx]
                v_stim_fin = v_stim[np.isfinite(v_stim)]
                v_ctrl = ctrl_t_db[:, b_idx]
                v_ctrl_fin = v_ctrl[np.isfinite(v_ctrl)]

                if len(v_stim_fin) >= 2 and len(v_ctrl_fin) >= 2:
                    try:
                        _, p_val = stats.ttest_ind(v_stim_fin, v_ctrl_fin, equal_var=False)
                        p_row[b_idx] = p_val if np.isfinite(p_val) else 1.0
                    except Exception:
                        p_row[b_idx] = 1.0

            vs_ctrl_p_matrix[r_idx, :] = p_row

            if use_fdr:
                try:
                    q_row = stats.false_discovery_control(p_row, method="bh")
                except Exception:
                    order = np.argsort(p_row)
                    ranks = np.arange(1, len(p_row) + 1)
                    q_row = np.minimum.accumulate((p_row[order] * len(p_row) / ranks)[::-1])[::-1]
                    q_row = np.clip(q_row, 0.0, 1.0)
                    q_orig = np.empty_like(q_row)
                    q_orig[order] = q_row
                    q_row = q_orig
                vs_ctrl_q_matrix[r_idx, :] = q_row
            else:
                vs_ctrl_q_matrix[r_idx, :] = p_row

        vs_ctrl_sig_matrix = vs_ctrl_q_matrix < alpha if use_fdr else vs_ctrl_p_matrix < alpha

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "conditions": conditions,
        "cond_labels": [c.clean_label for c in conditions],
        "within_matrix": within_matrix,
        "within_p_matrix": within_p_matrix,
        "within_q_matrix": within_q_matrix,
        "within_sig_matrix": within_sig_matrix,
        "has_control": has_ctrl,
        "stim_conditions": stim_conds,
        "stim_labels": stim_labels,
        "vs_ctrl_matrix": vs_ctrl_matrix,
        "vs_ctrl_p_matrix": vs_ctrl_p_matrix,
        "vs_ctrl_q_matrix": vs_ctrl_q_matrix,
        "vs_ctrl_sig_matrix": vs_ctrl_sig_matrix,
    }


def plot_binned_dynamics_heatmap(
    matrix_data: np.ndarray,
    sig_matrix: np.ndarray,
    p_matrix: np.ndarray,
    q_matrix: np.ndarray,
    bin_edges: np.ndarray,
    bin_centers: np.ndarray,
    condition_labels: List[str],
    array_name: str,
    band_name: str,
    band_range: Tuple[float, float],
    target: str,
    is_vs_control: bool,
    baseline_win: Tuple[float, float] = DEFAULT_BASELINE_WINDOW_MS,
    soft_mask: bool = DYNAMICS_SOFT_MASK,
    show_stars: bool = DYNAMICS_SHOW_STARS,
    mask_color: str = DYNAMICS_MASK_COLOR,
    use_fdr: bool = DYNAMICS_USE_FDR,
    alpha: float = DYNAMICS_ALPHA,
    multitaper_info: str = "",
    save_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Render publication-grade 2D Condition vs 20 ms Time Bin Heatmap.
    """
    n_conds, n_bins = matrix_data.shape
    if n_conds == 0 or n_bins == 0:
        return None

    fig_w = 14.5
    fig_h = max(4.0, 1.2 + 0.55 * n_conds)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=FIG_DPI)

    # Determine symmetric colormap range
    fin = matrix_data[np.isfinite(matrix_data)]
    abs_max = float(np.nanpercentile(np.abs(fin), 98)) if len(fin) > 0 else 3.0
    vlim = max(1.5, min(12.0, float(np.ceil(abs_max * 2.0) / 2.0)))

    extent = [float(bin_edges[0]), float(bin_edges[-1]), n_conds - 0.5, -0.5]
    im = ax.imshow(
        matrix_data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        extent=extent,
        interpolation="nearest",
        zorder=1,
    )

    # Overlay mask for non-significant bins & asterisks for significant bins
    for r in range(n_conds):
        for c in range(n_bins):
            val = matrix_data[r, c]
            is_sig = sig_matrix[r, c]
            if not is_sig:
                if soft_mask:
                    # Soft mask: translucent white overlay
                    rect = Rectangle(
                        (bin_edges[c], r - 0.5),
                        bin_edges[c + 1] - bin_edges[c],
                        1.0,
                        facecolor=mask_color,
                        alpha=0.68,
                        edgecolor="none",
                        zorder=3,
                    )
                    ax.add_patch(rect)
                else:
                    # Hard/full mask: fully opaque mask (solid white / background color)
                    rect = Rectangle(
                        (bin_edges[c], r - 0.5),
                        bin_edges[c + 1] - bin_edges[c],
                        1.0,
                        facecolor=mask_color,
                        alpha=1.0,
                        edgecolor=mask_color,
                        linewidth=0.25,
                        zorder=3,
                    )
                    ax.add_patch(rect)
            elif show_stars:
                stat_val = q_matrix[r, c] if use_fdr else p_matrix[r, c]
                stars = "***" if stat_val < 0.001 else ("**" if stat_val < 0.01 else "*")
                star_color = "white" if abs(val) > 0.62 * vlim else "#111111"
                ax.text(
                    bin_centers[c],
                    r,
                    stars,
                    ha="center",
                    va="center",
                    color=star_color,
                    fontsize=9,
                    fontweight="bold",
                    zorder=4,
                )

    # Subtle horizontal dividers between condition rows
    for r in range(1, n_conds):
        ax.axhline(r - 0.5, color="#c8c8c8", linewidth=0.75, linestyle=":", zorder=5)

    # Vertical stimulation onset dashed line at 0 ms
    ax.axvline(0.0, color="#111111", linestyle="--", linewidth=1.5, alpha=0.9, zorder=6)

    # Colorbar on the right
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02, aspect=20)
    cbar_label = (
        "Excess Perturbation vs Control (ΔdB)"
        if is_vs_control
        else "LFP Band Power Change vs Baseline (dB)"
    )
    cbar.set_label(cbar_label, fontsize=10, fontweight="bold", labelpad=10)
    cbar.ax.tick_params(labelsize=9)

    # Axes styling
    ax.set_yticks(np.arange(n_conds))
    ax.set_yticklabels(condition_labels, fontsize=10, fontweight="medium")

    # Major X ticks every 100 ms, minor ticks at all bin edges
    x_ticks = np.arange(int(np.ceil(bin_edges[0] / 100.0) * 100), int(bin_edges[-1]) + 1, 100)
    ax.set_xticks(x_ticks)
    ax.set_xticks(bin_edges, minor=True)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("Time Relative to Stimulation Onset (ms)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Condition", fontsize=11, fontweight="bold", labelpad=8)

    # Title & Subtitles honoring user title switches
    region_display = get_region_display_name(array_name)
    reg_color = get_region_color(array_name)
    comp_type = (
        "Excess Perturbation vs Reach Control (ΔΔdB)"
        if is_vs_control
        else "Within-Condition Potentiation vs Pre-Reach Baseline (ΔdB)"
    )
    title_text = f"{region_display} [{array_name}] — {band_name} Band ({band_range[0]}–{band_range[1]} Hz)\n{comp_type} | 20 ms Time Bins"

    if SHOW_TITLES and SHOW_SUBPLOT_TITLES:
        ax.set_title(title_text, fontsize=12, fontweight="bold", pad=12, color=reg_color)

    # Diagnostic stats footnote
    crit = 'Benjamini-Hochberg FDR q < ' + str(alpha) if use_fdr else 'Uncorrected p < ' + str(alpha)
    if show_stars:
        stats_note = (
            f"Stats: {crit} (* q<0.05, ** q<0.01, *** q<0.001) | Bins: 20 ms | Baseline: [{baseline_win[0]:.0f}, {baseline_win[1]:.0f}] ms"
        )
    else:
        stats_note = (
            f"Stats: {crit} (colored = significant, masked = non-significant) | Bins: 20 ms | Baseline: [{baseline_win[0]:.0f}, {baseline_win[1]:.0f}] ms"
        )
    if multitaper_info:
        stats_note += f" | {multitaper_info}"
    ax.text(0.0, -0.16, stats_note, transform=ax.transAxes, fontsize=8.5, color="#555555", ha="left", va="top")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Time-Binned Dynamics Heatmap -> {save_path.name}")

        # Export corresponding CSV tables
        try:
            clean_cols = [f"{bc:.0f}ms" for bc in bin_centers]
            df_vals = pd.DataFrame(matrix_data, index=condition_labels, columns=clean_cols)
            df_vals.to_csv(save_path.with_suffix(".csv"))
            df_p = pd.DataFrame(p_matrix, index=condition_labels, columns=clean_cols)
            df_p.to_csv(save_path.parent / f"{save_path.stem}_pval.csv")
            df_q = pd.DataFrame(q_matrix, index=condition_labels, columns=clean_cols)
            df_q.to_csv(save_path.parent / f"{save_path.stem}_qval.csv")
        except Exception as e:
            _log(f"Warning: Could not save CSV tables for {save_path.name}: {e}")

    return fig


# =============================================================================
# Execution Pipeline per Target & Window
# =============================================================================

def run_lfp_rsa_for_target(
    session: str,
    target: str,
    lfp_dir: Path,
    save_dir: Path,
    compare_conditions: Optional[Sequence[int]] = None,
    control_br_indices: Optional[Sequence[int]] = None,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    selected_bands: Optional[Sequence[str]] = None,
    windows: Optional[Dict[str, Tuple[float, float]]] = None,
    selected_windows: Optional[Sequence[str]] = None,
    metric: str = "euclidean",
    baseline_win: Tuple[float, float] = DEFAULT_BASELINE_WINDOW_MS,
    win_ms: float = DEFAULT_MULTITAPER_WIN_MS,
    nw: float = DEFAULT_MULTITAPER_NW,
    k: int = DEFAULT_MULTITAPER_K,
    subtract_control: bool = False,
    metadata_csv: Optional[Path] = None,
    plot_silhouette: bool = GENERATE_SILHOUETTE_FIGURES,
    plot_rdm: bool = GENERATE_TRIAL_RDM_FIGURES,
    plot_combined_rdm: bool = GENERATE_COMBINED_4ARRAY_RDM,
    plot_subtraction: bool = GENERATE_SUBTRACTION_FIGURES,
    plot_response_magnitude: bool = GENERATE_RESPONSE_MAGNITUDE_FIGURES,
    plot_disambiguation: bool = GENERATE_DISAMBIGUATION_FIGURE,
    plot_single_array_windows: bool = GENERATE_SINGLE_ARRAY_WINDOWS_FIGURE,
    single_array_target: str = SINGLE_ARRAY_WINDOWS_TARGET,
    plot_dynamics_heatmaps: bool = GENERATE_CONDITION_TIME_HEATMAPS,
    dynamics_target_arrays: Union[str, List[str]] = DYNAMICS_TARGET_ARRAYS,
    dynamics_time_range_ms: Tuple[float, float] = DYNAMICS_TIME_RANGE_MS,
    dynamics_bin_width_ms: float = DYNAMICS_BIN_WIDTH_MS,
    dynamics_win_ms: float = DYNAMICS_MULTITAPER_WIN_MS,
    dynamics_soft_mask: bool = DYNAMICS_SOFT_MASK,
    dynamics_show_stars: bool = DYNAMICS_SHOW_STARS,
    dynamics_mask_color: str = DYNAMICS_MASK_COLOR,
    dynamics_use_fdr: bool = DYNAMICS_USE_FDR,
    dynamics_alpha: float = DYNAMICS_ALPHA,
    dynamics_bands: Optional[List[str]] = DYNAMICS_BANDS,
    n_perm: int = N_PERMUTATIONS,
) -> Dict[str, Any]:
    """
    Execute complete single-session LFP RSA pipeline for one reaching target across
    requested temporal analysis windows and frequency bands.
    """
    target_str = target.upper()
    _log(f"\n=======================================================")
    _log(f"Running LFP Multitaper RSA: Session '{session}', Target {target_str}")
    _log(f"   Method: DPSS Multitaper (win={win_ms}ms, NW={nw}, K={k} tapers)")
    _log(f"   Baseline: {baseline_win} ms")
    _log(f"   Subtract Matched Control: {subtract_control}")
    _log(f"   Plot Settings: Silhouette={plot_silhouette}, Trial RDM={plot_rdm}, Combined 4-Array RDM={plot_combined_rdm}, Single Array Windows={plot_single_array_windows} (Target: {single_array_target}), Dynamics Heatmaps={plot_dynamics_heatmaps} (Target: {dynamics_target_arrays}), Subtraction={plot_subtraction}, Response Mag={plot_response_magnitude}, Disambiguation={plot_disambiguation}")
    _log(f"=======================================================")

    ensure_dir(save_dir)

    # 1. Load conditions once per target
    conditions = load_lfp_conditions_for_target(
        session=session,
        target=target_str,
        lfp_dir=lfp_dir,
        compare_conditions=compare_conditions,
        control_br_indices=control_br_indices,
        metadata_csv=metadata_csv,
    )

    if len(conditions) < 2:
        _log(f"Insufficient conditions ({len(conditions)}) for Target {target_str} RSA. Skipping.")
        return {}

    # 2. Build electrode-to-region mapping once per target
    ua_ids = conditions[0].ua_ids
    region_map = build_region_channel_indices(ua_ids)
    for reg, idxs in region_map.items():
        _log(f"   Region {reg}: {len(idxs)} channels")

    if bands is None:
        bands = DEFAULT_BANDS

    # Resolve bands to run (default: all bands)
    if selected_bands is not None:
        bands_to_run = [b for b in selected_bands if b in bands]
    else:
        bands_to_run = list(bands.keys())

    # Resolve windows to run (default: all windows in RSA_ANALYSIS_WINDOWS)
    if windows is not None:
        active_windows = windows
    elif selected_windows is not None and "all" not in [w.lower() for w in selected_windows]:
        active_windows = {k: v for k, v in RSA_ANALYSIS_WINDOWS.items() if k in selected_windows}
        if not active_windows:
            active_windows = RSA_ANALYSIS_WINDOWS
    else:
        active_windows = RSA_ANALYSIS_WINDOWS

    cond_tag = "_".join(str(c.br_idx) for c in conditions)
    multitaper_info = f"DPSS (win={int(win_ms)}ms, NW={nw}, K={k})"

    _log(f"   Processing {len(active_windows)} analysis windows: {list(active_windows.keys())}")
    _log(f"   Processing {len(bands_to_run)} frequency bands: {bands_to_run}")

    all_window_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    need_rsa_computation = (
        plot_silhouette
        or plot_rdm
        or plot_combined_rdm
        or plot_single_array_windows
        or plot_subtraction
        or plot_response_magnitude
    )

    windows_to_process = active_windows if need_rsa_computation else {}
    if not need_rsa_computation:
        _log(f"   [RSA Analysis Windows] Skipped (all RSA and Silhouette figures disabled).")

    # 3. Outer Loop: Temporal Analysis Windows
    for win_tag, resp_win in windows_to_process.items():
        _log(f"\n=======================================================")
        _log(f"Target {target_str} | Temporal Window: {win_tag} [{resp_win[0]}, {resp_win[1]}] ms")
        _log(f"=======================================================")

        win_save_dir = save_dir / win_tag
        ensure_dir(win_save_dir)

        band_results: Dict[str, Dict[str, Any]] = {}

        # 4. Inner Loop: Frequency Bands
        for band_name in bands_to_run:
            band_range = bands[band_name]
            _log(f"\n--- Band: {band_name} ({band_range[0]} - {band_range[1]} Hz) | Window: {win_tag} ---")

            results_by_array: Dict[str, Dict[str, Any]] = {}

            for array_name in DEFAULT_ARRAYS:
                ch_idxs = region_map.get(array_name, [])
                if len(ch_idxs) < 2:
                    _log(f"Array {array_name} has too few channels ({len(ch_idxs)}). Skipping.")
                    continue

                _log(f"   Array: {array_name} ({len(ch_idxs)} ch) -> computing multitaper RDM...")

                RDM, block_sizes, block_labels, feat_list, ctrl_mean = build_lfp_rdm(
                    conditions=conditions,
                    ch_indices=ch_idxs,
                    band=band_range,
                    metric=metric,
                    baseline_win=baseline_win,
                    resp_win=resp_win,
                    win_ms=win_ms,
                    nw=nw,
                    k=k,
                    subtract_control=subtract_control,
                )

                S, P, Q = compute_pairwise_silhouette_matrix(RDM, block_sizes, n_perm=n_perm)
                within_dist = compute_within_condition_consistency(RDM, block_sizes)

                results_by_array[array_name] = {
                    "RDM": RDM,
                    "S": S,
                    "P": P,
                    "Q": Q,
                    "within_dist": within_dist,
                    "block_sizes": block_sizes,
                    "block_labels": block_labels,
                    "ch_indices": ch_idxs,
                    "fixed_params_summary": getattr(conditions[0], "fixed_params_summary", "") if conditions else "",
                }

                # 1. Save individual trial-by-trial RDM Heatmap
                if plot_rdm:
                    out_rdm = win_save_dir / f"lfp_rdm_Target_{target_str}_{array_name}_{band_name}_{win_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
                    plot_lfp_rdm_figure(
                        RDM=RDM,
                        block_sizes=block_sizes,
                        block_labels=block_labels,
                        array_name=array_name,
                        band_name=band_name,
                        target=target_str,
                        window_name=win_tag,
                        resp_win=resp_win,
                        baseline_win=baseline_win,
                        metric_label="Distance (Euclidean dB)" if metric == "euclidean" else "Dissimilarity (1 - r)",
                        multitaper_info=multitaper_info,
                        save_path=out_rdm,
                    )

                # Save pairwise silhouette, p-value, and q-value tables
                clean_labs = [re.sub(r'\s*\(n=\d+\)', '', lab) for lab in block_labels]
                df_sil = pd.DataFrame(S, index=clean_labs, columns=clean_labs)
                df_sil.to_csv(win_save_dir / f"lfp_silhouette_Target_{target_str}_{array_name}_{band_name}_{win_tag}.csv")
                df_p = pd.DataFrame(P, index=clean_labs, columns=clean_labs)
                df_p.to_csv(win_save_dir / f"lfp_pval_Target_{target_str}_{array_name}_{band_name}_{win_tag}.csv")
                df_q = pd.DataFrame(Q, index=clean_labs, columns=clean_labs)
                df_q.to_csv(win_save_dir / f"lfp_qval_Target_{target_str}_{array_name}_{band_name}_{win_tag}.csv")

            # 2. Combined 4-Array RDM Figure (All 4 arrays on the SAME plot)
            if plot_combined_rdm and results_by_array:
                out_combined_rdm = win_save_dir / f"lfp_rdm_4arrays_Target_{target_str}_{band_name}_{win_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
                plot_combined_4array_rdm_figure(
                    results_by_array=results_by_array,
                    band_name=band_name,
                    band_range=band_range,
                    target=target_str,
                    window_name=win_tag,
                    resp_win=resp_win,
                    baseline_win=baseline_win,
                    metric_label="Euclidean Distance (dB)" if metric == "euclidean" else "Dissimilarity (1 - r)",
                    multitaper_info=multitaper_info,
                    subtract_control=subtract_control,
                    save_path=out_combined_rdm,
                )
                top_rdm = save_dir / f"lfp_rdm_4arrays_Target_{target_str}_{band_name}_{win_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
                try:
                    import shutil
                    shutil.copyfile(out_combined_rdm, top_rdm)
                except Exception:
                    pass

            # 3. Combined 4-Array Silhouette Figure (All 4 arrays on the SAME plot)
            if plot_silhouette and results_by_array:
                out_combined_sil = win_save_dir / f"lfp_silhouette_4arrays_Target_{target_str}_{band_name}_{win_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
                plot_combined_4array_silhouette_figure(
                    results_by_array=results_by_array,
                    band_name=band_name,
                    band_range=band_range,
                    target=target_str,
                    window_name=win_tag,
                    resp_win=resp_win,
                    multitaper_info=multitaper_info,
                    subtract_control=subtract_control,
                    baseline_win=baseline_win,
                    metric=metric,
                    save_path=out_combined_sil,
                )
                # Copy to top-level save_dir for quick reference
                top_sil = save_dir / f"lfp_silhouette_4arrays_Target_{target_str}_{band_name}_{win_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
                try:
                    import shutil
                    shutil.copyfile(out_combined_sil, top_sil)
                except Exception:
                    pass

            # 4. Subtraction Analysis Figure (Stim - Control ΔdB Excess Perturbation across 4 arrays)
            if plot_subtraction:
                out_sub = win_save_dir / f"lfp_subtraction_profiles_Target_{target_str}_{band_name}_{win_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
                plot_subtraction_analysis_figure(
                    conditions=conditions,
                    region_map=region_map,
                    band=band_range,
                    band_name=band_name,
                    target=target_str,
                    window_name=win_tag,
                    resp_win=resp_win,
                    baseline_win=baseline_win,
                    win_ms=win_ms,
                    nw=nw,
                    k=k,
                    save_path=out_sub,
                )
                top_sub = save_dir / f"lfp_subtraction_profiles_Target_{target_str}_{band_name}_{win_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
                try:
                    import shutil
                    shutil.copyfile(out_sub, top_sub)
                except Exception:
                    pass

            # 5. 4-Panel Array Response Magnitude Figure
            if plot_response_magnitude:
                out_resp = win_save_dir / f"lfp_response_magnitude_Target_{target_str}_{band_name}_{win_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
                plot_array_response_magnitude_figure(
                    conditions=conditions,
                    region_map=region_map,
                    band=band_range,
                    band_name=band_name,
                    target=target_str,
                    window_name=win_tag,
                    resp_win=resp_win,
                    baseline_win=baseline_win,
                    win_ms=win_ms,
                    nw=nw,
                    k=k,
                    subtract_control=subtract_control,
                    save_path=out_resp,
                )
                top_resp = save_dir / f"lfp_response_magnitude_Target_{target_str}_{band_name}_{win_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
                try:
                    import shutil
                    shutil.copyfile(out_resp, top_resp)
                except Exception:
                    pass

            band_results[band_name] = results_by_array

        all_window_results[win_tag] = band_results

    # 5b. Single Array Across Temporal Windows Figure (e.g. M1i / M1inferior compared across all active windows)
    if plot_single_array_windows and all_window_results:
        target_arrays = resolve_target_arrays_for_windows(single_array_target)
        _log(f"\n=======================================================")
        _log(f"Generating Single-Array Across-Windows Figures for Target {target_str}: Arrays {target_arrays}")
        _log(f"=======================================================")

        for band_name in bands_to_run:
            band_range = bands[band_name]
            for arr in target_arrays:
                windows_data = {}
                for w_tag, b_res in all_window_results.items():
                    if band_name in b_res and arr in b_res[band_name]:
                        windows_data[w_tag] = b_res[band_name][arr]

                if not windows_data:
                    continue

                if plot_silhouette:
                    out_sil_win = save_dir / f"lfp_silhouette_windows_{arr}_Target_{target_str}_{band_name}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
                    plot_single_array_windows_silhouette_figure(
                        windows_dict=windows_data,
                        array_name=arr,
                        band_name=band_name,
                        band_range=band_range,
                        target=target_str,
                        active_windows=active_windows,
                        baseline_win=baseline_win,
                        multitaper_info=multitaper_info,
                        subtract_control=subtract_control,
                        metric=metric,
                        save_path=out_sil_win,
                    )

                if plot_combined_rdm:
                    out_rdm_win = save_dir / f"lfp_rdm_windows_{arr}_Target_{target_str}_{band_name}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
                    plot_single_array_windows_rdm_figure(
                        windows_dict=windows_data,
                        array_name=arr,
                        band_name=band_name,
                        band_range=band_range,
                        target=target_str,
                        active_windows=active_windows,
                        baseline_win=baseline_win,
                        metric_label="Euclidean Distance (dB)" if metric == "euclidean" else "Dissimilarity (1 - r)",
                        multitaper_info=multitaper_info,
                        subtract_control=subtract_control,
                        save_path=out_rdm_win,
                    )

    # 6. High-Temporal-Precision Stim Artifact Disambiguation Analysis
    if plot_disambiguation:
        _log(f"\n=======================================================")
        _log(f"Running High-Temporal-Precision Stim Artifact Disambiguation: Target {target_str}")
        _log(f"=======================================================")
        disambig_path = save_dir / f"lfp_stim_artifact_disambiguation_Target_{target_str}.{FIG_FORMAT.lower()}"
        plot_time_resolved_stim_artifact_disambiguation(
            conditions=conditions,
            region_map=region_map,
            target=target_str,
            save_dir=save_dir,
            win_ms=40.0,
            step_ms=10.0,
            time_range_ms=(-100.0, 400.0),
            baseline_win=baseline_win,
            nw=nw,
            k=k,
            save_path=disambig_path,
        )
    else:
        _log(f"   [Disambiguation] Skipped per plot settings (GENERATE_DISAMBIGUATION_FIGURE=False).")

    # 7. Time-Binned Spectral Dynamics Heatmaps (Condition vs 20 ms Time Bins)
    if plot_dynamics_heatmaps:
        dynamics_arrays = resolve_target_arrays_for_windows(dynamics_target_arrays)
        _log(f"\n=======================================================")
        _log(f"Generating Time-Binned Spectral Dynamics Heatmaps: Target {target_str}")
        _log(f"   Target Arrays ({len(dynamics_arrays)}): {dynamics_arrays}")
        _log(f"   Time Range: {dynamics_time_range_ms} ms (bin width: {dynamics_bin_width_ms} ms, sliding win: {dynamics_win_ms} ms)")
        _log(f"   Mask Mode: {'Soft' if dynamics_soft_mask else 'Full'} (color={dynamics_mask_color}) | Stars: {dynamics_show_stars} | FDR: {dynamics_use_fdr} (alpha={dynamics_alpha})")
        _log(f"=======================================================")

        dyn_bands_to_run = [b for b in dynamics_bands if b in bands] if dynamics_bands else bands_to_run

        dynamics_dir = save_dir / "time_binned_dynamics"
        ensure_dir(dynamics_dir)

        for arr in dynamics_arrays:
            ch_idxs = region_map.get(arr, [])
            if len(ch_idxs) < 2:
                _log(f"Array {arr} has too few channels ({len(ch_idxs)}). Skipping dynamics.")
                continue

            for band_name in dyn_bands_to_run:
                band_range = bands[band_name]
                band_tag = band_name.replace(" ", "_")
                _log(f"   [Dynamics] Array: {arr} | Band: {band_name} ({band_range[0]}–{band_range[1]} Hz)...")

                dyn_data = compute_time_binned_condition_dynamics(
                    conditions=conditions,
                    ch_indices=ch_idxs,
                    band=band_range,
                    baseline_win=baseline_win,
                    time_range_ms=dynamics_time_range_ms,
                    bin_width_ms=dynamics_bin_width_ms,
                    win_ms=dynamics_win_ms,
                    nw=nw,
                    k=k,
                    use_fdr=dynamics_use_fdr,
                    alpha=dynamics_alpha,
                )

                # Figure 1: Within-Condition Potentiation vs Pre-Reach Baseline
                out_within = dynamics_dir / f"lfp_dynamics_within_cond_Target_{target_str}_{arr}_{band_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
                plot_binned_dynamics_heatmap(
                    matrix_data=dyn_data["within_matrix"],
                    sig_matrix=dyn_data["within_sig_matrix"],
                    p_matrix=dyn_data["within_p_matrix"],
                    q_matrix=dyn_data["within_q_matrix"],
                    bin_edges=dyn_data["bin_edges"],
                    bin_centers=dyn_data["bin_centers"],
                    condition_labels=dyn_data["cond_labels"],
                    array_name=arr,
                    band_name=band_name,
                    band_range=band_range,
                    target=target_str,
                    is_vs_control=False,
                    baseline_win=baseline_win,
                    soft_mask=dynamics_soft_mask,
                    show_stars=dynamics_show_stars,
                    mask_color=dynamics_mask_color,
                    use_fdr=dynamics_use_fdr,
                    alpha=dynamics_alpha,
                    multitaper_info=multitaper_info,
                    save_path=out_within,
                )
                try:
                    import shutil
                    shutil.copyfile(out_within, save_dir / out_within.name)
                except Exception:
                    pass

                # Figure 2: Stimulation Condition vs Matched Control Difference
                if dyn_data["has_control"] and dyn_data["vs_ctrl_matrix"] is not None:
                    out_vs_ctrl = dynamics_dir / f"lfp_dynamics_vs_control_Target_{target_str}_{arr}_{band_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
                    plot_binned_dynamics_heatmap(
                        matrix_data=dyn_data["vs_ctrl_matrix"],
                        sig_matrix=dyn_data["vs_ctrl_sig_matrix"],
                        p_matrix=dyn_data["vs_ctrl_p_matrix"],
                        q_matrix=dyn_data["vs_ctrl_q_matrix"],
                        bin_edges=dyn_data["bin_edges"],
                        bin_centers=dyn_data["bin_centers"],
                        condition_labels=dyn_data["stim_labels"],
                        array_name=arr,
                        band_name=band_name,
                        band_range=band_range,
                        target=target_str,
                        is_vs_control=True,
                        baseline_win=baseline_win,
                        soft_mask=dynamics_soft_mask,
                        show_stars=dynamics_show_stars,
                        mask_color=dynamics_mask_color,
                        use_fdr=dynamics_use_fdr,
                        alpha=dynamics_alpha,
                        multitaper_info=multitaper_info,
                        save_path=out_vs_ctrl,
                    )
                    try:
                        import shutil
                        shutil.copyfile(out_vs_ctrl, save_dir / out_vs_ctrl.name)
                    except Exception:
                        pass

    return {
        "conditions": conditions,
        "region_map": region_map,
        "all_window_results": all_window_results,
    }


# =============================================================================
# CLI & Main Entrypoint
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LFP Representational Similarity Analysis (LFP RSA) for Utah array recordings using Multitaper."
    )
    parser.add_argument("--session", type=str, default=None, help="Session ID (e.g. NRR_RW022).")
    parser.add_argument("--lfp-dir", type=str, default=None, help="Directory containing aligned_lfp__*.npz files.")
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        choices=["auto", "A", "B", "both", "all"],
        help="Reaching target button ('auto' analyzes both A and B).",
    )
    parser.add_argument(
        "--band",
        type=str,
        default="all",
        choices=["all", "both", "auto"] + list(DEFAULT_BANDS.keys()),
        help=f"Frequency band(s) to analyze. Default: 'all' runs across all {list(DEFAULT_BANDS.keys())}.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=str,
        default=["all"],
        help=f"Analysis windows: {list(RSA_ANALYSIS_WINDOWS.keys())}, or 'all' (default: all).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        choices=["euclidean", "correlation", "cosine"],
        help=f"RSA distance metric (default: {DEFAULT_METRIC}). 'euclidean' preserves power magnitude & dose differences; 'correlation' normalizes trial amplitudes.",
    )
    parser.add_argument("--multitaper-win", type=float, default=DEFAULT_MULTITAPER_WIN_MS, help="Multitaper window (ms).")
    parser.add_argument("--multitaper-nw", type=float, default=DEFAULT_MULTITAPER_NW, help="Multitaper NW product.")
    parser.add_argument("--multitaper-k", type=int, default=DEFAULT_MULTITAPER_K, help="Multitaper number of tapers K.")
    parser.add_argument(
        "--subtract-control",
        dest="subtract_control",
        action="store_true",
        default=False,
        help="Subtract matched reach-only control mean response (default: False; potentiation is measured vs pre-reach baseline [-450, -250] ms).",
    )
    parser.add_argument(
        "--no-subtract-control",
        dest="subtract_control",
        action="store_false",
        help="Disable subtracting matched reach-only control mean response.",
    )
    parser.add_argument("--compare-conditions", nargs="+", type=int, default=None, help="Conditions/BR files to compare.")
    parser.add_argument("--control-br", nargs="+", type=int, default=None, help="Explicit control BR indices.")
    parser.add_argument("--save-dir", type=str, default=None, help="Output figure directory.")
    parser.add_argument(
        "--resp-window",
        nargs=2,
        type=float,
        default=None,
        metavar=("START_MS", "END_MS"),
        help="Custom response time window in ms relative to stim onset (overrides --windows).",
    )
    parser.add_argument(
        "--base-window",
        nargs=2,
        type=float,
        default=None,
        metavar=("START_MS", "END_MS"),
        help="Baseline time window in ms relative to stim onset (default: -250 -50).",
    )
    # Plot generation settings CLI overrides
    parser.add_argument("--plot-silhouette", dest="plot_silhouette", action="store_true", default=None, help="Generate 4-array condition silhouette separation figures.")
    parser.add_argument("--no-plot-silhouette", dest="plot_silhouette", action="store_false", help="Disable silhouette figures.")
    parser.add_argument("--plot-rdm", dest="plot_rdm", action="store_true", default=None, help="Generate trial-by-trial RDM heatmaps.")
    parser.add_argument("--no-plot-rdm", dest="plot_rdm", action="store_false", help="Disable trial-by-trial RDM heatmaps.")
    parser.add_argument("--plot-combined-rdm", dest="plot_combined_rdm", action="store_true", default=None, help="Generate combined 4-array trial-by-trial RDM figure.")
    parser.add_argument("--no-plot-combined-rdm", dest="plot_combined_rdm", action="store_false", help="Disable combined 4-array RDM figure.")
    parser.add_argument("--plot-subtraction", dest="plot_subtraction", action="store_true", default=None, help="Generate stim minus control subtraction profiles.")
    parser.add_argument("--no-plot-subtraction", dest="plot_subtraction", action="store_false", help="Disable subtraction figures.")
    parser.add_argument("--plot-response-magnitude", dest="plot_response_magnitude", action="store_true", default=None, help="Generate 4-panel response magnitude figures.")
    parser.add_argument("--no-plot-response-magnitude", dest="plot_response_magnitude", action="store_false", help="Disable response magnitude figures.")
    parser.add_argument("--plot-disambiguation", dest="plot_disambiguation", action="store_true", default=None, help="Generate artifact disambiguation figure.")
    parser.add_argument("--plot-single-array-windows", dest="plot_single_array_windows", action="store_true", default=None, help="Generate single-array comparison across temporal windows.")
    parser.add_argument("--no-plot-single-array-windows", dest="plot_single_array_windows", action="store_false", help="Disable single-array windows figures.")
    parser.add_argument("--single-array-target", type=str, default=None, help="Target array for across-windows plot (e.g. 'M1i', 'M1inferior', 'SMA', 'all').")
    parser.add_argument("--plot-dynamics", dest="plot_dynamics", action="store_true", default=None, help="Generate time-binned spectral dynamics heatmaps (Condition vs 20ms bins).")
    parser.add_argument("--no-plot-dynamics", dest="plot_dynamics", action="store_false", help="Disable time-binned spectral dynamics heatmaps.")
    parser.add_argument("--dynamics-arrays", nargs="+", type=str, default=None, help="Target arrays for dynamics heatmaps (e.g. 'M1i', 'all').")
    parser.add_argument("--dynamics-soft-mask", dest="dynamics_soft_mask", action="store_true", default=None, help="Use soft mask (faded non-significant bins).")
    parser.add_argument("--dynamics-hard-mask", dest="dynamics_soft_mask", action="store_false", help="Use hard mask (greyed out non-significant bins).")
    parser.add_argument("--dynamics-stars", dest="dynamics_show_stars", action="store_true", default=None, help="Display significance stars on dynamics heatmaps.")
    parser.add_argument("--no-dynamics-stars", dest="dynamics_show_stars", action="store_false", help="Hide significance stars on dynamics heatmaps.")
    parser.add_argument("--dynamics-mask-color", type=str, default=None, help="Hex color for masked non-significant bins (default: '#ffffff').")
    parser.add_argument("--dynamics-no-fdr", dest="dynamics_use_fdr", action="store_false", default=None, help="Disable FDR correction (use uncorrected p < 0.05).")
    parser.add_argument("--show-indices", dest="show_indices", action="store_true", default=None, help="Show condition index prefix (e.g. 'BR 11:').")
    parser.add_argument("--no-show-indices", dest="show_indices", action="store_false", help="Hide condition index prefix.")
    parser.add_argument("--show-trial-counts", dest="show_trial_counts", action="store_true", default=None, help="Show trial counts in condition labels.")
    parser.add_argument("--no-show-trial-counts", dest="show_trial_counts", action="store_false", help="Hide trial counts in condition labels.")
    parser.add_argument("--show-titles", dest="show_titles", action="store_true", default=None, help="Show figure titles.")
    parser.add_argument("--no-titles", dest="show_titles", action="store_false", help="Fully turn off all title label generation.")
    parser.add_argument("--show-suptitle", dest="show_suptitle", action="store_true", default=None, help="Show main multi-line suptitle.")
    parser.add_argument("--no-suptitle", dest="show_suptitle", action="store_false", help="Hide main multi-line suptitle.")
    parser.add_argument("--show-subplot-titles", dest="show_subplot_titles", action="store_true", default=None, help="Show individual subplot titles.")
    parser.add_argument("--no-subplot-titles", dest="show_subplot_titles", action="store_false", help="Hide individual subplot titles.")
    parser.add_argument("--n-perm", type=int, default=N_PERMUTATIONS, help=f"Number of permutations for silhouette significance test (default: {N_PERMUTATIONS}).")
    return parser.parse_args()


def main() -> None:
    global SHOW_CONDITION_INDICES, SHOW_TRIAL_COUNTS, SHOW_TITLES, SHOW_SUPTITLE, SHOW_SUBPLOT_TITLES, GENERATE_SINGLE_ARRAY_WINDOWS_FIGURE, SINGLE_ARRAY_WINDOWS_TARGET, GENERATE_CONDITION_TIME_HEATMAPS, DYNAMICS_TARGET_ARRAYS, DYNAMICS_SOFT_MASK, DYNAMICS_SHOW_STARS, DYNAMICS_MASK_COLOR, DYNAMICS_USE_FDR
    args = parse_args()

    if args.show_indices is not None:
        SHOW_CONDITION_INDICES = args.show_indices
    if args.show_trial_counts is not None:
        SHOW_TRIAL_COUNTS = args.show_trial_counts
    if args.show_titles is not None:
        SHOW_TITLES = args.show_titles
    if args.show_suptitle is not None:
        SHOW_SUPTITLE = args.show_suptitle
    if args.show_subplot_titles is not None:
        SHOW_SUBPLOT_TITLES = args.show_subplot_titles
    if args.plot_single_array_windows is not None:
        GENERATE_SINGLE_ARRAY_WINDOWS_FIGURE = args.plot_single_array_windows
    if args.single_array_target is not None:
        SINGLE_ARRAY_WINDOWS_TARGET = args.single_array_target
    if args.plot_dynamics is not None:
        GENERATE_CONDITION_TIME_HEATMAPS = args.plot_dynamics
    if args.dynamics_arrays is not None:
        DYNAMICS_TARGET_ARRAYS = args.dynamics_arrays
    if args.dynamics_soft_mask is not None:
        DYNAMICS_SOFT_MASK = args.dynamics_soft_mask
    if args.dynamics_show_stars is not None:
        DYNAMICS_SHOW_STARS = args.dynamics_show_stars
    if args.dynamics_mask_color is not None:
        DYNAMICS_MASK_COLOR = args.dynamics_mask_color
    if args.dynamics_use_fdr is not None:
        DYNAMICS_USE_FDR = args.dynamics_use_fdr

    # Session resolution
    env_sess = os.environ.get("RCP_SESSION")
    session = args.session or env_sess or (PARAMS.session if PARAMS and hasattr(PARAMS, "session") else "NRR_RW022")

    # Resolve LFP checkpoint directory
    if args.lfp_dir:
        lfp_dir = Path(args.lfp_dir)
    elif OUT_BASE is not None and (OUT_BASE / "checkpoints" / "UA_LFP").exists():
        lfp_dir = OUT_BASE / "checkpoints" / "UA_LFP"
    elif SESSION_LOC is not None and (SESSION_LOC / "results" / "checkpoints" / "UA_LFP").exists():
        lfp_dir = SESSION_LOC / "results" / "checkpoints" / "UA_LFP"
    else:
        lfp_dir = Path.cwd() / "results" / "checkpoints" / "UA_LFP"

    # Resolve Save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    elif OUT_BASE is not None:
        save_dir = OUT_BASE / "figures" / "lfp_rsa"
    elif SESSION_LOC is not None:
        save_dir = SESSION_LOC / "results" / "figures" / "lfp_rsa"
    else:
        save_dir = Path.cwd() / "results" / "figures" / "lfp_rsa"
    ensure_dir(save_dir)

    # Active conditions
    active_process_only = resolve_process_only()
    compare_conds = args.compare_conditions if args.compare_conditions is not None else active_process_only
    control_brs = args.control_br

    base_win = (float(args.base_window[0]), float(args.base_window[1])) if args.base_window else DEFAULT_BASELINE_WINDOW_MS

    # Window resolution
    custom_windows = None
    if args.resp_window:
        custom_windows = {f"custom_{int(args.resp_window[0])}_to_{int(args.resp_window[1])}ms": (float(args.resp_window[0]), float(args.resp_window[1]))}
    elif args.windows and "all" not in [w.lower() for w in args.windows]:
        custom_windows = {w: RSA_ANALYSIS_WINDOWS[w] for w in args.windows if w in RSA_ANALYSIS_WINDOWS}
        if not custom_windows:
            custom_windows = RSA_ANALYSIS_WINDOWS
    else:
        custom_windows = RSA_ANALYSIS_WINDOWS

    # Band resolution
    if args.band.lower() in ("all", "both", "auto"):
        bands_to_run = list(DEFAULT_BANDS.keys())
    else:
        bands_to_run = [args.band]

    # Target resolution
    if args.target.lower() in ("auto", "both", "all"):
        targets = ["A", "B"]
    else:
        targets = [args.target.upper()]

    # Resolve Plot Generation Toggles (CLI overrides global settings)
    plot_silhouette = args.plot_silhouette if args.plot_silhouette is not None else GENERATE_SILHOUETTE_FIGURES
    plot_rdm = args.plot_rdm if args.plot_rdm is not None else GENERATE_TRIAL_RDM_FIGURES
    plot_combined_rdm = args.plot_combined_rdm if args.plot_combined_rdm is not None else GENERATE_COMBINED_4ARRAY_RDM
    plot_subtraction = args.plot_subtraction if args.plot_subtraction is not None else GENERATE_SUBTRACTION_FIGURES
    plot_response_magnitude = args.plot_response_magnitude if args.plot_response_magnitude is not None else GENERATE_RESPONSE_MAGNITUDE_FIGURES
    plot_disambiguation = args.plot_disambiguation if args.plot_disambiguation is not None else GENERATE_DISAMBIGUATION_FIGURE
    plot_single_array_windows = args.plot_single_array_windows if args.plot_single_array_windows is not None else GENERATE_SINGLE_ARRAY_WINDOWS_FIGURE
    single_array_target = args.single_array_target if args.single_array_target is not None else SINGLE_ARRAY_WINDOWS_TARGET
    plot_dynamics = args.plot_dynamics if args.plot_dynamics is not None else GENERATE_CONDITION_TIME_HEATMAPS
    dynamics_target = args.dynamics_arrays if args.dynamics_arrays is not None else DYNAMICS_TARGET_ARRAYS
    dynamics_soft = args.dynamics_soft_mask if args.dynamics_soft_mask is not None else DYNAMICS_SOFT_MASK
    dynamics_stars = args.dynamics_show_stars if args.dynamics_show_stars is not None else DYNAMICS_SHOW_STARS
    dynamics_mask_col = args.dynamics_mask_color if args.dynamics_mask_color is not None else DYNAMICS_MASK_COLOR
    dynamics_fdr = args.dynamics_use_fdr if args.dynamics_use_fdr is not None else DYNAMICS_USE_FDR
    n_perm = int(args.n_perm)

    _log(f"Starting LFP Multitaper RSA Analysis:")
    _log(f"   Session: {session}")
    _log(f"   LFP Dir: {lfp_dir}")
    _log(f"   Save Dir: {save_dir}")
    _log(f"   Windows: {list(custom_windows.keys()) if custom_windows else list(RSA_ANALYSIS_WINDOWS.keys())}")
    _log(f"   Bands to Run ({len(bands_to_run)}): {bands_to_run}")
    _log(f"   Metric: {args.metric}")
    _log(f"   Multitaper: win={args.multitaper_win}ms, NW={args.multitaper_nw}, K={args.multitaper_k}")
    _log(f"   Subtract Control: {args.subtract_control}")
    _log(f"   Baseline Window: {base_win} ms")
    _log(f"   Compare Conditions: {compare_conds}")
    _log(f"   Plot Settings: Silhouette={plot_silhouette}, Trial RDM={plot_rdm}, Combined 4-Array RDM={plot_combined_rdm}, Single Array Windows={plot_single_array_windows} (Target: {single_array_target}), Dynamics Heatmaps={plot_dynamics} (Target: {dynamics_target}), Subtraction={plot_subtraction}, Response Mag={plot_response_magnitude}, Disambiguation={plot_disambiguation}")
    _log(f"   N Permutations: {n_perm}")

    for tgt in targets:
        run_lfp_rsa_for_target(
            session=session,
            target=tgt,
            lfp_dir=lfp_dir,
            save_dir=save_dir,
            compare_conditions=compare_conds,
            control_br_indices=control_brs,
            bands=DEFAULT_BANDS,
            selected_bands=bands_to_run,
            windows=custom_windows,
            metric=args.metric,
            baseline_win=base_win,
            win_ms=args.multitaper_win,
            nw=args.multitaper_nw,
            k=args.multitaper_k,
            subtract_control=args.subtract_control,
            metadata_csv=METADATA_CSV,
            plot_silhouette=plot_silhouette,
            plot_rdm=plot_rdm,
            plot_combined_rdm=plot_combined_rdm,
            plot_subtraction=plot_subtraction,
            plot_response_magnitude=plot_response_magnitude,
            plot_disambiguation=plot_disambiguation,
            plot_single_array_windows=plot_single_array_windows,
            single_array_target=single_array_target,
            plot_dynamics_heatmaps=plot_dynamics,
            dynamics_target_arrays=dynamics_target,
            dynamics_soft_mask=dynamics_soft,
            dynamics_show_stars=dynamics_stars,
            dynamics_mask_color=dynamics_mask_col,
            dynamics_use_fdr=dynamics_fdr,
            n_perm=n_perm,
        )

    _log("LFP RSA Pipeline Complete.")


if __name__ == "__main__":
    main()
