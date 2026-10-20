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
import matplotlib.gridspec as gridspec
from scipy import signal, stats

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
DEFAULT_BASELINE_WINDOW_MS: Tuple[float, float] = (-950.0, -650.0)

# 4 Physiological & Methodological Analysis Windows
RSA_ANALYSIS_WINDOWS: Dict[str, Tuple[float, float]] = {
    "-200_to_0ms": (-250.0, -50.0),       # Pre-stim reach initiation / motor preparation phase
    "0_to_150ms": (0.0, 150.0),         # Early stimulus delivery phase (pulses + immediate evoked response)
    "0_to_350ms": (0.0, 350.0),         # Full reach perturbation phase
    "150_to_350ms": (150.0, 350.0),     # Post-stimulus clean reach execution / recovery phase (100% artifact-free)
}
DEFAULT_RESPONSE_WINDOW_MS: Tuple[float, float] = (150.0, 350.0)

# Multitaper parameters (Slepian DPSS tapers)
DEFAULT_MULTITAPER_WIN_MS: float = 100.0   # Window duration in ms
DEFAULT_MULTITAPER_NW: float = 2.0         # Time-halfbandwidth product NW
DEFAULT_MULTITAPER_K: int = 2              # Number of tapers (variance reduction = 1/2)

# Primary frequency bands (avoiding 60 Hz and 120 Hz line noise regions)
DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "Beta": (13.0, 25.0),
    "LowGamma": (25.0, 55.0),
    "HighGamma": (65.0, 115.0),
    "Alpha": (8.0, 12.0),
    "Theta": (4.0, 8.0),
}
NOTCH_REGIONS: List[Tuple[float, float]] = [(55.0, 65.0), (115.0, 125.0)]

DEFAULT_PRIMARY_BAND: str = "LowGamma"
DEFAULT_ARRAYS: List[str] = ["M1i", "M1s", "PMd", "SMA"]

MIN_TRIALS_PER_CONDITION: int = 3
N_PERMUTATIONS: int = 1000
FDR_ALPHA: float = 0.05
FIG_DPI: int = 300
FIG_FORMAT: str = "png"

# Color definitions matching project standard
REGION_COLORS: Dict[str, str] = {
    "M1i": "#2ca02c",   # Green (Primary Motor Deep / Infragranular)
    "M1s": "#d62728",   # Red (Primary Motor Superficial / Supragranular)
    "PMd": "#ff7f0e",   # Orange (Dorsal Premotor)
    "SMA": "#1f77b4",   # Blue (Supplementary Motor Area)
}
COLOR_CTRL: str = "#2c3e50"
COLOR_STIM_DEFAULT: str = "#e74c3c"


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

    @property
    def n_trials(self) -> int:
        return self.broadband.shape[0]

    @property
    def n_channels(self) -> int:
        return self.broadband.shape[1]

    @property
    def n_time(self) -> int:
        return len(self.time_ms)


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
        if br_idx < 0:
            m = re.search(r'_(\d{3})(?:_|$|\.)', sess_id)
            if m:
                br_idx = int(m.group(1))

        if br_idx < 0:
            continue

        csv_meta = session_meta.get(br_idx, {})
        notes = str(csv_meta.get("notes", "")).strip()

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

        # Check Control condition status
        notes_clean = notes.lower() if notes else ""
        sess_clean = sess_id.lower()
        is_control = bool(
            (control_br_indices and br_idx in control_br_indices)
            or bool(data.get("is_control", False))
            or (peri_pulses is not None and peri_pulses == 0)
            or (csv_pulses is not None and csv_pulses == 0)
            or any(kw in sess_clean for kw in ("control", "baseline", "0p", "0pulse", "nostim", "no_stim"))
            or any(kw in notes_clean for kw in ("control", "baseline", "0p", "0pulse", "no stim", "nostim", "no_stim"))
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
                best_ctrl = max(avail_ctrls, key=lambda c: c["n_trials"])
                _log(f"Target {target_str}: Control not in process_only; selected BR {best_ctrl['br_idx']} ({best_ctrl['n_trials']} trials).")
                selected = req_matched + [best_ctrl]
            else:
                selected = req_matched
    else:
        avail_ctrls = [c for c in parsed_candidates if c["is_control"]]
        if avail_ctrls:
            best_ctrl = max(avail_ctrls, key=lambda c: c["n_trials"])
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
            base_label = f"BR_{br} Control"
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
        )
        conditions.append(cond)

    # Sort conditions: Control first, then stim conditions sorted by pulse count / BR index
    conditions.sort(key=lambda c: (0 if c.is_control else 1, c.pulse_count or 0, c.br_idx))
    _log(f"Loaded {len(conditions)} conditions for Target {target_str}:")
    for c in conditions:
        _log(f"   BR {c.br_idx:03d}: {c.label} (ctrl={c.is_control}, pulses={c.pulse_count}, dur={c.stim_dur_ms:.1f}ms)")

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
    Returns:
        psd: (N_signals, N_rfft_bins)
        freqs: (N_rfft_bins,) in Hz.
    """
    n_signals, n_samples = traces_2d.shape
    Nw = int(win_ms / 1000.0 * fs)
    if Nw % 2 == 0:
        Nw += 1
    Nw = min(Nw, n_samples)
    Nw = max(Nw, 7)

    # Compute DPSS tapers (K, Nw)
    tapers, _ = signal.windows.dpss(Nw, nw, k, return_ratios=True)
    tapers = np.ascontiguousarray(tapers, dtype=np.float32)

    # Center-slice traces to Nw
    if n_samples > Nw:
        start_idx = (n_samples - Nw) // 2
        seg = traces_2d[:, start_idx:start_idx + Nw]
    else:
        seg = traces_2d

    n_fft = max(1024, _next_power_of_two(Nw * 2))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    # GPU acceleration if torch available
    if use_gpu and TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            t_torch = torch.tensor(tapers, device=device, dtype=torch.float32)  # (K, Nw)
            x_torch = torch.tensor(np.ascontiguousarray(seg), device=device, dtype=torch.float32)  # (N, Nw)

            tapered = x_torch.unsqueeze(1) * t_torch.unsqueeze(0)
            fft_res = torch.fft.rfft(tapered, n=n_fft, dim=-1)
            power = torch.abs(fft_res) ** 2
            psd_torch = power.mean(dim=1)  # average across K tapers -> (N, N_rfft_bins)
            psd = psd_torch.detach().cpu().numpy()
            del t_torch, x_torch, tapered, fft_res, power, psd_torch
            torch.cuda.empty_cache()
            return psd, freqs
        except Exception:
            pass  # Fall back to CPU

    # CPU vectorized computation
    seg_clean = np.nan_to_num(seg, nan=0.0)
    tapered = seg_clean[:, None, :] * tapers[None, :, :]
    fft_res = np.fft.rfft(tapered, n=n_fft, axis=-1)
    power = np.abs(fft_res) ** 2
    psd = power.mean(axis=1)  # (N, N_rfft_bins)
    return psd, freqs


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
      1. Extract baseline window segment [-950, -650] ms.
      2. Extract active response window segment (e.g. [-200, 0], [0, 150], [0, 350], [150, 350]).
      3. Compute DPSS multitaper PSD for baseline and response segments.
      4. Integrate PSD across band frequencies [f_lo, f_hi].
      5. Compute dB power change: 10 * log10(P_resp / P_base).

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
        base_mask = (time_ms >= -850.0) & (time_ms <= -650.0)
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
        band_mask = (freqs >= 13.0) & (freqs <= 30.0)

    # Average PSD over band frequencies
    p_base = np.nanmean(psd_base[:, band_mask], axis=-1)  # (N_signals,)
    p_resp = np.nanmean(psd_resp[:, band_mask], axis=-1)  # (N_signals,)

    eps = 1e-12
    p_base = np.maximum(p_base, eps)
    p_resp = np.maximum(p_resp, eps)
    db_power = 10.0 * np.log10(p_resp / p_base)
    db_power = np.nan_to_num(db_power, nan=0.0, posinf=0.0, neginf=0.0)

    return db_power.reshape(n_trials, n_sel_ch)


# =============================================================================
# Representational Similarity Analysis (LFP RDM & Separation)
# =============================================================================

def build_lfp_rdm(
    conditions: List[LfpCondition],
    ch_indices: Sequence[int],
    band: Tuple[float, float],
    metric: str = "correlation",
    baseline_win: Tuple[float, float] = DEFAULT_BASELINE_WINDOW_MS,
    resp_win: Tuple[float, float] = DEFAULT_RESPONSE_WINDOW_MS,
    win_ms: float = DEFAULT_MULTITAPER_WIN_MS,
    nw: float = DEFAULT_MULTITAPER_NW,
    k: int = DEFAULT_MULTITAPER_K,
    subtract_control: bool = True,
    zscore_features: bool = True,
) -> Tuple[np.ndarray, List[int], List[str], List[np.ndarray], Optional[np.ndarray]]:
    """
    Construct trial-by-trial LFP Representational Dissimilarity Matrix (RDM).

    When subtract_control=True, subtracts the matched reach-only control condition mean
    response to isolate stimulation-induced excess perturbation (Delta_P = P_stim - P_ctrl).
    """
    cond_features: List[np.ndarray] = []
    block_sizes = [c.n_trials for c in conditions]
    block_labels = [c.label for c in conditions]

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

    # Standardize channels across trials
    if zscore_features and all_features.shape[0] > 1:
        mu = np.nanmean(all_features, axis=0, keepdims=True)
        sd = np.nanstd(all_features, axis=0, keepdims=True)
        sd[sd < 1e-8] = 1.0
        all_features = (all_features - mu) / sd

    if metric == "correlation":
        feat_c = all_features - np.nanmean(all_features, axis=1, keepdims=True)
        norms = np.linalg.norm(feat_c, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        feat_norm = feat_c / norms
        corr = np.dot(feat_norm, feat_norm.T)
        corr = np.clip(corr, -1.0, 1.0)
        RDM = np.clip(1.0 - corr, 0.0, 2.0)
        np.fill_diagonal(RDM, 0.0)
    else:
        diff = all_features[:, None, :] - all_features[None, :, :]
        RDM = np.sqrt(np.sum(diff ** 2, axis=-1))
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

    # Benjamini-Hochberg FDR correction
    Q = np.ones((K, K), dtype=float)
    if p_vals:
        if _fdr_adjust_pairs is not None:
            try:
                adj = _fdr_adjust_pairs(p_vals)
                for (i, j), q_val in zip(pairs, adj):
                    Q[i, j] = q_val
                    Q[j, i] = q_val
            except Exception:
                pass
        else:
            p_arr = np.asarray(p_vals)
            order = np.argsort(p_arr)
            ranks = np.arange(1, len(p_arr) + 1)
            q_arr = np.minimum.accumulate((p_arr[order] * len(p_arr) / ranks)[::-1])[::-1]
            q_arr = np.clip(q_arr, 0.0, 1.0)
            q_orig = np.empty_like(q_arr)
            q_orig[order] = q_arr
            for (i, j), q_val in zip(pairs, q_orig):
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
    metric_label: str = "Dissimilarity (1 - r)",
    multitaper_info: str = "DPSS (win=100ms, NW=2.0, K=2)",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Generate publication-ready trial-by-trial LFP RDM heatmap with condition blocks."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=FIG_DPI)

    cmap = "viridis" if "1 -" in metric_label or "Distance" in metric_label else "RdBu_r"
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

    ax.set_xticks(centers)
    ax.set_xticklabels(block_labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(centers)
    ax.set_yticklabels(block_labels, fontsize=8.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_label, fontsize=10)

    color_reg = REGION_COLORS.get(array_name, "#2c3e50")
    ax.set_title(
        f"LFP Multitaper RDM Heatmap | Array: {array_name} | Band: {band_name}\n"
        f"Target {target} | Window: [{int(resp_win[0])}, {int(resp_win[1])}] ms ({window_name}) | Method: {multitaper_info} (N={RDM.shape[0]} trials)",
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


def plot_combined_4array_silhouette_figure(
    results_by_array: Dict[str, Dict[str, Any]],
    band_name: str,
    band_range: Tuple[float, float],
    target: str,
    window_name: str,
    resp_win: Tuple[float, float],
    multitaper_info: str = "DPSS (win=100ms, NW=2.0, K=2)",
    subtract_control: bool = True,
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
        clean_labels = [re.sub(r'\s*\(n=\d+\)', '', lab) for lab in block_labels]
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
                elif i < j:
                    s_val = S[i, j]
                    star = p_stars_label(Q[i, j])
                    star_str = f"\n{star}" if star and star != "n.s." else ""
                    txt = f"{s_val:+.2f}{star_str}" if np.isfinite(s_val) else "—"
                    col = "white" if abs(s_val) > v_abs * 0.55 else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, fontweight="bold", color=col)
                else:
                    s_val = S[i, j]
                    txt = f"{s_val:+.2f}" if np.isfinite(s_val) else "—"
                    col = "white" if abs(s_val) > v_abs * 0.55 else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=col)

        ax.set_xticks(np.arange(K))
        ax.set_xticklabels(clean_labels, rotation=45, ha="right", fontsize=8.5)
        ax.set_yticks(np.arange(K))
        ax.set_yticklabels(clean_labels, fontsize=8.5)
        ax.set_title(f"{array_name} ({n_ch} channels)", fontsize=12, fontweight="bold", color=color_reg, pad=10)

    # Shared colorbar at bottom
    if im_ref is not None:
        cbar_ax = fig.add_axes([0.25, 0.04, 0.50, 0.022])
        cb = fig.colorbar(im_ref, cax=cbar_ax, orientation="horizontal")
        cb.set_label("Pairwise Condition Silhouette Score (S)  [Separation Metric]", fontsize=10.5, fontweight="bold")
        cb.ax.tick_params(labelsize=9)

    feat_desc = "Stim-minus-Control Excess Perturbation (ΔdB)" if subtract_control else "Baseline-Normalized Band Power (dB)"
    fig.suptitle(
        f"LFP Condition Separation Matrix across Utah Arrays (Silhouette Score)\n"
        f"Band: {band_name} ({int(band_range[0])}–{int(band_range[1])} Hz) | Window: [{int(resp_win[0])}, {int(resp_win[1])}] ms ({window_name}) | Target {target}\n"
        f"Features: {feat_desc} | Method: {multitaper_info} | * q<0.05, ** q<0.01, *** q<0.001 (FDR BH)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(top=0.91, bottom=0.10, left=0.08, right=0.95, hspace=0.35, wspace=0.25)

    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Combined 4-Array Silhouette figure -> {save_path.name}")
    return fig


def plot_subtraction_analysis_figure(
    conditions: List[LfpCondition],
    region_map: Dict[str, List[int]],
    band: Tuple[float, float],
    band_name: str,
    target: str,
    window_name: str,
    resp_win: Tuple[float, float],
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
        ax.set_title(f"{array_name} (n={len(ch_idx)} ch)", fontsize=11.5, fontweight="bold", color=color_reg)
        if idx % 2 == 0:
            ax.set_ylabel("Stim − Control ΔPower (dB)", fontsize=9.5, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=1)

    fig.suptitle(
        f"LFP Stimulation-minus-Control Excess Perturbation (ΔdB re Matched Reach-Only Control)\n"
        f"Cortical Utah Arrays: M1i, M1s, PMd, SMA | Band: {band_name} ({int(band[0])}–{int(band[1])} Hz)\n"
        f"Window: [{int(resp_win[0])}, {int(resp_win[1])}] ms ({window_name}) | Target {target} (Mean ± 95% CI)",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
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
    win_ms: float = DEFAULT_MULTITAPER_WIN_MS,
    nw: float = DEFAULT_MULTITAPER_NW,
    k: int = DEFAULT_MULTITAPER_K,
    subtract_control: bool = True,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Generate 4-panel dose-response / array response magnitude figure:
    M1i, M1s, PMd, SMA showing baseline-normalized (and control-subtracted) band power change.
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
        ax.set_title(f"{array_name} (n={len(ch_idx)} ch)", fontsize=11, fontweight="bold", color=color_reg)
        y_lbl = "Excess Power relative to Control (dB)" if subtract_control else "Power Change vs Baseline (dB)"
        ax.set_ylabel(y_lbl, fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    sub_title = "Control-Subtracted Stim Perturbation" if subtract_control else "Baseline-Normalized Response"
    fig.suptitle(
        f"LFP Multitaper Band Power Response across Utah Arrays ({sub_title})\n"
        f"Band: {band_name} ({int(band[0])}–{int(band[1])} Hz) | Window: [{int(resp_win[0])}, {int(resp_win[1])}] ms ({window_name}) | Target {target} (Mean ± 95% CI)",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Response Magnitude figure -> {save_path.name}")
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
    metric: str = "correlation",
    baseline_win: Tuple[float, float] = DEFAULT_BASELINE_WINDOW_MS,
    win_ms: float = DEFAULT_MULTITAPER_WIN_MS,
    nw: float = DEFAULT_MULTITAPER_NW,
    k: int = DEFAULT_MULTITAPER_K,
    subtract_control: bool = True,
    metadata_csv: Optional[Path] = None,
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

    # Resolve windows to run (default: all 4 windows)
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

    # 3. Outer Loop: Temporal Analysis Windows
    for win_tag, resp_win in active_windows.items():
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

                S, P, Q = compute_pairwise_silhouette_matrix(RDM, block_sizes, n_perm=N_PERMUTATIONS)
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
                }

                # Save individual trial-by-trial RDM Heatmap
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
                    metric_label=f"Dissimilarity ({'1 - r' if metric == 'correlation' else 'Euclidean'})",
                    multitaper_info=multitaper_info,
                    save_path=out_rdm,
                )

                # Save pairwise silhouette table
                clean_labs = [re.sub(r'\s*\(n=\d+\)', '', lab) for lab in block_labels]
                df_sil = pd.DataFrame(S, index=clean_labs, columns=clean_labs)
                out_csv = win_save_dir / f"lfp_silhouette_Target_{target_str}_{array_name}_{band_name}_{win_tag}.csv"
                df_sil.to_csv(out_csv)

            # Combined 4-Array Silhouette Figure (All 4 arrays on the SAME plot)
            if results_by_array:
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
                    save_path=out_combined_sil,
                )
                # Copy to top-level save_dir for quick reference
                top_sil = save_dir / f"lfp_silhouette_4arrays_Target_{target_str}_{band_name}_{win_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
                try:
                    import shutil
                    shutil.copyfile(out_combined_sil, top_sil)
                except Exception:
                    pass

            # Subtraction Analysis Figure (Stim - Control ΔdB Excess Perturbation across 4 arrays)
            out_sub = win_save_dir / f"lfp_subtraction_profiles_Target_{target_str}_{band_name}_{win_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
            plot_subtraction_analysis_figure(
                conditions=conditions,
                region_map=region_map,
                band=band_range,
                band_name=band_name,
                target=target_str,
                window_name=win_tag,
                resp_win=resp_win,
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

            # 4-Panel Array Response Magnitude Figure
            out_resp = win_save_dir / f"lfp_response_magnitude_Target_{target_str}_{band_name}_{win_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
            plot_array_response_magnitude_figure(
                conditions=conditions,
                region_map=region_map,
                band=band_range,
                band_name=band_name,
                target=target_str,
                window_name=win_tag,
                resp_win=resp_win,
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
        help="Analysis windows: '-200_to_0ms', '0_to_150ms', '0_to_350ms', '150_to_350ms', or 'all'.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="correlation",
        choices=["correlation", "euclidean"],
        help="RSA distance metric (default: correlation = 1 - Pearson r).",
    )
    parser.add_argument("--multitaper-win", type=float, default=DEFAULT_MULTITAPER_WIN_MS, help="Multitaper window (ms).")
    parser.add_argument("--multitaper-nw", type=float, default=DEFAULT_MULTITAPER_NW, help="Multitaper NW product.")
    parser.add_argument("--multitaper-k", type=int, default=DEFAULT_MULTITAPER_K, help="Multitaper number of tapers K.")
    parser.add_argument(
        "--no-subtract-control",
        dest="subtract_control",
        action="store_false",
        default=True,
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
        help="Baseline time window in ms relative to stim onset (default: -950 -650).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
        )

    _log("LFP RSA Pipeline Complete.")


if __name__ == "__main__":
    main()
