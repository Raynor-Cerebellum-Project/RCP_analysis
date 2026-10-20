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
    1. Behavioral RDM / RSM Heatmap: Trial-by-trial kinematic dissimilarity matrix with
       condition block boundaries and labels.
    2. Condition Separation & Consistency Matrix: Pairwise silhouette separation scores
       with permutation testing (p-values & FDR stars), plus within-condition consistency.
    3. Time-Resolved Trajectory Variability Comparison: Within-condition 3D variability,
       excess variability ratio (Stim/Ctrl), and 3D divergence from control across time.
    4. Multi-Condition Directional Perturbation Traces & Vector Diagram: Time-resolved
       Delta X, Delta Y, Delta Z profiles, and 2D/3D perturbation vector hodographs.
    5. Consolidated Publication Summary Figure: 4-panel multi-metric summary.
    6. Multi-Target Joint Comparison: Cross-target RDM (Target A + Target B) testing
       target separability under stimulation.
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
from matplotlib.patches import Patch, Ellipse
from matplotlib.colors import LinearSegmentedColormap
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
        _p_stars,
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
DEFAULT_PLOT_WINDOW_MS: Tuple[float, float] = (0.0, 500.0)

# Dual-window Behavioral RSA analysis windows:
# Region 1 (Early Window): Pre-divergence / expected control-like phase
RSA_EARLY_WINDOW_MS: Tuple[float, float] = (0.0, 75.0)
# Region 2 (Perturbation Window): Active stimulation deflection & divergence phase
RSA_LATE_WINDOW_MS: Tuple[float, float] = (75.0, 350.0)

# Backward-compatibility alias (unused by core pipeline)
EXTENDED_PLOT_WINDOW_MS: Tuple[float, float] = (-200.0, 500.0)

MIN_REACHES_PER_CONDITION: int = 3
N_PERMUTATIONS: int = 1000
FDR_ALPHA: float = 0.05
FIG_FORMAT: str = "png"  # Preselected figure format: "png" or "svg"
FIG_DPI: int = 300

# Color definitions
COLOR_CTRL: str = "#2c3e50"
COLOR_CTRL_LIGHT: str = "#7f8c8d"
COLOR_STIM_DEFAULT: str = "#e74c3c"


# =============================================================================
# Logging & Helper Utilities
# =============================================================================

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
        )


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
            # Use that control and keep strictly requested conditions.
            _log(f"Target {target_str}: Control condition(s) {sorted(set(c['br_idx'] for c in ctrl_in_req))} provided in process_only with reaches for Target {target_str}. Using requested control.")
            selected_candidates = req_matched
        else:
            # Control NOT provided in process_only, or target is not available in the process_only control session.
            # Look for the control session with the largest amount of trials for this target!
            avail_ctrls = [c for c in parsed_candidates if c["is_control"] and c["n_events"] > 0]
            if avail_ctrls:
                best_ctrl = max(avail_ctrls, key=lambda c: c["n_events"])
                _log(f"Target {target_str}: Control not in process_only or target unavailable in requested control; selected BR {best_ctrl['br_idx']} as control ({best_ctrl['n_events']} trials for Target {target_str}).")
                selected_candidates = req_matched + [best_ctrl]
            else:
                _log(f"Target {target_str}: No control conditions available with reaches for Target {target_str}; proceeding with requested conditions.")
                selected_candidates = req_matched
    else:
        # No process_only filtering: take control session with largest amount of trials for this target + all stim conditions
        avail_ctrls = [c for c in parsed_candidates if c["is_control"] and c["n_events"] > 0]
        if avail_ctrls:
            best_ctrl = max(avail_ctrls, key=lambda c: c["n_events"])
            _log(f"Target {target_str}: Selected BR {best_ctrl['br_idx']} as control with largest trials ({best_ctrl['n_events']}) for Target {target_str}.")
            selected_candidates = [best_ctrl] + [c for c in parsed_candidates if not c["is_control"]]
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

        if align_task_frame:
            try:
                R_task, p_start = p3d.compute_task_rotation_matrix(
                    dlc_df=df_dlc,
                    session_landmarks=session_landmarks,
                    square_size_mm=square_size_mm,
                )
                pts_cam = np.stack([x_all, y_all, z_all], axis=0) - p_start.reshape(3, 1, 1)
                pts_task = np.einsum('ji,jkl->ikl', R_task, pts_cam)
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
            win_mask = (rel_t >= 0.0) & (rel_t <= 400.0)

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
        )
        conditions.append(rc)

    # Sort conditions: Control conditions first, followed by stim conditions sorted by BR index
    conditions.sort(key=lambda c: (0 if c.is_control else 1, c.br_idx))
    _log(f"Loaded {len(conditions)} valid conditions for Target {target_str} (Control count: {sum(c.is_control for c in conditions)}):")
    for c in conditions:
        ctrl_mark = " [CONTROL]" if c.is_control else ""
        _log(f"   • {c.label}{ctrl_mark}")
    return conditions


# =============================================================================
# Behavioral Representational Similarity Analysis (Kinematic RDM / RSA)
# =============================================================================

def compute_pairwise_reach_distances(
    x1: np.ndarray, y1: np.ndarray, z1: np.ndarray,
    x2: np.ndarray, y2: np.ndarray, z2: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise 3D Euclidean distances between reaches in group 1 (N1 trials)
    and group 2 (N2 trials), averaged over the reaching time axis.
    
    Returns:
        dist_mat: (N1, N2) array of mean 3D Euclidean distances in mm.
    """
    # x1: (N1, T), x2: (N2, T)
    n1, t1 = x1.shape
    n2, t2 = x2.shape
    assert t1 == t2, "Time axes must match."

    # Pairwise 3D Euclidean distance at each time sample, averaged over time
    dx = x1[:, None, :] - x2[None, :, :]  # (N1, N2, T)
    dy = y1[:, None, :] - y2[None, :, :]
    dz = z1[:, None, :] - z2[None, :, :]
    point_dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)  # (N1, N2, T)
    return np.nanmean(point_dist, axis=2)              # (N1, N2)


def build_behavioral_rdm(
    conditions: List[ReachCondition],
    metric: str = "euclidean",
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Construct full trial-by-trial Kinematic Dissimilarity Matrix (RDM / RSM).

    Args:
        conditions: List of ReachCondition objects.
        metric: 'euclidean' (mean 3D distance in mm) or 'correlation' (1 - Pearson r).

    Returns:
        RDM: (N_total, N_total) dissimilarity matrix.
        block_sizes: Number of trials in each condition block.
        block_labels: Formatted label for each condition block.
    """
    block_sizes = [c.n_trials for c in conditions]
    block_labels = [c.label for c in conditions]
    n_total = sum(block_sizes)

    RDM = np.zeros((n_total, n_total), dtype=float)

    # Flatten feature vectors for correlation metric
    flattened_features: List[np.ndarray] = []
    if metric == "correlation":
        for c in conditions:
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

    # Euclidean distance metric (mm)
    starts = np.r_[0, np.cumsum(block_sizes)[:-1]]
    for i, c1 in enumerate(conditions):
        s1, e1 = starts[i], starts[i] + block_sizes[i]
        for j, c2 in enumerate(conditions):
            s2, e2 = starts[j], starts[j] + block_sizes[j]
            if i <= j:
                d_sub = compute_pairwise_reach_distances(c1.x, c1.y, c1.z, c2.x, c2.y, c2.z)
                if i == j:
                    np.fill_diagonal(d_sub, 0.0)
                RDM[s1:e1, s2:e2] = d_sub
                RDM[s2:e2, s1:e1] = d_sub.T

    return RDM, block_sizes, block_labels


def compute_two_group_silhouette(D: np.ndarray, in_group_b: np.ndarray) -> float:
    """Fast two-group silhouette score for precomputed distance matrix D."""
    if "_two_group_silhouette" in globals() and callable(_two_group_silhouette):
        return _two_group_silhouette(D, in_group_b)
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

            s_obs = compute_two_group_silhouette(D_sub, in_b)
            S[i, j] = s_obs
            S[j, i] = s_obs

            # Permutation test
            if n_perm > 0 and np.isfinite(s_obs):
                null_scores = np.zeros(n_perm)
                for p in range(n_perm):
                    shuffled_b = rng.permutation(in_b)
                    null_scores[p] = compute_two_group_silhouette(D_sub, shuffled_b)
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


def compute_within_condition_stereotypy(
    D: np.ndarray,
    block_sizes: List[int],
) -> np.ndarray:
    """Compute within-condition mean pairwise distance (stereotypy in mm) along diagonal."""
    K = len(block_sizes)
    within_scores = np.zeros(K)
    starts = np.r_[0, np.cumsum(block_sizes)[:-1]]

    for i in range(K):
        ni = block_sizes[i]
        si, ei = starts[i], starts[i] + ni
        D_within = D[si:ei, si:ei]
        if ni > 1:
            iu = np.triu_indices(ni, k=1)
            within_scores[i] = float(np.nanmean(D_within[iu]))
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
# Multi-Condition Directional Perturbation Analysis
# =============================================================================

def compute_directional_perturbations(
    conditions: List[ReachCondition],
) -> Dict[int, Dict[str, Any]]:
    """
    Decompose reach trajectories into directional perturbation components
    relative to matched control reference:
        Delta X(t) = Lateral displacement (mm; left < 0 < right)
        Delta Y(t) = Forward displacement (mm; undershoot < 0 < overshoot)
        Delta Z(t) = Vertical displacement (mm; dip < 0 < lift)
        ||Delta r(t)|| = 3D vector norm (mm)
    """
    ctrl_candidates = [c for c in conditions if c.is_control]
    ctrl_cond = max(ctrl_candidates, key=lambda c: c.n_trials) if ctrl_candidates else None
    if ctrl_cond is None:
        _log("Warning: No control condition found; cannot compute directional perturbations.")
        return {}

    mx_c, my_c, mz_c = ctrl_cond.mean_trajectory()

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

        # Endpoint shifts (averaged over last 50 ms of movement window)
        end_idx = max(1, int(n_t * 0.15))
        end_dx = np.nanmean(dx_trials[:, -end_idx:], axis=1)
        end_dy = np.nanmean(dy_trials[:, -end_idx:], axis=1)
        end_dz = np.nanmean(dz_trials[:, -end_idx:], axis=1)
        end_norm = np.sqrt(end_dx ** 2 + end_dy ** 2 + end_dz ** 2)

        # Absolute endpoint positions in physical task frame
        abs_end_x = np.nanmean(c.x[:, -end_idx:], axis=1)
        abs_end_y = np.nanmean(c.y[:, -end_idx:], axis=1)
        abs_end_z = np.nanmean(c.z[:, -end_idx:], axis=1)
        mean_abs_end_x = float(np.nanmean(abs_end_x))
        mean_abs_end_y = float(np.nanmean(abs_end_y))
        mean_abs_end_z = float(np.nanmean(abs_end_z))

        ctrl_mean_end_x = float(np.nanmean(mx_c[-end_idx:]))
        ctrl_mean_end_y = float(np.nanmean(my_c[-end_idx:]))
        ctrl_mean_end_z = float(np.nanmean(mz_c[-end_idx:]))

        mean_x, mean_y, mean_z = c.mean_trajectory()

        # Statistical test vs zero (or vs control)
        p_dx = float(stats.ttest_1samp(end_dx, 0.0).pvalue) if n_trials > 1 and np.any(end_dx != 0) else 1.0
        p_dy = float(stats.ttest_1samp(end_dy, 0.0).pvalue) if n_trials > 1 and np.any(end_dy != 0) else 1.0
        p_dz = float(stats.ttest_1samp(end_dz, 0.0).pvalue) if n_trials > 1 and np.any(end_dz != 0) else 1.0
        p_norm = float(stats.ttest_1samp(end_norm, 0.0).pvalue) if n_trials > 1 and np.any(end_norm != 0) else 1.0

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
            "ctrl_mean_end": (ctrl_mean_end_x, ctrl_mean_end_y, ctrl_mean_end_z),
            "ctrl_mean_traj": (mx_c, my_c, mz_c),
            "mean_traj": (mean_x, mean_y, mean_z),
            "p_dx": p_dx, "p_dy": p_dy, "p_dz": p_dz, "p_norm": p_norm,
            "stim_dur_ms": c.stim_dur_ms,
            "n_trials": n_trials,
        }

    return results


# =============================================================================
# Visualizations & Plotting Functions
# =============================================================================

def draw_staggered_stim_block(
    ax: plt.Axes,
    stim_conditions: List[Dict[str, Any]],
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    ladder_height_ratio: float = 0.12,
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

    # 2. Condition-specific drop-lines at stimulus termination
    if show_drop_lines:
        for c in sorted_stims:
            dur = float(c["stim_dur_ms"])
            color = c.get("color", "#e74c3c")
            ax.axvline(dur, color=color, linestyle=":", linewidth=1.1, alpha=0.55, zorder=1)

    # 3. Top-margin duration ladder
    if show_ladder:
        cur_ylim = ax.get_ylim()
        ymin = y_min if y_min is not None else cur_ylim[0]
        ymax = y_max if y_max is not None else cur_ylim[1]
        yrange = ymax - ymin
        if yrange <= 0:
            yrange = 1.0

        n_bars = len(sorted_stims)
        bar_height_total = yrange * ladder_height_ratio
        bar_step = bar_height_total / max(1, n_bars)
        y_top = ymax - (yrange * 0.02)

        for i, c in enumerate(reversed(sorted_stims)):
            dur = float(c["stim_dur_ms"])
            color = c.get("color", "#e74c3c")
            y_bar = y_top - (i * bar_step)
            # Horizontal bar
            ax.plot([0.0, dur], [y_bar, y_bar], color=color, linewidth=2.4, solid_capstyle="butt", zorder=7)
            # End tick
            tick_dy = bar_step * 0.35
            ax.plot([dur, dur], [y_bar - tick_dy, y_bar + tick_dy], color=color, linewidth=1.8, zorder=7)
            # Duration callout text
            dur_str = f"{dur:.1f}ms" if dur < 10 and dur % 1 != 0 else f"{int(dur)}ms"
            ax.text(dur + 2.0, y_bar, dur_str, color=color, fontsize=6.5, va="center", ha="left", fontweight="bold", zorder=8)


def draw_confidence_ellipse(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    n_std: float = 2.0,
    edgecolor: str = "black",
    **kwargs,
) -> Optional[Ellipse]:
    """Draw a covariance confidence ellipse representing 2D trajectory dispersion."""
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
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
        width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-6))
        ellipse = Ellipse(
            xy=(np.mean(x_val), np.mean(y_val)),
            width=width,
            height=height,
            angle=theta,
            edgecolor=edgecolor,
            facecolor="none",
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
    Shorten condition label for matrix tick display while showing:
    - Condition prefix: BR X
    - Control vs Stim parameters:
        * Control: BR X (Ctrl, n=N)
        * Stim: BR X (40p, 400Hz, ch 1-8, n=N) or BR X (2Hz-cont, ch 1-32, n=N)
    """
    m = re.match(r"(BR\s*\d+)", lbl, re.IGNORECASE)
    prefix = m.group(1) if m else lbl

    # Extract trial count if not provided
    if n_tri is None:
        m_n = re.search(r"\(n=(\d+)\)", lbl)
        if m_n:
            n_tri = int(m_n.group(1))
    n_str = f"n={n_tri}" if n_tri is not None else ""

    if "control" in lbl.lower():
        parts = ["Ctrl"]
        if n_str:
            parts.append(n_str)
        return f"{prefix} ({', '.join(parts)})"

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

    if parts:
        return f"{prefix} ({', '.join(parts)})"
    return prefix


def plot_behavioral_rsa_matrix_figure(
    RDM: np.ndarray,
    S: np.ndarray,
    Q: np.ndarray,
    within_stereotypy: np.ndarray,
    block_sizes: List[int],
    block_labels: List[str],
    target: str = "A",
    metric_label: str = "Euclidean Distance (mm)",
    window_ms: Optional[Tuple[float, float]] = None,
    window_description: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Consolidated 2-panel Behavioral RSA Figure:
        Panel A: Trial-by-Trial Kinematic Dissimilarity Matrix (RDM) with condition boundaries.
        Panel B: Condition-Level Pairwise Silhouette Matrix with FDR significance stars.
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14.0, 6.2), dpi=FIG_DPI)

    # -------------------------------------------------------------
    # Panel A: Trial-by-Trial Kinematic RDM
    # -------------------------------------------------------------
    vmax_a = np.nanpercentile(RDM[RDM > 0], 98) if np.any(RDM > 0) else 50.0
    cmap_rdm = get_rdm_colormap()

    im_a = ax_a.imshow(
        RDM, aspect="equal", origin="upper", interpolation="nearest",
        cmap=cmap_rdm, vmin=0.0, vmax=vmax_a,
    )
    win_tag = f" ({int(window_ms[0])}–{int(window_ms[1])} ms)" if window_ms else ""
    ax_a.set_title(f"A. Trial-by-Trial Kinematic RDM (Target {target}{win_tag})", fontsize=11, fontweight="bold", loc="left")

    cum = np.cumsum(block_sizes)
    starts = np.r_[0, cum[:-1]]
    boundaries = cum[:-1] - 0.5
    centers = starts + (np.asarray(block_sizes) / 2.0) - 0.5

    ax_a.set_xticks(boundaries, minor=True)
    ax_a.set_yticks(boundaries, minor=True)
    ax_a.grid(which="minor", color="#2c3e50", linestyle="-", linewidth=1.0, alpha=0.6)
    ax_a.set_xticks([])
    ax_a.set_yticks([])

    short_labels = [to_short_condition_label(l, sz) for l, sz in zip(block_labels, block_sizes)]

    for c, lab in zip(centers, short_labels):
        ax_a.text(c, -0.02, lab, transform=ax_a.get_xaxis_transform(),
                  ha="right", va="top", rotation=35, fontsize=8)
        ax_a.text(-0.02, c, lab, transform=ax_a.get_yaxis_transform(),
                  ha="right", va="center", fontsize=8)

    cb_a = fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)
    cb_a.set_label(f"Dissimilarity: {metric_label}", fontsize=9)

    # -------------------------------------------------------------
    # Panel B: Condition Pairwise Silhouette Matrix
    # -------------------------------------------------------------
    K = len(block_labels)
    finite_s = S[np.isfinite(S)]
    vmax_b = float(np.max(np.abs(finite_s))) if len(finite_s) > 0 else 0.5
    vmax_b = max(vmax_b, 0.2)

    cmap_b = plt.colormaps["RdBu_r"].copy()
    cmap_b.set_bad("#eaecee")

    im_b = ax_b.imshow(
        np.ma.masked_invalid(S), cmap=cmap_b, vmin=-vmax_b, vmax=vmax_b,
        origin="upper", interpolation="nearest", aspect="equal",
    )
    ax_b.set_title(f"B. Condition Separation (Silhouette) & Consistency (Target {target}{win_tag})", fontsize=11, fontweight="bold", loc="left")

    ax_b.set_xticks(range(K))
    ax_b.set_yticks(range(K))
    ax_b.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=8)
    ax_b.set_yticklabels(short_labels, fontsize=8)

    ax_b.set_xticks(np.arange(K + 1) - 0.5, minor=True)
    ax_b.set_yticks(np.arange(K + 1) - 0.5, minor=True)
    ax_b.grid(which="minor", color="white", linewidth=1.0)
    ax_b.tick_params(which="minor", length=0)

    # Annotate cells with silhouette scores and FDR stars
    for i in range(K):
        for j in range(K):
            if i == j:
                # Diagonal: Within-condition dispersion / stereotypy in mm
                score_str = f"[{within_stereotypy[i]:.1f}]"
                ax_b.text(j, i, score_str, ha="center", va="center", fontsize=7.5, color="#1b2631", fontweight="bold")
            elif np.isfinite(S[i, j]):
                val = S[i, j]
                q_val = Q[i, j]
                star = p_stars_label(q_val)
                txt = f"{val:+.2f}"
                if star and star != "n.s.":
                    txt += f"\n{star}"
                color = "white" if abs(val) > 0.55 * vmax_b else "black"
                ax_b.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=color)

    cb_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cb_b.set_label("Pairwise Silhouette Score (Separation)", fontsize=9)

    # Full condition details legend line placed outside the plots at bottom of figure
    if len(block_labels) > 7:
        mid = int(np.ceil(len(block_labels) / 2))
        line1 = "   |   ".join(block_labels[:mid])
        line2 = "   |   ".join(block_labels[mid:])
        full_legend_text = f"{line1}\n{line2}"
        fig.text(0.5, 0.045, full_legend_text, ha="center", va="top", fontsize=7.2, color="#2c3e50", fontweight="medium")
        fig.tight_layout(rect=[0, 0.08, 1, 0.94 if window_description else 1.0])
    else:
        full_legend_text = "   |   ".join(block_labels)
        fig.text(0.5, 0.04, full_legend_text, ha="center", va="top", fontsize=7.5, color="#2c3e50", fontweight="medium")
        fig.tight_layout(rect=[0, 0.06, 1, 0.94 if window_description else 1.0])

    if window_description and window_ms:
        fig.suptitle(
            f"Behavioral RSA & Condition Separation — Target {target} ({int(window_ms[0])}–{int(window_ms[1])} ms : {window_description})",
            fontsize=12.0, fontweight="bold", y=0.985,
        )

    fig.text(
        0.98, 0.01,
        "Off-diagonal: Pairwise silhouette S (FDR * q<0.05, ** q<0.01, *** q<0.001). Diagonal: [Within-condition mean pairwise distance mm].",
        ha="right", va="bottom", fontsize=7.5, color="#5d6d7e", style="italic",
    )
    if save_path:
        out_file = save_path.with_suffix(f".{FIG_FORMAT.lower()}")
        fig.savefig(out_file, format=FIG_FORMAT.lower(), dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Behavioral RSA matrix figure: {out_file.name}")
        plt.close(fig)
    return fig


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
    metric_label: str = "Mean 3D Distance (mm)",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Consolidated 4-panel Dual-Window Behavioral RSA Figure:
        Top Row (Early Window: 0–100 ms, Pre-Divergence / Control-Like):
            Panel A: Trial-by-Trial Kinematic RDM
            Panel B: Condition Separation (Pairwise Silhouette)
        Bottom Row (Perturbation Window: 100–350 ms, Active Deflection / Divergent):
            Panel C: Trial-by-Trial Kinematic RDM
            Panel D: Condition Separation (Pairwise Silhouette)
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
    short_labels = [to_short_condition_label(l, sz) for l, sz in zip(block_labels, block_sizes)]

    # -------------------------------------------------------------
    # Panel A: Early RDM (0–100 ms)
    # -------------------------------------------------------------
    im_a = ax_a.imshow(
        RDM_early, aspect="equal", origin="upper", interpolation="nearest",
        cmap=cmap_rdm, vmin=0.0, vmax=vmax_rdm,
    )
    ax_a.set_title(
        f"A. Early Kinematic RDM ({int(window_early[0])}–{int(window_early[1])} ms)",
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
    # Panel B: Early Silhouette Matrix (0–100 ms)
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
        f"B. Early Condition Separation ({int(window_early[0])}–{int(window_early[1])} ms) — Target {target}",
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
                txt = f"{val:+.2f}"
                if star and star != "n.s.":
                    txt += f"\n{star}"
                color = "white" if abs(val) > 0.55 * vmax_s else "black"
                ax_b.text(j, i, txt, ha="center", va="center", fontsize=7.0, color=color)

    cb_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
    cb_b.set_label("Pairwise Silhouette S", fontsize=8.5)

    # -------------------------------------------------------------
    # Panel C: Late / Perturbation RDM (100–350 ms)
    # -------------------------------------------------------------
    im_c = ax_c.imshow(
        RDM_late, aspect="equal", origin="upper", interpolation="nearest",
        cmap=cmap_rdm, vmin=0.0, vmax=vmax_rdm,
    )
    ax_c.set_title(
        f"C. Perturbation Kinematic RDM ({int(window_late[0])}–{int(window_late[1])} ms)",
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
    # Panel D: Late / Perturbation Silhouette Matrix (100–350 ms)
    # -------------------------------------------------------------
    im_d = ax_d.imshow(
        np.ma.masked_invalid(S_late), cmap=cmap_b, vmin=-vmax_s, vmax=vmax_s,
        origin="upper", interpolation="nearest", aspect="equal",
    )
    ax_d.set_title(
        f"D. Perturbation Condition Separation ({int(window_late[0])}–{int(window_late[1])} ms) — Target {target}",
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
                txt = f"{val:+.2f}"
                if star and star != "n.s.":
                    txt += f"\n{star}"
                color = "white" if abs(val) > 0.55 * vmax_s else "black"
                ax_d.text(j, i, txt, ha="center", va="center", fontsize=7.0, color=color)

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
        fig.tight_layout(rect=[0, 0.09, 1, 0.95])
    elif len(block_labels) > 4:
        mid = int(np.ceil(len(block_labels) / 2))
        line1 = "   |   ".join(block_labels[:mid])
        line2 = "   |   ".join(block_labels[mid:])
        full_legend_text = f"{line1}\n{line2}"
        fig.text(0.5, 0.045, full_legend_text, ha="center", va="top", fontsize=7.2, color="#2c3e50", fontweight="medium")
        fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    else:
        full_legend_text = "   |   ".join(block_labels)
        fig.text(0.5, 0.04, full_legend_text, ha="center", va="top", fontsize=7.5, color="#2c3e50", fontweight="medium")
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    fig.suptitle(
        f"Behavioral RSA Temporal Epoch Comparison: Early ({int(window_early[0])}–{int(window_early[1])} ms) vs. Perturbation ({int(window_late[0])}–{int(window_late[1])} ms) — Target {target}",
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
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Consolidated 2-Panel Trajectory Variability & Net Deflection Figure:
        Panel A: Within-Condition 3D Trajectory Variability V(t) in mm.
        Panel B: Net 3D Trajectory Deflection from Control Reference D_to_ctrl(t) in mm.
    """
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True, dpi=FIG_DPI)

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

        # Panel A: Within-Condition Variability V(t)
        ax_a.plot(t, v_med, color=color, linestyle=ls, linewidth=lw, label=label, zorder=zorder)
        ax_a.fill_between(t, v_med - v_sem, v_med + v_sem, color=color, alpha=0.15, zorder=zorder - 1)

        # Panel B: Divergence from Control Trajectory
        if not is_ctrl:
            ax_b.plot(t, div_m, color=color, linestyle="-", linewidth=lw, label=label, zorder=zorder)
            ax_b.fill_between(t, div_m - div_sem, div_m + div_sem, color=color, alpha=0.15, zorder=zorder - 1)
        else:
            ax_b.plot(t, div_m, color=color, linestyle="--", linewidth=1.5, alpha=0.7, label=f"{label} (Baseline)", zorder=2)
            ax_b.fill_between(t, np.maximum(0, div_m - div_sem), div_m + div_sem, color=color, alpha=0.10, zorder=1)

    # 20 mm Significance / Heuristic Threshold Line on Panel B
    ax_b.axhline(
        20.0,
        color="#c0392b",
        linestyle="--",
        linewidth=1.3,
        alpha=0.85,
        zorder=2,
        label="Significance Threshold (20 mm)",
    )

    # Ensure Y-max accommodates 20 mm line
    c_ymin_b, c_ymax_b = ax_b.get_ylim()
    if c_ymax_b < 25.0:
        ax_b.set_ylim(c_ymin_b, 25.0)

    # Apply Staggered Stim Windows, despine, and clean styling to both panels
    for ax in [ax_a, ax_b]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.2, zorder=1)
        ax.grid(True, linestyle="--", alpha=0.35)
        # Expand Y-max slightly to accommodate top-margin duration ladder without clipping curves
        c_ymin, c_ymax = ax.get_ylim()
        n_ymax = c_ymax + (c_ymax - c_ymin) * 0.14
        ax.set_ylim(c_ymin, n_ymax)
        draw_staggered_stim_block(ax, stim_info, y_min=c_ymin, y_max=n_ymax)

    ax_a.set_ylabel("3D Variability (mm)", fontsize=9.5, fontweight="medium")
    ax_a.set_title(f"A. Within-Condition Trajectory Variability Across Time (Target {target})", fontsize=10.5, fontweight="bold", loc="left")

    ax_b.set_ylabel("3D Deflection from Ctrl (mm)", fontsize=9.5, fontweight="medium")
    ax_b.set_xlabel("Time Relative to Stim Onset (ms)", fontsize=10, fontweight="medium")
    ax_b.set_title("B. Net 3D Trajectory Deflection from Control Reference", fontsize=10.5, fontweight="bold", loc="left")

    # Single unified figure legend placed cleanly OUTSIDE the plots at top
    h_a, l_a = ax_a.get_legend_handles_labels()
    h_b, l_b = ax_b.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(h_a + h_b, l_a + l_b):
        if l not in by_label:
            by_label[l] = h
    handles = list(by_label.values())
    labels = list(by_label.keys())
    ncol = min(7, int(np.ceil(len(labels) / 2))) if len(labels) > 7 else len(labels)
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=ncol,
        frameon=True, framealpha=0.95, fontsize=8,
    )
    fig.subplots_adjust(top=0.85 if len(labels) > 7 else 0.89, hspace=0.22, left=0.08, right=0.96, bottom=0.09)

    if save_path:
        out_file = save_path.with_suffix(f".{FIG_FORMAT.lower()}")
        fig.savefig(out_file, format=FIG_FORMAT.lower(), dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Time-Resolved Variability figure: {out_file.name}")
        plt.close(fig)
    return fig


def plot_target_directional_deviation_traces(
    dir_results: Dict[int, Dict[str, Any]],
    target: str = "A",
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
        ax.set_xlabel("Time Relative to Stim Onset (ms)", fontsize=9.0)
        ax.grid(True, linestyle="--", alpha=0.35)

        # Expand Y-max slightly and draw staggered stim window block
        c_ymin, c_ymax = ax.get_ylim()
        n_ymax = c_ymax + (c_ymax - c_ymin) * 0.14
        ax.set_ylim(c_ymin, n_ymax)
        draw_staggered_stim_block(ax, stim_info, y_min=c_ymin, y_max=n_ymax)

    # Single unified figure legend placed cleanly OUTSIDE the subplots at top
    handles, labels = ax_x.get_legend_handles_labels()
    ncol = min(7, int(np.ceil(len(labels) / 2))) if len(labels) > 7 else len(labels)
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.985),
        ncol=ncol,
        frameon=True, framealpha=0.95, fontsize=8,
    )
    fig.suptitle(f"3D Directional Perturbation Time Profiles — Target {target}", fontsize=11.5, fontweight="bold", y=1.03 if len(labels) > 7 else 1.02)
    fig.subplots_adjust(top=0.82 if len(labels) > 7 else 0.86, left=0.06, right=0.96, bottom=0.12, wspace=0.24)

    if save_path:
        out_file = save_path.with_suffix(f".{FIG_FORMAT.lower()}")
        fig.savefig(out_file, format=FIG_FORMAT.lower(), dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Directional Deviation figure: {out_file.name}")
        plt.close(fig)
    return fig


def plot_combined_spatial_perturbations_figure(
    dir_results_by_target: Dict[str, Dict[int, Dict[str, Any]]],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Consolidated 4-panel Spatial Perturbation Figure (Combined Targets A & B):
        Top Row (Spatial Workspace):
            Panel A: Touchscreen-Face Endpoint Perturbations for Targets A & B on the same plot
                     in the shared physical task coordinate frame (Lateral X vs Vertical Z).
            Panel B: True 3D Spatial Trajectories & Perturbation Vectors across the reaching volume.
        Bottom Row (Perturbation Dynamics):
            Panel C: Net 3D Perturbation Magnitude ||Delta r(t)|| across time for Target A.
            Panel D: Net 3D Perturbation Magnitude ||Delta r(t)|| across time for Target B.
                     (Panels C & D share the vertical Y-axis for direct quantitative comparison).
    """
    fig = plt.figure(figsize=(16.5, 11.5), dpi=FIG_DPI)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.25)

    ax_face = fig.add_subplot(gs[0, 0])
    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_norm_a = fig.add_subplot(gs[1, 0])
    ax_norm_b = fig.add_subplot(gs[1, 1])

    # Extract all unique stimulation conditions across targets for consistent color mapping
    all_stim_conds = sorted(list(set(
        k for res in dir_results_by_target.values() for k, v in res.items() if not v["is_control"]
    )))
    palette = plt.cm.tab10(np.linspace(0.0, 1.0, max(10, len(all_stim_conds) + 1)))
    cond_colors = {cond: palette[i % len(palette)] for i, cond in enumerate(all_stim_conds)}

    # Track global stim duration across targets
    stim_dur_max = 0.0
    for d_tgt in dir_results_by_target.values():
        for v in d_tgt.values():
            stim_dur_max = max(stim_dur_max, v.get("stim_dur_ms", 0.0) or 0.0)

    # -------------------------------------------------------------------------
    # Panel A: Touchscreen Face (Targets A & B on the same plot & coordinates)
    # -------------------------------------------------------------------------
    targets_present = sorted(dir_results_by_target.keys())
    for tgt in targets_present:
        d_tgt = dir_results_by_target[tgt]
        ctrl_d = next((v for v in d_tgt.values() if v["is_control"]), None)

        if ctrl_d:
            ctrl_x, ctrl_y, ctrl_z = ctrl_d["ctrl_mean_end"]
            # Control endpoint cluster and confidence ellipse
            ax_face.scatter(
                ctrl_d["abs_end_x"], ctrl_d["abs_end_z"],
                color=COLOR_CTRL, alpha=0.22, s=22, edgecolors="none",
            )
            draw_confidence_ellipse(
                ax_face, ctrl_d["abs_end_x"], ctrl_d["abs_end_z"],
                n_std=1.5, edgecolor=COLOR_CTRL, linestyle=":", linewidth=1.3, alpha=0.7,
            )
            # Control mean marker
            ax_face.scatter(
                ctrl_x, ctrl_z, marker="D", s=110, color=COLOR_CTRL,
                edgecolors="#ffffff", linewidths=1.5, zorder=8,
                label=f"Control (Tgt {tgt})",
            )
            # Target banner label near button location
            ax_face.text(
                ctrl_x, ctrl_z + 14.0, f"Target {tgt} (Ctrl n={ctrl_d['n_trials']})",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                color="#1a252f",
                bbox=dict(boxstyle="round,pad=0.28", fc="#f2f4f4", ec="#bdc3c7", alpha=0.9),
                zorder=10,
            )

        # Plot each stimulation condition
        for cond_num, v in d_tgt.items():
            if v["is_control"]:
                continue
            color = cond_colors.get(cond_num, COLOR_STIM_DEFAULT)
            mx = v["mean_abs_end_x"]
            mz = v["mean_abs_end_z"]
            pts_x = v["abs_end_x"]
            pts_z = v["abs_end_z"]

            # Individual reach trial endpoints
            ax_face.scatter(pts_x, pts_z, color=color, alpha=0.35, s=24, edgecolors="none")
            # Dispersion ellipse
            draw_confidence_ellipse(
                ax_face, pts_x, pts_z, n_std=1.5, edgecolor=color,
                linestyle="--", linewidth=1.1, alpha=0.7,
            )
            # Perturbation vector arrow from control to stim
            if ctrl_d:
                ax_face.annotate(
                    "", xy=(mx, mz), xytext=(ctrl_x, ctrl_z),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=13),
                    zorder=6,
                )
            # Stim mean endpoint marker
            ax_face.scatter(
                mx, mz, color=color, s=75, edgecolors="#1b2631", linewidths=1.2,
                zorder=7, label=f"BR {cond_num} (Tgt {tgt})" if len(targets_present) > 1 else v["label"],
            )

    ax_face.set_xlabel("Lateral Physical Position X (mm) [Left <-> Right]", fontsize=9.5, fontweight="medium")
    ax_face.set_ylabel("Vertical Physical Position Z (mm) [Height]", fontsize=9.5, fontweight="medium")
    ax_face.set_title(
        "A. Touchscreen-Face Endpoint Perturbations (Shared Task Coordinates)",
        fontsize=10.5, fontweight="bold", loc="left",
    )
    ax_face.grid(True, linestyle="--", alpha=0.35)

    # -------------------------------------------------------------------------
    # Panel B: True 3D Spatial Trajectories & Perturbation Vectors
    # -------------------------------------------------------------------------
    for tgt in targets_present:
        d_tgt = dir_results_by_target[tgt]
        ctrl_d = next((v for v in d_tgt.values() if v["is_control"]), None)

        if ctrl_d:
            cmx, cmy, cmz = ctrl_d["ctrl_mean_traj"]
            ax_3d.plot(
                cmx, cmy, cmz, color=COLOR_CTRL, linestyle="--", linewidth=2.0, alpha=0.85,
                label=f"Ctrl Mean (Tgt {tgt})" if tgt == "A" or "A" not in dir_results_by_target else None,
            )
            ctrl_x, ctrl_y, ctrl_z = ctrl_d["ctrl_mean_end"]
            ax_3d.scatter([ctrl_x], [ctrl_y], [ctrl_z], marker="D", s=85, color=COLOR_CTRL, edgecolors="#ffffff", linewidths=1.2, zorder=8)

        for cond_num, v in d_tgt.items():
            if v["is_control"]:
                continue
            color = cond_colors.get(cond_num, COLOR_STIM_DEFAULT)
            smx, smy, smz = v["mean_traj"]
            ax_3d.plot(smx, smy, smz, color=color, linestyle="-", linewidth=1.6, alpha=0.65)
            mx = v["mean_abs_end_x"]
            my = v["mean_abs_end_y"]
            mz = v["mean_abs_end_z"]
            if ctrl_d:
                ax_3d.plot([ctrl_x, mx], [ctrl_y, my], [ctrl_z, mz], color=color, linewidth=2.4, zorder=8)
            ax_3d.scatter([mx], [my], [mz], color=color, s=65, edgecolors="#1b2631", linewidths=1.2, zorder=9)

    ax_3d.set_xlabel("Lateral X (mm)", fontsize=8.0)
    ax_3d.set_ylabel("Forward Y (mm)", fontsize=8.0)
    ax_3d.set_zlabel("Vertical Z (mm)", fontsize=8.0)
    ax_3d.set_title("B. True 3D Spatial Trajectories & Perturbation Vectors", fontsize=10.5, fontweight="bold", loc="left")
    # Viewpoint oriented such that lateral X is exactly parallel to the bottom of the figure
    ax_3d.view_init(elev=18, azim=-90)

    # -------------------------------------------------------------------------
    # Panels C & D: Net 3D Perturbation Magnitude ||Delta r(t)|| for Targets A & B
    # -------------------------------------------------------------------------
    for ax, tgt, panel_letter in [
        (ax_norm_a, "A", "C"),
        (ax_norm_b, "B", "D"),
    ]:
        if tgt in dir_results_by_target:
            d_tgt = dir_results_by_target[tgt]
            tot_reaches = sum(v["n_trials"] for v in d_tgt.values())
            title = f"{panel_letter}. Net 3D Perturbation Magnitude ||Δr(t)|| (Target {tgt}, {tot_reaches} reaches)"
            for cond_num, v in sorted(d_tgt.items(), key=lambda x: x[0]):
                if v["is_control"]:
                    continue
                color = cond_colors.get(cond_num, COLOR_STIM_DEFAULT)
                t = v["time"]
                m = v["mean_dnorm"]
                s = v["sem_dnorm"]
                ax.plot(t, m, color=color, linestyle="-", linewidth=1.8, label=v["label"], zorder=3)
                ax.fill_between(t, m - s, m + s, color=color, alpha=0.15)

            ax.axvline(0.0, color="#7f8c8d", linestyle=":", linewidth=1.1)
            ax.set_title(title, fontsize=10.0, fontweight="bold", loc="left")
            ax.set_xlabel("Time Relative to Stim Onset (ms)", fontsize=9.0)
            ax.set_ylabel("3D Deflection ||Δr|| (mm)", fontsize=9.0)
            ax.grid(True, linestyle="--", alpha=0.35)
        else:
            ax.text(
                0.5, 0.5, f"Target {tgt} not analyzed or no valid reaches",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#7f8c8d",
            )
            ax.set_title(title, fontsize=10.0, fontweight="bold", loc="left")

    # Match Y-axis limits across Panels C and D for direct comparison and draw staggered stim blocks
    max_ylim = max(ax_norm_a.get_ylim()[1], ax_norm_b.get_ylim()[1])
    n_max_ylim = max_ylim * 1.15
    ax_norm_a.set_ylim(0.0, n_max_ylim)
    ax_norm_b.set_ylim(0.0, n_max_ylim)

    if "A" in dir_results_by_target:
        stims_a = [
            {"cond_num": k, "label": v["label"], "color": cond_colors.get(k, COLOR_STIM_DEFAULT), "stim_dur_ms": v.get("stim_dur_ms", 0.0) or 0.0, "is_control": False}
            for k, v in dir_results_by_target["A"].items() if not v["is_control"]
        ]
        draw_staggered_stim_block(ax_norm_a, stims_a, y_min=0.0, y_max=n_max_ylim)

    if "B" in dir_results_by_target:
        stims_b = [
            {"cond_num": k, "label": v["label"], "color": cond_colors.get(k, COLOR_STIM_DEFAULT), "stim_dur_ms": v.get("stim_dur_ms", 0.0) or 0.0, "is_control": False}
            for k, v in dir_results_by_target["B"].items() if not v["is_control"]
        ]
        draw_staggered_stim_block(ax_norm_b, stims_b, y_min=0.0, y_max=n_max_ylim)

    # Build clean unified figure legend placed outside the plots at top
    legend_handles = []
    legend_labels = []

    ctrl_a = next((v for v in dir_results_by_target.get("A", {}).values() if v["is_control"]), None)
    ctrl_b = next((v for v in dir_results_by_target.get("B", {}).values() if v["is_control"]), None)
    if ctrl_a and ctrl_b:
        ctrl_lbl = f"Control (Tgt A n={ctrl_a['n_trials']}, Tgt B n={ctrl_b['n_trials']})"
    elif ctrl_a:
        ctrl_lbl = f"Control (n={ctrl_a['n_trials']})"
    elif ctrl_b:
        ctrl_lbl = f"Control (n={ctrl_b['n_trials']})"
    else:
        ctrl_lbl = "Control"

    ctrl_handle = plt.Line2D([0], [0], color=COLOR_CTRL, linestyle="--", linewidth=2.0, marker="D", markersize=6)
    legend_handles.append(ctrl_handle)
    legend_labels.append(ctrl_lbl)

    for c_num in all_stim_conds:
        v_a = dir_results_by_target.get("A", {}).get(c_num)
        v_b = dir_results_by_target.get("B", {}).get(c_num)
        v_sample = v_a or v_b
        base_lbl = v_sample.get("base_label") if v_sample else f"BR {c_num}"

        # Clean base_lbl from any leftover trial string if present
        base_lbl = re.sub(r"\s*\(n=\d+\)", "", base_lbl).strip()

        if v_a is not None and v_b is not None:
            n_str = f"n_A={v_a['n_trials']}, n_B={v_b['n_trials']}"
        elif v_a is not None:
            n_str = f"n={v_a['n_trials']}"
        elif v_b is not None:
            n_str = f"n={v_b['n_trials']}"
        else:
            n_str = ""

        lbl = f"{base_lbl} ({n_str})" if n_str else base_lbl
        col = cond_colors.get(c_num, COLOR_STIM_DEFAULT)
        h = plt.Line2D([0], [0], color=col, linestyle="-", linewidth=2.0, marker="o", markersize=5)
        legend_handles.append(h)
        legend_labels.append(lbl)

    ncol = min(7, int(np.ceil(len(legend_labels) / 2))) if len(legend_labels) > 7 else len(legend_labels)
    fig.legend(
        legend_handles, legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, 0.965),
        ncol=ncol,
        frameon=True, framealpha=0.95, fontsize=8,
    )

    fig.suptitle(
        "Consolidated 3D & Touchscreen Spatial Perturbation Profiles (Targets A & B)",
        fontsize=12.0, fontweight="bold", y=0.995,
    )
    fig.subplots_adjust(top=0.87 if len(legend_labels) > 7 else 0.91, hspace=0.28, wspace=0.22, left=0.06, right=0.96, bottom=0.07)

    if save_path:
        out_file = save_path.with_suffix(f".{FIG_FORMAT.lower()}")
        fig.savefig(out_file, format=FIG_FORMAT.lower(), dpi=FIG_DPI, bbox_inches="tight")
        _log(f"Saved Consolidated Spatial Perturbation figure: {out_file.name}")
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

    cond_tag = "_".join(map(str, sorted(c.cond_num for c in conditions)))

    # -------------------------------------------------------------------------
    # 2a. Region 1: Early Window RSA (0 to 100 ms) — Expected Control-Like
    # -------------------------------------------------------------------------
    conds_early = [c.slice_to_window(rsa_early_window) for c in conditions]
    RDM_early, block_sizes_early, block_labels_early = build_behavioral_rdm(conds_early, metric="euclidean")
    S_early, P_early, Q_early = compute_pairwise_silhouette_matrix(RDM_early, block_sizes_early, n_perm=N_PERMUTATIONS)
    stereotypy_early = compute_within_condition_stereotypy(RDM_early, block_sizes_early)

    early_tag = f"{int(rsa_early_window[0])}_to_{int(rsa_early_window[1])}ms"
    out_rsa_early = save_dir / f"behavioral_rsa_matrix_Target_{target_str}_window_{early_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
    plot_behavioral_rsa_matrix_figure(
        RDM=RDM_early,
        S=S_early,
        Q=Q_early,
        within_stereotypy=stereotypy_early,
        block_sizes=block_sizes_early,
        block_labels=block_labels_early,
        target=target_str,
        metric_label="Mean 3D Distance (mm)",
        window_ms=rsa_early_window,
        window_description="Early Window",
        save_path=out_rsa_early,
    )

    # -------------------------------------------------------------------------
    # 2b. Region 2: Perturbation Window RSA (100 to 350 ms)
    # -------------------------------------------------------------------------
    conds_late = [c.slice_to_window(rsa_late_window) for c in conditions]
    RDM_late, block_sizes_late, block_labels_late = build_behavioral_rdm(conds_late, metric="euclidean")
    S_late, P_late, Q_late = compute_pairwise_silhouette_matrix(RDM_late, block_sizes_late, n_perm=N_PERMUTATIONS)
    stereotypy_late = compute_within_condition_stereotypy(RDM_late, block_sizes_late)

    late_tag = f"{int(rsa_late_window[0])}_to_{int(rsa_late_window[1])}ms"
    out_rsa_late = save_dir / f"behavioral_rsa_matrix_Target_{target_str}_window_{late_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
    plot_behavioral_rsa_matrix_figure(
        RDM=RDM_late,
        S=S_late,
        Q=Q_late,
        within_stereotypy=stereotypy_late,
        block_sizes=block_sizes_late,
        block_labels=block_labels_late,
        target=target_str,
        metric_label="Mean 3D Distance (mm)",
        window_ms=rsa_late_window,
        window_description="Perturbation Window",
        save_path=out_rsa_late,
    )

    # Also save as the standard unwindowed filename for pipeline compatibility
    out_rsa_std = save_dir / f"behavioral_rsa_matrix_Target_{target_str}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
    plot_behavioral_rsa_matrix_figure(
        RDM=RDM_late,
        S=S_late,
        Q=Q_late,
        within_stereotypy=stereotypy_late,
        block_sizes=block_sizes_late,
        block_labels=block_labels_late,
        target=target_str,
        metric_label="Mean 3D Distance (mm)",
        window_ms=rsa_late_window,
        window_description="Perturbation Window",
        save_path=out_rsa_std,
    )

    # -------------------------------------------------------------------------
    # 2c. Consolidated Dual-Window Comparison Figure (0-100 ms vs 100-350 ms)
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
        metric_label="Mean 3D Distance (mm)",
        save_path=out_rsa_dual,
    )

    # -------------------------------------------------------------------------
    # 3. Time-Resolved Trajectory Variability (Across full 0-500 ms window)
    # -------------------------------------------------------------------------
    var_results = compute_time_resolved_variability(conditions)
    out_var = save_dir / f"trajectory_variability_time_resolved_Target_{target_str}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
    plot_time_resolved_variability_figure(
        var_results=var_results,
        target=target_str,
        save_path=out_var,
    )

    # -------------------------------------------------------------------------
    # 4. Multi-Condition Directional Perturbations (Across full 0-500 ms window)
    # -------------------------------------------------------------------------
    dir_results = compute_directional_perturbations(conditions)
    out_dir = save_dir / f"directional_deviations_Target_{target_str}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
    plot_target_directional_deviation_traces(
        dir_results=dir_results,
        target=target_str,
        save_path=out_dir,
    )

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
        "RDM": RDM_late,
        "S": S_late,
        "Q": Q_late,
        "within_stereotypy": stereotypy_late,
        "var_results": var_results,
        "dir_results": dir_results,
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
        help="Global time window (ms) for reaching trajectories (default: 0 500).",
    )
    parser.add_argument(
        "--rsa-early-window",
        nargs=2,
        type=float,
        default=None,
        metavar=("START_MS", "END_MS"),
        help="Early time window (ms) for RSA analysis (default: 0 100).",
    )
    parser.add_argument(
        "--rsa-late-window",
        nargs=2,
        type=float,
        default=None,
        metavar=("START_MS", "END_MS"),
        help="Late/Perturbation time window (ms) for RSA analysis (default: 100 350).",
    )
    parser.add_argument(
        "--no-task-align",
        dest="align_task_frame",
        action="store_false",
        default=True,
        help="Disable rotation into calibrated monkey task frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
            keypoint=args.keypoint,
            align_task_frame=args.align_task_frame,
        )
        if res.get("conditions"):
            results_all[tgt] = res

    # Consolidated Multi-Target Spatial Perturbations Figure (Screen-Face, 3D Vectors, 3D Magnitudes)
    dir_results_by_target = {
        tgt: res["dir_results"]
        for tgt, res in results_all.items()
        if "dir_results" in res and res["dir_results"]
    }
    if dir_results_by_target:
        tgt_keys = sorted(dir_results_by_target.keys())
        tgt_tag = "_and_".join(f"Target_{t}" for t in tgt_keys)
        all_conds = sorted(list(set(c for d in dir_results_by_target.values() for c in d.keys())))
        cond_tag = "_".join(map(str, all_conds))
        out_spatial = save_dir / f"combined_spatial_perturbations_{tgt_tag}_conds_{cond_tag}.{FIG_FORMAT.lower()}"
        plot_combined_spatial_perturbations_figure(
            dir_results_by_target=dir_results_by_target,
            save_path=out_spatial,
        )

    _log("Behavioral RSA & Trajectory Variability Pipeline Complete.")


if __name__ == "__main__":
    main()
