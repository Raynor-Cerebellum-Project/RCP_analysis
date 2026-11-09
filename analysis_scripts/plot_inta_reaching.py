"""
plot_inta_reaching.py
=====================

Characterization of Single-Unit Anterior Interposed Nucleus (IntA) Neural Responses
During Normal Reaching (NPRW Intan Read-Write Probe in Monkey Nike)

All analyses and visualizations are centered on IR Beam Crossing (t = 0 ms),
identically to stimulation delivery trials and the rest of the VELES pipeline.

Key Visualizations Generated:
  1. Staggered Single-Unit Characterization Profiles (Target A vs Target B Separated):
     - Separate side-by-side columns for Target A (Left Reach) and Target B (Right Reach)
     - 4 Staggered Subplots per column (NO confusing dual-axis overlays!):
       * Row 1: Middle Finger Hand Forward Position (Normalized 0 to 1)
       * Row 2: Middle Finger Hand Speed (cm/s)
       * Row 3: Spike Raster (Individual Reaching Trials, NO blank rows!)
       * Row 4: Instantaneous Firing Rate PSTH (Mean ± SEM in Hz with peak marked)
     - Individual per-target dedicated figures (target_A/ and target_B/) plus unified comparison.
     - Clear Blackrock Condition Header (e.g. Control BR_001 vs 400Hz 40p Stim BR_018).
  2. All-Units Population Overview Gallery:
     - Active modulated single-units displayed in a clean, legible multi-panel grid.
     - Sorted chronologically by response peak latency relative to IR crossing (t = 0).
     - Both Target A and Target B PSTH curves overlaid on each unit's panel.
  3. Population Peak Firing Time & Epoch Peaking Diagram:
     - Firing rate peak latency distribution relative to Peak Speed and Touch Endpoint.
     - Sorted population heatmap and epoch breakdown.
  4. Kinematic Encoding & Tuning Curves:
     - Cross-correlation analysis and speed/acceleration tuning curves.
  5. Population Patterns & State-Space Dynamics:
     - PCA state-space trajectories tracing reaching execution through neural space.
"""

from __future__ import annotations

import os
import re
import sys
import json
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Ensure matplotlib cache directory is writable in sandbox / headless environments
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import stats
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# -----------------------------------------------------------------------------
# VELES / RCP_analysis CONFIG INTEGRATION
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import RCP_analysis as rcp
except Exception:
    rcp = None

try:
    import spikeinterface as si
except Exception:
    si = None

try:
    import RCP_analysis.python.functions.config_loading as cfg
    from RCP_analysis.python.functions.config_loading import (
        PARAMS,
        OUT_BASE,
        PERI_ROOT,
        SESSION_LOC,
        METADATA_CSV,
    )
    CONFIG_LOADED = True
except Exception:
    CONFIG_LOADED = False
    PARAMS = None
    OUT_BASE = REPO_ROOT / "results"
    PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim"
    SESSION_LOC = None
    METADATA_CSV = None

# Check environment variables passed by VELES.py
loc_env = os.environ.get("RCP_LOCATION")
if loc_env:
    p_loc = Path(loc_env)
    if p_loc.is_absolute() and p_loc.exists():
        SESSION_LOC = p_loc
    elif PARAMS and hasattr(PARAMS, "data_root") and PARAMS.data_root:
        SESSION_LOC = (Path(PARAMS.data_root) / p_loc).resolve()
    elif SESSION_LOC is None:
        SESSION_LOC = (REPO_ROOT / p_loc).resolve()
    OUT_BASE = SESSION_LOC / "results"
    PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim"

# Respect PROCESS_ONLY from environment or params.yaml if available
PROCESS_ONLY_ENV = os.environ.get("RCP_PROCESS_ONLY")
if PROCESS_ONLY_ENV:
    try:
        PROCESS_ONLY = json.loads(PROCESS_ONLY_ENV)
    except Exception:
        PROCESS_ONLY = [int(x.strip()) for x in PROCESS_ONLY_ENV.split(",") if x.strip().isdigit()]
elif PARAMS and hasattr(PARAMS, "preprocessing") and PARAMS.preprocessing.get("process_only"):
    PROCESS_ONLY = PARAMS.preprocessing.get("process_only")
else:
    PROCESS_ONLY = []


# =============================================================================
# CONSTANTS & TIME WINDOWS (ALL CENTERED ON IR BEAM CROSSING t = 0)
# =============================================================================
WIN_PLOT_MS   = (-500.0, 500.0)
BIN_PSTH_MS   = 10.0   # Bin width in ms for PSTHs
SIGMA_PSTH_MS = 25.0   # Gaussian kernel std in ms

# Pre-reach baseline window used for firing rate estimation (matching plot_firing_rates.py)
BASELINE_WIN_MS = (-800.0, -650.0)

# Reach execution window (centered around IR beam crossing and forward reach)
REACH_WIN_MS = (-50.0, 350.0)

# Colors for reach directions
COLORS = {
    "Target A": "#1f77b4",  # Deep Royal Blue
    "Target B": "#d95f02",  # Vibrant Deep Amber / Orange
    "A": "#1f77b4",
    "B": "#d95f02",
}


# =============================================================================
# 1. KINEMATIC & SPIKE UTILITIES
# =============================================================================
def find_keypoint_indices(names: List[str], keypoint_base: str = "middle") -> Tuple[Optional[int], Optional[int]]:
    """
    Find column indices corresponding to keypoint_base_x and keypoint_base_y.
    Defaults to 'middle' (middle finger) matching plot_behavioral_rsa.py and plot_firing_rates.py.
    """
    clean_names = [str(n).lower().strip() for n in names]
    for base in [keypoint_base, "middle", "middle_finger", "wrist", "hand"]:
        x_name = f"{base}_x"
        y_name = f"{base}_y"
        idx_x = None
        idx_y = None
        for i, name in enumerate(clean_names):
            if name == x_name or name.endswith(f"_{x_name}"):
                idx_x = i
            elif name == y_name or name.endswith(f"_{y_name}"):
                idx_y = i
        if idx_x is not None and idx_y is not None:
            return idx_x, idx_y

    return None, None


def smooth_spike_train(
    spike_times_ms: Any,
    time_grid_ms: np.ndarray,
    sigma_ms: float = SIGMA_PSTH_MS,
) -> np.ndarray:
    """
    Gaussian-smoothed instantaneous firing rate (Hz) evaluated on time_grid_ms.
    Centered relative to IR beam crossing (t = 0 ms).
    Safely handles empty arrays or edge cases.
    """
    if len(time_grid_ms) == 0:
        return np.zeros(0)

    try:
        spk = np.atleast_1d(spike_times_ms).astype(float)
    except Exception:
        return np.zeros_like(time_grid_ms)

    if spk.size == 0 or not np.any(np.isfinite(spk)):
        return np.zeros_like(time_grid_ms)

    dt_s = (time_grid_ms[1] - time_grid_ms[0]) / 1000.0
    half_bin = 0.5 * (time_grid_ms[1] - time_grid_ms[0])
    bin_edges = np.append(time_grid_ms - half_bin, time_grid_ms[-1] + half_bin)

    counts, _ = np.histogram(spk, bins=bin_edges)
    rates_raw = counts / dt_s  # in Hz

    # Gaussian kernel smoothing
    k_half = int(np.ceil(3.5 * sigma_ms / (dt_s * 1000.0)))
    k_t = np.arange(-k_half, k_half + 1) * (dt_s * 1000.0)
    kernel = np.exp(-0.5 * (k_t / sigma_ms) ** 2)
    kernel /= np.sum(kernel)

    smoothed = np.convolve(rates_raw, kernel, mode="same")
    return smoothed


# =============================================================================
# 2. DATASET LOADER (REAL PREPROCESSED DATA & BLACKROCK CONDITIONS)
# =============================================================================
class IntAReachingDataset:
    """Container for single-unit IntA spike trains, kinematics, and target labels."""
    def __init__(
        self,
        t_beh_ms: np.ndarray,
        kinematics: Dict[str, np.ndarray],  # 'speed': (n_trials, n_t), 'dist': (n_trials, n_t), etc.
        target_labels: np.ndarray,          # 'A' or 'B'
        unit_spikes: Dict[int, List[np.ndarray]],  # unit_id -> list of spike time arrays per trial (in ms relative to IR crossing)
        unit_ids: List[int],
        condition_label: str = "Control Baseline",
        condition_type: str = "control_reach",
        br_indices: List[int] = None,
        session_name: str = "",
        is_sorted: bool = False,
        monkey: str = "Nike",
        stim_dur_ms: float = 0.0,
        blank_win_ms: Optional[Tuple[float, float]] = None,
    ):
        self.t_beh_ms = t_beh_ms
        self.kinematics = kinematics
        self.target_labels = target_labels
        self.unit_spikes = unit_spikes
        self.unit_ids = unit_ids
        self.n_trials = len(target_labels)
        self.condition_label = condition_label
        self.condition_type = condition_type
        self.br_indices = br_indices or []
        self.session_name = session_name
        self.is_sorted = is_sorted
        self.monkey = monkey
        self.stim_dur_ms = stim_dur_ms
        self.blank_win_ms = blank_win_ms

    def get_br_condition_tag(self) -> str:
        """Return standardized condition tag for filenames and directories (e.g. 'BR_018', 'all_controls')."""
        if self.br_indices:
            if len(self.br_indices) == 1:
                return f"BR_{self.br_indices[0]:03d}"
            elif len(self.br_indices) <= 3:
                return "BR_" + "_".join(f"{b:03d}" for b in sorted(self.br_indices))
            else:
                return f"BR_{self.br_indices[0]:03d}_to_{self.br_indices[-1]:03d}"
        if self.condition_type == "control_reach":
            return "all_controls"
        return "condition"


def load_condition_kinematics(
    out_base: Path,
    sess_name: str,
    br_num: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load normalized forward distance [0 to 1] and instantaneous speed (cm/s)
    for the middle finger using Camera 1, identically matching plot_firing_rates.py.
    """
    peri_dir = out_base / "checkpoints" / "PeriStim"
    if not peri_dir.exists():
        return None, None, None

    cands = list(peri_dir.rglob(f"*__{sess_name}__BR_{br_num:03d}*.npz"))
    if not cands:
        cands = list(peri_dir.rglob(f"*__BR_{br_num:03d}*.npz"))
    if not cands:
        return None, None, None

    cand_file = None
    for c in cands:
        if not c.name.startswith("._"):
            cand_file = c
            break

    if cand_file is None:
        return None, None, None

    try:
        pdata = np.load(cand_file, allow_pickle=True)
        t_beh = np.asarray(pdata["beh_rel_t"], float) if "beh_rel_t" in pdata else None

        # Camera 1 is primary (side/top view tracking), fallback to Cam 0
        cam_pos = pdata.get("beh_cam1_segs")
        cam_vel = pdata.get("beh_cam1_vel_segs")
        raw_names = pdata.get("beh_cam1_names")

        if cam_pos is None or (isinstance(cam_pos, np.ndarray) and cam_pos.size == 0):
            cam_pos = pdata.get("beh_cam0_segs")
            cam_vel = pdata.get("beh_cam0_vel_segs")
            raw_names = pdata.get("beh_cam0_names")

        if cam_pos is None or t_beh is None:
            return None, None, None

        names_list = [str(x) for x in np.asarray(raw_names).ravel()] if raw_names is not None else []
        beh_labels_raw = [rcp.strip_cam_prefix(n) for n in names_list] if rcp is not None else names_list
        keypoints_order = tuple(PARAMS.kinematics.get("keypoints", ("middle", "wrist"))) if (PARAMS and hasattr(PARAMS, "kinematics") and PARAMS.kinematics.get("keypoints")) else ("middle", "wrist")
        beh_labels_display = rcp.simple_beh_labels(beh_labels_raw, keypoints_order) if rcp is not None else beh_labels_raw

        # 1. Compute Normalized Distance matching plot_firing_rates.py
        norm_dist = None
        if rcp is not None:
            try:
                norm_dist = rcp.compute_normalized_distance_from_xy_3d(
                    cam_pos,
                    beh_labels_display,
                    t_beh,
                    keypoint="middle",
                    ref_time_ms=-600.0,
                    mode="max_abs",
                )
            except Exception:
                norm_dist = None

        if norm_dist is None and cam_pos.ndim == 3:
            # Inline fallback: middle finger distance from -600 ms reference normalized by max_abs
            idx_x, idx_y = find_keypoint_indices(names_list, "middle")
            if idx_x is None:
                idx_x, idx_y = 0, 1
            n_tr, _, n_t = cam_pos.shape
            norm_dist = np.full((n_tr, n_t), np.nan, dtype=float)
            for tr in range(n_tr):
                x = cam_pos[tr, idx_x, :]
                y = cam_pos[tr, idx_y, :]
                base_m = (t_beh >= -650.0) & (t_beh <= -550.0)
                x0 = np.nanmedian(x[base_m]) if np.any(base_m) else x[0]
                y0 = np.nanmedian(y[base_m]) if np.any(base_m) else y[0]
                d = np.sqrt((x - x0)**2 + (y - y0)**2)
                denom = np.nanmax(np.abs(d))
                norm_dist[tr] = d / denom if denom > 0 and np.isfinite(denom) else np.zeros_like(d)

        # 2. Instantaneous Speed
        idx_x, idx_y = find_keypoint_indices(names_list, "middle")
        if idx_x is None and len(names_list) >= 2:
            idx_x, idx_y = 0, 1

        spd = None
        if cam_vel is not None and cam_vel.ndim == 3 and idx_x is not None and idx_y is not None:
            vx = cam_vel[:, idx_x, :]
            vy = cam_vel[:, idx_y, :]
            spd = np.sqrt(vx**2 + vy**2)
        elif norm_dist is not None:
            dt_s = np.nanmedian(np.diff(t_beh)) / 1000.0
            spd = np.abs(np.gradient(norm_dist, dt_s, axis=-1))

        return norm_dist, spd, t_beh
    except Exception as e:
        print(f"[IntA Analysis] Note: Could not load PeriStim kinematics for BR_{br_num:03d}: {e}")
        return None, None, None


def compute_trial_kinematic_landmarks(
    s_tr: np.ndarray,
    d_tr: np.ndarray,
    t_beh_ref: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Compute key kinematic milestone latencies (ms) relative to IR crossing (t = 0 ms):
      1. Peak Acceleration: maximum acceleration (dv/dt) in [-100, t_peak_speed] ms.
      2. Peak Speed: maximum speed in [0, 450] ms.
      3. Peak Deceleration: maximum deceleration (-dv/dt) in [t_peak_speed, t_endpoint + 60] ms.
      4. Endpoint / Touch: peak normalized distance in [100, 500] ms.
    """
    dt_s = float(np.nanmedian(np.diff(t_beh_ref))) / 1000.0 if len(t_beh_ref) > 1 else 0.010
    s_clean = np.nan_to_num(s_tr, nan=0.0)
    acc_tr = np.gradient(s_clean, dt_s)

    # 1. Peak Speed
    spd_m = (t_beh_ref >= 0.0) & (t_beh_ref <= 450.0)
    if np.any(spd_m) and np.any(np.isfinite(s_tr[spd_m])):
        t_pk_spd = float(t_beh_ref[spd_m][np.nanargmax(s_tr[spd_m])])
    else:
        t_pk_spd = 120.0

    # 2. Peak Acceleration (forward reach onset before peak speed)
    acc_m = (t_beh_ref >= -100.0) & (t_beh_ref <= t_pk_spd)
    if np.any(acc_m) and np.any(np.isfinite(acc_tr[acc_m])):
        t_pk_acc = float(t_beh_ref[acc_m][np.nanargmax(acc_tr[acc_m])])
    else:
        t_pk_acc = max(0.0, t_pk_spd * 0.45)

    # 3. Touch Endpoint (reach apex / touch hold in [100, 500] ms)
    arr_m = (t_beh_ref >= 100.0) & (t_beh_ref <= 500.0)
    if np.any(arr_m) and np.any(np.isfinite(d_tr[arr_m])):
        t_end = float(t_beh_ref[arr_m][np.nanargmax(d_tr[arr_m])])
    else:
        t_end = max(350.0, t_pk_spd + 180.0)

    # 4. Peak Deceleration (slowing down between peak speed and touch endpoint)
    dec_m = (t_beh_ref >= t_pk_spd) & (t_beh_ref <= min(500.0, t_end + 60.0))
    if np.any(dec_m) and np.any(np.isfinite(acc_tr[dec_m])):
        t_pk_dec = float(t_beh_ref[dec_m][np.nanargmin(acc_tr[dec_m])])
    else:
        t_pk_dec = t_pk_spd + (t_end - t_pk_spd) * 0.55

    return t_pk_acc, t_pk_spd, t_pk_dec, t_end


def load_real_reaching_dataset(
    peri_dir: Optional[Path] = None,
    process_only: Optional[List[int]] = None,
    condition_filter: Optional[str] = None,
    all_controls: bool = False,
) -> Optional[IntAReachingDataset]:
    """
    Load reaching data aligned to IR beam crossing (t = 0 ms).
    Searches both PeriStim_sorted and PeriStim, grouping by Blackrock Condition.
    """
    import pickle

    out_base = OUT_BASE if OUT_BASE is not None else (REPO_ROOT / "results")
    if peri_dir is None:
        peri_dir = PERI_ROOT if PERI_ROOT is not None and PERI_ROOT.exists() else (out_base / "checkpoints" / "PeriStim")

    peri_sorted_root = out_base / "checkpoints" / "spikesorting" / "PeriStim_sorted"

    # 1. Parse Metadata CSV to map Blackrock files (BR_File) to condition parameters
    meta_by_br: Dict[int, Dict[str, Any]] = {}
    sess_dir = SESSION_LOC if SESSION_LOC is not None else (peri_dir.parents[2] if len(peri_dir.parents) >= 3 else None)

    meta_csv = None
    if METADATA_CSV and Path(METADATA_CSV).exists():
        meta_csv = Path(METADATA_CSV)
    elif sess_dir and (sess_dir / "Metadata").exists():
        cands = [f for f in (sess_dir / "Metadata").glob("*metadata*.csv") if not f.name.startswith("._")]
        if cands:
            meta_csv = cands[0]

    if meta_csv and meta_csv.exists():
        try:
            mdf = pd.read_csv(meta_csv, encoding="latin1")
            for _, r in mdf.iterrows():
                b_val = r.get("BR_File", r.get("br_idx", r.get("Condition")))
                if pd.notna(b_val):
                    try:
                        meta_by_br[int(b_val)] = r.to_dict()
                    except Exception:
                        pass
        except Exception as e:
            print(f"[IntA Analysis] Note: Could not parse metadata CSV: {e}")

    # 2. Determine target Blackrock condition(s)
    target_brs: Optional[List[int]] = None
    if process_only and len(process_only) > 0 and not all_controls:
        target_brs = [int(x) for x in process_only]
    elif condition_filter and condition_filter.isdigit():
        target_brs = [int(condition_filter)]

    # 3. Discover Candidate Checkpoint Files
    # Candidate search roots
    scan_roots = []
    if target_brs:
        # User specified particular BR conditions: scan both stim and control
        if peri_sorted_root.exists():
            scan_roots.extend([peri_sorted_root / "stim_reaches", peri_sorted_root / "control_reaches"])
        if peri_dir.exists():
            scan_roots.extend([peri_dir / "stim_reaches", peri_dir / "control_reaches"])
    else:
        # Default: load control reaches (baseline reach-only)
        if (peri_sorted_root / "control_reaches").exists():
            scan_roots.append(peri_sorted_root / "control_reaches")
        if (peri_dir / "control_reaches").exists():
            scan_roots.append(peri_dir / "control_reaches")

    discovered_files: List[Path] = []
    for s_root in scan_roots:
        if not s_root.exists():
            continue
        for f in sorted(list(s_root.rglob("*.npz"))):
            if f.name.startswith("._") or f in discovered_files:
                continue

            # Extract BR index from filename
            m = re.search(r"BR_(\d+)", f.name, re.IGNORECASE)
            b_num = int(m.group(1)) if m else -1

            if target_brs:
                if b_num in target_brs:
                    discovered_files.append(f)
            else:
                # Default control search: verify it's baseline
                is_ctrl = ("control" in str(f).lower() or
                           meta_by_br.get(b_num, {}).get("Notes", "").lower() in ("baseline", "control"))
                if is_ctrl:
                    discovered_files.append(f)

    if not discovered_files:
        print(f"[IntA Analysis] No candidate reach files found in {scan_roots} for target conditions: {target_brs}")
        return None

    # Prefer PeriStim_sorted files if available
    sorted_files = [f for f in discovered_files if "PeriStim_sorted" in str(f)]
    use_presegmented = (len(sorted_files) > 0)
    files_to_load = sorted_files if use_presegmented else discovered_files

    print(f"[IntA Analysis] Selected {len(files_to_load)} files for condition evaluation "
          f"({'PeriStim_sorted true units' if use_presegmented else 'PeriStim checkpoints'}).")

    # 4. Construct Descriptive Condition Label & Stimulation Parameters
    def _get_meta(m_dict: Dict[str, Any], *keys: str) -> Any:
        if not m_dict:
            return None
        for k in keys:
            if k in m_dict and pd.notna(m_dict[k]) and str(m_dict[k]).strip() != "":
                return m_dict[k]
        low_map = {str(k).lower().strip(): v for k, v in m_dict.items()}
        for k in keys:
            lk = k.lower().strip()
            if lk in low_map and pd.notna(low_map[lk]) and str(low_map[lk]).strip() != "":
                return low_map[lk]
        return None

    found_brs = sorted(list(set([
        int(re.search(r"BR_(\d+)", f.name, re.IGNORECASE).group(1))
        for f in files_to_load if re.search(r"BR_(\d+)", f.name, re.IGNORECASE)
    ])))

    is_stim = any("stim_reaches" in str(f) for f in files_to_load)
    stim_dur_ms = 0.0

    if len(found_brs) == 1:
        b_idx = found_brs[0]
        meta = meta_by_br.get(b_idx, {})
        notes = str(_get_meta(meta, "Notes", "notes") or "").strip()
        freq = _get_meta(meta, "Stim_Frequency_Hz", "Frequency", "stim_freq_hz", "stim_freq")
        pulses = _get_meta(meta, "Pulses", "stim_pulse_count", "pulse_count", "num_pulses")
        dur = _get_meta(meta, "Stim_Duration_ms", "Duration", "stim_dur_nominal_ms", "stim_dur_ms", "duration_ms")

        # Parse stim duration
        if dur is not None and str(dur).lower().strip() not in ("cont", "continuous"):
            try:
                if float(dur) > 0:
                    stim_dur_ms = float(dur)
            except Exception:
                pass
        if stim_dur_ms <= 0.0 and pulses is not None and freq is not None:
            try:
                if float(freq) > 0 and float(pulses) > 0:
                    stim_dur_ms = float(pulses) * 1000.0 / float(freq)
            except Exception:
                pass

        if notes.lower() in ("baseline", "control") or (not freq and not pulses and not is_stim and stim_dur_ms <= 0):
            cond_label = f"Control Baseline Reaches [BR {b_idx:03d}] (Reach-Only, No Stimulation)"
            cond_type = "control_reach"
            stim_dur_ms = 0.0
        else:
            p_val = pulses
            if p_val is None and freq is not None and stim_dur_ms > 0:
                try:
                    p_val = int(round(float(freq) * stim_dur_ms / 1000.0))
                except Exception:
                    pass
            p_str = f"{int(p_val)} pulses" if p_val is not None else ""
            d_str = f"{float(dur):.1f} ms" if (dur is not None and str(dur).lower().strip() not in ("cont", "continuous")) else (f"{stim_dur_ms:.1f} ms" if stim_dur_ms > 0 else "")
            f_str = f"{float(freq):.0f} Hz" if freq is not None else ""
            specs = ", ".join([s for s in [f_str, p_str, d_str] if s])
            cond_label = f"Stimulation {specs} [BR {b_idx:03d}]"
            cond_type = "stim_reach"
    else:
        if is_stim:
            cond_label = f"Stimulation Reaches"
            cond_type = "stim_reach"
        else:
            br_str_list = ", ".join([f"BR_{b:03d}" for b in found_brs])
            cond_label = f"Control Baseline Reaches [{br_str_list}]"
            cond_type = "control_reach"

    print(f"[IntA Analysis] Condition: {cond_label}")

    # 5. Load manual trial removal records
    manual_remove_csv = REPO_ROOT / "config" / "manual_trial_remove.csv"
    manual_drop_dict: Dict[Tuple[str, int], set] = {}
    if manual_remove_csv.exists():
        try:
            mdf = pd.read_csv(manual_remove_csv)
            for _, r in mdf.iterrows():
                s_fn = str(r["intan_filename"]).strip()
                br_idx_val = int(r["br_idx"]) if pd.notna(r["br_idx"]) else -1
                drop_str = str(r["trials_to_drop"]) if pd.notna(r["trials_to_drop"]) else ""
                drop_list = [int(x.strip()) for x in drop_str.replace(";", ",").split(",") if x.strip().isdigit()]
                manual_drop_dict[(s_fn, br_idx_val)] = set(drop_list)
        except Exception as e:
            print(f"[IntA Analysis] Note: Could not parse manual_trial_remove.csv: {e}")

    # 6. Load spike sorting alignment if using raw PeriStim
    sort_curated_dir = None
    session_index_file = None
    shifts_df = None

    if not use_presegmented:
        candidate_sort_dirs = [
            out_base / "checkpoints" / "spikesorting" / "NPRW_concat_blanking" / "analyzer_manual",
            out_base / "checkpoints" / "spikesorting" / "NPRW_concat_blanking" / "sorting_curated",
            out_base / "checkpoints" / "spikesorting" / "NPRW_concat_blanking" / "analyzer_SLAy",
        ]
        for cdir in candidate_sort_dirs:
            if cdir.exists() and (cdir.parent / "session_index.pkl").exists():
                sort_curated_dir = cdir
                session_index_file = cdir.parent / "session_index.pkl"
                break

        shifts_csv_file = sess_dir / "Metadata" / "br_to_intan_shifts.csv" if sess_dir else None
        if shifts_csv_file and shifts_csv_file.exists():
            try:
                shifts_df = pd.read_csv(shifts_csv_file)
            except Exception:
                pass

    sorting_obj = None
    sess_names = []
    session_boundaries = []
    if si is not None and sort_curated_dir and session_index_file and shifts_df is not None:
        try:
            if "analyzer" in sort_curated_dir.name.lower():
                analyzer_obj = si.load_sorting_analyzer(sort_curated_dir)
                sorting_obj = analyzer_obj.sorting
            else:
                sorting_obj = si.load(sort_curated_dir)
            with open(session_index_file, "rb") as f:
                idx_info = pickle.load(f)
            sess_names = list(idx_info.get("sess_names", []))
            session_boundaries = list(idx_info.get("session_boundaries", []))
            print(f"[IntA Analysis] Loaded Kilosort4 sorting: {len(sorting_obj.unit_ids)} units.")
        except Exception:
            sorting_obj = None

    # 7. Iterate through files and extract aligned trials
    all_speeds: List[np.ndarray] = []
    all_dists: List[np.ndarray] = []
    all_labels: List[str] = []
    all_t_peak_accel: List[float] = []
    all_t_peak_speed: List[float] = []
    all_t_peak_decel: List[float] = []
    all_t_endpoint: List[float] = []
    unit_spikes_merged: Dict[int, List[np.ndarray]] = {}
    unit_ids_list: List[int] = []
    t_beh_ref: Optional[np.ndarray] = None
    sess_name_ref = ""

    WIN_EXTRACT_MIN = -800.0
    WIN_EXTRACT_MAX =  800.0

    for f in files_to_load:
        try:
            data = np.load(f, allow_pickle=True)
            sess_name = str(data.get("sess", data.get("session_name", f.stem)))
            if not sess_name_ref:
                sess_name_ref = sess_name

            t_beh = np.asarray(data["beh_rel_t"], float) if "beh_rel_t" in data else np.arange(-800, 801, 10, float)
            if t_beh_ref is None:
                t_beh_ref = t_beh

            m_br = re.search(r"BR_(\d+)", f.stem, re.IGNORECASE)
            br_num = int(m_br.group(1)) if m_br else -1

            # Load ground-truth middle finger kinematics matching plot_firing_rates.py
            cond_dist, cond_spd, cond_t = load_condition_kinematics(out_base, sess_name, br_num)
            if cond_t is not None and t_beh_ref is None:
                t_beh_ref = cond_t

            # A. PRE-SEGMENTED SORTED FILES (PeriStim_sorted)
            if use_presegmented and "trial_spikes" in data:
                uids = np.asarray(data["unit_ids"], int)
                if not unit_ids_list:
                    unit_ids_list = list(uids)
                    for u in unit_ids_list:
                        unit_spikes_merged[u] = []

                spk_mat = data["trial_spikes"]  # shape: (n_trials, n_units)
                n_tr_file = spk_mat.shape[0]
                target_str = str(data.get("target", "A")).upper().replace("TARGET_", "")
                tr_idx_file = data.get("trial_indices", np.arange(n_tr_file))

                # Use ground-truth middle finger kinematics from PeriStim
                if cond_dist is not None and cond_spd is not None and len(cond_dist) > np.max(tr_idx_file):
                    norm_pos_file = cond_dist[tr_idx_file]
                    spd_file = cond_spd[tr_idx_file]
                    if cond_t is not None:
                        t_beh_ref = cond_t
                elif "norm_pos" in data and len(data["norm_pos"]) == n_tr_file:
                    norm_pos_file = data["norm_pos"]
                    spd_file = data["speed"] if "speed" in data else np.zeros((n_tr_file, len(t_beh_ref)))
                else:
                    norm_pos_file = np.zeros((n_tr_file, len(t_beh_ref)))
                    spd_file = np.zeros((n_tr_file, len(t_beh_ref)))

                for tr_i in range(n_tr_file):
                    all_labels.append(target_str)
                    all_speeds.append(spd_file[tr_i:tr_i+1])
                    all_dists.append(norm_pos_file[tr_i:tr_i+1])

                    # Landmark calculations per trial
                    s_tr = spd_file[tr_i]
                    d_tr = norm_pos_file[tr_i]
                    t_acc, t_spd, t_dec, t_end = compute_trial_kinematic_landmarks(s_tr, d_tr, t_beh_ref)
                    all_t_peak_accel.append(t_acc)
                    all_t_peak_speed.append(t_spd)
                    all_t_peak_decel.append(t_dec)
                    all_t_endpoint.append(t_end)

                    for u_idx, uid in enumerate(unit_ids_list):
                        spk_pts = spk_mat[tr_i, u_idx]
                        unit_spikes_merged[uid].append(np.asarray(spk_pts, float) if spk_pts is not None else np.zeros(0))

                continue

            # B. STANDARD PERISTIM FILES
            event_ms = np.asarray(data["event_ms"], float) if "event_ms" in data else np.zeros(0, float)
            vel_segs = data.get("beh_cam1_vel_segs", data.get("beh_cam0_vel_segs"))
            pos_segs = data.get("beh_cam1_segs", data.get("beh_cam0_segs"))
            names = [str(x) for x in data.get("beh_cam1_names", data.get("beh_cam0_names", []))]

            if event_ms.size == 0:
                continue
            n_tr = len(event_ms)

            # Filter out manual trial drops
            drops_for_file = set()
            for (m_sfn, m_b), drop_set in manual_drop_dict.items():
                if (m_sfn in f.stem or m_sfn == sess_name) and (m_b == br_num or m_b == -1):
                    drops_for_file.update(drop_set)

            valid_trial_indices = [i for i in range(n_tr) if i not in drops_for_file]
            if not valid_trial_indices:
                continue

            # Use ground-truth middle finger kinematics from Camera 1
            if cond_dist is not None and cond_spd is not None and len(cond_dist) >= n_tr:
                dst_arr = cond_dist[valid_trial_indices]
                spd_arr = cond_spd[valid_trial_indices]
                if cond_t is not None:
                    t_beh_ref = cond_t
            else:
                idx_x, idx_y = find_keypoint_indices(names, "middle")
                if idx_x is None and len(names) >= 2:
                    idx_x, idx_y = 0, 1
                if vel_segs is not None and vel_segs.ndim == 3 and idx_x is not None and idx_y is not None:
                    vx = vel_segs[:, idx_x, :]
                    vy = vel_segs[:, idx_y, :]
                    spd_arr = np.sqrt(vx**2 + vy**2)[valid_trial_indices]
                else:
                    spd_arr = np.zeros((len(valid_trial_indices), len(t_beh)))
                dst_arr = np.zeros((len(valid_trial_indices), len(t_beh)))

            # Compute landmark times per valid trial
            for tr_local in range(len(spd_arr)):
                s_tr = spd_arr[tr_local]
                d_tr = dst_arr[tr_local]
                t_acc, t_spd, t_dec, t_end = compute_trial_kinematic_landmarks(s_tr, d_tr, t_beh_ref)
                all_t_peak_accel.append(t_acc)
                all_t_peak_speed.append(t_spd)
                all_t_peak_decel.append(t_dec)
                all_t_endpoint.append(t_end)

            # Target labels
            if "trial_labels" in data and len(data["trial_labels"]) == n_tr:
                lbls = [str(data["trial_labels"][i]).upper() for i in valid_trial_indices]
            else:
                target_str = "A" if "target_a" in str(f).lower() else ("B" if "target_b" in str(f).lower() else "A")
                lbls = [target_str] * len(valid_trial_indices)

            all_speeds.append(spd_arr)
            all_dists.append(dst_arr)
            all_labels.extend(lbls)

            # Slicing spikes from sorting or MUA
            # Match session boundary for sorted spikes
            matched_row = shifts_df[shifts_df["br_idx"] == br_num] if shifts_df is not None and "br_idx" in shifts_df else []
            shift_samp = matched_row["shift_sample"].values[0] if len(matched_row) > 0 else 0.0

            matched_sname = None
            if sorting_obj is not None:
                for sn in sess_names:
                    if sn in f.stem:
                        matched_sname = sn
                        break

            if sorting_obj is not None and matched_sname and matched_sname in sess_names:
                s_idx = sess_names.index(matched_sname)
                t0, t1 = session_boundaries[s_idx], session_boundaries[s_idx + 1]
                fs_sort = float(sorting_obj.get_sampling_frequency())

                if not unit_ids_list:
                    unit_ids_list = [int(u) for u in sorting_obj.unit_ids]
                    for u in unit_ids_list:
                        unit_spikes_merged[u] = []

                for uid in unit_ids_list:
                    st = sorting_obj.get_unit_spike_train(uid)
                    st_sess = st[(st >= t0) & (st < t1)] - t0
                    if len(st_sess) > 0 and rcp is not None:
                        st_br_ms = rcp.intan_samples_to_br_ms(st_sess, shift_samp, fs_intan=fs_sort, ppm_correction=-13.951)
                    else:
                        st_br_ms = np.zeros(0, float)

                    for tr_orig in valid_trial_indices:
                        if tr_orig < len(event_ms):
                            ev = event_ms[tr_orig]
                            in_win = (st_br_ms >= ev + WIN_EXTRACT_MIN) & (st_br_ms <= ev + WIN_EXTRACT_MAX)
                            unit_spikes_merged[uid].append(st_br_ms[in_win] - ev)
                        else:
                            unit_spikes_merged[uid].append(np.zeros(0, float))

            elif "NPRW_peak_ms_dedup" in data:
                # Fallback to matched filter channel threshold crossings
                peaks_dict = data["NPRW_peak_ms_dedup"].item() if hasattr(data["NPRW_peak_ms_dedup"], "item") else data["NPRW_peak_ms_dedup"]
                if not unit_ids_list:
                    unit_ids_list = [int(k) for k in sorted(peaks_dict.keys(), key=lambda x: int(x))]
                    for u in unit_ids_list:
                        unit_spikes_merged[u] = []

                for uid in unit_ids_list:
                    all_spk = np.asarray(peaks_dict.get(uid, peaks_dict.get(str(uid), [])), float).ravel()
                    for tr_orig in valid_trial_indices:
                        if tr_orig < len(event_ms):
                            ev = event_ms[tr_orig]
                            in_win = (all_spk >= ev + WIN_EXTRACT_MIN) & (all_spk <= ev + WIN_EXTRACT_MAX)
                            unit_spikes_merged[uid].append(all_spk[in_win] - ev)
                        else:
                            unit_spikes_merged[uid].append(np.zeros(0, float))

        except Exception as e:
            print(f"[IntA Analysis] Warn: Error reading {f.name}: {e}")
            continue

    if not all_speeds or t_beh_ref is None or len(all_labels) == 0:
        print("[IntA Analysis] No valid trial segments extracted.")
        return None

    # Merge arrays
    merged_speeds = np.vstack(all_speeds)
    merged_dists = np.vstack(all_dists)
    merged_labels = np.array(all_labels)

    kinematics = {
        "speed": merged_speeds,
        "norm_pos": merged_dists,
        "dist": merged_dists,
        "median_t_peak_accel": float(np.nanmedian(all_t_peak_accel)) if all_t_peak_accel else 60.0,
        "median_t_peak_speed": float(np.nanmedian(all_t_peak_speed)) if all_t_peak_speed else 120.0,
        "median_t_peak_decel": float(np.nanmedian(all_t_peak_decel)) if all_t_peak_decel else 220.0,
        "median_t_endpoint": float(np.nanmedian(all_t_endpoint)) if all_t_endpoint else 350.0,
    }

    n_A = int(np.sum(merged_labels == "A"))
    n_B = int(np.sum(merged_labels == "B"))
    print(f"[IntA Analysis] Dataset Ready: {len(unit_ids_list)} IntA units across {len(merged_labels)} total reaches "
          f"(Target A: {n_A}, Target B: {n_B}) | Condition: {cond_label}")

    if cond_type == "stim_reach" or is_stim:
        blank_win_ms = (-20.0, max(stim_dur_ms, 0.0) + 20.0)
        print(f"[IntA Analysis] Stimulation Condition: stim_dur={stim_dur_ms:.1f} ms | "
              f"Blanking Window = [{blank_win_ms[0]:.1f}, {blank_win_ms[1]:.1f}] ms (fully greyed out)")
    else:
        blank_win_ms = None

    return IntAReachingDataset(
        t_beh_ms=t_beh_ref,
        kinematics=kinematics,
        target_labels=merged_labels,
        unit_spikes=unit_spikes_merged,
        unit_ids=unit_ids_list,
        condition_label=cond_label,
        condition_type=cond_type,
        br_indices=found_brs,
        session_name=sess_name_ref,
        is_sorted=(sorting_obj is not None or use_presegmented),
        stim_dur_ms=stim_dur_ms,
        blank_win_ms=blank_win_ms,
    )


# =============================================================================
# 3. FIGURE 1: STAGGERED FORMATION & CLEAR TARGET A vs TARGET B SEPARATION
# =============================================================================
def _render_target_columns_4row(
    fig: plt.Figure,
    gs: gridspec.GridSpec,
    configs: List[Tuple[str, np.ndarray, np.ndarray, str, int]],
    dataset: IntAReachingDataset,
    trial_spikes: List[np.ndarray],
    med_t_peak_spd: float,
    med_t_endpoint: float,
    max_spd: float,
    max_fr: float,
    t_grid: np.ndarray,
    search_mask: np.ndarray,
    sub_t: np.ndarray,
) -> None:
    """Helper to render 4 staggered rows (Position, Speed, Raster, PSTH) into given subplots."""
    norm_pos_all = dataset.kinematics.get("norm_pos", dataset.kinematics.get("dist"))
    spd_all = dataset.kinematics.get("speed")
    med_t_peak_accel = dataset.kinematics.get("median_t_peak_accel", 60.0)
    med_t_peak_decel = dataset.kinematics.get("median_t_peak_decel", 220.0)
    blank_win = dataset.blank_win_ms
    b0, b1 = blank_win if blank_win is not None else (None, None)

    for tgt_name, tr_group, mat_rate, col_color, col_idx in configs:
        n_tr_tgt = len(tr_group)
        is_first_col = (col_idx == 0)

        # ---------------------------------------------------------------------
        # Row 1: Middle Finger Normalized Position [0–1] (Matching plot_firing_rates.py)
        # ---------------------------------------------------------------------
        ax_pos = fig.add_subplot(gs[0, col_idx])
        if norm_pos_all is not None and len(tr_group) > 0:
            pos_sub = norm_pos_all[tr_group]
            mean_p = np.nanmean(pos_sub, axis=0)
            # 95% CI or SEM band
            if rcp is not None:
                try:
                    c_mean, c_sd = rcp.mean_ci95_from_trial_traces(pos_sub)
                    sem_p = np.squeeze(c_sd)
                except Exception:
                    sem_p = np.nanstd(pos_sub, axis=0) / np.sqrt(max(len(pos_sub), 1))
            else:
                sem_p = np.nanstd(pos_sub, axis=0) / np.sqrt(max(len(pos_sub), 1))

            ax_pos.plot(dataset.t_beh_ms, mean_p, color=col_color, lw=2.4, label=f"Mean Position ({tgt_name})")
            ax_pos.fill_between(dataset.t_beh_ms, mean_p - sem_p, mean_p + sem_p, color=col_color, alpha=0.25)

        ax_pos.axvline(0, color="red", ls="--", lw=1.5, label="IR Cross (t=0)" if blank_win is None else "Stim Onset (t=0)")
        ax_pos.axvline(med_t_peak_accel, color="#0072B2", ls=":", lw=1.5, label=f"Peak Accel ({med_t_peak_accel:.0f} ms)")
        ax_pos.axvline(med_t_peak_spd, color="#7570b3", ls=":", lw=1.6, label=f"Peak Spd ({med_t_peak_spd:.0f} ms)")
        ax_pos.axvline(med_t_peak_decel, color="#d95f02", ls=":", lw=1.5, label=f"Peak Decel ({med_t_peak_decel:.0f} ms)")
        ax_pos.axvline(med_t_endpoint, color="forestgreen", ls="-.", lw=1.5, label=f"Endpoint ({med_t_endpoint:.0f} ms)")

        if blank_win is not None:
            ax_pos.axvspan(b0, b1, color="gray", alpha=0.35, zorder=5, label="Stim Blanking" if is_first_col else None)

        ax_pos.set_xlim(WIN_PLOT_MS)
        ax_pos.set_ylim(-0.1, 1.1)
        ax_pos.set_title(f"1. Middle Finger Normalized Position [0–1] | {tgt_name}", fontsize=12, fontweight="bold", pad=8)
        if is_first_col:
            ax_pos.set_ylabel("Norm. Position [0–1]", fontsize=11, fontweight="bold")
        ax_pos.legend(loc="upper left", frameon=False, fontsize=8.5)
        ax_pos.tick_params(labelbottom=False)
        ax_pos.grid(True, linestyle=":", alpha=0.4)

        # ---------------------------------------------------------------------
        # Row 2: Middle Finger Speed [cm/s]
        # ---------------------------------------------------------------------
        ax_spd = fig.add_subplot(gs[1, col_idx], sharex=ax_pos)
        if spd_all is not None and len(tr_group) > 0:
            spd_sub = spd_all[tr_group]
            mean_s = np.nanmean(spd_sub, axis=0)
            sem_s = np.nanstd(spd_sub, axis=0) / np.sqrt(max(len(spd_sub), 1))
            ax_spd.plot(dataset.t_beh_ms, mean_s, color=col_color, lw=2.4, label=f"Mean Speed ({tgt_name})")
            ax_spd.fill_between(dataset.t_beh_ms, mean_s - sem_s, mean_s + sem_s, color=col_color, alpha=0.25)

        ax_spd.axvline(0, color="red", ls="--", lw=1.5)
        ax_spd.axvline(med_t_peak_accel, color="#0072B2", ls=":", lw=1.5)
        ax_spd.axvline(med_t_peak_spd, color="#7570b3", ls=":", lw=1.6)
        ax_spd.axvline(med_t_peak_decel, color="#d95f02", ls=":", lw=1.5)
        ax_spd.axvline(med_t_endpoint, color="forestgreen", ls="-.", lw=1.5)

        if blank_win is not None:
            ax_spd.axvspan(b0, b1, color="gray", alpha=0.35, zorder=5)

        ax_spd.set_xlim(WIN_PLOT_MS)
        ax_spd.set_ylim(0, max_spd)
        ax_spd.set_title(f"2. Middle Finger Speed (cm/s) | {tgt_name}", fontsize=12, fontweight="bold", pad=8)
        if is_first_col:
            ax_spd.set_ylabel("Speed (cm/s)", fontsize=11, fontweight="bold")
        ax_spd.legend(loc="upper right", frameon=False, fontsize=8.5)
        ax_spd.tick_params(labelbottom=False)
        ax_spd.grid(True, linestyle=":", alpha=0.4)

        # ---------------------------------------------------------------------
        # Row 3: Spike Raster (NO EMPTY TRIALS, 1 to N_valid)
        # ---------------------------------------------------------------------
        ax_rast = fig.add_subplot(gs[2, col_idx], sharex=ax_pos)
        for row_num, tr in enumerate(tr_group):
            spks = trial_spikes[tr]
            if isinstance(spks, (list, np.ndarray)) and len(spks) > 0:
                spks_arr = np.asarray(spks, float)
                if blank_win is not None:
                    in_win = spks_arr[(spks_arr >= WIN_PLOT_MS[0]) & (spks_arr <= WIN_PLOT_MS[1]) & ((spks_arr < b0) | (spks_arr > b1))]
                else:
                    in_win = spks_arr[(spks_arr >= WIN_PLOT_MS[0]) & (spks_arr <= WIN_PLOT_MS[1])]
                if len(in_win) > 0:
                    ax_rast.vlines(in_win, row_num + 0.6, row_num + 1.4, color=col_color, lw=1.3, alpha=0.9)

        ax_rast.axvline(0, color="red", ls="--", lw=1.5)
        ax_rast.axvline(med_t_peak_accel, color="#0072B2", ls=":", lw=1.5)
        ax_rast.axvline(med_t_peak_spd, color="#7570b3", ls=":", lw=1.6)
        ax_rast.axvline(med_t_peak_decel, color="#d95f02", ls=":", lw=1.5)
        ax_rast.axvline(med_t_endpoint, color="forestgreen", ls="-.", lw=1.5)

        if blank_win is not None:
            ax_rast.axvspan(b0, b1, color="gray", alpha=0.45, zorder=5)

        ax_rast.set_xlim(WIN_PLOT_MS)
        ax_rast.set_ylim(n_tr_tgt + 0.5, 0.5)  # Inverted so Trial 1 is at top
        ax_rast.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax_rast.set_title(f"3. Spike Raster (Individual Reaching Trials, n = {n_tr_tgt}) | {tgt_name}",
                          fontsize=12, fontweight="bold", pad=8)
        if is_first_col:
            ax_rast.set_ylabel("Trial Number (1 to N)", fontsize=11, fontweight="bold")
        ax_rast.tick_params(labelbottom=False)
        ax_rast.grid(True, linestyle=":", alpha=0.3)

        # ---------------------------------------------------------------------
        # Row 4: Instantaneous Firing Rate PSTH (Mean ± SEM in Hz)
        # ---------------------------------------------------------------------
        ax_psth = fig.add_subplot(gs[3, col_idx], sharex=ax_pos)
        if len(mat_rate) > 0:
            m_r = np.nanmean(mat_rate, axis=0)
            s_r = np.nanstd(mat_rate, axis=0) / np.sqrt(max(len(mat_rate), 1))
            if blank_win is not None:
                mask_b = (t_grid >= b0) & (t_grid <= b1)
                m_r_plot = np.copy(m_r)
                s_r_plot = np.copy(s_r)
                m_r_plot[mask_b] = np.nan
                s_r_plot[mask_b] = np.nan
            else:
                m_r_plot = m_r
                s_r_plot = s_r

            ax_psth.plot(t_grid, m_r_plot, color=col_color, lw=2.6, label=f"PSTH ({tgt_name})")
            ax_psth.fill_between(t_grid, m_r_plot - s_r_plot, m_r_plot + s_r_plot, color=col_color, alpha=0.25)

            # Mark Peak Firing Rate
            if np.any(search_mask):
                sub_p = m_r[search_mask]
                pk_idx = np.argmax(sub_p)
                pk_t = float(sub_t[pk_idx])
                pk_val = float(sub_p[pk_idx])
                ax_psth.scatter(pk_t, pk_val, marker="*", s=160, color="crimson", zorder=6,
                                label=f"Peak: {pk_val:.1f} Hz ({pk_t:+.0f} ms)")
                ax_psth.annotate(f"Peak: {pk_val:.1f} Hz\n({pk_t:+.0f} ms)", (pk_t, pk_val),
                                 textcoords="offset points", xytext=(8, 4), fontsize=9, fontweight="bold", color="crimson")

        ax_psth.axvspan(BASELINE_WIN_MS[0], BASELINE_WIN_MS[1], color="gray", alpha=0.10, label="Baseline Rest")
        ax_psth.axvline(0, color="red", ls="--", lw=1.5)
        ax_psth.axvline(med_t_peak_accel, color="#0072B2", ls=":", lw=1.5)
        ax_psth.axvline(med_t_peak_spd, color="#7570b3", ls=":", lw=1.6)
        ax_psth.axvline(med_t_peak_decel, color="#d95f02", ls=":", lw=1.5)
        ax_psth.axvline(med_t_endpoint, color="forestgreen", ls="-.", lw=1.5)

        if blank_win is not None:
            ax_psth.axvspan(b0, b1, color="gray", alpha=0.35, zorder=5, label=f"Stim Blanking ({b0:+.0f} to {b1:+.0f} ms)" if is_first_col else None)

        ax_psth.set_xlim(WIN_PLOT_MS)
        ax_psth.set_ylim(0, max_fr)
        ax_psth.set_title(f"4. Instantaneous Firing Rate (Mean ± SEM in Hz) | {tgt_name}", fontsize=12, fontweight="bold", pad=8)
        if is_first_col:
            ax_psth.set_ylabel("Firing Rate (Hz)", fontsize=11, fontweight="bold")
        ax_psth.set_xlabel("Time from IR Beam Crossing (ms)", fontsize=11, fontweight="bold")
        ax_psth.legend(loc="upper left", frameon=False, fontsize=8.5)
        ax_psth.grid(True, linestyle=":", alpha=0.4)


def plot_inta_single_unit_profile(
    dataset: IntAReachingDataset,
    unit_id: int,
    out_path: Path,
    br_tag: Optional[str] = None,
) -> None:
    """
    Figure 1: Single-Unit IntA Firing Profile Centered on IR Beam Crossing (t = 0 ms).
    Outputs:
      1. Dedicated single-target figure for Target A in single_unit_profiles/target_A/
      2. Dedicated single-target figure for Target B in single_unit_profiles/target_B/
      3. Side-by-side comparison figure in single_unit_profiles/comparison_A_vs_B/ (and out_path)
    Each with 4 staggered subplots per target:
      * Row 1: Middle Finger Normalized Position [0–1] (matching plot_firing_rates.py)
      * Row 2: Middle Finger Speed [cm/s]
      * Row 3: Spike Raster (Individual Reaching Trials, NO blank rows!)
      * Row 4: Instantaneous Firing Rate PSTH [Hz]
    """
    if br_tag is None:
        br_tag = dataset.get_br_condition_tag()

    trial_spikes = dataset.unit_spikes.get(unit_id, [])
    if len(trial_spikes) == 0:
        return

    idx_A = np.flatnonzero(dataset.target_labels == "A")
    idx_B = np.flatnonzero(dataset.target_labels == "B")
    n_A, n_B = len(idx_A), len(idx_B)
    has_both = (n_A > 0 and n_B > 0)

    t_grid = np.arange(WIN_PLOT_MS[0], WIN_PLOT_MS[1] + 1.0, BIN_PSTH_MS)

    # Landmark times
    med_t_peak_spd = dataset.kinematics.get("median_t_peak_speed", 120.0)
    med_t_endpoint = dataset.kinematics.get("median_t_endpoint", 350.0)

    # Compute PSTHs per target
    rates_A = [smooth_spike_train(trial_spikes[tr], t_grid, sigma_ms=SIGMA_PSTH_MS) for tr in idx_A] if n_A > 0 else []
    rates_B = [smooth_spike_train(trial_spikes[tr], t_grid, sigma_ms=SIGMA_PSTH_MS) for tr in idx_B] if n_B > 0 else []
    mat_A = np.array(rates_A) if len(rates_A) > 0 else np.zeros((0, len(t_grid)))
    mat_B = np.array(rates_B) if len(rates_B) > 0 else np.zeros((0, len(t_grid)))

    # Global max speed and firing rate for shared Y-axis scaling
    spd_all = dataset.kinematics.get("speed")
    max_spd = float(np.nanpercentile(spd_all, 99)) * 1.15 if spd_all is not None and spd_all.size > 0 else 0.05
    max_spd = max(max_spd, 0.02)

    all_rates = np.vstack([mat_A, mat_B]) if (len(mat_A) > 0 and len(mat_B) > 0) else (mat_A if len(mat_A) > 0 else mat_B)
    max_fr = float(np.nanmax(np.nanmean(all_rates, axis=0))) * 1.35 if len(all_rates) > 0 else 20.0
    max_fr = max(max_fr, 10.0)

    search_mask = (t_grid >= -150.0) & (t_grid <= 450.0)
    if dataset.blank_win_ms is not None:
        b0, b1 = dataset.blank_win_ms
        search_mask = search_mask & ((t_grid < b0) | (t_grid > b1))
    sub_t = t_grid[search_mask]

    unit_type_str = "Spike-Sorted Single Unit" if dataset.is_sorted else "Channel Contact MUA"
    base_dir = out_path if out_path.is_dir() else out_path.parent
    target_a_dir = base_dir / "target_A"
    target_b_dir = base_dir / "target_B"
    comp_dir = base_dir / "comparison_A_vs_B"

    # 1. DEDICATED SINGLE-TARGET FIGURE: TARGET A
    if n_A > 0:
        target_a_dir.mkdir(parents=True, exist_ok=True)
        fig_A = plt.figure(figsize=(9.5, 13))
        gs_A = gridspec.GridSpec(4, 1, height_ratios=[1.0, 1.0, 1.4, 1.2], hspace=0.35)
        _render_target_columns_4row(
            fig_A, gs_A, [("Target A", idx_A, mat_A, COLORS["A"], 0)],
            dataset, trial_spikes, med_t_peak_spd, med_t_endpoint, max_spd, max_fr,
            t_grid, search_mask, sub_t
        )
        fig_A.suptitle(
            f"Anterior Interposed Nucleus (IntA) | {unit_type_str} {unit_id:03d} [{br_tag}]\n"
            f"Condition: {dataset.condition_label} | Monkey: {dataset.monkey}\n"
            f"Centered on IR Beam Crossing (t = 0 ms) | Target A (n = {n_A} Reaches)",
            fontsize=13, fontweight="bold", y=0.995,
        )
        tgt_a_out = target_a_dir / f"IntA_unit_{unit_id:03d}_{br_tag}_TargetA.png"
        fig_A.savefig(tgt_a_out, dpi=200, bbox_inches="tight")
        plt.close(fig_A)

    # 2. DEDICATED SINGLE-TARGET FIGURE: TARGET B
    if n_B > 0:
        target_b_dir.mkdir(parents=True, exist_ok=True)
        fig_B = plt.figure(figsize=(9.5, 13))
        gs_B = gridspec.GridSpec(4, 1, height_ratios=[1.0, 1.0, 1.4, 1.2], hspace=0.35)
        _render_target_columns_4row(
            fig_B, gs_B, [("Target B", idx_B, mat_B, COLORS["B"], 0)],
            dataset, trial_spikes, med_t_peak_spd, med_t_endpoint, max_spd, max_fr,
            t_grid, search_mask, sub_t
        )
        fig_B.suptitle(
            f"Anterior Interposed Nucleus (IntA) | {unit_type_str} {unit_id:03d} [{br_tag}]\n"
            f"Condition: {dataset.condition_label} | Monkey: {dataset.monkey}\n"
            f"Centered on IR Beam Crossing (t = 0 ms) | Target B (n = {n_B} Reaches)",
            fontsize=13, fontweight="bold", y=0.995,
        )
        tgt_b_out = target_b_dir / f"IntA_unit_{unit_id:03d}_{br_tag}_TargetB.png"
        fig_B.savefig(tgt_b_out, dpi=200, bbox_inches="tight")
        plt.close(fig_B)

    # 3. SIDE-BY-SIDE TARGET A vs TARGET B COMPARISON
    if has_both:
        comp_dir.mkdir(parents=True, exist_ok=True)
        fig_comp = plt.figure(figsize=(16, 13))
        gs_comp = gridspec.GridSpec(4, 2, height_ratios=[1.0, 1.0, 1.4, 1.2], hspace=0.35, wspace=0.22)
        configs = [("Target A", idx_A, mat_A, COLORS["A"], 0),
                   ("Target B", idx_B, mat_B, COLORS["B"], 1)]
        _render_target_columns_4row(
            fig_comp, gs_comp, configs,
            dataset, trial_spikes, med_t_peak_spd, med_t_endpoint, max_spd, max_fr,
            t_grid, search_mask, sub_t
        )
        fig_comp.suptitle(
            f"Anterior Interposed Nucleus (IntA) | {unit_type_str} {unit_id:03d} [{br_tag}]\n"
            f"Condition: {dataset.condition_label} | Monkey: {dataset.monkey}\n"
            f"Centered on IR Beam Crossing (t = 0 ms) | Target A (n = {n_A}) vs Target B (n = {n_B})",
            fontsize=14, fontweight="bold", y=0.995,
        )
        comp_out = comp_dir / f"IntA_unit_{unit_id:03d}_{br_tag}_TargetA_vs_TargetB.png"
        fig_comp.savefig(comp_out, dpi=200, bbox_inches="tight")
        plt.close(fig_comp)

    print(f"  [saved] IntA_unit_{unit_id:03d}_{br_tag} (per target profiles)")


# =============================================================================
# 3B. MULTI-CONDITION COMPARISON PROFILES
# =============================================================================
def get_condition_palette(br_indices: List[int], datasets_by_br: Dict[int, IntAReachingDataset]) -> Dict[int, str]:
    """
    Assign distinct, publication-grade colors to each Blackrock condition:
      - Control conditions: Royal Blue ('#1f77b4')
      - Stim conditions: Distinct high-contrast palette based on stim duration or order
    """
    stim_colors_by_duration = {
        2.5:   "#8c564b",  # Brown / Rust
        5.0:   "#9467bd",  # Purple
        10.0:  "#e377c2",  # Magenta / Pink
        20.0:  "#17becf",  # Teal / Cyan
        50.0:  "#2ca02c",  # Green
        100.0: "#d62728",  # Crimson / Red
    }
    fallback_palette = [
        "#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#17becf", "#8c564b", "#e377c2", "#bcbd22", "#333333"
    ]
    cond_colors: Dict[int, str] = {}
    fallback_idx = 0

    for b in sorted(br_indices):
        ds = datasets_by_br.get(b)
        if ds is not None and ds.condition_type == "control_reach":
            cond_colors[b] = "#1f77b4"  # Royal blue for control
        elif ds is not None and ds.stim_dur_ms in stim_colors_by_duration:
            cond_colors[b] = stim_colors_by_duration[ds.stim_dur_ms]
        else:
            cond_colors[b] = fallback_palette[fallback_idx % len(fallback_palette)]
            fallback_idx += 1

    return cond_colors


def _render_multi_condition_target_columns_4row(
    fig: plt.Figure,
    gs: gridspec.GridSpec,
    configs: List[Tuple[str, str, int]],  # (target_display_name, target_letter, col_idx)
    datasets_by_br: Dict[int, IntAReachingDataset],
    cond_colors: Dict[int, str],
    unit_id: int,
    med_t_peak_accel: float,
    med_t_peak_spd: float,
    med_t_peak_decel: float,
    med_t_endpoint: float,
    max_spd: float,
    max_fr: float,
    t_grid: np.ndarray,
) -> None:
    """Helper to render 4 staggered rows (Position, Speed, Separated Raster, Overlaid PSTH) for multi-condition comparison."""
    for tgt_name, tgt_letter, col_idx in configs:
        is_first_col = (col_idx == 0)

        # ---------------------------------------------------------------------
        # Row 1: Forward Hand Position (Norm 0-1) across Conditions
        # ---------------------------------------------------------------------
        ax_pos = fig.add_subplot(gs[0, col_idx])
        for b_idx, ds in datasets_by_br.items():
            idx_tgt = np.flatnonzero(ds.target_labels == tgt_letter)
            if len(idx_tgt) == 0:
                continue
            norm_pos = ds.kinematics.get("norm_pos", ds.kinematics.get("dist"))
            if norm_pos is not None and len(norm_pos) > np.max(idx_tgt):
                pos_sub = norm_pos[idx_tgt]
                m_p = np.nanmean(pos_sub, axis=0)
                s_p = np.nanstd(pos_sub, axis=0) / np.sqrt(max(len(pos_sub), 1))
                c_col = cond_colors.get(b_idx, "#333333")
                ax_pos.plot(ds.t_beh_ms, m_p, color=c_col, lw=2.2, label=f"BR {b_idx:03d} (n={len(idx_tgt)})")
                ax_pos.fill_between(ds.t_beh_ms, m_p - s_p, m_p + s_p, color=c_col, alpha=0.18)

        ax_pos.axvline(0, color="red", ls="--", lw=1.5, label="IR Cross (t=0)")
        ax_pos.axvline(med_t_peak_accel, color="#0072B2", ls=":", lw=1.5, label=f"Peak Accel ({med_t_peak_accel:.0f} ms)")
        ax_pos.axvline(med_t_peak_spd, color="#7570b3", ls=":", lw=1.6, label=f"Peak Speed ({med_t_peak_spd:.0f} ms)")
        ax_pos.axvline(med_t_peak_decel, color="#d95f02", ls=":", lw=1.5, label=f"Peak Decel ({med_t_peak_decel:.0f} ms)")
        ax_pos.axvline(med_t_endpoint, color="forestgreen", ls="-.", lw=1.5, label=f"Touch Endpoint ({med_t_endpoint:.0f} ms)")

        ax_pos.set_xlim(WIN_PLOT_MS)
        ax_pos.set_ylim(-0.08, 1.08)
        ax_pos.set_title(f"1. Forward Hand Position (Norm 0–1) | {tgt_name}", fontsize=12, fontweight="bold", pad=8)
        if is_first_col:
            ax_pos.set_ylabel("Norm. Position [0–1]", fontsize=10.5, fontweight="bold")
            ax_pos.legend(loc="upper left", frameon=False, fontsize=8.0, ncol=min(len(datasets_by_br), 3))
        ax_pos.tick_params(labelbottom=False)
        ax_pos.grid(True, linestyle=":", alpha=0.4)

        # ---------------------------------------------------------------------
        # Row 2: Instantaneous Speed (cm/s) across Conditions
        # ---------------------------------------------------------------------
        ax_spd = fig.add_subplot(gs[1, col_idx], sharex=ax_pos)
        for b_idx, ds in datasets_by_br.items():
            idx_tgt = np.flatnonzero(ds.target_labels == tgt_letter)
            if len(idx_tgt) == 0:
                continue
            spd_arr = ds.kinematics.get("speed")
            if spd_arr is not None and len(spd_arr) > np.max(idx_tgt):
                spd_sub = spd_arr[idx_tgt]
                m_s = np.nanmean(spd_sub, axis=0)
                s_s = np.nanstd(spd_sub, axis=0) / np.sqrt(max(len(spd_sub), 1))
                c_col = cond_colors.get(b_idx, "#333333")
                ax_spd.plot(ds.t_beh_ms, m_s, color=c_col, lw=2.2, label=f"BR {b_idx:03d}")
                ax_spd.fill_between(ds.t_beh_ms, m_s - s_s, m_s + s_s, color=c_col, alpha=0.18)

        ax_spd.axvline(0, color="red", ls="--", lw=1.5)
        ax_spd.axvline(med_t_peak_accel, color="#0072B2", ls=":", lw=1.5)
        ax_spd.axvline(med_t_peak_spd, color="#7570b3", ls=":", lw=1.6)
        ax_spd.axvline(med_t_peak_decel, color="#d95f02", ls=":", lw=1.5)
        ax_spd.axvline(med_t_endpoint, color="forestgreen", ls="-.", lw=1.5)

        ax_spd.set_xlim(WIN_PLOT_MS)
        ax_spd.set_ylim(0, max_spd)
        ax_spd.set_title(f"2. Instantaneous Speed (cm/s) | {tgt_name}", fontsize=12, fontweight="bold", pad=8)
        if is_first_col:
            ax_spd.set_ylabel("Speed (cm/s)", fontsize=10.5, fontweight="bold")
        ax_spd.tick_params(labelbottom=False)
        ax_spd.grid(True, linestyle=":", alpha=0.4)

        # ---------------------------------------------------------------------
        # Row 3: Spike Raster (Grouped & Separated by Condition Block)
        # ---------------------------------------------------------------------
        ax_rast = fig.add_subplot(gs[2, col_idx], sharex=ax_pos)
        curr_y = 1
        block_centers = []
        divider_ys = []
        total_trials = 0

        for b_idx, ds in datasets_by_br.items():
            idx_tgt = np.flatnonzero(ds.target_labels == tgt_letter)
            n_tr = len(idx_tgt)
            if n_tr == 0:
                continue
            total_trials += n_tr
            y_start = curr_y
            if y_start > 1:
                divider_ys.append(y_start - 0.5)

            c_col = cond_colors.get(b_idx, "#333333")
            spk_list = ds.unit_spikes.get(unit_id, [])
            blank_win = ds.blank_win_ms
            b0, b1 = blank_win if blank_win is not None else (None, None)

            for tr_local, tr_orig in enumerate(idx_tgt):
                y_tr = curr_y + tr_local
                spks = np.asarray(spk_list[tr_orig], float) if tr_orig < len(spk_list) else np.zeros(0, float)
                spks = spks[np.isfinite(spks)]
                if blank_win is not None:
                    spks = spks[(spks < b0) | (spks > b1)]
                if len(spks) > 0:
                    ax_rast.vlines(spks, y_tr - 0.38, y_tr + 0.38, color=c_col, lw=1.2, alpha=0.9)

            # Draw condition-specific blanking rectangle across this condition's trial rows
            if blank_win is not None:
                rect = mpatches.Rectangle(
                    (b0, y_start - 0.48),
                    b1 - b0,
                    n_tr + 0.05,
                    facecolor="gray",
                    alpha=0.35,
                    zorder=2,
                    edgecolor="none",
                )
                ax_rast.add_patch(rect)

            # In-plot condition tag
            short_desc = f"{ds.stim_dur_ms:.0f}ms stim" if ds.stim_dur_ms > 0 else "Control"
            blank_info = f" [blank: {b0:+.0f}..{b1:+.0f}ms]" if blank_win is not None else ""
            tag_str = f"BR {b_idx:03d}: {short_desc}{blank_info} (n={n_tr})"
            ax_rast.text(
                WIN_PLOT_MS[0] + 12, y_start - 0.1, tag_str,
                fontsize=8.0, fontweight="bold", color=c_col, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.90, edgecolor=c_col, lw=0.9),
                zorder=6,
            )

            center_y = (y_start + (curr_y + n_tr - 1)) / 2.0
            block_centers.append((center_y, f"BR {b_idx:03d}\n(n={n_tr})"))
            curr_y += n_tr + 1

        # Dividing lines between condition blocks
        for div_y in divider_ys:
            ax_rast.axhline(div_y, color="black", lw=1.2, ls="--", alpha=0.85, zorder=5)

        ax_rast.axvline(0, color="red", ls="--", lw=1.5)
        ax_rast.axvline(med_t_peak_accel, color="#0072B2", ls=":", lw=1.5)
        ax_rast.axvline(med_t_peak_spd, color="#7570b3", ls=":", lw=1.6)
        ax_rast.axvline(med_t_peak_decel, color="#d95f02", ls=":", lw=1.5)
        ax_rast.axvline(med_t_endpoint, color="forestgreen", ls="-.", lw=1.5)

        ax_rast.set_xlim(WIN_PLOT_MS)
        ax_rast.set_ylim(max(curr_y - 0.5, 1.5), 0.5)  # Inverted so Trial 1 is at top
        if block_centers:
            ax_rast.set_yticks([bc[0] for bc in block_centers])
            ax_rast.set_yticklabels([bc[1] for bc in block_centers], fontsize=8.5, fontweight="bold")
        if is_first_col:
            ax_rast.set_ylabel("Condition & Trials", fontsize=11, fontweight="bold")
        ax_rast.set_title(f"3. Spike Raster (Separated by Condition, n = {total_trials}) | {tgt_name}", fontsize=12, fontweight="bold", pad=8)
        ax_rast.tick_params(labelbottom=False)
        ax_rast.grid(True, linestyle=":", alpha=0.3)

        # ---------------------------------------------------------------------
        # Row 4: Instantaneous Firing Rate PSTH (One Colored Line per Condition)
        # ---------------------------------------------------------------------
        ax_psth = fig.add_subplot(gs[3, col_idx], sharex=ax_pos)
        for b_idx, ds in datasets_by_br.items():
            idx_tgt = np.flatnonzero(ds.target_labels == tgt_letter)
            if len(idx_tgt) == 0:
                continue
            c_col = cond_colors.get(b_idx, "#333333")
            spk_list = ds.unit_spikes.get(unit_id, [])
            rates = [smooth_spike_train(spk_list[tr], t_grid, sigma_ms=SIGMA_PSTH_MS) for tr in idx_tgt]
            mat_rate = np.array(rates)
            m_r = np.nanmean(mat_rate, axis=0)
            s_r = np.nanstd(mat_rate, axis=0) / np.sqrt(max(len(mat_rate), 1))

            blank_win = ds.blank_win_ms
            if blank_win is not None:
                b0, b1 = blank_win
                mask_b = (t_grid >= b0) & (t_grid <= b1)
                m_r_plot = np.copy(m_r)
                s_r_plot = np.copy(s_r)
                m_r_plot[mask_b] = np.nan
                s_r_plot[mask_b] = np.nan
                lbl = f"BR {b_idx:03d}: {ds.stim_dur_ms:.0f}ms stim (n={len(idx_tgt)})"
                ax_psth.axvspan(b0, b1, color="gray", alpha=0.18, zorder=2)
            else:
                m_r_plot = m_r
                s_r_plot = s_r
                lbl = f"BR {b_idx:03d}: Control (n={len(idx_tgt)})"

            ax_psth.plot(t_grid, m_r_plot, color=c_col, lw=2.4, label=lbl)
            ax_psth.fill_between(t_grid, m_r_plot - s_r_plot, m_r_plot + s_r_plot, color=c_col, alpha=0.15)

            # Search peak outside blanking
            search_mask = (t_grid >= -150.0) & (t_grid <= 450.0)
            if blank_win is not None:
                search_mask = search_mask & ((t_grid < b0) | (t_grid > b1))
            if np.any(search_mask):
                sub_t = t_grid[search_mask]
                sub_p = m_r[search_mask]
                if np.any(np.isfinite(sub_p)):
                    pk_idx = np.nanargmax(sub_p)
                    pk_t = float(sub_t[pk_idx])
                    pk_val = float(sub_p[pk_idx])
                    ax_psth.scatter(pk_t, pk_val, marker="o", s=50, color=c_col, edgecolors="black", lw=0.8, zorder=6)

        ax_psth.axvspan(BASELINE_WIN_MS[0], BASELINE_WIN_MS[1], color="gray", alpha=0.10, label="Baseline Rest")
        ax_psth.axvline(0, color="red", ls="--", lw=1.5)
        ax_psth.axvline(med_t_peak_accel, color="#0072B2", ls=":", lw=1.5)
        ax_psth.axvline(med_t_peak_spd, color="#7570b3", ls=":", lw=1.6)
        ax_psth.axvline(med_t_peak_decel, color="#d95f02", ls=":", lw=1.5)
        ax_psth.axvline(med_t_endpoint, color="forestgreen", ls="-.", lw=1.5)

        ax_psth.set_xlim(WIN_PLOT_MS)
        ax_psth.set_ylim(0, max_fr)
        ax_psth.set_title(f"4. Instantaneous Firing Rate (Mean ± SEM in Hz) | {tgt_name}", fontsize=12, fontweight="bold", pad=8)
        if is_first_col:
            ax_psth.set_ylabel("Firing Rate (Hz)", fontsize=11, fontweight="bold")
        ax_psth.set_xlabel("Time from IR Beam Crossing (ms)", fontsize=11, fontweight="bold")
        ax_psth.legend(loc="upper left", frameon=False, fontsize=8.0, ncol=1 if len(datasets_by_br) <= 4 else 2)
        ax_psth.grid(True, linestyle=":", alpha=0.4)


def plot_inta_multi_condition_comparison(
    datasets_by_br: Dict[int, IntAReachingDataset],
    unit_id: int,
    out_dir: Path,
    cond_colors: Dict[int, str],
) -> None:
    """
    Generate unified multi-condition comparison profiles for a single unit across all conditions:
      1. target_A/IntA_unit_{unit_id:03d}_multi_condition_TargetA.png
      2. target_B/IntA_unit_{unit_id:03d}_multi_condition_TargetB.png
      3. comparison_A_vs_B/IntA_unit_{unit_id:03d}_multi_condition_TargetA_vs_TargetB.png
    Each with 4 staggered rows:
      - Row 1: Forward Hand Position (Norm 0-1) [separate colored line per condition]
      - Row 2: Instantaneous Speed (cm/s) [separate colored line per condition]
      - Row 3: Spike Raster [separated trial blocks by condition, horizontal dividers, condition blanking shaded]
      - Row 4: Firing Rate PSTH [separate colored line per condition, NaN during condition blanking]
    """
    if not datasets_by_br:
        return

    # Check presence of trials for Target A and Target B
    n_tr_A = sum(int(np.sum(ds.target_labels == "A")) for ds in datasets_by_br.values())
    n_tr_B = sum(int(np.sum(ds.target_labels == "B")) for ds in datasets_by_br.values())
    has_A = (n_tr_A > 0)
    has_B = (n_tr_B > 0)
    has_both = (has_A and has_B)

    if not has_A and not has_B:
        return

    # Metadata from first dataset
    first_ds = next(iter(datasets_by_br.values()))
    monkey_name = first_ds.monkey
    session_ref = first_ds.session_name
    unit_type_str = "Spike-Sorted Single Unit" if any(ds.is_sorted for ds in datasets_by_br.values()) else "Channel Contact MUA"

    # Kinematic landmarks across datasets
    med_t_peak_accel = float(np.nanmedian([ds.kinematics.get("median_t_peak_accel", 60.0) for ds in datasets_by_br.values()]))
    med_t_peak_spd   = float(np.nanmedian([ds.kinematics.get("median_t_peak_speed", 120.0) for ds in datasets_by_br.values()]))
    med_t_peak_decel = float(np.nanmedian([ds.kinematics.get("median_t_peak_decel", 220.0) for ds in datasets_by_br.values()]))
    med_t_endpoint   = float(np.nanmedian([ds.kinematics.get("median_t_endpoint", 350.0) for ds in datasets_by_br.values()]))

    # Global max speed
    all_spds = [ds.kinematics.get("speed") for ds in datasets_by_br.values() if ds.kinematics.get("speed") is not None]
    if all_spds:
        concat_spd = np.concatenate([s.ravel() for s in all_spds if s.size > 0])
        max_spd = float(np.nanpercentile(concat_spd, 99)) * 1.15 if concat_spd.size > 0 else 0.05
    else:
        max_spd = 0.05
    max_spd = max(max_spd, 0.02)

    # Global max firing rate for this unit across all conditions & targets
    t_grid = np.arange(WIN_PLOT_MS[0], WIN_PLOT_MS[1] + 1.0, BIN_PSTH_MS)
    max_fr_val = 15.0
    for ds in datasets_by_br.values():
        spk_list = ds.unit_spikes.get(unit_id, [])
        for tgt_letter in ["A", "B"]:
            idx = np.flatnonzero(ds.target_labels == tgt_letter)
            if len(idx) > 0:
                rates = [smooth_spike_train(spk_list[tr], t_grid, sigma_ms=SIGMA_PSTH_MS) for tr in idx]
                m_r = np.nanmean(rates, axis=0)
                if np.any(np.isfinite(m_r)):
                    max_fr_val = max(max_fr_val, float(np.nanmax(m_r)) * 1.35)
    max_fr = max(max_fr_val, 15.0)

    # Subdirectories
    target_a_dir = out_dir / "target_A"
    target_b_dir = out_dir / "target_B"
    comp_dir = out_dir / "comparison_A_vs_B"

    # Condition summary tag for header
    cond_tags = ", ".join([f"BR {b:03d}" for b in datasets_by_br.keys()])

    # 1. DEDICATED TARGET A COMPARISON FIGURE
    if has_A:
        target_a_dir.mkdir(parents=True, exist_ok=True)
        fig_h = max(13.0, 10.0 + 0.06 * n_tr_A)
        fig_A = plt.figure(figsize=(10.5, fig_h))
        gs_A = gridspec.GridSpec(4, 1, height_ratios=[1.0, 1.0, max(1.5, 0.05 * n_tr_A), 1.2], hspace=0.35)
        _render_multi_condition_target_columns_4row(
            fig_A, gs_A, [("Target A", "A", 0)],
            datasets_by_br, cond_colors, unit_id,
            med_t_peak_accel, med_t_peak_spd, med_t_peak_decel, med_t_endpoint,
            max_spd, max_fr, t_grid,
        )
        fig_A.suptitle(
            f"Anterior Interposed Nucleus (IntA) | {unit_type_str} {unit_id:03d} [Multi-Condition Comparison]\n"
            f"Conditions: {cond_tags} | Monkey: {monkey_name}\n"
            f"Centered on IR Beam Crossing (t = 0 ms) | Target A ({n_tr_A} Total Reaches across {len(datasets_by_br)} Conditions)",
            fontsize=13, fontweight="bold", y=0.995,
        )
        tgt_a_out = target_a_dir / f"IntA_unit_{unit_id:03d}_multi_condition_TargetA.png"
        fig_A.savefig(tgt_a_out, dpi=200, bbox_inches="tight")
        plt.close(fig_A)

    # 2. DEDICATED TARGET B COMPARISON FIGURE
    if has_B:
        target_b_dir.mkdir(parents=True, exist_ok=True)
        fig_h = max(13.0, 10.0 + 0.06 * n_tr_B)
        fig_B = plt.figure(figsize=(10.5, fig_h))
        gs_B = gridspec.GridSpec(4, 1, height_ratios=[1.0, 1.0, max(1.5, 0.05 * n_tr_B), 1.2], hspace=0.35)
        _render_multi_condition_target_columns_4row(
            fig_B, gs_B, [("Target B", "B", 0)],
            datasets_by_br, cond_colors, unit_id,
            med_t_peak_accel, med_t_peak_spd, med_t_peak_decel, med_t_endpoint,
            max_spd, max_fr, t_grid,
        )
        fig_B.suptitle(
            f"Anterior Interposed Nucleus (IntA) | {unit_type_str} {unit_id:03d} [Multi-Condition Comparison]\n"
            f"Conditions: {cond_tags} | Monkey: {monkey_name}\n"
            f"Centered on IR Beam Crossing (t = 0 ms) | Target B ({n_tr_B} Total Reaches across {len(datasets_by_br)} Conditions)",
            fontsize=13, fontweight="bold", y=0.995,
        )
        tgt_b_out = target_b_dir / f"IntA_unit_{unit_id:03d}_multi_condition_TargetB.png"
        fig_B.savefig(tgt_b_out, dpi=200, bbox_inches="tight")
        plt.close(fig_B)

    # 3. SIDE-BY-SIDE TARGET A vs TARGET B COMPARISON FIGURE
    if has_both:
        comp_dir.mkdir(parents=True, exist_ok=True)
        n_tr_max = max(n_tr_A, n_tr_B)
        fig_h = max(13.0, 10.0 + 0.06 * n_tr_max)
        fig_comp = plt.figure(figsize=(18, fig_h))
        gs_comp = gridspec.GridSpec(4, 2, height_ratios=[1.0, 1.0, max(1.5, 0.05 * n_tr_max), 1.2], hspace=0.35, wspace=0.22)
        configs = [("Target A", "A", 0), ("Target B", "B", 1)]
        _render_multi_condition_target_columns_4row(
            fig_comp, gs_comp, configs,
            datasets_by_br, cond_colors, unit_id,
            med_t_peak_accel, med_t_peak_spd, med_t_peak_decel, med_t_endpoint,
            max_spd, max_fr, t_grid,
        )
        fig_comp.suptitle(
            f"Anterior Interposed Nucleus (IntA) | {unit_type_str} {unit_id:03d} [Multi-Condition Comparison]\n"
            f"Conditions: {cond_tags} | Monkey: {monkey_name}\n"
            f"Centered on IR Beam Crossing (t = 0 ms) | Target A (n = {n_tr_A}) vs Target B (n = {n_tr_B}) across {len(datasets_by_br)} Conditions",
            fontsize=14, fontweight="bold", y=0.995,
        )
        comp_out = comp_dir / f"IntA_unit_{unit_id:03d}_multi_condition_TargetA_vs_TargetB.png"
        fig_comp.savefig(comp_out, dpi=200, bbox_inches="tight")
        plt.close(fig_comp)

    print(f"  [saved] IntA_unit_{unit_id:03d} [multi-condition comparison]")
# =============================================================================
# 4. FIGURE 2: PEAK FIRING TIME & POPULATION LATENCY DIAGRAM
# =============================================================================
def compute_inta_population_modulation_stats(dataset: IntAReachingDataset) -> pd.DataFrame:
    """
    Quantifies peak firing latency and reach modulation across all recorded IntA single units:
      - Aligned to IR Beam Crossing (t = 0 ms)
      - Peak firing time t_peak_ms determined in reach movement window [-150, 450] ms
      - Classified into functional reach phases relative to Peak Speed and Endpoint
      - Paired Wilcoxon signed-rank test across trials (baseline [-500, -200] ms vs reach [-50, 350] ms)
      - Directional tuning test (Target A vs Target B)
    """
    records = []
    base_dur_s = (BASELINE_WIN_MS[1] - BASELINE_WIN_MS[0]) / 1000.0
    reach_dur_s = (REACH_WIN_MS[1] - REACH_WIN_MS[0]) / 1000.0

    med_t_peak_spd = dataset.kinematics.get("median_t_peak_speed", 120.0)
    med_t_endpoint = dataset.kinematics.get("median_t_endpoint", 350.0)

    t_grid = np.arange(WIN_PLOT_MS[0], WIN_PLOT_MS[1] + 1.0, BIN_PSTH_MS)

    for uid in dataset.unit_ids:
        spikes = dataset.unit_spikes.get(uid, [])
        if len(spikes) == 0:
            continue

        base_rates = []
        reach_rates = []
        reach_rates_A = []
        reach_rates_B = []
        trial_psths = []

        for tr in range(dataset.n_trials):
            spks = np.asarray(spikes[tr], float) if tr < len(spikes) else np.zeros(0, float)
            target = dataset.target_labels[tr]

            b_cnt = np.sum((spks >= BASELINE_WIN_MS[0]) & (spks <= BASELINE_WIN_MS[1]))
            r_cnt = np.sum((spks >= REACH_WIN_MS[0]) & (spks <= REACH_WIN_MS[1]))

            r_b = b_cnt / base_dur_s
            r_m = r_cnt / reach_dur_s

            base_rates.append(r_b)
            reach_rates.append(r_m)
            if target == "A":
                reach_rates_A.append(r_m)
            else:
                reach_rates_B.append(r_m)

            trial_psths.append(smooth_spike_train(spks, t_grid, sigma_ms=SIGMA_PSTH_MS))

        base_rates = np.array(base_rates)
        reach_rates = np.array(reach_rates)
        reach_rates_A = np.array(reach_rates_A)
        reach_rates_B = np.array(reach_rates_B)

        mean_base  = float(np.mean(base_rates)) if len(base_rates) > 0 else 0.0
        mean_reach = float(np.mean(reach_rates)) if len(reach_rates) > 0 else 0.0
        mean_reach_A = float(np.mean(reach_rates_A)) if len(reach_rates_A) > 0 else np.nan
        mean_reach_B = float(np.mean(reach_rates_B)) if len(reach_rates_B) > 0 else np.nan

        # Mean PSTH to find peak firing time
        mean_psth = np.nanmean(trial_psths, axis=0) if len(trial_psths) > 0 else np.zeros_like(t_grid)
        blank_win = dataset.blank_win_ms
        search_mask = (t_grid >= -150.0) & (t_grid <= 450.0)
        if blank_win is not None:
            b0, b1 = blank_win
            search_mask = search_mask & ((t_grid < b0) | (t_grid > b1))
        if np.any(search_mask):
            sub_t = t_grid[search_mask]
            sub_p = mean_psth[search_mask]
            max_idx = np.argmax(sub_p)
            t_peak = float(sub_t[max_idx])
            peak_val = float(sub_p[max_idx])
        else:
            t_peak = 0.0
            peak_val = mean_reach

        # Functional reach phase classification
        if t_peak < 0.0:
            phase = "Pre-Crossing"
        elif t_peak < med_t_peak_spd:
            phase = "Acceleration"
        elif t_peak < med_t_endpoint:
            phase = "Deceleration / Pre-Endpoint"
        else:
            phase = "Post-Touch / Endpoint"

        # Paired Wilcoxon signed-rank test
        diff = reach_rates - base_rates
        if np.all(diff == 0) or len(diff) < 5:
            p_val = 1.0
        else:
            try:
                _, p_val = stats.wilcoxon(reach_rates, base_rates, alternative="two-sided")
            except Exception:
                p_val = 1.0

        is_sig = (p_val < 0.05)
        if is_sig:
            cat = "Facilitated" if (mean_reach > mean_base) else "Suppressed"
        else:
            cat = "Unmodulated"

        # Directional tuning test (Target A vs Target B)
        if len(reach_rates_A) >= 3 and len(reach_rates_B) >= 3:
            try:
                _, p_dir = stats.mannwhitneyu(reach_rates_A, reach_rates_B, alternative="two-sided")
            except Exception:
                p_dir = 1.0
        else:
            p_dir = 1.0
        is_dir = (p_dir < 0.05)

        # Modulation Index: (Reach - Base) / (Reach + Base)
        mi = (mean_reach - mean_base) / (mean_reach + mean_base + 1e-6)

        records.append({
            "unit_id": uid,
            "mean_baseline_hz": mean_base,
            "mean_reach_hz": mean_reach,
            "mean_reach_A_hz": mean_reach_A,
            "mean_reach_B_hz": mean_reach_B,
            "peak_fr_hz": peak_val,
            "t_peak_ms": t_peak,
            "reach_phase": phase,
            "delta_hz": mean_reach - mean_base,
            "modulation_index": mi,
            "p_val_modulation": p_val,
            "category": cat,
            "p_val_direction": p_dir,
            "is_directional": is_dir,
            "is_sorted": dataset.is_sorted,
            "condition_label": dataset.condition_label,
        })

    return pd.DataFrame(records)


def plot_inta_population_firing_rate_heatmap(
    dataset: IntAReachingDataset,
    out_path: Path,
    time_window_ms: Tuple[float, float] = (-500.0, 500.0),
    baseline_window_ms: Tuple[float, float] = (-800.0, -650.0),
    cmap: str = "RdBu_r",
    z_score: bool = True,
    bin_ms: float = 10.0,
    sigma_ms: float = 25.0,
) -> None:
    """
    Figure: Population Firing Rate Heatmap of IntA Single Units Over Time (-500 to 500 ms).
    - Pre-reach baseline reference: [-800, -650] ms (matching plot_firing_rates.py).
    - Rows (units) sorted chronologically by response peak latency.
    - Colormap: RdBu_r (0 baseline centered at white; red = excitation, blue = suppression).
    - Includes aligned kinematic reach speed on top and landmark timing lines.
    - Handles Target A, Target B, and Combined reaching conditions.
    """
    if dataset is None or len(dataset.unit_ids) == 0 or dataset.n_trials == 0:
        return

    # Evaluation time grid for Gaussian smoothing and baseline reference
    t_start = min(baseline_window_ms[0], time_window_ms[0])
    t_end = max(baseline_window_ms[1], time_window_ms[1])
    t_grid_full = np.arange(t_start, t_end + 1.0, bin_ms)
    base_mask = (t_grid_full >= baseline_window_ms[0]) & (t_grid_full <= baseline_window_ms[1])
    plot_mask = (t_grid_full >= time_window_ms[0]) & (t_grid_full <= time_window_ms[1])
    t_plot = t_grid_full[plot_mask]

    # Target groupings
    idx_A = np.flatnonzero(dataset.target_labels == "A")
    idx_B = np.flatnonzero(dataset.target_labels == "B")
    n_A, n_B = len(idx_A), len(idx_B)
    has_both = (n_A > 0 and n_B > 0)

    # Compute baseline-subtracted firing rates for each unit
    all_unit_deltas_A = []
    all_unit_deltas_B = []
    all_unit_deltas_all = []

    for uid in dataset.unit_ids:
        spk_list = dataset.unit_spikes.get(uid, [])
        tr_deltas = []
        for tr in range(dataset.n_trials):
            spk = spk_list[tr] if tr < len(spk_list) else np.zeros(0, float)
            fr = smooth_spike_train(spk, t_grid_full, sigma_ms=sigma_ms)
            b_val = np.nanmean(fr[base_mask]) if np.any(base_mask) else 0.0
            tr_deltas.append(fr - b_val)
        tr_deltas = np.array(tr_deltas)  # (n_trials, n_eval)

        # Average per target
        if n_A > 0:
            all_unit_deltas_A.append(np.nanmean(tr_deltas[idx_A], axis=0)[plot_mask])
        if n_B > 0:
            all_unit_deltas_B.append(np.nanmean(tr_deltas[idx_B], axis=0)[plot_mask])
        all_unit_deltas_all.append(np.nanmean(tr_deltas, axis=0)[plot_mask])

    mat_all = np.array(all_unit_deltas_all)  # (n_units, n_time_plot)
    mat_A = np.array(all_unit_deltas_A) if n_A > 0 else mat_all
    mat_B = np.array(all_unit_deltas_B) if n_B > 0 else mat_all
    n_units = len(dataset.unit_ids)

    blank_win = dataset.blank_win_ms
    b0, b1 = blank_win if blank_win is not None else (None, None)

    # Sort units by peak response latency in reach epoch [-200, 400] ms based on Target A
    search_m = (t_plot >= -200.0) & (t_plot <= 400.0)
    if blank_win is not None:
        search_m = search_m & ((t_plot < b0) | (t_plot > b1))
    sort_base = mat_A if (n_A > 0 and mat_A.shape[0] > 0) else mat_all
    if np.any(search_m):
        peak_idx = np.argmax(sort_base[:, search_m], axis=1)
        peak_times = t_plot[search_m][peak_idx]
    else:
        peak_idx = np.argmax(sort_base, axis=1)
        peak_times = t_plot[peak_idx]
    sort_order = np.argsort(peak_times, kind="stable")

    mat_all = mat_all[sort_order]
    if n_A > 0:
        mat_A = mat_A[sort_order]
    if n_B > 0:
        mat_B = mat_B[sort_order]
    sorted_uids = [dataset.unit_ids[i] for i in sort_order]

    # Z-score or symmetric firing rate scale
    if z_score:
        def _zscore_mat(m):
            mu = np.nanmean(m, axis=1, keepdims=True)
            sig = np.clip(np.nanstd(m, axis=1, keepdims=True), 1e-3, None)
            return (m - mu) / sig
        mat_all_disp = _zscore_mat(mat_all)
        mat_A_disp = _zscore_mat(mat_A) if n_A > 0 else mat_all_disp
        mat_B_disp = _zscore_mat(mat_B) if n_B > 0 else mat_all_disp
        vmin, vmax = -2.5, 2.5
        cb_label = "Modulation z-score (std)"
    else:
        vlim = float(np.nanpercentile(np.abs(np.concatenate([mat_A, mat_B] if has_both else [mat_all])), 98))
        vlim = max(vlim, 1.0)
        vmin, vmax = -vlim, vlim
        cb_label = "Δ Firing Rate (Hz) vs [-800, -650] ms"
        mat_all_disp = mat_all
        mat_A_disp = mat_A
        mat_B_disp = mat_B

    if blank_win is not None:
        blank_cols = (t_plot >= b0) & (t_plot <= b1)
        mat_all_disp = np.copy(mat_all_disp)
        mat_all_disp[:, blank_cols] = np.nan
        if n_A > 0:
            mat_A_disp = np.copy(mat_A_disp)
            mat_A_disp[:, blank_cols] = np.nan
        if n_B > 0:
            mat_B_disp = np.copy(mat_B_disp)
            mat_B_disp[:, blank_cols] = np.nan

    # Landmarks & Kinematics
    norm_pos_all = dataset.kinematics.get("norm_pos", dataset.kinematics.get("dist"))
    spd_all = dataset.kinematics.get("speed")
    med_t_peak_accel = dataset.kinematics.get("median_t_peak_accel", 60.0)
    med_t_peak_spd   = dataset.kinematics.get("median_t_peak_speed", 120.0)
    med_t_peak_decel = dataset.kinematics.get("median_t_peak_decel", 220.0)
    med_t_endpoint   = dataset.kinematics.get("median_t_endpoint", 350.0)

    # Setup Figure (3 Rows: 1. Normalized Position, 2. Speed, 3. Firing Rate Heatmap)
    if has_both:
        cols_config = [
            ("Target A (Left Reaches)", mat_A_disp, idx_A, COLORS.get("Target A", "#1f77b4")),
            ("Target B (Right Reaches)", mat_B_disp, idx_B, COLORS.get("Target B", "#d95f02")),
        ]
        fig = plt.figure(figsize=(17, 12.0))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1.0, 1.0, 3.8], hspace=0.18, wspace=0.18,
                               left=0.08, right=0.90, top=0.88, bottom=0.07)
    else:
        tgt_name = "Target A (Left Reach)" if n_A > 0 else ("Target B (Right Reach)" if n_B > 0 else "All Reaches")
        col_color = COLORS.get("Target A" if n_A > 0 else "Target B", "#1f77b4")
        disp_mat = mat_A_disp if n_A > 0 else (mat_B_disp if n_B > 0 else mat_all_disp)
        tr_indices = idx_A if n_A > 0 else (idx_B if n_B > 0 else np.arange(dataset.n_trials))
        cols_config = [(f"{tgt_name} (n = {len(tr_indices)})", disp_mat, tr_indices, col_color)]
        fig = plt.figure(figsize=(11, 12.0))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1.0, 1.0, 3.8], hspace=0.18,
                               left=0.11, right=0.88, top=0.88, bottom=0.07)

    im_ref = None
    for col_idx, (col_title, mat_disp, tr_grp, col_c) in enumerate(cols_config):
        # ---------------------------------------------------------------------
        # Row 1: Normalized Position [0–1]
        # ---------------------------------------------------------------------
        ax_pos = fig.add_subplot(gs[0, col_idx])
        if norm_pos_all is not None and len(tr_grp) > 0:
            pos_sub = norm_pos_all[tr_grp]
            m_p = np.nanmean(pos_sub, axis=0)
            s_p = np.nanstd(pos_sub, axis=0) / np.sqrt(max(len(pos_sub), 1))
            ax_pos.plot(dataset.t_beh_ms, m_p, color=col_c, lw=2.2, label="Mean Position")
            ax_pos.fill_between(dataset.t_beh_ms, m_p - s_p, m_p + s_p, color=col_c, alpha=0.25)

        ax_pos.axvline(0, color="red", ls="--", lw=1.5, label="IR Cross (t=0)" if blank_win is None else "Stim Onset (t=0)")
        ax_pos.axvline(med_t_peak_accel, color="#0072B2", ls=":", lw=1.5, label=f"Peak Accel ({med_t_peak_accel:.0f} ms)")
        ax_pos.axvline(med_t_peak_spd, color="#7570b3", ls=":", lw=1.6, label=f"Peak Speed ({med_t_peak_spd:.0f} ms)")
        ax_pos.axvline(med_t_peak_decel, color="#d95f02", ls=":", lw=1.5, label=f"Peak Decel ({med_t_peak_decel:.0f} ms)")
        ax_pos.axvline(med_t_endpoint, color="forestgreen", ls="-.", lw=1.5, label=f"Touch Endpoint ({med_t_endpoint:.0f} ms)")

        if blank_win is not None:
            ax_pos.axvspan(b0, b1, color="gray", alpha=0.35, zorder=5, label="Stim Blanking" if col_idx == 0 else None)

        ax_pos.set_xlim(time_window_ms)
        ax_pos.set_ylim(-0.08, 1.08)
        ax_pos.set_title(col_title, fontsize=12.5, fontweight="bold", pad=8)
        if col_idx == 0:
            ax_pos.set_ylabel("Norm. Position [0–1]", fontsize=10.5, fontweight="bold")
            ax_pos.legend(loc="upper left", frameon=False, fontsize=8.0, ncol=3)
        ax_pos.tick_params(labelbottom=False)
        ax_pos.grid(True, linestyle=":", alpha=0.4)

        # ---------------------------------------------------------------------
        # Row 2: Kinematic Speed (cm/s)
        # ---------------------------------------------------------------------
        ax_spd = fig.add_subplot(gs[1, col_idx], sharex=ax_pos)
        if spd_all is not None and len(tr_grp) > 0:
            spd_sub = spd_all[tr_grp]
            m_s = np.nanmean(spd_sub, axis=0)
            s_s = np.nanstd(spd_sub, axis=0) / np.sqrt(max(len(spd_sub), 1))
            ax_spd.plot(dataset.t_beh_ms, m_s, color=col_c, lw=2.2, label="Mean Speed")
            ax_spd.fill_between(dataset.t_beh_ms, m_s - s_s, m_s + s_s, color=col_c, alpha=0.25)

        ax_spd.axvline(0, color="red", ls="--", lw=1.5)
        ax_spd.axvline(med_t_peak_accel, color="#0072B2", ls=":", lw=1.5)
        ax_spd.axvline(med_t_peak_spd, color="#7570b3", ls=":", lw=1.6)
        ax_spd.axvline(med_t_peak_decel, color="#d95f02", ls=":", lw=1.5)
        ax_spd.axvline(med_t_endpoint, color="forestgreen", ls="-.", lw=1.5)

        if blank_win is not None:
            ax_spd.axvspan(b0, b1, color="gray", alpha=0.35, zorder=5)

        ax_spd.set_xlim(time_window_ms)
        if col_idx == 0:
            ax_spd.set_ylabel("Speed (cm/s)", fontsize=10.5, fontweight="bold")
            ax_spd.legend(loc="upper right", frameon=False, fontsize=8.0)
        ax_spd.tick_params(labelbottom=False)
        ax_spd.grid(True, linestyle=":", alpha=0.4)

        # ---------------------------------------------------------------------
        # Row 3: Population Firing Rate Heatmap
        # ---------------------------------------------------------------------
        ax_hm = fig.add_subplot(gs[2, col_idx], sharex=ax_pos)
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="gray")
        im = ax_hm.imshow(
            mat_disp,
            extent=[time_window_ms[0], time_window_ms[1], n_units - 0.5, -0.5],
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        if im_ref is None:
            im_ref = im

        ax_hm.axvline(0, color="red", ls="--", lw=1.5)
        ax_hm.axvline(med_t_peak_accel, color="#0072B2", ls=":", lw=1.5)
        ax_hm.axvline(med_t_peak_spd, color="#7570b3", ls=":", lw=1.6)
        ax_hm.axvline(med_t_peak_decel, color="#d95f02", ls=":", lw=1.5)
        ax_hm.axvline(med_t_endpoint, color="forestgreen", ls="-.", lw=1.5)

        if blank_win is not None:
            ax_hm.axvspan(b0, b1, color="gray", alpha=0.55, zorder=10)

        ax_hm.set_xlim(time_window_ms)
        ax_hm.set_ylim(n_units - 0.5, -0.5)  # Row 0 (earliest peak) at top
        ax_hm.set_xlabel("Time from IR Beam Crossing (ms)", fontsize=11, fontweight="bold")
        if col_idx == 0:
            sort_lbl = "Target A" if n_A > 0 else "All Reaches"
            ax_hm.set_ylabel(f"IntA Unit (sorted by {sort_lbl} peak response latency)", fontsize=11, fontweight="bold")
            step = 1 if n_units <= 25 else (5 if n_units <= 60 else 10)
            y_ticks = np.arange(0, n_units, step)
            ax_hm.set_yticks(y_ticks)
            ax_hm.set_yticklabels([f"u{sorted_uids[i]}" for i in y_ticks], fontsize=8.5)
        else:
            ax_hm.tick_params(labelleft=False)

    # Colorbar
    cbar_ax = fig.add_axes([0.915, 0.08, 0.02, 0.52])
    cbar = fig.colorbar(im_ref, cax=cbar_ax)
    cbar.set_label(cb_label, fontsize=11, fontweight="bold")

    # Overall Header
    metric_name = "Normalized Modulation (z-score)" if z_score else "Δ Firing Rate (Hz)"
    cond_title = dataset.condition_label
    br_tag = dataset.get_br_condition_tag()
    if len(cond_title) > 65:
        cond_title = cond_title[:62] + "..."
    blank_str = f" | Stim Blanking: [{b0:+.0f}, {b1:+.0f}] ms" if blank_win is not None else ""
    fig.suptitle(
        f"IntA Population Firing Rates ({metric_name}) [{br_tag}] | Monkey Nike\n"
        f"{cond_title} | n = {n_units} units{blank_str}\n"
        f"Pre-Reach Baseline Reference: [{baseline_window_ms[0]:.0f}, {baseline_window_ms[1]:.0f}] ms",
        fontsize=12,
        fontweight="bold",
        y=0.96,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  [saved] Population Firing Rate Heatmap -> {out_path.name}")


def plot_inta_unit_quality_diagnostic(
    dataset: IntAReachingDataset,
    unit_id: int,
    out_path: Path,
    analyzer: Optional[Any] = None,
    br_tag: Optional[str] = None,
) -> None:
    """
    Spike Sorting Quality & Isolation Diagnostic Card for a single unit:
      - Panel A: Action Potential Waveform on Peak Contacts (from template or analyzer)
      - Panel B: Inter-Spike Interval (ISI) Distribution with 1.5 ms Refractory Line
      - Panel C: Spike Autocorrelogram (ACG, [-50, +50] ms) showing Refractory Dip
      - Panel D: Trial-by-Trial Firing Rate Stability across Session Timeline
    """
    if br_tag is None:
        br_tag = dataset.get_br_condition_tag()

    trial_spikes = dataset.unit_spikes.get(unit_id, [])
    if len(trial_spikes) == 0:
        return

    # Aggregate all spikes and ISIs
    all_spikes_list = []
    all_isis_list = []
    reach_frs = []
    base_frs = []
    blank_win = dataset.blank_win_ms
    b0, b1 = blank_win if blank_win is not None else (None, None)

    for tr in range(dataset.n_trials):
        spks = np.asarray(trial_spikes[tr], float) if tr < len(trial_spikes) else np.zeros(0, float)
        spks = spks[np.isfinite(spks)]
        if blank_win is not None:
            spks_clean = spks[(spks < b0) | (spks > b1)]
        else:
            spks_clean = spks
        all_spikes_list.append(spks_clean)

        # Firing rates in reach vs baseline
        r_dur = (REACH_WIN_MS[1] - REACH_WIN_MS[0]) / 1000.0
        b_dur = (BASELINE_WIN_MS[1] - BASELINE_WIN_MS[0]) / 1000.0
        if blank_win is not None:
            r_cnt = np.sum((spks >= REACH_WIN_MS[0]) & (spks <= REACH_WIN_MS[1]) & ((spks < b0) | (spks > b1)))
            ov_b0 = max(REACH_WIN_MS[0], b0)
            ov_b1 = min(REACH_WIN_MS[1], b1)
            blank_dur_s = max(0.0, (ov_b1 - ov_b0) / 1000.0)
            effective_r_dur = max(r_dur - blank_dur_s, 0.05)
            reach_frs.append(r_cnt / effective_r_dur)
        else:
            r_cnt = np.sum((spks >= REACH_WIN_MS[0]) & (spks <= REACH_WIN_MS[1]))
            reach_frs.append(r_cnt / r_dur if r_dur > 0 else 0.0)
        b_cnt = np.sum((spks >= BASELINE_WIN_MS[0]) & (spks <= BASELINE_WIN_MS[1]))
        base_frs.append(b_cnt / b_dur if b_dur > 0 else 0.0)

        if len(spks_clean) >= 2:
            spks_sort = np.sort(spks_clean)
            all_isis_list.extend(np.diff(spks_sort))

    all_isis = np.array(all_isis_list)
    total_spikes = sum(len(s) for s in all_spikes_list)

    # Refractory violation statistics (< 1.5 ms)
    valid_isis = all_isis[all_isis > 0]
    n_violations = int(np.sum(valid_isis < 1.5))
    violation_pct = (n_violations / max(len(valid_isis), 1)) * 100.0

    if violation_pct < 1.0:
        badge_color = "#1b7837"
        status_str = "CLEAN SINGLE UNIT (< 1% violations)"
    elif violation_pct < 2.5:
        badge_color = "#d95f02"
        status_str = "LIKELY MUA / CLUSTER (1-2.5% violations)"
    else:
        badge_color = "#b2182b"
        status_str = "HIGH CONTAMINATION / MUA (> 2.5% violations)"

    fig = plt.figure(figsize=(13.5, 10.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.34, wspace=0.28)

    # -------------------------------------------------------------------------
    # Panel A: Mean Action Potential Waveform
    # -------------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    template = None

    if analyzer is not None:
        try:
            if hasattr(analyzer, "get_template"):
                template = analyzer.get_template(unit_id)
            elif hasattr(analyzer, "get_extension"):
                tpl_ext = analyzer.get_extension("templates")
                if tpl_ext is not None:
                    all_tpls = tpl_ext.get_templates()
                    u_list = list(analyzer.unit_ids)
                    if unit_id in u_list:
                        template = all_tpls[u_list.index(unit_id)]
        except Exception:
            template = None

    if template is not None and isinstance(template, np.ndarray) and template.ndim == 2:
        n_samp, n_chan = template.shape
        fs_wf = 30000.0
        t_wf_ms = (np.arange(n_samp) - n_samp // 3) / fs_wf * 1000.0

        ptp = np.ptp(template, axis=0)
        peak_ch = int(np.argmax(ptp))
        peak_amp = float(ptp[peak_ch])
        top_chans = np.argsort(ptp)[-4:][::-1]

        for ch in top_chans:
            if ch != peak_ch:
                ax_a.plot(t_wf_ms, template[:, ch], color="gray", lw=1.2, alpha=0.55, label=f"Ch {ch}")
        ax_a.plot(t_wf_ms, template[:, peak_ch], color="#08519c", lw=2.5, label=f"Peak Ch {peak_ch} ({peak_amp:.1f} uV)")
        ax_a.axvline(0, color="black", ls=":", lw=1.0)
        ax_a.set_xlabel("Time (ms)", fontsize=10, fontweight="bold")
        ax_a.set_ylabel("Amplitude (uV)", fontsize=10, fontweight="bold")
        ax_a.set_title(f"A. Action Potential Waveform (Peak Contact {peak_ch})", fontsize=11, fontweight="bold")
        ax_a.legend(loc="upper right", frameon=False, fontsize=8.5)
        ax_a.grid(True, linestyle=":", alpha=0.4)
    else:
        # Fallback informative display
        ax_a.text(0.5, 0.55,
                  f"Kilosort4 Template / Waveform\n\nUnit ID: {unit_id:03d}\nTotal Spikes Analyzed: {total_spikes:,}\n"
                  f"Reach Mean FR: {np.mean(reach_frs):.1f} Hz\nBaseline Rest FR: {np.mean(base_frs):.1f} Hz\n\n"
                  f"For multi-channel geometry waveforms, open:\nphy template-gui params.py",
                  ha="center", va="center", fontsize=9.5, transform=ax_a.transAxes,
                  bbox=dict(boxstyle="round,pad=0.6", fc="#f7f7f7", ec="#cccccc", lw=1.5))
        ax_a.set_title(f"A. Unit Isolation & Firing Statistics | Unit {unit_id:03d}", fontsize=11, fontweight="bold")
        ax_a.axis("off")

    # -------------------------------------------------------------------------
    # Panel B: Inter-Spike Interval (ISI) Distribution
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    if len(valid_isis) > 0:
        isi_sub = valid_isis[valid_isis <= 50.0]
        bins_isi = np.arange(0, 51.0, 1.0)
        ax_b.hist(isi_sub, bins=bins_isi, color="#2b5c8f", edgecolor="black", alpha=0.75, label="Observed ISIs")
        ax_b.axvline(1.5, color="crimson", ls="--", lw=2.2, label="1.5 ms Refractory Limit")
        ax_b.set_xlim(0, 50)
        ax_b.set_xlabel("Inter-Spike Interval (ms)", fontsize=10, fontweight="bold")
        ax_b.set_ylabel("Spike Count", fontsize=10, fontweight="bold")
        ax_b.set_title(f"B. ISI Distribution (Violations < 1.5 ms: {violation_pct:.2f}%)", fontsize=11, fontweight="bold")
        ax_b.legend(loc="upper right", frameon=False, fontsize=8.5)
        ax_b.annotate(status_str, xy=(0.96, 0.76), xycoords="axes fraction", ha="right",
                      fontsize=9.0, fontweight="bold", color=badge_color,
                      bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=badge_color, lw=1.6))
        ax_b.grid(True, linestyle=":", alpha=0.3)
    else:
        ax_b.text(0.5, 0.5, "Insufficient spikes for ISI distribution", ha="center", va="center", transform=ax_b.transAxes)
        ax_b.set_title("B. ISI Distribution", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel C: Spike Autocorrelogram (ACG)
    # -------------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    lags = []
    for spks in all_spikes_list:
        if len(spks) >= 2:
            diffs = spks[:, None] - spks[None, :]
            np.fill_diagonal(diffs, np.nan)
            sub_d = diffs[np.isfinite(diffs) & (np.abs(diffs) <= 50.0)]
            lags.extend(sub_d)

    if len(lags) > 0:
        bins_acg = np.arange(-50.5, 51.5, 1.0)
        ax_c.hist(lags, bins=bins_acg, color="#4393c3", edgecolor="#2166ac", alpha=0.8, label="Spike ACG")
        ax_c.axvline(0, color="crimson", ls="--", lw=1.5)
        ax_c.axvspan(-1.5, 1.5, color="crimson", alpha=0.18, label="Refractory Notch (±1.5 ms)")
        ax_c.set_xlim(-50, 50)
        ax_c.set_xlabel("Time Lag (ms)", fontsize=10, fontweight="bold")
        ax_c.set_ylabel("Coincidence Count", fontsize=10, fontweight="bold")
        ax_c.set_title("C. Spike Autocorrelogram (Dip to 0 at t=0 ms)", fontsize=11, fontweight="bold")
        ax_c.legend(loc="upper right", frameon=False, fontsize=8.5)
        ax_c.grid(True, linestyle=":", alpha=0.3)
    else:
        ax_c.text(0.5, 0.5, "Insufficient spikes for autocorrelogram", ha="center", va="center", transform=ax_c.transAxes)
        ax_c.set_title("C. Spike Autocorrelogram", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel D: Trial-by-Trial Firing Rate Stability
    # -------------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    trial_nums = np.arange(1, dataset.n_trials + 1)
    ax_d.plot(trial_nums, reach_frs, "o-", color="#1f77b4", lw=1.8, ms=4, alpha=0.75, label="Reach FR ([-50, 350] ms)")
    ax_d.plot(trial_nums, base_frs, "s--", color="#7f7f7f", lw=1.4, ms=3, alpha=0.6, label="Baseline Rest FR ([-500, -200] ms)")
    ax_d.set_xlabel("Trial Number (Chronological)", fontsize=10, fontweight="bold")
    ax_d.set_ylabel("Firing Rate (Hz)", fontsize=10, fontweight="bold")
    ax_d.set_title(f"D. Trial-by-Trial Stability across Session (n = {dataset.n_trials} Trials)", fontsize=11, fontweight="bold")
    ax_d.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax_d.set_ylim(bottom=0)
    ax_d.grid(True, linestyle=":", alpha=0.4)

    # Master Figure Super-Title
    fig.suptitle(
        f"Anterior Interposed Nucleus (IntA) | Spike-Sorted Single Unit {unit_id:03d} Quality Diagnostics [{br_tag}]\n"
        f"Condition: {dataset.condition_label} | Monkey: {dataset.monkey}\n"
        f"Total Spikes: {total_spikes:,} | Reach FR: {np.mean(reach_frs):.1f} Hz | Baseline: {np.mean(base_frs):.1f} Hz | ISI Violations (< 1.5 ms): {violation_pct:.2f}%",
        fontsize=13, fontweight="bold", y=0.995,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path.name}")


# =============================================================================
# 8. MAIN PIPELINE & CLI
# =============================================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-Unit IntA Reaching Characterization (Monkey Nike, NPRW Probe)"
    )
    parser.add_argument(
        "--data_dir", "--data-dir", dest="data_dir", type=Path, default=None,
        help="Path to peri_stim checkpoint directory (defaults to PERI_ROOT from config)."
    )
    parser.add_argument(
        "--out_dir", "--out-dir", dest="out_dir", type=Path, default=None,
        help="Directory to save generated IntA figures and statistics (defaults to results/figures/sorted_IntA)."
    )
    parser.add_argument(
        "--units", type=int, nargs="*", default=None,
        help="Specific unit IDs to generate Figure 1 single-unit profile plots for."
    )
    parser.add_argument(
        "--br_idx", "--br-idx", dest="br_idx", type=int, nargs="*", default=None,
        help="Specific Blackrock condition index or indices to evaluate (e.g. --br_idx 1 or --br_idx 18)."
    )
    parser.add_argument(
        "--all_controls", "--all-controls", dest="all_controls", action="store_true", default=False,
        help="Load all control reaches across the entire session to maximize statistical power."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Determine output folder
    if args.out_dir is not None:
        out_root = Path(args.out_dir)
    elif OUT_BASE is not None:
        out_root = OUT_BASE / "figures" / "sorted_IntA"
    else:
        out_root = REPO_ROOT / "results" / "figures" / "sorted_IntA"
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" ANTERIOR INTERPOSED NUCLEUS (IntA) REACHING CHARACTERIZATION")
    print(" Centered on IR Beam Crossing (t = 0 ms) | Monkey Nike (NPRW Probe)")
    print("=" * 80)
    print(f"Output Directory: {out_root}")

    # Determine Blackrock condition filter
    eval_brs = args.br_idx or PROCESS_ONLY
    if eval_brs:
        try:
            eval_brs = sorted(list(set(int(x) for x in eval_brs)))
        except Exception:
            eval_brs = []

    # Attempt to load SortingAnalyzer from checkpoints/spikesorting/NPRW_concat_blanking/analyzer
    analyzer = None
    if si is not None:
        cand_folders = [
            out_root.parents[1] / "checkpoints" / "spikesorting" / "NPRW_concat_blanking" / "analyzer_manual",
            out_root.parents[1] / "checkpoints" / "spikesorting" / "NPRW_concat_blanking" / "analyzer_SLAy",
            out_root.parents[1] / "checkpoints" / "spikesorting" / "NPRW_concat_blanking" / "analyzer",
            out_root.parents[2] / "checkpoints" / "spikesorting" / "NPRW_concat_blanking" / "analyzer_manual",
            out_root.parents[2] / "checkpoints" / "spikesorting" / "NPRW_concat_blanking" / "analyzer_SLAy",
            out_root.parents[2] / "checkpoints" / "spikesorting" / "NPRW_concat_blanking" / "analyzer",
        ]
        analyzer_folder = next((p for p in cand_folders if p.exists()), None)
        if analyzer_folder is not None:
            try:
                analyzer = si.load_sorting_analyzer(analyzer_folder)
                print(f"[IntA Analysis] Loaded SortingAnalyzer from: {analyzer_folder}")
            except Exception as e:
                print(f"[IntA Analysis] Note loading analyzer: {e}")

    # =========================================================================
    # CASE A: SINGLE CONDITION (OR DEFAULT ALL CONTROLS)
    # =========================================================================
    if not eval_brs or len(eval_brs) <= 1:
        dataset = load_real_reaching_dataset(
            peri_dir=args.data_dir,
            process_only=eval_brs,
            all_controls=args.all_controls,
        )

        if dataset is None or len(dataset.unit_ids) == 0:
            print("\n[IntA Analysis] No reaching data available to process.")
            print("[IntA Analysis] Please verify checkpoints in PeriStim_sorted or PeriStim.")
            return

        br_tag = dataset.get_br_condition_tag()
        print(f"\n[IntA Analysis] Active Condition Tag: {br_tag}")

        cond_dir = out_root / br_tag
        cond_dir.mkdir(parents=True, exist_ok=True)

        # 1. Population Modulation Statistics
        print(f"\n[IntA Analysis] Step 1: Quantifying reach modulation fraction and statistics for [{br_tag}]...")
        df_stats = compute_inta_population_modulation_stats(dataset)

        csv_path_cond = cond_dir / f"IntA_unit_modulation_stats_{br_tag}.csv"
        csv_path_root = out_root / f"IntA_unit_modulation_stats_{br_tag}.csv"
        df_stats.to_csv(csv_path_cond, index=False)
        df_stats.to_csv(csv_path_root, index=False)
        df_stats.to_csv(out_root / "IntA_unit_modulation_stats.csv", index=False)
        print(f"  [saved] Statistics exported to: {csv_path_cond.name}")

        phase_counts = df_stats["reach_phase"].value_counts()
        n_tot = len(df_stats)
        print("---------------------------------------------------------------------------")
        print(f"  Total IntA Units Analyzed: {n_tot}")
        print(f"  Condition Evaluated: {dataset.condition_label} [{br_tag}]")
        print("  Reaching Epoch Peaking Breakdown (relative to Peak Speed & Endpoint):")
        for ph in ["Pre-Crossing", "Acceleration", "Deceleration / Pre-Endpoint", "Post-Touch / Endpoint"]:
            cnt = phase_counts.get(ph, 0)
            print(f"  - {ph:28s}: {cnt:3d} ({cnt/max(n_tot,1)*100:5.1f}%)")
        dir_cnt = int(df_stats["is_directional"].sum())
        print(f"  - Directionally Tuned (Target A vs B): {dir_cnt}/{n_tot} ({dir_cnt/max(n_tot,1)*100:.1f}%)")
        print("---------------------------------------------------------------------------")

        # 2. Single-Unit Profiles & Spike Sorting Quality Diagnostics
        print(f"\n[IntA Analysis] Step 2: Generating single-unit profiles and quality diagnostics for [{br_tag}]...")
        single_dir = cond_dir / "single_unit_profiles"
        quality_dir = cond_dir / "quality_diagnostics"
        single_dir.mkdir(parents=True, exist_ok=True)
        quality_dir.mkdir(parents=True, exist_ok=True)

        units_to_plot = args.units if (args.units is not None and len(args.units) > 0) else dataset.unit_ids
        if args.units is None:
            active_uids = df_stats[df_stats["peak_fr_hz"] >= 2.0]["unit_id"].tolist()
            if active_uids:
                units_to_plot = active_uids
            else:
                units_to_plot = dataset.unit_ids[:25]

        for uid in units_to_plot:
            plot_inta_single_unit_profile(dataset, uid, single_dir, br_tag=br_tag)
            out_qual = quality_dir / f"IntA_unit_{uid:03d}_{br_tag}_quality.png"
            plot_inta_unit_quality_diagnostic(dataset, uid, out_qual, analyzer=analyzer, br_tag=br_tag)

        # 3. Population Firing Rate Heatmap Figure (-500 to 500 ms, RdBu_r, Baseline -800 to -650 ms)
        print(f"\n[IntA Analysis] Step 3: Generating population firing rate heatmap for [{br_tag}] (-500 to 500 ms, RdBu_r)...")
        heatmap_png_cond = cond_dir / f"figure_IntA_population_firing_rate_heatmap_{br_tag}.png"
        heatmap_svg_cond = cond_dir / f"figure_IntA_population_firing_rate_heatmap_{br_tag}.svg"
        plot_inta_population_firing_rate_heatmap(dataset, heatmap_png_cond, z_score=True)
        plot_inta_population_firing_rate_heatmap(dataset, heatmap_svg_cond, z_score=True)

        heatmap_png_root = out_root / f"figure_IntA_population_firing_rate_heatmap_{br_tag}.png"
        heatmap_svg_root = out_root / f"figure_IntA_population_firing_rate_heatmap_{br_tag}.svg"
        try:
            shutil.copy2(heatmap_png_cond, heatmap_png_root)
            shutil.copy2(heatmap_svg_cond, heatmap_svg_root)
            shutil.copy2(heatmap_png_cond, out_root / "figure_IntA_population_firing_rate_heatmap.png")
            shutil.copy2(heatmap_svg_cond, out_root / "figure_IntA_population_firing_rate_heatmap.svg")
        except Exception:
            pass

    # =========================================================================
    # CASE B: MULTIPLE CONDITIONS (INDIVIDUAL ANALYSES + COMPARISON OVERLAY)
    # =========================================================================
    else:
        print(f"\n[IntA Analysis] Multiple conditions detected in process_only: {eval_brs}")
        print("[IntA Analysis] Running individual isolated analyses for EACH condition followed by multi-condition comparison...")

        datasets_by_br: Dict[int, IntAReachingDataset] = {}
        all_active_uids: set = set()

        for b_idx in eval_brs:
            print("\n" + "=" * 75)
            print(f"[IntA Analysis] >>> Loading and analyzing Condition BR_{b_idx:03d} individually...")
            print("=" * 75)

            ds_cond = load_real_reaching_dataset(
                peri_dir=args.data_dir,
                process_only=[b_idx],
                all_controls=False,
            )

            if ds_cond is None or len(ds_cond.unit_ids) == 0 or ds_cond.n_trials == 0:
                print(f"[IntA Analysis] Skipping BR_{b_idx:03d}: no valid trials or units found.")
                continue

            datasets_by_br[b_idx] = ds_cond
            br_tag = ds_cond.get_br_condition_tag()
            cond_dir = out_root / br_tag
            cond_dir.mkdir(parents=True, exist_ok=True)

            # 1. Condition Statistics
            print(f"\n[IntA Analysis] Step 1: Quantifying reach modulation fraction and statistics for [{br_tag}]...")
            df_stats = compute_inta_population_modulation_stats(ds_cond)
            csv_path_cond = cond_dir / f"IntA_unit_modulation_stats_{br_tag}.csv"
            df_stats.to_csv(csv_path_cond, index=False)
            print(f"  [saved] Statistics exported to: {csv_path_cond.name}")

            active_in_cond = df_stats[df_stats["peak_fr_hz"] >= 2.0]["unit_id"].tolist()
            all_active_uids.update(active_in_cond)

            # 2. Condition Single-Unit Profiles & Quality Diagnostics
            print(f"\n[IntA Analysis] Step 2: Generating single-unit profiles and quality diagnostics for [{br_tag}]...")
            single_dir = cond_dir / "single_unit_profiles"
            quality_dir = cond_dir / "quality_diagnostics"
            single_dir.mkdir(parents=True, exist_ok=True)
            quality_dir.mkdir(parents=True, exist_ok=True)

            units_to_plot = args.units if (args.units is not None and len(args.units) > 0) else ds_cond.unit_ids
            if args.units is None:
                units_to_plot = active_in_cond if active_in_cond else ds_cond.unit_ids[:25]

            for uid in units_to_plot:
                plot_inta_single_unit_profile(ds_cond, uid, single_dir, br_tag=br_tag)
                out_qual = quality_dir / f"IntA_unit_{uid:03d}_{br_tag}_quality.png"
                plot_inta_unit_quality_diagnostic(ds_cond, uid, out_qual, analyzer=analyzer, br_tag=br_tag)

            # 3. Condition Heatmap (sorted by Target A)
            print(f"\n[IntA Analysis] Step 3: Generating population firing rate heatmap for [{br_tag}] (-500 to 500 ms, RdBu_r)...")
            heatmap_png_cond = cond_dir / f"figure_IntA_population_firing_rate_heatmap_{br_tag}.png"
            heatmap_svg_cond = cond_dir / f"figure_IntA_population_firing_rate_heatmap_{br_tag}.svg"
            plot_inta_population_firing_rate_heatmap(ds_cond, heatmap_png_cond, z_score=True)
            plot_inta_population_firing_rate_heatmap(ds_cond, heatmap_svg_cond, z_score=True)

        # 4. Multi-Condition Unified Comparison Figure Generation
        if len(datasets_by_br) > 1:
            print("\n" + "=" * 80)
            print(f"[IntA Analysis] >>> Generating Multi-Condition Comparison Across {len(datasets_by_br)} Conditions...")
            print("=" * 80)
            multi_comp_dir = out_root / "multi_condition_comparison"
            multi_comp_dir.mkdir(parents=True, exist_ok=True)
            cond_colors = get_condition_palette(list(datasets_by_br.keys()), datasets_by_br)

            all_uids = set()
            for ds in datasets_by_br.values():
                all_uids.update(ds.unit_ids)
            all_uids_sorted = sorted(list(all_uids))

            if args.units is not None and len(args.units) > 0:
                multi_units_to_plot = [u for u in args.units if u in all_uids]
            elif all_active_uids:
                multi_units_to_plot = sorted(list(all_active_uids))
            else:
                multi_units_to_plot = all_uids_sorted[:25]

            print(f"[IntA Analysis] Generating multi-condition comparison profiles for {len(multi_units_to_plot)} units into: {multi_comp_dir.name}")
            for uid in multi_units_to_plot:
                plot_inta_multi_condition_comparison(datasets_by_br, uid, multi_comp_dir, cond_colors)

    # Clean up any previously generated gallery or population summary figures
    for old_name in [
        "figure_all_units_gallery.png",
        "figure2_IntA_peak_firing_time_diagram.png",
        "figure3_IntA_kinematic_encoding_summary.png",
        "figure4_IntA_population_patterns_and_dynamics.png",
    ]:
        old_f = out_root / old_name
        if old_f.exists():
            try:
                old_f.unlink()
                print(f"  [removed] {old_name}")
            except Exception:
                pass

    # Suppressed per user request: focusing exclusively on individual unit profiles per target
    # (Functions remain defined in script for future reference)

    print("\n" + "=" * 80)
    print(" INTA REACHING ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f" View figures in: {out_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
