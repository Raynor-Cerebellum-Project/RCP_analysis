#!/usr/bin/env python3
"""
compare_sessions_ua.py

Cross-session UA (Utah Array) comparison.
Generates:
1. Per-condition peri-stimulus plots (kinematics + UA heatmaps by region + raw HPF traces with spikes)
2. Cross-session comparison heatmaps (mean per window, change from baseline, stim-control difference)
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml
import RCP_analysis.python.functions.config_loading as cfg

# Try to import spikeinterface for raw trace loading
try:
    import spikeinterface as si
    HAS_SI = True
except ImportError:
    HAS_SI = False
    print("[warn] spikeinterface not available - raw trace plots will be disabled")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
REGION_ORDER = ["SMA", "PMd", "M1i", "M1s"]
REGION_COLORS = {
    "SMA": "#1f77b4",   # blue
    "PMd": "#ff7f0e",   # orange
    "M1i": "#2ca02c",   # green
    "M1s": "#d62728",   # red
}

DEFAULT_TIME_WINDOWS = [
    (-600, -425, "win1_baseline"),
    (-400, -225, "win2_early"),
    (-200, -25, "win3_late"),
    (0, 175, "win4_post"),
]

# Plot settings
COLORMAP = "RdBu_r"
VMIN_UA, VMAX_UA = -100.0, 100.0
VMIN_DIFF, VMAX_DIFF = -100.0, 100.0
KINEMATICS_YLIM = (-4, 4)

# Raw trace settings
TRACE_WINDOW_MS = (-600, 400)  # Window for raw traces relative to movement onset
TRACE_YLIM_UV = (-100, 100)    # µV range for raw traces


@dataclass
class SessionSpec:
    """One session with control and stim condition indices."""
    session: str
    control_br: int
    stim_br: int


def get_default_electrode_mapping_csv() -> Path:
    monkey = str(cfg.PARAMS.monkey)

    if monkey not in ("Nike", "Ada"):
        raise ValueError(
            f"Unknown PARAMS.monkey={monkey!r}; expected 'Nike' or 'Ada'"
        )

    return (
        Path(__file__).resolve().parents[3]
        / "config"
        / f"electrode_port_mapping_{monkey}.csv"
    )


@dataclass
class ComparisonConfig:
    """Configuration for comparison."""
    data_root: Path = field(default_factory=lambda: Path("."))
    electrode_mapping_csv: Path = field(default_factory=get_default_electrode_mapping_csv)
    session_specs: list[SessionSpec] = field(default_factory=list)
    target: str = 'A'
    time_windows: list[tuple[float, float, str]] = field(default_factory=lambda: DEFAULT_TIME_WINDOWS.copy())
    output_dir: Path = field(default_factory=lambda: Path("comparison_results"))
    min_spike_count: int = 1  


def load_config(config_path: Path) -> ComparisonConfig:
    """Load configuration from YAML."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    
    cfg = ComparisonConfig()
    
    if 'data_root' in raw:
        cfg.data_root = Path(raw['data_root'])
    
    if 'electrode_mapping_csv' in raw:
        csv_path = Path(raw['electrode_mapping_csv'])
        if not csv_path.is_absolute() and not csv_path.exists():
            config_dir = config_path.parent
            candidate = config_dir / csv_path
            if candidate.exists():
                csv_path = candidate
            else:
                script_dir = Path(__file__).parent
                candidate = script_dir / csv_path
                if candidate.exists():
                    csv_path = candidate
        cfg.electrode_mapping_csv = csv_path
    
    if 'session_specs' in raw:
        for sess_name, spec in raw['session_specs'].items():
            cfg.session_specs.append(SessionSpec(
                session=sess_name,
                control_br=int(spec['control']),
                stim_br=int(spec['stim']),
            ))
    
    if 'target' in raw:
        cfg.target = raw['target']
    if 'output_dir' in raw:
        cfg.output_dir = Path(raw['output_dir'])
    if 'time_windows' in raw:
        cfg.time_windows = [(w[0], w[1], w[2]) for w in raw['time_windows']]

    if 'min_spike_count' in raw:
        cfg.min_spike_count = int(raw['min_spike_count'])
    
    return cfg


# -----------------------------------------------------------------------------
# Electrode Mapping
# -----------------------------------------------------------------------------
@dataclass
class ElectrodeMapping:
    """Electrode mapping information."""
    df: pd.DataFrame
    nsp_to_elec: dict[int, int] = field(default_factory=dict)
    elec_to_nsp: dict[int, int] = field(default_factory=dict)
    elec_to_region: dict[int, str] = field(default_factory=dict)
    region_to_elecs: dict[str, list[int]] = field(default_factory=dict)
    region_grids: dict[str, np.ndarray] = field(default_factory=dict)
    
    def __post_init__(self):
        self._build_mappings()
    
    def _build_mappings(self):
        """Build all mapping dictionaries from DataFrame."""
        for region in REGION_ORDER:
            self.region_to_elecs[region] = []
            self.region_grids[region] = np.zeros((8, 8), dtype=int)
        
        for _, row in self.df.iterrows():
            elec_id = int(row['ElectrodeID'])
            nsp_id = int(row['NSP_ID'])
            region = str(row['Array']).strip()
            grid_row = int(row['GridRow'])
            grid_col = int(row['GridCol'])
            
            self.nsp_to_elec[nsp_id] = elec_id
            self.elec_to_nsp[elec_id] = nsp_id
            self.elec_to_region[elec_id] = region
            
            if region in self.region_to_elecs:
                self.region_to_elecs[region].append(elec_id)
            
            if region in self.region_grids:
                self.region_grids[region][grid_row, grid_col] = elec_id


def load_electrode_mapping(csv_path: Path) -> ElectrodeMapping:
    """Load electrode mapping from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Electrode mapping CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    required_cols = ['Array', 'Port', 'GridRow', 'GridCol', 'ElectrodeID', 'NSP_ID']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in electrode mapping: {missing}")
    
    return ElectrodeMapping(df=df)


# -----------------------------------------------------------------------------
# Data Loading - PeriStim NPZ (firing rates)
# -----------------------------------------------------------------------------
@dataclass
class UAData:
    """UA (Utah Array) data for one condition."""
    session: str
    br_idx: int
    condition_type: str
    
    ua_med: np.ndarray
    ua_var: np.ndarray
    ua_rel_t: np.ndarray
    
    beh_rel_t: Optional[np.ndarray] = None
    middle_x_med: Optional[np.ndarray] = None
    middle_y_med: Optional[np.ndarray] = None
    middle_x_std: Optional[np.ndarray] = None
    middle_y_std: Optional[np.ndarray] = None
    
    n_trials: int = 0
    n_channels: int = 0
    ua_ids_1based: Optional[np.ndarray] = None
    ua_port: str = 'A'
    
    # For raw trace plotting
    session_folder: Optional[Path] = None
    
    @property
    def label(self) -> str:
        return f"{self.session}_BR{self.br_idx:03d}_{self.condition_type}"


@dataclass
class RawTraceData:
    """Raw trace data for HPF plotting."""
    recording: object  # SpikeInterface recording
    fs: float
    peak_ms_dict: dict  # channel_idx -> spike times (ms)
    target_ch_elecs: np.ndarray  # electrode IDs per channel
    target_ch_regions: np.ndarray  # region indices per channel
    region_names: list[str]
    rec_start_ms: float = 0.0
    movement_times_ms: Optional[np.ndarray] = None  # movement onset times in LOCAL recording time
    ridx_to_meta: dict = field(default_factory=dict)  # recording index -> metadata


def find_session_folder(data_root: Path, session: str) -> Optional[Path]:
    """Find session folder by pattern *_SESSION."""
    pattern = f"*_{session}"
    matches = list(data_root.glob(pattern))
    
    if not matches:
        all_dirs = [d for d in data_root.iterdir() if d.is_dir() and session in d.name]
        matches = all_dirs
    
    if not matches:
        print(f"[warn] No folder found for {session} in {data_root}")
        return None
    
    if len(matches) > 1:
        matches = sorted(matches)
    
    return matches[-1] if matches else None


def find_npz_file(
    data_root: Path,
    session: str,
    br_idx: int,
    target: str,
) -> Optional[Path]:
    """Find the PeriStim NPZ file for a specific session/BR."""
    session_folder = find_session_folder(data_root, session)
    if session_folder is None:
        return None
    
    peri_root = session_folder / "results" / "checkpoints" / "PeriStim"
    
    if not peri_root.exists():
        print(f"[warn] PeriStim not found: {peri_root}")
        return None
    
    for subdir in ["control_reaches", "stim_reaches", "continuous_stim", "at_rest"]:
        for search_dir in [peri_root / subdir / f"target_{target}", peri_root / subdir]:
            if not search_dir.exists():
                continue
            
            patterns = [
                f"peristim__*{session}*BR_{br_idx:03d}*.npz",
                f"peristim__*BR_{br_idx:03d}*.npz",
                f"*BR_{br_idx:03d}*.npz",
                f"*BR_{br_idx}_*.npz",
            ]
            
            for pattern in patterns:
                matches = list(search_dir.glob(pattern))
                if matches:
                    return matches[0]
    
    print(f"[warn] No PeriStim NPZ for {session} BR{br_idx}")
    return None


def find_aligned_file(session_folder: Path, br_idx: int) -> Optional[Path]:
    """Find the Aligned NPZ file for a given BR index."""
    aligned_dir = session_folder / "results" / "checkpoints" / "Aligned"
    
    if not aligned_dir.exists():
        print(f"    [warn] Aligned directory not found: {aligned_dir}")
        return None
    
    pat = f"*__BR_{br_idx:03d}.npz"
    
    search_dirs = [
        aligned_dir / "stim_reaches",
        aligned_dir / "control_reaches",
        aligned_dir / "at_rest",
        aligned_dir / "continuous_stim",
        aligned_dir,
    ]
    
    for d in search_dirs:
        if d.exists():
            cands = sorted(d.glob(pat))
            if cands:
                return cands[-1]
    
    print(f"    [warn] No Aligned NPZ matching {pat}")
    return None


def find_pp_folder(session_folder: Path, br_idx: int) -> Optional[Path]:
    """Find the preprocessed SpikeInterface folder for UA data."""
    ckpt_dir = session_folder / "results" / "checkpoints" / "UA"
    
    if not ckpt_dir.exists():
        print(f"    [warn] UA checkpoint directory not found: {ckpt_dir}")
        return None
    
    all_folders = sorted(ckpt_dir.glob("pp_*"))
    
    # For Blackrock, the condition ID is in the folder name
    target = f"_{br_idx:03d}__NS6"
    for f in all_folders:
        if target in f.name:
            return f
    
    # Fallback: try without leading zeros
    target_alt = f"_{br_idx}__NS6"
    for f in all_folders:
        if target_alt in f.name:
            return f
    
    print(f"    [warn] No pp_* folder matching BR{br_idx}")
    return None


def _extract_middle_finger(
    beh_pos_med: np.ndarray,
    beh_pos_segs: np.ndarray,
    beh_names: list[str],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract middle finger X and Y from behavior data."""
    if beh_pos_med is None or beh_names is None or len(beh_names) == 0:
        return None, None, None, None
    
    beh_names_lower = [str(n).lower() for n in beh_names]
    
    middle_x_idx = None
    middle_y_idx = None
    
    for i, name in enumerate(beh_names_lower):
        if 'middle' in name:
            if '_x' in name or name.endswith('x'):
                middle_x_idx = i
            elif '_y' in name or name.endswith('y'):
                middle_y_idx = i
    
    if middle_x_idx is None and middle_y_idx is None:
        return None, None, None, None
    
    middle_x_med = beh_pos_med[middle_x_idx] if middle_x_idx is not None else None
    middle_y_med = beh_pos_med[middle_y_idx] if middle_y_idx is not None else None
    
    middle_x_std = None
    middle_y_std = None
    
    if beh_pos_segs is not None and beh_pos_segs.size > 0:
        if middle_x_idx is not None and middle_x_idx < beh_pos_segs.shape[1]:
            middle_x_std = np.nanstd(beh_pos_segs[:, middle_x_idx, :], axis=0)
        if middle_y_idx is not None and middle_y_idx < beh_pos_segs.shape[1]:
            middle_y_std = np.nanstd(beh_pos_segs[:, middle_y_idx, :], axis=0)
    
    return middle_x_med, middle_y_med, middle_x_std, middle_y_std


def load_ua_data(npz_path: Path, condition_type: str, session_folder: Path = None) -> Optional[UAData]:
    """Load UA firing rate data from PeriStim NPZ file."""
    try:
        npz = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"[error] Load failed {npz_path}: {e}")
        return None
    
    ua_med = npz.get("UA_med", None)
    ua_rel_t = npz.get("UA_rel_t", None)
    
    if ua_med is None or ua_rel_t is None:
        print(f"[warn] No UA_med/UA_rel_t in {npz_path.name}")
        return None
    
    if ua_med.size == 0:
        print(f"[warn] UA_med is empty in {npz_path.name}")
        return None
    
    ua_var = npz.get("UA_var", np.zeros_like(ua_med))
    
    sess = str(npz.get("sess", "unknown"))
    br_idx = int(npz.get("br_idx", 0))
    n_trials = int(npz.get("n_trials", 0))
    ua_ids_1based = npz.get("ua_ids_1based", None)
    ua_port = str(npz.get("ua_port", "A"))
    
    beh_rel_t = npz.get("beh_rel_t", None)
    
    beh_cam0_pos_med = npz.get("beh_cam0_pos_med", None)
    beh_cam0_segs = npz.get("beh_cam0_segs", None)
    beh_cam0_names = npz.get("beh_cam0_names", None)
    if beh_cam0_names is not None and isinstance(beh_cam0_names, np.ndarray):
        beh_cam0_names = beh_cam0_names.tolist()
    
    beh_cam1_pos_med = npz.get("beh_cam1_pos_med", None)
    beh_cam1_segs = npz.get("beh_cam1_segs", None)
    beh_cam1_names = npz.get("beh_cam1_names", None)
    if beh_cam1_names is not None and isinstance(beh_cam1_names, np.ndarray):
        beh_cam1_names = beh_cam1_names.tolist()
    
    middle_x_med, middle_y_med, middle_x_std, middle_y_std = None, None, None, None
    
    if beh_cam0_pos_med is not None and beh_cam0_names:
        middle_x_med, middle_y_med, middle_x_std, middle_y_std = _extract_middle_finger(
            beh_cam0_pos_med, beh_cam0_segs, beh_cam0_names
        )
    
    if middle_x_med is None and beh_cam1_pos_med is not None and beh_cam1_names:
        middle_x_med, middle_y_med, middle_x_std, middle_y_std = _extract_middle_finger(
            beh_cam1_pos_med, beh_cam1_segs, beh_cam1_names
        )
    
    return UAData(
        session=sess,
        br_idx=br_idx,
        condition_type=condition_type,
        ua_med=ua_med,
        ua_var=ua_var,
        ua_rel_t=ua_rel_t,
        beh_rel_t=beh_rel_t,
        middle_x_med=middle_x_med,
        middle_y_med=middle_y_med,
        middle_x_std=middle_x_std,
        middle_y_std=middle_y_std,
        n_trials=n_trials,
        n_channels=ua_med.shape[0],
        ua_ids_1based=ua_ids_1based,
        ua_port=ua_port,
        session_folder=session_folder,
    )


def load_raw_trace_data(session_folder: Path, br_idx: int, debug: bool = True) -> Optional[RawTraceData]:
    """Load raw trace data from Aligned NPZ and SpikeInterface recording."""
    if not HAS_SI:
        print("    [warn] SpikeInterface not available, skipping raw traces")
        return None
    
    # Find aligned file
    aligned_path = find_aligned_file(session_folder, br_idx)
    if aligned_path is None:
        return None
    
    # Find preprocessed folder
    pp_folder = find_pp_folder(session_folder, br_idx)
    if pp_folder is None:
        return None
    
    # Load aligned NPZ
    try:
        z = np.load(aligned_path, allow_pickle=True)
    except Exception as e:
        print(f"    [error] Failed to load aligned NPZ: {e}")
        return None
    
    if debug:
        print(f"    [debug] Aligned NPZ fields: {list(z.files)[:15]}...")
    
    # Load SpikeInterface recording
    try:
        rec = si.load(str(pp_folder))
        fs = float(rec.get_sampling_frequency())
    except Exception as e:
        print(f"    [error] Failed to load SI recording: {e}")
        return None
    
    # Extract spike times
    pk_obj = z.get("ua_peak_ms_dedup", z.get("ua_peak_ms", None))
    if pk_obj is not None:
        if isinstance(pk_obj, np.ndarray) and pk_obj.dtype == object:
            peak_ms_dict = pk_obj.reshape(-1)[0] if pk_obj.size > 0 else {}
        else:
            peak_ms_dict = pk_obj
    else:
        peak_ms_dict = {}
    if peak_ms_dict is None:
        peak_ms_dict = {}
    
    # Extract channel info
    target_ch_elecs = z.get("ua_elec", np.array([]))
    if isinstance(target_ch_elecs, np.ndarray) and target_ch_elecs.ndim == 0:
        target_ch_elecs = target_ch_elecs.item()
    target_ch_elecs = np.asarray(target_ch_elecs, int).ravel()
    
    target_ch_regions = z.get("ua_region", np.array([]))
    if isinstance(target_ch_regions, np.ndarray) and target_ch_regions.ndim == 0:
        target_ch_regions = target_ch_regions.item()
    target_ch_regions = np.asarray(target_ch_regions, int).ravel()
    
    region_names = z.get("ua_region_names", [])
    if isinstance(region_names, np.ndarray):
        region_names = region_names.tolist()
    
    # --- Find movement onset times ---
    # Try multiple possible field names
    movement_times_ms = None
    
    # List of possible field names for movement/event times
    event_field_candidates = [
        "event_ms",           # Primary expected field
        "movement_onset_ms",  # Alternative name
        "reach_onset_ms",     # Alternative name
        "trial_onset_ms",     # Alternative name
        "anchor_ms",          # The alignment anchor times
        "stim_ms",            # Stimulation times (if no movement times available)
    ]
    
    if debug:
        print(f"    [debug] Looking for movement times in fields...")
    
    for field_name in event_field_candidates:
        if field_name in z.files:
            candidate = z[field_name]
            if candidate is not None:
                candidate = np.asarray(candidate, float).ravel()
                if candidate.size > 0:
                    if debug:
                        print(f"    [debug] Found '{field_name}': {candidate.size} events, range [{candidate.min():.1f}, {candidate.max():.1f}] ms")
                    movement_times_ms = candidate
                    break
    
    # If still no movement times, try to get them from trial_labels or other sources
    if movement_times_ms is None or movement_times_ms.size == 0:
        # Check if there's align_meta with timing info
        if "align_meta_raw" in z.files or "align_meta" in z.files:
            meta_key = "align_meta_raw" if "align_meta_raw" in z.files else "align_meta"
            try:
                raw_meta = z[meta_key]
                if raw_meta.ndim == 0:
                    raw_meta = raw_meta.item()
                if isinstance(raw_meta, str):
                    meta = json.loads(raw_meta)
                elif isinstance(raw_meta, dict):
                    meta = raw_meta
                else:
                    meta = {}
                
                # Check for timing info in metadata
                if 'anchor_times_ms' in meta:
                    movement_times_ms = np.asarray(meta['anchor_times_ms'], float).ravel()
                    if debug:
                        print(f"    [debug] Found anchor_times_ms in meta: {movement_times_ms.size} events")
            except Exception as e:
                if debug:
                    print(f"    [debug] Failed to parse align_meta: {e}")
    
    # Fallback: use the time at t=0 relative to the recording
    # This assumes the aligned data is centered around movement onset
    if movement_times_ms is None or movement_times_ms.size == 0:
        # Get recording duration and create dummy event at center or use trial count
        n_trials = int(z.get("n_beh", z.get("n_trials", 1)))
        if n_trials > 0:
            # Try to infer from the peri-stimulus window
            # The UA_rel_t typically spans something like [-600, 600] ms
            # Movement onset is at t=0 in that frame
            # We need the ABSOLUTE times in the recording
            
            # Check if there's info about when trials occurred
            if "trial_start_ms" in z.files:
                trial_starts = np.asarray(z["trial_start_ms"], float).ravel()
                if trial_starts.size > 0:
                    movement_times_ms = trial_starts
                    if debug:
                        print(f"    [debug] Using trial_start_ms: {movement_times_ms.size} events")
        
        if movement_times_ms is None or movement_times_ms.size == 0:
            print(f"    [warn] Could not find movement times - trace plots will be disabled")
            # Create placeholder with recording midpoint for at least 1 trial
            rec_dur_ms = rec.get_num_frames() / fs * 1000.0
            movement_times_ms = np.array([rec_dur_ms / 2])
            if debug:
                print(f"    [debug] Using recording midpoint as fallback: {movement_times_ms[0]:.1f} ms")
    
    # Build ridx_to_meta mapping
    ridx_to_meta = {}
    ch_ids = rec.get_channel_ids()
    
    for ridx, cid in enumerate(ch_ids):
        m = re.search(r'UAe(\d+)', str(cid))
        eid = int(m.group(1)) if m else -1
        
        if eid < 0 and ridx < len(target_ch_elecs):
            eid = int(target_ch_elecs[ridx])
        
        oid = ridx
        if eid > 0 and target_ch_elecs.size > 0:
            matches = np.where(target_ch_elecs == eid)[0]
            if matches.size > 0:
                oid = int(matches[0])
        
        ridx_to_meta[ridx] = {"eid": eid, "oid": oid, "cid": cid}
    
    print(f"    [info] Loaded raw traces: {rec.get_num_channels()} ch, fs={fs:.0f} Hz")
    print(f"    [info] Movement events: {len(movement_times_ms) if movement_times_ms is not None else 0}")
    
    return RawTraceData(
        recording=rec,
        fs=fs,
        peak_ms_dict=peak_ms_dict,
        target_ch_elecs=target_ch_elecs,
        target_ch_regions=target_ch_regions,
        region_names=list(region_names) if region_names else [],
        rec_start_ms=0.0,
        movement_times_ms=movement_times_ms,
        ridx_to_meta=ridx_to_meta,
    )


def extract_trace_segment(
    rec,
    ch_row: int,
    center_ms: float,
    win_ms: tuple[float, float],
    fs: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract one trace window (in µV) for a channel around an event."""
    ch_id = rec.get_channel_ids()[ch_row]
    s0 = int(round(center_ms / 1000.0 * fs))
    i0 = int(s0 + round(win_ms[0] / 1000.0 * fs))
    i1 = int(s0 + round(win_ms[1] / 1000.0 * fs))
    
    if i0 < 0 or i1 > rec.get_num_frames() or i1 <= i0:
        return None, None
    
    try:
        y = rec.get_traces(start_frame=i0, end_frame=i1,
                           channel_ids=[ch_id], return_in_uV=True).squeeze()
        t = (np.arange(i0, i1) - s0) / fs * 1000.0
        return t, y
    except Exception as e:
        print(f"    [warn] Failed to extract trace: {e}")
        return None, None


# -----------------------------------------------------------------------------
# Channel Ordering
# -----------------------------------------------------------------------------
# def build_channel_order(
#     ua_data: UAData,
#     elec_mapping: ElectrodeMapping,
#     time_windows: list[tuple[float, float, str]],
# ) -> dict[str, list[int]]:
#     """Build channel ordering by region, sorted by mean activity."""
#     n_channels = ua_data.n_channels
#     ua_port = ua_data.ua_port.upper()
    
#     channel_mean_activity = np.zeros(n_channels)
    
#     for ch_idx in range(n_channels):
#         values = []
#         for win_start, win_end, _ in time_windows:
#             mask = (ua_data.ua_rel_t >= win_start) & (ua_data.ua_rel_t <= win_end)
#             if mask.any():
#                 values.append(np.nanmean(ua_data.ua_med[ch_idx, mask]))
#         channel_mean_activity[ch_idx] = np.nanmean(values) if values else 0
    
#     nsp_offset = 0 if ua_port == 'A' else 128
    
#     region_channels: dict[str, list[tuple[int, float]]] = {r: [] for r in REGION_ORDER}
    
#     for ch_idx in range(n_channels):
#         nsp_id = ch_idx + 1 + nsp_offset
#         elec_id = elec_mapping.nsp_to_elec.get(nsp_id, -1)
        
#         if elec_id > 0:
#             region = elec_mapping.elec_to_region.get(elec_id, None)
#             if region in region_channels:
#                 region_channels[region].append((ch_idx, channel_mean_activity[ch_idx]))
    
#     result = {}
#     for region in REGION_ORDER:
#         sorted_channels = sorted(region_channels[region], key=lambda x: -x[1])
#         result[region] = [ch_idx for ch_idx, _ in sorted_channels]
    
#     return result

def build_channel_order(
    ua_data: UAData,
    elec_mapping: ElectrodeMapping,
    time_windows: list[tuple[float, float, str]],
    raw_trace_data: Optional[RawTraceData] = None,
    min_spike_count: int = 1,
) -> dict[str, list[int]]:
    """Build channel ordering by region, sorted by mean activity.
    
    Parameters
    ----------
    ua_data : UAData
        UA firing rate data
    elec_mapping : ElectrodeMapping
        Electrode mapping information
    time_windows : list
        Time windows for computing mean activity
    raw_trace_data : RawTraceData, optional
        Raw trace data with spike times (for filtering by spike count)
    min_spike_count : int
        Minimum number of spikes required to include a channel (default: 1)
    
    Returns
    -------
    dict[str, list[int]]
        Channel indices per region, sorted by mean activity
    """
    n_channels = ua_data.n_channels
    ua_port = ua_data.ua_port.upper()
    
    channel_mean_activity = np.zeros(n_channels)
    
    for ch_idx in range(n_channels):
        values = []
        for win_start, win_end, _ in time_windows:
            mask = (ua_data.ua_rel_t >= win_start) & (ua_data.ua_rel_t <= win_end)
            if mask.any():
                values.append(np.nanmean(ua_data.ua_med[ch_idx, mask]))
        channel_mean_activity[ch_idx] = np.nanmean(values) if values else 0
    
    nsp_offset = 0 if ua_port == 'A' else 128
    
    # Build set of channels with sufficient spikes
    channels_with_spikes = set(range(n_channels))  # Default: include all
    
    if raw_trace_data is not None and raw_trace_data.peak_ms_dict:
        channels_with_spikes = set()
        
        for ch_idx in range(n_channels):
            meta = raw_trace_data.ridx_to_meta.get(ch_idx, {"eid": ch_idx, "oid": ch_idx})
            oid = meta["oid"]
            eid = meta["eid"]
            
            # Get spike count for this channel
            ch_peaks = np.asarray(raw_trace_data.peak_ms_dict.get(oid, []), float).ravel()
            if ch_peaks.size == 0 and eid > 0:
                ch_peaks = np.asarray(raw_trace_data.peak_ms_dict.get(eid, []), float).ravel()
            
            spike_count = ch_peaks.size
            
            if spike_count >= min_spike_count:
                channels_with_spikes.add(ch_idx)
        
        n_excluded = n_channels - len(channels_with_spikes)
        if n_excluded > 0:
            print(f"    [filter] Excluding {n_excluded} channels with < {min_spike_count} spikes")
    
    region_channels: dict[str, list[tuple[int, float]]] = {r: [] for r in REGION_ORDER}
    
    for ch_idx in range(n_channels):
        # Skip channels without sufficient spikes
        if ch_idx not in channels_with_spikes:
            continue
        
        nsp_id = ch_idx + 1 + nsp_offset
        elec_id = elec_mapping.nsp_to_elec.get(nsp_id, -1)
        
        if elec_id > 0:
            region = elec_mapping.elec_to_region.get(elec_id, None)
            if region in region_channels:
                region_channels[region].append((ch_idx, channel_mean_activity[ch_idx]))
    
    result = {}
    for region in REGION_ORDER:
        sorted_channels = sorted(region_channels[region], key=lambda x: -x[1])
        result[region] = [ch_idx for ch_idx, _ in sorted_channels]
    
    return result


def get_flat_channel_order(channel_order: dict[str, list[int]]) -> tuple[list[int], list[str], list[int]]:
    """Flatten channel order into single list with region boundaries."""
    flat_order = []
    region_labels = []
    region_boundaries = [0]
    
    for region in REGION_ORDER:
        channels = channel_order.get(region, [])
        flat_order.extend(channels)
        region_labels.extend([region] * len(channels))
        region_boundaries.append(len(flat_order))
    
    return flat_order, region_labels, region_boundaries[:-1]


# -----------------------------------------------------------------------------
# Window Computations
# -----------------------------------------------------------------------------
def compute_window_values(
    ua_med: np.ndarray,
    ua_rel_t: np.ndarray,
    time_windows: list[tuple[float, float, str]],
) -> np.ndarray:
    """Compute mean value per channel per window."""
    n_channels = ua_med.shape[0]
    n_windows = len(time_windows)
    
    result = np.full((n_channels, n_windows), np.nan)
    
    for win_idx, (win_start, win_end, _) in enumerate(time_windows):
        mask = (ua_rel_t >= win_start) & (ua_rel_t <= win_end)
        if mask.any():
            result[:, win_idx] = np.nanmean(ua_med[:, mask], axis=1)
    
    return result


def compute_change_from_baseline(window_values: np.ndarray) -> np.ndarray:
    """Compute change from baseline (win1)."""
    baseline = window_values[:, 0:1]
    return window_values[:, 1:] - baseline


# -----------------------------------------------------------------------------
# Plotting: Per-Condition Peri-Stimulus
# -----------------------------------------------------------------------------
# def plot_peristim_figure(
#     ua_data: UAData,
#     elec_mapping: ElectrodeMapping,
#     channel_order: dict[str, list[int]],
#     raw_trace_data: Optional[RawTraceData],
#     out_path: Path,
#     title: str = "",
#     vmin: float = VMIN_UA,
#     vmax: float = VMAX_UA,
#     trace_window_ms: tuple[float, float] = TRACE_WINDOW_MS,
#     trace_ylim_uv: tuple[float, float] = TRACE_YLIM_UV,
#     trial_idx: int = 0,
# ):
#     """
#     Plot peri-stimulus figure with:
#     - Kinematics (X and Y on same panel)
#     - UA heatmaps by region (SMA, PMd, M1i, M1s)
#     - Raw HPF traces with spike markers (top, middle, bottom channel per region)
#     - Electrode layout (4 grids at bottom)
#     """
#     flat_order, region_labels, region_boundaries = get_flat_channel_order(channel_order)
#     n_channels_total = len(flat_order)
    
#     if n_channels_total == 0:
#         print(f"[warn] No channels to plot for {out_path.name}")
#         return
    
#     # Reorder UA data
#     ua_ordered = ua_data.ua_med[flat_order, :]
    
#     # Check kinematics
#     has_kinematics = ua_data.middle_x_med is not None or ua_data.middle_y_med is not None
    
#     # Check if we can plot raw traces
#     can_plot_traces = (
#         raw_trace_data is not None and 
#         raw_trace_data.movement_times_ms is not None and 
#         len(raw_trace_data.movement_times_ms) > trial_idx
#     )
    
#     n_regions = len(REGION_ORDER)
#     region_sizes = [len(channel_order.get(r, [])) for r in REGION_ORDER]
    
#     # Calculate height ratios for MAIN sections
#     kin_height = 2.5
#     heatmap_section_height = sum([max(size * 0.06, 1.2) for size in region_sizes]) + 1.0
#     trace_section_height = (2.0 if can_plot_traces else 1.5) * n_regions + 0.5
#     probe_height = 2.5
    
#     total_height = kin_height + heatmap_section_height + trace_section_height + probe_height + 1.5
    
#     fig = plt.figure(figsize=(12, total_height))
    
#     # Main GridSpec with 4 sections: kinematics, heatmaps, traces, probe
#     main_gs = GridSpec(4, 1, figure=fig, 
#                        height_ratios=[kin_height, heatmap_section_height, trace_section_height, probe_height],
#                        hspace=0.3)  # Spacing between main sections
    
#     # --- Kinematics Panel ---
#     ax_kin = fig.add_subplot(main_gs[0])
    
#     if has_kinematics and ua_data.beh_rel_t is not None:
#         t = ua_data.beh_rel_t
        
#         if ua_data.middle_x_med is not None:
#             ax_kin.plot(t, ua_data.middle_x_med, 'b-', linewidth=1.5, label='Middle X')
#             if ua_data.middle_x_std is not None:
#                 ax_kin.fill_between(t,
#                                     ua_data.middle_x_med - ua_data.middle_x_std,
#                                     ua_data.middle_x_med + ua_data.middle_x_std,
#                                     color='blue', alpha=0.2)
        
#         if ua_data.middle_y_med is not None:
#             ax_kin.plot(t, ua_data.middle_y_med, 'r-', linewidth=1.5, label='Middle Y')
#             if ua_data.middle_y_std is not None:
#                 ax_kin.fill_between(t,
#                                     ua_data.middle_y_med - ua_data.middle_y_std,
#                                     ua_data.middle_y_med + ua_data.middle_y_std,
#                                     color='red', alpha=0.2)
        
#         ax_kin.axvline(0, color='k', linestyle='--', alpha=0.5, linewidth=1)
#         ax_kin.axhline(0, color='gray', linestyle='-', alpha=0.3)
#         ax_kin.set_ylabel("Position (normalized)", fontsize=10)
#         ax_kin.set_ylim(KINEMATICS_YLIM)
#         ax_kin.set_xlim(ua_data.ua_rel_t.min(), ua_data.ua_rel_t.max())
#         ax_kin.legend(loc='upper right', fontsize=9)
#         ax_kin.set_title("Middle Finger Position", fontsize=11)
#         ax_kin.set_xticklabels([])
#     else:
#         ax_kin.text(0.5, 0.5, "No kinematics data available", transform=ax_kin.transAxes,
#                     ha='center', va='center', fontsize=12, color='gray')
#         ax_kin.set_xlim(ua_data.ua_rel_t.min(), ua_data.ua_rel_t.max())
#         ax_kin.set_xticklabels([])
    
#     # --- UA Heatmaps Section (nested GridSpec with tight spacing) ---
#     heatmap_heights = [max(size * 0.06, 1.2) for size in region_sizes]
#     heatmap_gs = main_gs[1].subgridspec(n_regions, 2, width_ratios=[20, 1], 
#                                          height_ratios=heatmap_heights,
#                                          hspace=0.08, wspace=0.05)  # Tight spacing within heatmaps
    
#     channel_start = 0
#     heatmap_ims = []
    
#     for region_idx, region in enumerate(REGION_ORDER):
#         region_channels = channel_order.get(region, [])
#         n_region = len(region_channels)
        
#         ax = fig.add_subplot(heatmap_gs[region_idx, 0])
        
#         if n_region == 0:
#             ax.text(0.5, 0.5, f"{region}: No channels", transform=ax.transAxes,
#                     ha='center', va='center', fontsize=10, color='gray')
#             ax.set_xlim(ua_data.ua_rel_t.min(), ua_data.ua_rel_t.max())
#             heatmap_ims.append(None)
#             channel_start += n_region
#             continue
        
#         region_data = ua_ordered[channel_start:channel_start + n_region, :]
        
#         im = ax.imshow(
#             region_data,
#             aspect='auto',
#             origin='upper',
#             extent=[ua_data.ua_rel_t.min(), ua_data.ua_rel_t.max(), n_region - 0.5, -0.5],
#             cmap=COLORMAP,
#             vmin=vmin,
#             vmax=vmax,
#         )
#         heatmap_ims.append(im)
        
#         ax.axvline(0, color='k', linestyle='--', alpha=0.7, linewidth=1)
#         ax.set_ylabel(f"{region}\n({n_region} ch)", fontsize=10, color=REGION_COLORS.get(region, 'black'))
#         ax.set_xticklabels([])
        
#         channel_start += n_region
    
#     # Add single colorbar for all heatmaps
#     valid_im = next((im for im in heatmap_ims if im is not None), None)
#     if valid_im is not None:
#         ax_cb = fig.add_subplot(heatmap_gs[:, 1])
#         cbar = plt.colorbar(valid_im, cax=ax_cb)
#         cbar.set_label("Δ FR (Hz)", fontsize=9)
    
#     # --- Raw HPF Traces Section (nested GridSpec with tight spacing) ---
#     trace_gs = main_gs[2].subgridspec(n_regions, 1, hspace=0.1)  # Tight spacing within traces
    
#     channel_start = 0
    
#     for region_idx, region in enumerate(REGION_ORDER):
#         region_channels = channel_order.get(region, [])
#         n_region = len(region_channels)
        
#         ax = fig.add_subplot(trace_gs[region_idx])
        
#         if n_region == 0:
#             ax.text(0.5, 0.5, f"{region}: No channels", transform=ax.transAxes,
#                     ha='center', va='center', fontsize=10, color='gray')
#             ax.set_xlim(trace_window_ms)
#             channel_start += n_region
#             continue
        
#         if not can_plot_traces:
#             ax.text(0.5, 0.5, f"{region}: Raw traces not available", transform=ax.transAxes,
#                     ha='center', va='center', fontsize=10, color='gray')
#             ax.set_xlim(trace_window_ms)
#             ax.set_ylabel(f"{region}\n(µV)", fontsize=9, color=REGION_COLORS.get(region, 'black'))
#             if region_idx == n_regions - 1:
#                 ax.set_xlabel("Time from IR crossing / Stim (ms)", fontsize=10)
#             channel_start += n_region
#             continue
        
#         # Get movement time for this trial
#         movement_time_ms = raw_trace_data.movement_times_ms[trial_idx]
#         rec = raw_trace_data.recording
#         fs = raw_trace_data.fs
        
#         # Select top, middle, bottom channels
#         if n_region >= 3:
#             trace_indices = [0, n_region // 2, n_region - 1]
#             trace_labels = ['Top', 'Mid', 'Bot']
#         elif n_region == 2:
#             trace_indices = [0, 1]
#             trace_labels = ['Top', 'Bot']
#         else:
#             trace_indices = [0]
#             trace_labels = ['Ch']
        
#         trace_colors = ['darkblue', 'green', 'darkred'][:len(trace_indices)]
        
#         # Get the actual channel indices in the original data
#         region_ch_indices = [flat_order[channel_start + ti] for ti in trace_indices]
        
#         y_offset_step = (trace_ylim_uv[1] - trace_ylim_uv[0]) * 0.5
        
#         for i, (tr_idx, ch_idx, label, color) in enumerate(zip(trace_indices, region_ch_indices, trace_labels, trace_colors)):
#             # Extract raw trace
#             t, y = extract_trace_segment(rec, ch_idx, movement_time_ms, trace_window_ms, fs)
            
#             if t is None or y is None:
#                 continue
            
#             # Plot trace with offset for visibility
#             offset = (len(trace_indices) - 1 - i) * y_offset_step
#             ax.plot(t, y + offset, color=color, linewidth=0.6, alpha=0.8, label=label)
            
#             # Get spike times for this channel
#             meta = raw_trace_data.ridx_to_meta.get(ch_idx, {"eid": ch_idx, "oid": ch_idx})
#             oid = meta["oid"]
#             eid = meta["eid"]
            
#             ch_peaks_abs = np.asarray(raw_trace_data.peak_ms_dict.get(oid, []), float).ravel()
#             if ch_peaks_abs.size == 0 and eid > 0:
#                 ch_peaks_abs = np.asarray(raw_trace_data.peak_ms_dict.get(eid, []), float).ravel()
            
#             # Convert to relative times
#             pk_rel = ch_peaks_abs - movement_time_ms
#             pk_rel_win = pk_rel[(pk_rel >= trace_window_ms[0]) & (pk_rel <= trace_window_ms[1])]
            
#             # Plot spike markers
#             if pk_rel_win.size > 0 and t is not None:
#                 peak_idx = np.searchsorted(t, pk_rel_win)
#                 valid_mask = (peak_idx >= 0) & (peak_idx < len(y))
#                 peak_idx = peak_idx[valid_mask]
#                 pk_rel_clean = pk_rel_win[valid_mask]
                
#                 if len(peak_idx) > 0:
#                     ax.plot(pk_rel_clean, y[peak_idx] + offset, 'o', 
#                             color=color, markersize=3, alpha=0.8, markerfacecolor='none')
        
#         ax.axvline(0, color='k', linestyle='--', alpha=0.5, linewidth=1)
#         ax.set_ylabel(f"{region}\n(µV)", fontsize=9, color=REGION_COLORS.get(region, 'black'))
#         ax.set_xlim(trace_window_ms)
#         ax.set_ylim(-100, 250)
#         ax.legend(loc='upper right', fontsize=7, ncol=3)
        
#         if region_idx == n_regions - 1:
#             ax.set_xlabel("Time from IR crossing / Stim (ms)", fontsize=10)
#         else:
#             ax.set_xticklabels([])
        
#         channel_start += n_region
    
#     # --- Electrode Layout (separate section at bottom) ---
#     ax_probe = fig.add_subplot(main_gs[3])
#     _plot_probe_layout(ax_probe, elec_mapping, channel_order, ua_data)
    
#     # Title
#     if title:
#         trial_info = f" (Trial {trial_idx + 1})" if can_plot_traces else ""
#         fig.suptitle(title + trial_info, fontsize=13, y=0.995, fontweight='bold')
    
#     # Save
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_path, dpi=150, bbox_inches='tight')
#     fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     print(f"[saved] {out_path.name}")

def plot_peristim_figure(
    ua_data: UAData,
    elec_mapping: ElectrodeMapping,
    channel_order: dict[str, list[int]],
    raw_trace_data: Optional[RawTraceData],
    out_path: Path,
    title: str = "",
    vmin: float = VMIN_UA,
    vmax: float = VMAX_UA,
    trace_window_ms: tuple[float, float] = TRACE_WINDOW_MS,
    trace_ylim_uv: tuple[float, float] = TRACE_YLIM_UV,
    trial_idx: int = 0,
    waveform_pre_ms: float = 1,
    waveform_post_ms: float = 1.5,
    max_waveforms: int = 100,
):
    """
    Plot peri-stimulus figure with:
    - Kinematics (X and Y on same panel)
    - UA heatmaps by region (SMA, PMd, M1i, M1s)
    - Raw HPF traces with spike markers (top, middle, bottom channel per region)
    - Spike waveform overlays for displayed channels
    - Electrode layout (4 grids at bottom)
    """
    flat_order, region_labels, region_boundaries = get_flat_channel_order(channel_order)
    n_channels_total = len(flat_order)
    
    if n_channels_total == 0:
        print(f"[warn] No channels to plot for {out_path.name}")
        return
    
    # Reorder UA data
    ua_ordered = ua_data.ua_med[flat_order, :]
    
    # Check kinematics
    has_kinematics = ua_data.middle_x_med is not None or ua_data.middle_y_med is not None
    
    # Check if we can plot raw traces
    can_plot_traces = (
        raw_trace_data is not None and
        raw_trace_data.movement_times_ms is not None and
        len(raw_trace_data.movement_times_ms) > trial_idx
    )
    
    n_regions = len(REGION_ORDER)
    region_sizes = [len(channel_order.get(r, [])) for r in REGION_ORDER]
    
    # Calculate height ratios for MAIN sections
    kin_height = 2.5
    heatmap_section_height = sum([max(size * 0.06, 1.2) for size in region_sizes]) + 1.0
    trace_section_height = (2.0 if can_plot_traces else 1.5) * n_regions + 0.5
    waveform_section_height = 3 * n_regions if can_plot_traces else 0  # New waveform section
    probe_height = 2.5
    
    total_height = kin_height + heatmap_section_height + trace_section_height + waveform_section_height + probe_height + 2.0
    
    # Adjust grid spec based on whether we have waveforms
    if can_plot_traces:
        n_main_sections = 5  # kin, heatmap, traces, waveforms, probe
        height_ratios = [kin_height, heatmap_section_height, trace_section_height, waveform_section_height, probe_height]
    else:
        n_main_sections = 4  # kin, heatmap, traces, probe
        height_ratios = [kin_height, heatmap_section_height, trace_section_height, probe_height]
    
    fig = plt.figure(figsize=(18, total_height))
    
    # Main GridSpec
    main_gs = GridSpec(n_main_sections, 1, figure=fig,
                       height_ratios=height_ratios,
                       hspace=0.3)
    
    # --- Kinematics Panel ---
    ax_kin = fig.add_subplot(main_gs[0])
    
    if has_kinematics and ua_data.beh_rel_t is not None:
        t = ua_data.beh_rel_t
        
        if ua_data.middle_x_med is not None:
            ax_kin.plot(t, ua_data.middle_x_med, 'b-', linewidth=1.5, label='Middle X')
            if ua_data.middle_x_std is not None:
                ax_kin.fill_between(t,
                                    ua_data.middle_x_med - ua_data.middle_x_std,
                                    ua_data.middle_x_med + ua_data.middle_x_std,
                                    color='blue', alpha=0.2)
        
        if ua_data.middle_y_med is not None:
            ax_kin.plot(t, ua_data.middle_y_med, 'r-', linewidth=1.5, label='Middle Y')
            if ua_data.middle_y_std is not None:
                ax_kin.fill_between(t,
                                    ua_data.middle_y_med - ua_data.middle_y_std,
                                    ua_data.middle_y_med + ua_data.middle_y_std,
                                    color='red', alpha=0.2)
        
        ax_kin.axvline(0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax_kin.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax_kin.set_ylabel("Position (normalized)", fontsize=10)
        ax_kin.set_ylim(KINEMATICS_YLIM)
        ax_kin.set_xlim(ua_data.ua_rel_t.min(), ua_data.ua_rel_t.max())
        ax_kin.legend(loc='upper right', fontsize=9)
        ax_kin.set_title("Middle Finger Position", fontsize=11)
        ax_kin.set_xticklabels([])
    else:
        ax_kin.text(0.5, 0.5, "No kinematics data available", transform=ax_kin.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
        ax_kin.set_xlim(ua_data.ua_rel_t.min(), ua_data.ua_rel_t.max())
        ax_kin.set_xticklabels([])
    
    # --- UA Heatmaps Section ---
    heatmap_heights = [max(size * 0.06, 1.2) for size in region_sizes]
    heatmap_gs = main_gs[1].subgridspec(n_regions, 2, width_ratios=[20, 1],
                                         height_ratios=heatmap_heights,
                                         hspace=0.08, wspace=0.05)
    
    channel_start = 0
    heatmap_ims = []
    
    for region_idx, region in enumerate(REGION_ORDER):
        region_channels = channel_order.get(region, [])
        n_region = len(region_channels)
        
        ax = fig.add_subplot(heatmap_gs[region_idx, 0])
        
        if n_region == 0:
            ax.text(0.5, 0.5, f"{region}: No channels", transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
            ax.set_xlim(ua_data.ua_rel_t.min(), ua_data.ua_rel_t.max())
            heatmap_ims.append(None)
            channel_start += n_region
            continue
        
        region_data = ua_ordered[channel_start:channel_start + n_region, :]
        
        im = ax.imshow(
            region_data,
            aspect='auto',
            origin='upper',
            extent=[ua_data.ua_rel_t.min(), ua_data.ua_rel_t.max(), n_region - 0.5, -0.5],
            cmap=COLORMAP,
            vmin=vmin,
            vmax=vmax,
        )
        heatmap_ims.append(im)
        
        ax.axvline(0, color='k', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_ylabel(f"{region}\n({n_region} ch)", fontsize=10, color=REGION_COLORS.get(region, 'black'))
        ax.set_xticklabels([])
        
        channel_start += n_region
    
    # Add single colorbar for all heatmaps
    valid_im = next((im for im in heatmap_ims if im is not None), None)
    if valid_im is not None:
        ax_cb = fig.add_subplot(heatmap_gs[:, 1])
        cbar = plt.colorbar(valid_im, cax=ax_cb)
        cbar.set_label("Δ FR (Hz)", fontsize=9)
    
    # --- Raw HPF Traces Section ---
    trace_gs = main_gs[2].subgridspec(n_regions, 1, hspace=0.1)
    
    # Store channel info for waveform plotting
    displayed_channels_per_region = {}
    
    channel_start = 0
    
    for region_idx, region in enumerate(REGION_ORDER):
        region_channels = channel_order.get(region, [])
        n_region = len(region_channels)
        
        ax = fig.add_subplot(trace_gs[region_idx])
        
        displayed_channels_per_region[region] = []
        
        if n_region == 0:
            ax.text(0.5, 0.5, f"{region}: No channels", transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
            ax.set_xlim(trace_window_ms)
            channel_start += n_region
            continue
        
        if not can_plot_traces:
            ax.text(0.5, 0.5, f"{region}: Raw traces not available", transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
            ax.set_xlim(trace_window_ms)
            ax.set_ylabel(f"{region}\n(µV)", fontsize=9, color=REGION_COLORS.get(region, 'black'))
            if region_idx == n_regions - 1:
                ax.set_xlabel("Time from IR crossing / Stim (ms)", fontsize=10)
            channel_start += n_region
            continue
        
        # Get movement time for this trial
        movement_time_ms = raw_trace_data.movement_times_ms[trial_idx]
        rec = raw_trace_data.recording
        fs = raw_trace_data.fs
        
        # Select top, middle, bottom channels
        if n_region >= 3:
            trace_indices = [0, n_region // 2, n_region - 1]
            trace_labels = ['Top', 'Mid', 'Bot']
        elif n_region == 2:
            trace_indices = [0, 1]
            trace_labels = ['Top', 'Bot']
        else:
            trace_indices = [0]
            trace_labels = ['Ch']
        
        trace_colors = ['darkblue', 'green', 'darkred'][:len(trace_indices)]
        
        # Get the actual channel indices in the original data
        region_ch_indices = [flat_order[channel_start + ti] for ti in trace_indices]
        
        # Store for waveform plotting
        displayed_channels_per_region[region] = list(zip(region_ch_indices, trace_labels, trace_colors))
        
        y_offset_step = (trace_ylim_uv[1] - trace_ylim_uv[0]) * 0.5
        
        for i, (tr_idx, ch_idx, label, color) in enumerate(zip(trace_indices, region_ch_indices, trace_labels, trace_colors)):
            # Extract raw trace
            t, y = extract_trace_segment(rec, ch_idx, movement_time_ms, trace_window_ms, fs)
            
            if t is None or y is None:
                continue
            
            # Plot trace with offset for visibility
            offset = (len(trace_indices) - 1 - i) * y_offset_step
            ax.plot(t, y + offset, color=color, linewidth=0.6, alpha=0.8, label=label)
            
            # Get spike times for this channel
            meta = raw_trace_data.ridx_to_meta.get(ch_idx, {"eid": ch_idx, "oid": ch_idx})
            oid = meta["oid"]
            eid = meta["eid"]
            
            ch_peaks_abs = np.asarray(raw_trace_data.peak_ms_dict.get(oid, []), float).ravel()
            if ch_peaks_abs.size == 0 and eid > 0:
                ch_peaks_abs = np.asarray(raw_trace_data.peak_ms_dict.get(eid, []), float).ravel()
            
            # Convert to relative times
            pk_rel = ch_peaks_abs - movement_time_ms
            pk_rel_win = pk_rel[(pk_rel >= trace_window_ms[0]) & (pk_rel <= trace_window_ms[1])]
            
            # Plot spike markers
            if pk_rel_win.size > 0 and t is not None:
                peak_idx = np.searchsorted(t, pk_rel_win)
                valid_mask = (peak_idx >= 0) & (peak_idx < len(y))
                peak_idx = peak_idx[valid_mask]
                pk_rel_clean = pk_rel_win[valid_mask]
                
                if len(peak_idx) > 0:
                    ax.plot(pk_rel_clean, y[peak_idx] + offset, 'o',
                            color=color, markersize=3, alpha=0.8, markerfacecolor='none')
        
        ax.axvline(0, color='k', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylabel(f"{region}\n(µV)", fontsize=9, color=REGION_COLORS.get(region, 'black'))
        ax.set_xlim(trace_window_ms)
        ax.set_ylim(-100, 250)
        ax.legend(loc='upper right', fontsize=7, ncol=3)
        
        if region_idx == n_regions - 1:
            ax.set_xlabel("Time from IR crossing / Stim (ms)", fontsize=10)
        else:
            ax.set_xticklabels([])
        
        channel_start += n_region
    
    # --- Spike Waveform Section ---
    if can_plot_traces:
        # 3 channels per region × 3 time conditions = 9 columns
        # Time conditions: All spikes, Pre-stim only, Post-stim only
        n_time_conditions = 3
        n_channels_per_region = 3  # Top, Mid, Bot
        n_cols = n_time_conditions * n_channels_per_region  # 9 columns
        
        # Define time windows for spike selection (relative to movement onset)
        waveform_time_windows = [
            ("All", None, None),           # All spikes from session
            ("Pre", -600.0, 0.0),          # Pre-stim window
            ("Post", 0.0, 600.0),          # Post-stim window
        ]
        
        waveform_gs = main_gs[3].subgridspec(n_regions, n_cols, hspace=0.4, wspace=0.25)
        
        rec = raw_trace_data.recording
        fs = raw_trace_data.fs
        movement_times = raw_trace_data.movement_times_ms
        
        for region_idx, region in enumerate(REGION_ORDER):
            channels_info = displayed_channels_per_region.get(region, [])
            
            if not channels_info:
                # Empty region - create placeholder spanning all columns
                ax = fig.add_subplot(waveform_gs[region_idx, :])
                ax.text(0.5, 0.5, f"{region}: No waveforms", transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='gray')
                ax.axis('off')
                continue
            
            # Pad channels_info to always have 3 entries
            while len(channels_info) < n_channels_per_region:
                channels_info.append((None, "", "gray"))
            
            for time_idx, (time_label, t_start, t_end) in enumerate(waveform_time_windows):
                for ch_idx_in_region, (ch_idx, ch_label, color) in enumerate(channels_info[:n_channels_per_region]):
                    col_idx = time_idx * n_channels_per_region + ch_idx_in_region
                    ax = fig.add_subplot(waveform_gs[region_idx, col_idx])
                    
                    if ch_idx is None:
                        ax.axis('off')
                        continue
                    
                    # Get spike times for this channel
                    meta = raw_trace_data.ridx_to_meta.get(ch_idx, {"eid": ch_idx, "oid": ch_idx})
                    oid = meta["oid"]
                    eid = meta["eid"]
                    
                    ch_peaks_abs = np.asarray(raw_trace_data.peak_ms_dict.get(oid, []), float).ravel()
                    if ch_peaks_abs.size == 0 and eid > 0:
                        ch_peaks_abs = np.asarray(raw_trace_data.peak_ms_dict.get(eid, []), float).ravel()
                    
                    # Filter spikes based on time window
                    if t_start is not None and t_end is not None and movement_times is not None:
                        # Select spikes that fall within the time window for ANY trial
                        selected_spikes = []
                        for mv_time in movement_times:
                            # Convert to relative time
                            rel_times = ch_peaks_abs - mv_time
                            # Find spikes in this window
                            mask = (rel_times >= t_start) & (rel_times < t_end)
                            selected_spikes.extend(ch_peaks_abs[mask])
                        ch_peaks_local = np.array(selected_spikes) - raw_trace_data.rec_start_ms
                    else:
                        # All spikes
                        ch_peaks_local = ch_peaks_abs - raw_trace_data.rec_start_ms
                    
                    # Remove duplicates (spikes might be selected from overlapping trial windows)
                    if len(ch_peaks_local) > 0:
                        ch_peaks_local = np.unique(ch_peaks_local)
                    
                    # Extract waveforms
                    waveforms, t_wf = extract_spike_waveforms(
                        rec, ch_idx, ch_peaks_local, fs,
                        pre_ms=waveform_pre_ms,
                        post_ms=waveform_post_ms,
                        max_spikes=max_waveforms,
                    )
                    
                    n_spikes = waveforms.shape[0] if waveforms is not None else 0
                    
                    if waveforms is not None and n_spikes > 0:
                        # Plot individual waveforms
                        for wf in waveforms:
                            ax.plot(t_wf, wf, color=color, alpha=0.1, linewidth=0.4)
                        
                        # Plot mean waveform with 1.96 SEM shading
                        wf_mean = np.mean(waveforms, axis=0)
                        wf_sem = 1.96 * np.std(waveforms, axis=0) / np.sqrt(n_spikes)
                        
                        ax.fill_between(t_wf, wf_mean - wf_sem, wf_mean + wf_sem,
                                        color=color, alpha=0.3)
                        ax.plot(t_wf, wf_mean, color=color, linewidth=1.5)
                        
                        # Add vertical line at peak
                        ax.axvline(0, ls='--', color='gray', alpha=0.5, linewidth=0.8)
                        
                        # Compute peak-to-trough
                        p2t = np.max(wf_mean) - np.min(wf_mean)
                        
                        # Title: show region/channel only for first time condition
                        if time_idx == 0:
                            title_str = f"{region} {ch_label}\n{time_label} n={n_spikes}"
                        else:
                            title_str = f"{time_label} n={n_spikes}"
                        
                        ax.set_title(title_str, fontsize=7, color=color, pad=2)
                    else:
                        ax.text(0.5, 0.5, f"n=0", transform=ax.transAxes,
                                ha='center', va='center', fontsize=8, color='gray')
                        if time_idx == 0:
                            ax.set_title(f"{region} {ch_label}\n{time_label}", fontsize=7, color=color, pad=2)
                        else:
                            ax.set_title(f"{time_label}", fontsize=7, color=color, pad=2)
                    
                    ax.set_xlim(-waveform_pre_ms, waveform_post_ms)
                    ax.set_ylim(-60, 60)
                    
                    # X-axis label only on bottom row
                    if region_idx == n_regions - 1:
                        ax.set_xlabel("ms", fontsize=7)
                    else:
                        ax.set_xticklabels([])
                    
                    # Y-axis label only on first column of each time condition
                    if ch_idx_in_region == 0:
                        ax.set_ylabel("µV", fontsize=7)
                    else:
                        ax.set_yticklabels([])
                    
                    # Reduce tick label size
                    ax.tick_params(axis='both', labelsize=6)
        
        # Add column headers for time conditions
        for time_idx, (time_label, _, _) in enumerate(waveform_time_windows):
            col_center = time_idx * n_channels_per_region + (n_channels_per_region - 1) / 2
            # Add text above the first row
            fig.text(
                0.15 + (col_center + 0.5) / n_cols * 0.7,  # x position (adjusted for margins)
                0.98 - (kin_height + heatmap_section_height + trace_section_height) / total_height - 0.02,
                f"── {time_label} Spikes ──",
                ha='center', va='bottom', fontsize=9, fontweight='bold'
            )
        
        probe_gs_idx = 4
    else:
        probe_gs_idx = 3
    
    # --- Electrode Layout ---
    ax_probe = fig.add_subplot(main_gs[probe_gs_idx])
    _plot_probe_layout(ax_probe, elec_mapping, channel_order, ua_data)
    
    # Title
    if title:
        trial_info = f" (Trial {trial_idx + 1})" if can_plot_traces else ""
        fig.suptitle(title + trial_info, fontsize=13, y=0.995, fontweight='bold')
    
    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path.name}")

def _plot_probe_layout(
    ax: plt.Axes,
    elec_mapping: ElectrodeMapping,
    channel_order: dict[str, list[int]],
    ua_data: UAData,
):
    """Plot electrode layout as 4 separate 8x8 grids."""
    ax.set_aspect('equal')
    ax.axis('off')
    # ax.set_title("Electrode Layout (by region)", fontsize=10, pad=20)
    
    grid_spacing = 10
    electrode_size = 0.8
    
    present_channels = set()
    for region, channels in channel_order.items():
        present_channels.update(channels)
    
    nsp_offset = 0 if ua_data.ua_port.upper() == 'A' else 128
    present_elecs = set()
    for ch_idx in present_channels:
        nsp_id = ch_idx + 1 + nsp_offset
        elec_id = elec_mapping.nsp_to_elec.get(nsp_id, -1)
        if elec_id > 0:
            present_elecs.add(elec_id)
    
    for region_idx, region in enumerate(REGION_ORDER):
        x_offset = region_idx * grid_spacing
        
        ax.text(x_offset + 3.5, -1.5, region, ha='center', va='bottom', fontsize=10,
                fontweight='bold', color=REGION_COLORS.get(region, 'black'))
        
        grid = elec_mapping.region_grids.get(region, np.zeros((8, 8)))
        
        for row in range(8):
            for col in range(8):
                elec_id = int(grid[row, col])
                x = x_offset + col
                y = 7 - row
                
                if elec_id > 0 and elec_id in present_elecs:
                    color = REGION_COLORS.get(region, 'gray')
                    alpha = 0.8
                else:
                    color = 'lightgray'
                    alpha = 0.3
                
                rect = mpatches.Rectangle(
                    (x - electrode_size/2, y - electrode_size/2),
                    electrode_size, electrode_size,
                    facecolor=color, edgecolor='black', alpha=alpha, linewidth=0.5
                )
                ax.add_patch(rect)
    
    ax.set_xlim(-1, len(REGION_ORDER) * grid_spacing)
    ax.set_ylim(-3, 9)


# -----------------------------------------------------------------------------
# Plotting: Cross-Session Comparison Heatmaps
# -----------------------------------------------------------------------------
def plot_cross_session_windows(
    all_data: list[UAData],
    all_window_values: list[np.ndarray],
    all_channel_orders: list[dict[str, list[int]]],
    time_windows: list[tuple[float, float, str]],
    out_path: Path,
    title: str = "",
    vmin: float = VMIN_UA,
    vmax: float = VMAX_UA,
    show_baseline: bool = True,
):
    """Plot cross-session comparison heatmap."""
    n_sessions = len(all_data)
    n_windows = len(time_windows) if show_baseline else len(time_windows) - 1
    
    if n_sessions == 0:
        print(f"[warn] No data for {out_path.name}")
        return
    
    ref_channel_order = all_channel_orders[0]
    flat_order, region_labels, region_boundaries = get_flat_channel_order(ref_channel_order)
    n_channels = len(flat_order)
    
    if n_channels == 0:
        print(f"[warn] No channels for {out_path.name}")
        return
    
    data_matrix = np.full((n_channels, n_sessions * n_windows), np.nan)
    
    for sess_idx, (ua_data, window_values, channel_order) in enumerate(zip(all_data, all_window_values, all_channel_orders)):
        sess_flat_order, _, _ = get_flat_channel_order(channel_order)
        
        for ref_ch_idx, ref_ch in enumerate(flat_order):
            if ref_ch in sess_flat_order:
                sess_ch_idx = sess_flat_order.index(ref_ch)
                data_matrix[ref_ch_idx, sess_idx * n_windows:(sess_idx + 1) * n_windows] = window_values[sess_ch_idx, :]
    
    fig_width = max(12, n_sessions * n_windows * 0.8 + 4)
    fig_height = max(10, n_channels * 0.06 + 4)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    im = ax.imshow(
        data_matrix,
        aspect='auto',
        origin='upper',
        cmap=COLORMAP,
        vmin=vmin,
        vmax=vmax,
    )
    
    x_labels = []
    x_positions = []
    for sess_idx, ua_data in enumerate(all_data):
        for win_idx in range(n_windows):
            x_positions.append(sess_idx * n_windows + win_idx)
            if show_baseline:
                win_label = time_windows[win_idx][2].replace("win", "W").replace("_", "\n")
            else:
                win_label = f"Δ{time_windows[win_idx + 1][2]}".replace("win", "W").replace("_", "\n")
            x_labels.append(f"{win_label}")
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha='right')
    
    for sess_idx, ua_data in enumerate(all_data):
        x_center = sess_idx * n_windows + (n_windows - 1) / 2
        session_short = ua_data.session
        br_label = f"BR{ua_data.br_idx}"
        ax.text(x_center, -2, f"{session_short}\n{br_label}", ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    y_ticks = []
    y_labels_list = []
    
    region_boundaries_full = [0]
    for region in REGION_ORDER:
        n_region = len(ref_channel_order.get(region, []))
        region_boundaries_full.append(region_boundaries_full[-1] + n_region)
    
    for region_idx, region in enumerate(REGION_ORDER):
        start = region_boundaries_full[region_idx]
        end = region_boundaries_full[region_idx + 1]
        if end > start:
            y_center = (start + end) / 2
            y_ticks.append(y_center)
            y_labels_list.append(region)
            
            if region_idx > 0:
                ax.axhline(start - 0.5, color='white', linewidth=2)
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels_list, fontsize=10)
    
    for sess_idx in range(1, n_sessions):
        ax.axvline(sess_idx * n_windows - 0.5, color='white', linewidth=2)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_label("Δ FR (Hz)" if show_baseline else "Δ from baseline (Hz)", fontsize=10)
    
    ax.set_xlabel("Time Windows", fontsize=11)
    ax.set_ylabel("Channels (by region)", fontsize=11)
    
    if title:
        ax.set_title(title, fontsize=12, pad=50)
    
    fig.tight_layout()
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.45)
    fig.savefig(out_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path.name}")


def plot_stim_minus_control(
    control_data: list[UAData],
    stim_data: list[UAData],
    control_windows: list[np.ndarray],
    stim_windows: list[np.ndarray],
    control_orders: list[dict[str, list[int]]],
    time_windows: list[tuple[float, float, str]],
    out_path: Path,
    title: str = "",
):
    """Plot stim - control difference heatmap."""
    n_sessions = len(control_data)
    
    if n_sessions == 0:
        print(f"[warn] No data for {out_path.name}")
        return
    
    diff_windows = []
    for ctrl_win, stim_win in zip(control_windows, stim_windows):
        diff_windows.append(stim_win - ctrl_win)
    
    plot_cross_session_windows(
        control_data,
        diff_windows,
        control_orders,
        time_windows,
        out_path,
        title=title,
        vmin=VMIN_DIFF,
        vmax=VMAX_DIFF,
        show_baseline=True,
    )


def extract_spike_waveforms(
    rec,
    ch_row: int,
    spike_times_ms: np.ndarray,
    fs: float,
    pre_ms: float = 1,
    post_ms: float = 1.5,
    max_spikes: int = 100,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """
    Extract spike waveforms for a single channel.
    
    Parameters
    ----------
    rec : SpikeInterface recording
    ch_row : int
        Channel index in the recording
    spike_times_ms : np.ndarray
        Spike times in ms (in recording-local time)
    fs : float
        Sampling frequency in Hz
    pre_ms : float
        Time before spike peak to extract (ms)
    post_ms : float
        Time after spike peak to extract (ms)
    max_spikes : int
        Maximum number of spikes to extract (randomly sampled if more)
    
    Returns
    -------
    waveforms : np.ndarray or None
        Shape (n_spikes, n_samples), in µV
    t_wf : np.ndarray
        Time axis for waveforms (ms)
    """
    if spike_times_ms.size == 0:
        n_samples = int((pre_ms + post_ms) / 1000.0 * fs)
        return None, np.linspace(-pre_ms, post_ms, n_samples)
    
    ch_id = rec.get_channel_ids()[ch_row]
    n_frames = rec.get_num_frames()
    
    pre_samp = int(round(pre_ms / 1000.0 * fs))
    post_samp = int(round(post_ms / 1000.0 * fs))
    n_samples = pre_samp + post_samp
    
    # Subsample if too many spikes
    if len(spike_times_ms) > max_spikes:
        spike_times_ms = np.random.choice(spike_times_ms, max_spikes, replace=False)
    
    waveforms = []
    
    for spike_ms in spike_times_ms:
        center_samp = int(round(spike_ms / 1000.0 * fs))
        i0 = center_samp - pre_samp
        i1 = center_samp + post_samp
        
        # Skip if outside recording bounds
        if i0 < 0 or i1 > n_frames:
            continue
        
        try:
            wf = rec.get_traces(
                start_frame=i0,
                end_frame=i1,
                channel_ids=[ch_id],
                return_in_uV=True
            ).squeeze()
            
            if wf.shape[0] == n_samples:
                waveforms.append(wf)
        except Exception:
            continue
    
    t_wf = np.linspace(-pre_ms, post_ms, n_samples)
    
    if len(waveforms) == 0:
        return None, t_wf
    
    return np.array(waveforms), t_wf


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------
def run_analysis(cfg: ComparisonConfig):
    """Main analysis pipeline."""
    
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[info] Loading electrode mapping from {cfg.electrode_mapping_csv}")
    elec_mapping = load_electrode_mapping(cfg.electrode_mapping_csv)
    print(f"[info] Loaded {len(elec_mapping.nsp_to_elec)} electrode mappings")
    
    print(f"[info] Data root: {cfg.data_root}")
    print(f"[info] Output: {cfg.output_dir}")
    print(f"[info] Target: {cfg.target}")
    print(f"[info] Sessions: {len(cfg.session_specs)}")
    for spec in cfg.session_specs:
        print(f"       {spec.session}: ctrl=BR{spec.control_br}, stim=BR{spec.stim_br}")
    
    all_control_data: list[UAData] = []
    all_stim_data: list[UAData] = []
    all_control_windows: list[np.ndarray] = []
    all_stim_windows: list[np.ndarray] = []
    all_control_orders: list[dict[str, list[int]]] = []
    all_stim_orders: list[dict[str, list[int]]] = []
    all_control_deltas: list[np.ndarray] = []
    all_stim_deltas: list[np.ndarray] = []
    
    for spec in cfg.session_specs:
        print(f"\n{'='*60}")
        print(f"Processing {spec.session}")
        print(f"{'='*60}")
        
        # Find session folder
        session_folder = find_session_folder(cfg.data_root, spec.session)
        if session_folder is None:
            print(f"  [skip] Session folder not found")
            continue
        
        # --- Load Control ---
        print(f"\n  Loading control (BR{spec.control_br})...")
        ctrl_path = find_npz_file(cfg.data_root, spec.session, spec.control_br, cfg.target)
        
        if ctrl_path is None:
            print(f"  [skip] Control PeriStim data not found")
            continue
        
        ctrl_data = load_ua_data(ctrl_path, "control", session_folder)
        if ctrl_data is None:
            print(f"  [skip] Failed to load control data")
            continue
        
        print(f"  Control: {ctrl_data.n_channels} channels, {ctrl_data.n_trials} trials")
        
        # Load raw trace data for control
        print(f"  Loading raw traces for control...")
        ctrl_raw = load_raw_trace_data(session_folder, spec.control_br, debug=True)
        
        # --- Load Stim ---
        print(f"\n  Loading stim (BR{spec.stim_br})...")
        stim_path = find_npz_file(cfg.data_root, spec.session, spec.stim_br, cfg.target)
        
        if stim_path is None:
            print(f"  [skip] Stim PeriStim data not found")
            continue
        
        stim_data = load_ua_data(stim_path, "stim", session_folder)
        if stim_data is None:
            print(f"  [skip] Failed to load stim data")
            continue
        
        print(f"  Stim: {stim_data.n_channels} channels, {stim_data.n_trials} trials")
        
        # Load raw trace data for stim
        print(f"  Loading raw traces for stim...")
        stim_raw = load_raw_trace_data(session_folder, spec.stim_br, debug=True)
        
        # --- Build channel ordering (filtering out channels with no spikes) ---
        ctrl_order = build_channel_order(
            ctrl_data, elec_mapping, cfg.time_windows, 
            raw_trace_data=ctrl_raw, min_spike_count=cfg.min_spike_count
        )
        stim_order = build_channel_order(
            stim_data, elec_mapping, cfg.time_windows,
            raw_trace_data=stim_raw, min_spike_count=cfg.min_spike_count
        )
        
        print(f"  Channel counts (after filtering inactive channels):")
        for region in REGION_ORDER:
            ctrl_count = len(ctrl_order.get(region, []))
            stim_count = len(stim_order.get(region, []))
            print(f"    {region}: ctrl={ctrl_count}, stim={stim_count} channels")
        
        # --- Compute window values ---
        ctrl_windows = compute_window_values(ctrl_data.ua_med, ctrl_data.ua_rel_t, cfg.time_windows)
        stim_windows = compute_window_values(stim_data.ua_med, stim_data.ua_rel_t, cfg.time_windows)
        
        ctrl_deltas = compute_change_from_baseline(ctrl_windows)
        stim_deltas = compute_change_from_baseline(stim_windows)
        
        all_control_data.append(ctrl_data)
        all_stim_data.append(stim_data)
        all_control_windows.append(ctrl_windows)
        all_stim_windows.append(stim_windows)
        all_control_orders.append(ctrl_order)
        all_stim_orders.append(stim_order)
        all_control_deltas.append(ctrl_deltas)
        all_stim_deltas.append(stim_deltas)
        
        sess_out = cfg.output_dir / "per_session"
        
        print(f"\n  Generating peri-stimulus plots...")
        
        # Control
        plot_peristim_figure(
            ctrl_data, elec_mapping, ctrl_order, ctrl_raw,
            sess_out / f"{spec.session}_BR{spec.control_br:03d}_control_peristim.svg",
            title=f"{spec.session} - Control (BR{spec.control_br}) - n={ctrl_data.n_trials} trials",
            trial_idx=0,
        )
        
        # Stim
        plot_peristim_figure(
            stim_data, elec_mapping, stim_order, stim_raw,
            sess_out / f"{spec.session}_BR{spec.stim_br:03d}_stim_peristim.svg",
            title=f"{spec.session} - Stim (BR{spec.stim_br}) - n={stim_data.n_trials} trials",
            trial_idx=0,
        )
    
    if len(all_control_data) < 1:
        print("\n[warn] No sessions loaded, skipping cross-session plots")
        return
    
    print(f"\n{'='*60}")
    print("Generating cross-session comparison plots")
    print(f"{'='*60}")
    
    cross_out = cfg.output_dir / "cross_session"
    
    print("\n  Control: Mean per window...")
    plot_cross_session_windows(
        all_control_data,
        all_control_windows,
        all_control_orders,
        cfg.time_windows,
        cross_out / "cross_session_control_mean_windows.svg",
        title="Control Conditions: Mean Activity per Window",
        show_baseline=True,
    )
    
    print("\n  Control: Change from baseline...")
    plot_cross_session_windows(
        all_control_data,
        all_control_deltas,
        all_control_orders,
        cfg.time_windows,
        cross_out / "cross_session_control_delta_baseline.svg",
        title="Control Conditions: Change from Baseline (win1)",
        vmin=VMIN_DIFF, vmax=VMAX_DIFF,
        show_baseline=False,
    )
    
    print("\n  Stim: Mean per window...")
    plot_cross_session_windows(
        all_stim_data,
        all_stim_windows,
        all_stim_orders,
        cfg.time_windows,
        cross_out / "cross_session_stim_mean_windows.svg",
        title="Stim Conditions: Mean Activity per Window",
        show_baseline=True,
    )
    
    print("\n  Stim: Change from baseline...")
    plot_cross_session_windows(
        all_stim_data,
        all_stim_deltas,
        all_stim_orders,
        cfg.time_windows,
        cross_out / "cross_session_stim_delta_baseline.svg",
        title="Stim Conditions: Change from Baseline (win1)",
        vmin=VMIN_DIFF, vmax=VMAX_DIFF,
        show_baseline=False,
    )
    
    print("\n  Stim - Control difference...")
    plot_stim_minus_control(
        all_control_data,
        all_stim_data,
        all_control_windows,
        all_stim_windows,
        all_control_orders,
        cfg.time_windows,
        cross_out / "cross_session_stim_minus_control.svg",
        title="Stim - Control Difference per Window",
    )
    
    print(f"\n[done] All results saved to: {cfg.output_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cross-session UA comparison")
    parser.add_argument("--config", "-c", type=Path, required=True,
                        help="Path to YAML config file")
    
    args = parser.parse_args()
    
    config_path = args.config
    if not config_path.exists():
        script_dir = Path(__file__).parent
        config_path = script_dir / args.config
    
    if not config_path.exists():
        print(f"[error] Config not found: {args.config}")
        return
    
    cfg = load_config(config_path)
    run_analysis(cfg)


if __name__ == "__main__":
    main()