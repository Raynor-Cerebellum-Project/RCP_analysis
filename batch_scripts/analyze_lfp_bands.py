import gc
import os
import csv
import re
import numpy as np
from scipy import signal
from scipy.io import loadmat
from scipy.optimize import curve_fit
from pathlib import Path
import pandas as pd

# SpikeInterface
from probeinterface import Probe
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
import warnings
from joblib import Parallel, delayed
from scipy.optimize import curve_fit, OptimizeWarning

import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

# ---------- BASELINE TRIAL EXCLUSION ----------
# Format: "baseline_port{A,B}_target{A,B}": [list of 0-based trial indices to skip]
# Use this to remove false positive IR-triggered events (e.g., multiple triggers on same reach)
EXCLUDE_BASELINE_TRIALS = {
    "baseline_portA_targetA": [1, 3, 4, 6, 13, 14, 21, 24],
    "baseline_portB_targetA": [2, 6, 7, 9, 13, 18],
    "baseline_portA_targetB": [6, 10, 13, 15, 16, 17, 20, 21, 22, 24, 25, 30, 39, 40, 42, 47, 56, 57],
    "baseline_portB_targetB": [2, 3, 4, 6, 8, 10, 12, 15, 17, 20, 22, 23, 24, 26, 28, 30, 31, 35],
}

"""
    Analyzes LFP bands (Alpha, Beta, Low/High Gamma) for NPRW (Intan) and Utah Array (Blackrock)
    
    Pipeline:
      1. Extraction: Loads stimulation events and extracts epochs (-500ms to +1000ms) with padding.
      2. Blanking: Applies 'copy_baseline' blanking to remove stim artifacts (-5ms to +101ms).
      3. Local CMR (Intan): Subtracts common median within local radius (loaded from config). 
      4. Utah CMR: DISABLED.
      5. Filtering: Zero-phase bandpass filtering in specific frequency bands (using padded epochs).
      6. Cleaning/Rejection: 
         - No baseline correction or exponential/template subtraction.
         - Bad channels rejected based on IMPEDANCE thresholds (Intan > 7000 kOhm, Utah > 1000 kOhm).
      
    Associated Scripts:
      - debug_lfp_pipeline_plots.py: Visualizes the pipeline steps.
      - plot_lfp_check.py: General quality check of aligned LFP.
    
    Output:
        Aligned data .npz files containing windowed traces for each band.
        - NPRW: results/checkpoints/NPRW_LFP/aligned_lfp__{session_name}.npz
        - Utah: results/checkpoints/UA_LFP/aligned_lfp__{session_name}.npz
"""

# ---------- CONSTANTS ----------
TARGET_FS = 1000
BLANK_PRE_MS = 5.0
BLANK_POST_MS = 101.0 
EPOCH_PRE_MS = 500.0
EPOCH_POST_MS = 1000.0
PAD_MS = 1000.0 # Padding for zero-phase filtering
INTAN_IMPEDANCE_CUTOFF = 7000.0 # in kOhm
UTAH_IMPEDANCE_CUTOFF = 1000.0 # in kOhm

# ---------- CONFIG ----------
# Bands of interest
LFP_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 25),
    "low_gamma": (25, 60),
    "high_gamma": (60, 120)
}

# Filter order for butterworth
FILTER_ORDER = 4

# Output Directories
NPRW_LFP_CKPT_ROOT = OUT_BASE / "checkpoints" / "NPRW_LFP"
UA_LFP_CKPT_ROOT = OUT_BASE / "checkpoints" / "UA_LFP"
ALIGNED_CKPT_ROOT = OUT_BASE / "checkpoints" / "Aligned"
PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim"

NPRW_LFP_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
UA_LFP_CKPT_ROOT.mkdir(parents=True, exist_ok=True)

# NPRW Config
NPRW_CFG = PARAMS.probes.get("NPRW")
INTAN_STREAM = NPRW_CFG.get("neural_data_stream")
STIM_STREAM = NPRW_CFG.get("stim_data_stream")
AUX_STREAM = NPRW_CFG.get("aux_stream")
RADII = (NPRW_CFG.get("local_radius_inner"), NPRW_CFG.get("local_radius_outer"))

# Utah Config
UA_CFG = PARAMS.probes.get("UA")
XLS = rcp.ua_excel_path(REPO_ROOT, PARAMS.probes)
UA_MAP = rcp.load_UA_mapping_from_excel(XLS) if XLS else None
METADATA_CSV = METADATA_ROOT / f"{PARAMS.session}_metadata.csv"
SHIFT_CSV = METADATA_ROOT / "br_to_intan_shifts.csv"

# Impedance Config (Utah)
IMP_BASE = OUT_BASE.parents[0]  # One level above OUT_BASE (results)
IMP_FILES = {
    "A": IMP_BASE / "Impedances" / "Utah_imp_Bank_A_start",
    "B": IMP_BASE / "Impedances" / "Utah_imp_Bank_B_start",
}

# Global job kwargs
global_job_kwargs = dict(n_jobs=PARAMS.parallel_jobs, chunk_duration=PARAMS.chunk)
si.set_global_job_kwargs(**global_job_kwargs)


# =============================================================================
# Metadata Helpers (for Baseline Grouping)
# =============================================================================

def _normalize_metadata(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase with underscores."""
    df = df_raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _find_baseline_groups(df_norm: pd.DataFrame) -> dict:
    """
    Return mapping (ua_port_norm, depth_mm_norm) -> row indices of baseline rows.
    Mirrors the grouping logic from extract_peri_IR_and_concat_baseline.py.
    """
    if "notes" not in df_norm.columns:
        return {}

    is_baseline = df_norm["notes"].astype(str).str.lower().eq("baseline")
    df_base = df_norm.loc[is_baseline].copy()

    if df_base.empty:
        return {}

    # UA_port
    if "ua_port" in df_base.columns:
        df_base["_ua_port_norm"] = (
            df_base["ua_port"].astype(str).str.upper().str.strip().replace({"": "UNKNOWN"})
        )
    else:
        df_base["_ua_port_norm"] = "UNKNOWN"

    # Depth_mm
    if "depth_mm" in df_base.columns:
        depth_num = pd.to_numeric(df_base["depth_mm"], errors="coerce")
        df_base["_depth_mm_norm"] = depth_num.where(
            depth_num.notna(), df_base["depth_mm"].astype(str)
        ).replace({"": "n/a"})
    else:
        df_base["_depth_mm_norm"] = "n/a"

    groups = {}
    for (port, depth), g in df_base.groupby(["_ua_port_norm", "_depth_mm_norm"], sort=True):
        key = (str(port), str(depth))
        groups[key] = g.index.to_numpy()
    return groups


# =============================================================================
# Core Cleaning Functions
# =============================================================================

def filter_zero_phase(epochs, fs, low, high, order=4):
    """
    Apply zero-phase bandpass filter using filtfilt along axis 2 (time).
    
    Args:
        epochs (np.ndarray): (n_trials, n_ch, n_time)
        fs (float): Sampling rate (Hz)
        low (float): Low cutoff (Hz)
        high (float): High cutoff (Hz)
        order (int): Filter order
        
    Returns:
        np.ndarray: Filtered data
    """
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    
    b, a = signal.butter(order, [low_norm, high_norm], btype='band')
    
    # Apply filtfilt along time axis (axis=2)
    return signal.filtfilt(b, a, epochs, axis=2)

# =============================================================================
# Helper Functions
# =============================================================================

def get_bad_channels_intan(sess_path, threshold_kohm=2000.0):
    """
    Search for impedance.csv in the session path and identify bad channels.
    Returns list of channel names (e.g. 'A-000') that exceed threshold.
    """
    matches = list(sess_path.glob("*impedance*.csv"))
    if not matches:
        return []
        
    imp_file = matches[0]
    bad_ch_names = []
    print(f"  [Impedance] Loading {imp_file.name}...")
    try:
        # Intan impedance CSV: "Channel Name", "Impedance Magnitude at 1000 Hz (kOhm)", ...
        # Skips header
        with open(imp_file, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            # Intan often has some metadata lines before header? 
            # Usually header starts with "Channel Name"
            header = None
            for row in reader:
                if row and "Channel Name" in row[0]:
                    header = row
                    break
            
            if not header:
                 print("  [Warn] No header found in impedance file.")
                 return []

            # Find column
            imp_col = -1
            for i, h in enumerate(header):
                if "kOhm" in h and "1000" in h:
                    imp_col = i
                    break
            # Fallback if simplified header
            if imp_col == -1:
                 for i, h in enumerate(header):
                     if "kOhm" in h:
                         imp_col = i
                         break
            
            if imp_col == -1:
                print("  [Warn] Could not find kOhm column in impedance file.")
                return []
                
            for row in reader:
                if not row: continue
                try:
                    ch_name = row[0].strip()
                    imp_val = float(row[imp_col])
                    if imp_val > threshold_kohm:
                        bad_ch_names.append(ch_name)
                except:
                    continue
    except Exception as e:
        print(f"  [Warn] Failed to read impedance file: {e}")
        return []
        
    return bad_ch_names

# =============================================================================
# Utah Impedance Helper Functions
# =============================================================================

_imp_pat_elecnum = re.compile(
    r"\belec\s*\d+\s*-\s*(\d{1,3})\s+([0-9]+(?:\.[0-9]+)?)\s*(k?ohms?|kΩ|ohms?|Ω)\b",
    flags=re.IGNORECASE,
)

def _read_text_loose(p: Path) -> str:
    if not p.exists():
        raise FileNotFoundError(str(p))
    b = p.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1", "mac_roman"):
        try:
            return b.decode(enc)
        except Exception:
            pass
    filtered = bytes(ch for ch in b if 9 <= ch <= 126 or ch in (10, 13))
    return filtered.decode("latin-1", errors="ignore")

def _unit_to_kohm(val: float, unit: str) -> float:
    u = (unit or "").lower()
    if "k" in u:
        return float(val)
    return float(val) / 1000.0  # Ω → kΩ

def load_impedances_from_textedit_dump(path_like: str | Path) -> dict[int, float]:
    """
    Parse lines like 'elec1-5   201 kOhm' from a loose text dump.
    Returns { Elec#: impedance_kΩ }.
    """
    txt = _read_text_loose(Path(path_like))
    out: dict[int, float] = {}
    for m in _imp_pat_elecnum.finditer(txt):
        elec_str, val_str, unit = m.groups()
        try:
            elec = int(elec_str)
            val = float(val_str)
        except Exception:
            continue
        out[elec] = _unit_to_kohm(val, unit)
    return out

def get_bad_channels_utah(imp_files_dict, ua_elec_ids, threshold_kohm=2000.0):
    """
    Load impedances from known Utah files and return indices of bad channels 
    matching the current session's electrode IDs.
    
    Args:
        imp_files_dict: dict {port: Path}, e.g. {"A": path_to_A, "B": path_to_B}
        ua_elec_ids: (n_ch,) array of Electrode IDs (1-96 typically) corresponding to the data rows.
        threshold_kohm: Exclusion threshold.
        
    Returns:
        np.ndarray: Indices of rows in ua_elec_ids that have high impedance.
    """
    if ua_elec_ids is None or len(ua_elec_ids) == 0:
        return np.array([], dtype=int)
        
    # Load all ports
    imp_by_elec = {}
    for port, p in imp_files_dict.items():
        if p and p.exists():
            try:
                d = load_impedances_from_textedit_dump(p)
                imp_by_elec.update(d)
                print(f"  [Impedance] Parsed {len(d)} UA impedances from {p.name} (Port {port})")
            except Exception as e:
                print(f"  [Warn] Failed to parse UA impedance file {p}: {e}")
                
    bad_indices = []
    for r, eid in enumerate(ua_elec_ids):
        if eid is None or not np.isfinite(eid):
            continue
        z = imp_by_elec.get(int(eid)) # IDs are ints in the dict
        if z is not None and z > threshold_kohm:
            bad_indices.append(r)
            
    return np.array(bad_indices, dtype=int)

def get_intan_geometry():
    GEOM_PATH = rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
    mat_probe = loadmat(Path(GEOM_PATH))
    intan_geom = {}
    intan_geom["x"] = mat_probe["xcoords"].ravel()
    intan_geom["y"] = mat_probe["ycoords"].ravel()
    
    if "chanMap0ind" in mat_probe:
        intan_probe_mapping = intan_geom["device_index_0based"] = mat_probe["chanMap0ind"].ravel()
    else:
        raise ValueError("No 0-based chanmap in .mat geometry file.")
    return intan_geom, intan_probe_mapping

# --- Custom Blanking Implementation ---
class BlankingRecording(si.BaseRecording):
    def __init__(self, parent_recording, stim_indices, pre_ms=5.0, post_ms=20.0, mode='interp'):
        si.BaseRecording.__init__(self, parent_recording.get_sampling_frequency(), 
                               parent_recording.get_channel_ids(), 
                               parent_recording.get_dtype())
        
        # Copy properties to self (channel gains/offsets)
        parent_recording.copy_metadata(self)
        
        self._parent = parent_recording
        self._stim_indices = np.sort(stim_indices) # Ensure sorted
        self._pre_samples = int(pre_ms * self.get_sampling_frequency() / 1000.0)
        self._post_samples = int(post_ms * self.get_sampling_frequency() / 1000.0)
        self._mode = mode
        
        # Add segments
        for i in range(parent_recording.get_num_segments()):
            parent_segment = parent_recording._recording_segments[i]
            self.add_recording_segment(BlankingRecordingSegment(
                parent_segment, self._stim_indices, self._pre_samples, self._post_samples, mode=mode
            ))
            
        self._kwargs = {'parent_recording': parent_recording.to_dict(), 'stim_indices': stim_indices, 
                        'pre_ms': pre_ms, 'post_ms': post_ms, 'mode': mode}

class BlankingRecordingSegment(si.BaseRecordingSegment):
    """
    Custom SI segment that zeros out or interpolates over stir-related artifacts 
    defined by stim_indices.
    """
    def __init__(self, parent_segment, stim_indices, pre_samples, post_samples, mode='interp'):
        si.BaseRecordingSegment.__init__(self, **parent_segment.get_times_kwargs())
        self._parent_segment = parent_segment
        self._stim_indices = stim_indices
        self._pre = pre_samples
        self._post = post_samples
        self._mode = mode
        self._n_samples = parent_segment.get_num_samples() # Cache
        
    def get_num_samples(self):
        return self._parent_segment.get_num_samples()
        
    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self._parent_segment.get_traces(start_frame, end_frame, channel_indices)
        traces = traces.copy() # Make writable
        
        # Search range
        if len(self._stim_indices) == 0:
            return traces
            
        # We need to consider blanking window
        pad = self._pre + self._post
        i_start = np.searchsorted(self._stim_indices, start_frame - pad) 
        i_end = np.searchsorted(self._stim_indices, end_frame + pad)
        
        relevant = self._stim_indices[i_start:i_end]
        n_chunk = traces.shape[0]
        
        blank_duration = self._pre + self._post
        
        for t in relevant:
            # t is absolute sample of stim
            rel_t = int(t - start_frame)
            
            # Blanking Window Indices (Relative to chunk start)
            idx_L = rel_t - self._pre
            idx_R = rel_t + self._post
            
            # Determine overlap with current chunk
            fill_start = max(0, idx_L)
            fill_end = min(n_chunk, idx_R)
            
            if fill_start >= fill_end:
                 continue
                 
            # --- Logic per Mode ---
            if self._mode == 'copy_baseline':
                # Source Window (Absolute)
                # Ideally: [blank_start - duration, blank_start]
                abs_blank_start = t - self._pre
                abs_src_start = abs_blank_start - blank_duration
                abs_src_end = abs_blank_start
                
                # We need to map the 'fill' region (which is a subset of blanking window)
                # to the valid source region.
                # fill_start is relative index in chunk.
                # offset from blank_start = fill_start - idx_L
                offset = fill_start - idx_L
                length = fill_end - fill_start
                
                s_req_start = abs_src_start + offset
                s_req_end = s_req_start + length
                
                # Fetch baseline data (Source)
                # Handle edge case: src < 0
                if s_req_start < 0:
                     # Partial fetch not easily supported, just zero out?
                     # Or fetch what we can.
                     if s_req_end <= 0:
                          traces[fill_start:fill_end] = 0
                          continue
                     else:
                          # Split zero and valid
                          valid_len = s_req_end
                          traces[fill_start:fill_start + (length-valid_len)] = 0
                          try:
                              bsl = self._parent_segment.get_traces(0, s_req_end, channel_indices)
                              traces[fill_end-valid_len:fill_end] = bsl
                          except:
                              pass
                          continue

                try:
                    baseline = self._parent_segment.get_traces(s_req_start, s_req_end, channel_indices)
                    if baseline.shape[0] == length:
                         traces[fill_start:fill_end] = baseline
                except Exception:
                    pass # Safety
                continue

            elif self._mode == 'zero':
                 traces[fill_start:fill_end] = 0.0
                 continue
            
            else:
                # Default: Linear Interpolation ('interp')
                # Determine Anchor Values (Edge of blanking)
                # If idx_L is within chunk, use it. If < 0, use trace[0] (best effort)
                
                # We need one sample BEFORE the fill start effectively for slope?
                # Actually standard linear interp uses idx_L and idx_R values as anchors.
                
                # Anchor L
                if idx_L >= 0 and idx_L < n_chunk:
                    val_L = traces[idx_L]
                else:
                    # If anchor is outside, we can't easily get it without fetching.
                    # Fallback to nearest in chunk
                    val_L = traces[0]
                    
                # Anchor R
                if idx_R >= 0 and idx_R < n_chunk:
                    val_R = traces[idx_R]
                else:
                    val_R = traces[-1]
                
                width_total = idx_R - idx_L
                if width_total == 0: continue
                
                slope = (val_R - val_L) / width_total # Shape (N_ch,)
                
                # X coordinates for the fill region
                x_vals = np.arange(fill_start, fill_end) - idx_L
                
                # Vectorized fill
                traces[fill_start:fill_end, :] = val_L[None, :] + x_vals[:, None] * slope[None, :]
            
        return traces


# --- Time Reversal Wrapper for Backward Filtering ---
class TimeReversedRecordingSegment(si.BaseRecordingSegment):
    def __init__(self, parent_segment):
        si.BaseRecordingSegment.__init__(self, **parent_segment.get_times_kwargs())
        self._parent_segment = parent_segment
        self._n_samples = parent_segment.get_num_samples()

    def get_num_samples(self):
        return self._n_samples

    def get_traces(self, start_frame, end_frame, channel_indices):
        # Time Reversal Logic:
        # Requesting [start, end) in reversed time
        # corresponds to [N - end, N - start) in original time
        # Then flip the result.
        
        N = self._n_samples
        if start_frame is None: start_frame = 0
        if end_frame is None: end_frame = N
        
        # Original indices
        orig_start = N - end_frame
        orig_end = N - start_frame
        
        # Handle negatives/bounds if caller requests out of bounds (though SI handles bounds usually)
        # Assuming valid inputs or handling by parent
        
        traces = self._parent_segment.get_traces(orig_start, orig_end, channel_indices)
        
        # Flip along time axis (axis 0)
        return traces[::-1, :]

class TimeReversedRecording(si.BaseRecording):
    def __init__(self, parent_recording):
        si.BaseRecording.__init__(self, parent_recording.get_sampling_frequency(), 
                               parent_recording.get_channel_ids(), 
                               parent_recording.get_dtype())
        
        parent_recording.copy_metadata(self)
        self._parent = parent_recording
        
        for i in range(parent_recording.get_num_segments()):
            self.add_recording_segment(TimeReversedRecordingSegment(
                parent_recording._recording_segments[i]
            ))
            
        self._kwargs = {'parent_recording': parent_recording.to_dict()}

# --- Debug Plotting ---
def build_lfp_preprocessing_pipeline(
    recording: si.BaseRecording,
    stim_indices_native: np.ndarray,
    curr_sess_name: str = "unknown",
    local_radii: tuple | None = None) -> si.BaseRecording:
    """
    Custom pipeline with Manual Blanking Class.
    
    Args:
        local_radii: If set, use local CMR with these radii (inner, outer). 
                     If None, skip CMR (for Utah).
    """
    # 1. Blanking (Custom) - On RAW Data (30kHz)
    if len(stim_indices_native) > 0:
        print(f"    [SI Pipeline] Custom Blanking (Copy Baseline) ({len(stim_indices_native)} events, -{BLANK_PRE_MS}ms to +{BLANK_POST_MS}ms) on RAW...")
        rec_blanked = BlankingRecording(
            recording,
            stim_indices_native,
            pre_ms=BLANK_PRE_MS,
            post_ms=BLANK_POST_MS,
            mode='copy_baseline'
        )
    else:
        rec_blanked = recording
        
    # 2. CMR (Local for Intan, None for Utah)
    if local_radii is not None:
        r_in, r_out = local_radii
        print(f"    [SI Pipeline] Local CMR (Median, radius {r_in}-{r_out} um) ENABLED...")
        rec_cmr = spre.common_reference(rec_blanked, reference='local', local_radius=(r_in, r_out), operator='median')
        
    else:
        print("    [SI Pipeline] CMR Skipped (Utah or Config)...")
        rec_cmr = rec_blanked
        
    # 3. Resampling (30k -> 1k)
    # SI resample includes anti-aliasing filter
    print(f"    [SI Pipeline] Resampling to {TARGET_FS} Hz...")
    rec_resampled = spre.resample(rec_cmr, resample_rate=TARGET_FS)
    
    rec_clean = rec_resampled
    
    return rec_clean
    
    rec_clean = rec_resampled
    
    return rec_clean


def extract_single_epoch(rec_obj, center_idx, pre_samps, post_samps):
    """
    Extract a single epoch from a recording object.
    
    Args:
        rec_obj: SpikeInterface recording object
        center_idx: Center sample index (typically stim onset)
        pre_samps: Samples before center
        post_samps: Samples after center
    
    Returns:
        np.ndarray of shape (n_ch, n_time) or None if out of bounds
    """
    start = center_idx - pre_samps
    end = center_idx + post_samps
    
    # Check bounds
    if start < 0 or end > rec_obj.get_num_samples():
        return None
        
    # Get trace (n_samples, n_ch) -> Transpose to (n_ch, n_samples)
    return rec_obj.get_traces(start_frame=start, end_frame=end, return_in_uV=True).T


def slice_epoch(data, t_axis, target_win):
    """
    Slice epoch data to a specific time window.
    
    Args:
        data: np.ndarray of shape (n_trials, n_ch, n_time)
        t_axis: Time axis array
        target_win: Tuple of (start_ms, end_ms)
    
    Returns:
        Tuple of (sliced_data, sliced_time_axis)
    """
    mask = (t_axis >= target_win[0]) & (t_axis < target_win[1])
    return data[:, :, mask], t_axis[mask]


def process_nprw_session(sess_name: str, br_idx: int, intan_idx: int, shift_ms: float = 0.0):
    """
    Process one Intan session: lazy load, SI clean/filter/DS, epoch.
    """
    # Identify session folder
    matches = list(INTAN_ROOT.glob(f"*{sess_name}*"))
    valid_folders = [p for p in matches if p.is_dir()]
    if not valid_folders:
        print(f"[NPRW] Session folder not found for {sess_name}")
        return
    sess_path = valid_folders[0] # Take first match
    
    out_npz = NPRW_LFP_CKPT_ROOT / f"aligned_lfp__{sess_name}.npz"
    if out_npz.exists():
        print(f"[NPRW] Skipping {sess_name}, output exists.")
        return
        
    print(f"[NPRW] Processing {sess_name}")

    # Geometry
    intan_geom, intan_probe_mapping = get_intan_geometry()
    
    nprw_probe = Probe(ndim=2)
    nprw_probe.set_contacts(positions=np.c_[intan_geom["x"], intan_geom["y"]], shape_params={"width": 12.0})
    nprw_probe.set_device_channel_indices(intan_probe_mapping)
    
    # 2. Load Raw Recording (Lazy)
    rec = se.read_split_intan_files(sess_path, mode="concatenate", stream_name=INTAN_STREAM, use_names_as_ids=True)
    rec = spre.unsigned_to_signed(rec)
    rec_reordered = rcp.reorder_recording_to_geometry(rec, intan_probe_mapping)
    rec_reordered = rec_reordered.set_probe(nprw_probe)
    
    fs_native = rec_reordered.get_sampling_frequency()

    # 1. Load Resulting Stim Times
    stim = {}
    stim_start_samps_native = np.array([])
    stim_channels_meta = ["Stim_Trigger"]

    # Try loading from ALIGNED file first (as requested)
    aligned_path = ALIGNED_CKPT_ROOT / f"aligned__{sess_name}__Intan_{intan_idx:03d}__BR_{br_idx:03d}.npz"
    
    if aligned_path.exists():
        print(f"    [Stim] Loading from aligned file: {aligned_path.name}")
        try:
             aln = np.load(aligned_path, allow_pickle=True)
             if "stim_ms" in aln:
                 stim_ms_aligned = aln["stim_ms"]
                 if stim_ms_aligned.size > 0:
                      # Convert back to native samples
                      # stim_ms in aligned file is (sample / fs * 1000 - shift)
                      # So: sample = (ms + shift) * fs / 1000
                      stim_start_samps_native = ((stim_ms_aligned + shift_ms) * fs_native / 1000.0).astype(np.int64)
                      print(f"    [Stim] Found {len(stim_start_samps_native)} events in aligned file.")
                      stim_channels_meta = ["Aligned_Stim_MS"]
                      
                      # Metadata
                      if 'stim_channels' in aln: stim['stim_channels'] = aln['stim_channels']
                      if 'stim_amplitudes' in aln: stim['amplitudes'] = aln['stim_amplitudes']
                      if 'amplitudes' in aln: stim['amplitudes'] = aln['amplitudes']
                      if 'notes' in aln: stim['notes'] = aln['notes']
        except Exception as e:
            print(f"    [WARN] Failed to load aligned file: {e}")

    # Fallback to standard Stim stream if aligned failed
    # Fallback to standard Stim stream if aligned failed
    if len(stim_start_samps_native) == 0:
        stim_npz_dir = NPRW_AUX_DATA / f"{sess_path.name}_Intan_streams"
        stim_npz_path = stim_npz_dir / "stim_stream.npz"
        
        if stim_npz_path.exists():
            print(f"    [Stim] Loading existing stim file: {stim_npz_path.name}")
            # Use existing function or load npz directly
            try:
                 stim_data = rcp.load_stim_detection(stim_npz_path)
                 stim = stim_data # Assign for metadata usage later
                 # stim_data is dict, block_bounds_samples is key
                 block_bounds = stim_data.get("block_bounds_samples")
            except Exception as e:
                 print(f"    [Stim] Error loading existing file ({e}), re-extracting...")
                 stim_ext_arrays = rcp.extract_stim_npz(sess=sess_path, out_dir=NPRW_AUX_DATA, stim_stream_name=STIM_STREAM, chanmap_perm=intan_probe_mapping)
                 stim = stim_ext_arrays # Assign for metadata
                 block_bounds = stim_ext_arrays.get("block_bounds_samples")
        else:
             print(f"    [Stim] Extracting stim stream (first run)...")
             stim_ext_arrays = rcp.extract_stim_npz(sess=sess_path, out_dir=NPRW_AUX_DATA, stim_stream_name=STIM_STREAM, chanmap_perm=intan_probe_mapping)
             stim = stim_ext_arrays # Assign for metadata
             block_bounds = stim_ext_arrays.get("block_bounds_samples")

        stim_start_samps_native = block_bounds[:, 0] if (block_bounds is not None and block_bounds.size) else np.array([])
    
    # -------------------------------------------------------------------------
    # [Control/IR Fallback]
    # If no stim triggers found, load IR events from baseline PeriStim files
    # -------------------------------------------------------------------------
    stim_channels_meta = ["Stim_Trigger"]
    if len(stim_start_samps_native) == 0:
        print(f"    [Stim] No standard stim triggers found. Checking for baseline PeriStim files...")
        
        for target in ["Target_A", "Target_B"]:
            if len(stim_start_samps_native) > 0:
                break  # Already found
                
            target_dir = PERI_ROOT / target
            if not target_dir.exists():
                continue
                
            # Pattern: baseline__{session}*_target_{A,B}.npz
            pattern = f"baseline__*{sess_name}*_target_{target[-1]}.npz"
            matches = list(target_dir.glob(pattern))
            
            if matches:
                baseline_file = matches[0]
                print(f"    [Stim] Loading IR from baseline file: {baseline_file.name}")
                try:
                    bl_data = np.load(baseline_file, allow_pickle=True)
                    if "stim_ms" in bl_data and bl_data["stim_ms"].size > 0:
                        stim_ms_baseline = bl_data["stim_ms"]
                        # Convert ms back to native samples
                        # stim_ms in baseline file is in BR time (aligned)
                        # So: sample = (ms + shift) * fs / 1000
                        stim_start_samps_native = ((stim_ms_baseline + shift_ms) * fs_native / 1000.0).astype(np.int64)
                        stim_channels_meta = ["Baseline_IR"]
                        print(f"    [Stim] Found {len(stim_start_samps_native)} IR events from baseline file.")
                except Exception as e:
                    print(f"    [WARN] Failed to load baseline file: {e}")
        
        if len(stim_start_samps_native) == 0:
            print(f"    [WARN] No IR events found in baseline files for {sess_name}.")

    
    # (Raw Loading moved up)
    # rec = se.read_split_intan_files(...) 
    # already done above to get fs_native for conversion
    
    
    # 3. Build SI Pipeline
    # Blank(-5, +20) -> Local CMR -> DS(1000)
    rec_proc = build_lfp_preprocessing_pipeline(
        rec_reordered, 
        stim_indices_native=stim_start_samps_native,
        curr_sess_name=sess_name,
        local_radii=RADII
    )
    
    # 4. Get Traces (Parallel Epoch Extraction)
    # Be sure to extract PADDED epochs for zero-phase filtering
    print("  [NPRW] Extracting PADDED epochs directly from Lazy Recording (Parallel)...")
    
    # We need stim indices in the TARGET_FS domain
    # stim_start_samps_native are in fs_native (30000)
    # The pipeline resamples to TARGET_FS (typically 1000)
    stim_indices_res = (stim_start_samps_native * TARGET_FS / fs_native).astype(int)
    
    # Define generic parallel extractor
    def extract_single_epoch(rec_obj, center_idx, pre_samps, post_samps):
        start = center_idx - pre_samps
        end = center_idx + post_samps
        
        # Check bounds
        if start < 0 or end > rec_obj.get_num_samples():
            return None # Skip
            
        # return_in_uV=True handles scaling
        return rec_obj.get_traces(start_frame=start, end_frame=end, return_in_uV=True).T

    pad_samps = int(PAD_MS * TARGET_FS / 1000.0)
    pre_samps = int(EPOCH_PRE_MS * TARGET_FS / 1000.0) + pad_samps
    post_samps = int(EPOCH_POST_MS * TARGET_FS / 1000.0) + pad_samps
    
    exclusion_list = MANUAL_EXCLUSION.get(sess_name, [])
    if exclusion_list:
        print(f"  [NPRW] Excluding {len(exclusion_list)} trials manually: {exclusion_list}")

    segs_list = Parallel(n_jobs=-int(os.cpu_count() or 1), batch_size='auto')(
        delayed(extract_single_epoch)(rec_proc, idx, pre_samps, post_samps) 
        for i, idx in enumerate(stim_indices_res)
        if i not in exclusion_list
    )
    
    # Filter out Nones
    segs_list = [s for s in segs_list if s is not None]
    
    if not segs_list:
        print("  [Warn] No valid epochs found.")
        return

    segs_bb = np.stack(segs_list, axis=0) # (n_trials, n_ch, n_time)
    n_trials, n_ch, n_time = segs_bb.shape
    
    # Construct rel_t_epoch (With padding included)
    rel_t_epoch = (np.arange(n_time) / TARGET_FS * 1000.0) - (EPOCH_PRE_MS + PAD_MS)
    
    print(f"  [NPRW] Extracted {n_trials} epochs. Shape: {segs_bb.shape}")

    # Stim times to ms (just for meta)
    stim_ms = stim_start_samps_native * 1000.0 / fs_native
    
    # 6. Epoching: Extract Broadband (-500 to +1000)
    # Already done above!
    WIN_EPOCH = (-EPOCH_PRE_MS, EPOCH_POST_MS)
    
    # Sub-windows for final saving
    WIN_PRE = (-EPOCH_PRE_MS, -BLANK_PRE_MS)
    WIN_POST = (BLANK_POST_MS, EPOCH_POST_MS - BLANK_PRE_MS) # 500ms post-blank
    
    # Valid stim times
    valid_stim = stim_ms 
    
    results = {}
    
    if len(segs_bb) == 0:
        print(f"  [WARN] No valid segments found for {sess_name}. Skipping.")
        return
        
    # 7. Per-Trial Cleaning - DISABLED (Baseline/Exp Removal)
    print(f"  [Cleaning] Using RAW (resampled) epochs (Baseline/Exp/Template removal DISABLED).")
    segs_bb_clean = segs_bb # Use padded, raw
    
    # --- Step 7.5: Reject Bad Channels ---
    # Convert Channel IDs to Indices
    chan_ids = rec_proc.get_channel_ids()
    bad_ch_indices = []

    # 1. Impedance Check (ONLY)
    bad_names_imp = get_bad_channels_intan(sess_path, threshold_kohm=INTAN_IMPEDANCE_CUTOFF)
    if bad_names_imp:
         print(f"  [Rejection] Found {len(bad_names_imp)} bad channels from impedance file.")
         # map to index
         for name in bad_names_imp:
             # Assume channel IDs match names?
             # When use_names_as_ids=True, IDs are strings.
             idx = np.where(chan_ids == name)[0]
             if len(idx) > 0:
                 bad_ch_indices.extend(idx)
    
    # Combined Bad Channels
    bad_ch_indices = np.array(bad_ch_indices, dtype=int)
    bad_ch_indices = np.unique(bad_ch_indices)
    
    if len(bad_ch_indices) > 0:
         print(f"  [Rejection] Removing {len(bad_ch_indices)} bad channels (Impedance > 2000kOhm or file-spec).")
         segs_bb_clean[:, bad_ch_indices, :] = np.nan
    
    results['bad_channels'] = bad_ch_indices
    
    # Helper to slice extracted epoch into sub-windows
    def slice_epoch(data, t_axis, target_win):
        # find indices in t_axis
        mask = (t_axis >= target_win[0]) & (t_axis < target_win[1])
        return data[:, :, mask], t_axis[mask]
        
    # Slice Broadband Results (Crop Padding)
    results['broadband_pre'], t_pre = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_PRE)
    results['broadband_post'], t_post = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_POST)
    
    results['broadband_pre'], t_pre = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_PRE)
    results['broadband_post'], t_post = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_POST)
    
    # 9. Band Analysis (Zero-Phase Filter Cleaned Epochs)
    for band_name, (low, high) in LFP_BANDS.items():
        print(f"  Filtering Epochs {band_name} ({low}-{high} Hz)...", flush=True)
        # Apply filter to the FULL PADDED epochs
        filt_epochs = filter_zero_phase(segs_bb_clean, TARGET_FS, low, high)
        
        # Slice (Crop Padding)
        results[f'{band_name}_pre'], _ = slice_epoch(filt_epochs, rel_t_epoch, WIN_PRE)
        results[f'{band_name}_post'], _ = slice_epoch(filt_epochs, rel_t_epoch, WIN_POST)
    
    # Save
    save_dict = {
        "fs_lfp": TARGET_FS,
        "channel_geom_x": intan_geom["x"],
        "channel_geom_y": intan_geom["y"],
        "session": sess_name,
        "stim_ms": valid_stim, # Original indices relative to full recording
        "rel_time_pre": t_pre,
        "rel_time_post": t_post,
        # Metadata
        "stim_channels": stim_channels_meta if stim_channels_meta == ["Control_IR"] else stim.get('stim_channels', stim_channels_meta),
        "stim_amplitudes": stim.get('amplitudes', []),
        "stim_notes": stim.get('notes', ""),
        **results
    }
    
    print(f"  [DEBUG] Keys to save: {list(save_dict.keys())}", flush=True)
    
    np.savez_compressed(out_npz, **save_dict)
    print(f"  Saved -> {out_npz}")
    
    # Cleanup memory
    try:
        del rec, rec_reordered, rec_proc, results
        if 'traces' in locals(): del traces
    except:
        pass
    gc.collect()


# =============================================================================
# Grouped Baseline LFP Processing
# =============================================================================

def process_baseline_group(
    group_key: tuple,  # (port, depth)
    session_infos: list,  # List of dicts with session info
):
    """
    Process a group of baseline/control sessions by loading IR events from
    existing baseline PeriStim files and concatenating LFP epochs.
    
    Args:
        group_key: Tuple of (ua_port, depth_mm)
        session_infos: List of dicts with keys: session, br_idx, intan_idx, shift_ms
    """
    port, depth = group_key
    
    # Sanitize for filename
    def _sanitize(s):
        return str(s).replace(".", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
    
    depth_str = _sanitize(depth)
    port_str = _sanitize(port)
    
    out_npz = NPRW_LFP_CKPT_ROOT / f"aligned_lfp__baseline__Depth_{depth_str}_port_{port_str}.npz"
    
    if out_npz.exists():
        print(f"[Baseline LFP] Skipping group Depth={depth}, Port={port}, output exists.")
        return
        
    print(f"\n[Baseline LFP] Processing group: Depth={depth}, Port={port}")
    print(f"    Sessions: {[s['session'] for s in session_infos]}")
    
    # 1. Load IR events from baseline PeriStim file
    # Try Target_A first, then Target_B
    # Use glob to handle different naming conventions (e.g., 43.0 vs 43_0)
    ir_events_ms = None
    loaded_target = None  # Track which target was loaded
    for target in ["Target_A", "Target_B"]:
        target_dir = PERI_ROOT / target
        if not target_dir.exists():
            continue
            
        # Use flexible glob pattern for depth (handles 43.0 and 43_0)
        # Pattern: baseline__*_Depth_*{depth}*_port_{port}_target_{A,B}.npz
        pattern = f"baseline__*Depth*{depth.replace('.', '*')}*port*{port}*target_{target[-1]}.npz"
        matches = list(target_dir.glob(pattern))
        
        if matches:
            baseline_file = matches[0]
            print(f"    Loading IR events from: {baseline_file.name}")
            try:
                bl_data = np.load(baseline_file, allow_pickle=True)
                if "stim_ms" in bl_data and bl_data["stim_ms"].size > 0:
                    ir_events_ms = bl_data["stim_ms"]
                    loaded_target = target[-1]  # 'A' or 'B'
                    print(f"    Found {len(ir_events_ms)} IR events in baseline file (Target {loaded_target}).")
                    break
            except Exception as e:
                print(f"    [WARN] Failed to load baseline file: {e}")
    
    if ir_events_ms is None or len(ir_events_ms) == 0:
        print(f"    [WARN] No IR events found for group. Skipping.")
        return
    
    # 1.5. Apply Baseline Trial Exclusions
    exclusion_key = f"baseline_port{port}_target{loaded_target}"
    exclude_indices = EXCLUDE_BASELINE_TRIALS.get(exclusion_key, [])
    if exclude_indices:
        print(f"    [Exclusion] Removing {len(exclude_indices)} trials from {exclusion_key}: {exclude_indices}")
        # Create mask for valid trials
        valid_mask = np.ones(len(ir_events_ms), dtype=bool)
        for idx in exclude_indices:
            if 0 <= idx < len(ir_events_ms):
                valid_mask[idx] = False
        ir_events_ms = ir_events_ms[valid_mask]
        print(f"    [Exclusion] Remaining trials: {len(ir_events_ms)}")
    
    # 2. Set up probe geometry
    intan_geom, intan_probe_mapping = get_intan_geometry()
    
    nprw_probe = Probe(ndim=2)
    nprw_probe.set_contacts(positions=np.c_[intan_geom["x"], intan_geom["y"]], shape_params={"width": 12.0})
    nprw_probe.set_device_channel_indices(intan_probe_mapping)
    
    # 3. Process each session in the group
    all_epochs_bb = []
    all_epochs_raw = []
    
    for sess_info in session_infos:
        sess_name = sess_info["session"]
        shift_ms = sess_info["shift_ms"]
        
        # Find session folder
        matches = list(INTAN_ROOT.glob(f"*{sess_name}*"))
        valid_folders = [p for p in matches if p.is_dir()]
        if not valid_folders:
            print(f"    [WARN] Session folder not found: {sess_name}")
            continue
            
        sess_path = valid_folders[0]
        print(f"    Processing session: {sess_name}")
        
        try:
            # Load raw recording
            rec = se.read_split_intan_files(sess_path, mode="concatenate", stream_name=INTAN_STREAM, use_names_as_ids=True)
            rec = spre.unsigned_to_signed(rec)
            rec_reordered = rcp.reorder_recording_to_geometry(rec, intan_probe_mapping)
            rec_reordered = rec_reordered.set_probe(nprw_probe)
            
            fs_native = rec_reordered.get_sampling_frequency()
            rec_duration_ms = rec_reordered.get_total_duration() * 1000.0
            
            # Filter IR events for this session's recording duration
            # IR events are in BR-aligned ms, convert back to session time
            sess_ir_ms = ir_events_ms + shift_ms  # Add shift to get Intan time
            
            # Only keep events within this recording's duration
            valid_mask = (sess_ir_ms >= 0) & (sess_ir_ms < rec_duration_ms)
            sess_ir_ms_valid = sess_ir_ms[valid_mask]
            
            if len(sess_ir_ms_valid) == 0:
                print(f"        No valid IR events for this session. Skipping.")
                continue
                
            print(f"        Found {len(sess_ir_ms_valid)} valid IR events for this session.")
            
            # Convert to samples
            stim_start_samps = (sess_ir_ms_valid * fs_native / 1000.0).astype(np.int64)
            
            # Build preprocessing pipeline (no blanking for baseline)
            rec_proc = build_lfp_preprocessing_pipeline(
                rec_reordered,
                stim_indices_native=np.array([]),  # No blanking for control
                curr_sess_name=sess_name,
                local_radii=RADII
            )
            
            # Convert to resampled indices
            stim_indices_res = (stim_start_samps * TARGET_FS / fs_native).astype(np.int64)
            
            # Calculate epoch samples (PADDED)
            pad_samps = int(PAD_MS * TARGET_FS / 1000.0)
            pre_samps = int(EPOCH_PRE_MS * TARGET_FS / 1000) + pad_samps
            post_samps = int(EPOCH_POST_MS * TARGET_FS / 1000) + pad_samps
            
            n_samps_total = rec_proc.get_total_samples()
            
            # Extract epochs (sequential to avoid joblib pickling issues)
            segs_list = []
            for idx in stim_indices_res:
                if (idx - pre_samps >= 0) and (idx + post_samps <= n_samps_total):
                    seg = extract_single_epoch(rec_proc, idx, pre_samps, post_samps)
                    if seg is not None:
                        segs_list.append(seg)
            
            if len(segs_list) > 0:
                sess_epochs = np.stack(segs_list, axis=0)
                
                # --- Reject Bad Channels (Impedance Only) ---
                bad_names_imp = get_bad_channels_intan(sess_path)
                if bad_names_imp:
                     chan_ids = rec_proc.get_channel_ids()
                     bad_ch_indices = []
                     for name in bad_names_imp:
                         idx = np.where(chan_ids == name)[0]
                         if len(idx) > 0:
                             bad_ch_indices.extend(idx)
                     
                     if bad_ch_indices:
                         print(f"        [Rejection] Masking {len(bad_ch_indices)} bad channels (Imp).")
                         bad_ch_indices = np.unique(bad_ch_indices)
                         sess_epochs[:, bad_ch_indices, :] = np.nan
                
                all_epochs_bb.append(sess_epochs)
                print(f"        Extracted {len(segs_list)} epochs from session.")
            else:
                print(f"        [WARN] No valid epochs extracted from session.")
                
            del rec, rec_reordered, rec_proc
            gc.collect()
            
        except Exception as e:
            print(f"        [ERROR] Failed to process session {sess_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 4. Concatenate epochs across sessions
    if len(all_epochs_bb) == 0:
        print(f"    [WARN] No epochs extracted from any session in group. Skipping.")
        return
        
    concat_epochs = np.concatenate(all_epochs_bb, axis=0)
    print(f"    Concatenated epochs: {concat_epochs.shape} (trials, channels, time)")
    
    # 5. Clean epochs (DISABLED)
    print(f"    [Cleaning] Using RAW (resampled) epochs (Baseline/Exp/Template removal DISABLED).")
    concat_epochs_clean = concat_epochs # Use padded, raw
    
    # 6. Compute time vectors and slice
    # rel_t_epoch needs to account for padding
    n_time = concat_epochs_clean.shape[2]
    rel_t_epoch = (np.arange(n_time) / TARGET_FS * 1000.0) - (EPOCH_PRE_MS + PAD_MS)
    
    WIN_PRE = (-EPOCH_PRE_MS, -BLANK_PRE_MS)
    WIN_POST = (BLANK_POST_MS, EPOCH_POST_MS - BLANK_PRE_MS)
    
    bb_pre, t_pre = slice_epoch(concat_epochs_clean, rel_t_epoch, WIN_PRE)
    bb_post, t_post = slice_epoch(concat_epochs_clean, rel_t_epoch, WIN_POST)
    
    results = {
        "broadband_pre": bb_pre,
        "broadband_post": bb_post,
    }
    
    # 7. Band analysis
    for band_name, (low, high) in LFP_BANDS.items():
        print(f"    Filtering {band_name} ({low}-{high} Hz)...")
        filt_epochs = filter_zero_phase(concat_epochs_clean, TARGET_FS, low, high)
        results[f'{band_name}_pre'], _ = slice_epoch(filt_epochs, rel_t_epoch, WIN_PRE)
        results[f'{band_name}_post'], _ = slice_epoch(filt_epochs, rel_t_epoch, WIN_POST)
    
    # 8. Save
    save_dict = {
        "fs_lfp": TARGET_FS,
        "channel_geom_x": intan_geom["x"],
        "channel_geom_y": intan_geom["y"],
        "group_port": port,
        "group_depth": depth,
        "sessions": [s["session"] for s in session_infos],
        "stim_ms": ir_events_ms,  # All IR events from baseline file
        "n_trials": concat_epochs_clean.shape[0],
        "rel_time_pre": t_pre,
        "rel_time_post": t_post,
        "stim_channels": ["Baseline_IR"],
        **results
    }
    
    np.savez_compressed(out_npz, **save_dict)
    print(f"    Saved -> {out_npz}")
    
    gc.collect()


# =============================================================================
# Grouped Baseline LFP Processing - Utah Array
# =============================================================================

def process_baseline_group_utah(
    group_key: tuple,  # (port, depth)
    session_infos: list,  # List of dicts with session info
):
    """
    Process a group of baseline/control sessions for Utah Array by loading IR events from
    existing baseline PeriStim files and concatenating LFP epochs.
    
    Args:
        group_key: Tuple of (ua_port, depth_mm)
        session_infos: List of dicts with keys: session, br_idx, intan_idx, shift_ms, fs_intan
    """
    port, depth = group_key
    
    # Sanitize for filename
    def _sanitize(s):
        return str(s).replace(".", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
    
    depth_str = _sanitize(depth)
    port_str = _sanitize(port)
    
    out_npz = UA_LFP_CKPT_ROOT / f"aligned_lfp__baseline__Depth_{depth_str}_port_{port_str}.npz"
    
    if out_npz.exists():
        print(f"[Baseline UA] Skipping group Depth={depth}, Port={port}, output exists.")
        return
        
    print(f"\n[Baseline UA] Processing group: Depth={depth}, Port={port}")
    print(f"    Sessions: {[s['session'] for s in session_infos]}")
    
    # 1. Load IR events from baseline PeriStim file
    ir_events_ms = None
    loaded_target = None  # Track which target was loaded
    for target in ["Target_A", "Target_B"]:
        target_dir = PERI_ROOT / target
        if not target_dir.exists():
            continue
            
        # Use flexible glob pattern for depth
        pattern = f"baseline__*Depth*{depth.replace('.', '*')}*port*{port}*target_{target[-1]}.npz"
        matches = list(target_dir.glob(pattern))
        
        if matches:
            baseline_file = matches[0]
            print(f"    Loading IR events from: {baseline_file.name}")
            try:
                bl_data = np.load(baseline_file, allow_pickle=True)
                if "stim_ms" in bl_data and bl_data["stim_ms"].size > 0:
                    ir_events_ms = bl_data["stim_ms"]
                    loaded_target = target[-1]  # 'A' or 'B'
                    print(f"    Found {len(ir_events_ms)} IR events in baseline file (Target {loaded_target}).")
                    break
            except Exception as e:
                print(f"    [WARN] Failed to load baseline file: {e}")
    
    if ir_events_ms is None or len(ir_events_ms) == 0:
        print(f"    [WARN] No IR events found for group. Skipping.")
        return
    
    # 1.5. Apply Baseline Trial Exclusions
    exclusion_key = f"baseline_port{port}_target{loaded_target}"
    exclude_indices = EXCLUDE_BASELINE_TRIALS.get(exclusion_key, [])
    if exclude_indices:
        print(f"    [Exclusion] Removing {len(exclude_indices)} trials from {exclusion_key}: {exclude_indices}")
        valid_mask = np.ones(len(ir_events_ms), dtype=bool)
        for idx in exclude_indices:
            if 0 <= idx < len(ir_events_ms):
                valid_mask[idx] = False
        ir_events_ms = ir_events_ms[valid_mask]
        print(f"    [Exclusion] Remaining trials: {len(ir_events_ms)}")
    
    # 2. Process each session in the group
    all_epochs_bb = []
    ua_region_global = None
    ua_region_names_global = None
    ua_elec_global = None
    ua_nsp_global = None
    
    for sess_info in session_infos:
        br_idx = sess_info["br_idx"]
        shift_ms = sess_info["shift_ms"]
        fs_intan = sess_info.get("fs_intan", 30000.0)
        
        # Find Blackrock ns6 file
        ns6_files = [p for p in BR_ROOT.iterdir() if p.is_file() and f"_{br_idx:03d}" in p.name and p.suffix == '.ns6']
        
        if not ns6_files:
            print(f"    [WARN] No .ns6 file found for BR {br_idx:03d}")
            continue
            
        sess_path = ns6_files[0]
        print(f"    Processing BR {br_idx:03d}: {sess_path.name}")
        
        try:
            # Load Blackrock recording
            rec_raw = se.read_blackrock(str(sess_path), stream_name='nsx6', all_annotations=True)
            
            # Apply UA mapping
            rec_mapped, idx_rows, ua_elec, ua_nsp, ua_region, ua_region_names, ua_port_from_map = rcp.apply_ua_mapping_with_regions(
                rec_raw, UA_MAP, br_idx, METADATA_CSV
            )
            
            # Store region info from first session
            if ua_region_global is None:
                ua_region_global = ua_region
                ua_region_names_global = ua_region_names
                ua_elec_global = ua_elec
                ua_nsp_global = ua_nsp
            
            fs_native = rec_mapped.get_sampling_frequency()
            rec_duration_ms = rec_mapped.get_total_duration() * 1000.0
            
            # Convert IR events (in BR-aligned ms) to session time
            # IR events are relative to BR time, shift_ms converts Intan->BR
            # So for BR time we don't need to shift, IR events are already in BR time
            sess_ir_ms_valid = ir_events_ms[(ir_events_ms >= 0) & (ir_events_ms < rec_duration_ms)]
            
            if len(sess_ir_ms_valid) == 0:
                print(f"        No valid IR events for this session. Skipping.")
                continue
                
            print(f"        Found {len(sess_ir_ms_valid)} valid IR events for this session.")
            
            # Convert to samples
            stim_start_samps = (sess_ir_ms_valid * fs_native / 1000.0).astype(np.int64)
            
            # Build preprocessing pipeline (no blanking for baseline)
            # local_radii=None (default) -> Skips CMR which is correct for Utah per user request
            rec_proc = build_lfp_preprocessing_pipeline(
                rec_mapped,
                stim_indices_native=np.array([]),  # No blanking for control
                curr_sess_name=sess_path.name
            )
            
            # Convert to resampled indices
            stim_indices_res = (stim_start_samps * TARGET_FS / fs_native).astype(np.int64)
            
            # Calculate epoch samples (PADDED)
            pad_samps = int(PAD_MS * TARGET_FS / 1000.0)
            pre_samps = int(EPOCH_PRE_MS * TARGET_FS / 1000) + pad_samps
            post_samps = int(EPOCH_POST_MS * TARGET_FS / 1000) + pad_samps
            
            n_samps_total = rec_proc.get_total_samples()
            
            # Extract epochs (sequential to avoid joblib pickling issues)
            segs_list = []
            for idx in stim_indices_res:
                if (idx - pre_samps >= 0) and (idx + post_samps <= n_samps_total):
                    seg = extract_single_epoch(rec_proc, idx, pre_samps, post_samps)
                    if seg is not None:
                        segs_list.append(seg)
            
            if len(segs_list) > 0:
                sess_epochs = np.stack(segs_list, axis=0)
                all_epochs_bb.append(sess_epochs)
                print(f"        Extracted {len(segs_list)} epochs from session.")
            else:
                print(f"        [WARN] No valid epochs extracted from session.")
                
            del rec_raw, rec_mapped, rec_proc
            gc.collect()
            
        except Exception as e:
            print(f"        [ERROR] Failed to process BR {br_idx:03d}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 3. Concatenate epochs across sessions
    if len(all_epochs_bb) == 0:
        print(f"    [WARN] No epochs extracted from any session in group. Skipping.")
        return
        
    concat_epochs = np.concatenate(all_epochs_bb, axis=0)
    print(f"    Concatenated epochs: {concat_epochs.shape} (trials, channels, time)")
    
    # 4. Clean epochs (DISABLED)
    print(f"    [Cleaning] Using RAW (resampled) epochs (Baseline/Exp/Template removal DISABLED).")
    concat_epochs_clean = concat_epochs
    
    # --- Step 4.5: Reject Bad Channels (Impedance Only) ---
    # Need ua_elec_global which was stored in the loop
    bad_ch_indices = get_bad_channels_utah(IMP_FILES, ua_elec_global, threshold_kohm=UTAH_IMPEDANCE_CUTOFF)
    
    if len(bad_ch_indices) > 0:
         print(f"    [Rejection] Removing {len(bad_ch_indices)} bad channels (Imp > 1000kOhm).")
         concat_epochs_clean[:, bad_ch_indices, :] = np.nan
    
    # 5. Compute time vectors and slice
    # rel_t_epoch needs to account for padding
    n_time = concat_epochs_clean.shape[2]
    rel_t_epoch = (np.arange(n_time) / TARGET_FS * 1000.0) - (EPOCH_PRE_MS + PAD_MS)
    
    WIN_PRE = (-EPOCH_PRE_MS, -BLANK_PRE_MS)
    WIN_POST = (BLANK_POST_MS, EPOCH_POST_MS - BLANK_PRE_MS)
    
    bb_pre, t_pre = slice_epoch(concat_epochs_clean, rel_t_epoch, WIN_PRE)
    bb_post, t_post = slice_epoch(concat_epochs_clean, rel_t_epoch, WIN_POST)
    
    results = {
        "broadband_pre": bb_pre,
        "broadband_post": bb_post,
    }
    
    # 6. Band analysis
    for band_name, (low, high) in LFP_BANDS.items():
        print(f"    Filtering {band_name} ({low}-{high} Hz)...")
        filt_epochs = filter_zero_phase(concat_epochs_clean, TARGET_FS, low, high)
        results[f'{band_name}_pre'], _ = slice_epoch(filt_epochs, rel_t_epoch, WIN_PRE)
        results[f'{band_name}_post'], _ = slice_epoch(filt_epochs, rel_t_epoch, WIN_POST)
    
    # 7. Save
    save_dict = {
        "fs_lfp": TARGET_FS,
        "group_port": port,
        "group_depth": depth,
        "sessions": [s["session"] for s in session_infos],
        "br_indices": [s["br_idx"] for s in session_infos],
        "stim_ms": ir_events_ms,
        "n_trials": concat_epochs_clean.shape[0],
        "rel_time_pre": t_pre,
        "rel_time_post": t_post,
        "stim_channels": ["Baseline_IR"],
        # UA-specific metadata
        "ua_region": ua_region_global,
        "ua_region_names": ua_region_names_global,
        "ua_elec": ua_elec_global,
        "ua_nsp": ua_nsp_global,
        **results
    }
    
    np.savez_compressed(out_npz, **save_dict)
    print(f"    Saved -> {out_npz}")
    
    gc.collect()


def process_utah_session(sess_name: str, br_idx: int, shift_ms: float, fs_intan: float, shift_sample: float):
    matches = [p for p in BR_ROOT.iterdir() if p.is_dir() and f"_{br_idx:03d}" in p.name]
    
    if matches:
        # Found a directory
        dir_path = matches[0]
        # Look for ns files inside
        ns_files = list(dir_path.glob("*.ns*"))
        ns6 = [f for f in ns_files if f.suffix == '.ns6']
        ns2 = [f for f in ns_files if f.suffix == '.ns2']
        ns5 = [f for f in ns_files if f.suffix == '.ns5']
        
        if ns6: sess_path = ns6[0]
        elif ns2: sess_path = ns2[0]
        elif ns5: sess_path = ns5[0]
        elif ns_files: sess_path = ns_files[0]
        else:
            print(f"[UA] Found directory {dir_path.name} but no .ns files inside.")
            return
            
    else:
        # Check for files (flat structure)
        # Look for .ns files
        matches_files = [p for p in BR_ROOT.iterdir() if p.is_file() and f"_{br_idx:03d}" in p.name and p.suffix in ['.ns2', '.ns5', '.ns6']]
        if matches_files:
            ns6 = [f for f in matches_files if f.suffix == '.ns6']
            ns2 = [f for f in matches_files if f.suffix == '.ns2']
            if ns6:
                sess_path = ns6[0]
            elif ns2:
                sess_path = ns2[0]
            else:
                sess_path = matches_files[0]
        else:
            print(f"[UA] No session folder or files found for BR {br_idx:03d}")
            return

    # Construct unique identifier for output filename
    # If it's a file, sess_path.name includes extension. removing it for cleanliness in filename
    sess_id = sess_path.stem if sess_path.is_file() else sess_path.name
    
    out_npz = UA_LFP_CKPT_ROOT / f"aligned_lfp__{sess_id}.npz"
    if out_npz.exists():
        print(f"[UA] Skipping {sess_id}, output exists.")
        return
        
    print(f"[UA] Processing {sess_id} (Source: {sess_path.name})")
    
    # Load Recording
    # Load Recording
    try:
        # User requested to use nsx6
        print(f"  Loading Blackrock stream: nsx6")
        rec_raw = se.read_blackrock(str(sess_path), stream_name='nsx6', all_annotations=True)
    except Exception as e:
        print(f"[UA] Failed to load nsx6: {e}")
        return

    # Mapping
    rec_mapped, idx_rows, ua_elec, ua_nsp, ua_region, ua_region_names, ua_port = rcp.apply_ua_mapping_with_regions(
        rec_raw, UA_MAP, br_idx, METADATA_CSV
    )
    
    # Load Stim for Blanking AND Alignment (Using Intan info mapped to UA time)
    stim_npz_path, intan_session_name = rcp.stim_npz_path_from_br_idx(br_idx, METADATA_CSV, NPRW_AUX_DATA)
    if not stim_npz_path or not stim_npz_path.exists():
        print("[UA] No stim stream found (Intan). Skipping.")
        return
        
    stim = rcp.load_stim_detection(stim_npz_path)
    block_bounds = stim.get("block_bounds_samples", [])
    
    if block_bounds.size == 0:
        print("[UA] No stim events found.")
        stim_indices_native = np.array([], dtype=int)
        stim_ms_ua = np.array([])
    else:
        # Intan Samples -> Time -> Shift -> UA Time -> UA Native Samples
        # We need triggers in *native* UA samples for processing
        intan_stim_samps = block_bounds[:, 0]
        intan_stim_ms = intan_stim_samps * 1000.0 / fs_intan
        stim_ms_ua = intan_stim_ms - shift_ms
        
        fs_native = rec_mapped.get_sampling_frequency()
        stim_indices_native = np.floor(stim_ms_ua * fs_native / 1000.0).astype(int)

    # 2. Build SI Pipeline
    # Blank(-5, +20) -> Local CMR (None) -> DS(1000)
    # local_radii=None -> No CMR for Utah
    rec_proc = build_lfp_preprocessing_pipeline(
        rec_mapped, 
        stim_indices_native=stim_indices_native,
        curr_sess_name=sess_path.name,
        local_radii=None 
    )
    
    # 3. Get Traces (Eager Load)
    print("  [UA] Loading processed traces into memory...")
    traces = rec_proc.get_traces(return_in_uV=True).T
    n_channels, n_samples = traces.shape

    # Segment
    t_ms = np.arange(n_samples, dtype=float) * 1000.0 / TARGET_FS
    
    # --- Parameters ---
    WIN_EPOCH = (-EPOCH_PRE_MS, EPOCH_POST_MS)   # Epoch Window
    WIN_PRE   = (-EPOCH_PRE_MS, -BLANK_PRE_MS)    # Baseline Window
    WIN_POST  = (BLANK_POST_MS, EPOCH_POST_MS - BLANK_PRE_MS)   # Analysis Window (Post-Blank)
    
    results = {}
    
    print(f"  Segmenting Broadband Epochs ({WIN_EPOCH} ms)...", flush=True)
    segs_bb, _, rel_t_epoch, valid_stim = rcp.extract_peristim_segments(
        rate_hz=traces.astype(np.float32), 
        counts=None,
        t_ms=t_ms,
        stim_ms=stim_ms_ua,
        win_ms=WIN_EPOCH
    )
    
    if valid_stim is None:
        print(f"  [WARN] No valid segments found for {sess_path.name}. Skipping.")
        return

    # 7. Per-Trial Cleaning - DISABLED
    print(f"  [Cleaning] Using RAW (resampled) epochs (Baseline/Exp/Template removal DISABLED).")
    segs_bb_clean = segs_bb
    
    # --- Step 7.5: Reject Bad Channels ---
    # 1. Impedance (Utah specific)
    # Use global IMP_FILES and get_bad_channels_utah
    bad_ch_indices = get_bad_channels_utah(IMP_FILES, ua_elec, threshold_kohm=UTAH_IMPEDANCE_CUTOFF)
    
    if len(bad_ch_indices) > 0:
         print(f"  [Rejection] Removing {len(bad_ch_indices)} bad channels (Imp > {UTAH_IMPEDANCE_CUTOFF}kOhm).")
         # We set them to NaN in the SEGMENTS (broadband)
         segs_bb_clean[:, bad_ch_indices, :] = np.nan
    
    results['bad_channels'] = bad_ch_indices
    
    def slice_epoch(data, t_axis, target_win):
        mask = (t_axis >= target_win[0]) & (t_axis < target_win[1])
        return data[:, :, mask], t_axis[mask]
        
    results['broadband_pre'], t_pre = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_PRE)
    results['broadband_post'], t_post = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_POST)
    
    # 9. Band Analysis (Filter Continuous -> Segment)
    # This avoids padding issues for segments since we have continuous data
    for band_name, (low, high) in LFP_BANDS.items():
        print(f"  Filtering Continuous Trace {band_name} ({low}-{high} Hz)...", flush=True)
        # traces: (n_ch, n_samples). filter_zero_phase expects (trials, ch, time).
        # We pass fake trial dim: (1, n_ch, n_samples)
        filt_traces = filter_zero_phase(traces[None, :, :], TARGET_FS, low, high)[0] # Extract back
        
        # Segment fitlered traces
        # Use same valid_stim
        segs_filt, _, _, _ = rcp.extract_peristim_segments(
            rate_hz=filt_traces.astype(np.float32), 
            counts=None,
            t_ms=t_ms,
            stim_ms=valid_stim,
            win_ms=WIN_EPOCH
        )
        
        if segs_filt is not None:
             # Apply Rejection
             if len(bad_ch_indices) > 0:
                 segs_filt[:, bad_ch_indices, :] = np.nan
             
             results[f'{band_name}_pre'], _ = slice_epoch(segs_filt, rel_t_epoch, WIN_PRE)
             results[f'{band_name}_post'], _ = slice_epoch(segs_filt, rel_t_epoch, WIN_POST)

        
    # Save
    save_dict = {
        "fs_lfp": TARGET_FS,
        "ua_region": ua_region,
        "ua_region_names": ua_region_names,
        "ua_elec": ua_elec,
        "ua_nsp": ua_nsp,
        "session": sess_path.name,
        "stim_ms": valid_stim,
        "rel_time_pre": t_pre,
        "rel_time_post": t_post,
        **results
    }
    
    np.savez_compressed(out_npz, **save_dict)
    print(f"  Saved -> {out_npz}")
    
    del rec_raw, rec_mapped, traces, results
    gc.collect()


def quick_check_stim(sess_path: Path, stream_name: str, chanmap: np.ndarray, intan_idx: int, br_idx: int) -> bool:
    """
    Check if stim file or aligned file exists and has events.
    """
    sess_name = sess_path.name
    
    # 0. Check Aligned File First
    aligned_path = ALIGNED_CKPT_ROOT / f"aligned__{sess_name}__Intan_{intan_idx:03d}__BR_{br_idx:03d}.npz"
    if aligned_path.exists():
         try:
             aln = np.load(aligned_path, allow_pickle=True)
             if "stim_ms" in aln and aln["stim_ms"].size > 0:
                 print(f"    [Check] Found aligned file with {aln['stim_ms'].size} events: {aligned_path.name}")
                 return True
         except:
             pass

    stim_npz = NPRW_AUX_DATA / f"{sess_name}_Intan_streams" / "stim_stream.npz"
    
    # 1. Standard Stim Check
    has_stim = False
    
    # Try loading existing fast
    if stim_npz.exists():
        try:
            print(f"    [Check] Loading existing {stim_npz.name}")
            stim_data = rcp.load_stim_detection(stim_npz)
            bounds = stim_data.get("block_bounds_samples")
            if bounds is not None and bounds.size > 0:
                print(f"    [Check] Found {len(bounds)} stim blocks.")
                has_stim = True
            else:
                print("    [Check] Existing file has 0 stim blocks.")
        except Exception as e:
            print(f"    [Check] Failed to load existing: {e}")

    # Fallback to extraction if not found or empty?? 
    # Actually, if existing file is empty, it means we already tried extracting and found nothing.
    # So we shouldn't re-extract standard stim. 
    # But if file didn't exist, we try extraction.
    
    if not has_stim and not stim_npz.exists():
        try:
            print(f"    [Check] Extracting new stim file...")
            stim_ext = rcp.extract_stim_npz(sess_path, out_dir=NPRW_AUX_DATA, stim_stream_name=stream_name, chanmap_perm=chanmap)
            if stim_ext is not None:
                bounds = stim_ext.get("block_bounds_samples")
                if bounds is not None and bounds.size > 0:
                     has_stim = True
        except Exception as e:
            print(f"  [Check Stim] Failed: {e}")

    if has_stim:
        return True

    # -------------------------------------------------------------------------
    # [Control/IR Fallback Check]
    # If no stim, check if we have valid IR events from baseline PeriStim files
    # -------------------------------------------------------------------------
    print(f"    [Check] No stim found. Checking for baseline PeriStim files...")
    try:
        # Look for baseline files in PERI_ROOT / Target_A and Target_B
        sess_name = sess_path.name
        
        for target in ["Target_A", "Target_B"]:
            target_dir = PERI_ROOT / target
            if not target_dir.exists():
                continue
                
            # Pattern: baseline__{session}*_target_{A,B}.npz
            pattern = f"baseline__*{sess_name}*_target_{target[-1]}.npz"
            matches = list(target_dir.glob(pattern))
            
            if matches:
                baseline_file = matches[0]
                print(f"    [Check] Found baseline file: {baseline_file.name}")
                try:
                    bl_data = np.load(baseline_file, allow_pickle=True)
                    if "stim_ms" in bl_data and bl_data["stim_ms"].size > 0:
                        print(f"    [Check] Baseline file has {bl_data['stim_ms'].size} IR events.")
                        return True
                except Exception as e:
                    print(f"    [Check] Failed to load baseline file: {e}")
                    
        print(f"    [Check] No baseline PeriStim files found for {sess_name}.")
                    
    except Exception as e:
        print(f"    [Check] Baseline file check failed: {e}")
            
    return False

def main():
    if not SHIFT_CSV.exists():
        print(f"Error: {SHIFT_CSV} not found. Cannot proceed with aligned analysis.")
        return

    with SHIFT_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    print(f"Found {len(rows)} sessions in shift CSV.")
    
    # Pre-load geometry for stim checking
    try:
        intan_geom, intan_probe_mapping = get_intan_geometry()
    except Exception as e:
        print(f"[ERROR] Could not load Intan geometry: {e}")
        return
    
    # =========================================================================
    # Load Metadata to Identify Baseline Sessions
    # =========================================================================
    baseline_br_idxs = set()  # BR indices that are baseline sessions
    baseline_groups = {}  # {(port, depth): [session_info, ...]}
    
    if METADATA_CSV.exists():
        print(f"\nLoading metadata from {METADATA_CSV.name}...")
        try:
            df_meta_raw = pd.read_csv(METADATA_CSV)
            df_meta = _normalize_metadata(df_meta_raw)
            
            groups = _find_baseline_groups(df_meta)
            if groups:
                print(f"Found {len(groups)} baseline groups: {list(groups.keys())}")
                
                # Build mapping BR -> (port, depth)
                if "br_file" in df_meta.columns:
                    br_col = pd.to_numeric(df_meta["br_file"], errors="coerce")
                    for (port, depth), idxs in groups.items():
                        for i in idxs:
                            br = br_col.iloc[i]
                            if pd.notna(br):
                                baseline_br_idxs.add(int(br))
                                
                # Initialize groups for session collection
                for key in groups:
                    baseline_groups[key] = []
                    
        except Exception as e:
            print(f"[WARN] Failed to load metadata for baseline detection: {e}")
    else:
        print(f"[WARN] Metadata CSV not found: {METADATA_CSV}. Baseline grouping disabled.")
    
    print(f"Identified {len(baseline_br_idxs)} baseline BR indices: {sorted(baseline_br_idxs)}")
    
    # =========================================================================
    # Per-Session Loop (Skip Baseline Sessions)
    # =========================================================================
    for row in rows:
        session_name = row["session"]
        intan_idx = int(row["intan_idx"])
        br_idx = int(row["br_idx"])
        shift_ms = float(row["shift_ms"])
        fs_intan = float(row.get("fs_intan", 30000.0))
        shift_sample = float(row.get("shift_sample", 0.0))
        
        # Check if this is a baseline session
        if br_idx in baseline_br_idxs:
            print(f"\n[Baseline] Session {session_name} (BR={br_idx}) is baseline. Will process in group.")
            
            # Find which group this belongs to and add session info
            if METADATA_CSV.exists():
                try:
                    df_meta_raw = pd.read_csv(METADATA_CSV)
                    df_meta = _normalize_metadata(df_meta_raw)
                    
                    # Find the row for this br_idx
                    if "br_file" in df_meta.columns:
                        br_col = pd.to_numeric(df_meta["br_file"], errors="coerce")
                        mask = br_col == br_idx
                        if mask.any():
                            row_meta = df_meta.loc[mask].iloc[0]
                            port = row_meta.get("_ua_port_norm") if "_ua_port_norm" in row_meta else (
                                str(row_meta.get("ua_port", "UNKNOWN")).upper().strip() or "UNKNOWN"
                            )
                            depth = row_meta.get("_depth_mm_norm") if "_depth_mm_norm" in row_meta else (
                                str(row_meta.get("depth_mm", "n/a")).strip() or "n/a"
                            )
                            
                            # Normalize port/depth the same way as _find_baseline_groups
                            port = str(port).upper().strip() or "UNKNOWN"
                            try:
                                depth = float(depth)
                            except:
                                depth = str(depth)
                            depth = str(depth)
                            
                            group_key = (port, depth)
                            if group_key not in baseline_groups:
                                baseline_groups[group_key] = []
                            
                            baseline_groups[group_key].append({
                                "session": session_name,
                                "br_idx": br_idx,
                                "intan_idx": intan_idx,
                                "shift_ms": shift_ms,
                                "fs_intan": fs_intan,
                            })
                except Exception as e:
                    print(f"    [WARN] Failed to identify group for session: {e}")
            continue
        
        # Resolve Intan Path
        matches = list(INTAN_ROOT.glob(f"*{session_name}*"))
        valid_folders = [p for p in matches if p.is_dir()]
        
        if not valid_folders:
            print(f"[WARN] Session folder not found: {session_name}. Skipping pair.")
            continue
            
        sess_path = valid_folders[0]
        
        # --- Pre-Check Stim ---
        # We check stim BEFORE running either NPRW or UA analysis
        print(f"\nChecking stim for {session_name}...")
        has_stim = quick_check_stim(sess_path, STIM_STREAM, intan_probe_mapping, intan_idx, br_idx)
        
        if not has_stim:
            print(f"  -> No valid stim events found. Skipping both NPRW and UA for this session.")
            continue
            
        print(f"  -> Stim OK. Proceeding...")

        try:
            process_nprw_session(session_name, br_idx, intan_idx, shift_ms)
        except Exception as e:
            print(f"[ERROR] NPRW {session_name}: {e}")
            import traceback
            traceback.print_exc()

        try:
            process_utah_session(session_name, br_idx, shift_ms, fs_intan, shift_sample)
        except Exception as e:
            print(f"[ERROR] UA {br_idx}: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Process Baseline Groups
    # =========================================================================
    if baseline_groups:
        print(f"\n{'='*60}")
        print("Processing Baseline Groups")
        print(f"{'='*60}")
        
        for group_key, session_infos in baseline_groups.items():
            if not session_infos:
                continue
            
            # Process NPRW (Intan) baseline
            try:
                process_baseline_group(group_key, session_infos)
            except Exception as e:
                print(f"[ERROR] Baseline NPRW group {group_key}: {e}")
                import traceback
                traceback.print_exc()
            
            # Process Utah Array baseline
            try:
                process_baseline_group_utah(group_key, session_infos)
            except Exception as e:
                print(f"[ERROR] Baseline UA group {group_key}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()

