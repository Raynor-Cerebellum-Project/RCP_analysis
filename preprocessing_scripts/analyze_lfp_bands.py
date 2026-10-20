"""
    Analyzes LFP bands (Alpha, Beta, Low/High Gamma) for NPRW (Intan) and Utah Array (Blackrock)
    
    Pipeline:
      1. Extraction: Loads stimulation events and extracts epochs (-1000ms to +1000ms) with padding.
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
        - NPRW: results/checkpoints/NPRW_LFP/aligned_lfp    __{session_name}.npz
        - Utah: results/checkpoints/UA_LFP/aligned_lfp__{session_name}.npz
"""

import gc
import csv
import numpy as np
from scipy import signal
from scipy.io import loadmat
from pathlib import Path
import pandas as pd

import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se

import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *
from RCP_analysis.python.functions.artifact_correction import IPCA_Artifact_Correction, Template

# ---------- BASELINE TRIAL EXCLUSION ----------

from spikeinterface.core import BaseRecording, BaseRecordingSegment

class PerChannelIPCACorrectedRecordingSegment(BaseRecordingSegment):
    def __init__(self, parent_recording_segment, micro_map_per_channel, micro_corrected):
        BaseRecordingSegment.__init__(self, **parent_recording_segment.get_times_kwargs())
        self.parent_recording_segment = parent_recording_segment
        self.micro_map_per_channel = micro_map_per_channel  # {ch_idx: [(start, end), ...]}
        self.micro_corrected = micro_corrected
        
        # Build lookup structures per channel for fast searching
        self.channel_starts = {}
        self.channel_ends = {}
        
        for ch_idx, map_list in micro_map_per_channel.items():
            if map_list:
                starts = np.array([s for s, e in map_list])
                ends = np.array([e for s, e in map_list])
                order = np.argsort(starts)
                self.channel_starts[ch_idx] = starts[order]
                self.channel_ends[ch_idx] = ends[order]
            else:
                self.channel_starts[ch_idx] = np.array([], dtype=np.int64)
                self.channel_ends[ch_idx] = np.array([], dtype=np.int64)

    def get_num_samples(self):
        return self.parent_recording_segment.get_num_samples()

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        traces = traces.copy()
        
        # Handle different channel_indices types
        if channel_indices is None:
            channel_indices = np.arange(len(self.micro_map_per_channel))
        elif isinstance(channel_indices, slice):
            start = channel_indices.start if channel_indices.start is not None else 0
            stop = channel_indices.stop if channel_indices.stop is not None else len(self.micro_map_per_channel)
            step = channel_indices.step if channel_indices.step is not None else 1
            channel_indices = np.arange(start, stop, step)
        else:
            channel_indices = np.asarray(channel_indices)
        
        # Process each channel independently
        for local_idx, global_ch_idx in enumerate(channel_indices):
            if global_ch_idx not in self.channel_starts:
                continue
            
            starts = self.channel_starts[global_ch_idx]
            ends = self.channel_ends[global_ch_idx]
            
            if len(starts) == 0:
                continue
            
            lo = np.searchsorted(ends, start_frame, side='right')
            hi = np.searchsorted(starts, end_frame, side='left')
            
            for pulse_idx in range(lo, hi):
                p_start = starts[pulse_idx]
                p_end = ends[pulse_idx]
                
                overlap_start = max(start_frame, p_start)
                overlap_end = min(end_frame, p_end)
                
                if overlap_start < overlap_end:
                    chunk_idx_start = overlap_start - start_frame
                    chunk_idx_end = overlap_end - start_frame
                    patch_idx_start = overlap_start - p_start
                    patch_idx_end = overlap_end - p_start
                    
                    patch = self.micro_corrected[pulse_idx, patch_idx_start:patch_idx_end, global_ch_idx]
                    traces[chunk_idx_start:chunk_idx_end, local_idx] = patch
        
        return traces

class PerChannelIPCACorrectedRecording(BaseRecording):
    def __init__(self, parent_recording, micro_map_per_channel, micro_corrected):
        BaseRecording.__init__(self, 
                               parent_recording.get_sampling_frequency(), 
                               parent_recording.channel_ids, 
                               parent_recording.get_dtype())
        self.parent_recording = parent_recording
        parent_recording.copy_metadata(self)
        
        for segment_index in range(parent_recording.get_num_segments()):
            parent_segment = parent_recording._recording_segments[segment_index]
            self.add_recording_segment(
                PerChannelIPCACorrectedRecordingSegment(parent_segment, micro_map_per_channel, micro_corrected)
            )


# Format: "baseline_port{A,B}_target{A,B}": [list of 0-based trial indices to skip]
# Use this to remove false positive IR-triggered events (e.g., multiple triggers on same reach)
EXCLUDE_BASELINE_TRIALS = {
    # "baseline_portA_targetA": [1, 3, 4, 6, 13, 14, 21, 24],
    # "baseline_portB_targetA": [2, 6, 7, 9, 13, 18],
    # "baseline_portA_targetB": [6, 10, 13, 15, 16, 17, 20, 21, 22, 24, 25, 30, 39, 40, 42, 47, 56, 57],
    # "baseline_portB_targetB": [2, 3, 4, 6, 8, 10, 12, 15, 17, 20, 22, 23, 24, 26, 28, 30, 31, 35],
}


# ---------- CONSTANTS ----------
TARGET_FS = 1000
BLANK_PRE_MS = 5.0
BLANK_POST_MS = 101.0 
EPOCH_PRE_MS = 1000.0
EPOCH_POST_MS = 1000.0
PAD_MS = 1000.0 # Padding for zero-phase filtering
# PLOT_1F_DEBUG = False # Optional flag to save figure comparing 1/f correction

# --- IPCA Artifact Correction Settings ---
IPCA_PARAMS = PARAMS.IPCA_Params
USE_IPCA_CORRECTION  = IPCA_PARAMS.get("use_ipca", True)
IPCA_RANK            = IPCA_PARAMS.get("rank", 10)
IPCA_PULSE_WINDOW_MS = IPCA_PARAMS.get("pulse_window_ms", (-0.4, 0.3))

# ---------- CONFIG ----------
# Bands of interest
LFP_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 25),
    "low_gamma": (25, 58),
    "high_gamma": (62, 120)
}

# Filter order for butterworth
FILTER_ORDER = 4

# Output Directories
UA_LFP_CKPT_ROOT = OUT_BASE / "checkpoints" / "UA_LFP"
NPRW_LFP_CKPT_ROOT = OUT_BASE / "checkpoints" / "NPRW_LFP"
ALIGNED_CKPT_ROOT = OUT_BASE / "checkpoints" / "Aligned"
PERI_ROOT = OUT_BASE / "checkpoints" / "PeriStim"
UA_LFP_CKPT_ROOT.mkdir(parents=True, exist_ok=True)
NPRW_LFP_CKPT_ROOT.mkdir(parents=True, exist_ok=True)

# Stim / Intan Settings mapping
NPRW_CFG = PARAMS.probes.get("NPRW")
STIM_STREAM = NPRW_CFG.get("stim_data_stream")

# Utah Config
UA_CFG = PARAMS.probes.get("UA")
XLS = rcp.ua_excel_path(REPO_ROOT, PARAMS.probes)
UA_MAP = rcp.load_UA_mapping_from_excel(XLS) if XLS else None
METADATA_CSV = METADATA_ROOT / f"{PARAMS.session}_metadata.csv"
SHIFT_CSV = METADATA_ROOT / "br_to_intan_shifts.csv"

# Global job kwargs
global_job_kwargs = dict(n_jobs=PARAMS.parallel_jobs, chunk_duration=PARAMS.chunk)
si.set_global_job_kwargs(**global_job_kwargs)

# Cache for stim file checks to avoid redundant disk I/O
STIM_CACHE = {}

# What to actually process?
SKIP_EXISTING = False


# =============================================================================
# IPCA Helper Functions (matching UA_BR_analysis_mf.py logic)
# =============================================================================

def _get_stim_frequency_from_metadata(br_idx):
    """Get stimulation frequency from metadata CSV."""
    stim_freq = 400.0  # default
    if METADATA_CSV.exists():
        try:
            freq_map = rcp.get_metadata_mapping(METADATA_CSV, 'BR_File', 'Stim_Frequency_Hz')
            stim_freq = float(freq_map.get(br_idx, 400.0))
            if np.isnan(stim_freq):
                stim_freq = 400.0
        except Exception:
            pass
    return stim_freq


def _get_ua_port_from_metadata(br_idx):
    """Get UA port (A or B) from metadata CSV."""
    ua_port = "A"  # default
    if METADATA_CSV.exists():
        try:
            df_meta_raw = pd.read_csv(METADATA_CSV)
            df_meta = _normalize_metadata(df_meta_raw)
            if "br_file" in df_meta.columns:
                br_col = pd.to_numeric(df_meta["br_file"], errors="coerce")
                mask = br_col == br_idx
                if mask.any():
                    row_meta = df_meta.loc[mask].iloc[0]
                    port_val = str(row_meta.get("ua_port", "A")).upper().strip()
                    if port_val in ("A", "B"):
                        ua_port = port_val
        except Exception:
            pass
    return ua_port


# =============================================================================
# Peristim File Discovery
# =============================================================================

def find_peristim_files(br_idx_or_sess, br_idx_opt=None):
    """
    Find all PeriStim files for a given Blackrock index.
    Polymorphic: accepts find_peristim_files(br_idx) or find_peristim_files(sess_name, br_idx).
    """
    if isinstance(br_idx_or_sess, int):
        br_idx = br_idx_or_sess
        sess_name = br_idx_opt
    elif isinstance(br_idx_opt, int):
        br_idx = br_idx_opt
        sess_name = br_idx_or_sess
    else:
        try:
            br_idx = int(br_idx_opt)
            sess_name = str(br_idx_or_sess)
        except:
            try:
                br_idx = int(br_idx_or_sess)
                sess_name = None
            except:
                return []

    found = []
    pattern = f"peristim__*BR_{br_idx:03d}*.npz"
    for cat in ["control_reaches", "stim_reaches", "at_rest", "Grasp", "IMU", "continuous_stim"]:
        cat_dir = PERI_ROOT / cat
        if not cat_dir.exists(): 
            continue
        
        # Check main cat dir
        for p in sorted(cat_dir.glob(pattern)):
            if sess_name is None or str(sess_name) in p.name:
                found.append((cat, None, p))
             
        # Check target subdirs
        for target in ["target_A", "target_B"]:
            t_dir = cat_dir / target
            if t_dir.exists():
                for p in sorted(t_dir.glob(pattern)):
                    if sess_name is None or str(sess_name) in p.name:
                        found.append((cat, target, p))
    return found


# =============================================================================
# Metadata Helpers (for Baseline Grouping)
# =============================================================================

def _normalize_metadata(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase with underscores."""
    df = df_raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _get_stim_params_for_br(br_idx: int, is_stim_cat: bool = True) -> tuple[int, float, float]:
    """Robustly resolve n_pulses, stim_dur_ms, and stim_freq_hz for a given Blackrock condition."""
    if not is_stim_cat:
        return 0, 0.0, 0.0

    n_pulses = None
    stim_freq_hz = 400.0
    stim_dur_ms = None

    # 1. Check PeriStim checkpoint files for true detected pulses
    try:
        peri_files = find_peristim_files(br_idx)
        if peri_files:
            for _, _, p_path in peri_files:
                p_data = np.load(p_path, allow_pickle=True)
                if "stim_pulse_counts" in p_data and p_data["stim_pulse_counts"].size > 0:
                    n_pulses = int(p_data["stim_pulse_counts"].flat[0])
                if "stim_freq_hz" in p_data:
                    stim_freq_hz = float(p_data["stim_freq_hz"])
                if "stim_dur_nominal_ms" in p_data:
                    stim_dur_ms = float(p_data["stim_dur_nominal_ms"])
                if n_pulses is not None and n_pulses > 0:
                    break
    except Exception:
        pass

    # 2. Check metadata CSV via parse_session_metadata_from_csv
    if n_pulses is None or stim_dur_ms is None:
        try:
            sess_parsed = rcp.parse_session_metadata_from_csv(METADATA_CSV, br_file=br_idx)
            if sess_parsed:
                sf = sess_parsed.get("stim_freq_hz")
                if sf is not None and not np.isnan(sf) and sf > 0:
                    stim_freq_hz = float(sf)
                sd = sess_parsed.get("stim_dur_nominal_ms")
                if sd is not None and not np.isnan(sd) and sd > 0:
                    stim_dur_ms = float(sd)
                    if n_pulses is None and stim_freq_hz > 0:
                        n_pulses = int(round(stim_dur_ms * stim_freq_hz / 1000.0))
        except Exception:
            pass

    # 3. Direct normalized metadata search for pulse count
    if n_pulses is None and METADATA_CSV.exists():
        try:
            df_m = pd.read_csv(METADATA_CSV)
            df_norm = _normalize_metadata(df_m)
            br_col = None
            for c in ["br_file", "br_idx", "br", "condition"]:
                if c in df_norm.columns:
                    br_col = c
                    break
            if br_col:
                mask = pd.to_numeric(df_norm[br_col], errors="coerce") == br_idx
                if mask.any():
                    row = df_norm.loc[mask].iloc[0]
                    for pk in ["pulse_count", "pulses", "n_pulses", "pulse"]:
                        if pk in row and pd.notna(row[pk]):
                            n_pulses = int(float(row[pk]))
                            break
                    for fk in ["stim_frequency_hz", "stim_freq_hz", "frequency", "freq_hz"]:
                        if fk in row and pd.notna(row[fk]):
                            stim_freq_hz = float(row[fk])
                            break
                    for dk in ["stim_duration_nominal_ms", "stim_duration_ms", "stim_dur_ms", "duration_ms"]:
                        if dk in row and pd.notna(row[dk]):
                            stim_dur_ms = float(row[dk])
                            break
        except Exception:
            pass

    # Calculate duration from pulses if needed
    if n_pulses is not None and (stim_dur_ms is None or stim_dur_ms <= 0) and stim_freq_hz > 0:
        stim_dur_ms = float(n_pulses * 1000.0 / stim_freq_hz)

    # Defaults if completely undetermined
    if n_pulses is None:
        n_pulses = 40
    if stim_dur_ms is None or stim_dur_ms <= 0:
        stim_dur_ms = float(n_pulses * 1000.0 / stim_freq_hz) if (n_pulses and stim_freq_hz > 0) else 100.0

    return n_pulses, stim_dur_ms, stim_freq_hz



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
def blank_notch_epochs(epochs, fs, notch_freqs=[60.0, 120.0], bandwidth=10.0):
    """
    Blank (set to NaN) frequency bins around line noise frequencies.
    This leaves gaps in the spectrum rather than interpolating.
    
    Args:
        epochs: np.ndarray of shape (n_trials, n_channels, n_time)
        fs: Sampling rate (Hz)
        notch_freqs: Frequencies to blank (default [60, 120] Hz)
        bandwidth: Hz width to blank (default 10 Hz = 55-65 and 115-125 Hz)
        
    Returns:
        np.ndarray: Cleaned epochs, same shape
    """
    n_trials, n_channels, n_time = epochs.shape
    
    # Work in frequency domain
    fft_epochs = np.fft.rfft(epochs, axis=2)
    freqs = np.fft.rfftfreq(n_time, 1.0 / fs)
    df = freqs[1] - freqs[0]  # Frequency resolution
    
    for notch_freq in notch_freqs:
        if notch_freq <= 0 or notch_freq >= fs / 2:
            continue
        
        # Find bins to blank (e.g., 55-65 Hz for 60 Hz with bandwidth=10)
        half_width = bandwidth / 2
        blank_mask = (freqs >= notch_freq - half_width) & (freqs <= notch_freq + half_width)
        
        # Set these frequency bins to NaN (will propagate to time domain as NaN)
        # Actually, we can't use NaN in FFT and get back clean signal
        # Instead, set to zero which effectively removes that frequency content
        fft_epochs[:, :, blank_mask] = 0
    
    # Inverse FFT
    cleaned = np.fft.irfft(fft_epochs, n=n_time, axis=2)
    
    return cleaned.astype(epochs.dtype)

def filter_zero_phase(epochs, fs, low, high, order=4, chunk_size=32):
    """
    Apply zero-phase bandpass filter using filtfilt along axis 2 (time).
    Processes in chunks along the channel axis (axis=1) to prevent 
    massive memory OOM spikes when filtering long continuous recordings.
    
    Args:
        epochs (np.ndarray): (n_trials, n_ch, n_time)
        fs (float): Sampling rate (Hz)
        low (float): Low cutoff (Hz)
        high (float): High cutoff (Hz)
        order (int): Filter order
        chunk_size (int): Number of channels to filter at once
        
    Returns:
        np.ndarray: Filtered data
    """
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    
    b, a = signal.butter(order, [low_norm, high_norm], btype='band')
    
    out = np.empty_like(epochs)
    n_ch = epochs.shape[1]
    
    # Process channels in chunks to avoid allocating gigantic padded 
    # copies of the entire matrix inside scipy.signal.filtfilt
    for start_ch in range(0, n_ch, chunk_size):
        end_ch = min(start_ch + chunk_size, n_ch)
        out[:, start_ch:end_ch, :] = signal.filtfilt(b, a, epochs[:, start_ch:end_ch, :], axis=2)
        
    return out


def apply_ipca_correction(
    rec_ns6,
    starts_ua,
    ends_ua,
    br_idx=None,
    fs_intan=30000.0,
    shift_samp_intan=0.0,
    ua_region=None,
    ua_region_names=None,):
    """
    Applies IPCA artifact removal to stimulation windows, using cross-channel templates 
    learned individually per brain region.
    
    Updated to match UA_BR_analysis_mf.py logic with:
    - Clock drift correction (1.33% slower than nominal)
    - Per-channel lag calibration from CSV files
    - Per-channel corrected recording placement
    """
    if not USE_IPCA_CORRECTION or not starts_ua.size:
        return rec_ns6

    fs_ua = rec_ns6.get_sampling_frequency()
    n_total = rec_ns6.get_num_samples()
    print(f"[IPCA] Starting Incremental PCA artifact correction on {len(starts_ua)} blocks...")
    
    # Get stim frequency from metadata
    stim_freq_meta = _get_stim_frequency_from_metadata(br_idx) if br_idx else 400.0
    pulse_interval_ms = 1000.0 / stim_freq_meta
    print(f"[IPCA] Using stim frequency {stim_freq_meta} Hz (interval: {pulse_interval_ms:.2f} ms)")
    
    # Get UA port for region mapping and lag calibration
    ua_port = _get_ua_port_from_metadata(br_idx) if br_idx else "A"
    
    # Calculate sample indices for windows
    mw_start = int(IPCA_PULSE_WINDOW_MS[0] / 1000.0 * fs_ua)
    mw_end = int(IPCA_PULSE_WINDOW_MS[1] / 1000.0 * fs_ua)
    micro_n_time = mw_end - mw_start
    
    # ══════════════════════════════════════════════════════════════════════
    # LOAD PER-CHANNEL LAG CALIBRATION
    # ══════════════════════════════════════════════════════════════════════
    lag_csv = REPO_ROOT / 'config' / 'channel_lag_calibration' / f'channel_lags_port{ua_port}.csv'
    
    channel_lags_by_name = {}
    channel_lags_by_electrode = {}
    
    if lag_csv.exists():
        with open(lag_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ch_name = row['channel_id']
                elec_id = int(row['electrode']) if row.get('electrode') else None
                
                if row.get('lag_samples') and row['lag_samples'].strip():
                    lag_samp = int(row['lag_samples'])
                elif row.get('lag_ms'):
                    lag_ms = float(row['lag_ms'])
                    lag_samp = int(round(lag_ms / 1000.0 * fs_ua))
                else:
                    continue
                
                channel_lags_by_name[ch_name] = lag_samp
                if elec_id is not None and elec_id > 0:
                    channel_lags_by_electrode[elec_id] = lag_samp
        
        print(f"[IPCA] Loaded lags for {len(channel_lags_by_name)} channels from {lag_csv.name}")
        if channel_lags_by_name:
            lag_values = list(channel_lags_by_name.values())
            print(f"[IPCA] Lag range: {min(lag_values)} to {max(lag_values)} samples")
    else:
        print(f"[IPCA] No lag calibration file found: {lag_csv}")
    
    # ══════════════════════════════════════════════════════════════════════
    # CALCULATE ACTUAL PULSE INTERVAL (CLOCK DRIFT CORRECTED)
    # ══════════════════════════════════════════════════════════════════════
    CLOCK_DRIFT_FACTOR = 1.0133  # Empirically measured
    
    nominal_interval_samples = pulse_interval_ms / 1000.0 * fs_ua
    interval_ua_samples = nominal_interval_samples * CLOCK_DRIFT_FACTOR
    
    actual_freq = fs_ua / interval_ua_samples
    print(f"[IPCA] Stim timing correction applied:")
    print(f"[IPCA]   Nominal: {stim_freq_meta:.1f} Hz ({nominal_interval_samples:.2f} samples)")
    print(f"[IPCA]   Actual:  {actual_freq:.1f} Hz ({interval_ua_samples:.2f} samples)")
    print(f"[IPCA]   Correction: +{interval_ua_samples - nominal_interval_samples:.2f} samples/pulse")
    
    # ══════════════════════════════════════════════════════════════════════
    # GENERATE PULSE TIMES IN UA COORDINATES
    # starts_ua and ends_ua are already converted to UA sample index space
    # with PPM clock correction via rcp.intan_samples_to_br_samples.
    # ══════════════════════════════════════════════════════════════════════
    all_ua_pulse_times = []
    
    for anchor_ua, en_ua in zip(starts_ua, ends_ua):
        # Calculate number of pulses based on block duration in UA time
        stim_dur_ms = (en_ua - anchor_ua) / fs_ua * 1000.0
        num_pulses = int(np.floor(stim_dur_ms / pulse_interval_ms)) + 1
        
        # Generate subsequent pulses in UA time directly
        pulse_times_ua = anchor_ua + np.arange(num_pulses) * interval_ua_samples
        all_ua_pulse_times.extend(pulse_times_ua)
    
    intan_pulses_base_ua = np.round(all_ua_pulse_times).astype(np.int64)
    print(f"[IPCA] Generated {len(intan_pulses_base_ua)} base pulse times in UA coordinates")
    
    # ══════════════════════════════════════════════════════════════════════
    # APPLY PER-CHANNEL LAGS
    # ══════════════════════════════════════════════════════════════════════
    n_channels = rec_ns6.get_num_channels()
    channel_ids = rec_ns6.get_channel_ids()
    
    # Get electrode mapping for fallback
    try:
        elec_ids = np.array([int(ch) for ch in channel_ids])
    except (ValueError, TypeError):
        elec_ids = np.arange(1, n_channels + 1)
    
    elec_ids_for_lookup = elec_ids
    
    # Calculate fallback lag
    available_lags = list(channel_lags_by_name.values())
    fallback_lag = int(np.median(available_lags)) if available_lags else 17
    print(f"[IPCA] Fallback lag (median): {fallback_lag} samples")
    
    intan_pulses_per_channel = {}
    lag_stats = {'by_name': 0, 'by_electrode': 0, 'fallback': 0}
    
    for ch_idx, ch_name in enumerate(channel_ids):
        ch_name_str = str(ch_name)
        
        if ch_name_str in channel_lags_by_name:
            lag = channel_lags_by_name[ch_name_str]
            lag_stats['by_name'] += 1
        else:
            elec_id = int(elec_ids_for_lookup[ch_idx]) if ch_idx < len(elec_ids_for_lookup) else None
            if elec_id and elec_id > 0 and elec_id in channel_lags_by_electrode:
                lag = channel_lags_by_electrode[elec_id]
                lag_stats['by_electrode'] += 1
            else:
                lag = fallback_lag
                lag_stats['fallback'] += 1
        
        channel_pulse_times = intan_pulses_base_ua + lag
        valid = (channel_pulse_times >= 0) & (channel_pulse_times < n_total)
        intan_pulses_per_channel[ch_idx] = channel_pulse_times[valid]
    
    print(f"[IPCA] Lag lookup stats: {lag_stats['by_name']} by_name, "
          f"{lag_stats['by_electrode']} by_electrode, {lag_stats['fallback']} fallback")
    
    n_pulses = max(len(times) for times in intan_pulses_per_channel.values())
    print(f"[IPCA] Maximum {n_pulses} pulses across all channels")
    
    if n_pulses == 0:
        print("[IPCA] No pulses found, skipping correction.")
        return rec_ns6
    
    # ══════════════════════════════════════════════════════════════════════
    # EXTRACT WINDOWS WITH LAG-CORRECTED CENTERS
    # ══════════════════════════════════════════════════════════════════════
    micro_signal_list = []
    micro_map_per_channel = {ch_idx: [] for ch_idx in range(n_channels)}
    
    for pulse_idx in range(n_pulses):
        pulse_data = np.zeros((micro_n_time, n_channels), dtype=np.float32)
        
        for ch_idx, ch_name in enumerate(channel_ids):
            channel_pulses = intan_pulses_per_channel[ch_idx]
            
            if pulse_idx >= len(channel_pulses):
                continue
            
            artifact_center = channel_pulses[pulse_idx]
            p_start = artifact_center + mw_start
            p_end = artifact_center + mw_end
            
            if p_start >= 0 and p_end <= n_total:
                trace = rec_ns6.get_traces(
                    start_frame=p_start, end_frame=p_end,
                    channel_ids=[ch_name], return_in_uV=False
                ).flatten()
                pulse_data[:, ch_idx] = trace
            
            micro_map_per_channel[ch_idx].append((p_start, p_end))
        
        micro_signal_list.append(pulse_data)
    
    micro_signal_array = np.stack(micro_signal_list)
    print(f"[IPCA] Extracted {len(micro_signal_array)} artifacts. Shape: {micro_signal_array.shape}")
    
    # ══════════════════════════════════════════════════════════════════════
    # DIAGNOSTIC: Quick artifact alignment check
    # ══════════════════════════════════════════════════════════════════════
    test_ch_idx = min(10, n_channels - 1)
    test_ch_name = channel_ids[test_ch_idx]
    n_check = min(50, micro_signal_array.shape[0])
    window_center = micro_signal_array.shape[1] // 2
    
    peak_locs = []
    for i in range(n_check):
        wf = micro_signal_array[i, :, test_ch_idx]
        peak_locs.append(np.argmax(np.abs(np.diff(wf, prepend=wf[0]))))
    peak_locs = np.array(peak_locs)
    drift = peak_locs - window_center
    
    drift_std = np.std(drift)
    drift_range = drift.max() - drift.min()
    alignment_ok = drift_range <= 3
    
    status = "✓ ALIGNED" if alignment_ok else "✗ MISALIGNED"
    print(f"[DIAG] Artifact alignment: {status}")
    print(f"[DIAG]   Peak jitter: {drift_std:.2f} samples (range: {drift_range})")
    
    # ══════════════════════════════════════════════════════════════════════
    # REGION-WISE IPCA CORRECTION
    # ══════════════════════════════════════════════════════════════════════
    corrector = IPCA_Artifact_Correction(rank=IPCA_RANK)
    micro_corrected = micro_signal_array.copy()
    
    # Map channels to regions
    ua_region = np.asarray(ua_region)
    ua_region_names = np.asarray(ua_region_names)

    if ua_region.shape[0] != n_channels:
        raise ValueError(
            f"ua_region length mismatch: got {ua_region.shape[0]}, "
            f"expected {n_channels}"
        )

    target_ch_regions = []
    for reg_idx in ua_region:
        reg_idx = int(reg_idx)
        if reg_idx < 0:
            target_ch_regions.append("Unknown")
        else:
            target_ch_regions.append(str(ua_region_names[reg_idx]))

    region_to_idxs = {}
    for ch_idx, reg_name in enumerate(target_ch_regions):
        region_to_idxs.setdefault(reg_name, []).append(ch_idx)
    
    n_stim, n_time, _ = micro_signal_array.shape
    
    print("Cross-channel IPCA by region:")
    for reg, idxs in region_to_idxs.items():
        if not idxs:
            continue
        
        reg_signal = micro_signal_array[:, :, idxs]
        pooled_signal = np.transpose(reg_signal, (0, 2, 1)).reshape(-1, n_time)
        
        reg_template = Template()
        _, reg_template = corrector.ipca_template_per_channel(pooled_signal, reg_template)
        
        print(f"  [{reg}] Shared subspace learned from {len(idxs)} channels.")
        
        for ch_idx in idxs:
            signal_ch = micro_signal_array[:, :, ch_idx].copy()
            corrected_ch = corrector.apply_template(signal_ch, reg_template)
            micro_corrected[:, :, ch_idx] = corrected_ch
    
    # ══════════════════════════════════════════════════════════════════════
    # BUILD CORRECTED RECORDING WITH PER-CHANNEL PLACEMENT
    # ══════════════════════════════════════════════════════════════════════
    print("[IPCA] Building per-channel IPCACorrectedRecording...")
    rec_corr_mem = PerChannelIPCACorrectedRecording(rec_ns6, micro_map_per_channel, micro_corrected)
    
    del micro_signal_array, micro_corrected
    gc.collect()
    
    return rec_corr_mem


# =============================================================================
# SI Pipeline and Epoch Extraction
# =============================================================================

def build_lfp_preprocessing_pipeline(
    recording: si.BaseRecording,
    stim_indices_native: np.ndarray,
    curr_sess_name: str = "unknown",
    local_radii: tuple | None = None) -> si.BaseRecording:
    """
    Custom pipeline with purely Bandpass and Resampling operations.
    Blanking / CMR are bypassed heavily for Utah IPCA.
    """
    # 1. Bandpass (1-200 Hz) BEFORE resampling
    print(f"    [SI Pipeline] Bandpass filter (1-200 Hz)...")
    rec_filtered = spre.bandpass_filter(
        recording, 
        freq_min=1.0, 
        freq_max=200.0,
        filter_order=FILTER_ORDER,
        dtype='float32',
        add_reflect_padding=True,
        ignore_low_freq_error=True,
    )

    # 2. Resampling (30k -> 1k)
    print(f"    [SI Pipeline] Resampling to {TARGET_FS} Hz...")
    rec_resampled = spre.resample(rec_filtered, resample_rate=TARGET_FS)
    
    return rec_resampled



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


# =============================================================================
# UA Mapping Helper
# =============================================================================

def _apply_native_ua_mapping(rec_raw, UA_MAP, br_idx, metadata_csv):
    """
    Apply centralized UA mapping from br_preproc.py, then adapt the result
    for this LFP script.

    Returns
    -------
    rec_mapped : SpikeInterface recording
        Recording restricted to mapped UA channels, with channel IDs renamed
        to 1-based electrode IDs as strings: "1", "2", ..., "256".
    ua_ids_1based : np.ndarray
        1-based UA electrode IDs corresponding to rec_mapped channel order.
    ua_port : str
        "A" or "B".
    ua_region : np.ndarray
        Region index per mapped channel, using convention:
          0 = SMA
          1 = PMd
          2 = M1i
          3 = M1s
    ua_region_names : np.ndarray
        Region names corresponding to ua_region indices.
    """

    if UA_MAP is None:
        print("    [WARN] UA_MAP is None; cannot apply UA mapping.")
        return None, None, "A", None, None

    (
        rec_renamed,
        idx_rows,
        ua_elec,
        ua_nsp,
        ua_region,
        ua_region_names,
        ua_port,
    ) = rcp.apply_ua_mapping_with_regions(
        rec_raw,
        UA_MAP,
        br_idx,
        metadata_csv,
        monkey=PARAMS.monkey,
    )

    # Keep only rows/channels that successfully mapped to a UA electrode.
    valid_rows = np.where(ua_elec > 0)[0]

    if valid_rows.size == 0:
        print("    [WARN] No mapped UA channels found after centralized mapping.")
        return None, None, ua_port, None, None

    # rec_renamed channel IDs are like "UAe001_NSP023".
    # This script wants channel IDs to be electrode IDs: "1", "2", ...
    rec_channel_ids = np.asarray(rec_renamed.get_channel_ids())
    selected_channel_ids = [rec_channel_ids[i] for i in valid_rows]

    ua_ids_1based = ua_elec[valid_rows].astype(int)
    mapped_region = ua_region[valid_rows].astype(np.int8)

    rec_mapped = rec_renamed.select_channels(
        channel_ids=selected_channel_ids
    ).rename_channels(
        new_channel_ids=[str(e) for e in ua_ids_1based]
    )

    return rec_mapped, ua_ids_1based, ua_port, mapped_region, ua_region_names

# =============================================================================
# Grouped Baseline LFP Processing - Utah Array
# =============================================================================

def process_baseline_group_utah(
    group_key: tuple,  # (port, depth)
    session_infos: list,  # List of dicts with session info
):
    """
    Process multiple baseline sessions for the same Port/Depth, aggregating epochs.
    Optimized to load each recording only once for all targets (A, B, None).
    """
    port, depth = group_key
    
    # Sanitize for filename
    def _sanitize(s):
        return str(s).replace(".", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
    
    depth_str = _sanitize(depth)
    port_str = _sanitize(port)
    
    # 1. Discovery phase: Identify which targets have data for each session
    all_events_by_br = {}  # {br_idx: {target_label: event_ms}}
    for sess_info in session_infos:
        br_idx = sess_info["br_idx"]
        all_events_by_br[br_idx] = {}
        for target in ["target_A", "target_B", ""]:
            cat_dir = PERI_ROOT / "control_reaches"
            if target:
                cat_dir = cat_dir / target
            if not cat_dir.exists():
                continue
            
            pattern = f"peristim__*{sess_info['session']}*BR_{br_idx:03d}*.npz"
            matches = list(cat_dir.glob(pattern))
            for p in matches:
                try:
                    bl_data = np.load(str(p), allow_pickle=True)
                    if "event_ms" in bl_data and bl_data["event_ms"].size > 0:
                        label = target[-1] if target else "N"
                        all_events_by_br[br_idx][label] = bl_data["event_ms"].flatten()
                except:
                    pass

    # 2. Session-major loop: Load each recording once and extract epochs for all labels
    all_epochs_by_label = {"A": [], "B": [], "N": []}
    active_sessions_by_label = {"A": [], "B": [], "N": []}
    ua_ids_global = None

    for sess_info in session_infos:
        br_idx = sess_info["br_idx"]
        if not all_events_by_br.get(br_idx):
            continue
            
        labels_to_proc = list(all_events_by_br[br_idx].keys())
        
        # Find Blackrock ns6 file
        ns6_files = [p for p in BR_ROOT.iterdir() if p.is_file() and f"_{br_idx:03d}" in p.name and p.suffix == '.ns6']
        if not ns6_files:
            continue
        
        sess_path = ns6_files[0]
        try:
            print(f"      [Baseline] Loading BR {br_idx:03d} for labels: {labels_to_proc}")
            rec_raw = se.read_blackrock(str(sess_path), stream_name='nsx6', all_annotations=True)
            rec_mapped, ua_ids_1based, ua_port, ua_region, ua_region_names = _apply_native_ua_mapping(rec_raw, UA_MAP, br_idx, METADATA_CSV)
            if rec_mapped is None:
                continue
            
            if ua_ids_global is None:
                ua_ids_global = ua_ids_1based
            
            fs_native = rec_mapped.get_sampling_frequency()
            rec_duration_ms = rec_raw.get_total_duration() * 1000.0
            
            # SI Pipeline (LFP Preprocessing)
            rec_proc = build_lfp_preprocessing_pipeline(
                rec_mapped,
                stim_indices_native=np.array([]),
                curr_sess_name=sess_path.name
            )
            n_samps_total = rec_proc.get_total_samples()
            
            pad_samps = int(PAD_MS * TARGET_FS / 1000.0)
            pre_samps = int(EPOCH_PRE_MS * TARGET_FS / 1000) + pad_samps
            post_samps = int(EPOCH_POST_MS * TARGET_FS / 1000) + pad_samps

            for label in labels_to_proc:
                sess_ir_ms = all_events_by_br[br_idx][label]
                
                # Filter events to session duration
                sess_ir_ms_valid = sess_ir_ms[(sess_ir_ms >= 0) & (sess_ir_ms < rec_duration_ms)]
                if len(sess_ir_ms_valid) == 0:
                    continue
                
                # Apply Baseline Trial Exclusions
                exclusion_key = f"baseline_port{port}_target{label}"
                exclude_indices = EXCLUDE_BASELINE_TRIALS.get(exclusion_key, [])
                if exclude_indices:
                    valid_mask = np.ones(len(sess_ir_ms_valid), dtype=bool)
                    for e_idx in exclude_indices:
                        if 0 <= e_idx < len(sess_ir_ms_valid):
                            valid_mask[e_idx] = False
                    sess_ir_ms_valid = sess_ir_ms_valid[valid_mask]

                if len(sess_ir_ms_valid) == 0:
                    continue
                
                # Extract epochs
                stim_start_samps = (sess_ir_ms_valid * fs_native / 1000.0).astype(np.int64)
                stim_indices_res = (stim_start_samps * TARGET_FS / fs_native).astype(np.int64)
                
                segs_list = []
                for idx in stim_indices_res:
                    if (idx - pre_samps >= 0) and (idx + post_samps <= n_samps_total):
                        seg = extract_single_epoch(rec_proc, idx, pre_samps, post_samps)
                        if seg is not None:
                            segs_list.append(seg)
                
                if segments := segs_list:
                    all_epochs_by_label[label].append(np.stack(segments, axis=0))
                    active_sessions_by_label[label].append(sess_info)
            
            del rec_raw, rec_mapped, rec_proc
            gc.collect()
            
        except Exception as e:
            print(f"        [ERROR] BR {br_idx}: {e}")

    # 3. Final processing per target label
    for label in ["A", "B", "N"]:
        if not all_epochs_by_label[label]:
            continue
            
        target_suffix = f"_target_{'target_'+label if label != 'N' else ''}".replace("target_target_", "target_")
        if label == "N":
            target_suffix = ""
        else:
            target_suffix = f"_target_{label}"
        
        out_name = f"aligned_lfp__baseline__Depth_{depth_str}_port_{port_str}{target_suffix}.npz"
        out_dir = UA_LFP_CKPT_ROOT / "control_reaches"
        if label != "N":
            out_dir = out_dir / f"target_{label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        final_out_npz = out_dir / out_name

        print(f"\n[Baseline UA] Aggregating Target {label}: {len(all_epochs_by_label[label])} sessions involved.")
        concat_epochs = np.concatenate(all_epochs_by_label[label], axis=0)
        n_time = concat_epochs.shape[2]
        rel_t_epoch = (np.arange(n_time) / TARGET_FS * 1000.0) - (EPOCH_PRE_MS + PAD_MS)

        # Spectral interpolation to remove 60 Hz line noise cleanly
        print(f"    [Cleaning] Spectral interpolation (60, 120 Hz)...")
        concat_epochs_clean = blank_notch_epochs(
            concat_epochs, 
            fs=TARGET_FS, 
            notch_freqs=[60.0, 120.0],
            bandwidth=10.0  # 55-65 Hz and 115-125 Hz
        )

        WIN_PRE = (-EPOCH_PRE_MS, -BLANK_PRE_MS)
        WIN_POST = (BLANK_POST_MS, EPOCH_POST_MS - BLANK_PRE_MS)
        WIN_FULL = (-EPOCH_PRE_MS, EPOCH_POST_MS)
        
        bb_full, t_full = slice_epoch(concat_epochs_clean, rel_t_epoch, WIN_FULL)
        _, t_pre = slice_epoch(concat_epochs_clean, rel_t_epoch, WIN_PRE)
        _, t_post = slice_epoch(concat_epochs_clean, rel_t_epoch, WIN_POST)
        
        results = {
            "broadband_full": bb_full,
        }
        
        # for band_name, (low, high) in LFP_BANDS.items():
        #     print(f"    Filtering {band_name}...")
        #     filt_epochs = filter_zero_phase(concat_epochs_clean, TARGET_FS, low, high)
        #     results[f'{band_name}_full'], _ = slice_epoch(filt_epochs, rel_t_epoch, WIN_FULL)
        #     results[f'{band_name}_pre'], _ = slice_epoch(filt_epochs, rel_t_epoch, WIN_PRE)
        #     results[f'{band_name}_post'], _ = slice_epoch(filt_epochs, rel_t_epoch, WIN_POST)
        
        active_sessions = active_sessions_by_label[label]
        save_dict = {
            "fs_lfp": TARGET_FS,
            "group_port": port,
            "group_depth": depth,
            "category": "control_reaches",
            "target": label if label != "N" else None,
            "sessions": [s["session"] for s in active_sessions],
            "br_indices": [s["br_idx"] for s in active_sessions],
            "n_trials": concat_epochs_clean.shape[0],
            "rel_time_pre": t_pre,
            "rel_time_post": t_post,
            "t_full_ms": t_full,
            "ua_ids_1based": ua_ids_global,
            "is_control": True,
            "n_pulses": 0,
            **results
        }
        
        np.savez_compressed(final_out_npz, **save_dict)
        print(f"    Saved -> {final_out_npz}")
    
    gc.collect()


# =============================================================================
# Helper: Resolve Blackrock Session ID and Path
# =============================================================================

def resolve_blackrock_session_id(br_idx: int, sess_name: str = "") -> tuple[Path | None, str]:
    """
    Given a Blackrock index (br_idx), find the Blackrock recording (.ns6, .ns2, etc.)
    and return (sess_path, sess_id).
    sess_id will be the directory name (if folder-based) or file stem (e.g. 'NRR_RW012_012').
    """
    fallback_id = f"{PARAMS.session}_{br_idx:03d}" if (PARAMS and hasattr(PARAMS, "session")) else f"BR_{br_idx:03d}"
    
    # Check directory structure
    matches_dir = [p for p in BR_ROOT.iterdir() if p.is_dir() and f"_{br_idx:03d}" in p.name]
    if matches_dir:
        dir_path = matches_dir[0]
        ns_files = list(dir_path.glob("*.ns*"))
        ns6 = [f for f in ns_files if f.suffix == '.ns6']
        ns2 = [f for f in ns_files if f.suffix == '.ns2']
        ns5 = [f for f in ns_files if f.suffix == '.ns5']
        sess_path = (ns6 or ns2 or ns5 or ns_files)[0] if (ns6 or ns2 or ns5 or ns_files) else None
        return sess_path, dir_path.name

    # Check flat file structure
    matches_files = [p for p in BR_ROOT.iterdir() if p.is_file() and f"_{br_idx:03d}" in p.name and p.suffix in ['.ns2', '.ns5', '.ns6']]
    if matches_files:
        ns6 = [f for f in matches_files if f.suffix == '.ns6']
        ns2 = [f for f in matches_files if f.suffix == '.ns2']
        sess_path = (ns6 or ns2 or matches_files)[0]
        return sess_path, sess_path.stem

    return None, fallback_id


# =============================================================================
# Main Utah Session Processing (Unified per Blackrock ID)
# =============================================================================

def process_utah_session(
    arg1,
    arg2=None,
    *args,
    shift_ms: float = 0.0,
    fs_intan: float = 30000.0,
    shift_sample: float = 0.0,
    sess_id: str | None = None,
    sess_path: Path | None = None,
    intan_rows: list[dict] | None = None,
    **kwargs
):
    if isinstance(arg1, int):
        br_idx = arg1
        if isinstance(arg2, list):
            intan_rows = arg2
        sess_name = kwargs.get("sess_name", "")
    else:
        sess_name = str(arg1)
        br_idx = int(arg2) if arg2 is not None else int(kwargs.get("br_idx", -1))
        if len(args) >= 1: shift_ms = float(args[0])
        if len(args) >= 2: fs_intan = float(args[1])
        if len(args) >= 3: shift_sample = float(args[2])

    if sess_path is None or sess_id is None:
        sess_path, sess_id = resolve_blackrock_session_id(br_idx, sess_name)

    if sess_path is None:
        print(f"[UA] No session folder or files found for BR {br_idx:03d}")
        return
    
    # Identify all peristim files to process for this Blackrock ID
    peristim_files = find_peristim_files(br_idx)

    print(f"  [UA] Found {len(peristim_files)} PeriStim files for BR={br_idx:03d}")
    for cat, target, p in peristim_files:
        print(f"      cat={cat}, target={target}, file={p.name}")

    # Group candidate events by (cat, target)
    events_by_cat_target: dict[tuple, list[np.ndarray]] = {}
    for cat, target, p in peristim_files:
        try:
            ps = np.load(p, allow_pickle=True)
            event_ms = None
            if "event_ms" in ps and ps["event_ms"].size > 0:
                event_ms = ps["event_ms"]
            elif "stim_ms" in ps and ps["stim_ms"].size > 0:
                event_ms = ps["stim_ms"]

            if event_ms is not None:
                events_by_cat_target.setdefault((cat, target), []).append(np.asarray(event_ms).flatten())
        except Exception as e:
            print(f"  [UA] Failed to load peristim {p.name}: {e}")

    # Pre-raw-load per-output skip check
    if SKIP_EXISTING:
        if events_by_cat_target:
            pending_events = {}
            for (cat, target), ev_list in events_by_cat_target.items():
                target_str = f"_target_{target}" if target else ""
                out_name = f"aligned_lfp__{sess_id}_{cat}{target_str}.npz"
                target_dir = UA_LFP_CKPT_ROOT / cat
                if target:
                    target_dir = target_dir / target
                expected_out = target_dir / out_name

                if expected_out.exists():
                    print(f"  [UA] Pre-skip existing output: {expected_out.name}")
                else:
                    pending_events[(cat, target)] = ev_list

            if not pending_events:
                print(f"[UA] Skipping {sess_id}; all PeriStim UA_LFP outputs already exist.")
                return
            events_by_cat_target = pending_events
        else:
            out_name = f"aligned_lfp__{sess_id}_stim_reaches.npz"
            expected_out = UA_LFP_CKPT_ROOT / "stim_reaches" / out_name
            if expected_out.exists():
                print(f"[UA] Skipping {sess_id}; fallback UA_LFP output already exists: {expected_out.name}")
                return

    print(f"[UA] Processing {sess_id} (Source: {sess_path.name})")
    
    # Load Recording
    try:
        print(f"  Loading Blackrock stream: nsx6")
        rec_raw = se.read_blackrock(str(sess_path), stream_name='nsx6', all_annotations=True)
    except Exception as e:
        print(f"[UA] Failed to load nsx6: {e}")
        return

    # Mapping
    rec_mapped, ua_ids_1based, ua_port, ua_region, ua_region_names = _apply_native_ua_mapping(rec_raw, UA_MAP, br_idx, METADATA_CSV)
    if rec_mapped is None:
        print("[WARN] No mapped channels found.")
        return
    
    fs_native = rec_mapped.get_sampling_frequency()
    ppm_corr = float(PARAMS.preprocessing.get("ppm_correction", -13.951))

    # IPCA artifact correction
    rows_to_check = intan_rows if intan_rows else [{
        "shift_sample": shift_sample, "fs_intan": fs_intan, "shift_ms": shift_ms
    }]

    all_starts_ua = []
    all_ends_ua = []
    for r in rows_to_check:
        i_fn = r.get("intan_filename") or r.get("session") or ""
        s_smp = float(r.get("shift_sample", 0.0))
        f_int = float(r.get("fs_intan", 30000.0))
        stim_npz_path, _ = rcp.stim_npz_path_from_br_idx(br_idx, METADATA_CSV, NPRW_AUX_DATA)
        if not (stim_npz_path and Path(stim_npz_path).exists()) and i_fn:
            cand = NPRW_AUX_DATA / f"{i_fn}_Intan_streams" / "stim_stream.npz"
            if cand.exists():
                stim_npz_path = cand
        if stim_npz_path and Path(stim_npz_path).exists():
            stim = rcp.load_stim_detection(stim_npz_path)
            b_bounds = stim.get("block_bounds_samples", [])
            if len(b_bounds) > 0:
                s_ua = rcp.intan_samples_to_br_samples(b_bounds[:, 0], s_smp, f_int, fs_native, ppm_corr)
                e_ua = rcp.intan_samples_to_br_samples(b_bounds[:, 1], s_smp, f_int, fs_native, ppm_corr)
                all_starts_ua.append(s_ua)
                all_ends_ua.append(e_ua)

    if all_starts_ua:
        starts_ua = np.concatenate(all_starts_ua)
        ends_ua = np.concatenate(all_ends_ua)
        n_total = rec_mapped.get_num_samples()
        ends_ua = np.minimum(ends_ua, n_total)
        valid = (ends_ua > starts_ua) & (starts_ua >= 0) & (ends_ua <= n_total)
        if valid.any():
            primary_fs_intan = float(rows_to_check[0].get("fs_intan", 30000.0))
            primary_shift_smp = float(rows_to_check[0].get("shift_sample", 0.0))
            rec_mapped = apply_ipca_correction(
                rec_mapped, starts_ua[valid], ends_ua[valid],
                br_idx=br_idx, fs_intan=primary_fs_intan, shift_samp_intan=primary_shift_smp,
                ua_region=ua_region, ua_region_names=ua_region_names
            )

    # Fallback if no peristim files found
    if not events_by_cat_target:
        if all_starts_ua:
            stim_ms_ua = starts_ua[valid] * 1000.0 / fs_native
            events_by_cat_target[("stim_reaches", None)] = [stim_ms_ua]
        else:
            print(f"  [UA] No events found for {sess_id}. Skipping.")
            return

    # Build SI Pipeline
    rec_proc = build_lfp_preprocessing_pipeline(
        rec_mapped, 
        stim_indices_native=np.array([], dtype=int),
        curr_sess_name=sess_path.name,
        local_radii=None 
    )

    pad_samps = int(PAD_MS * TARGET_FS / 1000.0)
    pre_samps = int(EPOCH_PRE_MS * TARGET_FS / 1000) + pad_samps
    post_samps = int(EPOCH_POST_MS * TARGET_FS / 1000) + pad_samps
    n_samps_total = rec_proc.get_total_samples()

    WIN_PRE = (-EPOCH_PRE_MS, -BLANK_PRE_MS)
    WIN_POST = (BLANK_POST_MS, EPOCH_POST_MS - BLANK_PRE_MS)
    WIN_FULL = (-EPOCH_PRE_MS, EPOCH_POST_MS)
    n_time = pre_samps + post_samps
    rel_t_epoch = (np.arange(n_time) / TARGET_FS * 1000.0) - (EPOCH_PRE_MS + PAD_MS)

    for (cat, target), ev_list in events_by_cat_target.items():
        target_str = f"_target_{target}" if target else ""
        out_name = f"aligned_lfp__{sess_id}_{cat}{target_str}.npz"
        target_dir = UA_LFP_CKPT_ROOT / cat
        if target:
            target_dir = target_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)
        final_out_npz = target_dir / out_name

        combined_ev = np.unique(np.concatenate(ev_list))
        print(f"  [UA] Segmenting {cat}{target_str} ({len(combined_ev)} events)...")

        stim_indices_res = (combined_ev * TARGET_FS / 1000.0).astype(np.int64)
        segs_list_bb = []
        valid_stim_ms = []

        for idx_res, stim_time in zip(stim_indices_res, combined_ev):
            if (idx_res - pre_samps >= 0) and (idx_res + post_samps <= n_samps_total):
                seg = extract_single_epoch(rec_proc, idx_res, pre_samps, post_samps)
                if seg is not None:
                    segs_list_bb.append(seg)
                    valid_stim_ms.append(stim_time)

        if not segs_list_bb:
            print(f"    [WARN] No valid segments for {cat}{target_str}.")
            continue

        segs_bb = np.stack(segs_list_bb, axis=0)
        segs_bb_clean = blank_notch_epochs(
            segs_bb, 
            fs=TARGET_FS, 
            notch_freqs=[60.0, 120.0],
            bandwidth=10.0
        )

        results = {}
        results['broadband_full'], t_full = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_FULL)
        _, t_pre = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_PRE)
        _, t_post = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_POST)

        is_stim_cat = "stim" in cat.lower()
        n_pulses, stim_dur_nominal, _ = _get_stim_params_for_br(br_idx, is_stim_cat=is_stim_cat)

        save_dict = {
            "fs_lfp": TARGET_FS,
            "ua_ids_1based": ua_ids_1based,
            "session": sess_id,
            "category": cat,
            "target": target if target else "none",
            "stim_ms": np.array(valid_stim_ms),
            "stim_dur_ms": stim_dur_nominal if is_stim_cat else 0.0,
            "n_pulses": n_pulses,
            "is_control": not is_stim_cat or n_pulses == 0,
            "rel_time_pre": t_pre,
            "rel_time_post": t_post,
            "t_full_ms": t_full,
            **results
        }
        
        np.savez_compressed(final_out_npz, **save_dict)
        print(f"    Saved UA LFP -> {final_out_npz}")

    del rec_proc, rec_mapped, rec_raw
    gc.collect()


# =============================================================================
# Main NPRW Session Processing (Concatenate all constituent NPRW files by BR ID)
# =============================================================================

def process_nprw_session(
    arg1,
    arg2=None,
    *args,
    intan_rows: list[dict] | None = None,
    sess_id: str | None = None,
    **kwargs
):
    """
    Process NPRW (Intan) neural stream for LFP band analysis.
    Concatenates all constituent NPRW/Intan files falling under the same Blackrock ID.
    
    Notes:
    - NO IPCA artifact correction is applied (not applicable for Intan/NPRW stim site).
    - Blanking is applied from 20 ms before stim to 20 ms post stim end 
      (-20 ms to stim_dur + 20 ms) to remove massive stimulation transients.
    - broadband_full has the blanked stimulation window set to NaN.
    """
    if isinstance(arg1, int):
        br_idx = arg1
        if isinstance(arg2, list):
            intan_rows = arg2
        sess_name = kwargs.get("sess_name", "")
    elif isinstance(arg2, list):
        sess_path = Path(arg1) if not isinstance(arg1, Path) else arg1
        br_idx = int(kwargs.get("br_idx", -1))
        intan_rows = arg2
        sess_name = sess_path.name
    else:
        sess_path = Path(arg1) if not isinstance(arg1, Path) else arg1
        sess_name = str(arg2)
        intan_idx = int(args[0]) if len(args) > 0 else int(kwargs.get("intan_idx", -1))
        br_idx = int(args[1]) if len(args) > 1 else int(kwargs.get("br_idx", -1))
        intan_rows = [{"intan_filename": sess_name, "intan_idx": intan_idx, "br_idx": br_idx}]

    if sess_id is None:
        _, sess_id = resolve_blackrock_session_id(br_idx, sess_name)

    if not intan_rows:
        intan_rows = [{"intan_filename": sess_name, "intan_idx": -1, "br_idx": br_idx}]

    print(f"\n[NPRW] Processing unified Blackrock ID {sess_id} (BR {br_idx:03d}, {len(intan_rows)} constituent Intan files)")

    # Load geometry & mapping
    try:
        geom_path = rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
        mat_probe = loadmat(Path(geom_path))
        if "chanMap0ind" in mat_probe:
            intan_probe_mapping = mat_probe["chanMap0ind"].ravel()
        else:
            print(f"[NPRW] No chanMap0ind in {geom_path}")
            return
    except Exception as e:
        print(f"[NPRW] Failed to load probe geometry: {e}")
        return

    # Find peristim files for this BR ID
    peristim_files = find_peristim_files(br_idx)

    # Pre-skip check
    if SKIP_EXISTING:
        if peristim_files:
            cats_targets = set((cat, target) for cat, target, _ in peristim_files)
            all_exist = True
            for cat, target in cats_targets:
                target_str = f"_target_{target}" if target else ""
                expected_out = NPRW_LFP_CKPT_ROOT / cat
                if target:
                    expected_out = expected_out / target
                expected_out = expected_out / f"aligned_lfp__{sess_id}_{cat}{target_str}.npz"
                if not expected_out.exists():
                    all_exist = False
                    break
            if all_exist:
                print(f"[NPRW] Skipping {sess_id}; all PeriStim NPRW outputs already exist.")
                return
        else:
            expected_out = NPRW_LFP_CKPT_ROOT / "stim_reaches" / f"aligned_lfp__{sess_id}_stim_reaches.npz"
            if expected_out.exists():
                print(f"[NPRW] Skipping {sess_id}; fallback NPRW output already exists: {expected_out.name}")
                return

    # Load and preprocess each unique constituent Intan recording
    intan_stream = NPRW_CFG.get("neural_data_stream")
    ppm_corr = float(PARAMS.preprocessing.get("ppm_correction", -13.951))
    clock_scale = 1.0 + (ppm_corr / 1e6)

    loaded_recordings: dict[str, dict] = {}
    for r in intan_rows:
        i_fn = r.get("intan_filename") or r.get("session")
        if not i_fn or i_fn in loaded_recordings:
            continue
        matches = list(INTAN_ROOT.glob(f"*{i_fn}*"))
        valid_folders = [p for p in matches if p.is_dir()]
        if not valid_folders:
            print(f"  [NPRW] Warning: Intan folder '{i_fn}' not found in {INTAN_ROOT}")
            continue
        sess_path_i = valid_folders[0]
        print(f"  [NPRW] Loading Intan stream: {intan_stream} from {sess_path_i.name}")
        try:
            rec_raw = se.read_split_intan_files(sess_path_i, mode="concatenate", stream_name=intan_stream, use_names_as_ids=True)
            rec_raw = spre.unsigned_to_signed(rec_raw)
            rec_mapped = rcp.reorder_recording_to_geometry(rec_raw, intan_probe_mapping)
            rec_proc = build_lfp_preprocessing_pipeline(
                rec_mapped, stim_indices_native=np.array([], dtype=int),
                curr_sess_name=sess_path_i.name, local_radii=None
            )
            fs_native = rec_mapped.get_sampling_frequency()

            stim_npz_path = NPRW_AUX_DATA / f"{sess_path_i.name}_Intan_streams" / "stim_stream.npz"
            stim_dur_ms = 100.0
            if stim_npz_path.exists():
                stim_det = rcp.load_stim_detection(stim_npz_path)
                b_bounds = stim_det.get("block_bounds_samples", [])
                if len(b_bounds) > 0:
                    durs = (b_bounds[:, 1] - b_bounds[:, 0]) * 1000.0 / fs_native
                    stim_dur_ms = float(np.median(durs))
            else:
                _, stim_dur_ms, _ = _get_stim_params_for_br(br_idx, is_stim_cat=True)

            loaded_recordings[i_fn] = {
                "rec_proc": rec_proc,
                "rec_raw": rec_raw,
                "rec_mapped": rec_mapped,
                "sess_path": sess_path_i,
                "intan_idx": int(r.get("intan_idx", -1)),
                "shift_ms": float(r.get("shift_ms", 0.0)),
                "shift_sample": float(r.get("shift_sample", 0.0)),
                "fs_intan": float(r.get("fs_intan", 30000.0)),
                "fs_native": fs_native,
                "stim_dur_ms": stim_dur_ms,
                "n_samps_proc": rec_proc.get_total_samples(),
            }
        except Exception as e:
            print(f"  [NPRW] Failed to load Intan stream {i_fn}: {e}")

    if not loaded_recordings:
        print(f"  [NPRW] No valid Intan recordings could be loaded for BR {br_idx:03d}. Skipping.")
        return

    # Group peristim files or stim detections by (cat, target)
    events_by_cat_target: dict[tuple, list[np.ndarray]] = {}

    if peristim_files:
        for cat, target, p in peristim_files:
            try:
                ps = np.load(p, allow_pickle=True)
                event_ms = None
                if "event_ms" in ps and ps["event_ms"].size > 0:
                    event_ms = ps["event_ms"]
                elif "stim_ms" in ps and ps["stim_ms"].size > 0:
                    event_ms = ps["stim_ms"]

                if event_ms is not None:
                    ev_br = np.asarray(event_ms, dtype=np.float64).flatten()
                    events_by_cat_target.setdefault((cat, target), []).append(ev_br)
            except Exception as e:
                print(f"  [NPRW] Failed to load peristim {p.name}: {e}")
    else:
        for i_fn, r_info in loaded_recordings.items():
            stim_npz_path = NPRW_AUX_DATA / f"{r_info['sess_path'].name}_Intan_streams" / "stim_stream.npz"
            if stim_npz_path.exists():
                stim_det = rcp.load_stim_detection(stim_npz_path)
                b_bounds = stim_det.get("block_bounds_samples", [])
                if len(b_bounds) > 0:
                    stim_samps = b_bounds[:, 0]
                    stim_ms_br = rcp.intan_samples_to_br_ms(stim_samps, r_info["shift_sample"], r_info["fs_intan"], ppm_corr)
                    events_by_cat_target.setdefault(("stim_reaches", None), []).append(stim_ms_br)

    if not events_by_cat_target:
        print(f"  [NPRW] No valid event sets found for {sess_id}. Skipping.")
        return

    ARTRMV_MS_BEFORE = 20.0
    ARTRMV_TAIL_MS = 20.0

    pad_samps = int(PAD_MS * TARGET_FS / 1000.0)
    pre_samps = int(EPOCH_PRE_MS * TARGET_FS / 1000) + pad_samps
    post_samps = int(EPOCH_POST_MS * TARGET_FS / 1000) + pad_samps
    n_time = pre_samps + post_samps
    rel_t_epoch = (np.arange(n_time) / TARGET_FS * 1000.0) - (EPOCH_PRE_MS + PAD_MS)

    for (cat, target), ev_list in events_by_cat_target.items():
        target_str = f"_target_{target}" if target else ""
        out_name = f"aligned_lfp__{sess_id}_{cat}{target_str}.npz"
        
        target_dir = NPRW_LFP_CKPT_ROOT / cat
        if target:
            target_dir = target_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)
        final_out_npz = target_dir / out_name

        if SKIP_EXISTING and final_out_npz.exists():
            print(f"  [NPRW] Skipping existing {final_out_npz.name}")
            continue

        segs_list_bb = []
        valid_stim_ms = []
        stim_dur_list = []

        all_candidate_events = np.unique(np.concatenate(ev_list))
        print(f"  [NPRW] Segmenting concatenated {cat}{target_str} ({len(all_candidate_events)} candidate events across {len(loaded_recordings)} files)...")

        for ev_br in all_candidate_events:
            for r_info in loaded_recordings.values():
                rec_proc = r_info["rec_proc"]
                shift_ms = r_info["shift_ms"]
                n_samps_total = r_info["n_samps_proc"]

                # Convert Blackrock event times to native Intan ms, then target_fs sample indices
                ev_intan_ms = (ev_br / clock_scale) + shift_ms
                idx_res = int(round(ev_intan_ms * TARGET_FS / 1000.0))

                if (idx_res - pre_samps >= 0) and (idx_res + post_samps <= n_samps_total):
                    seg = extract_single_epoch(rec_proc, idx_res, pre_samps, post_samps)
                    if seg is not None:
                        segs_list_bb.append(seg)
                        valid_stim_ms.append(ev_br)
                        stim_dur_list.append(r_info["stim_dur_ms"])
                        break

        if not segs_list_bb:
            print(f"    [WARN] No valid segments extracted for {cat}{target_str}.")
            continue

        segs_bb = np.stack(segs_list_bb, axis=0)
        sort_order = np.argsort(valid_stim_ms)
        segs_bb = segs_bb[sort_order]
        valid_stim_ms = [valid_stim_ms[i] for i in sort_order]

        stim_dur_ms = float(np.median(stim_dur_list)) if stim_dur_list else 100.0
        blank_win_start = -ARTRMV_MS_BEFORE
        blank_win_end = stim_dur_ms + ARTRMV_TAIL_MS

        WIN_PRE = (-EPOCH_PRE_MS, blank_win_start)
        WIN_POST = (blank_win_end, EPOCH_POST_MS - ARTRMV_MS_BEFORE)
        WIN_FULL = (-EPOCH_PRE_MS, EPOCH_POST_MS)

        segs_bb_clean = blank_notch_epochs(
            segs_bb,
            fs=TARGET_FS,
            notch_freqs=[60.0, 120.0],
            bandwidth=10.0
        )

        bb_full, t_full = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_FULL)
        _, t_pre = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_PRE)
        _, t_post = slice_epoch(segs_bb_clean, rel_t_epoch, WIN_POST)

        is_stim_cat = "stim" in cat.lower()
        bb_full_blanked = bb_full.copy()
        if is_stim_cat:
            blank_mask = (t_full >= blank_win_start) & (t_full <= blank_win_end)
            bb_full_blanked[:, :, blank_mask] = np.nan

        results = {
            'broadband_full': bb_full_blanked,
            'broadband_full_raw': bb_full,
        }

        n_pulses, stim_dur_ms, _ = _get_stim_params_for_br(br_idx, is_stim_cat=is_stim_cat)

        save_dict = {
            "fs_lfp": TARGET_FS,
            "session": sess_id,
            "br_idx": br_idx,
            "intan_folders": [r["sess_path"].name for r in loaded_recordings.values()],
            "intan_indices": [r["intan_idx"] for r in loaded_recordings.values()],
            "category": cat,
            "target": target if target else "none",
            "stim_ms": np.array(valid_stim_ms),
            "stim_dur_ms": stim_dur_ms,
            "n_pulses": n_pulses,
            "is_control": not is_stim_cat or n_pulses == 0,
            "rel_time_pre": t_pre,
            "rel_time_post": t_post,
            "t_full_ms": t_full,
            "n_channels": segs_bb.shape[1],
            **results
        }
        
        np.savez_compressed(final_out_npz, **save_dict)
        print(f"    Saved unified concatenated NPRW LFP ({len(valid_stim_ms)} trials from {len(loaded_recordings)} files) -> {final_out_npz}")

    del loaded_recordings
    gc.collect()


# =============================================================================
# Stim Check Helper
# =============================================================================

def quick_check_stim_for_br(br_idx: int, intan_rows: list[dict]) -> bool:
    """
    Check if stim file, peristim file, or aligned file exists and has events for this Blackrock ID.
    """
    # 1. Check if any peristim files exist on disk for this Blackrock ID
    peri = find_peristim_files(br_idx)
    if peri:
        return True

    # 2. Check each constituent Intan recording
    for r in intan_rows:
        i_fn = r.get("intan_filename") or r.get("session") or ""
        i_idx = int(r.get("intan_idx", -1))
        
        aligned_path = ALIGNED_CKPT_ROOT / f"aligned__{i_fn}__Intan_{i_idx:03d}__BR_{br_idx:03d}.npz"
        if aligned_path.exists():
            try:
                aln = np.load(aligned_path, allow_pickle=True)
                if "stim_ms" in aln and aln["stim_ms"].size > 0:
                    return True
            except:
                pass

        stim_npz = NPRW_AUX_DATA / f"{i_fn}_Intan_streams" / "stim_stream.npz"
        if stim_npz in STIM_CACHE:
            if STIM_CACHE[stim_npz]:
                return True
        elif stim_npz.exists():
            try:
                stim_data = rcp.load_stim_detection(stim_npz)
                bounds = stim_data.get("block_bounds_samples")
                has_s = bool(bounds is not None and len(bounds) > 0)
                STIM_CACHE[stim_npz] = has_s
                if has_s:
                    return True
            except:
                pass

    return False


def quick_check_stim(sess_path: Path, stream_name: str, intan_idx: int, br_idx: int) -> bool:
    return quick_check_stim_for_br(br_idx, [{"intan_filename": sess_path.name, "intan_idx": intan_idx}])


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    if not SHIFT_CSV.exists():
        print(f"Error: {SHIFT_CSV} not found. Cannot proceed with aligned analysis.")
        return

    with SHIFT_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    print(f"Found {len(rows)} entries in shift CSV.")

    PROCESS_ONLY = PARAMS.preprocessing.get("process_only")

    # Group shift rows by Blackrock ID (br_idx)
    br_to_intan_rows: dict[int, list[dict]] = {}
    for r in rows:
        try:
            b = int(r["br_idx"])
            br_to_intan_rows.setdefault(b, []).append(r)
        except Exception:
            continue

    if PROCESS_ONLY:
        keep_brs = set(int(x) for x in PROCESS_ONLY)
        br_to_intan_rows = {b: r_list for b, r_list in br_to_intan_rows.items() if b in keep_brs}
        print(f"[INFO] PROCESS_ONLY: filtered to BR indices: {sorted(br_to_intan_rows.keys())}")
    
    # Load Metadata to Identify Baseline Sessions
    baseline_br_idxs = set()
    baseline_groups = {}
    
    if METADATA_CSV.exists():
        print(f"\nLoading metadata from {METADATA_CSV.name}...")
        try:
            df_meta_raw = pd.read_csv(METADATA_CSV)
            df_meta = _normalize_metadata(df_meta_raw)
            
            groups = _find_baseline_groups(df_meta)
            if groups:
                print(f"Found {len(groups)} baseline groups: {list(groups.keys())}")
                
                if "br_file" in df_meta.columns:
                    br_col = pd.to_numeric(df_meta["br_file"], errors="coerce")
                    for (port, depth), idxs in groups.items():
                        for i in idxs:
                            br = br_col.iloc[i]
                            if pd.notna(br):
                                baseline_br_idxs.add(int(br))
                                
                for key in groups:
                    baseline_groups[key] = []
                    
        except Exception as e:
            print(f"[WARN] Failed to load metadata for baseline detection: {e}")
    else:
        print(f"[WARN] Metadata CSV not found: {METADATA_CSV}. Baseline grouping disabled.")
    
    print(f"Identified {len(baseline_br_idxs)} baseline BR indices: {sorted(baseline_br_idxs)}")
    
    # =========================================================================
    # Per-Blackrock ID Loop (Concatenates constituent NPRW files together)
    # =========================================================================
    for br_idx, intan_rows in sorted(br_to_intan_rows.items()):
        sess_path_br, sess_id = resolve_blackrock_session_id(br_idx)
        print(f"\n{'='*70}")
        print(f"[UNIFIED] Processing Blackrock ID BR_{br_idx:03d} -> Session ID: {sess_id}")
        print(f"          Constituent NPRW/Intan files ({len(intan_rows)}):")
        for r in intan_rows:
            i_name = r.get("intan_filename") or r.get("session")
            print(f"            - Intan {int(r.get('intan_idx', -1)):03d}: {i_name} (shift={float(r.get('shift_ms', 0)):.2f} ms)")
        print(f"{'='*70}")

        # Check if this is a baseline session
        if br_idx in baseline_br_idxs:
            print(f"\n[Baseline] BR {br_idx} is baseline. Registering constituent sessions for baseline grouping.")
            if METADATA_CSV.exists():
                try:
                    df_meta_raw = pd.read_csv(METADATA_CSV)
                    df_meta = _normalize_metadata(df_meta_raw)
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
                            port = str(port).upper().strip() or "UNKNOWN"
                            try:
                                depth = float(depth)
                            except:
                                depth = str(depth)
                            depth = str(depth)
                            
                            group_key = (port, depth)
                            if group_key not in baseline_groups:
                                baseline_groups[group_key] = []
                            
                            for r in intan_rows:
                                s_name = r.get("session") or Path(r.get("intan_filename", "")).stem
                                baseline_groups[group_key].append({
                                    "session": s_name,
                                    "br_idx": br_idx,
                                    "intan_idx": int(r.get("intan_idx", -1)),
                                    "shift_ms": float(r.get("shift_ms", 0.0)),
                                    "fs_intan": float(r.get("fs_intan", 30000.0)),
                                })
                except Exception as e:
                    print(f"    [WARN] Failed to identify group for session: {e}")
            continue

        # Check stim / peristim events
        has_stim = quick_check_stim_for_br(br_idx, intan_rows)
        if not has_stim:
            print(f"  -> No valid stim or peristim events found for BR {br_idx}. Skipping both NPRW and UA.")
            continue

        print(f"  -> Events confirmed. Proceeding with unified Blackrock ID {sess_id}...")

        # 1. Process Utah Array (ONCE per Blackrock ID)
        try:
            process_utah_session(br_idx, intan_rows, sess_id=sess_id, sess_path=sess_path_br)
        except Exception as e:
            print(f"[ERROR] UA BR_{br_idx:03d}: {e}")
            import traceback
            traceback.print_exc()

        # 2. Process NPRW (Concatenate all constituent NPRW files under Blackrock ID)
        try:
            process_nprw_session(br_idx, intan_rows, sess_id=sess_id)
        except Exception as e:
            print(f"[ERROR] NPRW BR_{br_idx:03d}: {e}")
            import traceback
            traceback.print_exc()

    # Process Baseline Groups
    if baseline_groups:
        print(f"\n{'='*60}")
        print("Processing Baseline Groups")
        print(f"{'='*60}")
        
        for group_key, session_infos in baseline_groups.items():
            if not session_infos:
                continue
            
            try:
                process_baseline_group_utah(group_key, session_infos)
            except Exception as e:
                print(f"[ERROR] Baseline UA group {group_key}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()