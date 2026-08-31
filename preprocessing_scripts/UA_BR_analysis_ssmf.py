""" 
    This script preprocesses the Blackrock data. Preprocessing and spike sorting are handled in Python using SpikeInterface (https://spikeinterface.readthedocs.io/)
    Steps:
        1. Load geometry and mapping (.xlsm file)
        2. Build .npz for auxiliary data (channels in .ns5 and .ns2 file, HR, touchscreen ... etc.)
        3. Extract target labels from touchscreen signal (with DLC kinematics fallback if touchscreen is unresponsive)
        4. Load neural data (.ns6) and config (location to data, geometry, and mapping)
        5. **Artifact Correction:**
        - Per-channel lag calibration using CSV lookup
        - Region-wise Incremental PCA (IPCA) artifact subtraction (rank 7 by default)
        - Generates diagnostic alignment plots
        6. High-pass filtering clean data
        7. **Spike Detection via Subspace Matched Filtering:**
        - Uses pre-computed extremum templates (median_extremum_basis_norm_UA_PortA_r3.npy)
        8. Saves .npz file per session with peaks, noise levels, and metadata
    Input:
        .ns6 files from Blackrock
    Output:
        pp_*.npz files after preprocessing
        aligned_*.npz files per condition with spike times
"""
import logging
logging.getLogger("spikeinterface").setLevel(logging.WARNING)
logging.getLogger("spikeinterface.core").setLevel(logging.WARNING)
logging.getLogger("spikeinterface.core.job_tools").setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore", message="The extractor is not serializable to file. The provenance will not be saved.")

import gc, csv
import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
from sklearn.decomposition import IncrementalPCA
import RCP_analysis as rcp
from spikeinterface.sortingcomponents.peak_detection import detect_peaks  # kept for fallback only
from RCP_analysis.python.functions.config_loading import *

import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    """Temporarily suppress stdout."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# FIX FOR WINDOWS MULTIPROCESSING IN SPIKEINTERFACE 0.104.1
# When child processes try to unpickle dynamically defined classes (IPCACorrectedRecording),
# SI tries to check the __version__ of the module. Since the module is __mp_main__, it crashes.
__version__ = "1.0.0"


# ---------- Config ----------
# Base paths from config_loading
SHIFT_CSV = METADATA_ROOT / "br_to_intan_shifts.csv"
BR_SESSION_FOLDERS = rcp.list_br_sessions(BR_ROOT)

PROCESS_ONLY = PARAMS.preprocessing.get("process_only")

RATES = PARAMS.UA_rate_est
BIN_MS     = RATES.get("bin_ms")
SIGMA_MS   = RATES.get("sigma_ms")
THRESH     = RATES.get("detect_threshold")
# THRESH = 16
PEAK_SIGN  = RATES.get("peak_sign")
ARTRMV_MS_BEFORE = float(RATES.get("remove_ms_before", 5.0))
ARTRMV_TAIL_MS   = float(RATES.get("remove_tail_ms_after", 5.0))


# --- IPCA Artifact Correction Settings ---
IPCA_PARAMS = PARAMS.IPCA_Params
USE_IPCA_CORRECTION  = IPCA_PARAMS.get("use_ipca", True)
IPCA_RANK            = IPCA_PARAMS.get("rank", 10)
IPCA_PULSE_WINDOW_MS = IPCA_PARAMS.get("pulse_window_ms", (-0.4, 0.3))
# ----------------------------------------

# --- Subspace Detector Settings ---
SUBSPCE_PARAMS = PARAMS.Subspace_Params
CFAR_ALPHA       = SUBSPCE_PARAMS.get("cfar_alpha", 1e-4)
LOCAL_SIGMA_GATE = SUBSPCE_PARAMS.get("local_sigma_gate", True)
AMP_GATE_K       = SUBSPCE_PARAMS.get("amp_gate_k", 3.5)
AMP_GATE_W_MS    = SUBSPCE_PARAMS.get("amp_gate_w_ms", 100.0)
# -----------------------------------


XLS = rcp.ua_excel_path(REPO_ROOT, PARAMS.probes)
UA_MAP = rcp.load_UA_mapping_from_excel(XLS) if XLS else None
if UA_MAP is None:
    raise RuntimeError("UA mapping required for mapping on NS6.")

# Sync channels
UA_CFG = PARAMS.probes.get("UA")
VOG_CH = int(UA_CFG.get("VOG_CH", 131))
CAMERA_SYNC_CH = int(UA_CFG.get("camera_sync_ch", 134))
HR_CH = int(UA_CFG.get("HR_ch", 136))
TRIANGLE_SYNC_CH = int(UA_CFG.get("triangle_sync_ch", 138))
TOUCHSCREEN_CH = int(UA_CFG.get("touchscreen_ch", 139))


TOUCHSCREEN_THRES_A = 2.0e6
TOUCHSCREEN_THRES_B = 3.3e6
TOUCHSCREEN_MIN_VALID = 1.5e6

UA_CKPT_OUT = UA_CKPT_ROOT

global_job_kwargs = dict(n_jobs=PARAMS.parallel_jobs, chunk_duration=PARAMS.chunk)
si.set_global_job_kwargs(**global_job_kwargs)


# ══════════════════════════════════════════════════════════════════════
# AMPLITUDE FILTERING OPTIONS
# ══════════════════════════════════════════════════════════════════════

# Absolute thresholds (µV) - same for all channels
MIN_AMPLITUDE_UV = None
MAX_AMPLITUDE_UV = 500.0

MIN_SNR = 2.5   # Minimum signal-to-noise ratio (amplitude / noise_level) - 2.0 is the absolute minimum


def filter_peaks_by_amplitude_fast(peaks, recording, noise_levels, 
                                    min_amplitude_uv=None, 
                                    max_amplitude_uv=None,
                                    min_snr=None,
                                    max_snr=None,
                                    peak_sign="neg",
                                    window_samples=20):
    """
    Fast vectorized amplitude/SNR filtering using batch waveform extraction.
    """
    from spikeinterface.core import extract_waveforms
    
    n_peaks = len(peaks)
    if n_peaks == 0:
        return peaks, np.array([]), np.array([])
    
    print(f"  [Amplitude Filter] Extracting amplitudes for {n_peaks:,} peaks...")
    
    fs = recording.get_sampling_frequency()
    n_channels = recording.get_num_channels()
    channel_ids = recording.get_channel_ids()
    
    # Get gain for µV conversion
    try:
        gains = recording.get_property('gain_to_uV')
        if gains is None:
            gains = np.ones(n_channels)
    except:
        gains = np.ones(n_channels)
    
    noise_levels_uv = noise_levels * gains
    
    # ══════════════════════════════════════════════════════════════════════
    # BATCH EXTRACTION: Load chunks of the recording and find amplitudes
    # ══════════════════════════════════════════════════════════════════════
    
    amplitudes = np.zeros(n_peaks, dtype=np.float32)
    
    # Sort peaks by sample index for efficient sequential reading
    sort_idx = np.argsort(peaks['sample_index'])
    sorted_peaks = peaks[sort_idx]
    
    # Process in large time chunks (e.g., 10 seconds at a time)
    chunk_duration_samples = int(10.0 * fs)  # 10 seconds
    n_samples_total = recording.get_num_samples()
    
    peak_idx = 0  # Current position in sorted_peaks
    
    for chunk_start in range(0, n_samples_total, chunk_duration_samples):
        chunk_end = min(chunk_start + chunk_duration_samples, n_samples_total)
        
        # Find peaks in this chunk (with padding for window)
        chunk_start_padded = max(0, chunk_start - window_samples)
        chunk_end_padded = min(n_samples_total, chunk_end + window_samples)
        
        # Find which peaks fall in this chunk
        peaks_in_chunk_mask = (
            (sorted_peaks['sample_index'] >= chunk_start) & 
            (sorted_peaks['sample_index'] < chunk_end)
        )
        peaks_in_chunk_idx = np.where(peaks_in_chunk_mask)[0]
        
        if len(peaks_in_chunk_idx) == 0:
            continue
        
        # Load entire chunk at once (FAST!)
        traces = recording.get_traces(
            start_frame=chunk_start_padded, 
            end_frame=chunk_end_padded,
            return_in_uV=True
        )  # Shape: (n_samples_in_chunk, n_channels)
        
        # Extract amplitude for each peak in this chunk
        for idx in peaks_in_chunk_idx:
            peak = sorted_peaks[idx]
            sample_idx = peak['sample_index']
            ch_idx = peak['channel_index']
            
            # Local indices within this chunk
            local_sample = sample_idx - chunk_start_padded
            s0 = max(0, local_sample - window_samples)
            s1 = min(traces.shape[0], local_sample + window_samples)
            
            snippet = traces[s0:s1, ch_idx]
            
            if peak_sign == "neg":
                amp = np.abs(np.min(snippet))
            elif peak_sign == "pos":
                amp = np.max(snippet)
            else:
                amp = np.max(np.abs(snippet))
            
            amplitudes[idx] = amp
        
        # Progress update
        n_done = peaks_in_chunk_idx[-1] + 1 if len(peaks_in_chunk_idx) > 0 else 0
        if n_done > 0 and n_done % 500000 < len(peaks_in_chunk_idx):
            print(f"    Processed {n_done:,}/{n_peaks:,} peaks...")
    
    # ══════════════════════════════════════════════════════════════════════
    # UN-SORT amplitudes back to original peak order
    # ══════════════════════════════════════════════════════════════════════
    unsort_idx = np.argsort(sort_idx)
    amplitudes = amplitudes[unsort_idx]
    
    # ══════════════════════════════════════════════════════════════════════
    # CALCULATE SNR (vectorized)
    # ══════════════════════════════════════════════════════════════════════
    channel_indices = peaks['channel_index']
    snr_values = amplitudes / noise_levels_uv[channel_indices]
    
    # ══════════════════════════════════════════════════════════════════════
    # APPLY THRESHOLDS (vectorized)
    # ══════════════════════════════════════════════════════════════════════
    keep_mask = np.ones(n_peaks, dtype=bool)
    
    reject_min_uv = 0
    reject_max_uv = 0
    reject_min_snr = 0
    reject_max_snr = 0
    
    if min_amplitude_uv is not None:
        bad = amplitudes < min_amplitude_uv
        reject_min_uv = np.sum(bad & keep_mask)
        keep_mask &= ~bad
    
    if max_amplitude_uv is not None:
        bad = amplitudes > max_amplitude_uv
        reject_max_uv = np.sum(bad & keep_mask)
        keep_mask &= ~bad
    
    if min_snr is not None:
        bad = snr_values < min_snr
        reject_min_snr = np.sum(bad & keep_mask)
        keep_mask &= ~bad
    
    if max_snr is not None:
        bad = snr_values > max_snr
        reject_max_snr = np.sum(bad & keep_mask)
        keep_mask &= ~bad
    
    # ══════════════════════════════════════════════════════════════════════
    # OUTPUT
    # ══════════════════════════════════════════════════════════════════════
    filtered_peaks = peaks[keep_mask]
    kept_amplitudes = amplitudes[keep_mask]
    kept_snr = snr_values[keep_mask]
    
    n_removed = n_peaks - len(filtered_peaks)
    print(f"[Amplitude Filter] Results:")
    print(f"  Total peaks: {n_peaks:,}")
    print(f"  Kept: {len(filtered_peaks):,} ({100*len(filtered_peaks)/n_peaks:.1f}%)")
    print(f"  Removed: {n_removed:,} ({100*n_removed/n_peaks:.1f}%)")
    
    if min_amplitude_uv is not None:
        print(f"    - Below {min_amplitude_uv} µV: {reject_min_uv:,}")
    if max_amplitude_uv is not None:
        print(f"    - Above {max_amplitude_uv} µV: {reject_max_uv:,}")
    if min_snr is not None:
        print(f"    - Below SNR {min_snr}: {reject_min_snr:,}")
    if max_snr is not None:
        print(f"    - Above SNR {max_snr}: {reject_max_snr:,}")
    
    if len(kept_amplitudes) > 0:
        print(f"  Kept amplitude range: {kept_amplitudes.min():.1f} - {kept_amplitudes.max():.1f} µV")
        print(f"  Kept SNR range: {kept_snr.min():.1f} - {kept_snr.max():.1f}")
    
    return filtered_peaks, kept_amplitudes, kept_snr

def main():
    sess_folders = BR_SESSION_FOLDERS

    print("Found session folders:", len(sess_folders))

    # Filter to PROCESS_ONLY BR indices if specified
    if PROCESS_ONLY:
        original_count = len(sess_folders)
        filtered = []
        for br_idx in PROCESS_ONLY:
            match = [s for s in sess_folders if int(s.name.split("_")[-1]) == br_idx]
            if not match:
                print(f"[WARN] PROCESS_ONLY BR index {br_idx} not found in session folders; skipping.")
                continue
            filtered.extend(match)
        print(f"[INFO] PROCESS_ONLY: {len(filtered)}/{original_count} sessions selected.")
        sess_folders = filtered

    # --- Subspace Basis Configuration ---
    # Prefer a pre-built (N, rank) orthonormal basis; fall back to building one
    # on-the-fly from the existing matched-filter template via truncated SVD.
    _basis_candidates = [
        REPO_ROOT / 'config' / "median_extremum_basis_norm_UA_PortA_r3.npy",
    ]
    _template_candidates = [
        REPO_ROOT / 'config' / "waveform_templates" / "median_extremum_templates_norm_UA_PortB.npy",
        REPO_ROOT / 'config' / 'waveform_templates' / "median_extremum_templates_norm_NPRW.npy", # just in case
    ]

    ua_basis = None
    for _bp in _basis_candidates:
        if _bp.exists():
            ua_basis = np.load(_bp)
            # print(f"[Subspace] Loaded pre-built basis: {_bp.name}, shape={ua_basis.shape}")
            break

    if ua_basis is None:
        # Build basis on-the-fly from the matched-filter template
        for _tp in _template_candidates:
            if _tp.exists():
                _tpl = np.load(_tp)
                ua_basis = rcp.build_subspace_basis(_tpl[np.newaxis, :], rank=3)
                # print(f"[Subspace] Built basis (rank=3) from template: {_tp.name}")
                break

    if ua_basis is None:
        raise RuntimeError(
            "[Subspace] No basis or template file found in config/. "
            "Please provide a pre-built basis or a matched-filter template."
        )


    for sess in sess_folders:
        print(f"\n=== Session: {sess.name} ===")
        
        # Check if outputs already exist
        out_dir = UA_CKPT_OUT / f"pp__{sess.name}__NS6"
        out_npz = UA_CKPT_OUT / f"rates__{sess.name}__bin{int(BIN_MS)}ms_sigma{int(SIGMA_MS)}ms.npz"
        
        # if out_dir.exists() and out_npz.exists():
        #     print(f"[SKIP] Both outputs already exist for {sess.name}")
        #     continue
        
        temp_bin_path = None # Initialize temp_bin_path
        try:
            touchscreen_sig, hr_sig, vog_sig, meta_ns5, meta_ns2 = rcp.extract_br_aux_streams_npz(sess, UA_AUX_DATA, CAMERA_SYNC_CH, TRIANGLE_SYNC_CH, TOUCHSCREEN_CH, HR_CH, VOG_CH) # Extract sync pulses and stuff
            fs_hr = meta_ns2["fs_hr"]
            fs_vog = meta_ns2["fs_vog"]
            fs_ts = meta_ns2["fs_ts"]
            
            rec_ns6 = se.read_blackrock(sess, stream_name = 'nsx6', all_annotations=True) # Load neural data
        except Exception as e:
            print(f"[ERROR] Failed to load data for session {sess.name} (possibly empty/corrupt): {e}")
            continue
        br_idx = int(sess.name.split('_')[-1])
        
        rec_ns6, idx_rows, ua_elec, ua_nsp, ua_region, ua_region_names, ua_port = rcp.apply_ua_mapping_with_regions(rec_ns6, UA_MAP, br_idx, METADATA_CSV, monkey=MONKEY)
        UA_probe = ua_region.copy()

        # artifact windows
        block_bounds = np.empty((0, 2), dtype=int)
        
        fs_ua = rec_ns6.get_sampling_frequency()

        
        if br_idx is None:
            print(f"[WARN] Could not parse BR index from session folder '{sess.name}'. Skipping artifact removal.")
        else:
            # 1) find Intan session via metadata + NPRW rates
            stim_npz_path, intan_session_name = rcp.stim_npz_path_from_br_idx(br_idx, METADATA_CSV, NPRW_AUX_DATA)
            # 2) and fetch the shift row for the same BR index
            br2fs_intan = rcp.get_metadata_mapping(SHIFT_CSV, 'br_idx', 'fs_intan')
            br2shift_samp_intan = rcp.get_metadata_mapping(SHIFT_CSV, 'br_idx', 'shift_sample')

            if not stim_npz_path or not stim_npz_path.exists():
                print(f"[WARN] stim_stream.npz not found for BR {br_idx:03d} (looked at {stim_npz_path}).")
            else:
                print(f"[MAP] BR {br_idx:03d} -> Intan session '{intan_session_name}'")

                stim = rcp.load_stim_detection(stim_npz_path)
                block_bounds = stim.get("block_bounds_samples", [])

        if block_bounds.size and br2shift_samp_intan is not None:
            fs_intan = float(br2fs_intan.get(br_idx, fs_ua)) # Default to fs_ua if unknown, better than hardcoded 30k
            shift_raw = br2shift_samp_intan.get(br_idx, None)
            if shift_raw is None or (isinstance(shift_raw, str) and shift_raw.strip() == ""):
                print(f"[WARN] Missing anchor_sample for br_idx={br_idx} in {SHIFT_CSV}. Skipping artifact removal.")
                block_bounds = np.empty((0, 2), dtype=int)
            else:
                shift_samp_intan = float(shift_raw)

            starts_intan = block_bounds[:, 0].astype(np.int64)
            ends_intan   = block_bounds[:, 1].astype(np.int64)

            # shift+scale into UA sample index space
            PPM_CORRECTION = -13.951
            
            scale_nominal = fs_ua / fs_intan
            scale_corrected = scale_nominal * (1.0 + PPM_CORRECTION / 1e6)

            starts_ua = np.round((starts_intan - shift_samp_intan) * scale_corrected).astype(np.int64)
            ends_ua   = np.round((ends_intan   - shift_samp_intan) * scale_corrected).astype(np.int64)

            n_total = rec_ns6.get_num_samples()
            ends_ua = np.minimum(ends_ua, n_total)
            valid = (ends_ua > starts_ua) & (starts_ua >= 0) & (ends_ua <= n_total)
            starts_ua = starts_ua[valid]
            ends_ua   = ends_ua[valid]
            starts_intan = starts_intan[valid]
            ends_intan = ends_intan[valid]      

            if starts_ua.size:
                if USE_IPCA_CORRECTION:
                    # print(f"[IPCA] Starting Incremental PCA artifact correction on {starts_ua.size} blocks...")
                    
                    # Setup constants
                    stim_freq_meta = float(rcp.get_metadata_mapping(METADATA_CSV, 'BR_File', 'Stim_Frequency_Hz').get(br_idx, 400.0))
                    if np.isnan(stim_freq_meta): stim_freq_meta = 400.0
                    pulse_interval_ms = 1000.0 / stim_freq_meta
                    # print(f"[IPCA] Using stim frequency {stim_freq_meta} Hz (interval: {pulse_interval_ms:.2f} ms)")
                    
                    mw_start = int(IPCA_PULSE_WINDOW_MS[0] / 1000.0 * fs_ua)
                    mw_end   = int(IPCA_PULSE_WINDOW_MS[1] / 1000.0 * fs_ua)
                    micro_n_time = mw_end - mw_start
                    
                    # ══════════════════════════════════════════════════════════════════════
                    # LOAD PER-CHANNEL LAG CALIBRATION
                    # Now uses DIRECT channel name lookup (no electrode mapping needed)
                    # ══════════════════════════════════════════════════════════════════════
                    lag_csv = REPO_ROOT / 'config' / 'channel_lag_calibration' / f'channel_lags_port{ua_port}.csv'

                    if not lag_csv.exists():
                        print(f"[ERROR] Lag calibration not found: {lag_csv}")
                        rec_hp = spre.highpass_filter(rec_ns6, freq_min=float(PARAMS.highpass_hz))
                        rec_artif_removed = rec_hp
                    else:
                        # Read per-channel lags - DIRECT lookup by channel name
                        channel_lags_by_name = {}      # {channel_name: lag_samples} - PRIMARY
                        channel_lags_by_electrode = {} # {electrode_id: lag_samples} - BACKUP
                        
                        with open(lag_csv, 'r', newline='') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                ch_name = row['channel_id']
                                elec_id = int(row['electrode']) if row.get('electrode') else None
                                
                                # Parse lag value
                                if row.get('lag_samples') and row['lag_samples'].strip():
                                    lag_samp = int(row['lag_samples'])
                                elif row.get('lag_ms'):
                                    lag_ms = float(row['lag_ms'])
                                    lag_samp = int(round(lag_ms / 1000.0 * fs_ua))
                                else:
                                    continue  # Skip invalid rows
                                
                                # Store by channel name (primary lookup)
                                channel_lags_by_name[ch_name] = lag_samp
                                
                                # Store by electrode ID (backup lookup)
                                if elec_id is not None and elec_id > 0:
                                    channel_lags_by_electrode[elec_id] = lag_samp
                        
                        # print(f"[IPCA] Loaded lags for {len(channel_lags_by_name)} channels from {lag_csv.name}")
                        
                        # Debug: show lag range
                        # if channel_lags_by_name:
                        #     lag_values = list(channel_lags_by_name.values())
                            # print(f"[IPCA] Lag range: {min(lag_values)} to {max(lag_values)} samples "
                            #     f"({min(lag_values)/fs_ua*1000:.3f} to {max(lag_values)/fs_ua*1000:.3f} ms)")
                        
                        # ══════════════════════════════════════════════════════════════════════
                        # STEP 1: GENERATE PULSE TIMES IN UA COORDINATES
                        # Apply PPM correction ONLY to the anchor (first pulse of each block),
                        # then generate subsequent pulses using UA timing directly.
                        # This avoids cumulative drift if the stimulator doesn't run on Intan's clock.
                        # ══════════════════════════════════════════════════════════════════════

                        # ══════════════════════════════════════════════════════════════════════
                        # CALCULATE ACTUAL PULSE INTERVAL (CLOCK DRIFT CORRECTED)
                        # Measured: stimulator runs ~1.33% slower than nominal
                        # 400 Hz nominal → 394.7 Hz actual (76 samples instead of 75)
                        # ══════════════════════════════════════════════════════════════════════
                        CLOCK_DRIFT_FACTOR = 1.0133  # Empirically measured across multiple sessions

                        nominal_interval_samples = pulse_interval_ms / 1000.0 * fs_ua
                        interval_ua_samples = nominal_interval_samples * CLOCK_DRIFT_FACTOR

                        # actual_freq = fs_ua / interval_ua_samples
                        # print(f"[IPCA] Stim timing correction applied:")
                        # print(f"[IPCA]   Nominal: {stim_freq_meta:.1f} Hz ({nominal_interval_samples:.2f} samples)")
                        # print(f"[IPCA]   Actual:  {actual_freq:.1f} Hz ({interval_ua_samples:.2f} samples)")
                        # print(f"[IPCA]   Correction: +{interval_ua_samples - nominal_interval_samples:.2f} samples/pulse")

                        

                        # Quick sanity check on stimulation frequency
                        # print(f"[DEBUG] Metadata stim frequency: {stim_freq_meta} Hz")
                        # print(f"[DEBUG] Pulse interval: {pulse_interval_ms} ms = {interval_ua_samples:.4f} UA samples")
                        # print(f"[DEBUG] fs_ua: {fs_ua} Hz")
                        # print(f"[DEBUG] fs_intan: {fs_intan} Hz")
                        # print(f"[DEBUG] scale_corrected: {scale_corrected}")
                        # print(f"[DEBUG] shift_samp_intan: {shift_samp_intan}")
                        # print(f"[DEBUG] First block: Intan [{starts_intan[0]} - {ends_intan[0]}]")
                        # print(f"[DEBUG] First block: UA [{starts_ua[0]} - {ends_ua[0]}]")

                        all_ua_pulse_times = []

                        for st_intan, en_intan in zip(starts_intan, ends_intan):
                            # Convert ONLY the block start (anchor) with PPM correction
                            anchor_ua = np.round((st_intan - shift_samp_intan) * scale_corrected).astype(np.int64)
                            
                            # Calculate number of pulses based on block duration in UA time
                            # (Use UA timing for duration calculation as well)
                            en_ua = np.round((en_intan - shift_samp_intan) * scale_corrected).astype(np.int64)
                            stim_dur_ms = (en_ua - anchor_ua) / fs_ua * 1000.0
                            num_pulses = int(np.floor(stim_dur_ms / pulse_interval_ms)) + 1
                            
                            # Generate subsequent pulses in UA time directly (no PPM on intervals)
                            pulse_times_ua = anchor_ua + np.arange(num_pulses) * interval_ua_samples
                            all_ua_pulse_times.extend(pulse_times_ua)

                        # Convert to integer array
                        intan_pulses_base_ua = np.round(all_ua_pulse_times).astype(np.int64)

                        # print(f"[IPCA] Generated {len(intan_pulses_base_ua)} base pulse times in UA coordinates")
                        # print(f"[IPCA] Using interval of {interval_ua_samples:.4f} UA samples ({pulse_interval_ms:.4f} ms)")

                        # ══════════════════════════════════════════════════════════════════════
                        # STEP 2: APPLY PER-CHANNEL LAGS
                        # Uses DIRECT channel name lookup (no NSP->Electrode mapping needed)
                        # ══════════════════════════════════════════════════════════════════════
                        n_channels = rec_ns6.get_num_channels()
                        channel_ids = rec_ns6.get_channel_ids()
                        
                        # Calculate fallback lag (median of all known lags)
                        available_lags = list(channel_lags_by_name.values())
                        fallback_lag = int(np.median(available_lags)) if available_lags else 17
                        # print(f"[IPCA] Fallback lag (median): {fallback_lag} samples")
                        
                        intan_pulses_per_channel = {}
                        lag_stats = {'by_name': 0, 'by_electrode': 0, 'fallback': 0}
                        
                        for ch_idx, ch_name in enumerate(channel_ids):
                            # Try direct channel name lookup first (most reliable)
                            if ch_name in channel_lags_by_name:
                                lag = channel_lags_by_name[ch_name]
                                lag_stats['by_name'] += 1
                                lookup_method = "by_name"
                            else:
                                # Fallback: try by electrode ID from ua_elec array
                                elec_id = int(ua_elec[ch_idx]) if ch_idx < len(ua_elec) else None
                                if elec_id and elec_id > 0 and elec_id in channel_lags_by_electrode:
                                    lag = channel_lags_by_electrode[elec_id]
                                    lag_stats['by_electrode'] += 1
                                    lookup_method = "by_electrode"
                                else:
                                    lag = fallback_lag
                                    lag_stats['fallback'] += 1
                                    lookup_method = "fallback"
                            
                            # Add lag to create per-channel pulse times
                            channel_pulse_times = intan_pulses_base_ua + lag
                            
                            # Filter valid times
                            n_total = rec_ns6.get_num_samples()
                            valid = (channel_pulse_times >= 0) & (channel_pulse_times < n_total)
                            intan_pulses_per_channel[ch_idx] = channel_pulse_times[valid]
                            
                            # Debug first 3 channels
                            # if ch_idx < 3:
                                # print(f"[DEBUG] Ch {ch_name}: lag={lag} samp [{lookup_method}], "
                                #       f"{len(intan_pulses_per_channel[ch_idx])} pulses")
                        
                        # print(f"[IPCA] Lag lookup stats: {lag_stats['by_name']} by_name, "
                        #       f"{lag_stats['by_electrode']} by_electrode, {lag_stats['fallback']} fallback")
                        
                        n_pulses = max(len(times) for times in intan_pulses_per_channel.values())
                        # print(f"[IPCA] Maximum {n_pulses} pulses across all channels")
                        
                        # ══════════════════════════════════════════════════════════════════════
                        # STEP 3: EXTRACT WINDOWS WITH LAG-CORRECTED CENTERS
                        # ══════════════════════════════════════════════════════════════════════
                        micro_signal_list = []
                        micro_map_per_channel = {ch_idx: [] for ch_idx in range(n_channels)}
                        
                        for pulse_idx in range(n_pulses):
                            pulse_data = np.zeros((micro_n_time, n_channels), dtype=np.float32)
                            
                            for ch_idx, ch_name in enumerate(channel_ids):
                                channel_pulses = intan_pulses_per_channel[ch_idx]
                                
                                if pulse_idx >= len(channel_pulses):
                                    continue
                                
                                # This is already lag-corrected artifact center
                                artifact_center = channel_pulses[pulse_idx]
                                
                                p_start = artifact_center + mw_start  # mw_start is negative
                                p_end = artifact_center + mw_end
                                
                                if p_start >= 0 and p_end <= rec_ns6.get_num_samples():
                                    trace = rec_ns6.get_traces(
                                        start_frame=p_start, end_frame=p_end,
                                        channel_ids=[ch_name], return_in_uV=False
                                    ).flatten()
                                    pulse_data[:, ch_idx] = trace
                                
                                micro_map_per_channel[ch_idx].append((p_start, p_end))
                            
                            micro_signal_list.append(pulse_data)
                        
                        micro_signal_array = np.stack(micro_signal_list)
                        # print(f"[IPCA] Extracted {len(micro_signal_array)} artifacts. Shape: {micro_signal_array.shape}")

                        # ══════════════════════════════════════════════════════════════════════
                        # DIAGNOSTIC: Quick artifact alignment check
                        # ══════════════════════════════════════════════════════════════════════
                        import matplotlib.pyplot as plt

                        test_ch_idx = min(10, n_channels - 1)
                        test_ch_name = rec_ns6.get_channel_ids()[test_ch_idx]
                        n_check = min(50, micro_signal_array.shape[0])
                        window_center = micro_signal_array.shape[1] // 2

                        # Find peak locations
                        peak_locs = []
                        for i in range(n_check):
                            wf = micro_signal_array[i, :, test_ch_idx]
                            peak_locs.append(np.argmax(np.abs(np.diff(wf, prepend=wf[0]))))
                        peak_locs = np.array(peak_locs)
                        drift = peak_locs - window_center

                        # Stats
                        drift_std = np.std(drift)
                        drift_range = drift.max() - drift.min()
                        alignment_ok = drift_range <= 3  # Good if peaks vary by ≤3 samples

                        # Single-panel diagnostic
                        fig, ax = plt.subplots(figsize=(10, 4))

                        # Overlay all waveforms (normalized)
                        time_ms = (np.arange(micro_signal_array.shape[1]) - window_center) / fs_ua * 1000
                        for i in range(min(30, n_check)):
                            wf = micro_signal_array[i, :, test_ch_idx]
                            wf_norm = (wf - wf.mean()) / (wf.std() + 1e-6)
                            ax.plot(time_ms, wf_norm, 'b-', alpha=0.15, lw=0.8)

                        # Mean waveform
                        mean_wf = micro_signal_array[:n_check, :, test_ch_idx].mean(axis=0)
                        mean_wf_norm = (mean_wf - mean_wf.mean()) / (mean_wf.std() + 1e-6)
                        ax.plot(time_ms, mean_wf_norm, 'r-', lw=2, label='Mean artifact')

                        ax.axvline(0, color='k', linestyle='--', alpha=0.5, lw=1)
                        ax.set_xlabel('Time from expected center (ms)', fontsize=11)
                        ax.set_ylabel('Normalized amplitude', fontsize=11)
                        ax.set_xlim(time_ms[0], time_ms[-1])

                        # Status indicator
                        status = "ALIGNED" if alignment_ok else "MISALIGNED"
                        status_color = "green" if alignment_ok else "red"

                        ax.set_title(
                            f'BR {br_idx:03d} | Ch {test_ch_name} | {n_check} pulses\n'
                            f'Peak jitter: {drift_std:.1f} samples (range: {drift_range}) — {status}',
                            fontsize=12, color=status_color if not alignment_ok else 'black'
                        )

                        ax.legend(loc='upper right')
                        ax.grid(True, alpha=0.3)

                        plt.tight_layout()
                        diag_path = UA_CKPT_OUT.parent.parent / "figures" / "artifact_alignment" / f"artifact_alignment_br{br_idx:03d}.png"
                        diag_path.parent.mkdir(parents=True, exist_ok=True)
                        plt.savefig(diag_path, dpi=120)
                        plt.close()

                        # Console summary
                        # print(f"[DIAG] Artifact alignment: {status}")
                        # print(f"[DIAG]   Peak jitter: {drift_std:.2f} samples (range: {drift_range})")
                        # print(f"[DIAG]   Saved: {diag_path.name}")
                        
                        # ══════════════════════════════════════════════════════════════════════
                        # STEP 4: REGION-WISE IPCA CORRECTION
                        # ══════════════════════════════════════════════════════════════════════
                        corrector = rcp.IPCA_Artifact_Correction(rank=IPCA_RANK)
                        
                        micro_corrected = micro_signal_array.copy()
                        
                        valid_idx = np.where(ua_elec > 0)[0]
                        target_ch_regions = [int(ua_region[i]) for i in valid_idx]
                        
                        region_to_idxs = {}
                        for valid_ch_idx, reg_idx in zip(valid_idx, target_ch_regions):
                            reg_name = ua_region_names[reg_idx] if reg_idx >= 0 else "Unknown"
                            region_to_idxs.setdefault(reg_name, []).append(valid_ch_idx)
                        
                        n_stim, n_time, _ = micro_signal_array.shape
                        
                        # print("Cross-channel IPCA by region:")
                        for reg, idxs in region_to_idxs.items():
                            if not idxs:
                                continue
                            
                            reg_signal = micro_signal_array[:, :, idxs]
                            pooled_signal = np.transpose(reg_signal, (0, 2, 1)).reshape(-1, n_time)
                            
                            reg_template = rcp.Template()
                            _, reg_template = corrector.ipca_template_per_channel(pooled_signal, reg_template)
                            
                            # print(f"  [{reg}] Shared subspace learned from {len(idxs)} channels.")
                            
                            for ch_idx in idxs:
                                signal_ch = micro_signal_array[:, :, ch_idx].copy()
                                corrected_ch = corrector.apply_template(signal_ch, reg_template)
                                micro_corrected[:, :, ch_idx] = corrected_ch
                        
                        # ══════════════════════════════════════════════════════════════════════
                        # STEP 5: BUILD CORRECTED RECORDING WITH PER-CHANNEL PLACEMENT
                        # ══════════════════════════════════════════════════════════════════════
                        # print("[IPCA] Building per-channel IPCACorrectedRecording...")
                        
                        rec_corr_mem = rcp.PerChannelIPCACorrectedRecording(rec_ns6, micro_map_per_channel, micro_corrected)
                        
                        # print("[IPCA] Highpass filtering after correction...")
                        rec_hp = spre.highpass_filter(rec_corr_mem, freq_min=float(PARAMS.highpass_hz))
                        rec_artif_removed = rec_hp
                        
                        del micro_signal_array, micro_corrected
                        gc.collect()
                        
                        # ══════════════════════════════════════════════════════════════════════
                        # DEBUG PLOTTING
                        # ══════════════════════════════════════════════════════════════════════
                        ch_si_id = None
                        el_out = ""
                        for ci, elec_id in enumerate(ua_elec):
                            if ua_port == 'B':
                                if int(elec_id) == 24:
                                    el_out = "24"
                                    ch_si_id = rec_ns6.get_channel_ids()[ci]
                                    break
                            else:
                                if int(elec_id) == 30:
                                    el_out = "30"
                                    ch_si_id = rec_ns6.get_channel_ids()[ci]
                                    break
                                
                        if ch_si_id is not None:
                            import matplotlib.pyplot as plt
                            MAX_DEBUG_BLOCKS = 20
                            n_blocks = min(len(starts_ua), MAX_DEBUG_BLOCKS)
                            fig, axes = plt.subplots(n_blocks, 3, figsize=(18, 2.5 * n_blocks))
                            if n_blocks == 1: axes = axes[np.newaxis, :]
                            
                            fs = rec_ns6.get_sampling_frequency()
                            win_samp_pre  = int(5.0  * fs / 1000.0)
                            win_samp_post = int(20.0 * fs / 1000.0)
                            time_axis = np.linspace(-5.0, 20.0, win_samp_pre + win_samp_post)
                            
                            for b_idx, s_anchor in enumerate(starts_ua[:n_blocks]):
                                s0 = int(s_anchor) - win_samp_pre
                                s1 = int(s_anchor) + win_samp_post
                                s0 = max(0, s0)
                                s1 = min(rec_ns6.get_num_samples(), s1)
                                n_samp = s1 - s0
                                t_ax = time_axis[:n_samp]
                                
                                raw_trace  = rec_ns6.get_traces(start_frame=s0, end_frame=s1, channel_ids=[ch_si_id], return_in_uV=True).flatten()
                                corr_trace = rec_corr_mem.get_traces(start_frame=s0, end_frame=s1, channel_ids=[ch_si_id], return_in_uV=True).flatten()
                                artifact_trace = raw_trace - corr_trace
                                
                                axes[b_idx, 0].plot(t_ax, raw_trace,      color='k',    lw=1,   alpha=0.8)
                                axes[b_idx, 0].axvline(0, color='r', linestyle='--', alpha=0.4)
                                axes[b_idx, 1].plot(t_ax, artifact_trace, color='r',    lw=1,   alpha=0.9)
                                axes[b_idx, 1].axvline(0, color='r', linestyle='--', alpha=0.4)
                                axes[b_idx, 2].plot(t_ax, corr_trace,     color='blue', lw=1.2, alpha=0.9)
                                axes[b_idx, 2].axvline(0, color='r', linestyle='--', alpha=0.4)
                                for c in range(3):
                                    axes[b_idx, c].set_ylabel(f"T{b_idx+1}", fontsize=7)
                                if b_idx == 0:
                                    axes[0, 0].set_title("Raw",             fontsize=9)
                                    axes[0, 1].set_title("Learned Artifact",fontsize=9)
                                    axes[0, 2].set_title("Corrected",       fontsize=9)
                                
                            for c in range(3):
                                axes[-1, c].set_xlabel("Time rel. stim onset (ms)")
                            fig.suptitle(f"IPCA Debug | Ch {el_out} | BR sess {br_idx:03d}")
                            fig.tight_layout()
                            out_fig = UA_CKPT_OUT.parent.parent / "figures" / "IPCA_debug"/ f"debug_MACRO_UA_br{br_idx:03d}_ch{el_out}.png"
                            out_fig.parent.mkdir(parents=True, exist_ok=True)
                            fig.savefig(out_fig, dpi=150)
                            plt.close(fig)
                            print(f"[IPCA] RAN --> SAVED DEBUG FIG FOR CH {el_out} in /figures/IPCA_debug/")

                else:
                    # Legacy 0-blanking method
                    rec_hp = spre.highpass_filter(rec_ns6, freq_min=float(PARAMS.highpass_hz))
                    dur_ms    = (ends_ua - starts_ua) * 1000.0 / fs_ua
                    ms_after  = float(dur_ms.max() + ARTRMV_TAIL_MS)

                    rec_artif_removed = spre.remove_artifacts(
                        rec_hp,
                        list_triggers=starts_ua.tolist(),
                        ms_before=ARTRMV_MS_BEFORE,
                        ms_after=ms_after,
                        mode="zeros",
                    )
            else:
                print("[WARN] all artifact intervals invalid after shift; skipping artifact removal.")
                rec_hp = spre.highpass_filter(rec_ns6, freq_min=float(PARAMS.highpass_hz))
                rec_artif_removed = rec_hp
        else:
            if not block_bounds.size:
                print("[WARN] no stim found, skipping artifact removal.")
            elif br2shift_samp_intan is None:
                print("[WARN] no shift found, skipping artifact removal.")
            rec_hp = spre.highpass_filter(rec_ns6, freq_min=float(PARAMS.highpass_hz))
            rec_artif_removed = rec_hp

            
        out_dir = UA_CKPT_OUT / f"pp__{sess.name}__NS6"; out_dir.mkdir(parents=True, exist_ok=True)
        
        # Custom IPCA Nodes are not easily fork-serializable. Force sequential write.
        skip_multiproc = USE_IPCA_CORRECTION and block_bounds.size > 0
        with suppress_stdout():
            if skip_multiproc:
                rec_artif_removed = rec_artif_removed.save(folder=out_dir, overwrite=True, n_jobs=1, progress_bar=False)
            else:
                rec_artif_removed = rec_artif_removed.save(folder=out_dir, overwrite=True, progress_bar=False)
            
        print(f"[{sess.name}] (ns6) saved preprocessed -> {out_dir}")
        
        # del rec_artif_removed
        if "rec_corr_mem" in locals(): del rec_corr_mem
        if "rec_hp" in locals(): del rec_hp
        gc.collect()
        
        n_seg = rec_artif_removed.get_num_segments()
        try:
            noise_levels = si.get_noise_levels(rec_artif_removed, method="mad", return_in_uV=False, n_jobs=1)
        except Exception:
            noise_levels = si.get_noise_levels(rec_artif_removed, method="mad", return_in_uV=False)
        
        # print(f"[INFO] Segments of recording: {n_seg}, Average noise level: {np.nanmean(noise_levels)}")
        
        
        # Set channel locations (required by some SI utilities; kept for compatibility)
        n_ch_total = rec_artif_removed.get_num_channels()
        locs = np.zeros((n_ch_total, 2))
        side = int(np.ceil(np.sqrt(n_ch_total)))
        for i in range(n_ch_total):
            locs[i, 0] = (i % side) * 400.0
            locs[i, 1] = (i // side) * 400.0
        rec_artif_removed.set_channel_locations(locs)
        
        # ── Subspace-CFAR detection ────────────────────────────────────────────
        # print(f"[Subspace] Running subspace-CFAR detector (α={CFAR_ALPHA})...")
        peaks = rcp.subspace_detect_cfar(
            rec_artif_removed, ua_basis,
            cfar_alpha=CFAR_ALPHA,
            peak_sign=PEAK_SIGN,
        )
        # print(f"[Subspace] Detected {len(peaks):,} raw peaks")

        # ── Optional local-σ amplitude gate ───────────────────────────────────
        # Raises the amplitude bar during high-noise bursts (e.g. chewing/movement).
        if LOCAL_SIGMA_GATE and len(peaks):
            peaks = rcp.filter_peaks_by_local_sigma(
                peaks, rec_artif_removed,
                k_amp=AMP_GATE_K,
                w_ms=AMP_GATE_W_MS,
                peak_sign=PEAK_SIGN,
            )

        # ── Optional hard amplitude / SNR post-filter ─────────────────────────
        if any([MIN_AMPLITUDE_UV, MAX_AMPLITUDE_UV, MIN_SNR]):
            print(f"[Filter] Applying amplitude/SNR filtering:")
            # if MIN_AMPLITUDE_UV:
            #     print(f"  - Min amplitude: {MIN_AMPLITUDE_UV} µV")
            # if MAX_AMPLITUDE_UV:
            #     print(f"  - Max amplitude: {MAX_AMPLITUDE_UV} µV")
            # if MIN_SNR:
            #     print(f"  - Min SNR: {MIN_SNR}")

            peaks, peak_amplitudes, peak_snr = filter_peaks_by_amplitude_fast(
                peaks,
                rec_artif_removed,
                noise_levels,
                min_amplitude_uv=MIN_AMPLITUDE_UV,
                max_amplitude_uv=MAX_AMPLITUDE_UV,
                min_snr=MIN_SNR,
                peak_sign=PEAK_SIGN,
            )
        else:
            peak_amplitudes = None
            peak_snr = None
        
        
        # Threshold to get touchscreen state
        ts_state_num = np.zeros(0, dtype=np.int8)
        ts_state_char = np.full(0, "N", dtype="U1")

        if touchscreen_sig is not None:
            import pandas as pd
            import matplotlib.pyplot as plt
            
            touchscreen_sig = np.asarray(touchscreen_sig, float)
            ts_state_num = np.zeros_like(touchscreen_sig, dtype=np.int8)
            ts_state_char = np.full(touchscreen_sig.shape, "N", dtype="U1")

            touchscreen_max = np.max(touchscreen_sig)
            auto_use_dlc = touchscreen_max < TOUCHSCREEN_MIN_VALID

            if auto_use_dlc:
                # print(f"  [TOUCH] Touchscreen max ({touchscreen_max:.2e}) < {TOUCHSCREEN_MIN_VALID:.2e}")
                print(f"[TOUCH] Falling back to DLC kinematics for touch state")
                
                # Load DLC data
                sess_parts = sess.name.split("_", 1)
                short_name = sess_parts[1] if len(sess_parts) > 1 and sess_parts[0].isdigit() else sess.name
                csv_files = list(BEHV_CKPT_ROOT.rglob(f"*{short_name}*aligned.csv"))
                
                if csv_files:
                    csv_path = csv_files[0]
                    try:
                        df = pd.read_csv(csv_path, header=[0, 1, 2])
                        flat_headers = ["_".join([str(c) for c in col if "Unnamed" not in str(c)]).strip() for col in df.columns]
                        df.columns = flat_headers
                    except:
                        df = pd.read_csv(csv_path, header=0)
                    
                    # Find middle finger y column for Cam 0
                    col_y = None
                    for i, h in enumerate(df.columns):
                        h_str = str(h).lower()
                        if "middle_y" in h_str and ("cam-0" in h_str or "cam0" in h_str or "camera 0" in h_str):
                            col_y = i
                            break
                    if col_y is None:
                        for i, h in enumerate(df.columns):
                            h_str = str(h).lower()
                            if "middle_y" in h_str and "cam" not in h_str:
                                col_y = i
                                break
                    
                    if col_y is not None:
                        ser_y_raw = pd.to_numeric(df.iloc[:, col_y], errors="coerce").values
                        n_valid_raw = np.sum(~np.isnan(ser_y_raw))
                        dlc_ok = n_valid_raw > 0

                        if not dlc_ok:
                            print(f"  [DLC WARNING] No valid DLC Y values found, leaving all timestamps unclassified")
                        else:
                            # Align DLC to timestamp grid
                            ts_sec = np.arange(len(touchscreen_sig), dtype=float) / fs_ts
                            t_ts = ts_sec - ts_sec[0]

                            # Try to use ns5_sample column for time alignment
                            ns5_col = None
                            for i, c in enumerate(df.columns):
                                if "ns5" in str(c).lower() and "sample" in str(c).lower():
                                    ns5_col = i
                                    break

                            if ns5_col is not None:
                                ns5_samples = pd.to_numeric(df.iloc[:, ns5_col], errors="coerce").values
                                ns5_fs = 30000.0
                                dlc_time_sec = ns5_samples / ns5_fs

                                from scipy.interpolate import interp1d
                                valid_mask = ~np.isnan(ser_y_raw) & ~np.isnan(dlc_time_sec)

                                if valid_mask.sum() > 10:
                                    interp_func = interp1d(
                                        dlc_time_sec[valid_mask],
                                        ser_y_raw[valid_mask],
                                        kind='linear',
                                        bounds_error=False,
                                        fill_value=np.nan
                                    )
                                    ser_y = interp_func(t_ts)
                                else:
                                    print(f"  [DLC WARNING] Not enough valid DLC points ({valid_mask.sum()}), falling back to frame-rate assumption")
                                    dlc_fps = 100.0
                                    idx_map = np.clip((t_ts * dlc_fps).astype(int), 0, len(ser_y_raw) - 1)
                                    ser_y = ser_y_raw[idx_map]
                            else:
                                print(f"  [DLC WARNING] No ns5_sample column found, using frame-rate assumption")
                                dlc_fps = 100.0
                                idx_map = np.clip((t_ts * dlc_fps).astype(int), 0, len(ser_y_raw) - 1)
                                ser_y = ser_y_raw[idx_map]

                            n_valid_aligned = np.sum(~np.isnan(ser_y))
                            if n_valid_aligned == 0:
                                print(f"  [DLC WARNING] DLC Y alignment produced 0 valid points, leaving all timestamps unclassified")
                                dlc_ok = False

                        if dlc_ok:
                            # Thresholds for state classification
                            DLC_THRESH_A = 250.0
                            DLC_THRESH_B = 150.0

                            ts_state_num[ser_y > DLC_THRESH_A] = 1
                            ts_state_num[ser_y < DLC_THRESH_B] = 2
                            ts_state_char[ts_state_num == 1] = "A"
                            ts_state_char[ts_state_num == 2] = "B"

                            n_A = np.sum(ts_state_num == 1)
                            n_B = np.sum(ts_state_num == 2)
                            n_N = np.sum(ts_state_num == 0)
                            print(f"  [DLC] Classified timestamps: A={n_A}, B={n_B}, N={n_N}")

                            DEBUG_DLC_PLOT = True
                            if DEBUG_DLC_PLOT:
                                import matplotlib.pyplot as plt

                                def safe_range_text(arr):
                                    arr = np.asarray(arr)
                                    arr = arr[~np.isnan(arr)]
                                    if arr.size == 0:
                                        return "NA-NA"
                                    return f"{np.nanmin(arr):.0f}-{np.nanmax(arr):.0f}"

                                debug_fig_dir = UA_CKPT_OUT.parent.parent / "figures/classification/"
                                debug_fig_dir.mkdir(parents=True, exist_ok=True)
                                debug_fig_path = debug_fig_dir / f"debug_DLC_classification_{sess.name}.png"

                                stim_onsets_sec = None
                                if 'starts_ua' in locals() and starts_ua is not None and len(starts_ua) > 0:
                                    ns6_start_time = rec_ns6.get_start_time()
                                    ns2_start_time = meta_ns2.get('start_time', 0.0) if meta_ns2 else 0.0
                                    ns6_to_ns2_offset = ns6_start_time - ns2_start_time
                                    stim_onsets_sec = (starts_ua / fs_ua) - ns6_to_ns2_offset

                                fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

                                # Top panel: DLC Y trace and thresholds
                                ax = axes[0]
                                ax.plot(ts_sec, ser_y, 'k-', lw=0.5, alpha=0.7, label='DLC middle_y')
                                ax.axhline(DLC_THRESH_A, color='red', ls='--', lw=1.5, label=f'A threshold ({DLC_THRESH_A})')
                                ax.axhline(DLC_THRESH_B, color='blue', ls='--', lw=1.5, label=f'B threshold ({DLC_THRESH_B})')

                                mask_A = ts_state_num == 1
                                mask_B = ts_state_num == 2
                                ax.scatter(ts_sec[mask_A], ser_y[mask_A], c='red', s=8, alpha=0.6, label=f'A classified ({n_A})')
                                ax.scatter(ts_sec[mask_B], ser_y[mask_B], c='blue', s=8, alpha=0.6, label=f'B classified ({n_B})')

                                if stim_onsets_sec is not None:
                                    for stim_t in stim_onsets_sec:
                                        if ts_sec[0] <= stim_t <= ts_sec[-1]:
                                            ax.axvspan(stim_t - 0.5, stim_t + 0.5, color='purple', alpha=0.15)
                                            ax.axvline(stim_t, color='purple', lw=0.8, alpha=0.5)

                                y_valid = ser_y[~np.isnan(ser_y)]
                                y_at_A = ser_y[mask_A]
                                y_at_B = ser_y[mask_B]
                                stats_text = (
                                    f"Valid DLC points: {len(y_valid)}/{len(ser_y)}\n"
                                    f"Y range: {safe_range_text(y_valid)}\n"
                                    f"A range: {safe_range_text(y_at_A)}\n"
                                    f"B range: {safe_range_text(y_at_B)}"
                                )
                                ax.text(
                                    0.99, 0.98, stats_text,
                                    transform=ax.transAxes,
                                    ha='right', va='top',
                                    fontsize=9,
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                                )

                                ax.set_ylabel("DLC middle_y")
                                ax.set_title(f"DLC classification debug: {sess.name}")
                                ax.legend(loc='upper left', fontsize=8)
                                ax.grid(True, alpha=0.3)

                                # Bottom panel: classified state over time
                                ax2 = axes[1]
                                ax2.scatter(ts_sec, ts_state_num, c=ts_state_num, cmap='viridis', s=4, alpha=0.7)
                                ax2.set_yticks([0, 1, 2])
                                ax2.set_yticklabels(['N', 'A', 'B'])
                                ax2.set_xlabel("Time in NS6 recording (s)")
                                ax2.set_ylabel("State")

                                if stim_onsets_sec is not None:
                                    for stim_t in stim_onsets_sec:
                                        if ts_sec[0] <= stim_t <= ts_sec[-1]:
                                            ax2.axvspan(stim_t - 0.5, stim_t + 0.5, color='purple', alpha=0.15)
                                            ax2.axvline(stim_t, color='purple', lw=0.8, alpha=0.5)

                                ax2.grid(True, alpha=0.3)

                                plt.tight_layout()
                                plt.savefig(debug_fig_path, dpi=150)
                                plt.close(fig)
                                print(f"  [DLC DEBUG] Saved classification plot: {debug_fig_path}") 
                    else:
                        print(f"  [WARNING] Could not find middle_y column in DLC CSV")
                else:
                    print(f"  [WARNING] No DLC CSV found for {short_name}")
            else:
                # print(f"  [TOUCHSCREEN] Using touchscreen signal (max: {touchscreen_max:.2e})")
                ts_state_num[touchscreen_sig >= TOUCHSCREEN_THRES_A] = 1
                ts_state_num[touchscreen_sig >= TOUCHSCREEN_THRES_B] = 2
                ts_state_char[ts_state_num == 1] = "A"
                ts_state_char[ts_state_num == 2] = "B"

            # print(f"  [TOUCHSCREEN] Final unique states: {np.unique(ts_state_num)}")
        else:
            print("[touchscreen] No touchscreen signal; leaving ts_state_* empty.")

        out_npz = UA_CKPT_OUT / f"rates__{sess.name}__bin{int(BIN_MS)}ms_sigma{int(SIGMA_MS)}ms.npz"
        
        save = dict(
            peaks=peaks,
            noise_levels=noise_levels,
                   
            ts_state_num=ts_state_num,
            ts_state_char=ts_state_char,

            # UA indices
            ua_index=idx_rows.astype(np.int16),
            ua_elec=ua_elec.astype(np.int16),
            ua_nsp=ua_nsp.astype(np.int16),
            ua_region=ua_region,
            ua_region_names=ua_region_names,
            ua_port=ua_port,
            
            # Other signals
            hr_sig=hr_sig,
            vog_sig=vog_sig,

            meta=dict(
                detect_threshold=THRESH,
                peak_sign=PEAK_SIGN,
                bin_ms=BIN_MS,
                sigma_ms=SIGMA_MS,
                fs_ua=fs_ua,
                fs_hr=fs_hr,
                fs_vog=fs_vog,
                fs_ts=fs_ts,
                n_channels=rec_artif_removed.get_num_channels(),
                session=str(sess.name),
                n_samples = rec_ns6.get_total_samples(),
                n_segs = rec_ns6.get_num_segments(),
                rec_dur = rec_ns6.get_total_duration(),
                rec_start_ms = rec_ns6.get_start_time()*1000,
                rec_end_ms = rec_ns6.get_end_time()*1000,

                meta_ns5=meta_ns5,
                meta_ns2=meta_ns2,
            ),
        )
        
        np.savez_compressed(out_npz, **save)
        print(f"[{sess.name}] saved peaks-> {out_npz}")

        # cleanup to keep memory stable on long batches
        del peaks, rec_ns6, rec_artif_removed
        gc.collect()
        
if __name__ == "__main__":
    main()
