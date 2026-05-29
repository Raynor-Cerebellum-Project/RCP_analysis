import gc, csv
import numpy as np
import warnings
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
from sklearn.decomposition import IncrementalPCA
import RCP_analysis as rcp
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from RCP_analysis.python.functions.config_loading import *
from RCP_analysis.python.functions.artifact_correction import IPCA_Artifact_Correction
from RCP_analysis.python.functions.config_loading import *

from spikeinterface.core import BaseRecording, BaseRecordingSegment

# FIX FOR WINDOWS MULTIPROCESSING IN SPIKEINTERFACE 0.104.1
# When child processes try to unpickle dynamically defined classes (IPCACorrectedRecording),
# SI tries to check the __version__ of the module. Since the module is __mp_main__, it crashes.
__version__ = "1.0.0"

# Suppress annoying SpikeInterface provenance warning when manually reconstructing memory arrays
warnings.filterwarnings("ignore", message="The extractor is not serializable to file. The provenance will not be saved.")

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
            # Convert slice to array
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
            
            # Find pulses that overlap this request
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
                    
                    # Extract this channel's corrected patch
                    patch = self.micro_corrected[pulse_idx, patch_idx_start:patch_idx_end, global_ch_idx]
                    
                    # Place it in the output
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

""" 
    This script preprocesses the Blackrock data.
    Input:
        .ns6 files from Blackrock
    Output:
        Checkpoint after preprocessing
        Checkpoint after matched-filtering and calculating MUA peak locations and firing rate
"""

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
USE_IPCA_CORRECTION  = True
IPCA_RANK            = 10
IPCA_PULSE_WINDOW_MS = (-0.4, 0.3)
# ----------------------------------------


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


TOUCHSCREEN_THRES_A = 1.0e6
TOUCHSCREEN_THRES_B = 3.0e6

UA_CKPT_OUT = UA_CKPT_ROOT

global_job_kwargs = dict(n_jobs=PARAMS.parallel_jobs, chunk_duration=PARAMS.chunk)
si.set_global_job_kwargs(**global_job_kwargs)


# ══════════════════════════════════════════════════════════════════════
# AMPLITUDE FILTERING OPTIONS
# You can use any combination of these:
# ══════════════════════════════════════════════════════════════════════

# Absolute thresholds (µV) - same for all channels
MIN_AMPLITUDE_UV = 25.0
MAX_AMPLITUDE_UV = 1000.0

MIN_SNR = 2.0                # Minimum signal-to-noise ratio (amplitude / noise_level)
                             # Typical values: 2.0 (permissive) to 5.0 (strict)


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

    # --- Matched Filter Template Configuration ---
    # Using a single unified template for all Utah Array channels
    template_path = REPO_ROOT / 'config' / "median_extremum_templates_norm_UA_PortB.npy"
    
    if template_path.exists():
        ua_template = np.load(template_path)
    else:
        # Fallback to general UA template if PortB specific is missing
        single_template_path = REPO_ROOT / 'config' / "median_extremum_templates_norm.npy"
        if single_template_path.exists():
            ua_template = np.load(single_template_path)
        else:
            print("[WARN] No matched filter templates found for Utah Arrays! Please generate them before running.")
            # Forcing placeholder to allow script to loop, but detection will be poor
            ua_template = np.zeros(60)


    for sess in sess_folders:
        print(f"=== Session: {sess.name} ===")
        
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
        
        rec_ns6, idx_rows, ua_elec, ua_nsp, ua_region, ua_region_names, ua_port = rcp.apply_ua_mapping_with_regions(rec_ns6, UA_MAP, br_idx, METADATA_CSV)
        UA_probe = ua_region.copy()

        # artifact windows
        block_bounds = np.empty((0, 2), dtype=int)
        
        fs_ua = rec_ns6.get_sampling_frequency()
        
        # Calculate ms_before for centering based on actual session sampling rate
        # If the template was generated at 30k but session is different, this ensures alignment
        ms_before = (np.argmin(ua_template) / fs_ua) * 1000.0

        
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
                print(f"[map] BR {br_idx:03d} -> Intan session '{intan_session_name}'")

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
                    print(f"[IPCA] Starting Incremental PCA artifact correction on {starts_ua.size} blocks...")
                    
                    # Setup constants
                    stim_freq_meta = float(rcp.get_metadata_mapping(METADATA_CSV, 'BR_File', 'Stim_Frequency_Hz').get(br_idx, 400.0))
                    if np.isnan(stim_freq_meta): stim_freq_meta = 400.0
                    pulse_interval_ms = 1000.0 / stim_freq_meta
                    print(f"[IPCA] Using stim frequency {stim_freq_meta} Hz (interval: {pulse_interval_ms:.2f} ms)")
                    
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
                        
                        print(f"[IPCA] Loaded lags for {len(channel_lags_by_name)} channels from {lag_csv.name}")
                        
                        # Debug: show lag range
                        if channel_lags_by_name:
                            lag_values = list(channel_lags_by_name.values())
                            print(f"[IPCA] Lag range: {min(lag_values)} to {max(lag_values)} samples "
                                f"({min(lag_values)/fs_ua*1000:.3f} to {max(lag_values)/fs_ua*1000:.3f} ms)")
                        
                        # ══════════════════════════════════════════════════════════════════════
                        # STEP 1: GENERATE PULSE TIMES IN UA COORDINATES
                        # Apply PPM correction ONLY to the anchor (first pulse of each block),
                        # then generate subsequent pulses using UA timing directly.
                        # This avoids cumulative drift if the stimulator doesn't run on Intan's clock.
                        # ══════════════════════════════════════════════════════════════════════
                        PPM_CORRECTION = -13.951
                        scale_corrected = (fs_ua / fs_intan) * (1.0 + PPM_CORRECTION / 1e6)

                        # Pulse interval in UA samples (nominal, from metadata)
                        interval_ua_samples_nominal = pulse_interval_ms / 1000.0 * fs_ua

                        # ══════════════════════════════════════════════════════════════════════
                        # CALCULATE ACTUAL PULSE INTERVAL (CLOCK DRIFT CORRECTED)
                        # Measured: stimulator runs ~1.33% slower than nominal
                        # 400 Hz nominal → 394.7 Hz actual (76 samples instead of 75)
                        # ══════════════════════════════════════════════════════════════════════
                        CLOCK_DRIFT_FACTOR = 1.0133  # Empirically measured across multiple sessions

                        nominal_interval_samples = pulse_interval_ms / 1000.0 * fs_ua
                        interval_ua_samples = nominal_interval_samples * CLOCK_DRIFT_FACTOR

                        actual_freq = fs_ua / interval_ua_samples
                        print(f"[IPCA] Stim timing correction applied:")
                        print(f"[IPCA]   Nominal: {stim_freq_meta:.1f} Hz ({nominal_interval_samples:.2f} samples)")
                        print(f"[IPCA]   Actual:  {actual_freq:.1f} Hz ({interval_ua_samples:.2f} samples)")
                        print(f"[IPCA]   Correction: +{interval_ua_samples - nominal_interval_samples:.2f} samples/pulse")

                        

                        # Quick sanity check on stimulation frequency
                        print(f"[DEBUG] Metadata stim frequency: {stim_freq_meta} Hz")
                        print(f"[DEBUG] Pulse interval: {pulse_interval_ms} ms = {interval_ua_samples:.4f} UA samples")
                        print(f"[DEBUG] fs_ua: {fs_ua} Hz")
                        print(f"[DEBUG] fs_intan: {fs_intan} Hz")
                        print(f"[DEBUG] scale_corrected: {scale_corrected}")
                        print(f"[DEBUG] shift_samp_intan: {shift_samp_intan}")
                        print(f"[DEBUG] First block: Intan [{starts_intan[0]} - {ends_intan[0]}]")
                        print(f"[DEBUG] First block: UA [{starts_ua[0]} - {ends_ua[0]}]")

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

                        print(f"[IPCA] Generated {len(intan_pulses_base_ua)} base pulse times in UA coordinates")
                        print(f"[IPCA] Using interval of {interval_ua_samples:.4f} UA samples ({pulse_interval_ms:.4f} ms)")

                        # ══════════════════════════════════════════════════════════════════════
                        # STEP 2: APPLY PER-CHANNEL LAGS
                        # Uses DIRECT channel name lookup (no NSP->Electrode mapping needed)
                        # ══════════════════════════════════════════════════════════════════════
                        n_channels = rec_ns6.get_num_channels()
                        channel_ids = rec_ns6.get_channel_ids()
                        
                        # Calculate fallback lag (median of all known lags)
                        available_lags = list(channel_lags_by_name.values())
                        fallback_lag = int(np.median(available_lags)) if available_lags else 17
                        print(f"[IPCA] Fallback lag (median): {fallback_lag} samples")
                        
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
                            if ch_idx < 3:
                                print(f"[DEBUG] Ch {ch_name}: lag={lag} samp [{lookup_method}], "
                                      f"{len(intan_pulses_per_channel[ch_idx])} pulses")
                        
                        print(f"[IPCA] Lag lookup stats: {lag_stats['by_name']} by_name, "
                              f"{lag_stats['by_electrode']} by_electrode, {lag_stats['fallback']} fallback")
                        
                        n_pulses = max(len(times) for times in intan_pulses_per_channel.values())
                        print(f"[IPCA] Maximum {n_pulses} pulses across all channels")
                        
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
                        print(f"[IPCA] Extracted {len(micro_signal_array)} artifacts. Shape: {micro_signal_array.shape}")

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
                        status = "✓ ALIGNED" if alignment_ok else "✗ MISALIGNED"
                        status_color = "green" if alignment_ok else "red"

                        ax.set_title(
                            f'BR {br_idx:03d} | Ch {test_ch_name} | {n_check} pulses\n'
                            f'Peak jitter: {drift_std:.1f} samples (range: {drift_range}) — {status}',
                            fontsize=12, color=status_color if not alignment_ok else 'black'
                        )

                        ax.legend(loc='upper right')
                        ax.grid(True, alpha=0.3)

                        plt.tight_layout()
                        diag_path = UA_CKPT_OUT.parent.parent / "figures" / f"artifact_alignment_br{br_idx:03d}.png"
                        diag_path.parent.mkdir(parents=True, exist_ok=True)
                        plt.savefig(diag_path, dpi=120)
                        plt.close()

                        # Console summary
                        print(f"[DIAG] Artifact alignment: {status}")
                        print(f"[DIAG]   Peak jitter: {drift_std:.2f} samples (range: {drift_range})")
                        print(f"[DIAG]   Saved: {diag_path.name}")
                        
                        # ══════════════════════════════════════════════════════════════════════
                        # STEP 4: REGION-WISE IPCA CORRECTION
                        # ══════════════════════════════════════════════════════════════════════
                        from RCP_analysis.python.functions.artifact_correction import Template
                        corrector = IPCA_Artifact_Correction(rank=IPCA_RANK)
                        
                        micro_corrected = micro_signal_array.copy()
                        
                        valid_idx = np.where(ua_elec > 0)[0]
                        target_ch_regions = [int(ua_region[i]) for i in valid_idx]
                        
                        region_to_idxs = {}
                        for valid_ch_idx, reg_idx in zip(valid_idx, target_ch_regions):
                            reg_name = ua_region_names[reg_idx] if reg_idx >= 0 else "Unknown"
                            region_to_idxs.setdefault(reg_name, []).append(valid_ch_idx)
                        
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
                        # STEP 5: BUILD CORRECTED RECORDING WITH PER-CHANNEL PLACEMENT
                        # ══════════════════════════════════════════════════════════════════════
                        print("[IPCA] Building per-channel IPCACorrectedRecording...")
                        
                        rec_corr_mem = PerChannelIPCACorrectedRecording(rec_ns6, micro_map_per_channel, micro_corrected)
                        
                        print("[IPCA] Highpass filtering after correction...")
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
                            out_fig = UA_CKPT_OUT.parent.parent / "figures" / f"debug_MACRO_UA_br{br_idx:03d}_ch{el_out}.png"
                            out_fig.parent.mkdir(parents=True, exist_ok=True)
                            fig.savefig(out_fig, dpi=150)
                            plt.close(fig)
                            print(f"!!! SAVED MACRO DEBUG FIG FOR CH {el_out}: {out_fig}")

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
        if skip_multiproc:
            print("[IPCA] Saving sequentially (n_jobs=1) because custom nodes cannot be pickled...")
            rec_artif_removed = rec_artif_removed.save(folder=out_dir, overwrite=True, n_jobs=1)
        else:
            rec_artif_removed = rec_artif_removed.save(folder=out_dir, overwrite=True)
            
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
        
        print(f"[INFO] Segments of recording: {n_seg}, Average noise level: {np.nanmean(noise_levels)}")
        
        
        # SpikeInterface peak detection requires channel locations (even for 1D matched filtering)
        n_ch_total = rec_artif_removed.get_num_channels()
        locs = np.zeros((n_ch_total, 2))
        side = int(np.ceil(np.sqrt(n_ch_total)))
        for i in range(n_ch_total):
            locs[i, 0] = (i % side) * 400.0
            locs[i, 1] = (i // side) * 400.0
        rec_artif_removed.set_channel_locations(locs)
        
        try:
            print("Detecting peaks simultaneously on all channels with generic template via matched filtering...")
            # Single template for all
            peaks = detect_peaks(
                rec_artif_removed,
                method="matched_filtering",
                detect_threshold=THRESH,
                peak_sign=PEAK_SIGN,
                ms_before=ms_before,
                prototype=ua_template,
                radius_um=0,
                n_jobs=1,
            )
                
        except Exception as exc:
            print(f"[WARN] Torch matched filtering failed ({exc}). Falling back to CPU locally_exclusive...")
            peaks = detect_peaks(
                rec_artif_removed,
                method="locally_exclusive", # Fallback logic
                detect_threshold=THRESH,
                peak_sign=PEAK_SIGN,
                noise_levels=noise_levels,
            )

        if any([MIN_AMPLITUDE_UV, MAX_AMPLITUDE_UV, MIN_SNR]):
            print(f"[Filter] Applying amplitude/SNR filtering:")
            if MIN_AMPLITUDE_UV:
                print(f"  - Min amplitude: {MIN_AMPLITUDE_UV} µV")
            if MAX_AMPLITUDE_UV:
                print(f"  - Max amplitude: {MAX_AMPLITUDE_UV} µV")
            if MIN_SNR:
                print(f"  - Min SNR: {MIN_SNR}")
            
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
            touchscreen_sig = np.asarray(touchscreen_sig, float)
            ts_state_num = np.zeros_like(touchscreen_sig, dtype=np.int8)
            ts_state_char = np.full(touchscreen_sig.shape, "N", dtype="U1")

            ts_state_num[touchscreen_sig >= TOUCHSCREEN_THRES_A] = 1
            ts_state_num[touchscreen_sig >= TOUCHSCREEN_THRES_B] = 2

            ts_state_char[ts_state_num == 1] = "A"
            ts_state_char[ts_state_num == 2] = "B"
            
            # --- Fallback to DLC kinematics if touchscreen signal is bad / unresponsive ---
            if np.max(ts_state_num) == 0:
                print(f"[touchscreen] Signal max voltage ({np.max(touchscreen_sig):.1f}) below threshold, falling back to DLC kinematics.")
                import pandas as pd
                
                # Base session name without date (e.g., NRR_RW022_001) if date is present
                sess_parts = sess.name.split("_", 1)
                short_name = sess_parts[1] if len(sess_parts) > 1 and sess_parts[0].isdigit() else sess.name
                
                csv_files = list(BEHV_CKPT_ROOT.rglob(f"*{short_name}*aligned.csv"))
                
                if csv_files:
                    csv_path = csv_files[0]
                    print(f"  > Reading DLC CSV: {csv_path.name}")
                    try:
                        df = pd.read_csv(csv_path, header=[0, 1, 2])
                        flat_headers = ["_".join([str(c) for c in col if "Unnamed" not in str(c)]).strip() for col in df.columns]
                        df.columns = flat_headers
                    except:
                        df = pd.read_csv(csv_path, header=0)
                    
                    # Try to extract the middle finger columns for Cam 0
                    cam_idx = 0
                    col_y = None
                    for i, h in enumerate(df.columns):
                         h_str = str(h).lower()
                         if "middle_y" in h_str and (f"cam-{cam_idx}" in h_str or f"cam{cam_idx}" in h_str or f"camera {cam_idx}" in h_str):
                             col_y = i
                             break
                    if col_y is None: # fallback
                         for i, h in enumerate(df.columns):
                              h_str = str(h).lower()
                              if "middle_y" in h_str and "cam" not in h_str:
                                  col_y = i
                                  break
                    
                    if col_y is not None:
                        ser_y = pd.to_numeric(df.iloc[:, col_y], errors="coerce")
                        dlc_fps = 100.0  # standard behavioral framing rate
                        
                        # Apply thresholds: Below 190 -> Target B, Above 240 -> Target A
                        dlc_state_num = np.zeros(len(ser_y), dtype=np.int8)
                        dlc_state_num[ser_y > 240.0] = 1 # A
                        dlc_state_num[ser_y < 190.0] = 2 # B
                        
                        # We often have erratic movement or noise. Optional: filter jumps (jump > 10.0). 
                        # However, given we just need the binary state map (and events are debounced down the pipeline), 
                        # we can rely on just thresholding the raw position, or replicate the basic jump logic.
                        diff = ser_y.diff().abs()
                        valid_mask = (diff <= 10.0) & (ser_y.diff(-1).abs() <= 10.0)
                        dlc_state_num[~valid_mask] = 0
                        
                        # Map 100Hz kinematics back to Blackrock fs_ts sampling rate
                        t_dlc = np.arange(len(ser_y)) / dlc_fps
                        t_ts = np.arange(len(touchscreen_sig)) / fs_ts
                        
                        idx_map = (t_ts * dlc_fps).astype(int)
                        # Clip indices to prevent out of bounds
                        idx_map = np.clip(idx_map, 0, len(dlc_state_num) - 1)
                        
                        ts_state_num = dlc_state_num[idx_map]
                        
                        ts_state_char = np.full(len(ts_state_num), "N", dtype="U1")
                        ts_state_char[ts_state_num == 1] = "A"
                        ts_state_char[ts_state_num == 2] = "B"
                        
                        print(f"  > Reassigned from DLC. Unique kinematic states: {np.unique(ts_state_num)}")
                    else:
                        print("  [WARN] Could not find Camera 0 middle_y column in CSV.")
                else:
                     print(f"  [WARN] DLC CSV not found for session footprint: *{short_name}*aligned.csv")

            print("Unique touchscreen states:", np.unique(ts_state_num))
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
