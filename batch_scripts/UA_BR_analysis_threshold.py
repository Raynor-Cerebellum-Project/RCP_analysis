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

# Suppress annoying SpikeInterface provenance warning when manually reconstructing memory arrays
warnings.filterwarnings("ignore", message="The extractor is not serializable to file. The provenance will not be saved.")

class IPCACorrectedRecordingSegment(BaseRecordingSegment):
    def __init__(self, parent_recording_segment, micro_map, micro_corrected):
        """
        micro_map: list of tuples (p_start, p_end) indicating where artifacts are
        micro_corrected: np.ndarray of shape (n_events, n_time, n_channels) holding the IPCA patches
        """
        BaseRecordingSegment.__init__(self, **parent_recording_segment.get_times_kwargs())
        self.parent_recording_segment = parent_recording_segment
        self.micro_map = micro_map
        self.micro_corrected = micro_corrected

    def get_num_samples(self):
        return self.parent_recording_segment.get_num_samples()

    def get_traces(self, start_frame, end_frame, channel_indices):
        # 1. Ask the parent for the clean disk-backed chunk
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, channel_indices)
        
        # We need a writeable copy to patch
        traces = traces.copy() 
        
        if self.micro_map is None or len(self.micro_map) == 0:
            return traces
            
        # 2. Find any artifacts that intersect with our current [start_frame, end_frame] chunk
        for p, (p_start, p_end) in enumerate(self.micro_map):
            # Check for overlap
            overlap_start = max(start_frame, p_start)
            overlap_end = min(end_frame, p_end)
            
            if overlap_start < overlap_end:
                # Calculate indices relative to our current trace chunk
                chunk_idx_start = overlap_start - start_frame
                chunk_idx_end = overlap_end - start_frame
                
                # Calculate indices relative to the pre-computed artifact patch
                patch_idx_start = overlap_start - p_start
                patch_idx_end = overlap_end - p_start
                
                # Slice the micro_corrected patch (only for the requested channels)
                if channel_indices is None:
                    # All channels
                    patch = self.micro_corrected[p, patch_idx_start:patch_idx_end, :]
                else:
                    # Specific channels
                    patch = self.micro_corrected[p, patch_idx_start:patch_idx_end, channel_indices]
                
                # Overwrite the disk trace with the IPCA-corrected data
                traces[chunk_idx_start:chunk_idx_end, :] = patch
                
        return traces

class IPCACorrectedRecording(BaseRecording):
    """
    A SpikeInterface BaseRecording that lazily streams disk data 
    and surgically inserts pre-computed IPCA artifact patches on the fly.
    """
    def __init__(self, parent_recording, micro_map, micro_corrected):
        BaseRecording.__init__(self, 
                               parent_recording.get_sampling_frequency(), 
                               parent_recording.channel_ids, 
                               parent_recording.get_dtype())
        self.parent_recording = parent_recording
        
        # Copy properties and annotations
        parent_recording.copy_metadata(self)
        
        # Create a matching segment
        for segment_index in range(parent_recording.get_num_segments()):
            parent_segment = parent_recording._recording_segments[segment_index]
            self.add_recording_segment(IPCACorrectedRecordingSegment(parent_segment, micro_map, micro_corrected))


""" 
    This script preprocesses the Blackrock data.
    Input:
        .ns6 files from Intan
    Output:
        Checkpoint after preprocessing
        Checkpoint after thresholding and calculating MUA peak locations and firing rate
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
PEAK_SIGN  = RATES.get("peak_sign")
ARTRMV_MS_BEFORE = float(RATES.get("remove_ms_before", 5.0))
ARTRMV_TAIL_MS   = float(RATES.get("remove_tail_ms_after", 5.0))


# --- IPCA Artifact Correction Settings ---
USE_IPCA_CORRECTION  = True
IPCA_RANK            = 10
IPCA_PULSE_WINDOW_MS = (-0.2, 0.3)
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

def _find_first_significant_peak(tr_local, ref_ratio=0.5):
    """
    Given a local trace fragment (already roughly centred on a peak),
    find the first peak in the absolute derivative that exceeds `ref_ratio` 
    of the local maximum derivative.
    Returns the index relative to the start of `tr_local`.
    """
    diff_abs = np.abs(np.diff(tr_local, prepend=tr_local[0]))
    pk_idx = np.argmax(diff_abs)
    max_d = diff_abs[pk_idx]
    if max_d < 1e-6:
        return pk_idx
    for k in range(len(diff_abs)):
        if diff_abs[k] > ref_ratio * max_d:
            return k
    return pk_idx

# For triangle pulse corrected stuff, save for later
def load_anchor_for_session(out_base: Path, session: str) -> tuple[int, float]:
    """
    Return (anchor_sample_intan, fs_intan) from figures/align_BR_to_Intan/br_to_intan_shifts.csv.
    Falls back to deriving anchor_sample from anchor_ms if needed.
    """
    shifts_csv = out_base / "figures" / "align_BR_to_Intan" / "br_to_intan_shifts.csv"
    if not shifts_csv.exists():
        return 0, 30000.0  # safe fallback

    with shifts_csv.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row.get("session") == session:
                fs_intan = float(row.get("fs_intan", 30000.0))
                if row.get("anchor_sample", "") != "":
                    return int(row["anchor_sample"]), fs_intan
                if row.get("anchor_ms", "") != "":
                    anchor_ms = float(row["anchor_ms"])
                    return int(round(anchor_ms * 1e-3 * fs_intan)), fs_intan
                break

    return 0, 30000.0

def main():
    sess_folders = BR_SESSION_FOLDERS

    print("Found session folders:", len(sess_folders))

    # Filter to PROCESS_ONLY BR indices if specified
    if PROCESS_ONLY:
        original_count = len(sess_folders)
        filtered = []
        for br_idx in PROCESS_ONLY:
            match = [s for s in sess_folders if int(s.name.split('_')[-1]) == br_idx]
            if not match:
                print(f"[WARN] PROCESS_ONLY BR index {br_idx} not found in session folders; skipping.")
                continue
            filtered.extend(match)
        print(f"[INFO] PROCESS_ONLY: {len(filtered)}/{original_count} sessions selected.")
        sess_folders = filtered

    for sess in sess_folders:
        print(f"=== Session: {sess.name} ===")
        
        # Check if outputs already exist
        out_dir = UA_CKPT_OUT / f"pp__{sess.name}__NS6"
        out_npz = UA_CKPT_OUT / f"rates__{sess.name}__bin{int(BIN_MS)}ms_sigma{int(SIGMA_MS)}ms.npz"
        
        # if out_dir.exists() and out_npz.exists():
        #     print(f"[SKIP] Both outputs already exist for {sess.name}")
        #     print(f"       - Preprocessed: {out_dir}")
        #     print(f"       - Rates: {out_npz}")
        #     continue
        
        temp_bin_path = None # Initialize temp_bin_path
        touchscreen_sig, hr_sig, vog_sig, meta_ns5, meta_ns2 = rcp.extract_br_aux_streams_npz(sess, UA_AUX_DATA, CAMERA_SYNC_CH, TRIANGLE_SYNC_CH, TOUCHSCREEN_CH, HR_CH, VOG_CH) # Extract sync pulses and stuff
        fs_hr = meta_ns2["fs_hr"]
        fs_vog = meta_ns2["fs_vog"]
        fs_ts = meta_ns2["fs_ts"]
        
        rec_ns6 = se.read_blackrock(sess, stream_name = 'nsx6', all_annotations=True) # Load neural data
        br_idx = int(sess.name.split('_')[-1])
        
        rec_ns6, idx_rows, ua_elec, ua_nsp, ua_region, ua_region_names, ua_port = rcp.apply_ua_mapping_with_regions(rec_ns6, UA_MAP, br_idx, METADATA_CSV)
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
                print(f"[map] BR {br_idx:03d} -> Intan session '{intan_session_name}'")

                stim = rcp.load_stim_detection(stim_npz_path)
                block_bounds = stim.get("block_bounds_samples", [])

        if block_bounds.size and br2shift_samp_intan is not None:
            fs_intan = float(br2fs_intan.get(br_idx, 30000.0))
            shift_raw = br2shift_samp_intan.get(br_idx, None)
            if shift_raw is None or (isinstance(shift_raw, str) and shift_raw.strip() == ""):
                print(f"[WARN] Missing anchor_sample for br_idx={br_idx} in {SHIFT_CSV}. Skipping artifact removal.")
                block_bounds = np.empty((0, 2), dtype=int)
            else:
                shift_samp_intan = float(shift_raw)

            starts_intan = block_bounds[:, 0].astype(np.int64)
            ends_intan   = block_bounds[:, 1].astype(np.int64)

            # shift+scale into UA sample index space
            scale = fs_ua / fs_intan
            starts_ua = np.round((starts_intan - shift_samp_intan) * scale).astype(np.int64)
            ends_ua   = np.round((ends_intan   - shift_samp_intan) * scale).astype(np.int64)

            n_total = rec_ns6.get_num_samples()
            ends_ua = np.minimum(ends_ua, n_total)
            valid = (ends_ua > starts_ua) & (starts_ua >= 0) & (ends_ua <= n_total)
            starts_ua = starts_ua[valid]
            ends_ua   = ends_ua[valid]

            if starts_ua.size:
                if USE_IPCA_CORRECTION:
                    print(f"[IPCA] Starting Incremental PCA artifact correction on {starts_ua.size} blocks...")
                    
                    # 1. Learn precise timestamps using derivative method
                    pulse_interval_ms = 1000.0 / PARAMS.stim_hz if hasattr(PARAMS, 'stim_hz') else 3.333
                    mw_start = int(IPCA_PULSE_WINDOW_MS[0] / 1000.0 * fs_ua)
                    mw_end   = int(IPCA_PULSE_WINDOW_MS[1] / 1000.0 * fs_ua)
                    micro_n_time = mw_end - mw_start
                    
                    fw_start = int(-ARTRMV_MS_BEFORE / 1000.0 * fs_ua)
                    fw_end   = int(ARTRMV_TAIL_MS / 1000.0 * fs_ua)
                    
                    # We need a reference channel to detect peaks. Using the first valid mapped channel.
                    valid_idx = np.where(ua_elec > 0)[0]
                    target_ch_ids = [rec_ns6.get_channel_ids()[i] for i in valid_idx]
                    target_ch_regions = [int(ua_region[i]) for i in valid_idx]
                    ref_ch_id = target_ch_ids[0]
                    
                    print("[IPCA] Scanning Blackrock raw file sequentially for artifact triggers...")
                    
                    micro_signal_list = []
                    micro_map = []  # (block_idx, p_start, p_end)
                    
                    interval_idx = int(pulse_interval_ms / 1000 * fs_ua)
                    search_radius_idx = int((pulse_interval_ms * 0.4) / 1000 * fs_ua)
                    refine_samp = max(int(0.2 / 1000.0 * fs_ua), 3)
                    
                    for i, (st, en) in enumerate(zip(starts_ua, ends_ua)):
                        # Coarse search
                        search_margin = int(20.0 / 1000 * fs_ua)
                        start_search  = st + fw_start - search_margin
                        end_search    = st + fw_end + search_margin
                        if start_search < 0 or end_search > n_total: continue
                        
                        tr_search = rec_ns6.get_traces(start_frame=start_search, end_frame=end_search, channel_ids=[rec_ns6.get_channel_ids()[valid_idx[0]]], return_in_uV=True)[:, 0]
                        tr_diff_search  = np.abs(np.diff(tr_search, prepend=tr_search[0]))
                        
                        predicted_centre = -fw_start + search_margin
                        slop = int(5.0 / 1000 * fs_ua)
                        win_lo = max(0, predicted_centre - slop)
                        win_hi = min(len(tr_search), predicted_centre + slop)
                        
                        peak_in_win = np.argmax(tr_diff_search[win_lo:win_hi])
                        anchor_idx  = win_lo + peak_in_win
                        max_diff    = tr_diff_search[anchor_idx]
                        
                        if max_diff < 5.0: continue
                        
                        first_pulse_in_search = None
                        for k in range(win_lo, win_hi):
                            if tr_diff_search[k] > 0.4 * max_diff:
                                first_pulse_in_search = k
                                break
                        if first_pulse_in_search is None: continue
                        
                        coarse_radius = max(refine_samp, 3)
                        rc_lo = max(0, first_pulse_in_search - coarse_radius)
                        rc_hi = min(len(tr_search), first_pulse_in_search + coarse_radius + 1)
                        if rc_hi <= rc_lo: continue
                        offset = _find_first_significant_peak(tr_search[rc_lo:rc_hi])
                        
                        coarse_centre = rc_lo + offset
                        true_start = start_search + coarse_centre
                        
                        train_start_idx = true_start
                        train_end_idx   = en + fw_end
                        
                        if train_end_idx <= train_start_idx: continue
                        
                        anchor_final = true_start
                        pulse_centres = [anchor_final]
                        curr = anchor_final
                        while True:
                            nxt = curr + interval_idx
                            if nxt > train_end_idx: break
                            s_lo = max(0, nxt - search_radius_idx)
                            s_hi = min(n_total, nxt + search_radius_idx)
                            if s_hi <= s_lo: break
                            
                            tr_diff = np.abs(np.diff(rec_ns6.get_traces(start_frame=s_lo, end_frame=s_hi, channel_ids=[rec_ns6.get_channel_ids()[valid_idx[0]]], return_in_uV=True)[:, 0]))
                            peak_idx = np.argmax(tr_diff)
                            
                            if tr_diff[peak_idx] < 0.15 * max_diff: break
                            
                            curr = s_lo + peak_idx
                            pulse_centres.append(curr)
                            
                        pulse_centres = sorted(set(pulse_centres))
                        
                        for p_idx in pulse_centres:
                            r_lo = max(0, p_idx - refine_samp)
                            r_hi = min(n_total, p_idx + refine_samp + 1)
                            if r_lo >= r_hi: continue
                            
                            true_centre = r_lo + _find_first_significant_peak(rec_ns6.get_traces(start_frame=r_lo, end_frame=r_hi, channel_ids=[rec_ns6.get_channel_ids()[valid_idx[0]]], return_in_uV=True)[:, 0])
                            p_start = true_centre + mw_start
                            p_end   = true_centre + mw_end
                            
                            if p_start >= 0 and p_end <= n_total:
                                micro_signal_list.append(rec_ns6.get_traces(start_frame=p_start, end_frame=p_end, return_in_uV=True).copy())
                                micro_map.append((p_start, p_end))

                    if not micro_signal_list:
                        print("[IPCA] No pulses successfully extracted, skipping correction.")
                        rec_hp = spre.bandpass_filter(rec_ns6, freq_min=float(PARAMS.highpass_hz), freq_max=(PARAMS.lowpass_hz))
                        rec_artif_removed = rec_hp
                    else:
                        micro_signal_array = np.stack(micro_signal_list) # (n_pulses, n_time, n_all_channels)
                        print(f"[IPCA] Extracted {len(micro_signal_array)} artifacts. Correcting...")
                        
                        corrector = IPCA_Artifact_Correction(rank=IPCA_RANK)
                        
                        # Group valid channels by region
                        region_to_idxs = {}
                        for valid_ch_idx, reg_idx in enumerate(target_ch_regions):
                            reg_name = ua_region_names[reg_idx] if reg_idx >= 0 else "Unknown"
                            region_to_idxs.setdefault(reg_name, []).append(valid_idx[valid_ch_idx])
                            
                        # Cross-channel IPCA using existing artifact_correction.py methods,
                        # learned separately for each brain region (SMA, PMd, M1i, M1s).
                        from RCP_analysis.python.functions.artifact_correction import Template
                        micro_corrected = micro_signal_array.copy()
                        
                        n_stim, n_time, _ = micro_signal_array.shape
                        
                        print("Cross-channel IPCA by region:")
                        for reg, idxs in region_to_idxs.items():
                            if not idxs: continue
                            
                            # Pool all pulses from channels within this region
                            reg_signal = micro_signal_array[:, :, idxs]  # (n_stim, n_time, n_reg_ch)
                            pooled_signal = np.transpose(reg_signal, (0, 2, 1)).reshape(-1, n_time)
                            
                            # Learn the shared template for this specific region
                            reg_template = Template()
                            _, reg_template = corrector.ipca_template_per_channel(pooled_signal, reg_template)
                            
                            print(f"  [{reg}] Shared subspace learned from {len(idxs)} channels.")
                            
                            # Apply this region's template to each channel in the region
                            for ch_idx in idxs:
                                signal_ch = micro_signal_array[:, :, ch_idx].copy()
                                baseline = np.mean(signal_ch[:, :3], axis=1, keepdims=True)
                                centered = signal_ch - baseline
                                
                                # Projection into the subspace
                                artifact = centered @ reg_template.weights.T @ reg_template.weights
                                
                                # Reconstruct corrected signal
                                corr_ch = centered - artifact + baseline
                                micro_corrected[:, :, ch_idx] = corr_ch
                                
                        print("[IPCA] Building dynamically-correcting IPCACorrectedRecording...")
                        rec_corr_mem = IPCACorrectedRecording(rec_ns6, micro_map, micro_corrected)
                        
                        print("[IPCA] Filter after correction...")
                        rec_hp = spre.highpass_filter(rec_corr_mem, freq_min=float(PARAMS.highpass_hz))
                        rec_artif_removed = rec_hp
                        
                        del micro_signal_array, micro_corrected
                        gc.collect()

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
            rec_artif_removed.save(folder=out_dir, overwrite=True, n_jobs=1)
        else:
            rec_artif_removed.save(folder=out_dir, overwrite=True)
            
        print(f"[{sess.name}] (ns6) saved preprocessed -> {out_dir}")
        
        # del rec_artif_removed
        if 'rec_corr_mem' in locals(): del rec_corr_mem
        if 'rec_hp' in locals(): del rec_hp
        gc.collect()
        
        n_seg = rec_artif_removed.get_num_segments()
        # Detect peaks
        noise_levels = si.get_noise_levels(rec_artif_removed, method="mad", return_in_uV=False) # They didn't write return_in_uV in their documentation
        
        print(f"[INFO] Segments of recording: {n_seg}, Average noise level: {np.nanmean(noise_levels)}")
        try:
            # Force n_jobs=1 for PyTorch backend to prevent GPU memory fragmentation / OOM
            peaks = detect_peaks(
                rec_artif_removed,
                method="by_channel_torch",
                method_kwargs=dict(
                    detect_threshold=THRESH,
                    peak_sign=PEAK_SIGN,
                    noise_levels=noise_levels,
                ),
                job_kwargs=dict(n_jobs=1),
            )
        except Exception as exc:
            print(f"[WARN] Torch peak detection failed ({exc}). Falling back to CPU locally_exclusive...")
            peaks = detect_peaks(
                rec_artif_removed,
                method="by_channel",
                detect_threshold=THRESH,
                peak_sign=PEAK_SIGN,
                noise_levels=noise_levels,
                n_jobs=PARAMS.parallel_jobs,
            )
        
        # Threshold to get touchscreen state
        ts_state_num = np.zeros(0, dtype=np.int8)
        ts_state_char = np.full(0, 'N', dtype='U1')

        if touchscreen_sig is not None:
            touchscreen_sig = np.asarray(touchscreen_sig, float)
            ts_state_num = np.zeros_like(touchscreen_sig, dtype=np.int8)
            ts_state_char = np.full(touchscreen_sig.shape, 'N', dtype='U1')

            ts_state_num[touchscreen_sig >= TOUCHSCREEN_THRES_A] = 1
            ts_state_num[touchscreen_sig >= TOUCHSCREEN_THRES_B] = 2

            ts_state_char[ts_state_num == 1] = 'A'
            ts_state_char[ts_state_num == 2] = 'B'
            
            # --- Fallback to DLC kinematics if touchscreen signal is bad / unresponsive ---
            if np.max(ts_state_num) == 0:
                print(f"[touchscreen] Signal max voltage ({np.max(touchscreen_sig):.1f}) below threshold, falling back to DLC kinematics.")
                import pandas as pd
                
                # Base session name without date (e.g., NRR_RW022_001) if date is present
                sess_parts = sess.name.split('_', 1)
                short_name = sess_parts[1] if len(sess_parts) > 1 and sess_parts[0].isdigit() else sess.name
                
                csv_files = list(BEHV_CKPT_ROOT.rglob(f"*{short_name}*aligned.csv"))
                
                if csv_files:
                    csv_path = csv_files[0]
                    print(f"  > Reading DLC CSV: {csv_path.name}")
                    try:
                        df = pd.read_csv(csv_path, header=[0, 1, 2])
                        flat_headers = ['_'.join([str(c) for c in col if 'Unnamed' not in str(c)]).strip() for col in df.columns]
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
                        ser_y = pd.to_numeric(df.iloc[:, col_y], errors='coerce')
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
                        
                        ts_state_char = np.full(len(ts_state_num), 'N', dtype='U1')
                        ts_state_char[ts_state_num == 1] = 'A'
                        ts_state_char[ts_state_num == 2] = 'B'
                        
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
