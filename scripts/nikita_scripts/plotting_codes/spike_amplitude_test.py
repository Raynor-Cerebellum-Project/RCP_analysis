import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION - Same style as your existing scripts
# ══════════════════════════════════════════════════════════════════════

# Which session to analyze
BR_IDX = 17  # e.g., BR_017 / Condition 17

# Region to analyze (or None for all)
TARGET_REGION = "M1 inferior"  # e.g., "M1 inferior", "M1 superior", "PMd", or None for all

# Specific electrodes to analyze (or None for all in region)
TARGET_ELECTRODES = None  # e.g., [115, 116, 133, 134, 135, 136, 137, 138, 139] or None

# Trial to focus on (or None to analyze all trials)
TARGET_TRIAL = 6  # e.g., 6 or None

# Peak sign used in detection
PEAK_SIGN = "neg"

# Window around peak to search for true extremum (in samples)
WINDOW_SAMPLES = 15

# Time window relative to stim onset (ms) - matches your plot x-axis
TIME_WINDOW_MS = (-300, 300)  # e.g., (-300, 300) to match your figure

# ══════════════════════════════════════════════════════════════════════
# LOAD DATA - Using your existing infrastructure
# ══════════════════════════════════════════════════════════════════════

import spikeinterface as si  # Add this import at the top if not already there

# Find session folder
BR_SESSION_FOLDERS = rcp.list_br_sessions(BR_ROOT)
sess = None
for s in BR_SESSION_FOLDERS:
    if int(s.name.split('_')[-1]) == BR_IDX:
        sess = s
        break

if sess is None:
    raise ValueError(f"Could not find session for BR_IDX={BR_IDX}")

print(f"Session: {sess.name}")
print(f"Condition/BR Index: {BR_IDX}")

# Load preprocessed recording
pp_dir = UA_CKPT_ROOT / f"pp__{sess.name}__NS6"
if not pp_dir.exists():
    raise FileNotFoundError(f"Preprocessed recording not found: {pp_dir}")

print(f"Loading preprocessed recording from: {pp_dir}")

# FIXED: Use si.load() instead of se.read_binary_folder()
rec = si.load(pp_dir)

# Load rates/peaks data
rates_files = list(UA_CKPT_ROOT.glob(f"rates__{sess.name}__*.npz"))
if not rates_files:
    raise FileNotFoundError(f"No rates file found for session {sess.name}")

rates_npz = rates_files[0]
print(f"Loading peaks from: {rates_npz.name}")
data = np.load(rates_npz, allow_pickle=True)

peaks = data['peaks']
noise_levels = data['noise_levels']
ua_elec = data['ua_elec']
ua_region = data['ua_region']
ua_region_names = data['ua_region_names']
ua_nsp = data.get('ua_nsp', np.zeros_like(ua_elec))

meta = data['meta'].item() if 'meta' in data else {}
fs = meta.get('fs_ua', rec.get_sampling_frequency())

n_channels = rec.get_num_channels()
channel_ids = rec.get_channel_ids()
n_samples = rec.get_num_samples()

print(f"\nRecording: {n_channels} channels, {n_samples:,} samples, {fs} Hz")
print(f"Total detected peaks: {len(peaks):,}")

# ══════════════════════════════════════════════════════════════════════
# LOAD TRIAL/STIM TIMING INFO
# ══════════════════════════════════════════════════════════════════════

# Get stim timing from Intan data (same as your preprocessing script)
stim_npz_path, intan_session_name = rcp.stim_npz_path_from_br_idx(BR_IDX, METADATA_CSV, NPRW_AUX_DATA)

if stim_npz_path and stim_npz_path.exists():
    stim = rcp.load_stim_detection(stim_npz_path)
    block_bounds_intan = stim.get("block_bounds_samples", np.empty((0, 2), dtype=int))
    
    # Convert to UA coordinates (same logic as your preprocessing)
    SHIFT_CSV = METADATA_ROOT / "br_to_intan_shifts.csv"
    br2fs_intan = rcp.get_metadata_mapping(SHIFT_CSV, 'br_idx', 'fs_intan')
    br2shift_samp_intan = rcp.get_metadata_mapping(SHIFT_CSV, 'br_idx', 'shift_sample')
    
    fs_intan = float(br2fs_intan.get(BR_IDX, fs))
    shift_raw = br2shift_samp_intan.get(BR_IDX, None)
    
    if shift_raw is not None and str(shift_raw).strip() != "":
        shift_samp_intan = float(shift_raw)
        PPM_CORRECTION = -13.951
        scale_corrected = (fs / fs_intan) * (1.0 + PPM_CORRECTION / 1e6)
        
        starts_intan = block_bounds_intan[:, 0].astype(np.int64)
        ends_intan = block_bounds_intan[:, 1].astype(np.int64)
        
        trial_starts_ua = np.round((starts_intan - shift_samp_intan) * scale_corrected).astype(np.int64)
        trial_ends_ua = np.round((ends_intan - shift_samp_intan) * scale_corrected).astype(np.int64)
        
        # Filter valid trials
        valid = (trial_ends_ua > trial_starts_ua) & (trial_starts_ua >= 0) & (trial_ends_ua <= n_samples)
        trial_starts_ua = trial_starts_ua[valid]
        trial_ends_ua = trial_ends_ua[valid]
        
        n_trials = len(trial_starts_ua)
        print(f"Found {n_trials} trials/stim blocks")
    else:
        print("[WARN] No shift data found, cannot determine trial times")
        trial_starts_ua = np.array([])
        trial_ends_ua = np.array([])
        n_trials = 0
else:
    print("[WARN] No stim file found, cannot determine trial times")
    trial_starts_ua = np.array([])
    trial_ends_ua = np.array([])
    n_trials = 0

# ══════════════════════════════════════════════════════════════════════
# FILTER BY REGION/ELECTRODES
# ══════════════════════════════════════════════════════════════════════

channel_mask = np.ones(n_channels, dtype=bool)

if TARGET_REGION is not None:
    region_idx = None
    for i, name in enumerate(ua_region_names):
        if name == TARGET_REGION:
            region_idx = i
            break
    
    if region_idx is None:
        print(f"Available regions: {list(ua_region_names)}")
        raise ValueError(f"Region '{TARGET_REGION}' not found")
    
    channel_mask &= (ua_region == region_idx)
    print(f"\nFiltering to region: {TARGET_REGION}")

if TARGET_ELECTRODES is not None:
    elec_mask = np.isin(ua_elec, TARGET_ELECTRODES)
    channel_mask &= elec_mask
    print(f"Filtering to electrodes: {TARGET_ELECTRODES}")

valid_ch_indices = np.where(channel_mask)[0]
print(f"Analyzing {len(valid_ch_indices)} channels:")

for ch_idx in valid_ch_indices:
    elec = int(ua_elec[ch_idx])
    reg_idx = int(ua_region[ch_idx])
    reg_name = ua_region_names[reg_idx] if reg_idx < len(ua_region_names) else "Unknown"
    print(f"  Ch {ch_idx:3d} | Elec {elec:3d} | {reg_name}")

# ══════════════════════════════════════════════════════════════════════
# FILTER PEAKS BY CHANNEL, TRIAL, AND TIME WINDOW
# ══════════════════════════════════════════════════════════════════════

# Filter peaks to selected channels
peak_ch_mask = np.isin(peaks['channel_index'], valid_ch_indices)
filtered_peaks = peaks[peak_ch_mask]
print(f"\nPeaks in selected channels: {len(filtered_peaks):,}")

# Filter by trial if specified
if TARGET_TRIAL is not None and n_trials > 0:
    if TARGET_TRIAL < 1 or TARGET_TRIAL > n_trials:
        raise ValueError(f"TARGET_TRIAL={TARGET_TRIAL} out of range (1-{n_trials})")
    
    trial_idx = TARGET_TRIAL - 1  # Convert to 0-indexed
    trial_start = trial_starts_ua[trial_idx]
    trial_end = trial_ends_ua[trial_idx]
    
    # Apply time window relative to trial start
    if TIME_WINDOW_MS is not None:
        win_start_samp = int(TIME_WINDOW_MS[0] / 1000.0 * fs)
        win_end_samp = int(TIME_WINDOW_MS[1] / 1000.0 * fs)
        analysis_start = trial_start + win_start_samp
        analysis_end = trial_start + win_end_samp
    else:
        analysis_start = trial_start
        analysis_end = trial_end
    
    # Ensure bounds are valid
    analysis_start = max(0, analysis_start)
    analysis_end = min(n_samples, analysis_end)
    
    time_mask = (filtered_peaks['sample_index'] >= analysis_start) & \
                (filtered_peaks['sample_index'] < analysis_end)
    filtered_peaks = filtered_peaks[time_mask]
    
    print(f"Trial {TARGET_TRIAL}: samples {trial_start:,} - {trial_end:,}")
    print(f"Analysis window: {TIME_WINDOW_MS[0]} to {TIME_WINDOW_MS[1]} ms relative to stim onset")
    print(f"Peaks in trial/window: {len(filtered_peaks):,}")

elif n_trials > 0 and TIME_WINDOW_MS is not None:
    # Analyze all trials within the time window
    all_trial_peaks = []
    for t_idx in range(n_trials):
        trial_start = trial_starts_ua[t_idx]
        win_start_samp = int(TIME_WINDOW_MS[0] / 1000.0 * fs)
        win_end_samp = int(TIME_WINDOW_MS[1] / 1000.0 * fs)
        analysis_start = max(0, trial_start + win_start_samp)
        analysis_end = min(n_samples, trial_start + win_end_samp)
        
        time_mask = (filtered_peaks['sample_index'] >= analysis_start) & \
                    (filtered_peaks['sample_index'] < analysis_end)
        all_trial_peaks.append(filtered_peaks[time_mask])
    
    filtered_peaks = np.concatenate(all_trial_peaks) if all_trial_peaks else filtered_peaks
    print(f"Analyzing all {n_trials} trials, window: {TIME_WINDOW_MS[0]} to {TIME_WINDOW_MS[1]} ms")
    print(f"Peaks across all trials/windows: {len(filtered_peaks):,}")

# ══════════════════════════════════════════════════════════════════════
# EXTRACT AMPLITUDES FOR FILTERED PEAKS
# ══════════════════════════════════════════════════════════════════════

print("\nExtracting peak amplitudes...")

n_peaks = len(filtered_peaks)
amplitudes = np.zeros(n_peaks, dtype=np.float32)

for i, peak in enumerate(filtered_peaks):
    if i % 1000 == 0 and i > 0:
        print(f"  Processed {i:,} / {n_peaks:,} peaks...")
    
    sample_idx = peak['sample_index']
    ch_idx = peak['channel_index']
    
    s0 = max(0, sample_idx - WINDOW_SAMPLES)
    s1 = min(n_samples, sample_idx + WINDOW_SAMPLES)
    
    ch_id = channel_ids[ch_idx]
    trace = rec.get_traces(
        start_frame=s0, end_frame=s1,
        channel_ids=[ch_id], return_in_uV=True
    ).flatten()
    
    if PEAK_SIGN == "neg":
        amplitudes[i] = np.abs(np.min(trace))
    elif PEAK_SIGN == "pos":
        amplitudes[i] = np.max(trace)
    else:
        amplitudes[i] = np.max(np.abs(trace))

print(f"Done. Extracted {n_peaks:,} amplitudes.\n")

# Compute noise-normalized amplitudes
ch_indices = filtered_peaks['channel_index']
amp_sigma = amplitudes / noise_levels[ch_indices]

# ══════════════════════════════════════════════════════════════════════
# DISPLAY STATISTICS
# ══════════════════════════════════════════════════════════════════════

print("=" * 70)
print(f"AMPLITUDE ANALYSIS: BR_{BR_IDX:03d} | {TARGET_REGION or 'All Regions'}")
if TARGET_TRIAL:
    print(f"Trial: {TARGET_TRIAL} | Window: {TIME_WINDOW_MS} ms")
print("=" * 70)

print(f"\n{'OVERALL STATISTICS':^70}")
print("-" * 70)
print(f"  Total peaks analyzed:  {n_peaks:,}")
print(f"  Min amplitude:         {amplitudes.min():.1f} µV  ({amp_sigma.min():.1f}σ)")
print(f"  Max amplitude:         {amplitudes.max():.1f} µV  ({amp_sigma.max():.1f}σ)")
print(f"  Mean amplitude:        {amplitudes.mean():.1f} µV  ({amp_sigma.mean():.1f}σ)")
print(f"  Median amplitude:      {np.median(amplitudes):.1f} µV  ({np.median(amp_sigma):.1f}σ)")
print(f"  Std amplitude:         {amplitudes.std():.1f} µV  ({amp_sigma.std():.1f}σ)")

print(f"\n{'PERCENTILES':^70}")
print("-" * 70)
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print(f"  {'Percentile':>12}  {'Amplitude (µV)':>15}  {'Amplitude (σ)':>15}")
for p in percentiles:
    amp_p = np.percentile(amplitudes, p)
    sig_p = np.percentile(amp_sigma, p)
    print(f"  {p:>11}%  {amp_p:>15.1f}  {sig_p:>15.1f}")

print(f"\n{'THRESHOLD ANALYSIS - Absolute (µV)':^70}")
print("-" * 70)
abs_thresholds = [10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100]
print(f"  {'Threshold':>12}  {'Removed':>12}  {'Kept':>12}  {'% Removed':>12}")
for thresh in abs_thresholds:
    n_removed = np.sum(amplitudes < thresh)
    n_kept = n_peaks - n_removed
    pct_removed = 100 * n_removed / n_peaks if n_peaks > 0 else 0
    marker = " <--" if thresh == 20 else ""
    print(f"  {thresh:>10} µV  {n_removed:>12,}  {n_kept:>12,}  {pct_removed:>11.1f}%{marker}")

print(f"\n{'THRESHOLD ANALYSIS - Relative (× noise)':^70}")
print("-" * 70)
sigma_thresholds = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]
print(f"  {'Threshold':>12}  {'Removed':>12}  {'Kept':>12}  {'% Removed':>12}")
for thresh in sigma_thresholds:
    n_removed = np.sum(amp_sigma < thresh)
    n_kept = n_peaks - n_removed
    pct_removed = 100 * n_removed / n_peaks if n_peaks > 0 else 0
    print(f"  {thresh:>11.1f}σ  {n_removed:>12,}  {n_kept:>12,}  {pct_removed:>11.1f}%")

print(f"\n{'PER-CHANNEL STATISTICS':^70}")
print("-" * 70)
print(f"  {'Elec':>5}  {'N_spk':>7}  {'Min':>7}  {'Med':>7}  {'Mean':>7}  {'Max':>7}  {'Noise':>7}  {'Med_σ':>7}")

for ch_idx in valid_ch_indices:
    ch_mask = filtered_peaks['channel_index'] == ch_idx
    ch_amps = amplitudes[ch_mask]
    n_spk = len(ch_amps)
    
    elec = int(ua_elec[ch_idx])
    noise = noise_levels[ch_idx]
    
    if n_spk > 0:
        min_amp = np.min(ch_amps)
        med_amp = np.median(ch_amps)
        mean_amp = np.mean(ch_amps)
        max_amp = np.max(ch_amps)
        med_sigma = med_amp / noise if noise > 0 else 0
        
        print(f"  {elec:>5}  {n_spk:>7,}  {min_amp:>7.1f}  {med_amp:>7.1f}  "
              f"{mean_amp:>7.1f}  {max_amp:>7.1f}  {noise:>7.1f}  {med_sigma:>7.1f}")
    else:
        print(f"  {elec:>5}  {n_spk:>7}  {'--':>7}  {'--':>7}  {'--':>7}  {'--':>7}  {noise:>7.1f}  {'--':>7}")

# ══════════════════════════════════════════════════════════════════════
# RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

# Find threshold that removes ~5-15% of spikes (likely noise)
for thresh in abs_thresholds:
    pct_removed = 100 * np.sum(amplitudes < thresh) / n_peaks if n_peaks > 0 else 0
    if 5 <= pct_removed <= 15:
        print(f"\n  Suggested threshold: {thresh} µV")
        print(f"  This would remove {pct_removed:.1f}% of detected spikes")
        break
else:
    # If no threshold in 5-15% range, find closest
    pct_at_20 = 100 * np.sum(amplitudes < 20) / n_peaks if n_peaks > 0 else 0
    print(f"\n  At 20 µV threshold: {pct_at_20:.1f}% of spikes would be removed")

# Show distribution of low-amplitude spikes
low_amp_mask = amplitudes < 30
n_low = np.sum(low_amp_mask)
print(f"\n  Spikes below 30 µV: {n_low:,} ({100*n_low/n_peaks:.1f}%)")

if n_low > 0:
    low_amps = amplitudes[low_amp_mask]
    print(f"  Distribution of low-amplitude spikes:")
    print(f"    < 10 µV: {np.sum(low_amps < 10):,}")
    print(f"    10-15 µV: {np.sum((low_amps >= 10) & (low_amps < 15)):,}")
    print(f"    15-20 µV: {np.sum((low_amps >= 15) & (low_amps < 20)):,}")
    print(f"    20-25 µV: {np.sum((low_amps >= 20) & (low_amps < 25)):,}")
    print(f"    25-30 µV: {np.sum((low_amps >= 25) & (low_amps < 30)):,}")

print("\n" + "=" * 70)