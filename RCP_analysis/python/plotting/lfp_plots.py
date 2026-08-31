"""
LFP analysis and plotting helpers.

This module contains the plotting/analysis functions extracted from
plot_lfp_cleaner.py. The top-level pipeline/entrypoint remains in
plot_lfp_cleaner.py for now.
"""

import warnings
import io
import re
from pathlib import Path

import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal, stats

from RCP_analysis.python.functions.lfp_config import (
    SESSION_FILTER,
    SESSION_ID_PATTERN,
    FIGURE_CONFIG,
    TIME_WINDOWS,
    ANALYSIS_CONFIG,
    PLOT_CONFIG,
    WAVELET_TEST_CONFIG,
    TORCH_AVAILABLE,
    torch,
)

from RCP_analysis.python.functions.lfp_tfr import (
    get_foi_with_notch_gaps,
    mask_notch_regions,
    zscore_normalize_spectrogram,
    compute_wavelet_spectrogram,
    compute_wavelet_spectrogram_batch_torch,
    compute_wavelet_spectrogram_batch_torch_adaptive,
    compute_morlet_adaptive_spectrogram,
    compute_stft_spectrogram,
    compute_multitaper_spectrogram,
    compute_paul_wavelet_spectrogram,
    compute_tfr_method,
)



# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_confidence_interval(data, axis=0, confidence=0.95):
    """
    Compute mean and confidence interval using SEM × z-score.
    
    Args:
        data: Array of values
        axis: Axis to compute statistics over
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        mean, ci_lower, ci_upper
    """
    z_score = stats.norm.ppf((1 + confidence) / 2)  # 1.96 for 95%
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = np.nanmean(data, axis=axis)
        n = np.sum(~np.isnan(data), axis=axis)
        sem = np.nanstd(data, axis=axis) / np.sqrt(np.maximum(n, 1))
        ci = sem * z_score
    
    return mean, mean - ci, mean + ci

def get_region_color(region_name):
    """Get color for a region, with fallback."""
    colors = PLOT_CONFIG['region_colors']
    # Handle various naming conventions
    for key in colors:
        if key.lower() in region_name.lower():
            return colors[key]
    return 'gray'

def add_band_shading(ax, bands=None, alpha=0.08, skip_notch=True):
    """Add subtle background shading for frequency bands, skipping notch regions."""
    if bands is None:
        bands = ANALYSIS_CONFIG['bands']
    
    notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)]) if skip_notch else []
    
    colors = ['#E91E63', '#9C27B0', '#3F51B5', '#4CAF50', '#FF9800', '#F44336']
    
    for i, (band_name, (f_lo, f_hi)) in enumerate(bands.items()):
        # Check if this band overlaps with any notch region
        for notch_lo, notch_hi in notch_regions:
            if f_lo < notch_hi and f_hi > notch_lo:
                # Band overlaps with notch - split the shading
                if f_lo < notch_lo:
                    ax.axvspan(f_lo, notch_lo, color=colors[i % len(colors)], alpha=alpha)
                if f_hi > notch_hi:
                    ax.axvspan(notch_hi, f_hi, color=colors[i % len(colors)], alpha=alpha)
                break
        else:
            # No overlap with any notch
            ax.axvspan(f_lo, f_hi, color=colors[i % len(colors)], alpha=alpha)

def add_stim_marker(ax, stim_duration_ms=100):
    """Add stimulation period marker to time-series plot."""
    ax.axvspan(0, stim_duration_ms, color='red', alpha=0.1, label='Stim')
    ax.axvline(0, color='red', ls='--', lw=1, alpha=0.7)

def compute_array_median_raw_spectrograms(data, groups, fs=1000,
                                          region_grids=None, elec_to_idx=None):
    """
    Compute raw Morlet spectrogram median across all valid trial/channel pairs
    for each array/region.

    This is the centralized raw array-median spectrogram helper used by:
      - generate_per_ua_array_median_spectrogram(...)
      - compute_control_median_baseline_norm(...)

    Return keys are intentionally preserved:
      'Sxx'
      'f'
      't_ms'
      'n_valid_channels'
      'n_valid_trial_channel_pairs'
    """

    spec_cfg = PLOT_CONFIG.get('spectrogram', {})

    freq_min = spec_cfg.get('freq_min', 1)
    freq_max = spec_cfg.get('freq_max', 120)
    freq_step = spec_cfg.get('freq_step', 1)

    tfr_method = spec_cfg.get('tfr_method', 'morlet').lower()

    n_cycles = spec_cfg.get(
        'wavelet_cycles',
        ANALYSIS_CONFIG.get('wavelet_cycles', 7)
    )

    adaptive_cycles_low = spec_cfg.get('adaptive_cycles_low', 3)
    adaptive_cycles_high = spec_cfg.get('adaptive_cycles_high', 7)
    adaptive_freq_low_max = spec_cfg.get('adaptive_freq_low_max', 15)
    adaptive_freq_high_min = spec_cfg.get('adaptive_freq_high_min', 16)

    if (
        tfr_method == 'morlet_adaptive'
        and adaptive_freq_high_min <= adaptive_freq_low_max
    ):
        print(
            "    WARNING: adaptive_freq_high_min must be greater than "
            "adaptive_freq_low_max. Expanding transition band automatically."
        )
        adaptive_freq_high_min = adaptive_freq_low_max + 15

    foi = get_foi_with_notch_gaps(freq_min, freq_max, freq_step)

    # Adaptive Morlet uses CPU path unless/until Torch adaptive support is added.
    # Torch GPU path supports fixed-cycle Morlet and adaptive Morlet.
    use_gpu = (
        tfr_method in ('morlet', 'morlet_adaptive')
        and ANALYSIS_CONFIG.get('use_gpu_wavelet', False)
        and ANALYSIS_CONFIG.get('gpu_backend', 'torch').lower() == 'torch'
        and TORCH_AVAILABLE
        and torch is not None
        and torch.cuda.is_available()
    )

    gpu_batch_size = int(ANALYSIS_CONFIG.get('gpu_batch_size', 32))

    if use_gpu:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA"

        if tfr_method == 'morlet_adaptive':
            print(
                f"    Torch GPU adaptive Morlet enabled: {gpu_name}, "
                f"{adaptive_cycles_low}->{adaptive_cycles_high} cycles, "
                f"{adaptive_freq_low_max}-{adaptive_freq_high_min} Hz transition, "
                f"batch_size={gpu_batch_size}"
            )
        else:
            print(
                f"    Torch GPU fixed Morlet enabled: {gpu_name}, "
                f"n_cycles={n_cycles}, batch_size={gpu_batch_size}"
            )
    else:
        if tfr_method == 'morlet_adaptive':
            print(
                "    Adaptive Morlet enabled: "
                f"{adaptive_cycles_low}->{adaptive_cycles_high} cycles, "
                f"{adaptive_freq_low_max}-{adaptive_freq_high_min} Hz transition "
                "(CPU path)"
            )
        else:
            print("    Torch GPU Morlet not enabled/available; using CPU fixed Morlet.")

    region_specs_raw = {}

    if region_grids is None or elec_to_idx is None:
        print("    WARNING: region_grids or elec_to_idx missing; cannot compute array median raw spectrograms.")
        return region_specs_raw

    bb_full, t_ms = get_broadband_full(data)

    if bb_full is None:
        print("    WARNING: broadband data missing; cannot compute array median raw spectrograms.")
        return region_specs_raw

    bb_full = np.asarray(bb_full)

    if bb_full.ndim != 3:
        print(f"    WARNING: expected broadband data shape trial x channel x time, got {bb_full.shape}")
        return region_specs_raw

    n_trials, n_channels, n_time = bb_full.shape

    for region_name, grid_elec in region_grids.items():
        print(f"    Computing raw array-median spectrogram for {region_name}...")

        region_traces = []
        valid_channel_indices = []

        grid_elec_arr = np.asarray(grid_elec)

        for elec_id in grid_elec_arr.ravel():
            if elec_id is None:
                continue

            try:
                if np.isnan(elec_id):
                    continue
            except Exception:
                pass

            ch_idx = elec_to_idx.get(int(elec_id), None)

            if ch_idx is None:
                continue

            if ch_idx < 0 or ch_idx >= n_channels:
                continue

            valid_channel_indices.append(ch_idx)

            for trial_idx in range(n_trials):
                trace = bb_full[trial_idx, ch_idx, :]

                if trace is None:
                    continue

                trace = np.asarray(trace)

                if trace.size != n_time:
                    continue

                if not np.isfinite(trace).any():
                    continue

                region_traces.append(trace.astype(np.float32))

        n_valid_channels = len(set(valid_channel_indices))
        n_valid_trial_channel_pairs = len(region_traces)

        if n_valid_trial_channel_pairs == 0:
            print(f"      No valid trial/channel pairs for {region_name}; skipping.")
            continue

        traces_arr = np.stack(region_traces, axis=0).astype(np.float32)

        Sxx_region_median_raw = None
        f_out_saved = None
        t_spec_ms_saved = None

        if use_gpu:
            try:
                if tfr_method == 'morlet_adaptive':
                    Sxx_all, f_out, t_out = compute_wavelet_spectrogram_batch_torch_adaptive(
                        traces_arr,
                        fs=fs,
                        foi=foi,
                        n_cycles_low=adaptive_cycles_low,
                        n_cycles_high=adaptive_cycles_high,
                        freq_low_max=adaptive_freq_low_max,
                        freq_high_min=adaptive_freq_high_min,
                        batch_size=gpu_batch_size,
                        device="cuda",
                    )
                else:
                    Sxx_all, f_out, t_out = compute_wavelet_spectrogram_batch_torch(
                        traces_arr,
                        fs=fs,
                        foi=foi,
                        n_cycles=n_cycles,
                        batch_size=gpu_batch_size,
                        device="cuda",
                    )

                Sxx_region_median_raw = np.nanmedian(Sxx_all, axis=0)
                f_out_saved = f_out
                t_spec_ms_saved = np.asarray(t_out) * 1000.0 + float(t_ms[0])

                print(
                    f"      {region_name}: GPU {tfr_method} complete, "
                    f"{n_valid_channels} channels, "
                    f"{n_valid_trial_channel_pairs} trial/channel pairs"
                )

            except Exception as e:
                print(f"      WARNING: Torch GPU {tfr_method} failed for {region_name}: {e}")
                print(f"      Falling back to CPU {tfr_method} for this region.")
                Sxx_region_median_raw = None

                if torch is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

        if Sxx_region_median_raw is None:
            specs = []
            f_out_saved = None
            t_spec_ms_saved = None

            if tfr_method == 'morlet_adaptive':
                method_cfg = {
                    'method': 'morlet_adaptive',
                    'n_cycles_low': adaptive_cycles_low,
                    'n_cycles_high': adaptive_cycles_high,
                    'freq_low_max': adaptive_freq_low_max,
                    'freq_high_min': adaptive_freq_high_min,
                }
            else:
                method_cfg = {
                    'method': 'morlet',
                    'n_cycles': n_cycles,
                }

            for trace in region_traces:
                Sxx, f_out, t_out = compute_tfr_method(
                    trace,
                    fs,
                    method_cfg,
                    foi=foi,
                    t_ms=t_ms,
                )

                if Sxx is None:
                    continue

                specs.append(Sxx)

                if f_out_saved is None:
                    f_out_saved = f_out
                    t_spec_ms_saved = np.asarray(t_out) * 1000.0 + float(t_ms[0])

            if len(specs) == 0:
                print(f"      No valid CPU spectrograms for {region_name}; skipping.")
                continue

            stacked = np.stack(specs, axis=0)
            Sxx_region_median_raw = np.nanmedian(stacked, axis=0)

            print(
                f"      {region_name}: CPU {tfr_method} complete, "
                f"{n_valid_channels} channels, "
                f"{n_valid_trial_channel_pairs} trial/channel pairs"
            )

        region_specs_raw[region_name] = {
            'Sxx': Sxx_region_median_raw,
            'f': f_out_saved,
            't_ms': t_spec_ms_saved,
            'n_valid_channels': n_valid_channels,
            'n_valid_trial_channel_pairs': n_valid_trial_channel_pairs,
        }

    return region_specs_raw

def get_trial_count(data):
    return int(data['broadband_full'].shape[0])

def compute_control_median_baseline_norm(data, groups, fs=1000,
                                         region_grids=None, elec_to_idx=None):
    """
    Compute control-reference mean/std from the baseline of the control
    array-level median spectrogram.
    """
    region_specs_raw = compute_array_median_raw_spectrograms(
        data=data,
        groups=groups,
        fs=fs,
        region_grids=region_grids,
        elec_to_idx=elec_to_idx,
    )

    baseline_win = ANALYSIS_CONFIG['baseline_window']
    norm = {}

    for region_name, spec in region_specs_raw.items():
        Sxx_control_median_raw = spec['Sxx']
        t_spec_ms = spec['t_ms']

        bl_mask = (
            (t_spec_ms >= baseline_win[0]) &
            (t_spec_ms <= baseline_win[1])
        )

        if not bl_mask.any():
            print(f"      [Skip] No baseline samples for control norm: {region_name}")
            continue

        baseline_values = Sxx_control_median_raw[:, bl_mask]

        mean_val = np.nanmean(baseline_values, axis=1, keepdims=True)
        std_val = np.nanstd(baseline_values, axis=1, keepdims=True)
        std_val[std_val < 1e-10] = 1e-10

        norm[region_name] = {
            'mean': mean_val,
            'std': std_val,
            'foi': spec['f'],
            'baseline_window': baseline_win,
            'tfr_method': PLOT_CONFIG.get('spectrogram', {}).get('tfr_method', 'morlet'),
            'n_valid_channels': spec['n_valid_channels'],
            'n_valid_trial_channel_pairs': spec['n_valid_trial_channel_pairs'],
        }

    return norm

def compute_control_spatial_baseline(data, fs=1000):
    """
    Compute control baseline envelope for selected spatial band.
    Shape returned: (1, n_channels, 1)
    """
    bb_full, t_ms = get_broadband_full(data)

    band_name = PLOT_CONFIG.get('selected_band', 'Beta')
    f_lo, f_hi = ANALYSIS_CONFIG['bands'][band_name]
    baseline_win = ANALYSIS_CONFIG['baseline_window']

    nyq = fs / 2
    if f_hi >= nyq:
        f_hi = nyq - 1

    b, a = signal.butter(4, [f_lo / nyq, f_hi / nyq], btype='band')

    filtered = signal.filtfilt(b, a, bb_full, axis=2)
    envelope = np.abs(signal.hilbert(filtered, axis=2))

    bl_mask = (t_ms >= baseline_win[0]) & (t_ms <= baseline_win[1])

    baseline_power = np.nanmean(envelope[:, :, bl_mask], axis=(0, 2))
    baseline_power[baseline_power < 1e-10] = 1e-10

    return baseline_power[None, :, None]

def get_time_axis(data):
    """Extract time axis in ms from data dict."""
    for key in ['t_full_ms', 't_ms', 'rel_time_full']:
        if key in data:
            return np.array(data[key])
    
    # If no full time axis found, try to construct from pre+post
    if 'rel_time_pre' in data and 'rel_time_post' in data:
        t_pre = np.array(data['rel_time_pre'])
        t_post = np.array(data['rel_time_post'])
        return np.concatenate([t_pre, t_post])
    
    raise KeyError("No time axis found in data (tried: t_full_ms, t_ms, rel_time_full, rel_time_pre+rel_time_post)")

def get_broadband_full(data):
    """
    Get full broadband data array and corresponding time axis.
    
    Prefers 'broadband_full' if available, otherwise constructs from pre+post.
    
    Returns:
        (broadband_full, t_ms): Arrays of shape (n_trials, n_channels, n_times) and (n_times,)
    """
    if 'broadband_full' in data:
        t_ms = get_time_axis(data)
        return np.array(data['broadband_full']), t_ms
    
    # Construct from pre + post (legacy format)
    if 'broadband_pre' not in data or 'broadband_post' not in data:
        raise KeyError("Data must contain either 'broadband_full' or both 'broadband_pre' and 'broadband_post'")
    
    pre = np.array(data['broadband_pre'])
    post = np.array(data['broadband_post'])
    
    # Get time axes - check multiple possible key names
    t_pre = None
    t_post = None
    
    for key in ['rel_time_pre', 't_pre_ms']:
        if key in data:
            t_pre = np.array(data[key])
            break
    
    for key in ['rel_time_post', 't_post_ms']:
        if key in data:
            t_post = np.array(data[key])
            break
    
    if t_pre is None or t_post is None:
        raise KeyError("Missing time axes for reconstruction (tried: rel_time_pre/t_pre_ms and rel_time_post/t_post_ms)")
    
    # Concatenate
    broadband_full = np.concatenate([pre, post], axis=-1)
    t_ms = np.concatenate([t_pre, t_post])
    
    return broadband_full, t_ms

def slice_time_window(data_3d, t_ms, t_start, t_end):
    """
    Slice 3D data array to a specific time window.
    
    Args:
        data_3d: Array of shape (n_trials, n_channels, n_times)
        t_ms: Time axis in ms
        t_start: Start time in ms (inclusive)
        t_end: End time in ms (inclusive)
    
    Returns:
        (sliced_data, sliced_t_ms): Sliced arrays
    """
    mask = (t_ms >= t_start) & (t_ms <= t_end)
    return data_3d[..., mask], t_ms[mask]


# =============================================================================
# ALTERNATIVE TFR METHODS FOR WAVELET COMPARISON
# =============================================================================

def run_wavelet_comparison(lfp_dir, fig_dir, label=""):
    """
    Run a side-by-side comparison of TFR methods on one session/region.

    Controlled by WAVELET_TEST_CONFIG. Outputs a grid figure with one column
    per method. Each method gets two rows: raw spectrogram + time-frequency
    resolution indicator (wavelet duration as a function of frequency).
    Also plots the mean LFP trace above for reference.

    This is the primary tool for diagnosing temporal smearing from the
    stimulation low-frequency transient.
    """
    cfg = WAVELET_TEST_CONFIG
    if not cfg.get('enabled', False):
        return

    lfp_dir = Path(lfp_dir)
    fig_dir = Path(fig_dir)

    print(f"\n{'='*60}")
    print(f"WAVELET COMPARISON MODE ({label})")
    print(f"{'='*60}")

    # --- Find session file ---
    files = sorted(lfp_dir.rglob("aligned_lfp__*.npz"))
    if not files:
        print(f"  [Skip] No aligned LFP npz files found in {lfp_dir}")
        return

    # Apply the same session filtering logic used in process_directory(...)
    files_filtered = []

    for npz_candidate in files:
        session_candidate_id = npz_candidate.stem.replace("aligned_lfp__", "")

        # Same SESSION_ID_PATTERN logic as process_directory(...)
        if SESSION_ID_PATTERN is not None:
            if not re.search(SESSION_ID_PATTERN, session_candidate_id, re.IGNORECASE):
                continue

        # Same SESSION_FILTER logic as process_directory(...)
        if SESSION_FILTER is not None:
            br_match = re.search(r'_(\d{3})(?:_|$|\.)', session_candidate_id)
            if br_match:
                br_idx = int(br_match.group(1))
                if br_idx not in SESSION_FILTER:
                    continue

        files_filtered.append(npz_candidate)

    if not files_filtered:
        print(
            f"  [Skip] No aligned LFP npz files matched "
            f"SESSION_FILTER={SESSION_FILTER}, SESSION_ID_PATTERN={SESSION_ID_PATTERN}"
        )
        return

    # Prefer stim/non-control files after applying the same filter.
    non_control_files = [
        f for f in files_filtered
        if 'baseline' not in f.stem.lower()
        and 'control' not in f.stem.lower()
    ]

    npz_path = non_control_files[0] if non_control_files else files_filtered[0]

    selected_session_id = npz_path.stem.replace("aligned_lfp__", "")
    selected_br_match = re.search(r'_(\d{3})(?:_|$|\.)', selected_session_id)
    selected_br_idx = int(selected_br_match.group(1)) if selected_br_match else None

    print(
        f"  Applied standard session filters: "
        f"SESSION_FILTER={SESSION_FILTER}, SESSION_ID_PATTERN={SESSION_ID_PATTERN}"
    )
    print(f"  Selected BR={selected_br_idx}: {npz_path.name}")

    session_id = npz_path.stem.replace('aligned_lfp__', '')
    print(f"  Session: {session_id}")

    try:
        data = dict(np.load(npz_path, allow_pickle=True))
    except Exception as e:
        print(f"  [Error] Failed to load {npz_path}: {e}")
        return

    if 'broadband_full' not in data:
        print(f"  [Skip] No broadband_full in {npz_path.name}")
        return

    groups, nsp_to_elec, elec_to_idx, region_grids, ua_ids = build_groups(data, data['broadband_full'].shape[1])
    fs = int(data.get('fs_lfp', 1000))
    bb_full, t_ms = get_broadband_full(data)
    bb_full = np.asarray(bb_full)
    n_trials, n_channels, n_time = bb_full.shape
    t_ms_arr = np.asarray(t_ms)

    # --- Find region ---
    test_region = cfg.get('test_region')
    region_name = None

    if region_grids and test_region is not None:
        if test_region in region_grids:
            region_name = test_region
        else:
            print(f"  [Warn] test_region={test_region!r} not found. Available: {list(region_grids.keys())}")

    if region_name is None and region_grids:
        region_name = next(iter(region_grids))

    print(f"  Region: {region_name}")

    # --- Collect ALL traces for this region ---
    region_traces = []
    valid_channel_indices = []

    if region_grids and region_name and elec_to_idx:
        grid_elec = np.asarray(region_grids[region_name])

        for elec_id in grid_elec.ravel():
            if elec_id is None:
                continue

            try:
                if np.isnan(float(elec_id)):
                    continue
            except Exception:
                pass

            ch_idx = elec_to_idx.get(int(elec_id))

            if ch_idx is None:
                continue

            if ch_idx < 0 or ch_idx >= n_channels:
                continue

            valid_channel_indices.append(ch_idx)

            for trial_idx in range(n_trials):
                trace = bb_full[trial_idx, ch_idx, :]

                if trace is None:
                    continue

                trace = np.asarray(trace)

                if trace.size != n_time:
                    continue

                if not np.isfinite(trace).any():
                    continue

                region_traces.append(trace.astype(np.float64))

    else:
        # Fallback: use all channels from first available group
        if groups:
            grp_idxs = groups[0][0]

            for ch_idx in grp_idxs:
                if ch_idx is None:
                    continue

                if ch_idx < 0 or ch_idx >= n_channels:
                    continue

                valid_channel_indices.append(ch_idx)

                for trial_idx in range(n_trials):
                    trace = bb_full[trial_idx, ch_idx, :]

                    if trace is None:
                        continue

                    trace = np.asarray(trace)

                    if trace.size != n_time:
                        continue

                    if not np.isfinite(trace).any():
                        continue

                    region_traces.append(trace.astype(np.float64))

    if not region_traces:
        print(f"  [Skip] No valid traces for region {region_name}")
        return

    print(f"  Using all available data from {region_name}: {len(set(valid_channel_indices))} channels, {len(region_traces)} trial/channel pairs")

    # --- Build FOI ---
    f_min, f_max = cfg.get('freq_range', (1, 60))
    foi = get_foi_with_notch_gaps(f_min, f_max, 1)
    normalize_method = cfg.get('normalize', 'zscore')
    baseline_win = cfg.get('baseline_window', (-950, -650))
    time_range = cfg.get('time_range_ms', (-300, 500))
    methods = [m for m in cfg.get('methods', []) if m.get('enabled', True)]

    if not methods:
        print("  [Skip] No methods defined in WAVELET_TEST_CONFIG['methods']")
        return

    print(f"  Computing {len(methods)} methods on {len(region_traces)} traces...")

    # --- Compute TFR for each method, then median across traces ---
    method_results = []  # list of (label, Sxx_median, f, t_ms_plot)

    for method_cfg in methods:
        mlabel = method_cfg.get('label', method_cfg.get('method', '?'))
        print(f"    [{mlabel}] ... ", end='', flush=True)

        specs = []
        f_out_saved = None
        t_ms_plot = None

        for trace in region_traces:
            try:
                Sxx, f_out, t_out = compute_tfr_method(trace, fs, method_cfg, foi=foi, t_ms=t_ms_arr)
            except Exception as e:
                print(f"\n      [Error] {e}")
                continue
            if Sxx is None:
                continue
            specs.append(Sxx)
            if f_out_saved is None:
                f_out_saved = np.asarray(f_out)
                t_ms_plot = np.asarray(t_out) * 1000.0 + float(t_ms_arr[0])

        if not specs:
            print(f"FAILED")
            method_results.append((mlabel, None, None, None))
            continue

        Sxx_median = np.nanmedian(np.stack(specs, axis=0), axis=0)  # (n_freq, n_time)

        # Normalize
        bl_mask = (t_ms_plot >= baseline_win[0]) & (t_ms_plot <= baseline_win[1])
        if normalize_method == 'zscore':
            Sxx_norm = zscore_normalize_spectrogram(Sxx_median, baseline_mask=bl_mask)
        elif normalize_method == 'dB':
            if bl_mask.any():
                bl_power = np.nanmean(Sxx_median[:, bl_mask], axis=1, keepdims=True)
                bl_power[bl_power < 1e-20] = 1e-20
                Sxx_norm = 10 * np.log10(Sxx_median / bl_power)
            else:
                Sxx_norm = Sxx_median.copy()
        else:
            Sxx_norm = Sxx_median.copy()

        print(f"done ({len(specs)}/{len(region_traces)} valid)")
        method_results.append((mlabel, Sxx_norm, f_out_saved, t_ms_plot))

    # Filter out failed methods
    valid_results = [(lbl, sxx, f, t) for lbl, sxx, f, t in method_results if sxx is not None]
    if not valid_results:
        print("  [Skip] All methods failed.")
        return

    # --- Compute mean LFP trace for reference ---
    mean_lfp = np.nanmean(np.stack(region_traces, axis=0), axis=0)  # (n_time,)
    # Trim to display time range
    t_display_mask = (t_ms_arr >= time_range[0]) & (t_ms_arr <= time_range[1])
    t_display = t_ms_arr[t_display_mask]
    mean_lfp_display = mean_lfp[t_display_mask]

    # --- Build resolution indicator: temporal resolution at each frequency ---
    # For Morlet: sigma_t = n_cycles / (2*pi*f) -> FWHM ~ 2.35*sigma_t
    # This shows HOW MUCH temporal smearing each method has
    def get_temporal_resolution_ms(method_cfg, freqs):
        method = method_cfg.get('method', 'morlet')
        freqs = np.asarray(freqs)
        freqs = freqs[freqs > 0]
        if method == 'morlet':
            nc = method_cfg.get('n_cycles', 7)
            sigma_t = nc / (2 * np.pi * freqs)
            return freqs, sigma_t * 2.35 * 1000  # FWHM in ms
        elif method == 'morlet_adaptive':
            nc_lo = method_cfg.get('n_cycles_low', 3)
            nc_hi = method_cfg.get('n_cycles_high', 7)
            f_lo_max = method_cfg.get('freq_low_max', 8)
            f_hi_min = method_cfg.get('freq_high_min', 30)
            nc_arr = np.where(
                freqs <= f_lo_max, nc_lo,
                np.where(freqs >= f_hi_min, nc_hi,
                         nc_lo + (freqs - f_lo_max) / (f_hi_min - f_lo_max) * (nc_hi - nc_lo))
            )
            sigma_t = nc_arr / (2 * np.pi * freqs)
            return freqs, sigma_t * 2.35 * 1000
        elif method == 'stft':
            win_ms = method_cfg.get('window_ms', 200)
            return freqs, np.full(len(freqs), win_ms)
        elif method == 'multitaper':
            win_ms = method_cfg.get('window_ms', 500)
            return freqs, np.full(len(freqs), win_ms)
        elif method == 'paul':
            m = method_cfg.get('order', 4)
            omega_0 = 2 * (m + 1)
            scale = omega_0 / (2 * np.pi * freqs)
            # Paul wavelet effective duration approx: 2*sqrt(m+0.5)/(2*pi) / freq * 1000
            return freqs, (2 * np.sqrt(m + 0.5) / (2 * np.pi * freqs)) * 1000
        return freqs, np.full(len(freqs), np.nan)

    # --- Plot ---
    n_methods = len(valid_results)
    vmin = cfg.get('vmin', -3)
    vmax = cfg.get('vmax', 3)
    cmap = cfg.get('cmap', 'jet')

    # Figure layout: 3 rows per method column
    # Row 0: mean LFP trace (shared, top)
    # Row 1: spectrogram
    # Row 2: temporal resolution curve
    fig_height = 3 + 4 * n_methods  # Fallback if stacking rows

    # Better: use a grid with shared top row
    fig = plt.figure(figsize=(4 * n_methods, 9), dpi=120)
    gs = fig.add_gridspec(
        3, n_methods,
        height_ratios=[1, 4, 1.5],
        hspace=0.45,
        wspace=0.35,
    )

    fig.suptitle(
        f"Wavelet Method Comparison\n"
        f"{session_id} | {region_name} | "
        f"{len(region_traces)} trial/ch pairs | {normalize_method} norm",
        fontsize=11, fontweight='bold', y=0.98,
    )

    for col_idx, (mlabel, Sxx_norm, f_out, t_ms_plot) in enumerate(valid_results):
        # --- Row 0: mean LFP reference ---
        ax_lfp = fig.add_subplot(gs[0, col_idx])
        ax_lfp.plot(t_display, mean_lfp_display, color='#333333', lw=0.8)
        ax_lfp.axvline(0, color='red', ls='--', lw=1, alpha=0.8)
        ax_lfp.axvspan(0, 100, color='red', alpha=0.08)
        ax_lfp.set_xlim(time_range)
        ax_lfp.set_title(mlabel, fontsize=9, fontweight='bold')
        ax_lfp.set_xlabel('')
        ax_lfp.set_xticklabels([])
        if col_idx == 0:
            ax_lfp.set_ylabel('LFP (µV)', fontsize=7)
        ax_lfp.tick_params(labelsize=7)
        ax_lfp.grid(True, alpha=0.2)

        # --- Row 1: Spectrogram ---
        ax_spec = fig.add_subplot(gs[1, col_idx])

        # Trim to display time range
        t_disp_mask = (t_ms_plot >= time_range[0]) & (t_ms_plot <= time_range[1])
        t_disp = t_ms_plot[t_disp_mask]
        Sxx_disp = Sxx_norm[:, t_disp_mask]

        # Trim frequency
        f_mask = (f_out >= f_min) & (f_out <= f_max)
        f_disp = f_out[f_mask]
        Sxx_disp = Sxx_disp[f_mask, :]

        if t_disp.size > 0 and f_disp.size > 0:
            im = ax_spec.imshow(
                Sxx_disp,
                aspect='auto',
                origin='lower',
                extent=[t_disp[0], t_disp[-1], f_disp[0], f_disp[-1]],
                vmin=vmin, vmax=vmax,
                cmap=cmap,
                interpolation='nearest',
            )
            plt.colorbar(im, ax=ax_spec, shrink=0.8, pad=0.02, label=normalize_method)

        ax_spec.axvline(0, color='white', ls='--', lw=1, alpha=0.9)
        ax_spec.set_xlim(time_range)
        ax_spec.set_ylim(f_min, f_max)
        ax_spec.set_xlabel('Time re. stim onset (ms)', fontsize=7)
        if col_idx == 0:
            ax_spec.set_ylabel('Frequency (Hz)', fontsize=7)
        ax_spec.tick_params(labelsize=7)

        # Mark notch regions
        notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])
        for nlo, nhi in notch_regions:
            if nlo < f_max and nhi > f_min:
                ax_spec.axhspan(max(nlo, f_min), min(nhi, f_max),
                                color='gray', alpha=0.3, zorder=5)

        # --- Row 2: Temporal resolution ---
        ax_res = fig.add_subplot(gs[2, col_idx])
        method_cfg_entry = methods[col_idx] if col_idx < len(methods) else {}
        res_freqs, res_ms = get_temporal_resolution_ms(method_cfg_entry, f_disp)
        if res_freqs is not None and len(res_ms) > 0:
            ax_res.plot(res_ms, res_freqs, color='navy', lw=1.5)
            ax_res.set_xlabel('Temporal resolution\n(FWHM, ms)', fontsize=7)
            if col_idx == 0:
                ax_res.set_ylabel('Freq (Hz)', fontsize=7)
            ax_res.set_ylim(f_min, f_max)
            ax_res.grid(True, alpha=0.2)
            ax_res.tick_params(labelsize=7)
            # Shade region > 100ms (larger than typical stim window) in red
            ax_res.axvline(100, color='red', ls=':', lw=1, alpha=0.6, label='100 ms')
            ax_res.legend(fontsize=6, loc='upper right')

    # --- Save ---
    out_dir = fig_dir / cfg.get('output_subdir', 'wavelet_comparison')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"wavelet_comparison__{session_id}__{region_name}__{normalize_method}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved wavelet comparison -> {out_path}")

def compute_self_baseline_norm_array_median_spectrograms(data, groups, fs=1000,
                                                        region_grids=None,
                                                        elec_to_idx=None,
                                                        session_label="control"):
    raw_specs = compute_array_median_raw_spectrograms(
        data=data,
        groups=groups,
        fs=fs,
        region_grids=region_grids,
        elec_to_idx=elec_to_idx,
    )

    baseline_win = ANALYSIS_CONFIG.get('baseline_window', (-950, -650))
    spec_cfg = PLOT_CONFIG.get('spectrogram', {})
    normalize_method = spec_cfg.get('normalize', 'zscore')

    norm_specs = {}

    for region_name, spec_info in raw_specs.items():
        Sxx_raw = spec_info.get('Sxx', None)
        f_spec = spec_info.get('f', None)
        t_spec_ms = spec_info.get('t_ms', None)

        if Sxx_raw is None or f_spec is None or t_spec_ms is None:
            continue

        bl_mask = (
            (t_spec_ms >= baseline_win[0]) &
            (t_spec_ms <= baseline_win[1])
        )

        if not np.any(bl_mask):
            print(
                f"      WARNING: no baseline bins for {session_label} {region_name}; "
                f"baseline={baseline_win}, "
                f"t_range={t_spec_ms[0]:.1f} to {t_spec_ms[-1]:.1f} ms"
            )
            continue

        if normalize_method == 'zscore':
            Sxx_norm = zscore_normalize_spectrogram(
                Sxx_raw,
                baseline_mask=bl_mask,
            )

        elif normalize_method == 'dB':
            baseline_power = np.nanmean(
                Sxx_raw[:, bl_mask],
                axis=1,
                keepdims=True,
            )
            baseline_power[baseline_power < 1e-20] = 1e-20
            Sxx_norm = 10 * np.log10(Sxx_raw / baseline_power)

        else:
            # For this subtraction, zscore is strongly preferred.
            # But keep this fallback explicit.
            Sxx_norm = Sxx_raw.copy()

        norm_specs[region_name] = {
            'Sxx': Sxx_norm,
            'Sxx_raw': Sxx_raw,
            'f': f_spec,
            't_ms': t_spec_ms,
            'baseline_window': baseline_win,
            'tfr_method': spec_cfg.get('tfr_method', 'unknown'),
            'n_valid_channels': spec_info.get('n_valid_channels', None),
            'n_valid_trial_channel_pairs': spec_info.get('n_valid_trial_channel_pairs', None),
        }

    return norm_specs


# =============================================================================
# FIGURE 1: SESSION SUMMARY (Extended to 120 Hz)
# =============================================================================

def generate_session_summary(data, session_id, fig_dir, groups, fs=1000):
    """
    Generate comprehensive session summary with PSD from 1-120 Hz.
    """
    try:
        bb_full, t_ms = get_broadband_full(data)
    except KeyError as e:
        print(f"    [Skip] No broadband data for session summary: {e}")
        return None
    
    # Slice to analysis windows
    bb_pre, t_pre = slice_time_window(
        bb_full, t_ms,
        TIME_WINDOWS['pre_start'], TIME_WINDOWS['pre_end']
    )
    bb_post, t_post = slice_time_window(
        bb_full, t_ms,
        TIME_WINDOWS['post_start'], TIME_WINDOWS['post_end']
    )
    
    n_trials, n_ch, n_time = bb_post.shape
    n_regions = len(groups)
    
    # Create figure - single row of PSD plots
    fig, axes = plt.subplots(1, n_regions, figsize=(4 * n_regions, 4))
    if n_regions == 1:
        axes = [axes]
    
    fig.suptitle(f'Session Summary: {session_id}\n'
                 f'({n_trials} trials, {n_ch} channels, {n_regions} regions)',
                 fontsize=14, fontweight='bold')
    
    freq_range = ANALYSIS_CONFIG['freq_range_display']
    notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])
    nperseg = min(n_time, 512)
    
    # PSD for each region
    for col, (grp_idxs, grp_name) in enumerate(groups):
        ax = axes[col]
        short_name = grp_name.split(' (')[0]
        color = get_region_color(short_name)
        
        # Compute PSD
        region_data = bb_post[:, grp_idxs, :]
        f, Pxx = signal.welch(region_data, fs=fs, nperseg=nperseg, axis=2)
        
        # Average across channels, then compute stats across trials
        Pxx_ch_mean = np.nanmean(Pxx, axis=1)  # (trials, freq)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Pxx_dB = 10 * np.log10(Pxx_ch_mean + 1e-20)
        
        # Compute mean and 95% CI
        psd_mean, psd_ci_lo, psd_ci_hi = compute_confidence_interval(Pxx_dB, axis=0)
        
        # === MASK NOTCH REGIONS ===
        psd_mean = mask_notch_regions(f, psd_mean, notch_regions)
        psd_ci_lo = mask_notch_regions(f, psd_ci_lo, notch_regions)
        psd_ci_hi = mask_notch_regions(f, psd_ci_hi, notch_regions)
        
        # Mask to frequency range (1-120 Hz)
        f_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        f_plot = f[f_mask]
        
        # Plot (NaN values will create gaps automatically)
        ax.plot(f_plot, psd_mean[f_mask], color=color, lw=2, label='Post')
        ax.fill_between(f_plot,
                        psd_ci_lo[f_mask],
                        psd_ci_hi[f_mask],
                        color=color, alpha=0.2)
        
        # Add pre-stim if available
        if bb_pre is not None:
            pre_data = bb_pre[:, grp_idxs, :]
            if pre_data.shape[2] > 0:
                f_pre, Pxx_pre = signal.welch(pre_data, fs=fs, nperseg=min(pre_data.shape[2], 512), axis=2)
                Pxx_pre_ch_mean = np.nanmean(Pxx_pre, axis=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    Pxx_pre_dB = 10 * np.log10(Pxx_pre_ch_mean + 1e-20)
                
                # Compute mean and 95% CI for pre
                psd_pre_mean, psd_pre_ci_lo, psd_pre_ci_hi = compute_confidence_interval(Pxx_pre_dB, axis=0)
                
                # === MASK NOTCH REGIONS FOR PRE ===
                psd_pre_mean = mask_notch_regions(f_pre, psd_pre_mean, notch_regions)
                psd_pre_ci_lo = mask_notch_regions(f_pre, psd_pre_ci_lo, notch_regions)
                psd_pre_ci_hi = mask_notch_regions(f_pre, psd_pre_ci_hi, notch_regions)
                
                f_pre_mask = (f_pre >= freq_range[0]) & (f_pre <= freq_range[1])
                
                # Plot pre mean line
                ax.plot(f_pre[f_pre_mask], psd_pre_mean[f_pre_mask], 
                    color=color, lw=1.5, ls='--', alpha=0.7, label='Pre')
                
                # Add CI shading for pre (lighter/more transparent than post)
                ax.fill_between(f_pre[f_pre_mask],
                                psd_pre_ci_lo[f_pre_mask],
                                psd_pre_ci_hi[f_pre_mask],
                                color=color, alpha=0.1)
        
        add_band_shading(ax)
        
        # === MARK NOTCH REGIONS WITH SHADING ===
        for f_lo, f_hi in notch_regions:
            ax.axvspan(f_lo, f_hi, color='gray', alpha=0.15, zorder=0)
        
        # Add 60 Hz line marker
        ax.axvline(60, color='gray', ls=':', alpha=0.5, label='60 Hz')
        
        ax.set_title(f'{short_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (dB re: 1 µV²/Hz)' if col == 0 else '')
        ax.set_xlim(freq_range)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    
    # Save
    out_dir = fig_dir / "Session_Summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"summary_{session_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved session summary -> {out_path.name}")
    
    return out_path

    
# =============================================================================
# FIGURE 2: RAW WAVEFORM RASTER
# =============================================================================

def generate_waveform_raster(data, session_id, fig_dir, groups, fs=1000, 
                              n_trials_show=30):
    """
    Plot raw LFP waveforms for representative channels across trials.
    """
    try:
        bb_full, t_ms = get_broadband_full(data)
    except KeyError as e:
        print(f"    [Skip] No broadband data for waveform raster: {e}")
        return None
    
    n_trials, n_ch, n_time = bb_full.shape
    n_trials_plot = min(n_trials_show, n_trials)
    n_regions = len(groups)
    
    # Create figure
    fig, axes = plt.subplots(n_regions, 1, figsize=(14, 3 * n_regions),
                             sharex=True)
    if n_regions == 1:
        axes = [axes]
    
    fig.suptitle(f'Raw LFP Waveforms: {session_id}\n'
                 f'({n_trials_plot} trials, 1 representative channel per region)',
                 fontsize=12, fontweight='bold')
    
    for ax_idx, (grp_idxs, grp_name) in enumerate(groups):
        ax = axes[ax_idx]
        short_name = grp_name.split(' (')[0]
        color = get_region_color(short_name)
        
        # Pick middle channel from group
        ch_idx = grp_idxs[len(grp_idxs) // 2]
        
        # Get traces
        traces = bb_full[:n_trials_plot, ch_idx, :]
        
        # Normalize for display
        trace_std = np.nanstd(traces)
        if trace_std < 1e-10:
            trace_std = 1.0
        
        # Plot each trial with vertical offset
        for trial_idx in range(n_trials_plot):
            trace = traces[trial_idx, :]
            offset = trial_idx * 4 * trace_std
            ax.plot(t_ms, trace + offset, color=color, alpha=0.6, lw=0.5)
        
        # Mark stimulation period
        add_stim_marker(ax)
        
        ax.set_ylabel(f'{short_name}\n(µV + offset)', fontsize=10)
        ax.set_xlim(t_ms[0], t_ms[-1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add channel info
        ax.text(1.02, 0.5, f'Ch {ch_idx}', transform=ax.transAxes,
                fontsize=9, va='center', ha='left')
    
    axes[-1].set_xlabel('Time relative to stim onset (ms)', fontsize=11)
    
    plt.tight_layout()
    
    # Save
    out_dir = fig_dir / "Waveform_Rasters"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"waveforms_{session_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved waveform raster -> {out_path.name}")
    
    return out_path


# =============================================================================
# FIGURE 4: INTER-REGIONAL COHERENCE
# =============================================================================

def generate_coherence_matrix(data, session_id, fig_dir, groups, fs=1000):
    """
    Compute and plot coherence between all pairs of regions.
    """
    try:
        bb_full, t_ms = get_broadband_full(data)
    except KeyError as e:
        print(f"    [Skip] No data for coherence analysis: {e}")
        return None
    
    n_trials, n_ch, n_time = bb_full.shape
    n_regions = len(groups)
    region_names = [g[1].split(' (')[0] for g in groups]
    
    if n_regions < 2:
        print("    [Skip] Need at least 2 regions for coherence")
        return None
    
    # Compute mean trace per region per trial
    region_traces = {}
    for grp_idxs, grp_name in groups:
        short_name = grp_name.split(' (')[0]
        region_traces[short_name] = np.nanmean(bb_full[:, grp_idxs, :], axis=1)
    
    # Coherence settings
    nperseg = min(n_time, ANALYSIS_CONFIG['coherence_nperseg'])
    
    # Get frequency axis
    f, _ = signal.coherence(region_traces[region_names[0]][0],
                            region_traces[region_names[0]][0],
                            fs=fs, nperseg=nperseg)
    
    bands = ANALYSIS_CONFIG['bands']
    n_bands = len(bands)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    fig.suptitle(f'Inter-Regional Coherence: {session_id}\n'
                 f'(averaged across {n_trials} trials)',
                 fontsize=12, fontweight='bold')
    
    for band_idx, (band_name, (f_lo, f_hi)) in enumerate(bands.items()):
        if band_idx >= len(axes):
            break
        ax = axes[band_idx]
        
        # Coherence matrix for this band
        coh_matrix = np.zeros((n_regions, n_regions))
        coh_sem = np.zeros((n_regions, n_regions))
        
        f_mask = (f >= f_lo) & (f <= f_hi)
        
        for i, reg_i in enumerate(region_names):
            for j, reg_j in enumerate(region_names):
                if i == j:
                    coh_matrix[i, j] = 1.0
                    continue
                
                # Compute coherence for each trial
                coh_trials = []
                for trial in range(n_trials):
                    try:
                        _, Cxy = signal.coherence(
                            region_traces[reg_i][trial],
                            region_traces[reg_j][trial],
                            fs=fs, nperseg=nperseg
                        )
                        coh_band = np.nanmean(Cxy[f_mask])
                        if np.isfinite(coh_band):
                            coh_trials.append(coh_band)
                    except:
                        pass
                
                if coh_trials:
                    coh_matrix[i, j] = np.mean(coh_trials)
                    coh_sem[i, j] = np.std(coh_trials) / np.sqrt(len(coh_trials))
        
        # Plot heatmap
        im = ax.imshow(coh_matrix, cmap='hot', vmin=0, vmax=1, aspect='auto')
        
        # Add text annotations
        for i in range(n_regions):
            for j in range(n_regions):
                val = coh_matrix[i, j]
                text_color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=9, color=text_color, fontweight='bold')
        
        ax.set_xticks(range(n_regions))
        ax.set_xticklabels(region_names, fontsize=9)
        ax.set_yticks(range(n_regions))
        ax.set_yticklabels(region_names, fontsize=9)
        ax.set_title(f'{band_name} ({f_lo}-{f_hi} Hz)', fontsize=10, fontweight='bold')
    
    # Hide unused axes
    for idx in range(n_bands, len(axes)):
        axes[idx].axis('off')
    
    # Colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Coherence')
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    
    # Save
    out_dir = fig_dir / "Coherence"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"coherence_{session_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved coherence matrix -> {out_path.name}")
    
    return out_path


# =============================================================================
# FIGURE 6: STIM VS CONTROL COMPARISON
# =============================================================================

def generate_stim_vs_control(data_stim, data_control, session_id, fig_dir, 
                              groups, fs=1000):
    """
    Compare stimulation and control conditions.
    """
    try:
        stim_full, t_stim = get_broadband_full(data_stim)
        ctrl_full, t_ctrl = get_broadband_full(data_control)
    except (KeyError, TypeError) as e:
        print(f"    [Skip] Missing data for stim vs control: {e}")
        return None
    
    # Slice to post-stim window for comparison
    bb_stim, _ = slice_time_window(
        stim_full, t_stim,
        TIME_WINDOWS['post_start'], TIME_WINDOWS['post_end']
    )
    bb_ctrl, _ = slice_time_window(
        ctrl_full, t_ctrl,
        TIME_WINDOWS['post_start'], TIME_WINDOWS['post_end']
    )
    
    n_regions = len(groups)
    bands = ANALYSIS_CONFIG['bands']
    
    # Create figure: PSD comparison and band power difference
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, n_regions, figure=fig)
    
    fig.suptitle(f'Stim vs Control: {session_id}\n'
                 f'(Stim: {bb_stim.shape[0]} trials, Control: {bb_ctrl.shape[0]} trials)',
                 fontsize=12, fontweight='bold')
    
    nperseg = min(bb_stim.shape[2], bb_ctrl.shape[2], 512)
    freq_range = ANALYSIS_CONFIG['freq_range_display']

    notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])
    
    for col, (grp_idxs, grp_name) in enumerate(groups):
        short_name = grp_name.split(' (')[0]
        
        # Validate indices
        valid_stim_idx = [i for i in grp_idxs if i < bb_stim.shape[1]]
        valid_ctrl_idx = [i for i in grp_idxs if i < bb_ctrl.shape[1]]
        
        if not valid_stim_idx or not valid_ctrl_idx:
            continue
        
        # Top row: PSD comparison
        ax_psd = fig.add_subplot(gs[0, col])
        
        # Stim PSD
        stim_data = bb_stim[:, valid_stim_idx, :]
        f, Pxx_stim = signal.welch(stim_data, fs=fs, nperseg=nperseg, axis=2)
        Pxx_stim_mean = np.nanmean(Pxx_stim, axis=(0, 1))
        
        # Control PSD
        ctrl_data = bb_ctrl[:, valid_ctrl_idx, :]
        _, Pxx_ctrl = signal.welch(ctrl_data, fs=fs, nperseg=nperseg, axis=2)
        Pxx_ctrl_mean = np.nanmean(Pxx_ctrl, axis=(0, 1))
        
        # Convert to dB
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psd_stim_dB = 10 * np.log10(Pxx_stim_mean + 1e-20)
            psd_ctrl_dB = 10 * np.log10(Pxx_ctrl_mean + 1e-20)

        psd_stim_dB = mask_notch_regions(f, psd_stim_dB, notch_regions)
        psd_ctrl_dB = mask_notch_regions(f, psd_ctrl_dB, notch_regions)
        
        f_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        
        ax_psd.plot(f[f_mask], psd_stim_dB[f_mask], 'r-', lw=2, label='Stim')
        ax_psd.plot(f[f_mask], psd_ctrl_dB[f_mask], 'b-', lw=2, label='Control')

        for f_lo, f_hi in notch_regions:
            ax_psd.axvspan(f_lo, f_hi, color='gray', alpha=0.15, zorder=0)
        
        add_band_shading(ax_psd)
        ax_psd.axvline(60, color='gray', ls=':', alpha=0.5)
        
        ax_psd.set_title(f'{short_name}', fontsize=11, fontweight='bold')
        ax_psd.set_xlabel('Frequency (Hz)')
        ax_psd.set_ylabel('Power (dB)' if col == 0 else '')
        ax_psd.set_xlim(freq_range)
        ax_psd.legend(loc='upper right', fontsize='small')
        ax_psd.grid(True, alpha=0.3)
        
        # Bottom row: Power difference by band
        ax_diff = fig.add_subplot(gs[1, col])
        
        band_names = list(bands.keys())
        x_pos = np.arange(len(band_names))
        
        diff_means = []
        diff_cis = []
        
        for band_name, (f_lo, f_hi) in bands.items():
            f_band_mask = (f >= f_lo) & (f <= f_hi)
            
            # Stim band power (per trial)
            stim_band = np.nanmean(Pxx_stim[:, :, f_band_mask], axis=(1, 2))
            ctrl_band = np.nanmean(Pxx_ctrl[:, :, f_band_mask], axis=(1, 2))
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stim_dB = 10 * np.log10(stim_band + 1e-20)
                ctrl_dB = 10 * np.log10(ctrl_band + 1e-20)
            
            # Difference (positive = stim > control)
            diff = np.mean(stim_dB) - np.mean(ctrl_dB)
            
            # Pooled SEM for difference
            sem_diff = np.sqrt(
                (np.std(stim_dB)**2 / len(stim_dB)) + 
                (np.std(ctrl_dB)**2 / len(ctrl_dB))
            ) * 1.96
            
            diff_means.append(diff)
            diff_cis.append(sem_diff)
        
        colors = ['red' if d > 0 else 'blue' for d in diff_means]
        ax_diff.bar(x_pos, diff_means, yerr=diff_cis, color=colors, alpha=0.7, 
                   capsize=4, edgecolor='black')
        ax_diff.axhline(0, color='black', ls='-', lw=1)
        
        ax_diff.set_xticks(x_pos)
        ax_diff.set_xticklabels(band_names, fontsize=8, rotation=45, ha='right')
        ax_diff.set_ylabel('Δ Power (dB)\n(Stim - Control)' if col == 0 else '')
        ax_diff.set_title('Power Difference', fontsize=10)
        ax_diff.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    out_dir = fig_dir / "Stim_vs_Control"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stim_vs_control_{session_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved stim vs control -> {out_path.name}")
    
    return out_path


def generate_vs_rest(data_a, data_rest, session_id, fig_dir, groups, fs=1000,
                     label_a="Stim", out_subdir="vs_Rest"):
    """
    Compare any condition (stim or control reaches) against at-rest LFP.

    Reuses the same PSD + band-power-difference layout as generate_stim_vs_control
    but with configurable axis labels and a separate output subdirectory so the
    three comparison types (stim-vs-control, stim-vs-rest, control-vs-rest) land
    in distinct folders.

    Parameters
    ----------
    data_a : dict
        LFP npz data for the primary condition (stim or control reaches).
    data_rest : dict
        LFP npz data for the at-rest baseline.
    session_id : str
        Identifier used in the figure title and filename.
    fig_dir : Path
        Root figure output directory.
    groups : list
        Channel-group list from build_groups().
    fs : int
        Sampling frequency in Hz (default 1000).
    label_a : str
        Human-readable label for the primary condition (default "Stim").
    out_subdir : str
        Subdirectory inside fig_dir to save figures (default "vs_Rest").
    """
    try:
        a_full, t_a = get_broadband_full(data_a)
        rest_full, t_rest = get_broadband_full(data_rest)
    except (KeyError, TypeError) as e:
        print(f"    [Skip] Missing data for {label_a} vs Rest: {e}")
        return None

    # Slice to post-stim window
    bb_a, _ = slice_time_window(a_full, t_a,
                                TIME_WINDOWS['post_start'], TIME_WINDOWS['post_end'])
    bb_rest, _ = slice_time_window(rest_full, t_rest,
                                   TIME_WINDOWS['post_start'], TIME_WINDOWS['post_end'])

    n_regions = len(groups)
    bands = ANALYSIS_CONFIG['bands']

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, n_regions, figure=fig)

    fig.suptitle(
        f'{label_a} vs At-Rest: {session_id}\n'
        f'({label_a}: {bb_a.shape[0]} trials, Rest: {bb_rest.shape[0]} trials)',
        fontsize=12, fontweight='bold'
    )

    nperseg = min(bb_a.shape[2], bb_rest.shape[2], 512)
    freq_range = ANALYSIS_CONFIG['freq_range_display']
    notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])

    for col, (grp_idxs, grp_name) in enumerate(groups):
        short_name = grp_name.split(' (')[0]

        valid_a_idx    = [i for i in grp_idxs if i < bb_a.shape[1]]
        valid_rest_idx = [i for i in grp_idxs if i < bb_rest.shape[1]]

        if not valid_a_idx or not valid_rest_idx:
            continue

        # --- Top row: PSD comparison ---
        ax_psd = fig.add_subplot(gs[0, col])

        a_data = bb_a[:, valid_a_idx, :]
        f, Pxx_a = signal.welch(a_data, fs=fs, nperseg=nperseg, axis=2)
        Pxx_a_mean = np.nanmean(Pxx_a, axis=(0, 1))

        rest_data = bb_rest[:, valid_rest_idx, :]
        _, Pxx_rest = signal.welch(rest_data, fs=fs, nperseg=nperseg, axis=2)
        Pxx_rest_mean = np.nanmean(Pxx_rest, axis=(0, 1))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psd_a_dB    = 10 * np.log10(Pxx_a_mean    + 1e-20)
            psd_rest_dB = 10 * np.log10(Pxx_rest_mean + 1e-20)

        psd_a_dB    = mask_notch_regions(f, psd_a_dB,    notch_regions)
        psd_rest_dB = mask_notch_regions(f, psd_rest_dB, notch_regions)

        f_mask = (f >= freq_range[0]) & (f <= freq_range[1])

        ax_psd.plot(f[f_mask], psd_a_dB[f_mask],    'r-', lw=2, label=label_a)
        ax_psd.plot(f[f_mask], psd_rest_dB[f_mask], 'k-', lw=2, label='Rest', alpha=0.7)

        for f_lo, f_hi in notch_regions:
            ax_psd.axvspan(f_lo, f_hi, color='gray', alpha=0.15, zorder=0)

        add_band_shading(ax_psd)
        ax_psd.set_title(f'{short_name}', fontsize=11, fontweight='bold')
        ax_psd.set_xlabel('Frequency (Hz)')
        ax_psd.set_ylabel('Power (dB)' if col == 0 else '')
        ax_psd.set_xlim(freq_range)
        ax_psd.legend(loc='upper right', fontsize='small')
        ax_psd.grid(True, alpha=0.3)

        # --- Bottom row: Band-power difference ---
        ax_diff = fig.add_subplot(gs[1, col])

        band_names = list(bands.keys())
        x_pos = np.arange(len(band_names))
        diff_means, diff_cis = [], []

        for band_name, (f_lo, f_hi) in bands.items():
            f_band_mask = (f >= f_lo) & (f <= f_hi)

            a_band    = np.nanmean(Pxx_a[:, :, f_band_mask],    axis=(1, 2))
            rest_band = np.nanmean(Pxx_rest[:, :, f_band_mask], axis=(1, 2))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a_dB    = 10 * np.log10(a_band    + 1e-20)
                rest_dB = 10 * np.log10(rest_band + 1e-20)

            diff = np.mean(a_dB) - np.mean(rest_dB)
            sem_diff = np.sqrt(
                (np.std(a_dB)**2    / max(len(a_dB),    1)) +
                (np.std(rest_dB)**2 / max(len(rest_dB), 1))
            ) * 1.96

            diff_means.append(diff)
            diff_cis.append(sem_diff)

        colors = ['red' if d > 0 else 'blue' for d in diff_means]
        ax_diff.bar(x_pos, diff_means, yerr=diff_cis, color=colors, alpha=0.7,
                    capsize=4, edgecolor='black')
        ax_diff.axhline(0, color='black', ls='-', lw=1)
        ax_diff.set_xticks(x_pos)
        ax_diff.set_xticklabels(band_names, fontsize=8, rotation=45, ha='right')
        ax_diff.set_ylabel(f'Δ Power (dB)\n({label_a} - Rest)' if col == 0 else '')
        ax_diff.set_title('Power Difference', fontsize=10)
        ax_diff.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    out_dir = fig_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_subdir.lower()}_{session_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved {label_a} vs Rest -> {out_path.name}")

    return out_path


# =============================================================================
# FIGURE 7: DOSE-RESPONSE ANALYSIS
# =============================================================================

def generate_dose_response(session_data_dict, base_session_id, fig_dir, groups, fs=1000):
    """
    Plot neural response as a function of stimulation dose (number of pulses).
    """
    if not session_data_dict or len(session_data_dict) < 2:
        print("    [Skip] Need multiple conditions for dose-response")
        return None
    
    pulse_counts = sorted(session_data_dict.keys())
    bands = ANALYSIS_CONFIG['bands']
    
    # Create figure
    n_bands = len(bands)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    fig.suptitle(f'Dose-Response Analysis: {base_session_id}\n'
                 f'Band power change vs number of stimulation pulses',
                 fontsize=12, fontweight='bold')
    
    # Get baseline (0 pulses / control) for reference
    baseline_powers = {}
    if 0 in session_data_dict:
        ctrl_data = session_data_dict[0].get('broadband_post')
        if ctrl_data is not None:
            nperseg = min(ctrl_data.shape[2], 256)
            f, Pxx_ctrl = signal.welch(ctrl_data, fs=fs, nperseg=nperseg, axis=2)
            
            for grp_idxs, grp_name in groups:
                short_name = grp_name.split(' (')[0]
                valid_idx = [i for i in grp_idxs if i < ctrl_data.shape[1]]
                if valid_idx:
                    baseline_powers[short_name] = np.nanmean(Pxx_ctrl[:, valid_idx, :], axis=(0, 1))
    
    for band_idx, (band_name, (f_lo, f_hi)) in enumerate(bands.items()):
        if band_idx >= len(axes):
            break
        ax = axes[band_idx]
        
        for grp_idxs, grp_name in groups:
            short_name = grp_name.split(' (')[0]
            color = get_region_color(short_name)
            
            power_means = []
            power_cis = []
            valid_pulses = []
            
            for n_pulses in pulse_counts:
                data = session_data_dict[n_pulses]
                
                try:
                    bb_full, t_ms = get_broadband_full(data)
                    bb_post, _ = slice_time_window(
                        bb_full, t_ms,
                        TIME_WINDOWS['post_start'], TIME_WINDOWS['post_end']
                    )
                except KeyError:
                    continue
                
                valid_idx = [i for i in grp_idxs if i < bb_post.shape[1]]
                if not valid_idx:
                    continue
                
                # Compute band power
                nperseg = min(bb_post.shape[2], 256)
                f, Pxx = signal.welch(bb_post[:, valid_idx, :], fs=fs, nperseg=nperseg, axis=2)
                
                f_mask = (f >= f_lo) & (f <= f_hi)
                band_power = np.nanmean(Pxx[:, :, f_mask], axis=(1, 2))  # (trials,)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    band_dB = 10 * np.log10(band_power + 1e-20)
                
                # Express relative to baseline if available
                if short_name in baseline_powers:
                    baseline_band = np.nanmean(baseline_powers[short_name][f_mask])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        baseline_dB = 10 * np.log10(baseline_band + 1e-20)
                    band_dB = band_dB - baseline_dB
                
                mean, ci_lo, ci_hi = compute_confidence_interval(band_dB, axis=0)
                
                power_means.append(mean)
                power_cis.append((ci_hi - ci_lo) / 2)
                valid_pulses.append(n_pulses)
            
            if valid_pulses:
                ax.errorbar(valid_pulses, power_means, yerr=power_cis,
                           marker='o', color=color, label=short_name,
                           capsize=3, lw=2, markersize=6)
        
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.set_xlabel('Number of Pulses' if band_idx >= 3 else '')
        ax.set_ylabel('Δ Power (dB)' if band_idx % 3 == 0 else '')
        ax.set_title(f'{band_name} ({f_lo}-{f_hi} Hz)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if band_idx == 0:
            ax.legend(loc='best', fontsize='small')
        
        if valid_pulses:
            ax.set_xticks(pulse_counts)
    
    # Hide unused axes
    for idx in range(n_bands, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save
    out_dir = fig_dir / "Dose_Response"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dose_response_{base_session_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved dose-response -> {out_path.name}")
    
    return out_path


# =============================================================================
# FIGURE 8: SPATIAL GIF
# =============================================================================

def render_spatial_grid(ax, mean_band_dB, t_bin_idx, grid_elec, elec_to_idx, 
                        vmin, vmax, smoothing=True):
    """
    Renders a spatial grid onto an axis, optionally with smoothing.
    """
    grid = np.full((8, 8), np.nan)
    points = []
    values = []
    
    for r in range(8):
        for c in range(8):
            elec_id = int(grid_elec[r, c])
            idx = elec_to_idx.get(elec_id, None)
            if idx is not None and 0 <= idx < mean_band_dB.shape[0]:
                val = mean_band_dB[idx, t_bin_idx]
                grid[r, c] = val
                if not np.isnan(val):
                    points.append((r, c))
                    values.append(val)
    
    if smoothing and len(points) >= 4:
        # Diffusion-based smoothing
        filled_grid = np.copy(grid)
        mask = np.isnan(filled_grid)
        
        if len(values) > 0:
            filled_grid[mask] = np.nanmean(np.asarray(values))
        
        # Iterative Laplace solver
        for _ in range(10):
            prev_grid = np.copy(filled_grid)
            for r in range(8):
                for c in range(8):
                    if mask[r, c]:
                        neighbors = []
                        if r > 0: neighbors.append(prev_grid[r-1, c])
                        if r < 7: neighbors.append(prev_grid[r+1, c])
                        if c > 0: neighbors.append(prev_grid[r, c-1])
                        if c < 7: neighbors.append(prev_grid[r, c+1])
                        if neighbors:
                            filled_grid[r, c] = np.nanmean(np.asarray(neighbors))
        
        im = ax.imshow(filled_grid, cmap='jet', vmin=vmin, vmax=vmax,
                       origin='upper', aspect='equal', interpolation='bicubic')
    else:
        im = ax.imshow(grid, cmap='jet', vmin=vmin, vmax=vmax,
                       origin='upper', aspect='equal')
    
    return im


def generate_spatial_gif(band_dB, t_ms, ua_ids_1based, session_id, fig_dir, 
                         groups, nsp_to_elec, elec_to_idx, region_grids):
    """
    Generate animated GIF of the spatial heatmap over time.
    """
    gif_cfg = ANALYSIS_CONFIG
    t_start, t_end = gif_cfg.get('gif_time_range', (-500, 500))
    t_step = gif_cfg.get('gif_step_ms', 10)
    vmin = PLOT_CONFIG['heatmap_vmin']
    vmax = PLOT_CONFIG['heatmap_vmax']
    
    n_trials = band_dB.shape[0]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_band_dB = np.nanmean(band_dB, axis=0)  # (n_ch, n_time)
    
    # Define time points for GIF frames
    gif_time_pts = np.arange(t_start, t_end + t_step, t_step)
    
    heatmap_dir = fig_dir / "Spatial_GIF"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    
    band_name = PLOT_CONFIG.get('selected_band', 'Beta')
    
    for grp_idxs, grp_name in groups:
        region_name = grp_name.split(' (')[0]
        grid_elec = region_grids.get(region_name)
        
        if grid_elec is None or len(grp_idxs) == 0:
            continue
        
        safe_region = region_name.replace(' ', '_')
        
        # Generate both blocky and smoothed versions
        for mode_label, use_smoothing in [('blocky', False), ('smoothed', True)]:
            frames = []
            print(f"      Generating GIF ({mode_label}) for {region_name}...")
            
            fig, ax = plt.subplots(figsize=(6, 5))
            
            for t_target in gif_time_pts:
                t_bin_idx = int(np.argmin(np.abs(t_ms - t_target)))
                t_actual = float(t_ms[t_bin_idx])
                
                if t_actual < t_ms[0] or t_actual > t_ms[-1]:
                    continue
                
                ax.clear()
                im = render_spatial_grid(ax, mean_band_dB, t_bin_idx, grid_elec, 
                                        elec_to_idx, vmin, vmax, smoothing=use_smoothing)
                
                ax.set_title(f"{region_name} | {band_name} ({mode_label.capitalize()})\n"
                            f"T = {t_actual:.0f} ms", fontsize=12)
                ax.set_xticks(np.arange(8))
                ax.set_yticks(np.arange(8))
                ax.set_xticklabels(np.arange(1, 9), fontsize=8)
                ax.set_yticklabels(np.arange(1, 9), fontsize=8)
                ax.set_ylabel("Row", fontsize=10)
                ax.set_xlabel("Col", fontsize=10)
                
                # Add colorbar only once
                if not hasattr(fig, '_colorbar_added'):
                    fig.colorbar(im, ax=ax, label="dB")
                    fig._colorbar_added = True
                
                # Convert to image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=80)
                buf.seek(0)
                frames.append(imageio.v3.imread(buf))
            
            plt.close(fig)
            
            if frames:
                out_path = heatmap_dir / f"spatial_{band_name}_{session_id}_{safe_region}_{mode_label}.gif"
                imageio.mimsave(out_path, frames, fps=gif_cfg.get('gif_fps', 10))
                print(f"      Saved spatial GIF -> {out_path.name}")


def generate_spatial_heatmaps(band_dB, t_ms, ua_ids_1based, session_id, fig_dir,
                               groups, elec_to_idx, region_grids):
    """
    Generate static spatial heatmaps at key timepoints.
    """
    time_pts = PLOT_CONFIG.get('spatial_timepoints_ms', [-200, -50, 0, 50, 100, 200, 300, 500])
    vmin = PLOT_CONFIG['heatmap_vmin']
    vmax = PLOT_CONFIG['heatmap_vmax']
    band_name = PLOT_CONFIG.get('selected_band', 'Beta')
    
    n_trials = band_dB.shape[0]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_band_dB = np.nanmean(band_dB, axis=0)
    
    heatmap_dir = fig_dir / "Spatial_Heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    
    for grp_idxs, grp_name in groups:
        region_name = grp_name.split(' (')[0]
        grid_elec = region_grids.get(region_name)
        
        if grid_elec is None or len(grp_idxs) == 0:
            continue
        
        safe_region = region_name.replace(' ', '_')
        
        for mode_label, use_smoothing in [('blocky', False), ('smoothed', True)]:
            fig, axes = plt.subplots(1, len(time_pts), figsize=(3.5 * len(time_pts), 4))
            if len(time_pts) == 1:
                axes = [axes]
            
            fig.suptitle(f"{session_id} — {region_name} ({mode_label.capitalize()})\n"
                        f"{band_name} Spatial Heatmap — Mean of {n_trials} Trials (dB)",
                        fontsize=14)
            
            im = None
            for t_idx, t_target in enumerate(time_pts):
                ax = axes[t_idx]
                t_bin_idx = int(np.argmin(np.abs(t_ms - t_target)))
                t_actual = float(t_ms[t_bin_idx])
                
                im = render_spatial_grid(ax, mean_band_dB, t_bin_idx, grid_elec,
                                        elec_to_idx, vmin, vmax, smoothing=use_smoothing)
                
                ax.set_title(f"T = {t_actual:.0f} ms", fontsize=11)
                ax.set_xticks(np.arange(8))
                ax.set_yticks(np.arange(8))
                ax.set_xticklabels(np.arange(1, 9), fontsize=7)
                ax.set_yticklabels(np.arange(1, 9), fontsize=7)
                if t_idx == 0:
                    ax.set_ylabel("Row", fontsize=9)
                ax.set_xlabel("Col", fontsize=8)
            
            fig.subplots_adjust(right=0.92, top=0.85, bottom=0.15, left=0.05, wspace=0.3)
            if im is not None:
                cbar_ax = fig.add_axes([0.94, 0.25, 0.01, 0.5])
                fig.colorbar(im, cax=cbar_ax, label="dB")
            
            out_path = heatmap_dir / f"spatial_{band_name}_{session_id}_{safe_region}_{mode_label}.png"
            plt.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"      Saved spatial heatmap -> {out_path.name}")


# =============================================================================
# TRIAL HEATMAPS (Per-trial band power visualization)
# =============================================================================

def generate_trial_heatmaps(data, session_id, fig_dir, groups, fs=1000):
    """
    Generate per-trial band power heatmaps for each region.
    """
    try:
        bb_full, t_ms = get_broadband_full(data)
    except KeyError as e:
        print(f"    [Skip] No data for trial heatmaps: {e}")
        return None
    
    n_trials, n_ch, n_time = bb_full.shape
    
    band_name = PLOT_CONFIG.get('selected_band', 'Beta')
    bands = ANALYSIS_CONFIG['bands']
    
    if band_name not in bands:
        band_name = 'Beta'
    
    f_lo, f_hi = bands[band_name]
    baseline_win = ANALYSIS_CONFIG['baseline_window']
    
    # Filter to band
    nyq = fs / 2
    if f_hi >= nyq:
        f_hi = nyq - 1
    
    try:
        b, a = signal.butter(4, [f_lo/nyq, f_hi/nyq], btype='band')
    except:
        print(f"    [Skip] Could not create filter for {band_name}")
        return None
    
    vmin = PLOT_CONFIG['heatmap_vmin']
    vmax = PLOT_CONFIG['heatmap_vmax']
    
    n_regions = len(groups)
    fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 8))
    if n_regions == 1:
        axes = [axes]
    
    fig.suptitle(f'{band_name} Band Power by Trial: {session_id}\n'
                 f'({f_lo}-{f_hi} Hz, trials sorted by mean post-stim power)',
                 fontsize=12, fontweight='bold')
    
    bl_mask = (t_ms >= baseline_win[0]) & (t_ms <= baseline_win[1])
    
    for col, (grp_idxs, grp_name) in enumerate(groups):
        ax = axes[col]
        short_name = grp_name.split(' (')[0]
        
        # Average across channels
        region_data = np.nanmean(bb_full[:, grp_idxs, :], axis=1)
        
        # Filter and compute envelope
        filtered = signal.filtfilt(b, a, region_data, axis=1)
        envelope = np.abs(signal.hilbert(filtered, axis=1))
        
        # Convert to dB
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            baseline_power = np.nanmean(envelope[:, bl_mask], axis=1, keepdims=True)
            baseline_power[baseline_power < 1e-10] = 1e-10
            envelope_dB = 10 * np.log10(envelope / baseline_power)
        
        # Sort trials by mean post-stim power
        post_mask = (t_ms >= 100) & (t_ms <= 500)
        mean_post_power = np.nanmean(envelope_dB[:, post_mask], axis=1)
        sort_idx = np.argsort(mean_post_power)[::-1]
        
        envelope_sorted = envelope_dB[sort_idx, :]
        
        # Plot
        im = ax.imshow(envelope_sorted, aspect='auto', cmap='jet',
                      vmin=vmin, vmax=vmax,
                      extent=[t_ms[0], t_ms[-1], n_trials, 0])
        
        ax.axvline(0, color='white', ls='--', lw=1)
        ax.axvline(100, color='white', ls=':', lw=0.5)
        
        ax.set_title(f'{short_name}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Trial (sorted)' if col == 0 else '')
        
        if col == n_regions - 1:
            cbar = fig.colorbar(im, ax=ax, label='Power (dB)')
    
    plt.tight_layout()
    
    # Save
    out_dir = fig_dir / "Trial_Heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trial_heatmap_{band_name}_{session_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved trial heatmaps -> {out_path.name}")
    
    return out_path


# =============================================================================
# PER-UA SPECTROGRAM FUNCTION (using Morlet wavelets)
# =============================================================================

def generate_per_ua_spectrograms(data, session_id, fig_dir, groups, fs=1000,
                                  ua_ids_1based=None, nsp_to_elec=None, 
                                  region_grids=None, elec_to_idx=None,
                                  trial_index=0, generate_all_trials=False,
                                  generate_trial_average=False):
    """
    Generate 8x8 spectrogram grids for each Utah Array region.
    Frequencies in notch regions are skipped entirely (broken y-axis).
    
    NOW USES:
    - Z-score normalization per channel
    - Median averaging across trials (instead of mean)
    """
    try:
        bb_full, t_ms = get_broadband_full(data)
    except KeyError as e:
        print(f"    [Skip] No broadband data for per-UA spectrograms: {e}")
        return None
    
    if region_grids is None or elec_to_idx is None:
        print("    [Skip] Need electrode mapping for per-UA spectrograms")
        return None
    
    n_trials, n_ch, n_time = bb_full.shape

    # Spectrogram settings
    spec_cfg = PLOT_CONFIG.get('spectrogram', {})
    freq_min = spec_cfg.get('freq_min', 1)
    freq_max = spec_cfg.get('freq_max', 120)
    freq_step = spec_cfg.get('freq_step', 1)
    
    # Get frequencies with notch gaps
    foi = get_foi_with_notch_gaps(freq_min, freq_max, freq_step)
    
    cmap = spec_cfg.get('cmap', 'viridis')
    normalize_method = spec_cfg.get('normalize', 'zscore')  # NEW: default to zscore
    n_cycles = spec_cfg.get('wavelet_cycles', 5)
    
    # Baseline window for z-score normalization
    baseline_win = ANALYSIS_CONFIG['baseline_window']
    
    # Output directory
    out_dir = fig_dir / "per_UA_activity"
    
    # Extract condition name
    br_match = re.search(r'_(\d{3})(?:_|$|\.)', session_id)
    target_match = re.search(r'target_([AB])$', session_id, re.IGNORECASE)
    target_suffix = f"_{target_match.group(1).upper()}" if target_match else ""

    if br_match:
        br_idx = int(br_match.group(1))
        condition_name = f"condition_{br_idx:02d}{target_suffix}"
    else:
        br_idx = data.get('br_idx', None)
        if br_idx is not None:
            if isinstance(br_idx, np.ndarray):
                br_idx = int(br_idx.item())
            condition_name = f"condition_{br_idx:02d}{target_suffix}"
        else:
            condition_name = session_id

    condition_dir = out_dir / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which trials to process
    trials_to_process = []
    
    if generate_trial_average:
        trials_to_process.append(('average', None))  # Special marker for averaging
    
    if generate_all_trials:
        trials_to_process.extend([('single', t) for t in range(n_trials)])
    elif not generate_trial_average:  # Only add default trial if not doing average-only
        if trial_index >= n_trials:
            print(f"    [Warn] Requested trial {trial_index} but only {n_trials} trials exist. Using trial 0.")
            trial_index = 0
        trials_to_process.append(('single', trial_index))
    
    # If both average and single trials requested, add default trial
    if generate_trial_average and not generate_all_trials:
        if trial_index >= n_trials:
            trial_index = 0
        trials_to_process.append(('single', trial_index))
    
    for grp_idxs, grp_name in groups:
        region_name = grp_name.split(' (')[0]
        grid_elec = region_grids.get(region_name)
        
        if grid_elec is None:
            print(f"    [Skip] No grid for region {region_name}")
            continue
        
        safe_region = region_name.replace(' ', '_')
        
        for trial_type, trial_idx in trials_to_process:
            
            if trial_type == 'average':
                # === TRIAL-AVERAGED SPECTROGRAM (using MEDIAN) ===
                print(f"      Computing trial-averaged (median) spectrogram for {region_name}...")
                
                # First pass: compute spectrograms for all trials, then MEDIAN average
                all_spectrograms = {}
                
                for row in range(8):
                    for col in range(8):
                        elec_id = int(grid_elec[row, col])
                        data_idx = elec_to_idx.get(elec_id, None)
                        
                        if data_idx is None or data_idx >= n_ch:
                            continue
                        
                        # Collect spectrograms across all trials for this electrode
                        trial_spectrograms = []
                        t_spec_ms_saved = None
                        f_out_saved = None
                        
                        for t_idx in range(n_trials):
                            trace = bb_full[t_idx, data_idx, :]
                            
                            if np.all(np.isnan(trace)) or np.std(trace) < 1e-10:
                                continue
                            
                            # Compute spectrogram (raw power, no normalization yet)
                            Sxx, f_out, t_out = compute_wavelet_spectrogram(
                                trace, fs, foi=foi,
                                baseline_window=None,  # Don't normalize yet
                                t_ms=t_ms,
                                n_cycles=n_cycles,
                                normalize=False
                            )
                            
                            if Sxx is not None:
                                trial_spectrograms.append(Sxx)
                                if t_spec_ms_saved is None:
                                    t_spec_ms_saved = t_out * 1000 + t_ms[0]
                                    f_out_saved = f_out
                        
                        # MEDIAN across trials (instead of mean)
                        if trial_spectrograms:
                            stacked = np.stack(trial_spectrograms, axis=0)  # (n_trials, n_freq, n_time)
                            
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                Sxx_median = np.nanmedian(stacked, axis=0)  # CHANGED: median
                            
                            # Z-score normalize the median spectrogram
                            bl_mask = (t_spec_ms_saved >= baseline_win[0]) & (t_spec_ms_saved <= baseline_win[1])
                            Sxx_normalized = zscore_normalize_spectrogram(Sxx_median, baseline_mask=bl_mask)
                            
                            all_spectrograms[(row, col)] = {
                                'Sxx': Sxx_normalized,
                                'f': f_out_saved,
                                't_ms': t_spec_ms_saved,
                                'elec_id': elec_id,
                                'n_trials_averaged': len(trial_spectrograms),
                            }
                
                if not all_spectrograms:
                    print(f"    [Skip] No valid spectrograms for {region_name} (trial average)")
                    continue
                
                # Create figure for trial average
                _plot_spectrogram_grid(
                    all_spectrograms, grid_elec, elec_to_idx, n_ch,
                    session_id, region_name, safe_region, condition_dir, fig_dir,
                    spec_cfg, normalize_method, n_trials,
                    trial_label=f"Trial Median (n={n_trials})",
                    filename_suffix="trial_median"
                )
                
            else:
                # === SINGLE TRIAL SPECTROGRAM ===
                trial_data = bb_full[trial_idx, :, :]
                
                # First pass: compute all spectrograms
                all_spectrograms = {}
                
                for row in range(8):
                    for col in range(8):
                        elec_id = int(grid_elec[row, col])
                        data_idx = elec_to_idx.get(elec_id, None)
                        
                        if data_idx is None or data_idx >= n_ch:
                            continue
                        
                        trace = trial_data[data_idx, :]
                        
                        if np.all(np.isnan(trace)) or np.std(trace) < 1e-10:
                            continue
                        
                        # Compute spectrogram (raw power)
                        Sxx, f_out, t_out = compute_wavelet_spectrogram(
                            trace, fs, foi=foi,
                            baseline_window=None,
                            t_ms=t_ms,
                            n_cycles=n_cycles,
                            normalize=False
                        )
                        
                        if Sxx is not None:
                            t_spec_ms = t_out * 1000 + t_ms[0]
                            
                            # Z-score normalize single trial
                            bl_mask = (t_spec_ms >= baseline_win[0]) & (t_spec_ms <= baseline_win[1])
                            Sxx_normalized = zscore_normalize_spectrogram(Sxx, baseline_mask=bl_mask)
                            
                            all_spectrograms[(row, col)] = {
                                'Sxx': Sxx_normalized,
                                'f': f_out,
                                't_ms': t_spec_ms,
                                'elec_id': elec_id
                            }
                
                if not all_spectrograms:
                    print(f"    [Skip] No valid spectrograms for {region_name} trial {trial_idx}")
                    continue
                
                # Create figure for single trial
                _plot_spectrogram_grid(
                    all_spectrograms, grid_elec, elec_to_idx, n_ch,
                    session_id, region_name, safe_region, condition_dir, fig_dir,
                    spec_cfg, normalize_method, n_trials,
                    trial_label=f"Trial {trial_idx}",
                    filename_suffix=f"trial_{trial_idx}"
                )
    
    return out_dir
    
def _plot_spectrogram_grid(all_spectrograms, grid_elec, elec_to_idx, n_ch,
                           session_id, region_name, safe_region, condition_dir, fig_dir,
                           spec_cfg, normalize_method, n_trials,
                           trial_label="", filename_suffix=""):
    """
    Helper function to plot 8x8 spectrogram grid.
    
    Updated to handle z-score normalized data with appropriate color scaling.
    """
    cmap = spec_cfg.get('cmap', 'viridis')
    
    # Determine color scale based on normalization method
    if normalize_method == 'zscore':
        vmin = spec_cfg.get('vmin_zscore', -3)
        vmax = spec_cfg.get('vmax_zscore', 3)
        cbar_label = 'Normalized Power (a.u.)'  # CHANGED
        cmap = 'jet'  # Diverging colormap better for z-scores
    elif normalize_method == 'dB':
        vmin = spec_cfg.get('vmin_db', -6)
        vmax = spec_cfg.get('vmax_db', 6)
        cbar_label = 'Power (dB re: baseline)'
        cmap = 'jet'
    else:  # raw
        all_power = np.concatenate([s['Sxx'].flatten() for s in all_spectrograms.values()])
        vmin = spec_cfg.get('vmin_uV2', 0)
        vmax = spec_cfg.get('vmax_uV2', None)
        if vmax is None:
            vmax = np.nanpercentile(all_power, 95)
        cbar_label = 'Power (µV²)'
    
    # Create 8x8 figure
    fig, axes = plt.subplots(8, 8, figsize=(24, 24))
    
    # Build subtitle showing frequency gaps
    notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])
    notch_str = ', '.join([f'{lo}-{hi} Hz' for lo, hi in notch_regions])
    
    # Update title to reflect normalization method
    norm_str = "Z-scored" if normalize_method == 'zscore' else normalize_method.upper()

    tfr_method = spec_cfg.get('tfr_method', 'morlet').lower()
    if tfr_method == 'morlet_adaptive':
        tfr_label = (
            f"Adaptive Morlet "
            f"({spec_cfg.get('adaptive_cycles_low', 3)}-"
            f"{spec_cfg.get('adaptive_cycles_high', 7)} cycles, "
            f"{spec_cfg.get('adaptive_freq_low_max', 15)}-"
            f"{spec_cfg.get('adaptive_freq_high_min', 30)} Hz transition)"
        )
    else:
        tfr_label = f"Morlet Wavelets ({spec_cfg.get('wavelet_cycles', 5)} cycles)"
    
    fig.suptitle(f'{session_id} — {region_name}\n'
            f'Time-Frequency Power ({trial_label}) | {tfr_label} | {norm_str}\n'
            f'(Notch regions skipped: {notch_str})',
            fontsize=16, fontweight='bold')
    
    im_for_cbar = None
    
    for row in range(8):
        for col in range(8):
            ax = axes[row, col]
            elec_id = int(grid_elec[row, col])
            
            if (row, col) in all_spectrograms:
                spec_data = all_spectrograms[(row, col)]
                Sxx = spec_data['Sxx']
                f_spec = spec_data['f']
                t_spec_ms = spec_data['t_ms']
                
                im = ax.pcolormesh(t_spec_ms, np.arange(len(f_spec)), Sxx,
                                  cmap=cmap, vmin=vmin, vmax=vmax,
                                  shading='auto', rasterized=True)
                
                # Set y-ticks to show actual frequencies
                freq_step = spec_cfg.get('freq_step', 1)
                tick_freqs = [1, 10, 20, 30, 40, 55, 65, 80, 100, 115]
                tick_positions = []
                tick_labels = []
                for tf in tick_freqs:
                    if tf in f_spec:
                        idx = np.where(f_spec == tf)[0][0]
                        tick_positions.append(idx)
                        tick_labels.append(str(tf))
                    elif any(abs(f_spec - tf) < freq_step):
                        idx = np.argmin(abs(f_spec - tf))
                        tick_positions.append(idx)
                        tick_labels.append(str(int(f_spec[idx])))
                
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(tick_labels)
                
                # Mark stim onset
                ax.axvline(0, color='white', ls='--', lw=0.8, alpha=0.8)
                
                # Mark where notch gaps are with lines
                for notch_lo, notch_hi in notch_regions:
                    below_idx = np.where(f_spec < notch_lo)[0]
                    above_idx = np.where(f_spec > notch_hi)[0]
                    if len(below_idx) > 0 and len(above_idx) > 0:
                        gap_y = below_idx[-1] + 0.5
                        ax.axhline(gap_y, color='white', ls='-', lw=1.5, alpha=0.8)
                
                im_for_cbar = im
                
                # Title with trial count if averaged
                title_str = f'E{elec_id}'
                if 'n_trials_averaged' in spec_data:
                    title_str += f' (n={spec_data["n_trials_averaged"]})'
                ax.set_title(title_str, fontsize=8, pad=2)
                
            else:
                ax.set_facecolor('#e0e0e0')
                data_idx = elec_to_idx.get(elec_id, None)
                if data_idx is None:
                    label = 'N/A'
                    color = 'gray'
                else:
                    label = 'Bad'
                    color = 'red'
                ax.text(0.5, 0.5, f'E{elec_id}\n{label}', 
                       ha='center', va='center', fontsize=8,
                       transform=ax.transAxes, color=color)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Axis labels only on edges
            if row == 7:
                ax.set_xlabel('Time (ms)', fontsize=7)
            else:
                ax.set_xticklabels([])
            
            if col == 0:
                ax.set_ylabel('Freq (Hz)', fontsize=7)
            else:
                ax.set_yticklabels([])
            
            ax.tick_params(axis='both', labelsize=6)
    
    # Add colorbar
    if im_for_cbar is not None:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im_for_cbar, cax=cbar_ax)
        cbar.set_label(cbar_label, fontsize=12)  # Now shows "Normalized Power (a.u.)"
        cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout(rect=[0, 0, 0.92, 0.94])
    
    # Save
    out_path = condition_dir / f"{safe_region}_{filename_suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"    Saved -> {out_path.relative_to(fig_dir)}")

def generate_per_ua_array_median_spectrogram(data, session_id, fig_dir, groups, fs=1000,
                                             ua_ids_1based=None, nsp_to_elec=None,
                                             region_grids=None, elec_to_idx=None,
                                             control_median_baseline_norm=None, 
                                             control_self_baseline_norm_array_median=None,):
    """
    Generate one 4-panel figure per session.

    Each subplot is one Utah array/region.
    For each region, this computes the median spectrogram across:
        1. all trials
        2. all valid channels/electrodes in that region

    Output shape per region:
        (n_freq, n_time)

    This is different from generate_per_ua_trial_average(), which gives one
    median-across-trials spectrogram per electrode in an 8x8 grid.
    """

    try:
        bb_full, t_ms = get_broadband_full(data)
    except KeyError as e:
        print(f"    [Skip] No broadband data for per-UA array median spectrogram: {e}")
        return None

    if region_grids is None or elec_to_idx is None:
        print("    [Skip] Need electrode mapping for per-UA array median spectrogram")
        return None

    # Spectrogram settings
    spec_cfg = PLOT_CONFIG.get('spectrogram', {})
    freq_step = spec_cfg.get('freq_step', 1)

    normalize_method = spec_cfg.get('normalize', 'zscore')
    baseline_win = ANALYSIS_CONFIG['baseline_window']

    # Output directory
    out_dir = fig_dir / "per_UA_activity"

    # Same condition naming logic as generate_per_ua_spectrograms()
    br_match = re.search(r'_(\d{3})(?:_|$|\.)', session_id)
    target_match = re.search(r'target_([AB])$', session_id, re.IGNORECASE)
    target_suffix = f"_{target_match.group(1).upper()}" if target_match else ""

    if br_match:
        br_idx = int(br_match.group(1))
        condition_name = f"condition_{br_idx:02d}{target_suffix}"
    else:
        br_idx = data.get('br_idx', None)
        if br_idx is not None:
            if isinstance(br_idx, np.ndarray):
                br_idx = int(br_idx.item())
            condition_name = f"condition_{br_idx:02d}{target_suffix}"
        else:
            condition_name = session_id

    condition_dir = out_dir / condition_name
    condition_dir.mkdir(parents=True, exist_ok=True)

    # Determine color scale
    cmap = spec_cfg.get('cmap', 'viridis')

    if normalize_method == 'zscore':
        vmin = spec_cfg.get('vmin_zscore', -3)
        vmax = spec_cfg.get('vmax_zscore', 3)
        cbar_label = 'Normalized Power (a.u.)'
        cmap = 'jet'
    elif normalize_method == 'dB':
        vmin = spec_cfg.get('vmin_db', -6)
        vmax = spec_cfg.get('vmax_db', 6)
        cbar_label = 'Power (dB re: baseline)'
        cmap = 'jet'
    else:
        vmin = spec_cfg.get('vmin_uV2', 0)
        vmax = spec_cfg.get('vmax_uV2', None)
        cbar_label = 'Power (µV²)'

    print("      Computing array-level median spectrograms across trials and channels...")

    region_specs_raw = compute_array_median_raw_spectrograms(
        data=data,
        groups=groups,
        fs=fs,
        region_grids=region_grids,
        elec_to_idx=elec_to_idx,
    )

    if not region_specs_raw:
        print("    [Skip] No valid raw array-level median spectrograms were computed")
        return None

    region_specs = {}

    for region_name, raw_spec in region_specs_raw.items():
        Sxx_region_median_raw = raw_spec['Sxx']
        t_spec_ms_saved = raw_spec['t_ms']

        Sxx_region_median = Sxx_region_median_raw.copy()

        # Normalize AFTER taking the median
        if normalize_method == 'zscore':
            bl_mask = (
                (t_spec_ms_saved >= baseline_win[0]) &
                (t_spec_ms_saved <= baseline_win[1])
            )

            Sxx_region_median = zscore_normalize_spectrogram(
                Sxx_region_median,
                baseline_mask=bl_mask,
            )

        elif normalize_method == 'dB':
            bl_mask = (
                (t_spec_ms_saved >= baseline_win[0]) &
                (t_spec_ms_saved <= baseline_win[1])
            )

            if bl_mask.any():
                baseline_power = np.nanmean(
                    Sxx_region_median[:, bl_mask],
                    axis=1,
                    keepdims=True,
                )
                baseline_power[baseline_power < 1e-20] = 1e-20

                Sxx_region_median = 10 * np.log10(
                    Sxx_region_median / baseline_power
                )

        region_specs[region_name] = {
            **raw_spec,
            'Sxx': Sxx_region_median,
        }

    if not region_specs:
        print("    [Skip] No valid array-level median spectrograms were computed")
        return None

    # If raw mode and vmax is None, compute global 95th percentile
    if normalize_method == 'raw' and vmax is None:
        all_power = np.concatenate([
            spec['Sxx'].ravel()
            for spec in region_specs.values()
            if spec['Sxx'] is not None
        ])
        vmax = np.nanpercentile(all_power, 95)

    # -------------------------------------------------------------------------
    # Plot 4-panel figure
    # -------------------------------------------------------------------------
    n_regions = len(region_specs)

    # Usually 4 arrays: SMA, PMd, M1i, M1s
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    notch_regions = ANALYSIS_CONFIG.get('notch_regions', [(55, 65), (115, 125)])
    notch_str = ', '.join([f'{lo}-{hi} Hz' for lo, hi in notch_regions])

    norm_str = "Z-scored" if normalize_method == 'zscore' else normalize_method.upper()

    tfr_method = spec_cfg.get('tfr_method', 'morlet').lower()

    if tfr_method == 'morlet_adaptive':
        adaptive_cycles_low = spec_cfg.get('adaptive_cycles_low', 3)
        adaptive_cycles_high = spec_cfg.get('adaptive_cycles_high', 7)
        adaptive_freq_low_max = spec_cfg.get('adaptive_freq_low_max', 15)
        adaptive_freq_high_min = spec_cfg.get('adaptive_freq_high_min', 30)

        if adaptive_freq_high_min <= adaptive_freq_low_max:
            adaptive_freq_high_min = adaptive_freq_low_max + 15

        tfr_label = (
            f"Adaptive Morlet "
            f"({adaptive_cycles_low}-{adaptive_cycles_high} cycles, "
            f"{adaptive_freq_low_max}-{adaptive_freq_high_min} Hz transition)"
        )
    else:
        tfr_label = f"Morlet Wavelets ({spec_cfg.get('wavelet_cycles', 5)} cycles)"

    fig.suptitle(
        f'{session_id}\n'
        f'Array-Level Median Spectrograms | Median Across All Trials and Channels\n'
        f'{tfr_label} | {norm_str} | Notch regions skipped: {notch_str}',
        fontsize=14,
        fontweight='bold'
    )

    im_for_cbar = None

    # Keep plotting order consistent with groups
    plot_idx = 0

    for grp_idxs, grp_name in groups:
        region_name = grp_name.split(' (')[0]

        if region_name not in region_specs:
            continue

        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]
        spec = region_specs[region_name]

        Sxx = spec['Sxx']
        f_spec = spec['f']
        t_spec_ms = spec['t_ms']

        im = ax.pcolormesh(
            t_spec_ms,
            np.arange(len(f_spec)),
            Sxx,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading='auto',
            rasterized=True
        )

        im_for_cbar = im

        # Mark stim onset
        ax.axvline(0, color='white', ls='--', lw=1.2, alpha=0.9)

        # Optional: mark end of stimulation if you consider 100 ms stimulation
        ax.axvline(100, color='white', ls=':', lw=1.0, alpha=0.7)

        # Mark notch gaps
        for notch_lo, notch_hi in notch_regions:
            below_idx = np.where(f_spec < notch_lo)[0]
            above_idx = np.where(f_spec > notch_hi)[0]

            if len(below_idx) > 0 and len(above_idx) > 0:
                gap_y = below_idx[-1] + 0.5
                ax.axhline(gap_y, color='white', ls='-', lw=1.5, alpha=0.8)

        # Frequency tick labels
        tick_freqs = [1, 10, 20, 30, 40, 55, 65, 80, 100, 115]
        tick_positions = []
        tick_labels = []

        for tf in tick_freqs:
            if tf in f_spec:
                idx = np.where(f_spec == tf)[0][0]
                tick_positions.append(idx)
                tick_labels.append(str(tf))
            else:
                idx = np.argmin(np.abs(f_spec - tf))
                if np.abs(f_spec[idx] - tf) <= freq_step:
                    tick_positions.append(idx)
                    tick_labels.append(str(int(f_spec[idx])))

        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

        ax.set_title(
            f'{region_name}\n'
            f'{spec["n_valid_channels"]} channels, '
            f'{spec["n_valid_trial_channel_pairs"]} trial-channel spectra',
            fontsize=11,
            fontweight='bold'
        )

        ax.set_xlabel('Time relative to event onset (ms)')
        ax.set_ylabel('Frequency (Hz)')

        ax.tick_params(axis='both', labelsize=9)

        plot_idx += 1

    # Hide unused axes if fewer than 4 regions
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    # Colorbar
    if im_for_cbar is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.18, 0.02, 0.65])
        cbar = fig.colorbar(im_for_cbar, cax=cbar_ax)
        cbar.set_label(cbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0, 0.88, 0.92])

    out_path = condition_dir / "array_median_across_trials_and_channels.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"    Saved array-level median spectrogram -> {out_path.relative_to(fig_dir)}")

    if control_median_baseline_norm is not None:
        out_path_ctrl = condition_dir / "array_median_across_trials_and_channels_CONTROL_MEDIAN_BASELINE_NORM.png"

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        fig.suptitle(
            f'{session_id}\n'
            f'Array-Level Median Spectrograms | Control Median-Baseline Normalized\n'
            f'{tfr_label} | Notch regions skipped: {notch_str}',
            fontsize=14,
            fontweight='bold'
        )

        im_for_cbar = None
        plot_idx = 0

        for grp_idxs, grp_name in groups:
            region_name = grp_name.split(' (')[0]

            if region_name not in region_specs_raw:
                continue
            if region_name not in control_median_baseline_norm:
                continue
            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]
            spec = region_specs_raw[region_name]

            Sxx_raw = spec['Sxx']
            mean_val = control_median_baseline_norm[region_name]['mean']
            std_val = control_median_baseline_norm[region_name]['std']

            Sxx_ctrl = (Sxx_raw - mean_val) / std_val

            f_spec = spec['f']
            t_spec_ms = spec['t_ms']

            im = ax.pcolormesh(
                t_spec_ms,
                np.arange(len(f_spec)),
                Sxx_ctrl,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading='auto',
                rasterized=True
            )

            im_for_cbar = im

            ax.axvline(0, color='white', ls='--', lw=1.2, alpha=0.9)
            ax.axvline(100, color='white', ls=':', lw=1.0, alpha=0.7)

            for notch_lo, notch_hi in notch_regions:
                below_idx = np.where(f_spec < notch_lo)[0]
                above_idx = np.where(f_spec > notch_hi)[0]
                if len(below_idx) > 0 and len(above_idx) > 0:
                    gap_y = below_idx[-1] + 0.5
                    ax.axhline(gap_y, color='white', ls='-', lw=1.5, alpha=0.8)

            tick_freqs = [1, 10, 20, 30, 40, 55, 65, 80, 100, 115]
            tick_positions = []
            tick_labels = []

            for tf in tick_freqs:
                if tf in f_spec:
                    idx = np.where(f_spec == tf)[0][0]
                    tick_positions.append(idx)
                    tick_labels.append(str(tf))
                else:
                    idx = np.argmin(np.abs(f_spec - tf))
                    if np.abs(f_spec[idx] - tf) <= freq_step:
                        tick_positions.append(idx)
                        tick_labels.append(str(int(f_spec[idx])))

            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)

            ax.set_title(
                f'{region_name}\n'
                f'{spec["n_valid_channels"]} channels, '
                f'{spec["n_valid_trial_channel_pairs"]} trial-channel spectra',
                fontsize=11,
                fontweight='bold'
            )

            ax.set_xlabel('Time relative to event onset (ms)')
            ax.set_ylabel('Frequency (Hz)')
            ax.tick_params(axis='both', labelsize=9)

            plot_idx += 1

        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')

        if im_for_cbar is not None:
            fig.subplots_adjust(right=0.88)
            cbar_ax = fig.add_axes([0.90, 0.18, 0.02, 0.65])
            cbar = fig.colorbar(im_for_cbar, cax=cbar_ax)
            cbar.set_label('Power vs control median baseline (z)', fontsize=12)
            cbar.ax.tick_params(labelsize=10)

        plt.tight_layout(rect=[0, 0, 0.88, 0.92])
        plt.savefig(out_path_ctrl, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"    Saved control median-baseline normalized array-level median spectrogram -> {out_path_ctrl.relative_to(fig_dir)}")

    # -------------------------------------------------------------------------
    # Optional third output:
    # ΔZ = stim self-baseline-normalized array median
    #      minus matched control self-baseline-normalized array median
    # -------------------------------------------------------------------------
    if (FIGURE_CONFIG.get('generate_self_baseline_minus_control_array_median', False)
           and control_self_baseline_norm_array_median is not None):
        out_path_diff = condition_dir / "array_median_across_trials_and_channels_SELF_BASELINE_MINUS_CONTROL_SELF_BASELINE.png"

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        diff_vlim = spec_cfg.get('self_minus_control_vlim', 3)
        diff_cmap = spec_cfg.get('self_minus_control_cmap', 'RdBu_r')

        fig.suptitle(
            f'{session_id}\n'
            f'Array-Level Median Spectrogram Contrast\n'
            f'ΔZ = stim self-baseline-normalized minus matched control self-baseline-normalized\n'
            f'{tfr_label} | Notch regions skipped: {notch_str}',
            fontsize=14,
            fontweight='bold'
        )

        im_for_cbar = None
        plot_idx = 0

        for grp_idxs, grp_name in groups:
            region_name = grp_name.split(' (')[0]

            if region_name not in region_specs:
                continue

            if region_name not in control_self_baseline_norm_array_median:
                continue

            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]

            stim_spec = region_specs[region_name]
            ctrl_spec = control_self_baseline_norm_array_median[region_name]

            Sxx_stim_self = stim_spec['Sxx']
            Sxx_ctrl_self = ctrl_spec['Sxx']

            f_spec = stim_spec['f']
            t_spec_ms = stim_spec['t_ms']

            if Sxx_stim_self.shape != Sxx_ctrl_self.shape:
                print(
                    f"      [Skip] Self-baseline stim-control subtraction shape mismatch "
                    f"for {region_name}: stim {Sxx_stim_self.shape}, "
                    f"control {Sxx_ctrl_self.shape}"
                )
                continue

            Sxx_diff = Sxx_stim_self - Sxx_ctrl_self

            im = ax.pcolormesh(
                t_spec_ms,
                np.arange(len(f_spec)),
                Sxx_diff,
                cmap=diff_cmap,
                vmin=-diff_vlim,
                vmax=diff_vlim,
                shading='auto',
                rasterized=True
            )

            im_for_cbar = im

            ax.axvline(0, color='black', ls='--', lw=1.2, alpha=0.9)
            ax.axvline(100, color='black', ls=':', lw=1.0, alpha=0.7)

            for notch_lo, notch_hi in notch_regions:
                below_idx = np.where(f_spec < notch_lo)[0]
                above_idx = np.where(f_spec > notch_hi)[0]

                if len(below_idx) > 0 and len(above_idx) > 0:
                    gap_y = below_idx[-1] + 0.5
                    ax.axhline(gap_y, color='black', ls='-', lw=1.5, alpha=0.8)

            tick_freqs = [1, 10, 20, 30, 40, 55, 65, 80, 100, 115]
            tick_positions = []
            tick_labels = []

            for tf in tick_freqs:
                if tf in f_spec:
                    idx = np.where(f_spec == tf)[0][0]
                    tick_positions.append(idx)
                    tick_labels.append(str(tf))
                else:
                    idx = np.argmin(np.abs(f_spec - tf))
                    if np.abs(f_spec[idx] - tf) <= freq_step:
                        tick_positions.append(idx)
                        tick_labels.append(str(int(f_spec[idx])))

            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels)

            ax.set_title(
                f'{region_name}\n'
                f'{stim_spec["n_valid_channels"]} channels, '
                f'{stim_spec["n_valid_trial_channel_pairs"]} trial-channel spectra',
                fontsize=11,
                fontweight='bold'
            )

            ax.set_xlabel('Time relative to event onset (ms)')
            ax.set_ylabel('Frequency (Hz)')
            ax.tick_params(axis='both', labelsize=9)

            plot_idx += 1

        for idx in range(plot_idx, len(axes)):
            axes[idx].axis('off')

        if im_for_cbar is not None:
            fig.subplots_adjust(right=0.88)
            cbar_ax = fig.add_axes([0.90, 0.18, 0.02, 0.65])
            cbar = fig.colorbar(im_for_cbar, cax=cbar_ax)
            cbar.set_label('ΔZ power', fontsize=12)
            cbar.ax.tick_params(labelsize=10)

        plt.tight_layout(rect=[0, 0, 0.88, 0.90])
        plt.savefig(out_path_diff, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(
            f"    Saved self-baseline minus matched-control array-level median spectrogram "
            f"-> {out_path_diff.relative_to(fig_dir)}"
        )

    return out_path


def build_groups(data, n_ch):
    """Helper to build channel groups and electrode mappings."""
    is_utah = 'ua_ids_1based' in data
    ua_ids = data.get('ua_ids_1based', None)
    
    nsp_to_elec = None
    elec_to_idx = None
    region_grids = None
    groups = []
    
    if is_utah:
        try:
            from RCP_analysis.python.functions.br_preproc import (
                load_electrode_mapping,
                get_region_grid,
                build_elec_to_data_idx
            )
            
            import RCP_analysis.python.functions.config_loading as cfg

            monkey = str(cfg.PARAMS.monkey)

            if monkey not in ("Nike", "Ada"):
                raise ValueError(f"Unknown PARAMS.monkey={monkey!r}; expected 'Nike' or 'Ada'")

            MAPPING_CSV = cfg.REPO_ROOT / "config" / f"electrode_port_mapping_{monkey}.csv"
            nsp_to_elec, UTAH_ELEC_GRIDS, elec_to_region = load_electrode_mapping(MAPPING_CSV)
            
            # Build groups by region
            sma_idxs, pmd_idxs, m1i_idxs, m1s_idxs = [], [], [], []
            
            for ch_idx in range(n_ch):
                nsp_id = int(ua_ids[ch_idx])
                elec_id = nsp_to_elec.get(nsp_id, -1)
                if elec_id < 1:
                    continue
                region = elec_to_region.get(elec_id)
                if region == "SMA":
                    sma_idxs.append(ch_idx)
                elif region == "PMd":
                    pmd_idxs.append(ch_idx)
                elif region in ["M1i", "M1 Inf"]:
                    m1i_idxs.append(ch_idx)
                elif region in ["M1s", "M1 Sup"]:
                    m1s_idxs.append(ch_idx)
            
            if sma_idxs: groups.append((sma_idxs, f"SMA (n={len(sma_idxs)})"))
            if pmd_idxs: groups.append((pmd_idxs, f"PMd (n={len(pmd_idxs)})"))
            if m1i_idxs: groups.append((m1i_idxs, f"M1i (n={len(m1i_idxs)})"))
            if m1s_idxs: groups.append((m1s_idxs, f"M1s (n={len(m1s_idxs)})"))
            
            # Build electrode mapping for spatial plots
            ua_port = 'A'
            elec_to_idx = build_elec_to_data_idx(ua_ids, nsp_to_elec, ua_port)
            
            # Region grids for spatial heatmaps
            region_grids = {
                'SMA': get_region_grid('SMA', UTAH_ELEC_GRIDS),
                'PMd': get_region_grid('PMd', UTAH_ELEC_GRIDS),
                'M1i': get_region_grid('M1i', UTAH_ELEC_GRIDS),
                'M1s': get_region_grid('M1s', UTAH_ELEC_GRIDS),
            }
            
        except Exception as e:
            print(f"    [Warn] Failed to load electrode mapping: {e}")
            is_utah = False
    
    if not groups:
        # Fallback: simple split into 4 groups
        ch_per_group = n_ch // 4
        groups = [
            (list(range(0, ch_per_group)), f"Group 1 (n={ch_per_group})"),
            (list(range(ch_per_group, 2*ch_per_group)), f"Group 2 (n={ch_per_group})"),
            (list(range(2*ch_per_group, 3*ch_per_group)), f"Group 3 (n={ch_per_group})"),
            (list(range(3*ch_per_group, n_ch)), f"Group 4 (n={n_ch - 3*ch_per_group})"),
        ]
    
    return groups, nsp_to_elec, elec_to_idx, region_grids, ua_ids


__all__ = [
    "compute_confidence_interval",
    "get_region_color",
    "add_band_shading",
    "add_stim_marker",
    "get_trial_count",
    "get_time_axis",
    "get_broadband_full",
    "slice_time_window",
    "compute_array_median_raw_spectrograms",
    "compute_control_median_baseline_norm",
    "compute_self_baseline_norm_array_median_spectrograms",
    "compute_control_spatial_baseline",
    "generate_session_summary",
    "generate_waveform_raster",
    "generate_trial_heatmaps",
    "generate_coherence_matrix",
    "generate_stim_vs_control",
    "generate_dose_response",
    "render_spatial_grid",
    "generate_spatial_gif",
    "generate_spatial_heatmaps",
    "generate_per_ua_spectrograms",
    "generate_per_ua_array_median_spectrogram",
    "run_wavelet_comparison",
    "build_groups",
]