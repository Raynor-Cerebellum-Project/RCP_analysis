import numpy as np
import matplotlib.pyplot as plt
import RCP_analysis.python.functions.config_loading as cfg
import RCP_analysis.python.functions.utils as rcp
import RCP_analysis.python.functions.br_preproc as br_preproc
import warnings
from pathlib import Path
from scipy import signal, stats
import pandas as pd
import csv

# Define output directory from config
NPRW_LFP_DIR = cfg.OUT_BASE / "checkpoints" / "NPRW_LFP"
UA_LFP_DIR = cfg.OUT_BASE / "checkpoints" / "UA_LFP"

NPRW_FIG_DIR = cfg.OUT_BASE / "figures" / "NPRW_LFP"
UA_FIG_DIR = cfg.OUT_BASE / "figures" / "UA_LFP"

NPRW_FIG_DIR.mkdir(parents=True, exist_ok=True)
UA_FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Global Plotting Configuration ---
FIGURE_CONFIG = {
    'process_original': False,          # The existing heatmap/trace plot
    'process_power_spectra': True,      # PSD (Pre vs Post)
    'process_lfp_coherence': False,     # (Stage 2: Disabled) Mag Squared Coherence 
    'process_phase_coherence': True,    # ITPC (Phase consistency across trials)
    'process_ersp': True,               # (Stage 2: New) Event-Related Spectral Perturbation
    'process_band_stats': True,         # (Stage 2: New) Band Power Stats (Pre vs Post)
}

PLOT_CONFIG = {
    # Opacity for individual trial traces
    "trace_alpha": 0.4,
    
    # Colormap for Heatmaps
    "heatmap_cmap": "RdBu_r",
    
    # Heatmap Scales (uV): Set value to enforce fixed scale, or None for dynamic (percentile)
    "scales": {
        'broadband': 30.0,
        'alpha': 10.0,
        'beta': 10.0,
        'low_gamma': 10.0,
        'high_gamma': 10.0
    },
    
    # Trace Y-Axis Scales (uV): Set value for fixed +/- limit, or None for dynamic
    "trace_scales": {
        'broadband': 200, 
        'alpha': 100,
        'beta': 100,
        'low_gamma': 100,
        'high_gamma': 100
    },

    # X-Axis Limits (ms): Set tuple (min, max) or None for default
    "x_limits": {
        "pre": (-500, -5), 
        "post": (101, 605)
    }
}

def get_representative_channel(data_array):
    """
    Find a channel index that has valid, non-flat data.
    data_array: (n_trials, n_ch, n_time)
    """
    if data_array is None:
        return 0
        
    n_ch = data_array.shape[1]
    
    # Try middle first
    mid = n_ch // 2
    
    # Scan outward from middle
    offsets = np.arange(n_ch)
    # Interleave 0, 1, -1, 2, -2...
    scan_order = []
    for i in range(n_ch):
        if mid + i < n_ch: scan_order.append(mid + i)
        if i > 0 and mid - i >= 0: scan_order.append(mid - i)
        
    for ch in scan_order:
        # Check if channel has data transparency
        traces = data_array[:, ch, :]
        if np.all(np.isnan(traces)):
             continue
        if np.nanstd(traces) < 1e-6: # Flat line
             continue
        return ch
        
    return mid # Fallback

def calculate_psd(data, fs):
    """
    Calculate Power Spectral Density averaged across trials.
    Returns: freq, psd_mean (n_ch, n_freq), psd_sem (n_ch, n_freq)
    data: (n_trials, n_ch, n_time)
    """
    n_trials, n_ch, n_time = data.shape
    
    # Calculate PSD for each trial/channel
    # nperseg = length of window. If too short, reduce it.
    nperseg = min(n_time, 256)
    
    f, Pxx = signal.welch(data, fs=fs, nperseg=nperseg, axis=2)
    
    # Valid Pxx: (n_trials, n_ch, n_freq)
    # Average across trials
    psd_mean = np.nanmean(Pxx, axis=0)
    psd_sem = np.nanstd(Pxx, axis=0) / np.sqrt(n_trials)
    
    return f, psd_mean, psd_sem

def calculate_coherence(data, rep_ch_idx, fs):
    """
    Calculate Magnitude Squared Coherence between rep_ch and all other channels.
    Returns: freq, coh_mean (n_ch, n_freq) - averaged across trials
    """
    n_trials, n_ch, n_time = data.shape
    nperseg = min(n_time, 256)
    
    rep_data = data[:, rep_ch_idx, :] # (n_trials, n_time)
    
    all_coh = []
    
    # Iterate channels
    for ch in range(n_ch):
        target_data = data[:, ch, :]
        
        # Coherence per trial? Or coherence of concatenated trials?
        # Usually better to do coherence of the concatenation if stationary, 
        # or avg coherence across trials. 
        # signal.coherence handles raw signals.
        
        # We will calculate coherence per trial and average, to match trial structure
        f, Cxy = signal.coherence(rep_data, target_data, fs=fs, nperseg=nperseg, axis=1)
        # Cxy: (n_trials, n_freq)
        
        mean_cxy = np.nanmean(Cxy, axis=0)
        all_coh.append(mean_cxy)
        
    all_coh = np.stack(all_coh, axis=0) # (n_ch, n_freq)
    return f, all_coh

def calculate_itpc(data, fs):
    """
    Inter-Trial Phase Coherence.
    Returns: freq, itpc (n_ch, n_freq)
    Using Morlet Wavelets or Hilbert? 
    Given the short windows, let's use FFT-based ITPC (Phase consistency of Fourier components).
    """
    n_trials, n_ch, n_time = data.shape
    
    # FFT
    # axis=2 is time
    fft_res = np.fft.rfft(data, axis=2)
    # Frequencies
    freqs = np.fft.rfftfreq(n_time, d=1/fs)
    
    # Normalize to unit length (phase only)
    # Add eps to avoid divide by zero
    mags = np.abs(fft_res)
    mags[mags == 0] = 1e-10
    
    norm_fft = fft_res / mags
    
    # ITPC = |mean vector length across trials|
    itpc = np.abs(np.nanmean(norm_fft, axis=0)) # (n_ch, n_freq)
    
    return freqs, itpc

def calculate_tf_itpc(data, fs, nperseg=64, noverlap=50):
    """
    Calculate Time-Frequency Inter-Trial Phase Coherence using Short-Time FFT.
    Returns: f, t, itpc_matrix (n_ch, n_freq, n_time_bins)
    data: (n_trials, n_ch, n_time)
    """
    n_trials, n_ch, n_time = data.shape
    
    # Check if data is long enough for STFT
    if n_time < nperseg:
        nperseg = n_time // 2
        noverlap = nperseg // 2
        
    # We need to perform STFT on each trial and channel
    # This can be heavy, so loop carefully or vectorize if possible.
    # scipy.signal.stft can handle (..., time)
    
    # stft returns (f, t, Zxx) where Zxx is (..., n_freq, n_time_bins)
    f, t, Zxx = signal.stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=2)
    # Zxx shape: (n_trials, n_ch, n_freq, n_time_bins)
    
    # Normalize magnitude to 1 over trials (phase only)
    mags = np.abs(Zxx)
    mags[mags == 0] = 1e-10
    Z_norm = Zxx / mags
    
    # ITPC = Magnitude of mean vector across trials
    itpc_matrix = np.abs(np.nanmean(Z_norm, axis=0)) # (n_ch, n_freq, n_time)
    
    return f, t, itpc_matrix

def calculate_ersp(data_pre, data_post, fs, nperseg=128, noverlap=100):
    """
    Calculate Event-Related Spectral Perturbation (ERSP) in dB.
    Returns: f, t_post, ersp_matrix (n_ch, n_freq, n_time_bins)
    normalized to baseline (data_pre).
    """
    # 1. Calculate Baseline Power (Pre)
    # Use same STFT parameters
    f_pre, t_pre, Zxx_pre = signal.stft(data_pre, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=2)
    # Power = |Zxx|^2
    P_pre = np.abs(Zxx_pre)**2 
    # Mean Baseline Power per freq across time and trials (or just trials? usually across time too for true baseline)
    # Shape: (n_trials, n_ch, n_freq, n_time)
    # We want mean scalar per freq/ch to normalize.
    mean_baseline = np.nanmean(P_pre, axis=(0, 3)) # (n_ch, n_freq)
    # Avoid zero
    mean_baseline[mean_baseline == 0] = 1e-10
    
    # 2. Calculate Post Power
    f_post, t_post, Zxx_post = signal.stft(data_post, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=2)
    P_post = np.abs(Zxx_post)**2
    # Mean across trials for the "Evoked+Induced" response? 
    # ERSP is usually: Mean(Power_trial) / Baseline
    mean_post_power = np.nanmean(P_post, axis=0) # (n_ch, n_freq, n_time)
    
    # 3. Normalize (dB): 10 * log10(Post / Baseline)
    # Broadcast baseline across time
    ersp_matrix = 10 * np.log10(mean_post_power / mean_baseline[:, :, None])
    
    return f_post, t_post, ersp_matrix

def calculate_band_power_stats(data_pre, data_post, fs):
    """
    Compare band power between Pre and Post using Paired T-Test.
    Returns: Dict of results per band
    """
    bands = {
        'Alpha': (4, 13),
        'Beta': (13, 30),
        'Low Gamma': (30, 60),
        'High Gamma': (60, 120)
    }
    
    results = {}
    
    # PSD calculation for spectral estimation
    # Calculate PSD per trial
    nperseg = min(data_pre.shape[2], 256)
    f, Pxx_pre = signal.welch(data_pre, fs=fs, nperseg=nperseg, axis=2) # (N, Ch, F)
    _, Pxx_post = signal.welch(data_post, fs=fs, nperseg=nperseg, axis=2)
    
    for band_name, (f_min, f_max) in bands.items():
        # Find freq indices
        idx_band = np.where((f >= f_min) & (f <= f_max))[0]
        
        if len(idx_band) == 0:
            results[band_name] = None
            continue
            
        # Mean Power in band (integrate or mean?) -> Mean for simplicity match units
        # Shape: (N, Ch)
        pow_pre = np.nanmean(Pxx_pre[:, :, idx_band], axis=2)
        pow_post = np.nanmean(Pxx_post[:, :, idx_band], axis=2)
        
        # Paired T-Test across trials, for each channel
        # ttest_rel returns statistic, pvalue arrays of shape (Ch,)
        t_stats, p_vals = stats.ttest_rel(pow_pre, pow_post, axis=0, nan_policy='omit')
        
        mean_pre = np.nanmean(pow_pre, axis=0)
        mean_post = np.nanmean(pow_post, axis=0)
        sem_pre = np.nanstd(pow_pre, axis=0) / np.sqrt(pow_pre.shape[0])
        sem_post = np.nanstd(pow_post, axis=0) / np.sqrt(pow_post.shape[0])
        
        results[band_name] = {
            't_stat': t_stats,
            'p_val': p_vals,
            'mean_pre': mean_pre,
            'mean_post': mean_post,
            'sem_pre': sem_pre,
            'sem_post': sem_post
        }
        
    return results

def add_band_annotations(ax):
    """Adds vertical spans for standard LFP bands."""
    bands = [
        (4, 13, 'green', 'Alpha'),
        (13, 30, 'yellow', 'Beta'),
        (30, 60, 'orange', 'Low Gamma'),
        (60, 120, 'red', 'High Gamma')
    ]
    
    for start, end, color, name in bands:
        ax.axvspan(start, end, color=color, alpha=0.1, label=name)
        
    # Get handles for legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # Only keep band labels in a specific order if present
    
    return by_label

def process_directory(lfp_dir, fig_dir, label="LFP"):
    if not lfp_dir.exists():
        print(f"Directory not found: {lfp_dir}")
        return

    # Find latest aligned file
    files = sorted(lfp_dir.glob("aligned_lfp__*.npz"))
    if not files:
        print(f"No aligned LFP npz files found in {lfp_dir}")
        return
    
    print(f"--- Processing {label} ({len(files)} files) ---")
    
    # Load Stimulation Summary
    summary_csv = cfg.METADATA_ROOT / "stimulation_summary.csv"
    df_summary = None
    if summary_csv.exists():
        try:
            df_summary = pd.read_csv(summary_csv)
            # Ensure columns are string/clean
            if 'Intan_Session' in df_summary.columns:
                df_summary['Intan_Session'] = df_summary['Intan_Session'].astype(str).str.strip()
            print(f"Loaded stimulation summary from {summary_csv}")
        except Exception as e:
            print(f"Warning: Failed to load stimulation summary: {e}")
    else:
        print(f"Warning: Stimulation summary not found at {summary_csv}")

    # Iterate through all files
    for npz_file in files:
        print(f"Loading {npz_file.name}...")
        
        try:
            data = np.load(npz_file, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            continue

        # Extract Info
        if 'fs_lfp' in data: fs = int(data['fs_lfp'])
        else: fs = 1000
            
        if 'rel_time_post' in data: t_ms = data['rel_time_post']
        elif 'rel_time_ms' in data: t_ms = data['rel_time_ms']
        else:
            try:
                n_p = data['broadband_post'].shape[2]
                t_ms = np.arange(n_p) * 1000.0 / fs
            except KeyError:
                 # If empty or corrupt
                 print(f"  [Skip] Missing broadband_post in {npz_file.name}")
                 continue
        
        # Session Name
        if 'session' in data: session = str(data['session'])
        else: session = npz_file.stem.replace("aligned_lfp__", "")
            
        if 'broadband_post' not in data:
            print(f"Error: 'broadband_post' key missing in {npz_file.name}. Skipping.")
            continue
            
        bb_data = data['broadband_post']
        n_trials, n_ch, n_time = bb_data.shape
        
        print(f"  Dimensions: {n_trials} trials, {n_ch} channels")

        # --- Channel Order ---
        sort_idx = np.arange(n_ch)
        
        # --- Metadata (Detailed: Lookup from Summary CSV) ---
        target_intan_session = session 
        
        active_str = "Unknown"
        freq_str = "Unknown"
        amp_str = ""
        found_row = None
        
        # 1. Try Lookup by BR Index (for Utah)
        if label == "Utah_Array" and df_summary is not None:
             try:
                parts = npz_file.stem.split('_')
                if parts[-1].isdigit(): b_idx = int(parts[-1])
                elif parts[-2].isdigit(): b_idx = int(parts[-2])
                else: b_idx = None
                
                if b_idx is not None and 'BR_Index' in df_summary.columns:
                     matches = df_summary[df_summary['BR_Index'] == b_idx]
                     if not matches.empty:
                         found_row = matches.iloc[0]
                         target_intan_session = str(found_row['Intan_Session'])
             except: pass
        
        # 2. Try Lookup by Session Name (NPRW or Fallback)
        if found_row is None and df_summary is not None:
             if 'Intan_Session' in df_summary.columns:
                 # Check exact match
                 matches = df_summary[df_summary['Intan_Session'] == session]
                 if not matches.empty:
                     found_row = matches.iloc[0]
                 else:
                     # Check if session is a prefix of Intan_Session (common if dates missing)
                     # or if Intan_Session is a prefix of session
                     # We iterate? No, vectorized check
                     # Use 'startswith'
                     mask = df_summary['Intan_Session'].str.startswith(session)
                     if mask.any():
                         found_row = df_summary[mask].iloc[0]

        # 3. Extract Values
        if found_row is not None:
             active_str = str(found_row.get('Stim_Channels', 'Unknown'))
             freq_str = str(found_row.get('Stim_Frequency', 'Unknown'))
             
             amp_val = found_row.get('Stim_Amplitudes', 'N/A')
             if pd.notna(amp_val) and str(amp_val).lower() != 'nan':
                 amp_str = f"| {amp_val} uA"
                 
             print(f"  [Meta] Found: Ch={active_str}, Freq={freq_str}")
        else:
             # Fallback to NPZ (Old Logic)
             print(f"  [Warn] No summary metadata found for {session}. Using NPZ fallback.")
             s_ch = data.get('stim_channels', [])
             if len(s_ch) > 0: active_str = str(s_ch)
             
             saved_amps = data.get('stim_amplitudes', [])
             if len(saved_amps) > 0:
                 u_amps = np.unique(saved_amps)
                 amp_str = f"| {u_amps} uA"

        # Title Construction
        title_str = f"Stim: {active_str} | {freq_str} {amp_str}"
        
        # --- Pick Representative Channel ---
        rep_ch_idx = get_representative_channel(data.get('broadband_post'))
        
        # --- 1. Original Plot (Heatmaps + Traces) ---
        if FIGURE_CONFIG.get('process_original', True):
            print("  Generating Original Heatmap/Trace Plot...")
            bands = ['broadband', 'alpha', 'beta', 'low_gamma', 'high_gamma']
            n_rows = len(bands)
            
            # Combine Pre/Post into single heatmap
            fig, axes = plt.subplots(n_rows, 3, figsize=(18, 3 * n_rows), constrained_layout=True,
                                    gridspec_kw={'width_ratios': [3, 1, 1]})
            fig.suptitle(f"{session}\n{title_str}\n(N={n_trials})", fontsize=14)
            
            for i, band in enumerate(bands):
                # Keys
                key_pre = f"{band}_pre"
                key_post = f"{band}_post"
                
                # Prepare Data
                d_pre = data.get(key_pre) # (N, Ch, T)
                d_post = data.get(key_post)
                
                # Calc Means
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m_pre = np.nanmean(d_pre, axis=0) if d_pre is not None else None
                    m_post = np.nanmean(d_post, axis=0) if d_post is not None else None
                
                # --- Scales ---
                limit = PLOT_CONFIG['scales'].get(band)
                
                if limit is None:
                    # Dynamic Shared Scale
                    vals_for_scale = []
                    if m_pre is not None: vals_for_scale.append(m_pre.flatten())
                    if m_post is not None: vals_for_scale.append(m_post.flatten())
                    
                    if len(vals_for_scale) > 0:
                        all_vals = np.concatenate(vals_for_scale)
                        valid_vals = all_vals[np.isfinite(all_vals)]
                        if valid_vals.size > 0:
                            vmin, vmax = np.percentile(valid_vals, [2, 98])
                            limit = max(abs(vmin), abs(vmax))
                        else:
                            limit = 50.0
                    else:
                        limit = 50.0
                
                cmap = PLOT_CONFIG['heatmap_cmap']

                # Time
                t_pre = data.get('rel_time_pre')
                if t_pre is None and d_pre is not None: t_pre = np.arange(d_pre.shape[2])
                
                t_post = data.get('rel_time_post')
                if t_post is None and d_post is not None: t_post = t_ms
                
                # --- HEATMAPS (Combined) ---
                t_gap_start = t_pre[-1]
                t_gap_end = t_post[0]
                
                if len(t_pre) > 1: dt_est = t_pre[1] - t_pre[0]
                else: dt_est = 1.0 
                
                n_gap = int(np.round((t_gap_end - t_gap_start) / dt_est)) - 1
                if n_gap < 1: n_gap = 1 
                
                # Concatenate Data
                ax = axes[i, 0]
                
                if m_pre is not None and m_post is not None:
                    gap_data = np.full((n_ch, n_gap), np.nan)
                    m_comb = np.hstack([m_pre[sort_idx, :], gap_data, m_post[sort_idx, :]])
                    
                    t_start = t_pre[0]
                    t_end = t_post[-1]
                    
                    im = ax.imshow(m_comb, aspect='auto', origin='upper', cmap=cmap, 
                                   vmin=-limit, vmax=limit, extent=[t_start, t_end, n_ch - 0.5, -0.5]) 
                    
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="uV")
                    ax.axvspan(t_gap_start, t_gap_end, color='gray', alpha=0.3, hatch='///')

                ax.set_ylabel(f"{band.capitalize()}\nCh Index")
                ax.set_title(f"Combined Activity")
                
                if i == n_rows - 1: ax.set_xlabel("Time (ms)")
                else: ax.set_xticklabels([])
                    
                if PLOT_CONFIG['x_limits']['pre'] and PLOT_CONFIG['x_limits']['post']:
                     ax.set_xlim(PLOT_CONFIG['x_limits']['pre'][0], PLOT_CONFIG['x_limits']['post'][1])

                # --- TRACES (Shared Y-Axis) ---
                rep_pre = d_pre[:, rep_ch_idx, :] if d_pre is not None else None
                rep_post = d_post[:, rep_ch_idx, :] if d_post is not None else None
                
                forced_ylim = PLOT_CONFIG['trace_scales'].get(band)
                
                if forced_ylim is not None:
                    y_range = (-forced_ylim, forced_ylim)
                else:
                    ylims = []
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if rep_pre is not None and np.any(np.isfinite(rep_pre)): 
                            ylims.extend([np.nanmin(rep_pre), np.nanmax(rep_pre)])
                        if rep_post is not None and np.any(np.isfinite(rep_post)):
                            ylims.extend([np.nanmin(rep_post), np.nanmax(rep_post)])
                    
                    ylims = [y for y in ylims if np.isfinite(y)]
                    if not ylims: y_range = (-100, 100)
                    else:
                         ymin, ymax = min(ylims), max(ylims)
                         if ymin == ymax: pad = 10.0
                         else: pad = (ymax - ymin) * 0.05
                         y_range = (ymin - pad, ymax + pad)
                
                t_alpha = PLOT_CONFIG['trace_alpha']

                # Pre Traces
                ax = axes[i, 1]
                if rep_pre is not None:
                    ax.plot(t_pre, rep_pre.T, color='gray', alpha=t_alpha, lw=0.5)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        rm = np.nanmean(rep_pre, axis=0)
                    ax.plot(t_pre, rm, color='blue', lw=1.5, label='Mean')
                ax.set_ylim(y_range)
                ax.set_title(f"Pre (Ch {rep_ch_idx})")
                ax.set_yticklabels([])
                
                if PLOT_CONFIG['x_limits']['pre'] is not None: ax.set_xlim(PLOT_CONFIG['x_limits']['pre'])
                elif t_pre is not None: ax.set_xlim(t_pre[0], t_pre[-1])
                
                if i == 0: ax.legend(loc='upper right', fontsize='x-small')
                
                # Post Traces
                ax = axes[i, 2]
                if rep_post is not None:
                    ax.plot(t_post, rep_post.T, color='gray', alpha=t_alpha, lw=0.5)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        rm = np.nanmean(rep_post, axis=0)
                    ax.plot(t_post, rm, color='red', lw=1.5, label='Mean')
                ax.set_ylim(y_range)
                ax.set_title(f"Post (Ch {rep_ch_idx})")
                ax.set_yticklabels([])
                
                if PLOT_CONFIG['x_limits']['post'] is not None: ax.set_xlim(PLOT_CONFIG['x_limits']['post'])
                elif t_post is not None: ax.set_xlim(t_post[0], t_post[-1])

            out_path = fig_dir / f"check_lfp_heatmap_{session}.png"
            try:
                 plt.savefig(out_path, dpi=150)
                 print(f"  Saved plot to {out_path}")
            except Exception as e:
                 print(f"  Error saving {out_path}: {e}")
            plt.close(fig)

        # --- Grab Broadband Data Single Time for Advanced Plots ---
        bb_pre = data.get("broadband_pre") # (N, Ch, T)
        bb_post = data.get("broadband_post")

        # --- Prepare Groups ---
        groups = []
        is_utah = (label == "Utah_Array")
        
        # --- Prepare Groups ---
        groups = []
        is_utah = (label == "Utah_Array")
        
        if is_utah:
            import csv
            
            # 1. Determine Port (A or B) AND Intan Session from Metadata
            ua_port = 'A' # Default
            intan_session = None
            try:
                parts = npz_file.stem.split('_')
                if parts[-1].isdigit():
                    br_idx = int(parts[-1])
                elif parts[-2].isdigit():
                    br_idx = int(parts[-2])
                else:
                    br_idx = None
                    
                if br_idx is not None and cfg.METADATA_CSV.exists():
                     with open(cfg.METADATA_CSV, 'r', newline='') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row.get('BR_File') and int(row['BR_File']) == br_idx:
                                p = row.get('UA_port', '').strip().upper()
                                if p in ['A', 'B']:
                                    ua_port = p
                                intan_session = row.get('Intan_Session', '').strip()
                                break
            except Exception as e:
                print(f"  [Warn] Failed to lookup metadata for port/intan_session: {e}")

            # 2. Load Mapping from Excel
            # The 'ua_ids_1based' in file might be missing or wrong, so we rebuild it using the master Excel map.
            try:
                xls_path = br_preproc.ua_excel_path(cfg.REPO_ROOT, cfg.PARAMS.probes)
                mapped_nsp = br_preproc.load_UA_mapping_from_excel(xls_path) # index=elec-1, val=nsp_id
                
                # Build Inverse Map: NSP_ID -> Elec_ID
                nsp_to_elec = {int(nsp): i+1 for i, nsp in enumerate(mapped_nsp) if nsp > 0}
                
                # 3. Map Channels to Regions
                # If Port A: Ch 0..127 -> NSP 1..128
                # If Port B: Ch 0..127 -> NSP 129..256
                offset = 128 if ua_port == 'B' else 0
                print(f"  [Info] Utah Array Port: {ua_port} (Offset: {offset})")
                
                sma_idxs, pmd_idxs, m1i_idxs, m1s_idxs = [], [], [], []
                
                for ch_idx in range(n_ch):
                    # Current NSP ID for this channel
                    curr_nsp = ch_idx + 1 + offset
                    
                    # Look up Electrode ID
                    elec_id = nsp_to_elec.get(curr_nsp, -1)
                    
                    if elec_id > 0:
                        if elec_id <= 64: sma_idxs.append(ch_idx)
                        elif elec_id <= 128: pmd_idxs.append(ch_idx)
                        elif elec_id <= 192: m1i_idxs.append(ch_idx)
                        elif elec_id <= 256: m1s_idxs.append(ch_idx)
                
                if sma_idxs: groups.append((sma_idxs, f"SMA (n={len(sma_idxs)})"))
                if pmd_idxs: groups.append((pmd_idxs, f"PMd (n={len(pmd_idxs)})"))
                if m1i_idxs: groups.append((m1i_idxs, f"M1 Inf (n={len(m1i_idxs)})"))
                if m1s_idxs: groups.append((m1s_idxs, f"M1 Sup (n={len(m1s_idxs)})"))
                
                if not groups:
                    print(f"  [Warn] No valid channel mappings found for Port {ua_port}. Check Excel map.")
                    is_utah = False

            except Exception as e:
                print(f"  [Warn] Failed to load/apply Excel mapping: {e}. Fallback to linear blocks.")
                is_utah = False

        if not is_utah:
            # Default 16-ch blocks (Intan)
            n_groups_default = 8
            ch_per_group = 16
            for g in range(n_groups_default):
                start = g * ch_per_group
                end = start + ch_per_group
                if start < n_ch:
                    actual_end = min(end, n_ch)
                    groups.append((range(start, actual_end), f"Ch {start}-{actual_end-1}"))

        # Setup Dynamic Grid
        n_plots = len(groups)
        n_cols = 2
        n_rows = int(np.ceil(n_plots / n_cols))
        figsize = (16, 3 * n_rows)

        # --- 2. Power Spectra (Overlay, Grouped) ---
        if FIGURE_CONFIG.get('process_power_spectra', False):
            print("  Generating Grouped Power Spectra...")
            # Dynamic Grid
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
            fig.suptitle(f"{session} - Power Spectra (Group Mean ± SEM)\n{title_str}", fontsize=14)
            axes_flat = axes.flatten()
            
            # Pre-calculate to avoid redundant calls if possible, but simplicity first.
            if bb_pre is not None:
                f_pre, psd_pre_all, sem_pre_all = calculate_psd(bb_pre, fs) # (n_ch, n_freq)
            else: psd_pre_all = None
                
            if bb_post is not None:
                f_post, psd_post_all, sem_post_all = calculate_psd(bb_post, fs)
            else: psd_post_all = None

            for i, (grp_idxs, grp_name) in enumerate(groups):
                if i >= len(axes_flat): break
                ax = axes_flat[i]
                
                # --- Pre ---
                if psd_pre_all is not None:
                    # Mean across group
                    g_psd = psd_pre_all[grp_idxs, :]
                    g_mean = np.nanmean(g_psd, axis=0) # (n_freq,)
                    ax.plot(f_pre, g_mean, color='blue', lw=2, label='Pre')

                # --- Post ---
                if psd_post_all is not None:
                    g_psd = psd_post_all[grp_idxs, :]
                    g_mean = np.nanmean(g_psd, axis=0)
                    ax.plot(f_post, g_mean, color='red', lw=2, label='Post')
                    
                add_band_annotations(ax)
                ax.set_title(grp_name)
                ax.set_xlim(0, 120)
                # Labels only on bottom/left edges of grid?
                # Simplify: Label all for clarity or just outer.
                ax.set_xlabel("Freq (Hz)")
                ax.set_ylabel("Power")
                if i == 0: ax.legend(loc='upper right', fontsize='small')
                ax.grid(True, alpha=0.3)
                
            # Hide unused axes
            for j in range(len(groups), len(axes_flat)):
                axes_flat[j].axis('off')

            out_path = fig_dir / f"check_lfp_psd_grouped_{session}.png"
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

            # --- 2b. PSD Ratio (Pre vs Post) ---
            print("  Generating Grouped PSD Ratio...")
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
            fig.suptitle(f"{session} - PSD Ratio 10*log10(Post/Pre)\n{title_str}", fontsize=14)
            axes_flat = axes.flatten()

            for i, (grp_idxs, grp_name) in enumerate(groups):
                if i >= len(axes_flat): break
                ax = axes_flat[i]
                
                if psd_pre_all is not None and psd_post_all is not None:
                     # Mean Group PSDs
                     g_pre = np.nanmean(psd_pre_all[grp_idxs, :], axis=0) # (F,)
                     g_post = np.nanmean(psd_post_all[grp_idxs, :], axis=0)
                     
                     # Ratio in dB
                     # Handle zeros
                     g_pre[g_pre == 0] = 1e-10
                     ratio_db = 10 * np.log10(g_post / g_pre)
                     
                     ax.plot(f_pre, ratio_db, color='k', lw=2)
                     ax.axhline(0, color='gray', linestyle='--')
                     
                add_band_annotations(ax)
                ax.set_title(grp_name)
                ax.set_xlim(0, 120)
                ax.set_xlabel("Freq (Hz)")
                ax.set_ylabel("dB Change")
                ax.grid(True, alpha=0.3)
                
            # Hide unused axes
            for j in range(len(groups), len(axes_flat)):
                axes_flat[j].axis('off')
            
            out_path = fig_dir / f"check_lfp_psd_ratio_{session}.png"
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

        # --- 3. LFP Coherence (Grouped: Channels vs Group Mean) ---
        if FIGURE_CONFIG.get('process_lfp_coherence', False):
            print("  Generating Grouped LFP Coherence...")

            fig, axes = plt.subplots(4, 2, figsize=(16, 16), constrained_layout=True)
            fig.suptitle(f"{session} - LFP Coherence (Channel vs Group Mean)\n{title_str}", fontsize=14)
            axes_flat = axes.flatten()

            # We need to calculate this per group.
            # Warning: This is computationally intensive (8 groups * 16 channels).
            # bb_post usually sufficient. User didn't specify Pre or Post.
            # Assuming Post-Stim is of interest.
            
            target_data = bb_post if bb_post is not None else bb_pre
            t_label = "Post-Stim" if bb_post is not None else "Pre-Stim"
            
            if target_data is not None:
                nperseg = min(target_data.shape[2], 256)
                
                for i, (grp_idxs, grp_name) in enumerate(groups):
                    if i >= len(axes_flat): break
                    ax = axes_flat[i]
                    
                    # 1. Get Group Mean Signal (n_trials, n_time)
                    # Use index slicing
                    # target_data: (N, Ch, T)
                    grp_data = target_data[:, grp_idxs, :]
                    grp_mean_sig = np.nanmean(grp_data, axis=1) # (N, T)
                    
                    # 2. Coherence of each channel in group vs grp_mean_sig
                    grp_coh_list = []
                    freqs = None
                    
                    for ch_idx in range(len(grp_idxs)):
                        # Single channel data
                        ch_sig = grp_data[:, ch_idx, :]
                        
                        f, Cxy = signal.coherence(ch_sig, grp_mean_sig, fs=fs, nperseg=nperseg, axis=1)
                        # Mean across trials
                        m_cxy = np.nanmean(Cxy, axis=0) # (n_freq,)
                        grp_coh_list.append(m_cxy)
                        if freqs is None: freqs = f
                    
                    coh_matrix = np.stack(grp_coh_list, axis=0) # (n_ch_in_grp, n_freq)
                    
                    # Plot
                    im = ax.imshow(coh_matrix, aspect='auto', origin='upper', cmap='inferno',
                                   extent=[freqs[0], freqs[-1], len(grp_idxs)-0.5, -0.5], vmin=0, vmax=1)
                    
                    # Bands
                    for b_start, b_end, b_col, _ in [(4, 13, 'green', 'A'), (13, 30, 'yellow', 'B'), (30, 60, 'orange', 'LG'), (60, 120, 'red', 'HG')]:
                        ax.axvline(b_start, color=b_col, linestyle='--', alpha=0.5)
                        ax.axvline(b_end, color=b_col, linestyle='--', alpha=0.5)

                    ax.set_title(f"{grp_name} ({t_label})")
                    ax.set_xlim(0, 120)
                    ax.set_yticks(range(0, len(grp_idxs), 4)) # Sparse ticks
                    # Correct labels to relative range 0-15? yes.
                    
                    if i >= 6: ax.set_xlabel("Freq (Hz)")
                    if i % 2 == 0: ax.set_ylabel("Ch (within group)")
            
            out_path = fig_dir / f"check_lfp_coherence_grouped_{session}.png"
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

        # --- 4. Time-Frequency ITPC (Grouped 4x2) ---
        if FIGURE_CONFIG.get('process_phase_coherence', False):
            print("  Generating Grouped T-F ITPC...")
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
            fig.suptitle(f"{session} - T-F ITPC (Post-Stim Group Mean)\n{title_str}", fontsize=14)
            axes_flat = axes.flatten()

            # Using Post-Stim for event response
            target_data = bb_post if bb_post is not None else bb_pre
            
            if target_data is not None:
                # Calculate ALL ITPC first? (n_ch, n_freq, n_time)
                # Calculating for 128 channels is acceptable.
                # Use slightly coarser STFT to be safe on memory/speed needed? 128/120 is fine.
                f, t, itpc_all = calculate_tf_itpc(target_data, fs, nperseg=128, noverlap=120)
                
                # Time vector
                t0 = data.get('rel_time_post')[0] if bb_post is not None else -500
                t_real = t * 1000.0 + t0

                for i, (grp_idxs, grp_name) in enumerate(groups):
                    if i >= len(axes_flat): break
                    ax = axes_flat[i]
                    
                    # Mean ITPC for this group
                    g_itpc = itpc_all[grp_idxs, :, :] # (n_grp, n_f, n_t)
                    g_mean_itpc = np.nanmean(g_itpc, axis=0) # (n_f, n_t)
                    
                    # Plot
                    im = ax.pcolormesh(t_real, f, g_mean_itpc, shading='gouraud', cmap='viridis', vmin=0, vmax=0.6) 
                    # lowered vmax slightly as means tend to be lower
                    
                    if i == 1: fig.colorbar(im, ax=ax, label="Mean ITPC")
                    
                    ax.set_title(grp_name)
                    ax.set_ylim(0, 120)
                    if i >= 6: ax.set_xlabel("Time (ms)")
                    if i % 2 == 0: ax.set_ylabel("Freq (Hz)")

            out_path = fig_dir / f"check_lfp_itpc_grouped_{session}.png"
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

        # --- 5. ERSP (Event-Related Spectral Perturbation) ---
        if FIGURE_CONFIG.get('process_ersp', False):
            print("  Generating Grouped ERSP...")
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
            fig.suptitle(f"{session} - ERSP (dB vs Pre-Stim Baseline)\n{title_str}", fontsize=14)
            axes_flat = axes.flatten()

            if bb_pre is not None and bb_post is not None:
                # Calculate ERSP for all channels
                # Use slightly coarser resolution for speed/memory if needed
                f_ersp, t_ersp_post, ersp_all = calculate_ersp(bb_pre, bb_post, fs, nperseg=128, noverlap=100)
                # ersp_all: (n_ch, n_freq, n_time)
                
                # Time vector (post)
                t0_post = data.get('rel_time_post')[0] if data.get('rel_time_post') is not None else 0
                t_real_post = t_ersp_post * 1000.0 + t0_post

                for i, (grp_idxs, grp_name) in enumerate(groups):
                    if i >= len(axes_flat): break
                    ax = axes_flat[i]
                    
                    # Mean ERSP for group
                    g_ersp = ersp_all[grp_idxs, :, :]
                    g_mean_ersp = np.nanmean(g_ersp, axis=0) # (F, T)
                    
                    # Plot Heatmap (RdBu_r centered at 0)
                    # Use vmin/vmax symmetric for dB
                    v_lim = 3.0 # +/- 3 dB
                    im = ax.pcolormesh(t_real_post, f_ersp, g_mean_ersp, shading='gouraud', cmap='RdBu_r', 
                                       vmin=-v_lim, vmax=v_lim)
                    
                    if i == 1: fig.colorbar(im, ax=ax, label="dB")
                    
                    ax.set_title(grp_name)
                    ax.set_ylim(0, 120)
                    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
                    ax.set_xlabel("Time (ms)")
                    ax.set_ylabel("Freq (Hz)")
            
            # Hide unused axes
            for j in range(len(groups), len(axes_flat)):
                axes_flat[j].axis('off')

            out_path = fig_dir / f"check_lfp_ersp_grouped_{session}.png"
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

        # --- 6. Band Power Stats (Bar Charts) ---
        if FIGURE_CONFIG.get('process_band_stats', False):
            print("  Generating Band Power Stats...")
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
            fig.suptitle(f"{session} - Band Power Change (Pre vs Post)\n{title_str}", fontsize=14)
            axes_flat = axes.flatten()
            
            if bb_pre is not None and bb_post is not None:
                # Calculate Stats
                stats_res = calculate_band_power_stats(bb_pre, bb_post, fs)
                
                # Bands to plot
                band_names = ['Alpha', 'Beta', 'Low Gamma', 'High Gamma']
                x_pos = np.arange(len(band_names))
                width = 0.35
                
                for i, (grp_idxs, grp_name) in enumerate(groups):
                    if i >= len(axes_flat): break
                    ax = axes_flat[i]
                    
                    # Collect data for this group
                    means_pre = []
                    means_post = []
                    errs_pre = []
                    errs_post = []
                    p_values = []
                    
                    for band in band_names:
                        res = stats_res.get(band)
                        if res is None:
                            means_pre.append(0); means_post.append(0)
                            errs_pre.append(0); errs_post.append(0)
                            p_values.append(1.0)
                            continue
                        
                        # Subset for group
                        g_mean_pre_chs = res['mean_pre'][grp_idxs]
                        g_mean_post_chs = res['mean_post'][grp_idxs]
                        
                        # Grand Mean across channels
                        gm_pre = np.nanmean(g_mean_pre_chs)
                        gm_post = np.nanmean(g_mean_post_chs)
                        
                        # Error? SEM across channels (Spatial variability)
                        g_sem_pre = np.nanstd(g_mean_pre_chs) / np.sqrt(len(grp_idxs))
                        g_sem_post = np.nanstd(g_mean_post_chs) / np.sqrt(len(grp_idxs))
                        
                        means_pre.append(gm_pre)
                        means_post.append(gm_post)
                        errs_pre.append(g_sem_pre)
                        errs_post.append(g_sem_post)
                        
                        # P-value: We did T-test per channel.
                        # How many channels were significant? Or perform T-test on the channel means?
                        # Let's do a Paired T-test on the CHANNEL MEANS for the group.
                        # i.e. Does the group consistently change?
                        t_grp, p_grp = stats.ttest_rel(g_mean_pre_chs, g_mean_post_chs, nan_policy='omit')
                        p_values.append(p_grp)

                    # Plot Bars
                    rects1 = ax.bar(x_pos - width/2, means_pre, width, yerr=errs_pre, label='Pre', color='blue', alpha=0.6, capsize=5)
                    rects2 = ax.bar(x_pos + width/2, means_post, width, yerr=errs_post, label='Post', color='red', alpha=0.6, capsize=5)
                    
                    # Add Significance Stars
                    max_y = max(max(means_pre), max(means_post)) * 1.1
                    for idx, p in enumerate(p_values):
                        if p < 0.05:
                            txt = '*' if p < 0.01 else '*'
                            # Height roughly above the bar
                            h = max(means_pre[idx] + errs_pre[idx], means_post[idx] + errs_post[idx])
                            ax.text(x_pos[idx], h * 1.05, txt, ha='center', va='bottom', fontsize=12, fontweight='bold')

                    ax.set_title(grp_name)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(band_names)
                    ax.set_ylabel("Power (uV^2/Hz)")
                    if i == 0: ax.legend()
                    ax.grid(axis='y', alpha=0.3)
            
            # Hide unused axes
            for j in range(len(groups), len(axes_flat)):
                axes_flat[j].axis('off')

            out_path = fig_dir / f"check_lfp_band_stats_{session}.png"
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

def plot_lfp_check():
    # 1. Process NPRW
    # process_directory(NPRW_LFP_DIR, NPRW_FIG_DIR, label="NPRW_Intan")
    
    # 2. Process Utah Array
    process_directory(UA_LFP_DIR, UA_FIG_DIR, label="Utah_Array")

if __name__ == "__main__":
    plot_lfp_check()
