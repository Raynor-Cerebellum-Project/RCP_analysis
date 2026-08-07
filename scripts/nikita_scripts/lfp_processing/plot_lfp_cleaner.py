"""
LFP Analysis Suite for Cerebellar Stimulation Experiment
=========================================================

Analyses included:
1. Session Summary PSD (1-120 Hz)
2. Raw Waveform Raster - See actual data across trials
3. Band Dynamics with proper 95% CI (SEM × 1.96)
4. Inter-Regional Coherence - Functional connectivity between arrays
5. Phase Lag Analysis - Information flow / lead-lag between regions
6. Dose-Response Analysis - Effect vs pulse count
7. Spatial GIF Animation - Activity spread over electrode grid
8. Stim vs Control Comparison

Key fixes from previous version:
- Frequency range extended to 120 Hz
- Confidence intervals now use SEM × 1.96 (not just SEM or SD)
- Removed unreliable peak detection
- Added meaningful inter-array analyses (coherence, phase)
- Uses Morlet wavelets for time-frequency analysis (no external dependencies)
"""

import warnings
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


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

from RCP_analysis.python.plotting.lfp_plots import (
    compute_confidence_interval,
    get_region_color,
    add_band_shading,
    add_stim_marker,
    get_trial_count,
    get_time_axis,
    get_broadband_full,
    slice_time_window,
    compute_array_median_raw_spectrograms,
    compute_control_median_baseline_norm,
    compute_self_baseline_norm_array_median_spectrograms,
    compute_control_spatial_baseline,
    generate_session_summary,
    generate_waveform_raster,
    generate_trial_heatmaps,
    generate_coherence_matrix,
    generate_stim_vs_control,
    generate_dose_response,
    render_spatial_grid,
    generate_spatial_gif,
    generate_spatial_heatmaps,
    generate_per_ua_spectrograms,
    generate_per_ua_array_median_spectrogram,
    run_wavelet_comparison,
    build_groups
)


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_session(data, session_id, fig_dir, groups, fs=1000,
                    data_control=None, session_data_dict=None,
                    nsp_to_elec=None, elec_to_idx=None, region_grids=None,
                    ua_ids_1based=None, control_median_baseline_norm=None,
                    control_self_baseline_norm_array_median=None,
                    control_spatial_baseline=None):
    """
    Run all enabled analyses for a session.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {session_id}")
    print(f"{'='*60}")
    
    figure_paths = []
    
    # 1. Session Summary (1-120 Hz PSD)
    if PLOT_CONFIG.get('generate_session_summary', True):
        print("  Generating session summary...")
        path = generate_session_summary(data, session_id, fig_dir, groups, fs)
        if path:
            figure_paths.append(path)
    
    # 2. Raw Waveform Raster
    if PLOT_CONFIG.get('generate_waveform_raster', True):
        print("  Generating waveform raster...")
        path = generate_waveform_raster(data, session_id, fig_dir, groups, fs)
        if path:
            figure_paths.append(path)
    
    # 4. Trial Heatmaps
    if PLOT_CONFIG.get('generate_trial_heatmaps', True):
        print("  Generating trial heatmaps...")
        path = generate_trial_heatmaps(data, session_id, fig_dir, groups, fs)
        if path:
            figure_paths.append(path)
    
    # 4b. Per-UA Spectrograms (8x8 grid per region)
    if PLOT_CONFIG.get('generate_per_ua_spectrograms', True):
        if elec_to_idx is not None and region_grids is not None:
            print("  Generating per-UA spectrogram grids (Morlet wavelets)...")
            default_trial = PLOT_CONFIG.get('per_ua_default_trial', 0)
            generate_all = PLOT_CONFIG.get('generate_all_trial_spectrograms', False)
            generate_avg = PLOT_CONFIG.get('generate_per_ua_trial_average', False)  # <-- ADD THIS
            path = generate_per_ua_spectrograms(
                data, session_id, fig_dir, groups, fs,
                ua_ids_1based=ua_ids_1based,
                nsp_to_elec=nsp_to_elec,
                region_grids=region_grids,
                elec_to_idx=elec_to_idx,
                trial_index=default_trial,
                generate_all_trials=generate_all,
                generate_trial_average=generate_avg,  # <-- ADD THIS
            )
            if path:
                figure_paths.append(path)
        else:
            print("  [Skip] Per-UA spectrograms require electrode mapping")


    # 4c. Per-UA array-level median spectrogram
    # Median across all trials and all channels/electrodes within each Utah array.
    if (PLOT_CONFIG.get('generate_per_ua_array_median', False) or
            PLOT_CONFIG.get('generate_control_median_baseline_norm_array_median', False) or
            PLOT_CONFIG.get('generate_self_baseline_minus_control_array_median', False)):
        if elec_to_idx is not None and region_grids is not None:
            print("  Generating per-UA array-level median spectrograms...")
            path = generate_per_ua_array_median_spectrogram(
                data=data,
                session_id=session_id,
                fig_dir=fig_dir,
                groups=groups,
                fs=fs,
                ua_ids_1based=ua_ids_1based,
                nsp_to_elec=nsp_to_elec,
                region_grids=region_grids,
                elec_to_idx=elec_to_idx,
                control_median_baseline_norm=control_median_baseline_norm,
                control_self_baseline_norm_array_median=control_self_baseline_norm_array_median,
            )
            if path:
                figure_paths.append(path)
        else:
            print("  [Skip] Per-UA array-level median spectrogram requires electrode mapping")
    
    
    # 5. Coherence Matrix
    if PLOT_CONFIG.get('generate_coherence_matrix', True):
        print("  Generating coherence matrix...")
        path = generate_coherence_matrix(data, session_id, fig_dir, groups, fs)
        if path:
            figure_paths.append(path)
    
    # 7. Stim vs Control
    if PLOT_CONFIG.get('generate_stim_vs_control', True) and data_control is not None:
        print("  Generating stim vs control comparison...")
        path = generate_stim_vs_control(data, data_control, session_id, fig_dir, groups, fs)
        if path:
            figure_paths.append(path)
    
    # 8. Dose Response
    if PLOT_CONFIG.get('generate_dose_response', True) and session_data_dict is not None:
        print("  Generating dose-response analysis...")
        path = generate_dose_response(session_data_dict, session_id, fig_dir, groups, fs)
        if path:
            figure_paths.append(path)
    
    # 9. Spatial visualizations (require electrode mapping)
    if (PLOT_CONFIG.get('generate_spatial_heatmaps', True) or 
        PLOT_CONFIG.get('generate_spatial_gif', True)):
        
        if elec_to_idx is not None and region_grids is not None:
            # Need to compute band power first
            try:
                bb_full, t_ms = get_broadband_full(data)
            except KeyError as e:
                print(f"    [Skip] No data for spatial visualizations: {e}")
                bb_full = None
            
            if bb_full is not None:          
                # Compute band power
                band_name = PLOT_CONFIG.get('selected_band', 'Beta')
                bands = ANALYSIS_CONFIG['bands']
                f_lo, f_hi = bands.get(band_name, (12, 25))
                baseline_win = ANALYSIS_CONFIG['baseline_window']
                
                nyq = fs / 2
                if f_hi >= nyq:
                    f_hi = nyq - 1
                
                try:
                    b, a = signal.butter(4, [f_lo/nyq, f_hi/nyq], btype='band')
                    
                    # Filter and compute envelope for all channels
                    filtered = signal.filtfilt(b, a, bb_full, axis=2)
                    envelope = np.abs(signal.hilbert(filtered, axis=2))
                    
                    # Baseline normalize
                    bl_mask = (t_ms >= baseline_win[0]) & (t_ms <= baseline_win[1])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        baseline_power = np.nanmean(envelope[:, :, bl_mask], axis=2, keepdims=True)
                        baseline_power[baseline_power < 1e-10] = 1e-10
                        band_dB = 10 * np.log10(envelope / baseline_power)
                    
                    # Generate normal spatial heatmaps
                    if PLOT_CONFIG.get('generate_spatial_heatmaps', True):
                        print("  Generating spatial heatmaps...")
                        generate_spatial_heatmaps(
                            band_dB,
                            t_ms,
                            ua_ids_1based,
                            session_id,
                            fig_dir,
                            groups,
                            elec_to_idx,
                            region_grids
                        )

                    # Generate CONTROL-NORM spatial heatmaps
                    if (PLOT_CONFIG.get('generate_control_norm_spatial_heatmaps', False)
                        and control_spatial_baseline is not None):

                        if control_spatial_baseline.shape[1] == envelope.shape[1]:
                            eps = 1e-12
                            band_dB_control = 10 * np.log10(
                                np.maximum(envelope, eps) / np.maximum(control_spatial_baseline, eps)
                            )

                            print("  Generating CONTROL-NORM spatial heatmaps...")
                            generate_spatial_heatmaps(
                                band_dB_control,
                                t_ms,
                                ua_ids_1based,
                                session_id + "_CONTROL_NORM",
                                fig_dir,
                                groups,
                                elec_to_idx,
                                region_grids
                            )
                        else:
                            print("    [Skip] Control spatial baseline channel mismatch")
                    
                    # Generate spatial GIF
                    if PLOT_CONFIG.get('generate_spatial_gif', True):
                        print("  Generating spatial GIF...")
                        generate_spatial_gif(band_dB, t_ms, ua_ids_1based, session_id,
                                            fig_dir, groups, nsp_to_elec, elec_to_idx, region_grids)
                
                except Exception as e:
                    print(f"    [Error] Spatial visualization failed: {e}")
        else:
            print("  [Skip] Spatial visualizations require electrode mapping")
    
    print(f"\n  Generated {len(figure_paths)} figure files")
    return figure_paths


# =============================================================================
# DIRECTORY PROCESSING
# =============================================================================

def process_directory(lfp_dir, fig_dir, label="LFP"):
    """Process all aligned LFP files in a directory, controls first."""
    from pathlib import Path
    
    if not lfp_dir.exists():
        print(f"Directory not found: {lfp_dir}")
        return

    files = sorted(lfp_dir.rglob("aligned_lfp__*.npz"))
    if not files:
        print(f"No aligned LFP npz files found in {lfp_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {label} ({len(files)} files found)")
    print(f"{'='*60}")
    
    # Show filter settings
    if SESSION_FILTER is not None:
        print(f"  Session filter (BR indices): {SESSION_FILTER}")
    if SESSION_ID_PATTERN is not None:
        print(f"  Session ID pattern: {SESSION_ID_PATTERN}")
    
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # PASS 1: Load and organize all sessions
    # =========================================================================
    print("\n--- Pass 1: Loading and organizing sessions ---")
    
    all_sessions = {}
    control_candidates_by_target = {}
    dose_response_cache = {}  # base_id -> {n_pulses: data}
    
    for npz_path in files:
        session_id = npz_path.stem.replace("aligned_lfp__", "")
        
        # Quick filter checks before loading
        if SESSION_ID_PATTERN is not None:
            if not re.search(SESSION_ID_PATTERN, session_id, re.IGNORECASE):
                continue
        
        if SESSION_FILTER is not None:
            br_match = re.search(r'_(\d{3})(?:_|$|\.)', session_id)
            if br_match:
                br_idx = int(br_match.group(1))
                if br_idx not in SESSION_FILTER:
                    continue
        
        print(f"  Loading {npz_path.name}...")
        
        try:
            data = dict(np.load(npz_path, allow_pickle=True))
        except Exception as e:
            print(f"    [Error] Failed to load: {e}")
            continue
        
        if 'broadband_full' not in data:
            print(f"    [Skip] No broadband_full data")
            continue
        
        # === EXTRACT TARGET (A or B) ===
        target_match = re.search(r'target_([AB])$', session_id, re.IGNORECASE)
        target = target_match.group(1).upper() if target_match else None
        
        # === DETERMINE IF CONTROL AND PULSE COUNT ===
        n_pulses = None
        is_control = False
        
        # Check metadata first
        if 'n_pulses' in data:
            n_pulses = int(data['n_pulses'])
        if 'is_control' in data:
            is_control = bool(data['is_control'])
        
        # Parse from filename if not in metadata
        if n_pulses is None:
            # Look for control/baseline indicators
            if ('control' in session_id.lower() or 
                'baseline' in session_id.lower() or
                '_0p_' in session_id or 
                '_0pulse' in session_id):
                n_pulses = 0
                is_control = True
            else:
                # Try to extract pulse count: _50p_, _100pulses, etc.
                pulse_match = re.search(r'_(\d+)p(?:ulse)?s?(?:_|$)', session_id, re.IGNORECASE)
                if pulse_match:
                    n_pulses = int(pulse_match.group(1))
                    is_control = (n_pulses == 0)
                else:
                    # Look up from aligned npz file matching this BR index to get stimulation duration
                    br_match = re.search(r'_(\d{3})(?:_|$|\.)', session_id)
                    if br_match:
                        br_idx = int(br_match.group(1))
                        try:
                            import RCP_analysis.python.functions.config_loading as cfg
                            import json
                            aligned_files = list(cfg.ALIGNED_CKPT_ROOT.rglob(f"*__BR_{br_idx:03d}.npz"))
                            if aligned_files:
                                aln_data = np.load(aligned_files[0], allow_pickle=True)
                                if 'align_meta' in aln_data:
                                    meta_raw = aln_data['align_meta'].item()
                                    meta = meta_raw if isinstance(meta_raw, dict) else json.loads(meta_raw)
                                    stim_dur = float(meta.get('recording_stim_dur', 0.0))
                                    if stim_dur > 0:
                                        n_pulses = int(round(stim_dur / 2.5))
                                        is_control = (n_pulses == 0)
                        except Exception as e:
                            print(f"    [Warn] Could not load stim duration from aligned file for BR {br_idx}: {e}")
                        
                        # Fallback to stim npz
                        if n_pulses is None:
                            try:
                                import RCP_analysis as rcp
                                import RCP_analysis.python.functions.config_loading as cfg
                                stim_npz_path, _ = rcp.stim_npz_path_from_br_idx(br_idx, cfg.METADATA_CSV, cfg.NPRW_AUX_DATA)
                                if stim_npz_path and stim_npz_path.exists():
                                    stim = rcp.load_stim_detection(stim_npz_path)
                                    block_bounds = stim.get("block_bounds_samples", [])
                                    if len(block_bounds) > 0:
                                        br2fs_intan = rcp.get_metadata_mapping(cfg.METADATA_ROOT / "br_to_intan_shifts.csv", 'br_idx', 'fs_intan')
                                        fs_intan = float(br2fs_intan.get(br_idx, 30000.0))
                                        block_durs = block_bounds[:, 1] - block_bounds[:, 0]
                                        median_dur_samples = np.median(block_durs)
                                        stim_dur_ms = median_dur_samples * 1000.0 / fs_intan
                                        n_pulses = stim_dur_ms / 2.5
                                        is_control = (n_pulses == 0)
                            except Exception as e:
                                pass
        
        # Extract base session ID (for dose-response grouping)
        base_id = re.sub(r'_\d+p(?:ulse)?s?(?:_|$)', '_', session_id, flags=re.IGNORECASE)
        base_id = re.sub(r'_control(?:_|$)', '_', base_id, flags=re.IGNORECASE)
        base_id = re.sub(r'_baseline(?:_|$)', '_', base_id, flags=re.IGNORECASE)
        base_id = base_id.rstrip('_')
        
        # Get sampling rate and channel info
        fs = int(data.get('fs_lfp', 1000))
        n_ch = data['broadband_full'].shape[1]
        
        # Build groups
        groups, nsp_to_elec, elec_to_idx, region_grids, ua_ids = build_groups(data, n_ch)
        
        session_info = {
            'npz_path': npz_path,
            'session_id': session_id,
            'base_id': base_id,
            'target': target,  # NEW: store target
            'data': data,
            'fs': fs,
            'groups': groups,
            'nsp_to_elec': nsp_to_elec,
            'elec_to_idx': elec_to_idx,
            'region_grids': region_grids,
            'ua_ids': ua_ids,
            'n_pulses': n_pulses,
            'is_control': is_control,
        }
        
        all_sessions[session_id] = session_info
        
        # === CACHE CONTROL CANDIDATES BY TARGET ===
        if is_control or n_pulses == 0:
            target_key = target if target is not None else '_default'
            n_trials = get_trial_count(data)

            control_candidates_by_target.setdefault(target_key, []).append({
                'data': data,
                'session_id': session_id,
                'n_trials': n_trials,
                'fs': fs,
                'groups': groups,
                'elec_to_idx': elec_to_idx,
                'region_grids': region_grids,
            })

            print(f"    -> Control/Baseline for target {target_key}, trials={n_trials}")
        else:
            print(f"    -> Stim session: {n_pulses if n_pulses else '?'} pulses, target {target}")
        
        # Build dose-response dict (group by target if available)
        dose_key = f"{base_id}_{target}" if target else base_id
        if dose_key not in dose_response_cache:
            dose_response_cache[dose_key] = {}
        if n_pulses is not None:
            dose_response_cache[dose_key][n_pulses] = data
    

    # =========================================================================
    # SELECT BEST CONTROL PER TARGET
    # =========================================================================
    control_data_by_target = {}
    control_median_baseline_norm_by_target = {}
    control_self_baseline_norm_array_median_by_target = {}
    control_spatial_baseline_by_target = {}

    for target_key, candidates in control_candidates_by_target.items():
        best = max(candidates, key=lambda x: x['n_trials'])

        control_data_by_target[target_key] = best['data']

        print(
            f"  Best control for target {target_key}: "
            f"{best['session_id']} ({best['n_trials']} trials)"
        )

        if PLOT_CONFIG.get('generate_control_median_baseline_norm_array_median', False):
            control_median_baseline_norm_by_target[target_key] = compute_control_median_baseline_norm(
                best['data'],
                best['groups'],
                fs=best['fs'],
                region_grids=best['region_grids'],
                elec_to_idx=best['elec_to_idx'],
            )   

        if PLOT_CONFIG.get('generate_self_baseline_minus_control_array_median', False):
            print(f"  Computing self-baseline-normalized control array medians for target {target_key}...")

            control_self_baseline_norm_array_median_by_target[target_key] = (
                compute_self_baseline_norm_array_median_spectrograms(
                    data=best['data'],
                    groups=best['groups'],
                    fs=best['fs'],
                    region_grids=best['region_grids'],
                    elec_to_idx=best['elec_to_idx'],
                    session_label=f"control target {target_key}",
                )
            )

        if PLOT_CONFIG.get('generate_control_norm_spatial_heatmaps', False):
            control_spatial_baseline_by_target[target_key] = compute_control_spatial_baseline(
                best['data'],
                fs=best['fs'],
            )

    
    # =========================================================================
    # PASS 2: Process sessions (controls already loaded)
    # =========================================================================
    print(f"\n--- Pass 2: Generating figures for {len(all_sessions)} sessions ---")
    
    # Show what controls we found
    print(f"  Controls found by target: {list(control_data_by_target.keys())}")
    
    processed_count = 0
    
    for session_id, info in all_sessions.items():
        target = info['target']

        target_key = target if target is not None else '_default'

        control_median_baseline_norm = control_median_baseline_norm_by_target.get(target_key)
        if control_median_baseline_norm is None:
            control_median_baseline_norm = control_median_baseline_norm_by_target.get('_default')

        control_self_baseline_norm_array_median = (
            control_self_baseline_norm_array_median_by_target.get(target_key)
        )
        if control_self_baseline_norm_array_median is None:
            control_self_baseline_norm_array_median = (
                control_self_baseline_norm_array_median_by_target.get('_default')
            )

        control_spatial_baseline = control_spatial_baseline_by_target.get(target_key)
        if control_spatial_baseline is None:
            control_spatial_baseline = control_spatial_baseline_by_target.get('_default')

        if info['is_control']:
            control_median_baseline_norm_for_session = None
            control_self_baseline_norm_array_median_for_session = None
        else:
            control_median_baseline_norm_for_session = control_median_baseline_norm
            control_self_baseline_norm_array_median_for_session = control_self_baseline_norm_array_median
        
        # === GET CONTROL DATA FOR THIS SESSION'S TARGET ===
        data_control = None
        if not info['is_control']:
            # Try to get control for this target
            if target and target in control_data_by_target:
                data_control = control_data_by_target[target]
            elif '_default' in control_data_by_target:
                data_control = control_data_by_target['_default']
            
            if data_control is not None:
                print(f"  Matched control for {session_id} (target {target})")
        
        # Get dose-response dict (only if multiple conditions exist)
        dose_key = f"{info['base_id']}_{target}" if target else info['base_id']
        session_data_dict = None
        if dose_key in dose_response_cache and len(dose_response_cache[dose_key]) > 1:
            session_data_dict = dose_response_cache[dose_key]
        
        # Process this session
        process_session(
            data=info['data'],
            session_id=info['session_id'],
            fig_dir=fig_dir,
            groups=info['groups'],
            fs=info['fs'],
            data_control=data_control,
            session_data_dict=session_data_dict,
            nsp_to_elec=info['nsp_to_elec'],
            elec_to_idx=info['elec_to_idx'],
            region_grids=info['region_grids'],
            ua_ids_1based=info['ua_ids'],
            control_median_baseline_norm=control_median_baseline_norm_for_session,
            control_self_baseline_norm_array_median=control_self_baseline_norm_array_median_for_session,
            control_spatial_baseline=control_spatial_baseline,
        )
        
        processed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Done processing {label}")
    print(f"  Processed: {processed_count} sessions")
    print(f"{'='*60}")


if __name__ == "__main__":
    import RCP_analysis.python.functions.config_loading as cfg
    
    # Define directories
    UA_LFP_DIR = cfg.OUT_BASE / "checkpoints" / "UA_LFP"
    NPRW_LFP_DIR = cfg.OUT_BASE / "checkpoints" / "NPRW_LFP"
    
    UA_FIG_DIR = cfg.OUT_BASE / "figures" / "UA_LFP"
    NPRW_FIG_DIR = cfg.OUT_BASE / "figures" / "NPRW_LFP"

    # -------------------------------------------------------------------------
    # WAVELET COMPARISON MODE
    # If WAVELET_TEST_CONFIG['enabled'] is True, run the wavelet comparison
    # instead of (or in addition to) the normal pipeline.
    # Edit WAVELET_TEST_CONFIG at the top of this file to configure.
    # -------------------------------------------------------------------------
    if WAVELET_TEST_CONFIG.get('enabled', False):
        print("\n[WAVELET TEST MODE ACTIVE]")
        run_wavelet_comparison(UA_LFP_DIR, UA_FIG_DIR, label="Utah_Array")
        run_wavelet_comparison(NPRW_LFP_DIR, NPRW_FIG_DIR, label="NPRW_Intan")
    else:
        # Normal pipeline
        process_directory(UA_LFP_DIR, UA_FIG_DIR, label="Utah_Array")
        process_directory(NPRW_LFP_DIR, NPRW_FIG_DIR, label="NPRW_Intan")