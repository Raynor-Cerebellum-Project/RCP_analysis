import gc, csv
import numpy as np
import warnings
import spikeinterface.extractors as se
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

warnings.filterwarnings("ignore")

def main():
    """
    Multi-Channel PPM Test
    Tests a single PPM correction value across multiple channels simultaneously.
    Shows first artifact onset (-0.5 to 3 ms) for each channel in a grid layout.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    TEST_PPM = -13.951  # Single PPM value to test
    
    # Representative electrodes (4 per array, spatially distributed)
    # Port B electrodes (NSP channels 129-256)
    ELECTRODES_PORT_B = {
        'SMA (Array 1)': [1, 11, 37, 64],           # corners + center
        'Dorsal PM (Array 2)': [65, 84, 103, 120],  # corners + center  
        'M1 Inferior (Array 3)': [137, 155, 174, 192],  # corners + center
        'M1 Superior (Array 4)': [193, 211, 237, 251]   # corners + center
    }
    
    # Port A electrodes (NSP channels 1-128)
    ELECTRODES_PORT_A = {
        'SMA (Array 1)': [5, 15, 29, 60],           # corners + center
        'Dorsal PM (Array 2)': [70, 77, 95, 128],   # corners + center
        'M1 Inferior (Array 3)': [129, 147, 165, 186],  # corners + center
        'M1 Superior (Array 4)': [196, 214, 233, 256]   # corners + center
    }
    
    MAX_DEBUG_TRIALS = 27  # How many stim blocks to show per channel
    WINDOW_MS = (-0.5, 3.0)  # Time window around artifact onset
    
    SHIFT_CSV = METADATA_ROOT / "br_to_intan_shifts.csv"
    BR_SESSION_FOLDERS = rcp.list_br_sessions(BR_ROOT)
    PROCESS_ONLY = PARAMS.preprocessing.get("process_only")
    # ═══════════════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print(f"MULTI-CHANNEL PPM TEST")
    print(f"Testing PPM = {TEST_PPM:+.3f}")
    print(f"{'='*70}\n")
    
    # Filter sessions if needed
    if PROCESS_ONLY:
        sess_folders = []
        for br_idx in PROCESS_ONLY:
            match = [s for s in BR_SESSION_FOLDERS if int(s.name.split("_")[-1]) == br_idx]
            sess_folders.extend(match)
        print(f"Processing {len(sess_folders)} sessions: {PROCESS_ONLY}")
    else:
        sess_folders = BR_SESSION_FOLDERS
    
    # Load UA mapping
    XLS = rcp.ua_excel_path(REPO_ROOT, PARAMS.probes)
    UA_MAP = rcp.load_UA_mapping_from_excel(XLS) if XLS else None
    if UA_MAP is None:
        raise RuntimeError("UA mapping required")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Process each session
    # ═══════════════════════════════════════════════════════════════════════════
    for sess in sess_folders:
        print(f"\n{'='*70}")
        print(f"Session: {sess.name}")
        print(f"{'='*70}")
        
        try:
            # Load neural data
            rec_ns6 = se.read_blackrock(sess, stream_name='nsx6', all_annotations=True)
            br_idx = int(sess.name.split('_')[-1])
            
            # Apply channel mapping
            rec_ns6, idx_rows, ua_elec, ua_nsp, ua_region, ua_region_names, ua_port = \
                rcp.apply_ua_mapping_with_regions(rec_ns6, UA_MAP, br_idx, METADATA_CSV)
            
            fs_ua = rec_ns6.get_sampling_frequency()
            n_total = rec_ns6.get_num_samples()
            
            print(f"  Sampling rate: {fs_ua} Hz")
            print(f"  Duration: {n_total / fs_ua:.1f} s")
            print(f"  Port: {ua_port}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to load: {e}")
            continue
        
        # ───────────────────────────────────────────────────────────────────────
        # Select electrodes based on port
        # ───────────────────────────────────────────────────────────────────────
        if ua_port == 'B':
            ELECTRODES_TO_PLOT = ELECTRODES_PORT_B
            print(f"  Using Port B electrode set")
        elif ua_port == 'A':
            ELECTRODES_TO_PLOT = ELECTRODES_PORT_A
            print(f"  Using Port A electrode set")
        else:
            print(f"  [WARN] Unknown port '{ua_port}', defaulting to Port B")
            ELECTRODES_TO_PLOT = ELECTRODES_PORT_B
        
        # Flatten electrode list for easier handling
        all_electrodes = []
        electrode_labels = []
        for region, elecs in ELECTRODES_TO_PLOT.items():
            for elec in elecs:
                all_electrodes.append(elec)
                electrode_labels.append(f"{region}\nElec {elec}")
        
        print(f"  Testing {len(all_electrodes)} electrodes across 4 arrays")
        
        # ───────────────────────────────────────────────────────────────────────
        # Get stim timing from Intan
        # ───────────────────────────────────────────────────────────────────────
        stim_npz_path, intan_session_name = rcp.stim_npz_path_from_br_idx(
            br_idx, METADATA_CSV, NPRW_AUX_DATA
        )
        
        if not stim_npz_path or not stim_npz_path.exists():
            print(f"  [SKIP] No stim data found")
            continue
        
        stim = rcp.load_stim_detection(stim_npz_path)
        block_bounds = stim.get("block_bounds_samples", [])
        
        if not block_bounds.size:
            print(f"  [SKIP] No stim blocks found")
            continue
        
        print(f"  Found {len(block_bounds)} stim blocks")
        
        # ───────────────────────────────────────────────────────────────────────
        # Get alignment parameters
        # ───────────────────────────────────────────────────────────────────────
        br2fs_intan = rcp.get_metadata_mapping(SHIFT_CSV, 'br_idx', 'fs_intan')
        br2shift_samp_intan = rcp.get_metadata_mapping(SHIFT_CSV, 'br_idx', 'shift_sample')
        
        fs_intan = float(br2fs_intan.get(br_idx, fs_ua))
        shift_raw = br2shift_samp_intan.get(br_idx, None)
        
        if shift_raw is None or (isinstance(shift_raw, str) and shift_raw.strip() == ""):
            print(f"  [SKIP] No alignment shift found")
            continue
        
        shift_samp_intan = float(shift_raw)
        starts_intan = block_bounds[:, 0].astype(np.int64)
        ends_intan = block_bounds[:, 1].astype(np.int64)
        
        scale_nominal = fs_ua / fs_intan
        scale_corrected = scale_nominal * (1.0 + TEST_PPM / 1e6)
        
        print(f"  Nominal scale:   {scale_nominal:.12f}")
        print(f"  Corrected scale: {scale_corrected:.12f} (PPM = {TEST_PPM:+.3f})")
        
        # Convert Intan times to Blackrock times
        starts_ua = np.round((starts_intan - shift_samp_intan) * scale_corrected).astype(np.int64)
        ends_ua = np.round((ends_intan - shift_samp_intan) * scale_corrected).astype(np.int64)
        
        # Validate
        ends_ua = np.minimum(ends_ua, n_total)
        valid = (ends_ua > starts_ua) & (starts_ua >= 0) & (ends_ua <= n_total)
        starts_ua = starts_ua[valid]
        ends_ua = ends_ua[valid]
        
        if not starts_ua.size:
            print(f"  [SKIP] No valid artifact windows")
            continue
        
        # ───────────────────────────────────────────────────────────────────────
        # Find channel IDs for target electrodes
        # ───────────────────────────────────────────────────────────────────────
        channel_map = {}  # elec_id -> (ch_idx, ch_name)
        for ci, elec_id in enumerate(ua_elec):
            if int(elec_id) in all_electrodes:
                ch_name = rec_ns6.get_channel_ids()[ci]
                channel_map[int(elec_id)] = (ci, ch_name)
        
        if not channel_map:
            print(f"  [SKIP] None of the target electrodes found in mapping")
            continue
        
        print(f"  Found {len(channel_map)}/{len(all_electrodes)} target electrodes")
        
        # ───────────────────────────────────────────────────────────────────────
        # Create multi-channel grid figure
        # ───────────────────────────────────────────────────────────────────────
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        n_trials = min(len(starts_ua), MAX_DEBUG_TRIALS)
        n_channels = len(all_electrodes)
        
        # Create figure with GridSpec for better control
        fig = plt.figure(figsize=(3.5 * n_channels, 1.5 * n_trials))
        gs = GridSpec(n_trials, n_channels, figure=fig, hspace=0.3, wspace=0.4)
        
        # Time window
        win_samp_pre = int(abs(WINDOW_MS[0]) * fs_ua / 1000.0)
        win_samp_post = int(WINDOW_MS[1] * fs_ua / 1000.0)
        time_axis = np.linspace(WINDOW_MS[0], WINDOW_MS[1], win_samp_pre + win_samp_post)
        
        # Plot each channel × trial combination
        for ch_idx, (elec_id, label) in enumerate(zip(all_electrodes, electrode_labels)):
            if elec_id not in channel_map:
                # Channel not found - leave blank
                for trial_idx in range(n_trials):
                    ax = fig.add_subplot(gs[trial_idx, ch_idx])
                    ax.text(0.5, 0.5, 'Not found', ha='center', va='center', 
                           fontsize=8, color='gray')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    if trial_idx == 0:
                        ax.set_title(label, fontsize=8, pad=3)
                continue
            
            _, ch_name = channel_map[elec_id]
            
            for trial_idx, s_anchor in enumerate(starts_ua[:n_trials]):
                ax = fig.add_subplot(gs[trial_idx, ch_idx])
                
                # Extract trace
                s0 = int(s_anchor) - win_samp_pre
                s1 = int(s_anchor) + win_samp_post
                s0 = max(0, s0)
                s1 = min(n_total, s1)
                n_samp = s1 - s0
                t_ax = time_axis[:n_samp]
                
                raw_trace = rec_ns6.get_traces(
                    start_frame=s0, end_frame=s1,
                    channel_ids=[ch_name], return_in_uV=True
                ).flatten()
                
                # Plot
                ax.plot(t_ax, raw_trace, color='k', lw=0.7, alpha=0.8)
                ax.axvline(0, color='r', linestyle='--', alpha=0.6, lw=1.2)
                
                # Formatting
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.2, linestyle=':')
                
                # Labels
                if trial_idx == 0:
                    ax.set_title(label, fontsize=8, pad=3)
                if ch_idx == 0:
                    ax.set_ylabel(f"Trial {trial_idx+1}\n(µV)", fontsize=7)
                if trial_idx == n_trials - 1:
                    ax.set_xlabel("Time (ms)", fontsize=7)
                else:
                    ax.set_xticklabels([])
                
                # Make plots compact
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
        # Overall title
        fig.suptitle(
            f"PPM = {TEST_PPM:+.3f} | BR {br_idx:03d} | Port {ua_port} | "
            f"Window: [{WINDOW_MS[0]}, {WINDOW_MS[1]}] ms",
            fontsize=11, fontweight='bold', y=0.995
        )
        
        # Save
        out_dir = UA_CKPT_ROOT.parent.parent / "figures" / "ppm_multichannel"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fig = out_dir / f"PPM_{TEST_PPM:+.3f}_br{br_idx:03d}_Port{ua_port}_multichannel.png"
        
        fig.savefig(out_fig, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Saved: {out_fig.name}")
        
        # Cleanup
        del rec_ns6
        gc.collect()

    print(f"\n{'='*70}")
    print("MULTI-CHANNEL TEST COMPLETE!")
    print(f"Output: {UA_CKPT_ROOT.parent.parent / 'figures' / 'ppm_multichannel'}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()