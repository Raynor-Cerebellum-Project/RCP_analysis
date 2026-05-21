import gc, csv
import numpy as np
import warnings
import spikeinterface.extractors as se
import RCP_analysis as rcp
from RCP_analysis.python.functions.config_loading import *

warnings.filterwarnings("ignore")

def main():
    """
    PPM Sweep Debug Figure Generator
    Quickly generates alignment debug figures for different PPM correction values.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════════
    PPM_VALUES = np.arange(-10.0, -15.0, -1.0)  # Test PPM from 5 to 25 in steps of 1
    DEBUG_ELECTRODE = 156  # Which electrode to plot
    MAX_DEBUG_BLOCKS = 20  # How many stim blocks to show per figure
    
    SHIFT_CSV = METADATA_ROOT / "br_to_intan_shifts.csv"
    BR_SESSION_FOLDERS = rcp.list_br_sessions(BR_ROOT)
    
    PROCESS_ONLY = PARAMS.preprocessing.get("process_only")
    # ═══════════════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print(f"PPM SWEEP DEBUG FIGURE GENERATOR")
    print(f"Testing {len(PPM_VALUES)} PPM values: {PPM_VALUES[0]:.1f} to {PPM_VALUES[-1]:.1f}")
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
            # Load only the neural data (skip aux streams for speed)
            rec_ns6 = se.read_blackrock(sess, stream_name='nsx6', all_annotations=True)
            br_idx = int(sess.name.split('_')[-1])
            
            # Apply channel mapping
            rec_ns6, idx_rows, ua_elec, ua_nsp, ua_region, ua_region_names, ua_port = \
                rcp.apply_ua_mapping_with_regions(rec_ns6, UA_MAP, br_idx, METADATA_CSV)
            
            fs_ua = rec_ns6.get_sampling_frequency()
            n_total = rec_ns6.get_num_samples()
            
            print(f"  Sampling rate: {fs_ua} Hz")
            print(f"  Duration: {n_total / fs_ua:.1f} s")
            
        except Exception as e:
            print(f"  [ERROR] Failed to load: {e}")
            continue
        
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
        print(f"  Nominal scale: {scale_nominal:.12f}")
        
        # ───────────────────────────────────────────────────────────────────────
        # Find the target electrode
        # ───────────────────────────────────────────────────────────────────────
        ch_si_id = None
        for ci, elec_id in enumerate(ua_elec):
            if int(elec_id) == DEBUG_ELECTRODE:
                ch_si_id = rec_ns6.get_channel_ids()[ci]
                break
        
        if ch_si_id is None:
            print(f"  [SKIP] Electrode {DEBUG_ELECTRODE} not found in mapping")
            continue
        
        print(f"  Plotting electrode {DEBUG_ELECTRODE} (channel: {ch_si_id})")
        
        # ───────────────────────────────────────────────────────────────────────
        # PPM SWEEP: Generate one figure per PPM value
        # ───────────────────────────────────────────────────────────────────────
        import matplotlib.pyplot as plt
        
        out_dir = UA_CKPT_ROOT.parent.parent / "figures" / "ppm_sweep" / f"br{br_idx:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for ppm_val in PPM_VALUES:
            scale_corrected = scale_nominal * (1.0 + ppm_val / 1e6)
            
            # Convert Intan times to Blackrock times
            starts_ua = np.round((starts_intan - shift_samp_intan) * scale_corrected).astype(np.int64)
            ends_ua = np.round((ends_intan - shift_samp_intan) * scale_corrected).astype(np.int64)
            
            # Validate
            ends_ua = np.minimum(ends_ua, n_total)
            valid = (ends_ua > starts_ua) & (starts_ua >= 0) & (ends_ua <= n_total)
            starts_ua = starts_ua[valid]
            ends_ua = ends_ua[valid]
            
            if not starts_ua.size:
                continue
            
            # ───────────────────────────────────────────────────────────────────
            # Create figure
            # ───────────────────────────────────────────────────────────────────
            n_blocks = min(len(starts_ua), MAX_DEBUG_BLOCKS)
            fig, axes = plt.subplots(n_blocks, 1, figsize=(14, 1.8 * n_blocks))
            if n_blocks == 1:
                axes = [axes]
            
            # Time window around each stim onset
            win_samp_pre = int(5.0 * fs_ua / 1000.0)   # 5 ms before
            win_samp_post = int(20.0 * fs_ua / 1000.0)  # 20 ms after
            time_axis = np.linspace(-5.0, 20.0, win_samp_pre + win_samp_post)
            
            for b_idx, s_anchor in enumerate(starts_ua[:n_blocks]):
                s0 = int(s_anchor) - win_samp_pre
                s1 = int(s_anchor) + win_samp_post
                s0 = max(0, s0)
                s1 = min(n_total, s1)
                n_samp = s1 - s0
                t_ax = time_axis[:n_samp]
                
                # Extract raw trace
                raw_trace = rec_ns6.get_traces(
                    start_frame=s0, end_frame=s1,
                    channel_ids=[ch_si_id], return_in_uV=True
                ).flatten()
                
                # Plot
                axes[b_idx].plot(t_ax, raw_trace, color='k', lw=0.8, alpha=0.8)
                axes[b_idx].axvline(0, color='r', linestyle='--', alpha=0.7, lw=1.5, 
                                   label='Predicted onset' if b_idx == 0 else '')
                axes[b_idx].set_ylabel(f"Block {b_idx+1}\n(µV)", fontsize=7)
                axes[b_idx].grid(True, alpha=0.2, linestyle=':')
                axes[b_idx].tick_params(labelsize=7)
                
                # Title on first subplot
                if b_idx == 0:
                    axes[b_idx].set_title(
                        f"PPM = {ppm_val:+.2f} | Electrode {DEBUG_ELECTRODE} | BR {br_idx:03d} | "
                        f"Scale: {scale_corrected:.12f}",
                        fontsize=9, fontweight='bold'
                    )
                    axes[b_idx].legend(loc='upper right', fontsize=7)
            
            axes[-1].set_xlabel("Time relative to predicted stim onset (ms)", fontsize=8)
            fig.tight_layout()
            
            # Save
            out_fig = out_dir / f"PPM_{ppm_val:+06.2f}.png"
            fig.savefig(out_fig, dpi=120, bbox_inches='tight')
            plt.close(fig)
            
            print(f"    ✓ PPM {ppm_val:+6.2f} → {out_fig.name}")
        
        # Cleanup
        del rec_ns6
        gc.collect()
        
        print(f"\n  Output: {out_dir}")
        print(f"  Generated {len(PPM_VALUES)} figures for visual comparison")

    print(f"\n{'='*70}")
    print("SWEEP COMPLETE!")
    print(f"Check: {UA_CKPT_OUT.parent.parent / 'figures' / 'ppm_sweep'}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()