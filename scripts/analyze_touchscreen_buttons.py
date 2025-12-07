
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import RCP_analysis as rcp

# ---- CONFIG ----
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE  = SESSION_LOC / "results"
ALIGNED_ROOT = OUT_BASE / "checkpoints" / "Aligned"
EVENTS_DIR   = OUT_BASE / "events"
FIGURES_OUT  = OUT_BASE / "figures" / "button_analysis"
FIGURES_OUT.mkdir(parents=True, exist_ok=True)

# Window for plotting around button press
WIN_SEC = (-0.5, 1.5)
BASELINE_WIN = (-0.5, -0.1) # For normalization
FS_ANALYSIS = 100.0 # Aligned data is usually 100Hz? Need to verify file.

def load_aligned_data(fpath):
    # fpath is a Path object
    if not fpath.exists():
        print(f"[WARN] Aligned file not found: {fpath}")
        return None
    
    print(f"  > Loading aligned data: {fpath.name}")
    with np.load(fpath, allow_pickle=True) as z:
        # Inspect keys
        # Expected: 'rates_ua', 'rates_intan', 'beh_cam1', 't_vec'
        data = dict(z)
    return data

def load_events(session_name):
    # Prefer kinematic events if available, else touchscreen
    p_kin = EVENTS_DIR / f"{session_name}_kinematic_events.npz"
    p_touch = EVENTS_DIR / f"{session_name}_touchscreen_events.npz"
    
    path = p_kin if p_kin.exists() else p_touch
    if not path.exists():
        print(f"[WARN] No event file found for {session_name}")
        return None
        
    print(f"  > Loading events: {path.name}")
    with np.load(path, allow_pickle=True) as z:
        events = z['events'] # Structured array
        fs_events = float(z['fs'])
        
    return events, fs_events

def extract_epochs(data_continuous, trigger_indices, win_samples):
    """
    Extract epochs from (N_time, N_ch) data.
    Returns (N_trials, N_ch, N_time)
    """
    n_trigs = len(trigger_indices)
    n_time, n_ch = data_continuous.shape
    win_len = win_samples[1] - win_samples[0]
    
    out = np.full((n_trigs, n_ch, win_len), np.nan)
    
    for i, idx in enumerate(trigger_indices):
        t0 = idx + win_samples[0]
        t1 = idx + win_samples[1]
        
        # Check bounds
        if t0 < 0 or t1 > n_time:
            # Partial fill?
            pad_pre = 0
            pad_post = 0
            
            sel_t0 = max(0, t0)
            sel_t1 = min(n_time, t1)
            
            chunk = data_continuous[sel_t0:sel_t1, :]
            
            # Place in out
            out_start = sel_t0 - t0
            out_end = out_start + (sel_t1 - sel_t0)
            
            if out_end > out_start:
                 out[i, :, out_start:out_end] = chunk.T
        else:
            out[i, :, :] = data_continuous[t0:t1, :].T
            
    return out

def get_channel_depths_intan():
    # Placeholder: assume linear spacing or load geometry
    # If not available, just return indices
    return None

def plot_session(sess_name, aligned, events, fs_events):
    # Unpack Data
    # Rates are usually (time, ch)
    ua = aligned.get('rates_ua') # Check shape
    intan = aligned.get('rates_nprw') # or 'rates_intan'
    if intan is None: intan = aligned.get('rates_intan')
    
    beh = aligned.get('beh_cam1') # (time, cols)
    
    # Check FS
    # Aligned data is usually resampled to specific rate (e.g. 100Hz or 1kHz)
    # The 't_vec' might help
    t_vec = aligned.get('t_vec')
    fs_aligned = 100.0 # Default assumption
    if t_vec is not None and len(t_vec) > 1:
        fs_aligned = 1.0 / np.median(np.diff(t_vec))
        
    # Events are in their own FS. We need to convert event Secs to Aligned Samples.
    # Event structure has 'start_sec'
    event_secs = events['start_sec']
    event_types = events['type']
    
    # Align
    # Convert secs to aligned indices
    # index = sec * fs_aligned
    event_idxs = (event_secs * fs_aligned).astype(int)
    
    # Separate A and B
    mask_a = event_types == 'A'
    mask_b = event_types == 'B'
    
    idxs_a = event_idxs[mask_a]
    idxs_b = event_idxs[mask_b]
    
    if len(idxs_a) == 0 and len(idxs_b) == 0:
        print("  [SKIP] No A or B events found.")
        return

    # Dimensions
    win_samp = (int(WIN_SEC[0] * fs_aligned), int(WIN_SEC[1] * fs_aligned))
    n_pts = win_samp[1] - win_samp[0]
    t_axis = np.linspace(WIN_SEC[0], WIN_SEC[1], n_pts)
    
    # Setup Figure
    # 2 Columns (A, B), 3 Rows (Kin, Intan, UA)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True, constrained_layout=True)
    fig.suptitle(f"{sess_name} - Button Aligned")
    
    # Names
    titles = ["Button A", "Button B"]
    event_sets = [idxs_a, idxs_b]
    
    for col, (ev_idxs, title) in enumerate(zip(event_sets, titles)):
        ax_kin = axes[0, col]
        ax_int = axes[1, col]
        ax_ua  = axes[2, col]
        
        ax_kin.set_title(f"{title} (n={len(ev_idxs)})")
        if len(ev_idxs) < 2:
            continue
            
        # 1. KINEMATICS
        # Look for middle_x, middle_y
        # If beh columns not labeled, assume last 2/3? Or check params?
        # For now, plot all columns or PCA?
        # User said "middle" was used for detection.
        # Let's plot indices 27, 28 if raw, or search headers if we knew them.
        # But 'beh_cam1' in aligned npz is likely a numpy array.
        # We will plot the first few valid columns or mean speed?
        # Let's plot euclidean speed if possible, or just raw X/Y.
        if beh is not None:
             # Extract
             beh_epochs = extract_epochs(beh, ev_idxs, win_samp) # (N, Ch, T)
             # Plot Median + IQR of 'Speed' or just X coordinate?
             # Let's guess X/Y are present.
             # Just plot all traces in gray and median in red?
             # Plotting 1st coordinate (X)
             x_data = beh_epochs[:, 27, :] if beh_epochs.shape[1] > 27 else beh_epochs[:, 0, :]
             
             # Normalize Baseline?
             bl_idx = (t_axis >= BASELINE_WIN[0]) & (t_axis <= BASELINE_WIN[1])
             means = np.nanmean(x_data, axis=0)
             sems  = np.nanstd(x_data, axis=0) / np.sqrt(x_data.shape[0])
             
             ax_kin.plot(t_axis, means, color='k')
             ax_kin.fill_between(t_axis, means-sems, means+sems, color='k', alpha=0.2)
             ax_kin.set_ylabel("Kinematics (X)")
             
        # 2. INTAN
        if intan is not None:
             intan_epochs = extract_epochs(intan, ev_idxs, win_samp) # (N, Ch, T)
             # Average across trials -> (Ch, T)
             psth = np.nanmean(intan_epochs, axis=0)
             
             # Vmin/Vmax
             vm = np.nanpercentile(np.abs(psth), 99)
             im = ax_int.imshow(psth, aspect='auto', origin='lower', 
                                extent=[WIN_SEC[0], WIN_SEC[1], 0, psth.shape[0]],
                                vmin=-vm, vmax=vm, cmap='bwr')
             ax_int.set_ylabel("Intan Channels")
             fig.colorbar(im, ax=ax_int, fraction=0.046, pad=0.04)

        # 3. UA
        if ua is not None:
             ua_epochs = extract_epochs(ua, ev_idxs, win_samp)
             psth_ua = np.nanmean(ua_epochs, axis=0)
             
             vm = np.nanpercentile(np.abs(psth_ua), 99)
             im = ax_ua.imshow(psth_ua, aspect='auto', origin='lower',
                               extent=[WIN_SEC[0], WIN_SEC[1], 0, psth_ua.shape[0]],
                               vmin=-vm, vmax=vm, cmap='viridis')
             ax_ua.set_ylabel("UA Channels")
             fig.colorbar(im, ax=ax_ua, fraction=0.046, pad=0.04)
        
        ax_ua.set_xlabel("Time from Press (s)")
        
    # Save
    out_path = FIGURES_OUT / f"{sess_name}_button_aligned.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  > Saved figure: {out_path}")

def main():
    print("--- Analysis: Touchscreen Buttons ---")
    
    # List aligned files to find sessions
    # Pattern: aligned_NRR_RW010_251125_135332_Intan_002_BR_002.npz
    files = list(ALIGNED_ROOT.glob("aligned_*.npz"))
    if not files:
        print(f"No aligned files found in {ALIGNED_ROOT}")
        return
    
    print(f"Found {len(files)} aligned sessions.")
    
    for f in files:
        # Parse Filename to get Session ID matching events
        # e.g., aligned_NRR_RW010_251125_135332_Intan_002_BR_002.npz
        # We need NRR_RW010_002
        try:
            # Strategies for parsing:
            # Handle double underscores by filtering empty strings
            parts = [p for p in f.stem.split('_') if p]
            
            # Expect starts with aligned
            if parts[0] != 'aligned':
                print(f"  [SKIP] Filename {f.name} doesn't start with 'aligned'")
                continue
                
            # Find Intan index
            if 'Intan' in parts:
                idx_intan_tag = parts.index('Intan')
                idx_val = parts[idx_intan_tag + 1]
                
                # Subject is parts before Date/Time. 
                # Assuming Date/Time are the two parts immediately preceding Intan
                # aligned (0), Subj..., Date, Time, Intan...
                
                subject_parts = parts[1 : idx_intan_tag - 2]
                subject_str = "_".join(subject_parts)
                
                session_name = f"{subject_str}_{idx_val}"
            else:
                # Fallback
                print(f"  [WARN] 'Intan' tag not found in {f.name}, guessing name...")
                session_name = f.stem.replace("aligned_", "").replace("__", "_")
        except Exception as e:
            print(f"  [WARN] Failed parsing {f.name}: {e}")
            continue
            
        print(f"\nProcessing: {session_name} (File: {f.name})")
        
        # Load
        aligned = load_aligned_data(f)
        if aligned is None: continue
        
        res = load_events(session_name)
        if res is None: continue
        events, fs_ev = res
        
        # Plot
        try:
            plot_session(session_name, aligned, events, fs_ev)
        except Exception as e:
            print(f"  [ERR] Plotting failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
