"""
Plot raw high-pass-filtered (HPF) traces for selected channels around
stimulation events for a given condition (BR index), grouping by UA Arrays.

Usage:
    python scripts/plot_ua_hpf_traces.py

Edit the USER SETTINGS section below to choose condition, channels, and window.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import pandas as pd
import spikeinterface as si
import RCP_analysis as rcp
from sklearn.decomposition import PCA

matplotlib.use("Agg")  # non-interactive backend
matplotlib.rcParams["svg.fonttype"] = "none"

# ──────────────────────────────────────────────────
# USER SETTINGS
# ──────────────────────────────────────────────────
CONDITION   = 16             # BR index  (= condition number)
# 'stim' = use stim_ms (stimulation times)
# 'movement' = use event_ms / IR crossing times (for control conditions)
EVENT_TYPE = "stim"  # Change to "stim" for stimulation conditions
PROBE       = "UA"           # Exclusively analyzing Utah Array channels
CHANNELS    = list(range(1, 256))   # Electrode IDs to plot (1–256)
WINDOW_MS   = (-300.0, 300.0)      # (start, end) relative to stim onset
Y_LIM       = (-100, 100)         # µV range; set to None for auto-scale
TRIALS_TO_PLOT = [5] #[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]                  # which trials to plot: e.g. [0, 1] or 'all'
PLOT_WAVEFORM = True          # If True, plot spike waveforms on the right
# ──────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parents[3]
PARAMS      = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
SESSION     = getattr(PARAMS, "session", "UNKNOWN")
OUT_BASE    = SESSION_LOC / "results"

ALIGNED_CKPT = OUT_BASE / "checkpoints" / "Aligned"
CKPT_DIR     = OUT_BASE / "checkpoints" / PROBE

FIG_DIR = OUT_BASE / "figures" / "hpf_traces"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def find_aligned_file(aligned_dir: Path, br_idx: int) -> Path:
    """Find the aligned .npz for a given BR index across all subdirectories."""
    pat = f"*__BR_{br_idx:03d}.npz"
    # Search in stim_reaches, control_reaches, at_rest, and top-level
    search_dirs = [
        aligned_dir / "stim_reaches",
        aligned_dir / "control_reaches",
        aligned_dir / "at_rest",
        aligned_dir / "continuous_stim",
        aligned_dir,
    ]
    for d in search_dirs:
        if d.exists():
            cands = sorted(d.glob(pat))
            if cands:
                return cands[-1]
    raise FileNotFoundError(f"No aligned npz matching {pat} in {aligned_dir}")


def find_pp_folder_for_intan(ckpt_dir: Path, intan_filename: str, probe: str, condition: int) -> Path:
    """
    Find the preprocessed SpikeInterface folder that matches a specific
    Intan session filename or Blackrock condition index.
    """
    all_folders = sorted(ckpt_dir.glob("pp_*"))
    
    if probe == "UA":
        # For Blackrock, the condition ID (br_idx) is in the folder name (e.g. pp__NRR_RW022_015__NS6)
        target = f"_{condition:03d}__NS6"
        for f in all_folders:
            if target in f.name:
                return f
                
    # Fallback to matching Intan filename substring (NPRW uses this)
    for f in all_folders:
        if intan_filename in f.name:
            return f
            
    # Legacy Fallback: try without the timestamp prefix
    parts = intan_filename.split("_")
    if len(parts) >= 2:
        session_prefix = "_".join(parts[:2])
        matches = [f for f in all_folders if session_prefix in f.name]
        if matches:
            return matches[-1]
            
    if all_folders:
        print(f"[warn] Could not match folder for condition {condition}; using last available folder: {all_folders[-1].name}")
        return all_folders[-1]
    raise FileNotFoundError(f"No preprocessed folders in {ckpt_dir}")


def extract_traces(rec, ch_row: int, center_ms: float, win_ms: tuple, fs: float):
    """Extract one trace window (in µV) for a channel around a stim event."""
    ch_id = rec.get_channel_ids()[ch_row]
    s0    = int(round(center_ms / 1000.0 * fs))
    i0    = int(s0 + round(win_ms[0] / 1000.0 * fs))
    i1    = int(s0 + round(win_ms[1] / 1000.0 * fs))
    if i0 < 0 or i1 > rec.get_num_frames() or i1 <= i0:
        return None, None
    y = rec.get_traces(start_frame=i0, end_frame=i1,
                       channel_ids=[ch_id], return_in_uV=True).squeeze()
    t = (np.arange(i0, i1) - s0) / fs * 1000.0
    return t, y



# ──────────────────────────────────────────────────
# SPIKE VALIDATION FUNCTIONS
# ──────────────────────────────────────────────────

def extract_spike_waveforms(
    rec, 
    ch_row: int, 
    spike_times_ms: np.ndarray, 
    fs: float,
    pre_ms: float = 0.5,
    post_ms: float = 1.5,
) -> np.ndarray:
    """
    Extract spike waveforms for a single channel.
    
    Parameters
    ----------
    rec : SpikeInterface recording
    ch_row : int
        Channel index in the recording
    spike_times_ms : np.ndarray
        Spike times in ms (in recording-local time)
    fs : float
        Sampling frequency in Hz
    pre_ms : float
        Time before spike peak to extract (ms)
    post_ms : float
        Time after spike peak to extract (ms)
    
    Returns
    -------
    waveforms : np.ndarray
        Shape (n_spikes, n_samples), in µV
    """
    if spike_times_ms.size == 0:
        return np.array([]).reshape(0, 0)
    
    ch_id = rec.get_channel_ids()[ch_row]
    n_frames = rec.get_num_frames()
    
    pre_samp = int(round(pre_ms / 1000.0 * fs))
    post_samp = int(round(post_ms / 1000.0 * fs))
    n_samples = pre_samp + post_samp
    
    waveforms = []
    
    for spike_ms in spike_times_ms:
        center_samp = int(round(spike_ms / 1000.0 * fs))
        i0 = center_samp - pre_samp
        i1 = center_samp + post_samp
        
        # Skip if outside recording bounds
        if i0 < 0 or i1 > n_frames:
            continue
        
        try:
            wf = rec.get_traces(
                start_frame=i0, 
                end_frame=i1,
                channel_ids=[ch_id], 
                return_in_uV=True
            ).squeeze()
            
            if wf.shape[0] == n_samples:
                waveforms.append(wf)
        except Exception:
            continue
    
    if len(waveforms) == 0:
        return np.array([]).reshape(0, n_samples)
    
    return np.array(waveforms)


def compute_waveform_pca(waveforms: np.ndarray, n_components: int = 2) -> tuple:
    """
    Compute PCA on spike waveforms.
    
    Parameters
    ----------
    waveforms : np.ndarray
        Shape (n_spikes, n_samples)
    n_components : int
        Number of PCA components
    
    Returns
    -------
    pca_coords : np.ndarray
        Shape (n_spikes, n_components)
    explained_var : np.ndarray
        Explained variance ratio per component
    """
    from sklearn.decomposition import PCA
    
    if waveforms.size == 0 or waveforms.shape[0] < n_components:
        return np.array([]).reshape(0, n_components), np.zeros(n_components)
    
    # Center waveforms
    waveforms_centered = waveforms - waveforms.mean(axis=0, keepdims=True)
    
    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(waveforms_centered)
    
    return pca_coords, pca.explained_variance_ratio_


def compute_isi(spike_times_ms: np.ndarray) -> np.ndarray:
    """
    Compute inter-spike intervals.
    
    Parameters
    ----------
    spike_times_ms : np.ndarray
        Spike times in ms (sorted)
    
    Returns
    -------
    isi_ms : np.ndarray
        Inter-spike intervals in ms
    """
    if spike_times_ms.size < 2:
        return np.array([])
    
    sorted_times = np.sort(spike_times_ms)
    return np.diff(sorted_times)


def compute_amplitudes(waveforms: np.ndarray) -> np.ndarray:
    """
    Compute peak-to-trough amplitude for each waveform.
    
    Parameters
    ----------
    waveforms : np.ndarray
        Shape (n_spikes, n_samples)
    
    Returns
    -------
    amplitudes : np.ndarray
        Peak-to-trough amplitude in µV for each spike
    """
    if waveforms.size == 0:
        return np.array([])
    
    # Peak-to-trough amplitude
    return np.max(waveforms, axis=1) - np.min(waveforms, axis=1)


def compute_isi_violation_rate(isi_ms: np.ndarray, refractory_ms: float = 1.0) -> float:
    """
    Compute the fraction of ISIs that violate the refractory period.
    
    Parameters
    ----------
    isi_ms : np.ndarray
        Inter-spike intervals in ms
    refractory_ms : float
        Refractory period threshold in ms
    
    Returns
    -------
    violation_rate : float
        Fraction of ISIs below threshold (0-1)
    """
    if isi_ms.size == 0:
        return 0.0
    
    n_violations = np.sum(isi_ms < refractory_ms)
    return n_violations / len(isi_ms)


def plot_spike_validation_figure(
    waveforms: np.ndarray,
    spike_times_ms: np.ndarray,
    fs: float,
    region_name: str,
    n_channels: int,
    out_path: Path,
    pre_ms: float = 0.5,
    post_ms: float = 1.5,
    max_waveforms_plot: int = 200,
    max_waveforms_pca: int = 5000,
):
    """
    Generate a 4-panel spike validation figure for one array/region.
    
    Panels:
        A. Waveform overlay + mean ± SD
        B. PCA scatter (PC1 vs PC2)
        C. Amplitude histogram
        D. ISI histogram (log scale)
    
    Parameters
    ----------
    waveforms : np.ndarray
        Shape (n_spikes, n_samples), in µV
    spike_times_ms : np.ndarray
        All spike times in ms (for ISI calculation)
    fs : float
        Sampling frequency
    region_name : str
        Name of the array/region
    n_channels : int
        Number of channels contributing spikes
    out_path : Path
        Output path for the figure
    pre_ms, post_ms : float
        Waveform window parameters
    max_waveforms_plot : int
        Maximum waveforms to overlay in panel A
    max_waveforms_pca : int
        Maximum waveforms for PCA (subsample if more)
    """
    n_spikes = waveforms.shape[0] if waveforms.size > 0 else 0
    n_samples = waveforms.shape[1] if waveforms.size > 0 else int((pre_ms + post_ms) / 1000.0 * fs)
    
    # Time axis for waveforms
    t_wf = np.linspace(-pre_ms, post_ms, n_samples)
    
    # Compute metrics
    amplitudes = compute_amplitudes(waveforms) if n_spikes > 0 else np.array([])
    isi_ms = compute_isi(spike_times_ms)
    isi_violation_rate = compute_isi_violation_rate(isi_ms, refractory_ms=1.0)
    
    # Subsample for PCA if needed
    if n_spikes > max_waveforms_pca:
        pca_idx = np.random.choice(n_spikes, max_waveforms_pca, replace=False)
        waveforms_pca = waveforms[pca_idx]
    else:
        waveforms_pca = waveforms
    
    pca_coords, explained_var = compute_waveform_pca(waveforms_pca) if n_spikes >= 2 else (np.array([]).reshape(0, 2), np.zeros(2))
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"{region_name} - Spike Validation\n"
        f"N = {n_spikes:,} spikes from {n_channels} channels | "
        f"ISI violations (<1ms): {isi_violation_rate*100:.1f}%",
        fontsize=12, fontweight='bold'
    )
    
    # ─────────────────────────────────────────────
    # Panel A: Waveform Overlay + Mean ± SD
    # ─────────────────────────────────────────────
    ax_wf = axes[0, 0]
    ax_wf.set_title("A. Waveform Overlay", fontsize=11, fontweight='bold')
    
    if n_spikes > 0:
        # Subsample for plotting
        if n_spikes > max_waveforms_plot:
            plot_idx = np.random.choice(n_spikes, max_waveforms_plot, replace=False)
            wf_plot = waveforms[plot_idx]
        else:
            wf_plot = waveforms
        
        # Plot individual waveforms (light gray)
        for wf in wf_plot:
            ax_wf.plot(t_wf, wf, color='gray', alpha=0.1, lw=0.5)
        
        # Plot mean ± 1.96 SEM
        wf_mean = np.mean(waveforms, axis=0)
        wf_se = 1.96 * np.std(waveforms, axis=0) / np.sqrt(waveforms.shape[0])
        
        ax_wf.fill_between(t_wf, wf_mean - wf_se, wf_mean + wf_se, 
                           color='blue', alpha=0.3, label='±1.96 SEM')
        ax_wf.plot(t_wf, wf_mean, color='blue', lw=2, label='Mean')
        
        ax_wf.axvline(0, ls='--', color='red', alpha=0.5, lw=1, label='Peak')
        ax_wf.legend(loc='upper right', fontsize=8)
        
        # Add waveform statistics
        peak_to_trough = np.max(wf_mean) - np.min(wf_mean)
        ax_wf.text(0.02, 0.98, f"P-T: {peak_to_trough:.1f} µV", 
                   transform=ax_wf.transAxes, fontsize=9, va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax_wf.text(0.5, 0.5, "No spikes detected", transform=ax_wf.transAxes,
                   ha='center', va='center', fontsize=12, color='gray')
    
    ax_wf.set_xlabel("Time from peak (ms)", fontsize=10)
    ax_wf.set_ylabel("Amplitude (µV)", fontsize=10)
    ax_wf.set_xlim(-pre_ms, post_ms)
    ax_wf.set_ylim(-100, 100)
    
    # ─────────────────────────────────────────────
    # Panel B: PCA Scatter
    # ─────────────────────────────────────────────
    ax_pca = axes[0, 1]
    ax_pca.set_title("B. PCA of Waveforms", fontsize=11, fontweight='bold')
    
    if pca_coords.size > 0 and pca_coords.shape[0] > 1:
        # Color by amplitude
        if amplitudes.size > max_waveforms_pca:
            amp_pca = amplitudes[pca_idx] if n_spikes > max_waveforms_pca else amplitudes
        else:
            amp_pca = amplitudes[:pca_coords.shape[0]]
        
        scatter = ax_pca.scatter(
            pca_coords[:, 0], pca_coords[:, 1],
            c=amp_pca, cmap='viridis', alpha=0.5, s=10, edgecolors='none',
            vmin=0, vmax=500  # Cap colorbar at 500 µV
        )
        cbar = plt.colorbar(scatter, ax=ax_pca, shrink=0.8)
        cbar.set_label("Amplitude (µV)", fontsize=9)
        
        # Add explained variance
        ax_pca.text(0.02, 0.98, 
                    f"PC1: {explained_var[0]*100:.1f}%\nPC2: {explained_var[1]*100:.1f}%",
                    transform=ax_pca.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax_pca.text(0.5, 0.5, "Insufficient spikes for PCA", transform=ax_pca.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
    
    ax_pca.set_xlabel("PC1", fontsize=10)
    ax_pca.set_ylabel("PC2", fontsize=10)
    
    # ─────────────────────────────────────────────
    # Panel C: Amplitude Histogram (Log Scale)
    # ─────────────────────────────────────────────
    ax_amp = axes[1, 0]
    ax_amp.set_title("C. Amplitude Distribution (Log Scale)", fontsize=11, fontweight='bold')
    
    if amplitudes.size > 0:
        # Use log-spaced bins for amplitude
        amp_min = max(amplitudes.min(), 1)  # Avoid log(0)
        amp_max = amplitudes.max()
        bins = np.logspace(np.log10(amp_min), np.log10(amp_max), 50)
        
        ax_amp.hist(amplitudes, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
        ax_amp.set_xscale('log')
        
        # Add statistics
        median_amp = np.median(amplitudes)
        ax_amp.axvline(median_amp, ls='--', color='red', lw=2, label=f'Median: {median_amp:.1f} µV')
        ax_amp.legend(loc='upper right', fontsize=9)
        
        # Add percentiles
        p25, p75 = np.percentile(amplitudes, [25, 75])
        ax_amp.text(0.02, 0.98, 
                    f"25th: {p25:.1f} µV\n75th: {p75:.1f} µV",
                    transform=ax_amp.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax_amp.text(0.5, 0.5, "No amplitudes to plot", transform=ax_amp.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
    
    ax_amp.set_xlabel("Peak-to-Trough Amplitude (µV)", fontsize=10)
    ax_amp.set_ylabel("Count", fontsize=10)
    
    # ─────────────────────────────────────────────
    # Panel D: ISI Histogram (Log Scale)
    # ─────────────────────────────────────────────
    ax_isi = axes[1, 1]
    ax_isi.set_title("D. Inter-Spike Interval Distribution", fontsize=11, fontweight='bold')
    
    if isi_ms.size > 0:
        # Use log-spaced bins
        isi_positive = isi_ms[isi_ms > 0]
        
        if isi_positive.size > 0:
            # Bins from 0.1 ms to max ISI (or 1000 ms)
            max_isi = min(np.max(isi_positive), 1000)
            bins = np.logspace(np.log10(0.1), np.log10(max_isi), 50)
            
            ax_isi.hist(isi_positive, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
            ax_isi.set_xscale('log')
            
            # Mark refractory period
            ax_isi.axvline(1.0, ls='--', color='red', lw=2, label='1 ms (refractory)')
            ax_isi.axvline(2.0, ls=':', color='orange', lw=1.5, label='2 ms')
            
            # Shade violation region
            ax_isi.axvspan(0.1, 1.0, color='red', alpha=0.1)
            
            ax_isi.legend(loc='upper right', fontsize=9)
            
            # Add violation stats
            n_below_1ms = np.sum(isi_positive < 1.0)
            n_below_2ms = np.sum(isi_positive < 2.0)
            ax_isi.text(0.02, 0.98, 
                        f"<1ms: {n_below_1ms} ({n_below_1ms/len(isi_positive)*100:.2f}%)\n"
                        f"<2ms: {n_below_2ms} ({n_below_2ms/len(isi_positive)*100:.2f}%)",
                        transform=ax_isi.transAxes, fontsize=9, va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax_isi.text(0.5, 0.5, "No positive ISIs", transform=ax_isi.transAxes,
                        ha='center', va='center', fontsize=12, color='gray')
    else:
        ax_isi.text(0.5, 0.5, "Insufficient spikes for ISI", transform=ax_isi.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
    
    ax_isi.set_xlabel("ISI (ms)", fontsize=10)
    ax_isi.set_ylabel("Count", fontsize=10)
    
    # ─────────────────────────────────────────────
    # Save figure
    # ─────────────────────────────────────────────
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.svg'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path.name}")


def collect_region_spikes(
    rec,
    region_channels: list,
    ridx_to_meta: dict,
    peak_ms_dict: dict,
    target_ch_elecs: np.ndarray,
    rec_start_ms: float,
    fs: float,
    pre_ms: float = 0.5,
    post_ms: float = 1.5,
) -> tuple:
    """
    Collect all spike waveforms and times for a region.
    
    Parameters
    ----------
    rec : SpikeInterface recording
    region_channels : list
        List of channel indices belonging to this region
    ridx_to_meta : dict
        Channel metadata mapping
    peak_ms_dict : dict
        Spike times dictionary from aligned file
    target_ch_elecs : np.ndarray
        Electrode IDs
    rec_start_ms : float
        Recording start time offset
    fs : float
        Sampling frequency
    pre_ms, post_ms : float
        Waveform extraction window
    
    Returns
    -------
    all_waveforms : np.ndarray
        Shape (n_total_spikes, n_samples)
    all_spike_times : np.ndarray
        All spike times in ms (recording-local)
    n_channels_with_spikes : int
        Number of channels that contributed spikes
    """
    all_waveforms = []
    all_spike_times = []
    n_channels_with_spikes = 0
    
    for ch_idx in region_channels:
        meta = ridx_to_meta.get(ch_idx, {"eid": ch_idx, "oid": ch_idx})
        eid = meta["eid"]
        oid = meta["oid"]
        
        # Get spike times for this channel
        ch_peaks_abs = np.asarray(peak_ms_dict.get(oid, []), float).ravel()
        
        # Fallback to electrode ID
        if ch_peaks_abs.size == 0 and eid > 0:
            ch_peaks_abs = np.asarray(peak_ms_dict.get(eid, []), float).ravel()
        
        if ch_peaks_abs.size == 0:
            continue
        
        # Convert to recording-local time
        ch_peaks_local = ch_peaks_abs - rec_start_ms
        
        # Filter to valid times (within recording)
        rec_dur_ms = rec.get_num_frames() / fs * 1000.0
        valid_mask = (ch_peaks_local >= pre_ms) & (ch_peaks_local <= rec_dur_ms - post_ms)
        ch_peaks_valid = ch_peaks_local[valid_mask]
        
        if ch_peaks_valid.size == 0:
            continue
        
        # Extract waveforms
        waveforms = extract_spike_waveforms(rec, ch_idx, ch_peaks_valid, fs, pre_ms, post_ms)
        
        if waveforms.size > 0:
            all_waveforms.append(waveforms)
            all_spike_times.extend(ch_peaks_valid.tolist())
            n_channels_with_spikes += 1
    
    # Concatenate all waveforms
    if all_waveforms:
        all_waveforms = np.vstack(all_waveforms)
    else:
        n_samples = int((pre_ms + post_ms) / 1000.0 * fs)
        all_waveforms = np.array([]).reshape(0, n_samples)
    
    all_spike_times = np.array(all_spike_times)
    
    return all_waveforms, all_spike_times, n_channels_with_spikes




def main():
    # ── Load aligned file for this condition ──
    aligned_path = find_aligned_file(ALIGNED_CKPT, CONDITION)
    print(f'{aligned_path}')
    z = np.load(aligned_path, allow_pickle=True)
    print(f"[info] Aligned file: {aligned_path.name}")

    # ── Parse metadata ──
    if "align_meta" in z.files:
        raw_meta = z["align_meta"].item() if z["align_meta"].ndim == 0 else z["align_meta"]
        meta = raw_meta if isinstance(raw_meta, dict) else json.loads(raw_meta)
    else:
        meta = {}
    intan_filename = meta.get("intan_filename", "")
    shift_ms = float(meta.get("shift_ms", 0.0))

    # # stim_ms are in the ALIGNED time base (Intan ms - shift_ms).
    # stim_ms = z.get("stim_ms", np.array([], dtype=float))
    # stim_ms = np.asarray(stim_ms, float).ravel()
    # if stim_ms.size == 0:
    #     print("[error] No stimulation events found in aligned file.")
    #     return

    if EVENT_TYPE == "stim":
        # Use stimulation times
        event_ms = z.get("stim_ms", np.array([], dtype=float))
        event_label = "stimulation"
    else:
        # Use movement onset / IR crossing times
        # Try multiple possible field names
        event_ms = None
        for field_name in ["event_ms", "movement_onset_ms", "reach_onset_ms", 
                           "ir_ms", "ir_crossing_ms", "anchor_ms", "trial_onset_ms"]:
            if field_name in z.files:
                candidate = z[field_name]
                if candidate is not None:
                    candidate = np.asarray(candidate, float).ravel()
                    if candidate.size > 0:
                        event_ms = candidate
                        print(f"[info] Using '{field_name}' as event times")
                        break
        
        if event_ms is None:
            event_ms = np.array([], dtype=float)
        event_label = "movement onset"
    
    event_ms = np.asarray(event_ms, float).ravel()
    if event_ms.size == 0:
        print(f"[error] No {event_label} events found in aligned file.")
        print(f"[debug] Available fields: {list(z.files)}")
        return

    # ── Get the recording's start time in the aligned frame ──
    # The SI recording starts at sample 0.
    if PROBE == "NPRW":
        probe_meta = z["nprw_meta"].item() if "nprw_meta" in z.files else {}
        rec_start_ms = float(probe_meta.get("rec_start_ms_aligned", 0.0))
        pk_obj = z.get("nprw_peak_ms_dedup", z.get("nprw_peak_ms", None))
        peak_ms_dict = pk_obj.reshape(-1)[0] if (isinstance(pk_obj, np.ndarray) and pk_obj.dtype == object) else pk_obj
        if peak_ms_dict is None: peak_ms_dict = {}
        target_ch_regions = []
        region_names = []
        target_ch_elecs = []
    else:  # UA
        # UA SpikeInterface recording (NS6) starts at Blackrock Local Time 0.0
        rec_start_ms = 0.0
        pk_obj = z.get("ua_peak_ms_dedup", z.get("ua_peak_ms", None))
        peak_ms_dict = pk_obj.reshape(-1)[0] if (isinstance(pk_obj, np.ndarray) and pk_obj.dtype == object) else pk_obj
        if peak_ms_dict is None: peak_ms_dict = {}
        target_ch_regions = z.get("ua_region", [])
        region_names = z.get("ua_region_names", [])
        target_ch_elecs = z.get("ua_elec", [])
        if isinstance(target_ch_elecs, np.ndarray) and target_ch_elecs.ndim == 0:
            target_ch_elecs = target_ch_elecs.item()  # Handle occasional object scalar
        target_ch_elecs = np.asarray(target_ch_elecs, int).ravel()
        ua_idx_rows = z.get("ua_idx_rows", [])
        if isinstance(ua_idx_rows, np.ndarray) and ua_idx_rows.ndim == 0:
            ua_idx_rows = ua_idx_rows.item()
        ua_idx_rows = np.asarray(ua_idx_rows, int).ravel()

    # Convert event times from aligned time base to recording-local time base
    event_local_ms = event_ms - rec_start_ms
    print(f"[info] Aligned event_ms range: {event_ms.min():.1f} – {event_ms.max():.1f} ms")
    print(f"[info] shift_ms (anchor offset) assumed: {rec_start_ms:.1f} ms")
    print(f"[info] Local event_ms range: {event_local_ms.min():.1f} – {event_local_ms.max():.1f} ms")

    # ── Resolve Stimulation Duration from Hardware Bounds ──
    from RCP_analysis.python.functions.config_loading import METADATA_CSV, NPRW_AUX_DATA, METADATA_ROOT
    stim_npz_path, _ = rcp.stim_npz_path_from_br_idx(CONDITION, METADATA_CSV, NPRW_AUX_DATA)
    fs_nprw = 30_000.0
    shifts_csv = OUT_BASE.parent / "Metadata" / "br_to_intan_shifts.csv"
    if shifts_csv.exists():
        br2fs_intan = rcp.get_metadata_mapping(shifts_csv, "br_idx", "fs_intan")
        fs_nprw = float(br2fs_intan.get(CONDITION, 30_000.0))
        
        # Override shift_ms dynamically if processing UA (Blackrock is anchor)
        if PROBE == "UA":
            br2shift_samp = rcp.get_metadata_mapping(shifts_csv, "br_idx", "shift_sample")
            if CONDITION in br2shift_samp and br2shift_samp[CONDITION]:
                shift_ms = float(br2shift_samp[CONDITION]) / fs_nprw * 1000.0
                rec_start_ms = 0.0
                event_local_ms = event_ms - rec_start_ms

    stim_dur = 0.0
    if stim_npz_path and stim_npz_path.exists():
        stim = rcp.load_stim_detection(stim_npz_path)
        block_bounds = stim.get("block_bounds_samples", np.empty((0, 2), dtype=np.int64))
        if len(block_bounds) > 0:
            if isinstance(block_bounds, list):
                block_bounds = np.array(block_bounds, dtype=np.int64)
            nprw_samps = block_bounds[:, 0].astype(np.int64)
            nprw_ends  = block_bounds[:, 1].astype(np.int64)
            stim_dur   = float((nprw_ends - nprw_samps).max() * 1000.0 / fs_nprw)
    print(f"[info] Extracted stim_dur: {stim_dur:.1f} ms")

    # ── Load SpikeInterface recording for the correct Intan session ──
    pp_folder = find_pp_folder_for_intan(CKPT_DIR, intan_filename, PROBE, CONDITION)
    rec = si.load(str(pp_folder))
    fs  = float(rec.get_sampling_frequency())
    n_ch = rec.get_num_channels()
    rec_dur_ms = rec.get_num_frames() / fs * 1000.0
    print(f"[info] Recording: {pp_folder.name}  |  fs={fs:.0f} Hz  |  {n_ch} ch  |  {rec_dur_ms:.1f} ms")
    print(f"[info] {event_label.capitalize()} events: {event_ms.size}  |  condition {CONDITION}")
    
    # Only compute stim duration for stim conditions
    if EVENT_TYPE != "stim":
        stim_dur = 0.0

    # Resolve channel metadata (Electrode ID and Original Index)
    # This map allows us to find the correct spikes in the .npz for any recording channel
    ridx_to_meta = {}
    ch_ids = rec.get_channel_ids()
    
    # Path to robust mapping CSV
    mapping_csv = REPO_ROOT / "scripts" / "electrode_port_mapping.csv"
    mapping_df = pd.read_csv(mapping_csv) if mapping_csv.exists() else None
    
    for ridx, cid in enumerate(ch_ids):
        # 1. Try to parse Electrode ID from name (e.g. UAe005 -> 5)
        m = re.search(r'UAe(\d+)', str(cid))
        eid = int(m.group(1)) if m else -1
        
        # 2. Fallback to indexing target_ch_elecs if parsing fails
        if eid < 0 and ridx < len(target_ch_elecs):
            eid = int(target_ch_elecs[ridx])
            
        # 3. Find where this Electrode is in the .npz mapping to align spikes
        # peak_ms_dict is usually keyed by the index into the 'peaks' array saved in the .npz
        # which corresponds to the index in target_ch_elecs
        oid = ridx
        if eid > 0:
            matches = np.where(target_ch_elecs == eid)[0]
            if matches.size > 0:
                oid = int(matches[0])
                
        ridx_to_meta[ridx] = {"eid": eid, "oid": oid, "cid": cid}

    # Resolve specifically requested channels for plotting
    resolved_channels = []
    if PROBE == "UA" and target_ch_elecs.size > 0:
        print(f"[info] Resolving Electrode IDs to recording indices...")
        for eid_req in CHANNELS:
            # Find which RIDX in the current recording has this Electrode ID
            # We check the metadata we just built from the channel IDs
            found = False
            for ridx, meta in ridx_to_meta.items():
                if meta["eid"] == eid_req:
                    resolved_channels.append(ridx)
                    found = True
                    break
            if not found:
                # Fallback to direct mapping if parsing didn't find it
                matches = np.where(target_ch_elecs == eid_req)[0]
                if matches.size > 0:
                    resolved_channels.append(int(matches[0]))
    else:
        # Fallback for NPRW or if no electrode mapping
        resolved_channels = [c for c in CHANNELS if 0 <= c < n_ch]

    if not resolved_channels:
        print(f"[error] No valid channels found for {CHANNELS} (recording has {n_ch} channels).")
        return



    # ── Filter valid events that fit within the recording ──
    valid = []
    for s in event_local_ms:
        if (s + WINDOW_MS[0]) >= 0 and (s + WINDOW_MS[1]) <= rec_dur_ms:
            valid.append(s)
    valid = np.array(valid)
    if valid.size == 0:
        print(f"[error] No {event_label} events fit within the recording for this window.")
        return
    print(f"[info] Valid {event_label} events for window: {valid.size} / {event_ms.size}")

    # ── Build figure ──
    if isinstance(TRIALS_TO_PLOT, str) and TRIALS_TO_PLOT.lower() == "all":
        plot_idxs = np.arange(len(valid))
    else:
        plot_idxs = np.array([i for i in TRIALS_TO_PLOT if i < len(valid)])

    # Group channels by region
    region_to_idxs = {}
    if PROBE == "UA" and len(target_ch_regions) > 0:
        for ch_idx in resolved_channels:
            if ch_idx < len(target_ch_regions):
                reg_idx = int(target_ch_regions[ch_idx])
                reg_name = region_names[reg_idx] if reg_idx >= 0 else "Unknown"
                region_to_idxs.setdefault(reg_name, []).append(ch_idx)
    else:
        region_to_idxs["All Channels"] = resolved_channels

    for trial_i in plot_idxs:
        center_time = valid[trial_i]
        
        for reg_name, reg_channels in region_to_idxs.items():
            if not reg_channels: continue
            
            n_rows = len(reg_channels)
            PLOT_WAVEFORM = globals().get("PLOT_WAVEFORM", False)
            if PLOT_WAVEFORM:
                fig = plt.figure(figsize=(15, 2.0 * n_rows))
                gs = matplotlib.gridspec.GridSpec(n_rows, 2, width_ratios=[8, 1])
                axes_trace = []
                axes_wv = []
                for r in range(n_rows):
                    ax_t = fig.add_subplot(gs[r, 0], sharex=axes_trace[0] if r>0 else None, sharey=axes_trace[0] if r>0 else None)
                    ax_w = fig.add_subplot(gs[r, 1], sharex=axes_wv[0] if r>0 else None, sharey=ax_t)
                    axes_trace.append(ax_t)
                    axes_wv.append(ax_w)
            else:
                fig, axes_raw = plt.subplots(n_rows, 1, figsize=(12, 2.0 * n_rows),
                                         sharex=True, sharey=True)
                axes_trace = np.atleast_1d(axes_raw)
                axes_wv = [None] * n_rows
                
            for r, ch in enumerate(reg_channels):
                ax = axes_trace[r]
                ax_wv = axes_wv[r]
                t, y = extract_traces(rec, ch, center_time, WINDOW_MS, fs)
                
                meta = ridx_to_meta.get(ch, {"eid": ch, "oid": ch})
                eid = meta["eid"]
                oid = meta["oid"]
                
                if t is not None:
                    ax.plot(t, y, lw=0.7, color="k")
                
                # Label with Electrode ID
                ch_label = f"Elec {eid}" if PROBE == "UA" and eid > 0 else f"Ch {ch}"
                ax.set_ylabel(f"{ch_label}\n(µV)", fontsize=8)

                # Overlay spikes - IMPORTANT: lookup by Alignment Index (oid) 
                # This ensures we get spikes for the physical electrode shown in the trace
                ch_peaks_abs_aligned = np.asarray(peak_ms_dict.get(oid, []), float).ravel()
                
                # Fallback to keying by Electrode ID directly (for some pipeline versions)
                if ch_peaks_abs_aligned.size == 0 and PROBE == "UA" and eid > 0:
                    ch_peaks_abs_aligned = np.asarray(peak_ms_dict.get(eid, []), float).ravel()
                
                event_ms_trial = center_time + rec_start_ms
                pk_rel = ch_peaks_abs_aligned - event_ms_trial


                
                # filter to window
                pk_rel_win = pk_rel[(pk_rel >= WINDOW_MS[0]) & (pk_rel <= WINDOW_MS[1])]
                
                # Get trace y-value at these relative times
                if pk_rel_win.size > 0 and t is not None:
                    peak_idx = np.searchsorted(t, pk_rel_win)
                    valid_idx = (peak_idx >= 0) & (peak_idx < len(t))
                    peak_idx = peak_idx[valid_idx]
                    pk_rel_clean = pk_rel_win[valid_idx]
                    ax.plot(pk_rel_clean, y[peak_idx], "ro", markersize=3, alpha=0.8)
                    
                    if ax_wv is not None:
                        wf_pre = int(1.0 * fs / 1000.0)
                        wf_post = int(2.0 * fs / 1000.0)
                        t_wv = np.arange(-wf_pre, wf_post) * 1000.0 / fs
                        for p_idx in peak_idx:
                            i0 = p_idx - wf_pre
                            i1 = p_idx + wf_post
                            if i0 >= 0 and i1 < len(y):
                                ax_wv.plot(t_wv, y[i0:i1], color="k", lw=0.5, alpha=0.5)
                        ax_wv.set_xlim(-1, 1.5)
                        ax_wv.axvline(0, ls="--", color="r", alpha=0.5)
                        if r == n_rows - 1:
                            ax_wv.set_xlabel("Time (ms)")

                if EVENT_TYPE == "stim":
                    ax.axvline(0.0, ls="--", lw=0.8, color="green", label="Stim onset")
                    if stim_dur > 0:
                        ax.axvline(stim_dur, ls="--", lw=0.8, color="green", label="Stim end")
                        ax.axvspan(0.0, stim_dur, color="gray", alpha=0.3, zorder=0)
                else:
                    ax.axvline(0.0, ls="--", lw=0.8, color="blue", label="Movement onset")
                
                if Y_LIM is not None:
                    ax.set_ylim(*Y_LIM)


            event_xlabel = "stimulation" if EVENT_TYPE == "stim" else "movement onset"
            axes_trace[-1].set_xlabel(f"Time relative to {event_xlabel} (ms)")
            axes_trace[0].set_xlim(*WINDOW_MS)

            event_type_label = "Stim" if EVENT_TYPE == "stim" else "Control"
            fig.suptitle(
                f"{SESSION}  •  {event_type_label} Condition {CONDITION} (BR_{CONDITION:03d})  •  "
                f"Region: {reg_name}  •  Trial {trial_i + 1}",
                fontsize=11, y=0.99 if n_rows < 10 else 1.0 - (0.5 / n_rows),
            )
            fig.tight_layout(rect=[0, 0, 1, 0.98 if n_rows < 10 else 1.0 - (1.0 / n_rows)])

            reg_clean = reg_name.replace(' ', '_').replace('/', '_')
            event_suffix = "stim" if EVENT_TYPE == "stim" else "ctrl"
            out_path = (
                FIG_DIR
                / f"{SESSION}__cond{CONDITION:03d}__{PROBE}__{event_suffix}__Region_{reg_clean}"
                  f"__trial{trial_i + 1}__win{int(WINDOW_MS[0])}-{int(WINDOW_MS[1])}ms.png"
            )
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {out_path}")


    # ──────────────────────────────────────────────────
    # SPIKE VALIDATION FIGURES (one per region/array)
    # ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Generating spike validation figures per array...")
    print("="*60)
    
    VALIDATION_DIR = FIG_DIR / "spike_validation"
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    
    for reg_name, reg_channels in region_to_idxs.items():
        if not reg_channels:
            print(f"[skip] {reg_name}: No channels")
            continue
        
        print(f"\n[processing] {reg_name} ({len(reg_channels)} channels)...")
        
        # Collect all spikes from this region
        all_waveforms, all_spike_times, n_ch_with_spikes = collect_region_spikes(
            rec=rec,
            region_channels=reg_channels,
            ridx_to_meta=ridx_to_meta,
            peak_ms_dict=peak_ms_dict,
            target_ch_elecs=target_ch_elecs,
            rec_start_ms=rec_start_ms,
            fs=fs,
            pre_ms=0.5,
            post_ms=1.5,
        )
        
        print(f"    Collected {all_waveforms.shape[0]:,} waveforms from {n_ch_with_spikes} channels")
        
        # Generate validation figure
        reg_clean = reg_name.replace(' ', '_').replace('/', '_')
        event_suffix = "stim" if EVENT_TYPE == "stim" else "ctrl"
        
        out_path = (
            VALIDATION_DIR 
            / f"{SESSION}__cond{CONDITION:03d}__{PROBE}__{event_suffix}__"
              f"SpikeValidation__{reg_clean}.png"
        )
        
        plot_spike_validation_figure(
            waveforms=all_waveforms,
            spike_times_ms=all_spike_times,
            fs=fs,
            region_name=f"{reg_name} ({SESSION} - Cond {CONDITION})",
            n_channels=n_ch_with_spikes,
            out_path=out_path,
            pre_ms=0.5,
            post_ms=1.5,
        )
    
    print("\n[done] Spike validation figures complete.")


if __name__ == "__main__":
    main()
