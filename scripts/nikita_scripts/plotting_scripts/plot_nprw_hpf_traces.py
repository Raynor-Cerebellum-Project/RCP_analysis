"""
Plot raw high-pass-filtered (HPF) traces for selected channels around
stimulation or movement onset events for a given condition (BR index), 
specifically for the NPRW (Neuropixels / RHS2000 linear probe).

Usage:
    python scripts/nikita_scripts/plotting_scripts/plot_nprw_hpf_traces.py

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
CONDITION        = 12                  # BR index (= condition number, e.g. 8, 12, 17, 23)
EVENT_TYPE       = "stim"              # 'stim' = stimulation times; 'movement' = reach/IR crossing times
PROBE            = "NPRW"              # Exclusively analyzing NPRW linear probe channels
CHANNELS         = list(range(0, 128)) # Channels to plot (0–127), or a custom list e.g. [10, 11, 12, 13]
CHANNELS_PER_FIG = 16                  # Number of channels per figure (e.g. 16 or 32); set to None to plot all in 1 fig
GROUP_BY_DEPTH   = True                # If True, order and label channel groups by probe depth (µm)
WINDOW_MS        = (-400.0, 400.0)     # (start, end) ms relative to event onset
Y_LIM            = (-100, 100)         # µV range; set to None for auto-scale
TRIALS_TO_PLOT   = [0]                 # Which trials to plot: e.g. [0], [0, 1], or 'all'
PLOT_WAVEFORM    = True                # If True, plot spike waveforms on the right panel
# ──────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parents[3]
PARAMS      = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
SESSION     = getattr(PARAMS, "session", "UNKNOWN")
OUT_BASE    = SESSION_LOC / "results"

ALIGNED_CKPT = OUT_BASE / "checkpoints" / "Aligned"
CKPT_DIR     = OUT_BASE / "checkpoints" / PROBE

FIG_DIR = OUT_BASE / "figures" / "nprw_hpf_traces"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def find_aligned_file(aligned_dir: Path, br_idx: int) -> Path:
    """Find the aligned .npz for a given BR index across all subdirectories."""
    pat = f"*__BR_{br_idx:03d}.npz"
    search_dirs = [
        aligned_dir / "stim_reaches",
        aligned_dir / "control_reaches",
        aligned_dir / "at_rest",
        aligned_dir / "continuous_stim",
        aligned_dir / "grasp",
        aligned_dir / "imu",
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
    
    # 1. Match exact Intan session filename (e.g. pp_local_30_150__interp_NRR_RW012_260116_145123)
    if intan_filename:
        for f in all_folders:
            if intan_filename in f.name:
                return f
        
        # 2. Try prefix without timestamp
        parts = intan_filename.split("_")
        if len(parts) >= 2:
            session_prefix = "_".join(parts[:2])
            matches = [f for f in all_folders if session_prefix in f.name]
            if matches:
                return matches[-1]

    # 3. If condition index is known, check shifts CSV for intan_filename
    shifts_csv = OUT_BASE.parent / "Metadata" / "br_to_intan_shifts.csv"
    if shifts_csv.exists() and condition is not None:
        try:
            br2intan = rcp.get_metadata_mapping(shifts_csv, "br_idx", "intan_filename")
            mapped_intan = br2intan.get(condition, "")
            if mapped_intan:
                for f in all_folders:
                    if mapped_intan in f.name:
                        return f
        except Exception:
            pass

    if all_folders:
        print(f"[warn] Could not match folder for condition {condition}; using last available folder: {all_folders[-1].name}")
        return all_folders[-1]
    raise FileNotFoundError(f"No preprocessed folders in {ckpt_dir}")


def extract_traces(rec, ch_row: int, center_ms: float, win_ms: tuple, fs: float):
    """Extract one trace window (in µV) for a channel around an event."""
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
    """
    if waveforms.size == 0 or waveforms.shape[0] < n_components:
        return np.array([]).reshape(0, n_components), np.zeros(n_components)
    
    waveforms_centered = waveforms - waveforms.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(waveforms_centered)
    
    return pca_coords, pca.explained_variance_ratio_


def compute_isi(spike_times_ms: np.ndarray) -> np.ndarray:
    """Compute inter-spike intervals in ms."""
    if spike_times_ms.size < 2:
        return np.array([])
    sorted_times = np.sort(spike_times_ms)
    return np.diff(sorted_times)


def compute_amplitudes(waveforms: np.ndarray) -> np.ndarray:
    """Compute peak-to-trough amplitude in µV for each waveform."""
    if waveforms.size == 0:
        return np.array([])
    return np.max(waveforms, axis=1) - np.min(waveforms, axis=1)


def compute_isi_violation_rate(isi_ms: np.ndarray, refractory_ms: float = 1.0) -> float:
    """Compute the fraction of ISIs that violate the refractory period (< 1 ms)."""
    if isi_ms.size == 0:
        return 0.0
    n_violations = np.sum(isi_ms < refractory_ms)
    return n_violations / len(isi_ms)


def plot_spike_validation_figure(
    waveforms: np.ndarray,
    spike_times_ms: np.ndarray,
    fs: float,
    group_title: str,
    n_channels: int,
    out_path: Path,
    pre_ms: float = 0.5,
    post_ms: float = 1.5,
    max_waveforms_plot: int = 200,
    max_waveforms_pca: int = 5000,
):
    """
    Generate a 4-panel spike validation figure for a channel group.
    
    Panels:
        A. Waveform overlay + mean ± 1.96 SEM
        B. PCA scatter (PC1 vs PC2)
        C. Amplitude histogram (log scale)
        D. ISI histogram (log scale)
    """
    n_spikes = waveforms.shape[0] if waveforms.size > 0 else 0
    n_samples = waveforms.shape[1] if waveforms.size > 0 else int((pre_ms + post_ms) / 1000.0 * fs)
    
    t_wf = np.linspace(-pre_ms, post_ms, n_samples)
    
    amplitudes = compute_amplitudes(waveforms) if n_spikes > 0 else np.array([])
    isi_ms = compute_isi(spike_times_ms)
    isi_violation_rate = compute_isi_violation_rate(isi_ms, refractory_ms=1.0)
    
    if n_spikes > max_waveforms_pca:
        pca_idx = np.random.choice(n_spikes, max_waveforms_pca, replace=False)
        waveforms_pca = waveforms[pca_idx]
    else:
        waveforms_pca = waveforms
    
    pca_coords, explained_var = compute_waveform_pca(waveforms_pca) if n_spikes >= 2 else (np.array([]).reshape(0, 2), np.zeros(2))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"{group_title} - Spike Validation\n"
        f"N = {n_spikes:,} spikes from {n_channels} channels | "
        f"ISI violations (<1ms): {isi_violation_rate*100:.1f}%",
        fontsize=12, fontweight='bold'
    )
    
    # ── Panel A: Waveform Overlay + Mean ± SEM ──
    ax_wf = axes[0, 0]
    ax_wf.set_title("A. Waveform Overlay", fontsize=11, fontweight='bold')
    
    if n_spikes > 0:
        if n_spikes > max_waveforms_plot:
            plot_idx = np.random.choice(n_spikes, max_waveforms_plot, replace=False)
            wf_plot = waveforms[plot_idx]
        else:
            wf_plot = waveforms
        
        for wf in wf_plot:
            ax_wf.plot(t_wf, wf, color='gray', alpha=0.1, lw=0.5)
        
        wf_mean = np.mean(waveforms, axis=0)
        wf_se = 1.96 * np.std(waveforms, axis=0) / np.sqrt(waveforms.shape[0])
        
        ax_wf.fill_between(t_wf, wf_mean - wf_se, wf_mean + wf_se, 
                           color='blue', alpha=0.3, label='±1.96 SEM')
        ax_wf.plot(t_wf, wf_mean, color='blue', lw=2, label='Mean')
        ax_wf.axvline(0, ls='--', color='red', alpha=0.5, lw=1, label='Peak')
        ax_wf.legend(loc='upper right', fontsize=8)
        
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
    
    # ── Panel B: PCA Scatter ──
    ax_pca = axes[0, 1]
    ax_pca.set_title("B. PCA of Waveforms", fontsize=11, fontweight='bold')
    
    if pca_coords.size > 0 and pca_coords.shape[0] > 1:
        if amplitudes.size > max_waveforms_pca:
            amp_pca = amplitudes[pca_idx] if n_spikes > max_waveforms_pca else amplitudes
        else:
            amp_pca = amplitudes[:pca_coords.shape[0]]
        
        scatter = ax_pca.scatter(
            pca_coords[:, 0], pca_coords[:, 1],
            c=amp_pca, cmap='viridis', alpha=0.5, s=10, edgecolors='none',
            vmin=0, vmax=500
        )
        cbar = plt.colorbar(scatter, ax=ax_pca, shrink=0.8)
        cbar.set_label("Amplitude (µV)", fontsize=9)
        
        ax_pca.text(0.02, 0.98, 
                    f"PC1: {explained_var[0]*100:.1f}%\nPC2: {explained_var[1]*100:.1f}%",
                    transform=ax_pca.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax_pca.text(0.5, 0.5, "Insufficient spikes for PCA", transform=ax_pca.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
    
    ax_pca.set_xlabel("PC1", fontsize=10)
    ax_pca.set_ylabel("PC2", fontsize=10)
    
    # ── Panel C: Amplitude Histogram (Log Scale) ──
    ax_amp = axes[1, 0]
    ax_amp.set_title("C. Amplitude Distribution (Log Scale)", fontsize=11, fontweight='bold')
    
    if amplitudes.size > 0:
        amp_min = max(amplitudes.min(), 1)
        amp_max = amplitudes.max()
        bins = np.logspace(np.log10(amp_min), np.log10(amp_max), 50)
        
        ax_amp.hist(amplitudes, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
        ax_amp.set_xscale('log')
        
        median_amp = np.median(amplitudes)
        ax_amp.axvline(median_amp, ls='--', color='red', lw=2, label=f'Median: {median_amp:.1f} µV')
        ax_amp.legend(loc='upper right', fontsize=9)
        
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
    
    # ── Panel D: ISI Histogram (Log Scale) ──
    ax_isi = axes[1, 1]
    ax_isi.set_title("D. Inter-Spike Interval Distribution", fontsize=11, fontweight='bold')
    
    if isi_ms.size > 0:
        isi_positive = isi_ms[isi_ms > 0]
        if isi_positive.size > 0:
            max_isi = min(np.max(isi_positive), 1000)
            bins = np.logspace(np.log10(0.1), np.log10(max_isi), 50)
            
            ax_isi.hist(isi_positive, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
            ax_isi.set_xscale('log')
            
            ax_isi.axvline(1.0, ls='--', color='red', lw=2, label='1 ms (refractory)')
            ax_isi.axvline(2.0, ls=':', color='orange', lw=1.5, label='2 ms')
            ax_isi.axvspan(0.1, 1.0, color='red', alpha=0.1)
            ax_isi.legend(loc='upper right', fontsize=9)
            
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
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.svg'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {out_path.name}")


def collect_group_spikes(
    rec,
    group_channels: list,
    peak_ms_dict: dict,
    rec_start_ms: float,
    fs: float,
    pre_ms: float = 0.5,
    post_ms: float = 1.5,
) -> tuple:
    """
    Collect all spike waveforms and times for a group of NPRW channels.
    """
    all_waveforms = []
    all_spike_times = []
    n_channels_with_spikes = 0
    ch_ids = rec.get_channel_ids()
    
    for ch_idx in group_channels:
        cid = ch_ids[ch_idx]
        
        # Robust lookup by int, int64, or string ID
        ch_peaks_abs = peak_ms_dict.get(ch_idx, None)
        if ch_peaks_abs is None:
            ch_peaks_abs = peak_ms_dict.get(int(ch_idx), None)
        if ch_peaks_abs is None:
            ch_peaks_abs = peak_ms_dict.get(np.int64(ch_idx), None)
        if ch_peaks_abs is None:
            ch_peaks_abs = peak_ms_dict.get(str(ch_idx), None)
        if ch_peaks_abs is None:
            ch_peaks_abs = peak_ms_dict.get(str(cid), [])
            
        ch_peaks_abs = np.asarray(ch_peaks_abs, float).ravel()
        if ch_peaks_abs.size == 0:
            continue
        
        ch_peaks_local = ch_peaks_abs - rec_start_ms
        rec_dur_ms = rec.get_num_frames() / fs * 1000.0
        valid_mask = (ch_peaks_local >= pre_ms) & (ch_peaks_local <= rec_dur_ms - post_ms)
        ch_peaks_valid = ch_peaks_local[valid_mask]
        
        if ch_peaks_valid.size == 0:
            continue
        
        waveforms = extract_spike_waveforms(rec, ch_idx, ch_peaks_valid, fs, pre_ms, post_ms)
        if waveforms.size > 0:
            all_waveforms.append(waveforms)
            all_spike_times.extend(ch_peaks_valid.tolist())
            n_channels_with_spikes += 1
            
    if all_waveforms:
        all_waveforms = np.vstack(all_waveforms)
    else:
        n_samples = int((pre_ms + post_ms) / 1000.0 * fs)
        all_waveforms = np.array([]).reshape(0, n_samples)
        
    all_spike_times = np.array(all_spike_times)
    return all_waveforms, all_spike_times, n_channels_with_spikes


def main():
    print("=" * 70)
    print(f"Plotting HPF Traces for NPRW Probe  |  Condition BR_{CONDITION:03d}")
    print("=" * 70)

    # ── 1. Load aligned file for this condition ──
    aligned_path = find_aligned_file(ALIGNED_CKPT, CONDITION)
    print(f"[info] Aligned file: {aligned_path.name}")
    z = np.load(aligned_path, allow_pickle=True)

    # ── 2. Parse metadata ──
    if "align_meta" in z.files:
        raw_meta = z["align_meta"].item() if z["align_meta"].ndim == 0 else z["align_meta"]
        meta = raw_meta if isinstance(raw_meta, dict) else json.loads(raw_meta)
    else:
        meta = {}
    intan_filename = meta.get("intan_filename", "")
    shift_ms = float(meta.get("shift_ms", 0.0))

    if EVENT_TYPE == "stim":
        event_ms = z.get("stim_ms", np.array([], dtype=float))
        event_label = "stimulation"
    else:
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
        return

    # ── 3. Time alignment for NPRW ──
    probe_meta = z["nprw_meta"].item() if "nprw_meta" in z.files else {}
    rec_start_ms = float(probe_meta.get("rec_start_ms_aligned", 0.0))
    if rec_start_ms == 0.0 and "rec_start_ms" in probe_meta:
        rec_start_ms = float(probe_meta["rec_start_ms"]) - shift_ms

    pk_obj = z.get("nprw_peak_ms_dedup", z.get("nprw_peak_ms", None))
    peak_ms_dict = pk_obj.reshape(-1)[0] if (isinstance(pk_obj, np.ndarray) and pk_obj.dtype == object) else pk_obj
    if peak_ms_dict is None:
        peak_ms_dict = {}

    event_local_ms = event_ms - rec_start_ms
    print(f"[info] Aligned event_ms range: {event_ms.min():.1f} – {event_ms.max():.1f} ms")
    print(f"[info] NPRW rec_start_ms in aligned frame: {rec_start_ms:.1f} ms")
    print(f"[info] NPRW local event_ms range: {event_local_ms.min():.1f} – {event_local_ms.max():.1f} ms")

    # ── 4. Stimulation Duration ──
    stim_dur = 0.0
    if EVENT_TYPE == "stim":
        from RCP_analysis.python.functions.config_loading import METADATA_CSV, NPRW_AUX_DATA
        stim_npz_path, _ = rcp.stim_npz_path_from_br_idx(CONDITION, METADATA_CSV, NPRW_AUX_DATA)
        fs_nprw = float(meta.get("fs_nprw", 30000.0))
        
        if stim_npz_path and stim_npz_path.exists():
            stim = rcp.load_stim_detection(stim_npz_path)
            block_bounds = stim.get("block_bounds_samples", np.empty((0, 2), dtype=np.int64))
            if len(block_bounds) > 0:
                if isinstance(block_bounds, list):
                    block_bounds = np.array(block_bounds, dtype=np.int64)
                nprw_samps = block_bounds[:, 0].astype(np.int64)
                nprw_ends  = block_bounds[:, 1].astype(np.int64)
                stim_dur   = float((nprw_ends - nprw_samps).max() * 1000.0 / fs_nprw)
        
        if stim_dur == 0.0:
            if "stim_dur_measured_ms" in z.files and len(z["stim_dur_measured_ms"]) > 0:
                stim_dur = float(np.max(z["stim_dur_measured_ms"]))
            elif "stim_dur_nominal_ms" in z.files:
                stim_dur = float(z["stim_dur_nominal_ms"])
    print(f"[info] Stimulation duration: {stim_dur:.1f} ms")

    # ── 5. Load SpikeInterface recording ──
    pp_folder = find_pp_folder_for_intan(CKPT_DIR, intan_filename, PROBE, CONDITION)
    rec = si.load(str(pp_folder))
    fs  = float(rec.get_sampling_frequency())
    n_ch = rec.get_num_channels()
    rec_dur_ms = rec.get_num_frames() / fs * 1000.0
    print(f"[info] Loaded recording: {pp_folder.name} | fs={fs:.0f} Hz | {n_ch} channels | {rec_dur_ms/1000.0:.1f} s")

    # ── 6. Channel resolution and Probe Geometry ──
    ch_ids = rec.get_channel_ids()
    try:
        locations = rec.get_channel_locations()
        has_locations = (locations is not None and len(locations) == n_ch)
    except Exception:
        locations = None
        has_locations = False

    resolved_channels = [c for c in CHANNELS if 0 <= c < n_ch]
    if not resolved_channels:
        print(f"[error] No valid channels found in {CHANNELS} (recording has {n_ch} channels).")
        return

    # If GROUP_BY_DEPTH is requested and locations exist, sort channels by physical depth (tip to surface)
    if GROUP_BY_DEPTH and has_locations:
        resolved_channels.sort(key=lambda c: locations[c][1])

    # ── 7. Partition channels into depth groups / figures ──
    group_size = CHANNELS_PER_FIG if (CHANNELS_PER_FIG is not None and CHANNELS_PER_FIG > 0) else len(resolved_channels)
    
    channel_groups = {}
    for g_idx in range(0, len(resolved_channels), group_size):
        chunk = resolved_channels[g_idx : g_idx + group_size]
        ch_min = min(chunk)
        ch_max = max(chunk)
        
        if has_locations:
            y_min = min(locations[c][1] for c in chunk)
            y_max = max(locations[c][1] for c in chunk)
            g_name = f"Group_{g_idx // group_size + 1}__Ch{ch_min:03d}-{ch_max:03d}__y{int(y_min)}-{int(y_max)}um"
            g_title = f"NPRW Group {g_idx // group_size + 1} (Ch {ch_min}–{ch_max} | Depth {int(y_min)}–{int(y_max)} µm)"
        else:
            g_name = f"Group_{g_idx // group_size + 1}__Ch{ch_min:03d}-{ch_max:03d}"
            g_title = f"NPRW Group {g_idx // group_size + 1} (Ch {ch_min}–{ch_max})"
            
        channel_groups[g_name] = {"channels": chunk, "title": g_title}

    print(f"[info] Formed {len(channel_groups)} channel group(s) ({group_size} channels/fig max).")

    # ── 8. Filter events fitting in recording window ──
    valid = []
    for s in event_local_ms:
        if (s + WINDOW_MS[0]) >= 0 and (s + WINDOW_MS[1]) <= rec_dur_ms:
            valid.append(s)
    valid = np.array(valid)
    if valid.size == 0:
        print(f"[error] No {event_label} events fit within the recording for window {WINDOW_MS}.")
        return
    print(f"[info] Valid {event_label} events: {valid.size} / {event_ms.size}")

    if isinstance(TRIALS_TO_PLOT, str) and TRIALS_TO_PLOT.lower() == "all":
        plot_idxs = np.arange(len(valid))
    else:
        plot_idxs = np.array([i for i in TRIALS_TO_PLOT if i < len(valid)])

    # ── 9. Plot Traces for each trial and group ──
    for trial_i in plot_idxs:
        center_time = valid[trial_i]
        event_ms_trial = center_time + rec_start_ms
        
        for g_name, g_info in channel_groups.items():
            grp_channels = g_info["channels"]
            grp_title = g_info["title"]
            n_rows = len(grp_channels)
            
            if PLOT_WAVEFORM:
                fig = plt.figure(figsize=(15, max(4.0, 1.8 * n_rows)))
                gs = matplotlib.gridspec.GridSpec(n_rows, 2, width_ratios=[8, 1])
                axes_trace = []
                axes_wv = []
                for r in range(n_rows):
                    ax_t = fig.add_subplot(gs[r, 0], sharex=axes_trace[0] if r > 0 else None, sharey=axes_trace[0] if r > 0 else None)
                    ax_w = fig.add_subplot(gs[r, 1], sharex=axes_wv[0] if r > 0 else None, sharey=ax_t)
                    axes_trace.append(ax_t)
                    axes_wv.append(ax_w)
            else:
                fig, axes_raw = plt.subplots(n_rows, 1, figsize=(12, max(4.0, 1.8 * n_rows)),
                                             sharex=True, sharey=True)
                axes_trace = np.atleast_1d(axes_raw)
                axes_wv = [None] * n_rows
                
            for r, ch in enumerate(grp_channels):
                ax = axes_trace[r]
                ax_wv = axes_wv[r]
                cid = ch_ids[ch]
                
                t, y = extract_traces(rec, ch, center_time, WINDOW_MS, fs)
                if t is not None:
                    ax.plot(t, y, lw=0.7, color="k")
                    
                # Y-label with depth coordinate if available
                if has_locations:
                    y_coord = locations[ch][1]
                    ch_label = f"Ch {ch}\n({y_coord:.0f}µm)"
                else:
                    ch_label = f"Ch {ch}"
                ax.set_ylabel(ch_label, fontsize=8)

                # Overlay spikes
                ch_peaks_abs = peak_ms_dict.get(ch, None)
                if ch_peaks_abs is None: ch_peaks_abs = peak_ms_dict.get(int(ch), None)
                if ch_peaks_abs is None: ch_peaks_abs = peak_ms_dict.get(np.int64(ch), None)
                if ch_peaks_abs is None: ch_peaks_abs = peak_ms_dict.get(str(ch), None)
                if ch_peaks_abs is None: ch_peaks_abs = peak_ms_dict.get(str(cid), [])
                ch_peaks_abs = np.asarray(ch_peaks_abs, float).ravel()

                pk_rel = ch_peaks_abs - event_ms_trial
                pk_rel_win = pk_rel[(pk_rel >= WINDOW_MS[0]) & (pk_rel <= WINDOW_MS[1])]
                
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
                        ax_wv.set_xlim(-1.0, 1.5)
                        ax_wv.axvline(0, ls="--", color="r", alpha=0.5)
                        if r == n_rows - 1:
                            ax_wv.set_xlabel("Time (ms)")

                # Stimulation / Movement indicators
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
                f"{grp_title}  •  Trial {trial_i + 1}",
                fontsize=11, y=0.99 if n_rows < 10 else 1.0 - (0.5 / n_rows),
            )
            fig.tight_layout(rect=[0, 0, 1, 0.98 if n_rows < 10 else 1.0 - (1.0 / n_rows)])

            event_suffix = "stim" if EVENT_TYPE == "stim" else "ctrl"
            out_path = (
                FIG_DIR
                / f"{SESSION}__cond{CONDITION:03d}__NPRW__{event_suffix}__{g_name}"
                  f"__trial{trial_i + 1}__win{int(WINDOW_MS[0])}-{int(WINDOW_MS[1])}ms.png"
            )
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] {out_path.name}")

    # ── 10. Spike Validation Figures (per channel group) ──
    print("\n" + "=" * 70)
    print("Generating spike validation figures per NPRW channel group...")
    print("=" * 70)
    
    VALIDATION_DIR = FIG_DIR / "spike_validation"
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    
    for g_name, g_info in channel_groups.items():
        grp_channels = g_info["channels"]
        grp_title = g_info["title"]
        
        print(f"\n[processing] {grp_title} ({len(grp_channels)} channels)...")
        all_waveforms, all_spike_times, n_ch_with_spikes = collect_group_spikes(
            rec=rec,
            group_channels=grp_channels,
            peak_ms_dict=peak_ms_dict,
            rec_start_ms=rec_start_ms,
            fs=fs,
            pre_ms=0.5,
            post_ms=1.5,
        )
        print(f"    Collected {all_waveforms.shape[0]:,} waveforms from {n_ch_with_spikes} channels")
        
        event_suffix = "stim" if EVENT_TYPE == "stim" else "ctrl"
        out_path = (
            VALIDATION_DIR 
            / f"{SESSION}__cond{CONDITION:03d}__NPRW__{event_suffix}__SpikeValidation__{g_name}.png"
        )
        
        plot_spike_validation_figure(
            waveforms=all_waveforms,
            spike_times_ms=all_spike_times,
            fs=fs,
            group_title=f"{grp_title} ({SESSION} - Cond {CONDITION})",
            n_channels=n_ch_with_spikes,
            out_path=out_path,
            pre_ms=0.5,
            post_ms=1.5,
        )

    print("\n[done] NPRW HPF trace plots and validation figures complete.")


if __name__ == "__main__":
    main()
