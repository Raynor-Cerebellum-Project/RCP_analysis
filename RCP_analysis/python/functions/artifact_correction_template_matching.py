from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Literal
import numpy as np
from scipy.signal import filtfilt, medfilt, windows
from sklearn.decomposition import PCA
import spikeinterface as si
try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

Backend = Literal["threading", "loky"]  # threads share memory; loky = processes (copies arrays)

# ----------------------------- Params -----------------------------------------
@dataclass
class PCAArtifactParams:
    # 1) Drift removal
    rolling_median_ms: float = 15.0
    gaussian_sigma_ms: float = 5.0
    gaussian_len_ms: float = 31.0

    # 2) Pulse-aligned matrices
    pre_samples: int = 13            # window start = trig_start - 13
    post_pad_samples: int = 15       # window end   = trig_end   + 15

    # 3) PCA template options
    center_snippets: bool = True
    first_pulse_special: bool = True
    exclude_first_n_for_pca: int = 1   # exclude N first pulses in each block when building template

    # 4) Subtraction options
    scale_amplitude: bool = True       # per-pulse, per-channel scalar

    # 5) Interpulse ramp
    interp_ramp: bool = True
    ramp_tail_ms: float = 1.0          # start ramp after artifact end + tail
    ramp_fraction: float = 1.0         # fraction of edge value to ramp to baseline

# ------------------------- NPZ loader -----------------------------------------

def load_stim_detection(npz_path: Path) -> Dict[str, np.ndarray]:
    """
    Required fields (your new format):
      - active_channels     : (n_active,)
      - trigger_pairs       : (n_pulses, 2) [start, end] in Intan index space
      - block_boundaries    : (n_blocks+1,) boundaries in Intan index space
      - pulse_sizes         : (n_pulses,)
    """
    with np.load(npz_path, allow_pickle=False) as z:
        active_channels = z.get("active_channels", np.array([], dtype=np.int64)).astype(np.int64)
        trigger_pairs = z["trigger_pairs"].astype(np.int64)
        block_boundaries = z["block_boundaries"].astype(np.int64)
        pulse_sizes = z["pulse_sizes"].astype(np.int64)

    return dict(
        active_channels=active_channels,
        trigger_pairs=trigger_pairs,
        block_boundaries=block_boundaries,
        pulse_sizes=pulse_sizes,
    )

# ---------------------- Helpers: drift removal, snippets, PCA -----------------
def _global_drift_remove(
    traces: np.ndarray,
    fs: float,
    p: PCAArtifactParams,
    *,
    n_jobs: Optional[int] = None,
    backend: Backend = "threading",
) -> np.ndarray:
    """
    Global drift removal:
      1) Rolling median (vectorized across channels)
      2) Zero-phase Gaussian smoothing removal via filtfilt

    If n_jobs is None -> single vectorized filtfilt across all channels (fast).
    If n_jobs is an int -> parallelize per-channel with joblib.

    backend='threading' (default) keeps memory shared (good for large arrays).
    backend='loky' uses processes (avoids GIL but copies data to workers).
    """
    out = traces

    # --- rolling median (vectorized across channels) ---
    med_window = max(1, int(round(p.rolling_median_ms * fs / 1000.0)))
    if med_window % 2 == 0:
        med_window += 1
    if med_window > 1:
        out = out - medfilt(out, kernel_size=(med_window, 1))

    # --- gaussian kernel for filtfilt ---
    gauss_window = max(3, int(round(p.gaussian_len_ms * fs / 1000.0)))
    if gauss_window % 2 == 0:
        gauss_window += 1
    sigma = max(1.0, p.gaussian_sigma_ms * fs / 1000.0)
    b = windows.gaussian(M=gauss_window, std=sigma, sym=True).astype(np.float32)
    b /= (b.sum() + 1e-12)

    # filtfilt padding requirement
    pad_needed = 3 * (len(b) - 1)
    if out.shape[0] <= pad_needed:
        return out

    padlen = min(pad_needed, out.shape[0] - 1)

    # === Path A: single vectorized filtfilt along axis=0 (recommended default) ===
    if n_jobs is None:
        smooth = filtfilt(b, [1.0], out, axis=0, padlen=padlen)
        return out - smooth

    # === Path B: joblib parallel per channel ===
    if not _HAS_JOBLIB:
        # Fallback to vectorized if joblib isn't available
        smooth = filtfilt(b, [1.0], out, axis=0, padlen=padlen)
        return out - smooth

    n_ch = out.shape[1]

    # Use threading by default: SciPy's filtfilt is C-coded and typically releases the GIL,
    # so threads can run concurrently *without* copying the big array.
    # Switch to backend='loky' if you prefer processes (at the cost of copying).
    def _one(ch: int) -> np.ndarray:
        x = out[:, ch]  # view
        y = filtfilt(b, [1.0], x, padlen=padlen)
        return (x - y).astype(out.dtype, copy=False)

    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_one)(ch) for ch in range(n_ch)
    )
    return np.stack(results, axis=1)


def _block_window_lengths(trigger_pairs_window: np.ndarray, pre: int, post_pad: int) -> np.ndarray:
    """
    For a block: per-pulse target window length = (end + post_pad) - (start - pre) + 1
    """
    starts = trigger_pairs_window[:, 0]
    ends   = trigger_pairs_window[:, 1]
    return (ends + post_pad) - (starts - pre) + 1

def _extract_pulse_snippets(
    clean: np.ndarray,                 # (n_samples, n_channels)
    trigger_pairs_window: np.ndarray,   # (m, 2) [start, end] inclusive indices
    pre: int,
    post_pad: int,
    target_len: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Extract snippets aligned to pulse start with window [start-pre : end+post_pad] (inclusive).
    Pads/truncates each to a uniform length L (default: max per-pulse length in for all pulses).

    Returns
    -------
    snips : (n_keep, L, n_channels) float32
    keep_mask : (m,) bool
    L : int
    """
    n_trigs = trigger_pairs_window.shape[0]
    n_Samps, n_channels = clean.shape

    if n_trigs == 0:
        return np.zeros((0, 0, n_channels), dtype=np.float32), np.zeros((0,), bool), 0

    # Per-pulse window bounds (inclusive end)
    starts = trigger_pairs_window[:, 0] - pre
    ends   = trigger_pairs_window[:, 1] + post_pad

    # Length each pulse would have if in-bounds:
    # because we’ll slice with b = end + 1 (python exclusive)
    win_lens = (ends - starts + 1).astype(int)

    # Choose uniform length
    L = int(target_len) if target_len is not None else int(np.max(win_lens))

    # Preallocate (n_trigs, L, nCannel). We'll compact at the end using keep mask.
    snips = np.empty((n_trigs, L, n_channels), dtype=np.float32)
    keep  = np.zeros((n_trigs,), dtype=bool)

    for i in range(n_trigs):
        a = int(starts[i])
        b = int(ends[i]) + 1  # python slice end

        if a < 0 or b > n_Samps:
            # Out of bounds -> skip
            continue

        S = clean[a:b, :]  # shape (win_lens[i], nChannels)

        # Pad/truncate to L (edge pad so last sample repeats)
        cur_len = S.shape[0]
        if cur_len < L:
            pad_len = L - cur_len
            # np.pad with mode='edge' avoids extra allocations from vstack
            S = np.pad(S, ((0, pad_len), (0, 0)), mode='edge')
        elif cur_len > L:
            S = S[:L, :]

        snips[i, :, :] = S.astype(np.float32, copy=False)
        keep[i] = True

    if not keep.any():
        # No valid pulses
        return np.zeros((0, L, n_channels), dtype=np.float32), keep, L

    # Compact to only valid pulses
    return snips[keep, :, :], keep, L

# TODO return array of PCA object or just rewrite to use 3 PCs
def _pca_template_per_channel(
    snips: np.ndarray,
    center: bool
) -> Tuple[np.ndarray, List[PCA]]:
    """
    Compute per-channel PCA templates and return fitted PCA objects.

    Parameters
    ----------
    snips : np.ndarray
        (n_pulses, L, n_channels) snippets
    center : bool
        If True, subtract timepoint-wise mean across pulses before PCA.

    Returns
    -------
    templ : np.ndarray
        (L, n_channels) template reconstructed from top-3 PCs.
    pcas : list of PCA
        One PCA object per channel (n_components <= 3).
    """
    n_pulses, L, n_ch = snips.shape
    if n_pulses == 0:
        return np.zeros((L, n_ch), dtype=np.float32), []
    if n_pulses == 1:
        return snips[0].astype(np.float32), []

    templates = np.zeros((L, n_ch), dtype=np.float32)
    pcas: List[PCA] = []

    for ch in range(n_ch):
        X = snips[:, :, ch].astype(np.float64, copy=False)
        k = min(3, n_pulses, L)

        mu = X.mean(axis=0)  # (L,)
        Xc = X - mu if center else X

        p = PCA(n_components=k, svd_solver="full")
        p.fit(Xc)
        pcas.append(p)

        # reconstruct the mean snippet from top-k PCs
        base = mu if center else p.mean_
        mean_snip = mu
        scores = (mean_snip - base) @ p.components_.T   # (k,)
        recon = base + scores @ p.components_           # (L,)

        templates[:, ch] = recon.astype(np.float32)

    return templates, pcas

# TODO: make sure we're not going between blocks
def _apply_interp_ramp(buf: np.ndarray, start_idx: int, end_idx: int, frac: float):
    """
    Linearly ramp from current edge value to 0 over [start_idx, end_idx).
    """
    if end_idx <= start_idx or start_idx < 0 or start_idx >= buf.shape[0]:
        return
    end_idx = min(end_idx, buf.shape[0])
    L = end_idx - start_idx
    if L <= 0:
        return
    v0 = buf[start_idx, :] * float(frac)
    ramp = v0[None, :] * (1.0 - np.linspace(0, 1, L, endpoint=False)[:, None])
    buf[start_idx:end_idx, :] -= ramp.astype(buf.dtype)

# --------------------------- Main pipeline ------------------------------------
def remove_stim_pca_offline(
    recording: si.BaseRecording,
    stim_npz_path: Path,
    params: Optional[PCAArtifactParams] = None,
    segment_index: int = 0,
) -> np.ndarray:
    """
    Returns cleaned traces for one segment as a NumPy array (n_samples, n_channels).

    Steps:
      1) Global drift removal
      2) Pulse-aligned matrices with [-13, +15 after pulse end] windows
      3) PCA template per block (exclude_first_for_pca if requested)
      4) Template subtraction (optional per-pulse scaling)
      5) Interpulse linear drift correction (artifact end + tail -> next trigger)
    """
    pca_params = params or PCAArtifactParams()
    fs = float(recording.get_sampling_frequency())
    ramp_tail = int(round(pca_params.ramp_tail_ms * fs / 1000.0))

    # Grab traces and pre-clean
    n_Samps = int(recording.get_num_samples(segment_index=segment_index))
    
    #TODO see if you can use get traces on small chunks of data to save memory and paralleize across chunks, move get_traces into global_drift_remove
    raw = recording.get_traces(
        start_frame=0, end_frame=n_Samps, channel_ids=None, segment_index=segment_index, return_scaled=True
    )
    import time

    t0 = time.time()
    clean = _global_drift_remove(raw, fs, pca_params, n_jobs=8, backend="threading")
    t1 = time.time()

    print(f"drift removal took {t1 - t0:.2f} seconds")
    
    # Load stim detection data
    stim_data = load_stim_detection(stim_npz_path)
    trigger_pairs = stim_data["trigger_pairs"]        # (n_pulses, 2)
    block_bounds  = stim_data["block_boundaries"]     # (n_blocks+1,)
    # pulse_sizes  = stim_data["pulse_sizes"]         # not needed explicitly here

    if trigger_pairs.shape[0] == 0 or block_bounds.size < 2:
        return clean  # nothing to do

    # Build per-block templates
    # We will (a) compute a uniform L per block = max window length among pulses in the block
    #     and (b) build PCA template using 2 to end pulses
    n_total_pulses = trigger_pairs.shape[0]
    n_blocks = block_bounds.size - 1
    templates_by_block: Dict[int, np.ndarray] = {}
    pulse_len_each_block: Dict[int, int] = {}

    for block in range(n_blocks):
        block_beg, block_end = int(block_bounds[block]), int(block_bounds[block + 1])
        if block_end <= block_beg:
            continue
        tp_block = trigger_pairs[block_beg:block_end, :]

        # Uniform window length per block (max of per-pulse [start-pre : end+post]), assuming pulse lengths are the same within a block
        all_pulse_len = _block_window_lengths(tp_block, pre=pca_params.pre_samples, post_pad=pca_params.post_pad_samples)
        pulse_len = int(np.max(all_pulse_len))
        pulse_len_each_block[block] = pulse_len

        # Pulses used for PCA template (optionally drop first one (or N))
        if pca_params.first_pulse_special and pca_params.exclude_first_n_for_pca > 0:
            use_rows = np.arange(tp_block.shape[0])[pca_params.exclude_first_n_for_pca:]
        else:
            use_rows = np.arange(tp_block.shape[0])

        if use_rows.size == 0:
            templates_by_block[block] = np.zeros((pulse_len, clean.shape[1]), dtype=np.float32)
            # no pulses so just zero for # pulses times # channels
            continue

        snips_block, keep_mask, L_real = _extract_pulse_snippets(
            clean, tp_block[use_rows], pre=pca_params.pre_samples, post_pad=pca_params.post_pad_samples, target_len=pulse_len
        )
        if snips_block.shape[0] == 0:
            templates_by_block[block], _ = np.zeros((pulse_len, clean.shape[1]), dtype=np.float32)
        else:
            templates_by_block[block], _ = _pca_template_per_channel(snips_block, center=pca_params.center_snippets)

    # Subtract templates pulse-by-pulse (with optional per-channel amplitude scaling)
    # Also apply interp ramp from (artifact_end + tail) to next trigger start.
    for pulse in range(n_total_pulses):
        # Which block is this pulse in?
        # Find b such that block_bounds[b] <= pulse < block_bounds[b+1]
        block = int(np.searchsorted(block_bounds[1:], pulse, side="right"))
        if block not in templates_by_block:
            continue
        template = templates_by_block[block]
        pulse_len = pulse_len_each_block[block]
        if template.size == 0:
            continue

        trig_start = int(trigger_pairs[pulse, 0])
        trig_end   = int(trigger_pairs[pulse, 1])

        # Apply subtraction on window [temp_beg:temp_end)
        temp_beg = trig_start - pca_params.pre_samples
        temp_end = temp_beg + pulse_len
        if temp_beg < 0 or temp_end > clean.shape[0]:
            # out of bounds; skip this pulse
            continue

        patch = clean[temp_beg:temp_end, :]           # (L_blk, n_ch)

        # TODO fix here for 3 PCs
        if pca_params.scale_amplitude:
            num = np.sum(patch * template, axis=0)                 # (n_ch,)
            den = np.sum(template * template, axis=0) + 1e-12
            amp = num / den                                   # (n_ch,)
            patch -= (template * amp[None, :]).astype(patch.dtype)
            
            # TODO projection = PCA.inverse_transform(PCA.transform(clean[a:bnd, ch]))
            # TODO clean[temp_beg:temp_end, ch] -= projection
            # TODO to get the final corrected data
        else:
            patch -= template.astype(patch.dtype)

        clean[temp_beg:temp_end, :] = patch

        # Interpulse ramp
        if pca_params.interp_ramp:
            ramp_start = min(trig_end + ramp_tail, clean.shape[0] - 1)
            # find next trigger start (absolute)
            if pulse + 1 < n_total_pulses:
                next_start = int(trigger_pairs[pulse + 1, 0])
                if next_start > ramp_start:
                    _apply_interp_ramp(clean, ramp_start, next_start, pca_params.ramp_fraction)

    return clean

# ------------------------------ wrap as SI recording ----------------
def cleaned_numpy_to_recording(
    cleaned: np.ndarray, recording_like: si.BaseRecording
) -> si.BaseRecording:
    """
    Convenience: turn cleaned array back into a SpikeInterface NumpyRecording,
    preserving sampling rate and channel ids.
    """
    from spikeinterface.core import NumpyRecording
    fs = float(recording_like.get_sampling_frequency())
    chan_ids = list(recording_like.get_channel_ids())
    rec = NumpyRecording(traces_list=[cleaned], sampling_frequency=fs)
    rec.set_channel_ids(chan_ids)
    # Propagate probe / geometry if present
    try:
        recording_like.copy_metadata(rec, only_main=False)
    except Exception:
        pass
    return rec
