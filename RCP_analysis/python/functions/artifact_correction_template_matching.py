from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
from scipy.signal import filtfilt, medfilt, windows
from sklearn.decomposition import PCA
import spikeinterface as si

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
    exclude_first_n_for_pca: int = 1   # exclude N first events in each block when building template

    # 4) Subtraction options
    scale_amplitude: bool = True       # per-event, per-channel scalar

    # 5) Interpulse ramp
    interp_ramp: bool = True
    ramp_tail_ms: float = 1.0          # start ramp after artifact end + tail
    ramp_fraction: float = 1.0         # fraction of edge value to ramp to baseline

# ------------------------- NPZ loader -----------------------------------------

def load_stim_detection(npz_path: Path) -> Dict[str, np.ndarray]:
    """
    Required fields (your new format):
      - active_channels     : (n_active,)
      - trigger_pairs       : (n_events, 2) [start, end] (ABSOLUTE sample indices)
      - block_boundaries    : (n_blocks+1,) boundaries in EVENT index space
      - pulse_sizes         : (n_events,)
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
def _global_drift_remove(traces: np.ndarray, fs: float, p: PCAArtifactParams) -> np.ndarray:
    """
    traces: (n_samples, n_channels)
    """
    out = traces

    # rolling median
    med_window = max(1, int(round(p.rolling_median_ms * fs / 1000.0)))
    if med_window % 2 == 0:
        med_window += 1
    if med_window > 1:
        out = out - medfilt(out, kernel_size=(med_window, 1))

    # TODO paralleize across channels
    # zero-phase gaussian smoothing removal
    gauss_window = max(3, int(round(p.gaussian_len_ms * fs / 1000.0)))
    if gauss_window % 2 == 0:
        gauss_window += 1
    sigma = max(1.0, p.gaussian_sigma_ms * fs / 1000.0)
    gaussian_win = windows.gaussian(M=gauss_window, std=sigma, sym=True).astype(np.float32)
    gaussian_win /= (gaussian_win.sum() + 1e-12)

    pad_needed = 3 * (len(gaussian_win) - 1)
    if out.shape[0] > pad_needed:
        for ch in range(out.shape[1]):
            out[:, ch] = out[:, ch] - filtfilt(gaussian_win, [1.0], out[:, ch],
                                               padlen=min(pad_needed, out.shape[0] - 1))
    return out

def _block_window_lengths(trigger_pairs_block: np.ndarray, pre: int, post_pad: int) -> np.ndarray:
    """
    For a block: per-event target window length = (end + post_pad) - (start - pre) + 1
    """
    starts = trigger_pairs_block[:, 0]
    ends   = trigger_pairs_block[:, 1]
    return (ends + post_pad) - (starts - pre) + 1

def _extract_block_snippets(
    clean: np.ndarray,             # (n_samples, n_channels)
    trigger_pairs_block: np.ndarray, # (m, 2)
    pre: int,
    post_pad: int,
    target_len: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Extracts snippets aligned to pulse start with window [start-pre : end+post_pad].
    Pads to a uniform 'target_len' (default = max per-event length in block).
    Returns:
      snips: (n_keep, L, n_channels)
      keep_mask: (m,) bool for which events were valid
      L: length used
    """
    m = trigger_pairs_block.shape[0]
    if m == 0:
        return np.zeros((0, 0, clean.shape[1]), dtype=clean.dtype), np.zeros((0,), bool), 0

    # per-event window bounds
    starts = trigger_pairs_block[:, 0] - pre
    ends   = trigger_pairs_block[:, 1] + post_pad
    win_lens = ends - starts + 1
    L = int(target_len) if target_len is not None else int(np.max(win_lens))

    #TODO: preallocate snips
    snips = []
    keep = np.zeros((m,), dtype=bool)
    nS, nC = clean.shape

    for i in range(m):
        a = int(starts[i])
        b = int(ends[i]) + 1  # python slice end
        if a < 0 or b > nS:
            # out of bounds -> skip
            continue

        S = clean[a:b, :]  # (win_lens[i], nC)
        # right-pad (edge) to L for PCA consistency
        if S.shape[0] < L:
            pad_len = L - S.shape[0]
            pad = np.repeat(S[-1:, :], pad_len, axis=0)
            S = np.vstack([S, pad])
        elif S.shape[0] > L:
            S = S[:L, :]

        snips.append(S)
        keep[i] = True

    if len(snips) == 0:
        return np.zeros((0, L, nC), dtype=clean.dtype), keep, L

    return np.stack(snips, axis=0), keep, L

# TODO return array of PCA object or just rewrite to use 3 PCs
def _pca_template_per_channel(snips: np.ndarray, center: bool) -> np.ndarray:
    """
    TODO rewrite this to use 3 PCs
    snips: (n_events, L, n_channels) -> template: (L, n_channels)
    Uses the first PC per channel; sign aligned to mean snippet.
    """
    n_ev, L, n_ch = snips.shape
    if n_ev == 0:
        return np.zeros((L, n_ch), dtype=np.float32)
    if n_ev == 1:
        return snips[0].astype(np.float32)

    templ = np.zeros((L, n_ch), dtype=np.float32)
    for ch in range(n_ch):
        X = snips[:, :, ch]
        if center:
            X = X - X.mean(axis=0, keepdims=True)
        p = PCA(n_components=3)
        p.fit(X)
        comp = p.components_[0:3]  # (L,)
        m = snips[:, :, ch].mean(axis=0)
        if np.sum(comp * m) < 0:
            comp = -comp
        templ[:, ch] = comp.astype(np.float32)
    return templ

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
    v0 = buf[start_idx, :].astype(np.float32) * float(frac)
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
      4) Template subtraction (optional per-event scaling)
      5) Interpulse linear drift correctio (artifact end + tail -> next trigger)
    """
    pca_params = params or PCAArtifactParams()
    fs = float(recording.get_sampling_frequency())
    ramp_tail = int(round(pca_params.ramp_tail_ms * fs / 1000.0))

    # Grab traces and pre-clean
    nS = int(recording.get_num_samples(segment_index=segment_index))
    
    #TODO see if you can use get traces on small chunks of data to save memory and paralleize across chunks, move get_traces into global_drift_remove
    raw = recording.get_traces(
        start_frame=0, end_frame=nS, channel_ids=None, segment_index=segment_index, return_scaled=True
    )
    # SpikeInterface returns (n_samples, n_channels)
    clean = _global_drift_remove(raw, fs, pca_params)

    # Load stim detection data
    D = load_stim_detection(stim_npz_path)
    trigger_pairs = D["trigger_pairs"]        # (n_events, 2)
    block_bounds  = D["block_boundaries"]     # (n_blocks+1,)
    # pulse_sizes  = D["pulse_sizes"]         # not needed explicitly here

    if trigger_pairs.shape[0] == 0 or block_bounds.size < 2:
        return clean  # nothing to do

    # Build per-block templates
    # We will (a) compute a uniform L per block = max window length among events in the block
    #     and (b) build PCA template using 2 to end pulses
    n_total_pulses = trigger_pairs.shape[0]
    n_blocks = block_bounds.size - 1
    templates_by_block: Dict[int, np.ndarray] = {}
    block_length_each_block: Dict[int, int] = {}

    for block in range(n_blocks):
        i0, i1 = int(block_bounds[block]), int(block_bounds[block + 1])
        if i1 <= i0:
            continue
        tp_block = trigger_pairs[i0:i1, :]

        # Uniform window length per block (max of per-event [start-pre : end+post])
        lengths_block = _block_window_lengths(tp_block, pre=pca_params.pre_samples, post_pad=pca_params.post_pad_samples)
        block_length = int(np.max(lengths_block))
        block_length_each_block[block] = block_length

        # Events used for PCA template (optionally drop first N)
        if pca_params.first_pulse_special and pca_params.exclude_first_n_for_pca > 0:
            use_rows = np.arange(tp_block.shape[0])[pca_params.exclude_first_n_for_pca:]
        else:
            use_rows = np.arange(tp_block.shape[0])

        if use_rows.size == 0:
            templates_by_block[block] = np.zeros((block_length, clean.shape[1]), dtype=np.float32)
            continue

        snips_block, keep_mask, L_real = _extract_block_snippets(
            clean, tp_block[use_rows], pre=pca_params.pre_samples, post_pad=pca_params.post_pad_samples, target_len=block_length
        )
        if snips_block.shape[0] == 0:
            templates_by_block[block] = np.zeros((block_length, clean.shape[1]), dtype=np.float32)
        else:
            templates_by_block[block] = _pca_template_per_channel(snips_block, center=pca_params.center_snippets)

    # Subtract templates event-by-event (with optional per-channel amplitude scaling)
    # Also apply interp ramp from (artifact_end + tail) to next trigger start.
    for ev in range(n_total_pulses):
        # Which block is this event in?
        # Find b such that block_bounds[b] <= ev < block_bounds[b+1]
        block = int(np.searchsorted(block_bounds[1:], ev, side="right"))
        if block not in templates_by_block:
            continue
        template = templates_by_block[block]
        block_length = block_length_each_block[block]
        if template.size == 0:
            continue

        trig_start = int(trigger_pairs[ev, 0])
        trig_end   = int(trigger_pairs[ev, 1])

        # Apply subtraction on window [a:b)
        a = trig_start - pca_params.pre_samples
        bnd = a + block_length
        if a < 0 or bnd > clean.shape[0]:
            # out of bounds; skip this event
            continue

        patch = clean[a:bnd, :]           # (L_blk, n_ch)

        # TODO fix here for 3 PCs
        if pca_params.scale_amplitude:
            num = np.sum(patch * template, axis=0)                 # (n_ch,)
            den = np.sum(template * template, axis=0) + 1e-12
            amp = num / den                                   # (n_ch,)
            patch -= (template * amp[None, :]).astype(patch.dtype)
            
            # TODO projection = PCA.inverse_transform(PCA.transform(patch))
            # TODO patch -= projection
            # TODO to get the final corrected data
        else:
            patch -= template.astype(patch.dtype)

        clean[a:bnd, :] = patch

        # Interpulse ramp
        if pca_params.interp_ramp:
            ramp_start = min(trig_end + ramp_tail, clean.shape[0] - 1)
            # find next trigger start (absolute)
            if ev + 1 < n_total_pulses:
                next_start = int(trigger_pairs[ev + 1, 0])
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
