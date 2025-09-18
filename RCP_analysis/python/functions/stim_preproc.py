import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class StimTriggerConfig:
    """
    Parameters that affect repeat (block) grouping.
    buffer_samples: The 'buffer' used on each side of a pulse when building snippets (in SAMPLES).
                    This is the same quantity your MATLAB code used as template_params.buffer.
    """
    buffer_samples: int

@dataclass
class StimTriggerResult:
    active_channels: np.ndarray            # (n_active_channels,)
    trigger_pairs: np.ndarray              # (n_events, 2) [start_sample, end_sample]
    trigger_starts: np.ndarray             # (n_events,) first column of trigger_pairs
    block_boundaries: np.ndarray           # (n_blocks+1,) event indices; blocks are [i:k) in event index space

def extract_stim_triggers_and_blocks(
    stim_data: np.ndarray,
    fs: float,
    cfg: StimTriggerConfig,
    use_first_active_channel: bool = True,
    channel_index: Optional[int] = None,
) -> StimTriggerResult:
    """
    Detect stimulation pulses and group them into repeating blocks.

    Parameters
    ----------
    stim_data : array, shape (n_channels, n_samples)
        Raw stim stream (as saved from Intan/USB ADC/etc.). Zero means baseline.
    fs : float
        Sampling rate (Hz).
    cfg : StimTriggerConfig
        Configuration including buffer size in SAMPLES (same semantics as MATLAB template_params.buffer).
    use_first_active_channel : bool
        If True, detect on the first channel that contains any non-zero sample.
    channel_index : Optional[int]
        If provided, force detection on this channel (overrides use_first_active_channel).

    Returns
    -------
    StimTriggerResult
    """
    if stim_data.ndim != 2:
        raise ValueError("stim_data must be (n_channels, n_samples)")

    # --- 1) active channels: any non-zero samples along time
    active_channels = np.flatnonzero((stim_data != 0).any(axis=1))
    if active_channels.size == 0:
        # nothing to do
        return StimTriggerResult(
            active_channels=np.array([], dtype=int),
            trigger_pairs=np.empty((0, 2), dtype=np.int64),
            trigger_starts=np.empty((0,), dtype=np.int64),
            block_boundaries=np.array([0], dtype=int),  # single empty block
        )

    # choose detection channel
    if channel_index is not None:
        det_ch = int(channel_index)
    else:
        det_ch = int(active_channels[0]) if use_first_active_channel else int(active_channels[0])

    sig = np.asarray(stim_data[det_ch, :], dtype=np.float64)
    if sig.size < 2:
        return StimTriggerResult(
            active_channels=active_channels,
            trigger_pairs=np.empty((0, 2), dtype=np.int64),
            trigger_starts=np.empty((0,), dtype=np.int64),
            block_boundaries=np.array([0], dtype=int),
        )

    # --- 2) edge detection
    d = np.diff(sig)
    edges_down = np.flatnonzero(d < 0) + 1  # MATLAB find(diff<0) -> index into next sample
    edges_up   = np.flatnonzero(d > 0) + 1

    # For each falling edge, find the first subsequent return-to-zero
    rz = []
    for idx in edges_down:
        nxt_zero = np.flatnonzero(sig[idx:] == 0)
        if nxt_zero.size:
            rz.append(idx + nxt_zero[0])  # absolute index where it returns to zero
    rz = np.asarray(rz, dtype=np.int64)

    # MATLAB logic:
    # trigs_beg = trigs1; if length(trigs2) > length(trigs1) then trigs_beg = trigs2; trigs_beg = trigs_beg(1:2:end);
    # trigs_end = trigs_rz(2:2:end);
    beg = edges_down
    if edges_up.size > edges_down.size:
        beg = edges_up
    beg = beg[::2]  # every other
    end_ = rz[1::2]  # every other, starting at the second

    n = int(min(beg.size, end_.size))
    if n == 0:
        trigger_pairs = np.empty((0, 2), dtype=np.int64)
    else:
        trigger_pairs = np.column_stack([beg[:n], end_[:n]]).astype(np.int64)

    # --- 3) block (repeat) boundaries
    # MATLAB: repeat_gap_threshold = 2 * (2 * buffer + 1);
    # This is in samples; events belong to new block when the gap between consecutive starts exceeds this threshold.
    repeat_gap_threshold = 2 * (2 * int(cfg.buffer_samples) + 1)
    if trigger_pairs.shape[0] == 0:
        block_boundaries = np.array([0], dtype=int)
    else:
        starts = trigger_pairs[:, 0]
        gaps = np.diff(starts)
        # indices *between* events where a new block should start (event index AFTER the gap)
        cut_points = np.flatnonzero(gaps > repeat_gap_threshold) + 1
        # boundaries as [0, *cuts..., n_events]; interpret as blocks: [b[i]:b[i+1]) over event indices
        block_boundaries = np.concatenate([[0], cut_points, [trigger_pairs.shape[0]]]).astype(int)

    return StimTriggerResult(
        active_channels=active_channels,
        trigger_pairs=trigger_pairs,
        trigger_starts=trigger_pairs[:, 0] if trigger_pairs.size else np.empty((0,), dtype=np.int64),
        block_boundaries=block_boundaries,
    )
