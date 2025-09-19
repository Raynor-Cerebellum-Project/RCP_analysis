import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class StimTriggerConfig:
    """
    Parameters that affect repeat (block) grouping.
    buffer_samples: The buffer used on each side of a pulse when building snippets (in SAMPLES)
                    Same as template_params.buffer in MATLAB
    """
    buffer_samples: int

@dataclass
class StimTriggerResult:
    active_channels: np.ndarray            # (n_active_channels,)
    trigger_pairs: np.ndarray              # (n_events, 2) [start_sample, end_sample]
    block_boundaries: np.ndarray           # (n_blocks+1,) event indices; blocks are [i:k) in event index space
    pulse_sizes: np.ndarray                # (n_events,) length of each pulse in samples

def extract_stim_triggers_and_blocks(
    stim_data: np.ndarray,
    fs: float,
    cfg: StimTriggerConfig,
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

    Returns
    -------
    StimTriggerResult
    """
    if stim_data.ndim != 2:
        raise ValueError("stim_data must be (n_channels, n_samples)")

    # 1) return active channels
    active_channels = np.flatnonzero((stim_data != 0).any(axis=1))
    if active_channels.size == 0:
        # nothing to do
        return StimTriggerResult(
            active_channels=np.array([], dtype=int),
            trigger_pairs=np.empty((0, 2), dtype=np.int64),
            block_boundaries=np.array([0], dtype=int),
            pulse_sizes=np.array([], dtype=int),
        )
    det_ch = int(active_channels[0])

    stim_signal = np.asarray(stim_data[det_ch, :], dtype=np.float64)
    if stim_signal.size < 2:
        return StimTriggerResult(
            active_channels=active_channels,
            trigger_pairs=np.empty((0, 2), dtype=np.int64),
            block_boundaries=np.array([0], dtype=int),
            pulse_sizes=np.array([], dtype=int),
        )

    # 2) edge detection
    diff = np.diff(stim_signal)
    falling_edge = np.flatnonzero(diff < 0) + 1
    rising_edge   = np.flatnonzero(diff > 0) + 1

    # For each falling edge, find the first subsequent return-to-zero
    rz = []
    for idx in falling_edge:
        end_of_pulse = np.flatnonzero(stim_signal[idx:] == 0)
        if end_of_pulse.size:
            rz.append(idx + end_of_pulse[0])  # absolute index where it returns to zero
    rz = np.asarray(rz, dtype=np.int64)

    beg = falling_edge
    if rising_edge.size > falling_edge.size:
        beg = rising_edge
    beg = beg[::2] # two falling edgers per biphasic pulse
    end_ = rz[1::2] # take every second return-to-zero, since it's biphasic

    n = int(min(beg.size, end_.size)) # number of pulses
    if n == 0:
        trigger_pairs = np.empty((0, 2), dtype=np.int64)
        pulse_sizes = np.array([], dtype=int)
    else:
        trigger_pairs = np.column_stack([beg[:n], end_[:n]]).astype(np.int64)
        pulse_sizes = trigger_pairs[:, 1] - trigger_pairs[:, 0]

    # --- 3) block (repeat) boundaries
    if trigger_pairs.shape[0] == 0:
        block_boundaries = np.array([0], dtype=int)
    else:
        # use median pulse size as representative
        pulse_size_ref = int(np.median(pulse_sizes))
        repeat_gap_threshold = 3 * pulse_size_ref
        starts = trigger_pairs[:, 0]
        gaps = np.diff(starts)
        cut_points = np.flatnonzero(gaps > repeat_gap_threshold) + 1
        block_boundaries = np.concatenate([[0], cut_points, [trigger_pairs.shape[0]]]).astype(int)

    return StimTriggerResult(
        active_channels=active_channels,
        trigger_pairs=trigger_pairs,
        block_boundaries=block_boundaries,
        pulse_sizes=pulse_sizes,
    )
