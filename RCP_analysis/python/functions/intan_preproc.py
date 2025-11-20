from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
from spikeinterface.core import ChannelSliceRecording

# Mapping
def reorder_recording_to_geometry(rec: si.BaseRecording, perm: np.ndarray | None) -> si.BaseRecording:
    """
    Reorder channels of `rec` according to a permutation `perm` which maps from device order to geometric order.

    Parameters
    ----------
    rec : BaseRecording
        Input recording.
    perm : array-like or None
        Permutation indices of length rec.get_num_channels(), or None to
        leave the recording unchanged.

    Returns
    -------
    BaseRecording
        Channel-sliced recording with reordered channels (or original if perm is None).
    """
    if perm is None:
        print("[WARN] No channel mapping provided; using device order.")
        return rec
    if perm.ndim != 1:
        raise ValueError(f"perm must be 1D, got shape {perm.shape}")
    if rec.get_num_channels() != perm.size:
        raise ValueError(f"Perm length {perm.size} != {rec.get_num_channels()} channels.")
    channel_ids = rec.get_channel_ids()
    return ChannelSliceRecording(rec, channel_ids=channel_ids[perm])

# Stim stuff
@dataclass
class StimTriggerResult:
    active_channels: np.ndarray            # (n_active_channels,)
    trigger_pairs: np.ndarray              # (n_pulses, 2) [start_sample, end_sample]
    block_bounds_samples: np.ndarray       # (n_blocks, 2) [block_start_sample, block_end_sample]
    pulse_sizes: np.ndarray                # (n_pulses,)

def extract_stim_triggers_and_blocks(
    stim_data: np.ndarray,   # (n_channels, n_samples)
) -> StimTriggerResult:
    """
    Detect stimulation pulses and group them into repeating blocks.

    Parameters
    ----------
    stim_data : array, shape (n_channels, n_samples)
        Raw stim stream (as saved from Intan/USB ADC/etc.). Zero means baseline.

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
            block_bounds_samples=np.empty((0, 2), dtype=np.int64),
            pulse_sizes=np.array([], dtype=int),
        )
    det_ch = int(active_channels[0])

    stim_signal = np.asarray(stim_data[det_ch, :], dtype=np.float32)
    if stim_signal.size < 2:
        return StimTriggerResult(
            active_channels=active_channels,
            trigger_pairs=np.empty((0, 2), dtype=np.int64),
            block_bounds_samples=np.empty((0, 2), dtype=np.int64),
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
    rz = np.asarray(rz, dtype=np.int32)

    beg = falling_edge
    if rising_edge.size > falling_edge.size:
        beg = rising_edge
    beg = beg[::2] # two falling edgers per biphasic pulse
    end_ = rz[1::2] # take every second return-to-zero, since it's biphasic

    n = int(min(beg.size, end_.size)) # number of pulses
    if n == 0:
        trigger_pairs = np.empty((0, 2), dtype=np.int32)
        pulse_sizes = np.array([], dtype=int)
    else:
        trigger_pairs = np.column_stack([beg[:n], end_[:n]]).astype(np.int32)
        pulse_sizes = trigger_pairs[:, 1] - trigger_pairs[:, 0]

    # --- 3) block (repeat) boundaries
    if trigger_pairs.shape[0] == 0:
        block_bounds_samples = np.empty((0, 2), dtype=np.int64)
    else:
        pulse_size_ref = int(np.median(pulse_sizes))
        repeat_gap_threshold = 50 * pulse_size_ref

        starts = trigger_pairs[:, 0]
        ends   = trigger_pairs[:, 1]

        gaps = np.diff(starts)
        cut_points = np.flatnonzero(gaps > repeat_gap_threshold) + 1
        block_boundaries_idx = np.concatenate([[0], cut_points, [trigger_pairs.shape[0]]]).astype(int)

        block_starts = starts[block_boundaries_idx[:-1]]
        block_ends   = ends[block_boundaries_idx[1:] - 1]
        block_bounds_samples = np.column_stack([block_starts, block_ends]).astype(np.int64)

    return StimTriggerResult(
        active_channels=active_channels,
        trigger_pairs=trigger_pairs,
        block_bounds_samples=block_bounds_samples,
        pulse_sizes=pulse_sizes,
)

def extract_stim_npz(
    sess_folder: Path,
    out_root: Path,
    stim_stream_name: str = "Stim channel",
    chanmap_perm: np.ndarray | None = None,
):
    bundle_dir = out_root / f"{sess_folder.name}_Intan_bundle"; bundle_dir.mkdir(parents=True, exist_ok=True)

    try:
        rec = se.read_split_intan_files(sess_folder, mode="concatenate", stream_name=stim_stream_name, use_names_as_ids=True)
    except Exception as e:
        print(f"[{sess_folder.name}] skip stream '{stim_stream_name}': {e}")
        return None
    rec_reordered = reorder_recording_to_geometry(rec, chanmap_perm)
    order = "geometry" if chanmap_perm is not None else "device"

    # load stim traces into memory
    stim_traces = rec_reordered.get_traces(return_scaled=True).T  # (n_channels, n_samples)
    stim_ext = extract_stim_triggers_and_blocks(stim_data=stim_traces)

    # collect everything you want to save
    arrays = {
        "stim_traces": stim_traces,
        "active_channels": stim_ext.active_channels.astype(np.int32),
        "trigger_pairs": stim_ext.trigger_pairs, # int64 (trigs, 2)
        "block_bounds_samples": stim_ext.block_bounds_samples, # int64 (blocks, 2)
        "pulse_sizes": stim_ext.pulse_sizes.astype(np.int32),
    }
    meta = dict(
        session=sess_folder.name,
        stream_name=stim_stream_name,
        fs_hz=rec_reordered.get_sampling_frequency(),
        n_channels=int(rec_reordered.get_num_channels()),
        order=order,
        note="Raw stim stream and derived trigger/block outputs."
    )

    out_npz = bundle_dir / "stim_stream.npz"
    np.savez_compressed(out_npz, **arrays, meta=json.dumps(meta))
    print(f"[STIM] saved stim stream + triggers -> {out_npz}")
    return out_npz, arrays

# AUX streams
def extract_aux_streams_npz(
    sess_folder: Path,
    out_root: Path,
    aux_streams: tuple[str, ...] = ("USB board ADC input channel",),
):
    bundle_dir = out_root / f"{sess_folder.name}_Intan_bundle"; bundle_dir.mkdir(parents=True, exist_ok=True)

    try:
        rec = se.read_split_intan_files(sess_folder, mode="concatenate", stream_name=aux_streams, use_names_as_ids=True)
        rec = spre.unsigned_to_signed(rec) # Convert UInt16 to int16
    except Exception as e:
        print(f"[{sess_folder.name}] skip stream '{aux_streams}': {e}")
        return None

    aux_traces = rec.get_traces(return_scaled=True).T  # (n_channels, n_samples)
    
    meta = dict(
        session=sess_folder.name,
        stream_name=aux_streams,
        fs_hz=rec.get_sampling_frequency(),
        n_channels=rec.get_num_channels(),
        channel_ids=list(rec.get_channel_ids()),
        dtype=str(rec.get_dtype()),
        shape=aux_traces.shape,
        units="uV",
        note="Aux stream stored as a single array aux_traces.",
    )
    
    out_npz = bundle_dir / f"aux_streams.npz"
    np.savez_compressed(out_npz, aux_traces=aux_traces, meta=json.dumps(meta))
    print(f"[AUX] saved stream '{aux_streams}' -> {out_npz}")
    return out_npz