from tqdm import tqdm
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

def _extract_stim_triggers_and_blocks(
    stim_data: np.ndarray,   # (n_channels, n_samples)
) -> StimTriggerResult:
    """
    Detect stimulation pulses and group them into blocks [beg end]

    Parameters:
    stim_data : array, shape (n_channels, n_samples)
        Raw stim stream, zero means baseline

    Returns: StimTriggerResult object
    """
    if stim_data.ndim != 2:
        raise ValueError("stim_data must be (n_channels, n_samples)")

    # 1) return active channels
    active_channels = np.flatnonzero((stim_data != 0).any(axis=1)) + 1  # 1-based channel IDs
    if active_channels.size == 0:
        # nothing to do
        return StimTriggerResult(
            active_channels=np.array([], dtype=int),
            trigger_pairs=np.empty((0, 2), dtype=np.int32),
            block_bounds_samples=np.empty((0, 2), dtype=np.int32),
            pulse_sizes=np.array([], dtype=int),
        )
    det_ch = int(active_channels[0])

    stim_signal = np.asarray(stim_data[det_ch, :], dtype=np.float32)
    if stim_signal.size < 2:
        return StimTriggerResult(
            active_channels=active_channels,
            trigger_pairs=np.empty((0, 2), dtype=np.int32),
            block_bounds_samples=np.empty((0, 2), dtype=np.int32),
            pulse_sizes=np.array([], dtype=int),
        )

    # 2) edge detection
    diff = np.diff(stim_signal)
    falling_edge = np.flatnonzero(diff < 0) + 1
    rising_edge   = np.flatnonzero(diff > 0) + 1

    # For each falling edge, find the first subsequent return-to-zero
    rz = []
    for idx in tqdm(falling_edge, desc="detecting pulse ends", leave=False):
        end_of_pulse = np.flatnonzero(stim_signal[idx:] == 0)
        if end_of_pulse.size:
            rz.append(idx + end_of_pulse[0])
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
        block_bounds_samples = np.empty((0, 2), dtype=np.int32)
    else:
        pulse_size_ref = int(np.median(pulse_sizes))
        repeat_gap_threshold = 50 * pulse_size_ref #TODO This right now categorizes everything higher than 10Hz stim as a "block"

        starts = trigger_pairs[:, 0]
        ends   = trigger_pairs[:, 1]

        gaps = np.diff(starts)
        cut_points = np.flatnonzero(gaps > repeat_gap_threshold) + 1
        block_boundaries_idx = np.concatenate([[0], cut_points, [trigger_pairs.shape[0]]]).astype(int)

        block_starts = starts[block_boundaries_idx[:-1]]
        block_ends   = ends[block_boundaries_idx[1:] - 1]
        block_bounds_samples = np.column_stack([block_starts, block_ends]).astype(np.int32)

    return StimTriggerResult(
        active_channels=active_channels,
        trigger_pairs=trigger_pairs,
        block_bounds_samples=block_bounds_samples,
        pulse_sizes=pulse_sizes,
)

STIM_CHUNK_S = 30.0


def scan_active_channels(rec, chunk_s=STIM_CHUNK_S):
    """Channels with any nonzero stim sample, 0-based. Memory bounded by chunk_s."""
    n_total = rec.get_num_samples()
    chunk = int(chunk_s * rec.get_sampling_frequency())
    active = np.zeros(rec.get_num_channels(), dtype=bool)
    for s0 in tqdm(range(0, n_total, chunk), desc="scanning stim channels", leave=False):
        tr = rec.get_traces(start_frame=s0, end_frame=min(s0 + chunk, n_total))
        active |= (tr != 0).any(axis=0)
        del tr
        if active.all():
            break
    return np.flatnonzero(active)


def load_channel_signal(rec, ch_index, chunk_s=STIM_CHUNK_S):
    """One channel as scaled float32, filled in chunks. ~86 MB for a 12 min session."""
    n_total = rec.get_num_samples()
    chunk = int(chunk_s * rec.get_sampling_frequency())
    ch_id = rec.get_channel_ids()[ch_index]
    out = np.empty(n_total, dtype=np.float32)
    for s0 in range(0, n_total, chunk):
        s1 = min(s0 + chunk, n_total)
        out[s0:s1] = rec.get_traces(start_frame=s0, end_frame=s1,
                                    channel_ids=[ch_id], return_scaled=True)[:, 0]
    return out


def load_traces_chunked(rec, ch_indices=None, chunk_s=STIM_CHUNK_S):
    """Build (n_channels, n_samples) scaled float32 traces without a full-size temp.

    ch_indices selects a 0-based channel subset; None loads all channels.
    """
    n_total = rec.get_num_samples()
    chunk = int(chunk_s * rec.get_sampling_frequency())
    all_ids = rec.get_channel_ids()
    ch_ids = all_ids if ch_indices is None else all_ids[np.asarray(ch_indices, int)]

    out = np.empty((len(ch_ids), n_total), dtype=np.float32)
    for s0 in tqdm(range(0, n_total, chunk), desc="loading stim traces", leave=False):
        s1 = min(s0 + chunk, n_total)
        tr = rec.get_traces(start_frame=s0, end_frame=s1,
                            channel_ids=list(ch_ids), return_scaled=True)
        out[:, s0:s1] = tr.T
        del tr
    return out
def _extract_stim_triggers_and_blocks_1d(stim_signal, active_channels_0based):
    """Pulse and block detection on a single detection channel.

    active_channels on the result stays 1-based, matching the original function.
    """
    active_0 = np.asarray(active_channels_0based, dtype=int)
    empty = StimTriggerResult(
        active_channels=active_0 + 1,
        trigger_pairs=np.empty((0, 2), dtype=np.int32),
        block_bounds_samples=np.empty((0, 2), dtype=np.int32),
        pulse_sizes=np.array([], dtype=int),
    )
    if stim_signal.size < 2:
        return empty

    diff = np.diff(stim_signal)
    falling_edge = np.flatnonzero(diff < 0) + 1
    rising_edge = np.flatnonzero(diff > 0) + 1
    del diff

    # first return-to-zero at or after each falling edge, as one sorted lookup
    zero_idx = np.flatnonzero(stim_signal == 0)
    if falling_edge.size and zero_idx.size:
        pos = np.searchsorted(zero_idx, falling_edge, side="left")
        rz = zero_idx[pos[pos < zero_idx.size]].astype(np.int32)
    else:
        rz = np.zeros(0, dtype=np.int32)
    del zero_idx

    beg = rising_edge if rising_edge.size > falling_edge.size else falling_edge
    beg = beg[::2]   # two falling edges per biphasic pulse
    end_ = rz[1::2]  # every second return-to-zero, since biphasic

    n = int(min(beg.size, end_.size))
    if n == 0:
        return empty

    trigger_pairs = np.column_stack([beg[:n], end_[:n]]).astype(np.int32)
    pulse_sizes = trigger_pairs[:, 1] - trigger_pairs[:, 0]

    pulse_size_ref = int(np.median(pulse_sizes))
    repeat_gap_threshold = 50 * pulse_size_ref
    starts, ends = trigger_pairs[:, 0], trigger_pairs[:, 1]
    cut_points = np.flatnonzero(np.diff(starts) > repeat_gap_threshold) + 1
    idx = np.concatenate([[0], cut_points, [trigger_pairs.shape[0]]]).astype(int)
    block_bounds_samples = np.column_stack([starts[idx[:-1]], ends[idx[1:] - 1]]).astype(np.int32)

    return StimTriggerResult(
        active_channels=active_0 + 1,
        trigger_pairs=trigger_pairs,
        block_bounds_samples=block_bounds_samples,
        pulse_sizes=pulse_sizes,
    )
    
def extract_stim_npz(
    sess: Path,
    out_dir: Path,
    stim_stream_name: str = "Stim channel",
    chanmap_perm: np.ndarray | None = None,
    save_traces: bool | str = True,
    det_channel: int | None = None,
):
    """Detect stim pulses/blocks and cache them, optionally with the raw traces.

    save_traces:
      True     - all channels, as before (~0.46 GB per recorded minute)
      "active" - only channels carrying stim; rows map via stim_traces_channels_0based
      False    - traces omitted, detection outputs only
    det_channel: 0-based override for the detection channel, default first active.
    """
    stim_npz_dir = out_dir / f"{sess.name}_Intan_streams"
    stim_npz_dir.mkdir(parents=True, exist_ok=True)

    try:
        rec = se.read_split_intan_files(sess, mode="concatenate",
                                        stream_name=stim_stream_name,
                                        use_names_as_ids=True)
    except Exception as e:
        print(f"[{sess.name}] skip stream '{stim_stream_name}': {e}")
        return None
    rec_reordered = reorder_recording_to_geometry(rec, chanmap_perm)
    order = "geometry" if chanmap_perm is not None else "device"

    n_total = rec_reordered.get_num_samples()
    n_ch = int(rec_reordered.get_num_channels())
    fs_hz = rec_reordered.get_sampling_frequency()

    active_0 = scan_active_channels(rec_reordered)
    print(f"[{sess.name}] {active_0.size} active stim channels (0-based): {active_0.tolist()}")

    if active_0.size == 0:
        stim_ext = _extract_stim_triggers_and_blocks_1d(np.zeros(0, np.float32), active_0)
        det_ch = None
    else:
        det_ch = int(active_0[0]) if det_channel is None else int(det_channel)
        stim_signal = load_channel_signal(rec_reordered, det_ch)
        stim_ext = _extract_stim_triggers_and_blocks_1d(stim_signal, active_0)
        del stim_signal

    stim_arrays = {
        "active_channels": stim_ext.active_channels.astype(np.int32),   # 1-based, as before
        "active_channels_0based": active_0.astype(np.int32),            # 0-based, for indexing
        "trigger_pairs": stim_ext.trigger_pairs,
        "block_bounds_samples": stim_ext.block_bounds_samples,
        "pulse_sizes": stim_ext.pulse_sizes.astype(np.int32),
    }

    if save_traces:
        traces_channels = None if save_traces is True else active_0
        n_rows = n_ch if traces_channels is None else traces_channels.size
        print(f"[{sess.name}] loading stim traces (~{n_rows * n_total * 4 / 1e9:.1f} GB in RAM)")
        stim_arrays["stim_traces"] = load_traces_chunked(rec_reordered, traces_channels)
        stim_arrays["stim_traces_channels_0based"] = (
            np.arange(n_ch, dtype=np.int32) if traces_channels is None
            else traces_channels.astype(np.int32)
        )

    meta = dict(
        session=sess.name,
        stream_name=stim_stream_name,
        fs_hz=fs_hz,
        n_channels=n_ch,
        n_samples=int(n_total),
        order=order,
        det_channel_0based=det_ch,
        det_channel_1based=None if det_ch is None else det_ch + 1,
        traces_saved=("all" if save_traces is True else ("active" if save_traces else "none")),
        note="Derived trigger/block outputs; stim_traces present per traces_saved.",
    )

    out_npz = stim_npz_dir / "stim_stream.npz"
    np.savez_compressed(out_npz, **stim_arrays, meta=json.dumps(meta))
    print(f"[STIM] saved -> {out_npz} "
          f"({stim_arrays['block_bounds_samples'].shape[0]} blocks, "
          f"{stim_arrays['trigger_pairs'].shape[0]} pulses)")
    return stim_arrays
    
# def extract_stim_npz(
#     sess: Path,
#     out_dir: Path,
#     stim_stream_name: str = "Stim channel",
#     chanmap_perm: np.ndarray | None = None,
# ):
#     stim_npz_dir = out_dir / f"{sess.name}_Intan_streams"; stim_npz_dir.mkdir(parents=True, exist_ok=True)

#     try:
#         rec = se.read_split_intan_files(sess, mode="concatenate", stream_name=stim_stream_name, use_names_as_ids=True)
#     except Exception as e:
#         print(f"[{sess.name}] skip stream '{stim_stream_name}': {e}")
#         return None
#     rec_reordered = reorder_recording_to_geometry(rec, chanmap_perm)
#     order = "geometry" if chanmap_perm is not None else "device"

#     # load stim traces into memory
#     stim_traces = rec_reordered.get_traces(return_scaled=True).T  # (n_channels, n_samples)
#     stim_ext = _extract_stim_triggers_and_blocks(stim_data=stim_traces)

#     # collect everything you want to save
#     stim_arrays = {
#         "stim_traces": stim_traces,
#         "active_channels": stim_ext.active_channels.astype(np.int32),
#         "trigger_pairs": stim_ext.trigger_pairs, # int32 (trigs, 2)
#         "block_bounds_samples": stim_ext.block_bounds_samples, # int32 (blocks, 2)
#         "pulse_sizes": stim_ext.pulse_sizes.astype(np.int32),
#     }
#     meta = dict(
#         session=sess.name,
#         stream_name=stim_stream_name,
#         fs_hz=rec_reordered.get_sampling_frequency(),
#         n_channels=int(rec_reordered.get_num_channels()),
#         order=order,
#         note="Raw stim stream and derived trigger/block outputs."
#     )

#     out_npz = stim_npz_dir / "stim_stream.npz"
#     np.savez_compressed(out_npz, **stim_arrays, meta=json.dumps(meta))
#     print(f"[STIM] saved stim stream + triggers -> {out_npz}")
#     return stim_arrays

# AUX streams
def extract_intan_aux_streams_npz(
    sess: Path,
    out_dir: Path,
    aux_streams: tuple[str, ...] = ("USB board ADC input channel",),
):
    aux_dir = out_dir / f"{sess.name}_Intan_streams"; aux_dir.mkdir(parents=True, exist_ok=True)

    try:
        rec = se.read_split_intan_files(sess, mode="concatenate", stream_name=aux_streams, use_names_as_ids=True)
        rec = spre.unsigned_to_signed(rec) # Convert UInt16 to int16
    except Exception as e:
        print(f"[{sess.name}] skip stream '{aux_streams}': {e}")
        return None

    aux_traces = rec.get_traces(return_scaled=True).T  # (n_channels, n_samples)
    
    meta = dict(
        session=sess.name,
        stream_name=aux_streams,
        fs_hz=rec.get_sampling_frequency(),
        n_channels=rec.get_num_channels(),
        channel_ids=list(rec.get_channel_ids()),
        dtype=str(rec.get_dtype()),
        shape=aux_traces.shape,
        units="uV",
        note="Aux stream stored as a single array aux_traces.",
    )
    
    out_npz = aux_dir / f"aux_streams.npz"
    np.savez_compressed(out_npz, aux_traces=aux_traces, meta=json.dumps(meta))
    print(f"[AUX] saved stream '{aux_streams}' -> {out_npz}")
    return out_npz