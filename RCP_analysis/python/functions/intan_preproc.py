from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import io, json, re, zipfile
import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from probeinterface import Probe
from dataclasses import dataclass

# ----------------------
# Geometry / mapping
# ----------------------
def load_stim_geometry(mat_or_csv: Path) -> Dict[str, np.ndarray]:
    mat_or_csv = Path(mat_or_csv)
    out: Dict[str, np.ndarray] = {}

    if mat_or_csv.suffix.lower() == ".mat":
        from scipy.io import loadmat
        m = loadmat(mat_or_csv)
        out["x"] = np.asarray(m["xcoords"]).ravel().astype(float)
        out["y"] = np.asarray(m["ycoords"]).ravel().astype(float)

        # Prefer explicit 0-based device mapping if present
        if "chanMap0ind" in m:
            out["device_index_0based"] = np.asarray(m["chanMap0ind"]).ravel().astype(int)
        # Fallback to 1-based if a file ever has it
        elif "device_index_1based" in m:
            out["device_index_1based"] = np.asarray(m["device_index_1based"]).ravel().astype(int)

        # Optional: carry a 'connected' mask if present
        if "connected" in m:
            out["connected_mask"] = (np.asarray(m["connected"]).ravel().astype(int) != 0)

    else:
        import pandas as pd
        df = pd.read_csv(mat_or_csv)
        out["x"] = df["x"].to_numpy(float)
        out["y"] = df["y"].to_numpy(float)
        if "device_index_1based" in df.columns:
            out["device_index_1based"] = df["device_index_1based"].to_numpy(int)

    return out

def make_probe_from_geom(geom: Dict[str, np.ndarray], radius_um: float = 5.0) -> Probe:
    x = np.asarray(geom["x"], float).ravel()
    y = np.asarray(geom["y"], float).ravel()
    assert x.size == y.size, "x/y must have same length"

    pr = Probe(ndim=2)
    pr.set_contacts(positions=np.c_[x, y], shapes="circle", shape_params={"radius": float(radius_um)})

    # Apply device-channel mapping (alignment of geometry vector to recording's channel order)
    if "device_index_0based" in geom:
        dev = np.asarray(geom["device_index_0based"], int).ravel()
        if dev.size != x.size:
            raise ValueError("device_index_0based length != #contacts")
        pr.set_device_channel_indices(dev)
    elif "device_index_1based" in geom:
        dev = np.asarray(geom["device_index_1based"], int).ravel() - 1
        if dev.size != x.size:
            raise ValueError("device_index_1based length != #contacts")
        pr.set_device_channel_indices(dev)

    return pr

def get_chanmap_perm_from_geom(geom: dict) -> np.ndarray:
    # geometry→device 0-based indices (normalized in loader)
    if "device_index_0based" in geom:
        return np.asarray(geom["device_index_0based"], int).ravel()
    if "device_index_1based" in geom:
        return np.asarray(geom["device_index_1based"], int).ravel() - 1
    raise ValueError("No chanmap in geometry.")

def make_identity_probe_from_geom(geom: dict, radius_um: float = 5.0):
    pr = make_probe_from_geom(geom, radius_um=radius_um)
    # Identity: channel i ↔ contact i (after you reorder the recording)
    pr.set_device_channel_indices(np.arange(pr.get_contact_count(), dtype=int))
    return pr

def reorder_recording_to_geometry(rec: si.BaseRecording, perm: np.ndarray) -> si.BaseRecording:
    ids = list(rec.get_channel_ids())
    if len(ids) != perm.size:
        raise ValueError(f"Perm length {perm.size} != {len(ids)} channels.")
    ordered_ids = [ids[i] for i in perm]        # put data into geometry order
    try:
        return rec.channel_slice(channel_ids=ordered_ids)
    except Exception:
        from spikeinterface.core import ChannelSliceRecording
        return ChannelSliceRecording(rec, channel_ids=ordered_ids)

# ----------------------
# Intan loading
# ----------------------
def read_intan_recording(
    folder: Path,
    stream_name: str = "RHS2000 amplifier channel",
) -> si.BaseRecording:
    """
    Read a single Intan session folder (RHS split files).
    """
    folder = Path(folder)
    rec = se.read_split_intan_files(folder, mode="concatenate", stream_name=stream_name, use_names_as_ids=True)
    # If UInt16, convert to signed int16 (common for Intan)
    if rec.get_dtype().kind == "u":
        rec = spre.unsigned_to_signed(rec)
        rec = spre.astype(rec, dtype="int16")
    return rec

def load_stim_triggers_from_npz(stim_npz_path: Path):
    stim_npz_path = Path(stim_npz_path)
    trigs = None
    blocks = None
    pulse_sizes = None
    meta = None

    with np.load(stim_npz_path, allow_pickle=False) as z:
        if "trigger_pairs" in z:
            trigs = z["trigger_pairs"][:, 0].astype(np.int64)
        else:
            trigs = np.zeros((0,), dtype=np.int64)

        if "block_boundaries" in z:
            blocks = z["block_boundaries"].astype(np.int64)

        if "pulse_sizes" in z:
            pulse_sizes = z["pulse_sizes"].astype(np.int64)

    # try meta.json
    try:
        with zipfile.ZipFile(stim_npz_path, "r") as zf:
            with zf.open("meta.json") as f:
                meta = json.load(f)
    except Exception:
        meta = None

    return trigs, blocks, pulse_sizes, meta

# ----------------------
# Preprocessing
# ----------------------
def local_cm_reference(
    rec: si.BaseRecording,
    freq_min: float = 300.0,
    inner_outer_radius_um: Tuple[float, float] = (30.0, 150.0),
) -> si.BaseRecording:
    """
    High-pass + local common-median reference using an annulus [inner, outer] in µm.
    """
    rec_hp = spre.highpass_filter(rec, freq_min=freq_min)
    rmin, rmax = inner_outer_radius_um
    rec_lref = spre.common_reference(rec_hp, reference="local", operator="median", local_radius=(float(rmin), float(rmax)))
    return rec_lref

def save_recording(rec: si.BaseRecording, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec.save(folder=out_dir, overwrite=True)

_SAN = re.compile(r'[^0-9A-Za-z_]+')
def _safe(s: str) -> str:
    return _SAN.sub('_', s).strip('_')

def _save_npz_streaming(npz_path: Path, items, meta: dict):
    """Write a streaming NPZ: items yields (name, np.ndarray). Meta is stored as JSON inside the zip."""
    with zipfile.ZipFile(str(npz_path), mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        # meta.json
        zf.writestr('meta.json', json.dumps(meta, indent=2))
        # each chunk becomes its own .npy inside the NPZ
        for key, arr in items:
            bio = io.BytesIO()
            np.lib.format.write_array(bio, np.asanyarray(arr), allow_pickle=False)
            zf.writestr(f'{_safe(key)}.npy', bio.getvalue())

def _iter_recording_chunks(rec: si.BaseRecording, chunk_s: float):
    fs = float(rec.get_sampling_frequency())
    step = int(max(1, round(chunk_s * fs)))
    n = int(rec.get_num_samples())
    for k, start in enumerate(range(0, n, step)):
        end = min(n, start + step)
        X = rec.get_traces(start_frame=start, end_frame=end, return_in_uV=True)
        yield (f'chunk_{k:04d}', X)

# ----------------------
# Stim stuff
# ----------------------

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
    stim_data: np.ndarray,   # (n_channels, n_samples)
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

def extract_and_save_stim_npz(
    sess_folder: Path,
    out_root: Path,
    stim_stream_name: str = "Stim channel",
    chanmap_perm: np.ndarray | None = None,
):
    bundle_dir = out_root / f"{sess_folder.name}_Intan_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    rec = read_intan_recording(sess_folder, stream_name=stim_stream_name)
    fs = float(rec.get_sampling_frequency())

    # reorder if needed
    if chanmap_perm is not None and rec.get_num_channels() == chanmap_perm.size:
        rec = reorder_recording_to_geometry(rec, chanmap_perm)
        order = "geometry"
    else:
        order = "device"

    # load stim traces into memory
    stim_traces = rec.get_traces(return_scaled=True).T  # (n_channels, n_samples)
    stim_ext = extract_stim_triggers_and_blocks(stim_data=stim_traces)

    # collect everything you want to save
    arrays = {
        "stim_traces": stim_traces.astype(np.float32),
        "active_channels": stim_ext.active_channels.astype(np.int64),
        "trigger_pairs": stim_ext.trigger_pairs.astype(np.int64),
        "block_boundaries": stim_ext.block_boundaries.astype(np.int64),
        "pulse_sizes": stim_ext.pulse_sizes.astype(np.int64),
    }
    meta = dict(
        session=sess_folder.name,
        stream_name=stim_stream_name,
        fs_hz=fs,
        n_channels=int(rec.get_num_channels()),
        order=order,
        note="Raw stim stream and derived trigger/block outputs."
    )

    out_npz = bundle_dir / "stim_stream.npz"
    np.savez_compressed(out_npz, **arrays, meta=json.dumps(meta))
    print(f"[STIM] saved stim stream + triggers -> {out_npz}")
    return out_npz

# ----------------------
# Other streams
# ----------------------

def extract_and_save_other_streams_npz(
    sess_folder: Path,
    out_root: Path,
    include_streams: tuple[str, ...] = ("USB board ADC input channel",),
    exclude_streams: tuple[str, ...] = ("Stim channel",),
    chunk_s: float = 60.0,
    chanmap_perm: np.ndarray | None = None,
):
    bundle_dir = out_root / f"{sess_folder.name}_Intan_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for stream_name in include_streams:
        if stream_name in exclude_streams:
            continue
        try:
            rec = read_intan_recording(sess_folder, stream_name=stream_name)

            if chanmap_perm is not None and rec.get_num_channels() == chanmap_perm.size:
                rec = reorder_recording_to_geometry(rec, chanmap_perm)
                order = "geometry"
                perm_geom_to_device = chanmap_perm.tolist()
                perm_device_to_geom = np.argsort(chanmap_perm).tolist()
            else:
                if chanmap_perm is not None:
                    print(f"[{sess_folder.name}] '{stream_name}': skip reordering "
                          f"(n_ch={rec.get_num_channels()} vs perm={chanmap_perm.size})")
                order = "device"
                perm_geom_to_device = None
                perm_device_to_geom = None

        except Exception as e:
            print(f"[{sess_folder.name}] skip stream '{stream_name}': {e}")
            continue

        fs = float(rec.get_sampling_frequency())
        meta = dict(
            session=sess_folder.name,
            stream_name=stream_name,
            fs_hz=fs,
            n_channels=int(rec.get_num_channels()),
            channel_ids=[str(x) for x in rec.get_channel_ids()],
            dtype=str(rec.get_dtype()),
            chunk_seconds=chunk_s,
            shape=[int(rec.get_num_samples()), int(rec.get_num_channels())],
            order=order,
            perm_geom_to_device=perm_geom_to_device,
            perm_device_to_geom=perm_device_to_geom,
            units="uV",
            note="Data stored as multiple .npy chunks inside this .npz (chunk_0000.npy, ...).",
        )
        out_npz = bundle_dir / f"{_safe(stream_name)}.npz"
        _save_npz_streaming(out_npz, _iter_recording_chunks(rec, chunk_s), meta)
        print(f"[AUX] saved stream '{stream_name}' -> {out_npz}")
        saved.append(out_npz)
    return saved

# ----------------------
# Convenience
# ----------------------
def list_intan_sessions(root: Path) -> list[Path]:
    root = Path(root)
    return sorted([p for p in root.iterdir() if p.is_dir()])

