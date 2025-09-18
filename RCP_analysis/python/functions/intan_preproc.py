from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import io, json, re, zipfile
import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from probeinterface import Probe
from .stim_preproc import StimTriggerConfig, extract_stim_triggers_and_blocks

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

def _append_npz_arrays(npz_path, arrays_dict):
    """
    Append arrays into an existing .npz (a zip). Keys become <key>.npy entries.
    Non-destructive for existing entries.
    """
    # write .npy bytes into the zip without touching existing members
    with zipfile.ZipFile(npz_path, mode="a", compression=zipfile.ZIP_DEFLATED) as zf:
        for key, arr in arrays_dict.items():
            with io.BytesIO() as buf:
                np.save(buf, arr)
                zf.writestr(f"{key}.npy", buf.getvalue())

def extract_and_save_stim_npz(
    sess_folder: Path,
    out_root: Path,
    stim_stream_name: str = "Stim channel",
    chunk_s: float = 60.0,
    chanmap_perm: np.ndarray | None = None,
    buffer_ms: float = 0.6,         # << how much pre/post “buffer” (samples) your MATLAB code used
    detect_on_channel: int | None = None,  # << optionally force detection channel
):
    bundle_dir = out_root / f"{sess_folder.name}_Intan_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    rec = read_intan_recording(sess_folder, stream_name=stim_stream_name)

    # Guarded reordering (only if counts match)
    if chanmap_perm is not None and rec.get_num_channels() == chanmap_perm.size:
        rec = reorder_recording_to_geometry(rec, chanmap_perm)
        order = "geometry"
        perm_geom_to_device = chanmap_perm.tolist()
        perm_device_to_geom = np.argsort(chanmap_perm).tolist()
    else:
        if chanmap_perm is not None:
            print(f"[{sess_folder.name}] '{stim_stream_name}': skip reordering "
                  f"(n_ch={rec.get_num_channels()} vs perm={chanmap_perm.size})")
        order = "device"
        perm_geom_to_device = None
        perm_device_to_geom = None

    fs = float(rec.get_sampling_frequency())

    # ------------ save the raw stim stream in chunked form (unchanged) ------------
    meta = dict(
        session=sess_folder.name,
        stream_name=stim_stream_name,
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
    out_npz = bundle_dir / "stim_stream.npz"
    _save_npz_streaming(out_npz, _iter_recording_chunks(rec, chunk_s), meta)
    print(f"[STIM] saved stim stream -> {out_npz}")

    # ------------ compute triggers/blocks from the stim stream ------------
    # For simplicity and robustness here, load the full stim stream into memory for detection.
    # (If this is too big, we can stream-detect later.)
    seg = 0
    n_samp = int(rec.get_num_samples(segment_index=seg))
    stim_traces = rec.get_traces(
        start_frame=0,
        end_frame=n_samp,
        channel_ids=None,   # all channels
        segment_index=seg,
        return_scaled=True
    )
    # stim_preproc expects (n_channels, n_samples)
    stim_data = stim_traces.T
    buffer_samples = int(round(buffer_ms * fs))
    cfg = StimTriggerConfig(buffer_samples=buffer_samples)

    res = extract_stim_triggers_and_blocks(
        stim_data=stim_data,
        fs=fs,
        cfg=cfg,
        use_first_active_channel=(detect_on_channel is None),
        channel_index=detect_on_channel,
    )

    # match MATLAB’s repeat_gap_threshold = 2 * (2*buffer + 1)
    repeat_gap_threshold_samples = np.array(
        2 * (2 * buffer_samples + 1), dtype=np.int64
    )

    # ------------ append results into the SAME NPZ ------------
    extras = {
        "active_channels": res.active_channels.astype(np.int64),
        "trigger_pairs": res.trigger_pairs.astype(np.int64),     # (n_events, 2)
        "trigger_starts": res.trigger_starts.astype(np.int64),   # (n_events,)
        "block_boundaries": res.block_boundaries.astype(np.int64),  # (n_blocks+1,)
        "repeat_gap_threshold_samples": repeat_gap_threshold_samples,
        "buffer_samples": np.array(buffer_samples, dtype=np.int64),
    }
    _append_npz_arrays(out_npz, extras)

    # also drop a few meta hints so downstream code can discover the fields easily
    # (we add/overwrite a tiny JSON sidecar inside the zip as text)
    try:
        import json, io, zipfile
        hints = {
            "stim_analysis_keys": list(extras.keys()),
            "stim_detection_channel": int(res.active_channels[0]) if res.active_channels.size else None,
        }
        with zipfile.ZipFile(out_npz, mode="a", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("stim_analysis_meta.json", json.dumps(hints, indent=2))
    except Exception as e:
        print(f"[STIM] warning: could not write stim_analysis_meta.json: {e}")

    return out_npz

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
# Convenience
# ----------------------
def list_intan_sessions(root: Path) -> list[Path]:
    root = Path(root)
    return sorted([p for p in root.iterdir() if p.is_dir()])

