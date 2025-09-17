from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, Tuple

import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from probeinterface import Probe


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


# ----------------------
# Convenience
# ----------------------
def list_intan_sessions(root: Path) -> list[Path]:
    root = Path(root)
    return sorted([p for p in root.iterdir() if p.is_dir()])

