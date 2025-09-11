# RCP_analysis/python/functions/br_preproc.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import h5py
import spikeinterface.extractors as se
from typing import Optional, Dict, Any


def load_session_geometry(
    sess_name: str,
    params: Any,  # your experimentParams / dataclass
    repo_root: Path,  # repo root to resolve relative geom paths
) -> Optional[dict]:
    """
    For UA (Blackrock) sessions, load UA geometry from YAML path.
    For other probes, return None (extend later for NN_DBS/NPRW).
    Returns a dict like: {"geom_corr_ind": np.ndarray|None, "xy": np.ndarray|None, "n_channels": int}
    """
    sessions = getattr(params, "sessions", {}) or {}
    probes = getattr(params, "probes", {}) or {}

    sess_cfg = sessions.get(sess_name, {})
    probe_tag = sess_cfg.get("probe") or (next(iter(probes.keys())) if probes else None)
    if probe_tag is None or probe_tag != "UA":
        return None  # only handle UA here

    probe_cfg = probes.get("UA", {})
    rel = probe_cfg.get("geom_mat_rel") or getattr(params, "geom_mat_rel", None)
    if not rel:
        print(f"[WARN] UA geometry path not set in params for session '{sess_name}'")
        return None

    geom_path = (repo_root / rel) if not str(rel).startswith("/") else Path(rel)
    if not geom_path.exists():
        print(f"[WARN] UA geometry file not found: {geom_path}")
        return None

    try:
        return load_br_geometry(geom_path)
    except Exception as e:
        print(f"[WARN] Could not load UA geometry from {geom_path}: {e}")
        return None


# -------- existing ----------
def read_blackrock_recording(sess_folder: Path, stream_id: str | None = None):
    """
    Load a Blackrock session folder as a SpikeInterface RecordingExtractor.
    sess_folder: path containing .nsx/.nev files
    stream_id:   optional stream specifier (e.g. "2", "5"), pass None for auto
    """
    sess_folder = Path(sess_folder)
    nsx_files = sorted(sess_folder.glob("*.ns*"))
    if not nsx_files:
        raise FileNotFoundError(f"No .nsx files found in {sess_folder}")
    rec = se.read_blackrock(str(sess_folder), stream_id=stream_id, all_annotations=True)
    return rec


# -------- new: UA geometry loader ----------
def _mat_read_any(path: Path) -> dict:
    """Read MATLAB file (classic or v7.3) into a simple dict-like namespace."""
    try:
        return loadmat(path, squeeze_me=True, struct_as_record=False)  # classic MAT
    except NotImplementedError:
        pass
    # v7.3 (HDF5) fallback
    out = {}
    with h5py.File(path, "r") as f:

        def fetch(name):
            obj = f[name]
            if isinstance(obj, h5py.Dataset):
                arr = np.array(obj)
                # match loadmat(squeeze_me=True) behavior a bit
                return np.squeeze(arr)
            return obj

        for k in f.keys():
            try:
                out[k] = fetch(k)
            except Exception:
                pass
    return out


def _maybe_to_1d(a) -> np.ndarray | None:
    try:
        arr = np.asarray(a)
        if arr.size == 0:
            return None
        return arr.ravel()
    except Exception:
        return None


def load_br_geometry(mat_path: Path) -> dict:
    """
    Load Utah Array geometry/mapping from a MATLAB file.

    Recognized patterns (any one is enough):
      - 'geom_corr_ind' / 'GeomCorrInd' / 'ua_index' : 1- or 0-based reorder index
      - 'ChanMap' / 'chan_map' / 'ChMap'            : 1-based device->geometry map
      - Row/Col with 'Chan' (or XY) to infer order   : returns XY and leaves index None

    Returns
    -------
    info : dict with keys:
        'geom_corr_ind' : np.ndarray[int] or None  (0-based index: device->geometry)
        'xy'            : np.ndarray[float] shape (nch, 2) or None
        'n_channels'    : int
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"UA geometry file not found: {mat_path}")

    M = _mat_read_any(mat_path)

    # 1) Direct index field(s)
    for key in ("geom_corr_ind", "GeomCorrInd", "ua_index", "UA_index"):
        if key in M:
            idx = _maybe_to_1d(M[key])
            if idx is not None:
                idx = idx.astype(int)
                # if it looks 1-based, shift to 0-based
                if idx.min() == 1 or idx.max() == idx.size:
                    idx = idx - 1
                return {"geom_corr_ind": idx, "xy": None, "n_channels": int(idx.size)}

    # 2) ChanMap style (1-based map device->geometry)
    for key in ("ChanMap", "chan_map", "ChMap", "ChannelMap", "channel_map"):
        if key in M:
            cm = _maybe_to_1d(M[key])
            if cm is not None:
                cm = cm.astype(int)
                if cm.min() == 1 or cm.max() == cm.size:
                    cm = cm - 1
                # ChanMap typically gives an order; convert to an index that reorders device->geometry.
                # If cm[k] = geo_idx of device k, then the index that reorders device rows into geometry
                # order is argsort over inverse mapping:
                # We want output[0] = device row corresponding to geometry 0, etc.
                inv = np.empty_like(cm)
                inv[cm] = np.arange(cm.size)
                return {"geom_corr_ind": inv, "xy": None, "n_channels": int(cm.size)}

    # 3) Row/Col or XY coordinates (no explicit index)
    rows = None
    cols = None
    for rk in ("Row", "row", "rows"):
        if rk in M:
            rows = _maybe_to_1d(M[rk])
            break
    for ck in ("Col", "col", "cols"):
        if ck in M:
            cols = _maybe_to_1d(M[ck])
            break
    xs = None
    ys = None
    # Some UA files store 'X', 'Y' or 'ChanX', 'ChanY'
    for xk in ("X", "x", "ChanX", "chan_x"):
        if xk in M:
            xs = _maybe_to_1d(M[xk])
            break
    for yk in ("Y", "y", "ChanY", "chan_y"):
        if yk in M:
            ys = _maybe_to_1d(M[yk])
            break

    xy = None
    if xs is not None and ys is not None and xs.size == ys.size:
        xy = np.column_stack([xs.astype(float), ys.astype(float)])
        n_channels = xy.shape[0]
        return {"geom_corr_ind": None, "xy": xy, "n_channels": int(n_channels)}

    if rows is not None and cols is not None and rows.size == cols.size:
        xy = np.column_stack([cols.astype(float), rows.astype(float)])
        n_channels = xy.shape[0]
        return {"geom_corr_ind": None, "xy": xy, "n_channels": int(n_channels)}

    # Nothing recognized
    raise ValueError(
        f"Could not find UA geometry mapping in {mat_path}. "
        "Expected one of: geom_corr_ind/ChanMap or Row/Col or X/Y."
    )
