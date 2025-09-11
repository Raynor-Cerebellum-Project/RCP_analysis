from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat
import h5py
import spikeinterface.extractors as se

# ---------- Excel -> mapping ----------

def _norm(s: str) -> str:
    return ''.join(c.lower() for c in s if c.isalnum() or c == '_')

def _find_col(cols, *names):
    norm = {_norm(c): c for c in cols}
    for n in names:
        k = _norm(n)
        if k in norm:
            return norm[k]
    return None

def _parse_nsp_channel(val) -> Optional[int]:
    """
    Accepts 'ch-15', 'CH 015', '15', 15, etc., returns int 15.
    """
    if val is None:
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    s = str(val)
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else None

def load_ua_geom_from_excel(xls_path: Path,
                            sheet: str | int | None = 0) -> Dict[str, Any]:
    """
    Read the first tab with the NSP map (e.g. 'Impedance Values from Automated').

    Expects 2 columns (case/space-insensitive):
      - NSP channel: e.g. 'NSP ch' values like 'ch-15'
      - Electrode number: e.g. 'Elec#' / 'Elec #' / 'Electrode'

    Returns dict:
      {
        'elec_to_nsp': np.ndarray shape (n_elec,), where entry g (0-based) is NSP ch number for geometry g
        'geom_corr_ind_nsp': np.ndarray shape (n_elec,), integer NSP channel numbers in geometry order,
        'n_channels': int
      }

    Note: 'geom_corr_ind_nsp' is expressed in NSP channel **numbers**, NOT row indices.
          Use align_geom_index_to_recording(...) to convert it to actual row indices for a RecordingExtractor.
    """
    xls_path = Path(xls_path)
    if not xls_path.exists():
        raise FileNotFoundError(f"Excel not found: {xls_path}")

    df = pd.read_excel(xls_path, sheet_name=sheet, engine="openpyxl")
    # Try to recover header if the first row isn't the real header
    if not any(df.columns.str.contains(r'NSP|Elec|Electrode', case=False, regex=True)):
        for hdr in range(1, min(8, len(df))):
            df2 = pd.read_excel(xls_path, sheet_name=sheet, header=hdr, engine="openpyxl")
            if any(df2.columns.str.contains(r'NSP|Elec|Electrode', case=False, regex=True)):
                df = df2
                break

    nsp_col  = _find_col(df.columns, 'NSP ch', 'NSP', 'NSP Channel', 'Channel', 'CH')
    elec_col = _find_col(df.columns, 'Elec#', 'Elec #', 'Elec', 'Electrode', 'Electrode #')
    if nsp_col is None or elec_col is None:
        raise ValueError(f"Could not find NSP/Electrode columns in {xls_path.name} "
                         f"(found columns: {list(df.columns)})")

    # Parse mapping
    nsp = df[nsp_col].map(_parse_nsp_channel)
    elec = df[elec_col].astype("Int64")  # pandas NA-safe int
    mask = nsp.notna() & elec.notna()
    nsp = nsp[mask].astype(int).to_numpy()
    elec = elec[mask].astype(int).to_numpy()

    # Normalize Electrode numbers to 1..N, then make 0-based geometry index
    max_elec = int(elec.max())
    elec_to_nsp = np.zeros(max_elec, dtype=int)  # 0 means "unknown"
    elec_to_nsp[elec - 1] = nsp

    # Geometry order is Elec# ascending (1..N) -> NSP numbers
    geom_corr_ind_nsp = elec_to_nsp.copy()  # NSP ch number per geometry position
    # Note: this is still in NSP numbering, not row indices

    return {
        "elec_to_nsp": elec_to_nsp,                 # (N,) values: NSP ch numbers
        "geom_corr_ind_nsp": geom_corr_ind_nsp,     # (N,) values: NSP ch numbers in geometry order
        "n_channels": int(max_elec),
    }

# ---------- NSP numbering -> Recording row indices ----------

def align_geom_index_to_recording(recording, geom_corr_ind_nsp: np.ndarray) -> np.ndarray:
    """
    Convert a geometry-order NSP channel list -> actual row indices for a SpikeInterface Recording.

    Strategy:
      1) Try to match NSP channel numbers to `recording.get_channel_ids()` directly.
      2) Otherwise, look for common Blackrock properties in `get_property_keys()`
         like 'electrode_id', 'nsx_chan_id', 'channel_name', and match by integer.
      3) Fall back to identity (warn) if nothing matches.

    Returns
    -------
    idx_rows : np.ndarray[int], shape (N,)
        0-based row indices s.t. data_geom = data_device[idx_rows, :]
    """
    import numpy as np

    ch_ids = np.array(recording.get_channel_ids())
    N = geom_corr_ind_nsp.size

    # Case 1: channel_ids are actual NSP numbers
    try:
        ch_ids_int = ch_ids.astype(int)
        id_to_row = {int(cid): i for i, cid in enumerate(ch_ids_int)}
        if all(int(n) in id_to_row for n in geom_corr_ind_nsp):
            return np.array([id_to_row[int(n)] for n in geom_corr_ind_nsp], dtype=int)
    except Exception:
        pass

    # Case 2: look into properties for a numeric NSP/electrode id
    for key in ("electrode_id", "nsx_chan_id", "nsp_channel", "channel_name"):
        if key in recording.get_property_keys():
            vals = recording.get_property(key)
            try:
                # Try to coerce values to int (e.g., "ch-15" -> 15)
                def v2i(v):
                    if isinstance(v, (int, np.integer)):
                        return int(v)
                    m = re.search(r'(\d+)', str(v))
                    return int(m.group(1)) if m else None
                ints = [v2i(v) for v in vals]
                id_to_row = {iv: i for i, iv in enumerate(ints) if iv is not None}
                if all(int(n) in id_to_row for n in geom_corr_ind_nsp):
                    return np.array([id_to_row[int(n)] for n in geom_corr_ind_nsp], dtype=int)
            except Exception:
                continue

    # Fallback: identity with warning
    print("[WARN] Could not align NSP numbers to recording channels; using identity mapping.")
    return np.arange(N, dtype=int)

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

# RCP_analysis/python/functions/br_preproc.py

def read_blackrock_recording(sess_folder: Path, stream_id: str | None = None):
    """
    If `sess_folder` contains multiple Blackrock bases (e.g. *_001.*, *_002.*),
    choose the first base and pass a specific file to neo instead of the folder.
    """
    sess_folder = Path(sess_folder)

    nsx_files = sorted(sess_folder.glob("*.ns*"))
    if not nsx_files:
        raise FileNotFoundError(f"No .nsx files found in {sess_folder}")

    # Group by base (prefix before extension)
    bases = {}
    for f in nsx_files:
        bases.setdefault(f.stem, []).append(f)

    if len(bases) > 1:
        # Choose one base deterministically (or make this configurable)
        base = sorted(bases.keys())[0]
        # Prefer an NS5 if present, else any NS*
        candidates = sorted(sess_folder.glob(f"{base}.ns5")) or sorted(sess_folder.glob(f"{base}.ns*"))
        if not candidates:
            raise FileNotFoundError(f"Could not locate a file for base {base} in {sess_folder}")
        file_path = str(candidates[0])
    else:
        # Single base: passing the directory is fine, but passing a file is safer
        base = next(iter(bases.keys()))
        candidates = sorted(sess_folder.glob(f"{base}.ns5")) or sorted(sess_folder.glob(f"{base}.ns*"))
        file_path = str(candidates[0])

    # Now open the specific file; neo will load the rest of the matching set
    return se.read_blackrock(file_path, stream_id=stream_id, all_annotations=True)

# UA geometry loader
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
