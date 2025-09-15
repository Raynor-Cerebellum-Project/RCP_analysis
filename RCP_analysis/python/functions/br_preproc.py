from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import re
import numpy as np
import pandas as pd
import spikeinterface.extractors as se

# Blackrock loader (can run when multiple trials exist)
def read_blackrock_recording(sess_folder: Path, stream_id: str | None = None):
    """
    Robustly load a Blackrock Recording.

    Accepts either:
      - a real directory containing *.ns* files (classic case), OR
      - a 'pseudo session' path whose .parent is the real folder and whose .name
        is the base (e.g., /.../Blackrock/Nike_reaching_002_001).
    """
    sess_folder = Path(sess_folder)

    if sess_folder.is_dir():
        # classic: directory with files
        nsx_files = sorted(sess_folder.glob("*.ns*"))
        if not nsx_files:
            raise FileNotFoundError(f"No .nsx files found in {sess_folder}")
        bases = sorted({f.stem for f in nsx_files})
        base = bases[0]
        root = sess_folder
    else:
        # pseudo session: parent is real folder, name is base
        root = sess_folder.parent
        base = sess_folder.name
        nsx_files = sorted(root.glob(f"{base}.ns*"))
        if not nsx_files:
            raise FileNotFoundError(f"No .nsx files found for base {base} in {root}")

    # Prefer .ns5 (30 kHz), fallback to any .ns*
    candidates = sorted(root.glob(f"{base}.ns5")) or sorted(root.glob(f"{base}.ns*"))
    if not candidates:
        raise FileNotFoundError(f"Could not locate a file for base {base} in {root}")

    return se.read_blackrock(str(candidates[0]), stream_id=stream_id, all_annotations=True)


# Parsing MAP file excel file (xlsm)
def _parse_nsp_channel(val) -> Optional[int]:
    """Accept 'ch-15', 'CH 015', '15', 15 -> 15."""
    if val is None:
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    m = re.search(r'(\d+)', str(val))
    return int(m.group(1)) if m else None

def load_UA_mapping_from_excel(
    xls_path: Path, sheet: str | int = 0, n_elec: int | None = None
) -> Dict[str, Any]:
    """
    Read the excel sheet that has columns like:
      - 'NSP ch' (values like 'ch-01')
      - 'Elec#'  (1..N)
    This function resolves the non-sequential mapping ("smattering") between electrode
    numbers and NSP channel IDs. The output array is indexed by electrode position:
        mapped_nsp[i] = NSP channel wired to Electrode #(i+1)
    Example:
        If the MAP file indicates that Electrode #1 is connected to NSP ch-146,
        then mapped_nsp[0] == 146.
    Input:
        xls_path: Path to the excel file
        sheet:    Sheet name or index (default 0)
        n_elec:   Optionally specify the number of electrodes (otherwise inferred)
    Returns:
      {
        'mapped_nsp': np.ndarray (N,), NSP channel numbers in **correct order**,
        'n_channels': int
      }
    """
    xls_path = Path(xls_path)
    if not xls_path.exists():
        raise FileNotFoundError(f"Excel not found: {xls_path}")

    df = pd.read_excel(xls_path, sheet_name=sheet)

    # Find likely columns by fuzzy match
    def find_col(names):
        cols = {re.sub(r'[^a-z0-9]+', '', c.lower()): c for c in map(str, df.columns)}
        for n in names:
            key = re.sub(r'[^a-z0-9]+', '', n.lower())
            if key in cols:
                return cols[key]
        return None

    nsp_col  = find_col(["NSP ch", "NSP", "NSP Channel", "Channel", "CH"])
    elec_col = find_col(["Elec#", "Elec #", "Electrode", "Electrode #"])

    if nsp_col is None or elec_col is None:
        raise ValueError(
            f"Could not find NSP/Electrode columns in {xls_path.name}. "
            f"Columns seen: {list(df.columns)}"
        )

    nsp  = df[nsp_col].map(_parse_nsp_channel)
    elec = pd.to_numeric(df[elec_col], errors="coerce").astype("Int64")

    mask = nsp.notna() & elec.notna()
    nsp  = nsp[mask].astype(int).to_numpy()
    elec = elec[mask].astype(int).to_numpy()

    max_elec = int(n_elec or elec.max())
    mapped_nsp = np.zeros(max_elec, dtype=int)  # zero => unknown
    mapped_nsp[elec - 1] = nsp

    return {"mapped_nsp": mapped_nsp, "n_channels": int(max_elec)}

# ------------------------
# Align mapping to Recording + optional XY
# ------------------------
def align_mapping_index_to_recording(recording, mapped_nsp: np.ndarray) -> np.ndarray:
    """
    Convert a mapping-order list of NSP channel numbers -> recording row indices.
    """
    ch_ids = np.array(recording.get_channel_ids())

    # Case 1: channel ids are NSP numbers
    try:
        id_to_row = {int(cid): i for i, cid in enumerate(ch_ids.astype(int))}
        if all(int(n) in id_to_row for n in mapped_nsp if n > 0):
            return np.array([id_to_row.get(int(n), -1) for n in mapped_nsp], dtype=int)
    except Exception:
        pass

    # Case 2: try properties
    for key in ("electrode_id", "nsx_chan_id", "nsp_channel", "channel_name"):
        if key in recording.get_property_keys():
            vals = recording.get_property(key)
            def v2i(v):
                if isinstance(v, (int, np.integer)):
                    return int(v)
                m = re.search(r'(\d+)', str(v))
                return int(m.group(1)) if m else None
            ints = [v2i(v) for v in vals]
            id_to_row = {iv: i for i, iv in enumerate(ints) if iv is not None}
            if all(int(n) in id_to_row for n in mapped_nsp if n > 0):
                return np.array([id_to_row.get(int(n), -1) for n in mapped_nsp], dtype=int)

    # Fallback: identity for the first N channels
    print("[WARN] Could not align NSP numbers to recording channels; using identity mapping.")
    N = mapped_nsp.size
    return np.arange(min(N, recording.get_num_channels()), dtype=int)


def ua_xy_geometry(n_elec: int, pitch_mm: float = 0.4) -> np.ndarray: ## NEED TO FIX
    """
    Simple Utah layouts (geometry order):
      64  => 8x8
      128 => 2 8x8 blocks
      256 => 4 8x8 blocks
    Returns (N,2) array of (x,y) in mm.
    """
    def block():
        r, c = np.mgrid[0:8, 0:8]
        return np.column_stack([c.ravel(), r.ravel()]) * pitch_mm

    if n_elec == 64:
        return block()
    if n_elec == 128:
        top = block()
        bot = block() + np.array([0, 8 * pitch_mm])
        return np.vstack([top, bot])
    if n_elec == 256:
        tl = block()
        tr = block() + np.array([8 * pitch_mm, 0])
        bl = block() + np.array([0, 8 * pitch_mm])
        br = block() + np.array([8 * pitch_mm, 8 * pitch_mm])
        return np.vstack([tl, tr, bl, br])
    raise ValueError(f"Unsupported UA size {n_elec} (expected 64/128/256)")


def apply_ua_geometry_to_recording(recording, mapped_nsp: np.ndarray, pitch_mm: float = 0.4) -> dict:
    idx_rows = align_mapping_index_to_recording(recording, mapped_nsp)

    # how many geometry entries we’ll actually place
    N = (idx_rows != -1).sum() if np.any(idx_rows == -1) else len(idx_rows)

    xy_geom = ua_xy_geometry(len(mapped_nsp), pitch_mm)[:N]

    # allocate for ALL channels to make row indexing valid
    num_ch = recording.get_num_channels()
    xy_rows = np.zeros((num_ch, 2), dtype=float)

    # only place the first N aligned entries
    # (optionally mask valid = idx_rows[:N] >= 0 if you want to be extra safe)
    xy_rows[idx_rows[:N]] = xy_geom

    recording.set_channel_locations(xy_rows)
    return {"idx_rows": idx_rows[:N], "xy_rows": xy_rows}

