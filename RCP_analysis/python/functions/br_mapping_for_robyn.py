from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import re
import numpy as np
import pandas as pd

# Parsing MAP file excel file (xlsm)
def _parse_nsp_channel(val) -> Optional[int]:
    """
    Parse excel mapping file into an NSP channel int
    Ex: converts 'ch-15' to 15
    """
    if val is None:
        return None
    if isinstance(val, (int, np.integer)):
        return int(val)
    m = re.search(r'(\d+)', str(val))
    return int(m.group(1)) if m else None

def load_UA_mapping_from_excel(xls_path: Path, sheet: str | int = 0, n_elec: int | None = None) -> Dict[str, Any]:
    """
    Read the excel sheet that has columns like:
      - 'NSP ch' (values like 'ch-01')
      - 'Elec#'  (1..N)
    This function resolves the non-sequential mapping ("smattering") between electrode
    numbers and NSP channel IDs. The output array is indexed by electrode position:
        mapped_nsp[i] = NSP channel wired to Electrode #(i+1)
    Example:
        If the MAP file indicates that Electrode #1 is connected to NSP ch-146,
        then mapped_nsp[0] == 146
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
        raise ValueError(f"Could not find NSP/Electrode columns in {xls_path.name}. Columns: {list(df.columns)}")

    nsp  = df[nsp_col].map(_parse_nsp_channel)
    elec = pd.to_numeric(df[elec_col], errors="coerce").astype("Int64")
    mask = nsp.notna() & elec.notna()
    nsp  = nsp[mask].astype(int).to_numpy()
    elec = elec[mask].astype(int).to_numpy()

    max_elec = int(n_elec or elec.max())
    mapped_nsp = np.zeros(max_elec, dtype=int)
    mapped_nsp[elec - 1] = nsp
    return {"mapped_nsp": mapped_nsp, "n_channels": int(max_elec)}

def align_mapping_index_to_recording(recording, mapped_nsp: np.ndarray) -> np.ndarray:
    """
    Align an array of NSP channel numbers (mapped_nsp) to the row
    indices of a SpikeInterface Recording.

    Returns an array of length len(mapped_nsp):
      - valid row index if that NSP channel is present in recording
      - -1 if not present
    """
    ch_ids = np.array(recording.get_channel_ids())
    id_to_row = {}

    # try direct numeric match
    try:
        id_to_row = {int(cid): i for i, cid in enumerate(ch_ids.astype(int))}
    except Exception:
        pass

    # fallback: check common property keys
    if not id_to_row:
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
                break

    # build output: -1 for missing
    idx_rows = np.full(mapped_nsp.shape, -1, dtype=int)
    for i, nsp in enumerate(mapped_nsp):
        if nsp in id_to_row:
            idx_rows[i] = id_to_row[nsp]

    return idx_rows

def apply_ua_mapping_properties(recording, mapped_nsp: np.ndarray):
    """
    Stamp UA mapping info onto the Recording without geometry.

    Per-channel properties (length == n_channels):
      - 'ua_electrode'   : Electrode number (1..N) for that recording row, or -1
      - 'ua_nsp_channel' : NSP channel id mapped to that row, or -1

    Per-recording annotation (any length):
      - 'ua_row_index_from_electrode' : np.ndarray len == len(mapped_nsp),
        where entry i is the recording row index for electrode (i+1), or -1 if absent.
    """
    idx_rows = align_mapping_index_to_recording(recording, mapped_nsp)  # shape = (N_elec,)
    n_ch = recording.get_num_channels()

    # Per-channel arrays
    ua_elec_per_row = -np.ones(n_ch, dtype=int)
    ua_nsp_per_row  = -np.ones(n_ch, dtype=int)

    # Fill only rows that exist in the recording
    for elec_idx0, row in enumerate(idx_rows):
        if 0 <= row < n_ch:
            ua_elec_per_row[row] = elec_idx0 + 1            # 1-based electrode number
            ua_nsp_per_row[row]  = int(mapped_nsp[elec_idx0])

    # Set per-channel properties (must be length n_ch)
    recording.set_property("ua_electrode", ua_elec_per_row)
    recording.set_property("ua_nsp_channel", ua_nsp_per_row)

    # Store the per-electrode -> row-index map as an annotation (any shape allowed)
    recording.set_annotation("ua_row_index_from_electrode", idx_rows.astype(int))

    mapped = int((ua_elec_per_row > 0).sum())
    print(f"[MAP] stamped UA mapping on {mapped}/{n_ch} rows (no geometry).")
