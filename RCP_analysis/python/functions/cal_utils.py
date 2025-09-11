from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
from scipy.io import loadmat


def _get_struct_field(mat_struct, key: str):
    """Robustly read a field from a MATLAB struct loaded by scipy.io.loadmat."""
    if mat_struct is None:
        return None
    # Attribute-style (object-like)
    try:
        val = getattr(mat_struct, key, None)
        if val is not None:
            return val
    except Exception:
        pass
    # Numpy void / record array style
    try:
        if hasattr(mat_struct, "dtype") and mat_struct.dtype and mat_struct.dtype.names:
            if key in mat_struct.dtype.names:
                return mat_struct[key]
    except Exception:
        pass
    return None


def load_sample_index_from_cal(
    sess_folder: Path,
    key_candidates: Iterable[str] = ("Blackrock_idx", "Intan_idx"),
) -> np.ndarray | None:
    """
    Load a sample index vector from the first '*Cal.mat' in sess_folder.

    Returns a 0-based int64 numpy array, or None if not found.
    """
    cal_files = sorted(Path(sess_folder).glob("*Cal.mat"))
    if not cal_files:
        return None

    M = loadmat(cal_files[0], squeeze_me=True, struct_as_record=False)
    Data = M.get("Data", None)
    if Data is None:
        return None

    for key in key_candidates:
        raw = _get_struct_field(Data, key)
        if raw is None:
            continue
        try:
            idx = np.asarray(raw).ravel().astype(np.int64)
        except Exception:
            continue
        # MATLAB (1-based) -> Python (0-based)
        return idx - 1
    return None
