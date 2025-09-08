
import numpy as np
from pathlib import Path
from scipy.io import loadmat

def _load_stim_matrix(path: Path) -> np.ndarray | None:
    """
    Return Stim_data/stim_data as a 2D float array or None.
    Supports both classic MAT and v7.3 (HDF5) files.
    """
    if not path.exists():
        return None

    # Try classic MAT first
    try:
        M = loadmat(path, squeeze_me=False, struct_as_record=False)
        key = "Stim_data" if "Stim_data" in M else ("stim_data" if "stim_data" in M else None)
        if key is None:
            return None
        arr = np.asarray(M[key])
        # ensure 2D float
        arr = np.array(arr, dtype=float)
        # normalize orientation: time should be the longer axis
        if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
            arr = arr.T
        return arr
    except NotImplementedError:
        # v7.3 fallback via h5py
        pass

    # HDF5 (MAT v7.3)
    try:
        with h5py.File(path, "r") as f:
            # typical top-level dataset names
            for key in ("Stim_data", "stim_data"):
                if key in f:
                    ds = f[key]
                    arr = np.array(ds, dtype=float)          # load to numpy
                    # normalize orientation: time should be the longer axis
                    if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                        arr = arr.T
                    return arr
            # if stored under a group, try a shallow scan
            for key in f.keys():
                obj = f[key]
                if isinstance(obj, h5py.Dataset):
                    continue
                # look for dataset children named Stim_data/stim_data
                for cand in ("Stim_data", "stim_data"):
                    if cand in obj:
                        arr = np.array(obj[cand], dtype=float)
                        if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                            arr = arr.T
                        return arr
    except Exception as e:
        print(f"[stim] HDF5 read failed for {path.name}: {e}")

    return None

def load_stim_geometry(sess_folder: Path) -> np.ndarray | None:
    """
    Return stim matrix with rows in GEOMETRY (MATLAB) order.
    The file is in DEVICE order; use G2D to reorder.
    """
    mat_dev = load_stim_device(sess_folder)
    if mat_dev is None:
        return None
    if mat_dev.shape[0] >= geom_corr_ind.size:
        # ✅ device (rows) -> geometry rows
        return mat_dev[geom_corr_ind, :]
    return mat_dev

def load_stim_device(sess_folder: Path) -> np.ndarray | None:
    """Load stim_data.mat as a 2D array (rows = device order, cols = time)."""
    mat = _load_stim_matrix(sess_folder / "stim_data.mat")    
    if mat is None:
        return None
    if mat.ndim == 1:
        # Single-channel vector -> treat as (1, T) device row
        return mat[np.newaxis, :]
    if mat.shape[0] != NCH:
        print(f"[stim] Unexpected #rows {mat.shape[0]} (expected {NCH}). Continuing anyway.")
    return np.asarray(mat, float)

