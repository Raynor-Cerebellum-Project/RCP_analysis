from __future__ import annotations
from pathlib import Path
import spikeinterface.extractors as se
import numpy as np
import h5py
from .cal_utils import load_sample_index_from_cal

NCH = 128


def read_intan_recording(
    sess_folder: Path,
    stream_name: str = "RHS2000 amplifier channel",
):
    """
    Load an Intan session folder as a SpikeInterface RecordingExtractor.

    sess_folder: path containing .rhd/.rhs files
    stream_name: name of the recording stream (default matches common RHS)
    """
    rec = se.read_intan(str(sess_folder), stream_name=stream_name)
    return rec


def load_intan_sample_index(sess_folder: Path) -> np.ndarray | None:
    """
    Convenience wrapper to fetch the Intan index from '*Cal.mat', 0-based.
    """
    return load_sample_index_from_cal(
        sess_folder,
        key_candidates=("Intan_idx", "RHS_idx", "IntanIndex"),
    )


# ---------------- Stim geometry helpers ----------------
def _load_geom_corr_ind_from_config() -> np.ndarray | None:
    """
    Try to load a geometry-row reorder index from a repo-level MAT.
    Looks for: <pkg file>/../../config/ImecPrimateStimRec128_042421.mat
    Accepts several key names; assumes MATLAB 1-based indices.
    """
    maybe = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "ImecPrimateStimRec128_042421.mat"
    )
    if not maybe.exists():
        return None
    try:
        M = loadmat(maybe, squeeze_me=True, struct_as_record=False)
        for key in ("geom_corr_ind", "GeomCorrInd", "neuropixel_index", "NP_index"):
            if key in M:
                arr = np.asarray(M[key]).ravel().astype(int)
                return arr - 1  # MATLAB 1-based -> 0-based
    except Exception as e:
        print(f"[stim] could not read geometry index from {maybe.name}: {e}")
    return None


geom_corr_ind = (
    _load_geom_corr_ind_from_config()
)  # None is OK; we’ll skip reordering if absent


def _load_stim_matrix(path: Path) -> np.ndarray | None:
    """
    Return Stim_data/stim_data as a 2D float array or None.
    Supports both classic MAT and v7.3 (HDF5) files.
    Normalizes orientation so that time is the longer axis (columns).
    """
    if not path.exists():
        return None

    # Try classic MAT first
    try:
        M = loadmat(path, squeeze_me=False, struct_as_record=False)
        key = (
            "Stim_data"
            if "Stim_data" in M
            else ("stim_data" if "stim_data" in M else None)
        )
        if key is not None:
            arr = np.asarray(M[key])
            arr = np.array(arr, dtype=float)
            if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                arr = arr.T
            return arr
    except NotImplementedError:
        # v7.3 fallback via h5py
        pass
    except Exception as e:
        print(f"[stim] MAT read failed for {path.name}: {e}")

    # HDF5 (MAT v7.3)
    try:
        with h5py.File(path, "r") as f:
            # typical top-level dataset names
            for key in ("Stim_data", "stim_data"):
                if key in f:
                    ds = f[key]
                    arr = np.array(ds, dtype=float)
                    if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                        arr = arr.T
                    return arr
            # look 1 level deep for common names
            for key in f.keys():
                obj = f[key]
                if isinstance(obj, h5py.Dataset):
                    continue
                for cand in ("Stim_data", "stim_data"):
                    if cand in obj:
                        arr = np.array(obj[cand], dtype=float)
                        if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                            arr = arr.T
                        return arr
    except Exception as e:
        print(f"[stim] HDF5 read failed for {path.name}: {e}")

    return None


def load_stim_device(source: Path) -> np.ndarray | None:
    """
    Load the stim matrix from a session folder (expects 'stim_data.mat') or from a .mat file path.
    Returns rows in DEVICE order (no geometry reindex applied).
    """
    if source.is_dir():
        mat = _load_stim_matrix(source / "stim_data.mat")
    else:
        mat = _load_stim_matrix(source)
    if mat is None:
        return None
    if mat.ndim == 1:
        return mat[np.newaxis, :]
    if NCH and mat.shape[0] != NCH:
        print(
            f"[stim] Unexpected #rows {mat.shape[0]} (expected {NCH}). Continuing anyway."
        )
    return np.asarray(mat, float)


def load_stim_geometry(source: Path) -> np.ndarray | None:
    """
    Return stim matrix with rows in GEOMETRY order if a reorder index is available.
    `source` may be a session folder (containing 'stim_data.mat') or a direct path to a .mat file.
    """
    mat_dev = load_stim_device(source)
    if mat_dev is None:
        return None
    if geom_corr_ind is not None and mat_dev.shape[0] >= geom_corr_ind.size:
        return mat_dev[geom_corr_ind, :]
    return mat_dev
