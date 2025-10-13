#!/usr/bin/env python3
"""
Convert aligned__*.npz (from your alignment pipeline) -> .mat

- Writes a single MATLAB struct 'aligned' with fields matching the NPZ keys.
- JSON in 'align_meta' is parsed to a MATLAB struct ('align_meta_struct').
- String lists (e.g., ua_region_names, beh_cam0_cols) become MATLAB cellstr.

Requires: numpy, scipy (for savemat). Optional: h5py if you enable the v7.3 fallback.

Usage:
  python npz_to_mat_aligned.py /path/to/Aligned   # folder containing aligned__*.npz
"""
from __future__ import annotations
from pathlib import Path
import json
import sys
import numpy as np
from scipy.io import savemat

# Optional: enable v7.3 fallback if you hit size limits
ENABLE_HDF5_FALLBACK = False
try:
    import h5py  # noqa: F401
except Exception:
    ENABLE_HDF5_FALLBACK = False


def _to_cellstr(seq) -> np.ndarray:
    """Convert a Python list/array of strings -> MATLAB cellstr via object ndarray."""
    if seq is None:
        return np.array([], dtype=object)
    if isinstance(seq, np.ndarray) and seq.dtype == object:
        # ensure pure strings inside
        return np.array([("" if s is None else str(s)) for s in seq.tolist()], dtype=object)
    # list/tuple or numeric array: cast elements to str
    return np.array([("" if s is None else str(s)) for s in np.asarray(seq).tolist()], dtype=object)


def _ensure_numeric(a):
    """Map None -> empty float32 array; keep numeric arrays as-is (cast to float32 for consistency)."""
    if a is None:
        return np.array([], dtype=np.float32)
    if isinstance(a, np.ndarray):
        # keep dtype if already numeric/bool/int
        if np.issubdtype(a.dtype, np.number) or a.dtype == bool:
            return a
        # object arrays that should be numeric: try to coerce
        try:
            return a.astype(np.float32)
        except Exception:
            return a
    return np.asarray(a)


def _parse_align_meta(align_meta):
    """JSON string -> dict suitable for MATLAB struct via savemat."""
    if align_meta is None:
        return {}
    try:
        if hasattr(align_meta, "item"):
            align_meta = align_meta.item()
        if isinstance(align_meta, (bytes, bytearray)):
            align_meta = align_meta.decode("utf-8", errors="ignore")
        if isinstance(align_meta, str):
            d = json.loads(align_meta)
        elif isinstance(align_meta, dict):
            d = align_meta
        else:
            return {}
        # prune non-serializable values and cast Paths, etc. to str
        out = {}
        for k, v in d.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
            elif isinstance(v, (list, tuple)):
                # if all scalar, keep; if mixed, stringify
                try:
                    arr = np.array(v)
                    if arr.dtype == object:
                        out[k] = _to_cellstr(v)
                    else:
                        out[k] = arr
                except Exception:
                    out[k] = str(v)
            else:
                out[k] = str(v)
        return out
    except Exception:
        return {}


def _build_mat_struct(z: np.lib.npyio.NpzFile) -> dict:
    """
    Prepare a dict with a single key 'aligned' -> sub-dict of fields.
    This maps directly to a MATLAB struct when passed to savemat.
    """
    # Pull everything present, default to empty arrays where appropriate
    def get(name, default=None):
        return z[name] if name in z.files else default

    aligned = {}

    # ---- Intan (on common grid) ----
    aligned["intan_rate_hz"]            = _ensure_numeric(get("intan_rate_hz", np.array([], dtype=np.float32)))
    aligned["intan_t_ms"]               = _ensure_numeric(get("intan_t_ms", np.array([], dtype=np.float32)))
    aligned["intan_t_ms_aligned"]       = _ensure_numeric(get("intan_t_ms_aligned", np.array([], dtype=np.float32)))
    aligned["intan_meta_raw"]           = get("intan_meta", None)  # could be dict/str/None
    aligned["intan_pcs"]                = _ensure_numeric(get("intan_pcs", np.array([], dtype=np.float32)))
    aligned["intan_explained_var"]      = _ensure_numeric(get("intan_explained_var", np.array([], dtype=np.float32)))
    aligned["intan_peaks"]              = _ensure_numeric(get("intan_peaks", np.array([], dtype=np.float32)))
    aligned["intan_peaks_t_ms"]         = _ensure_numeric(get("intan_peaks_t_ms", np.array([], dtype=np.float32)))
    aligned["intan_peaks_t_sample"]     = _ensure_numeric(get("intan_peaks_t_sample", np.array([], dtype=np.float32)))
    aligned["intan_peaks_t_ms_aligned"] = _ensure_numeric(get("intan_peaks_t_ms_aligned", np.array([], dtype=np.float32)))

    # ---- UA (resampled to common grid) ----
    aligned["ua_rate_hz"]               = _ensure_numeric(get("ua_rate_hz", np.array([], dtype=np.float32)))
    aligned["ua_t_ms"]                  = _ensure_numeric(get("ua_t_ms", np.array([], dtype=np.float32)))
    aligned["ua_t_ms_aligned"]          = _ensure_numeric(get("ua_t_ms_aligned", np.array([], dtype=np.float32)))
    aligned["ua_meta_raw"]              = get("ua_meta", None)
    aligned["ua_pcs"]                   = _ensure_numeric(get("ua_pcs", np.array([], dtype=np.float32)))
    aligned["ua_explained_var"]         = _ensure_numeric(get("ua_explained_var", np.array([], dtype=np.float32)))

    aligned["ua_row_to_elec"]           = _ensure_numeric(get("ua_row_to_elec", np.array([], dtype=np.int16)))
    aligned["ua_row_to_region"]         = _ensure_numeric(get("ua_row_to_region", np.array([], dtype=np.int8)))
    # names/cols as cellstr
    aligned["ua_region_names"]          = _to_cellstr(get("ua_region_names", np.array([], dtype=object)))
    aligned["ua_row_to_nsp"]            = _ensure_numeric(get("ua_row_to_nsp", np.array([], dtype=np.int16)))
    aligned["ua_row_index_from_electrode"] = _ensure_numeric(get("ua_row_index_from_electrode", np.array([], dtype=np.int16)))

    aligned["ua_peaks"]                 = _ensure_numeric(get("ua_peaks", np.array([], dtype=np.float32)))
    aligned["ua_peaks_t_ms"]            = _ensure_numeric(get("ua_peaks_t_ms", np.array([], dtype=np.float32)))
    aligned["ua_peaks_t_sample"]        = _ensure_numeric(get("ua_peaks_t_sample", np.array([], dtype=np.float32)))
    aligned["ua_peaks_t_ms_aligned"]    = _ensure_numeric(get("ua_peaks_t_ms_aligned", np.array([], dtype=np.float32)))

    # ---- Stim (absolute Intan ms) ----
    aligned["stim_ms"]                  = _ensure_numeric(get("stim_ms", np.array([], dtype=np.float32)))

    # ---- Behavior ----
    aligned["beh_ns5_sample"]           = _ensure_numeric(get("beh_ns5_sample", np.array([], dtype=np.int64)))
    aligned["beh_cam0"]                 = _ensure_numeric(get("beh_cam0", np.zeros((0, 0), dtype=np.float32)))
    aligned["beh_cam1"]                 = _ensure_numeric(get("beh_cam1", np.zeros((0, 0), dtype=np.float32)))
    aligned["beh_cam0_cols"]            = _to_cellstr(get("beh_cam0_cols", np.array([], dtype=object)))
    aligned["beh_cam1_cols"]            = _to_cellstr(get("beh_cam1_cols", np.array([], dtype=object)))
    aligned["beh_t_ms"]                 = _ensure_numeric(get("beh_t_ms", np.array([], dtype=np.float32)))
    aligned["beh_common_idx"]           = _ensure_numeric(get("beh_common_idx", np.array([], dtype=np.int64)))
    aligned["beh_common_valid"]         = _ensure_numeric(get("beh_common_valid", np.array([], dtype=bool)))

    # ---- Alignment meta (JSON -> struct; raw string also included) ----
    align_meta = get("align_meta", None)
    aligned["align_meta_raw"]    = "" if align_meta is None else (align_meta.item() if hasattr(align_meta, "item") else align_meta)
    aligned["align_meta_struct"] = _parse_align_meta(aligned["align_meta_raw"])

    # Convenience: expose common time as 't_ms_common'
    # (same as intan_t_ms_aligned / ua_t_ms_aligned)
    aligned["t_ms_common"] = aligned["intan_t_ms_aligned"]

    return {"aligned": aligned}


def _approx_bytes(x) -> int:
    """Rough size estimate to detect 2GB MAT v7 limit."""
    try:
        if isinstance(x, np.ndarray):
            return int(x.nbytes)
        if isinstance(x, dict):
            return sum(_approx_bytes(v) for v in x.values())
        if isinstance(x, (str, bytes, bytearray)):
            return len(x)
        return 0
    except Exception:
        return 0


def write_mat_v7(out_path: Path, data: dict):
    """Write using scipy.io.savemat (MAT v7)."""
    # MATLAB likes Fortran order for big 2D arrays; not strictly required
    def _f_orderize(d):
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray) and v.ndim >= 2 and v.flags.c_contiguous:
                d[k] = np.asfortranarray(v)
            elif isinstance(v, dict):
                _f_orderize(v)
        return d
    payload = _f_orderize({k: v for k, v in data.items()})
    savemat(out_path.as_posix(), payload, do_compression=True)


def write_mat_v73(out_path: Path, data: dict):
    """Minimal v7.3 writer via h5py (MATLAB can h5read)."""
    import h5py
    with h5py.File(out_path, "w") as f:
        grp = f.create_group("aligned")
        def write_any(g, key, val):
            if isinstance(val, dict):
                sub = g.create_group(key)
                for kk, vv in val.items():
                    write_any(sub, kk, vv)
            elif isinstance(val, np.ndarray):
                # strings/cellstr: store as variable-length UTF-8
                if val.dtype == object:
                    # convert to list of utf-8 strings
                    lst = [("" if x is None else str(x)) for x in val.tolist()]
                    dt = h5py.string_dtype(encoding="utf-8")
                    g.create_dataset(key, data=np.array(lst, dtype=dt), dtype=dt, compression="gzip")
                else:
                    g.create_dataset(key, data=val, compression="gzip")
            elif isinstance(val, (str, bytes, bytearray)):
                dt = h5py.string_dtype(encoding="utf-8")
                g.create_dataset(key, data=str(val), dtype=dt)
            elif isinstance(val, (int, float, bool)):
                g.create_dataset(key, data=val)
            else:
                # fallback to string
                dt = h5py.string_dtype(encoding="utf-8")
                g.create_dataset(key, data=str(val), dtype=dt)
        # write fields
        for k, v in data["aligned"].items():
            write_any(grp, k, v)


def convert_one(npz_path: Path, out_mat: Path, prefer_v73_if_large=True):
    with np.load(npz_path, allow_pickle=True) as z:
        data = _build_mat_struct(z)

    total_bytes = _approx_bytes(data)
    # Heuristic: if any single array likely exceeds ~1.9 GB, prefer v7.3
    big = total_bytes > 1_900_000_000

    if big and prefer_v73_if_large and ENABLE_HDF5_FALLBACK:
        print(f"[info] {npz_path.name}: payload ≈ {total_bytes/1e9:.2f} GB → writing v7.3 HDF5")
        write_mat_v73(out_mat, data)
    else:
        try:
            write_mat_v7(out_mat, data)
        except Exception as e:
            if ENABLE_HDF5_FALLBACK:
                print(f"[warn] v7 write failed for {npz_path.name}: {e}\n       → trying v7.3 HDF5")
                write_mat_v73(out_mat, data)
            else:
                raise


def main():
    if len(sys.argv) < 2:
        print("Usage: python npz_to_mat_aligned.py /path/to/Aligned")
        sys.exit(2)

    root = Path(sys.argv[1]).resolve()
    if not root.exists():
        print(f"[error] folder not found: {root}")
        sys.exit(2)

    npzs = sorted(root.glob("aligned__*.npz"))
    if not npzs:
        print(f"[warn] no files matched aligned__*.npz under {root}")
        sys.exit(0)

    out_root = root  # write next to inputs
    for p in npzs:
        out_mat = out_root / (p.stem + ".mat")
        try:
            convert_one(p, out_mat)
            print(f"[write] {out_mat}")
        except Exception as e:
            print(f"[error] failed for {p.name}: {e}")

if __name__ == "__main__":
    main()
