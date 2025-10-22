from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np
import json, csv, re, io, codecs
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from typing import Dict, Any, Optional, Tuple
from ..functions.intan_preproc import load_stim_geometry, make_identity_probe_from_geom

# ---------- Helper functions ----------
def _norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lookup = {_norm(str(c)): c for c in df.columns}
    for name in candidates:
        key = _norm(name)
        if key in lookup:
            return lookup[key]
    raise KeyError(f"Missing any of {candidates}; sample cols: {list(df.columns)[:8]}")

# Behavior CSV helpers
def _read_csv_text_fallback(p: Path) -> str:
    raw = p.read_bytes()
    try_order = []
    if raw.startswith(codecs.BOM_UTF8):       try_order = ["utf-8-sig"]
    elif raw.startswith(codecs.BOM_UTF16_LE): try_order = ["utf-16-le"]
    elif raw.startswith(codecs.BOM_UTF16_BE): try_order = ["utf-16-be"]
    try_order += ["utf-8", "cp1252", "latin1", "utf-16"]
    for enc in try_order:
        try: return raw.decode(enc)
        except UnicodeDecodeError: continue
    return raw.decode("latin1", errors="replace")

def _csv_dictreader(p: Path) -> csv.DictReader:
    text = _read_csv_text_fallback(p)
    rdr = csv.DictReader(io.StringIO(text))
    if not rdr.fieldnames:
        raise ValueError(f"{p.name}: no header row")
    return rdr

def _build_int_map_from_csv(p: Path, left_cols: tuple[str, ...], right_cols: tuple[str, ...]) -> dict[int,int]:
    rdr = _csv_dictreader(p)
    colmap = {_norm(c): c for c in rdr.fieldnames if c}
    def pick(names): 
        for n in names:
            k = _norm(n)
            if k in colmap: return colmap[k]
        raise KeyError(f"{p.name}: missing one of {names}; saw {rdr.fieldnames}")
    cL = pick(left_cols); cR = pick(right_cols)
    out: dict[int,int] = {}
    for row in rdr:
        try: out[int(float(str(row[cL]).strip()))] = int(float(str(row[cR]).strip()))
        except Exception: pass
    return out

def load_video_to_br_map(meta_csv: Path) -> dict[int,int]:
    return _build_int_map_from_csv(meta_csv, ("video_file","videofile","video"),
                                              ("br_file","brfile","br"))

def read_intan_to_br_map(csv_path: Path) -> dict[int,int]:
    return _build_int_map_from_csv(csv_path, ("intan_file","intanfile","intan","intanindex","intanfileindex"),
                                              ("br_file","brfile","br","brindex","brfileindex"))

_CAM_IN_NAME_RE = re.compile(r"Cam[-_]?([01])", re.IGNORECASE)

def _is_ns5_col(name: str) -> bool:
    # accept 'ns5_sample', 'ns5sample', and variants like 'ns5_sample_15'
    return re.fullmatch(r"(?:ns5[_ ]?sample)(?:_\d+)?", str(name).strip().lower()) is not None

def _flatten_cols_mi(mi: pd.MultiIndex) -> list[str]:
    """
    Flatten possibly-messy MultiIndex columns:
    - Drop 'nan' / 'Unnamed.*' levels
    - Join remaining parts with '_'
    - Normalize any ns5-sample-like column to 'ns5_sample'
    """
    flat = []
    for tup in mi.values:
        parts = [str(x) for x in tup if str(x).lower() not in ("nan", "none") and not str(x).startswith("Unnamed")]
        if not parts:
            parts = [""]
        name = "_".join(p.strip() for p in parts if p.strip())
        # Normalize ns5 column variants (e.g., "ns5_sample_15" -> "ns5_sample")
        if _is_ns5_col(name):
            name = "ns5_sample"
        flat.append(name)
    return flat

def _read_behavior_csv_robust(csv_path: Path, num_cam: int) -> tuple[pd.DataFrame, str]:
    """
    If num_cam > 1, read as a 2-level header (for both-cam CSVs).
    Else read as a single header (for single-cam CSVs).
    Always normalize any ns5* columns to 'ns5_sample'.
    """
    if num_cam not in (1, 2):
        raise ValueError("num_cam must be 1 or 2")

    if num_cam > 1:
        # Two-camera/both-cams format
        df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
        df.columns = _flatten_cols_mi(df.columns)  # uses your helper to clean MI and ns5 variants
    else:
        # Single-camera format
        df = pd.read_csv(csv_path, header=0, index_col=0)
        # normalize column names to strings then collapse ns5 variants
        cols = []
        for c in df.columns:
            name = str(c)
            if _is_ns5_col(name):  # e.g., 'ns5_sample_15' => 'ns5_sample'
                name = "ns5_sample"
            cols.append(name)
        df.columns = cols

    # Ensure we have a usable ns5 column and standardize its name
    ns5_col = next((c for c in ("ns5_sample", "cam0_ns5_sample", "cam1_ns5_sample") if c in df.columns), None)
    if ns5_col is None:
        # fallback: accept any column matching the pattern
        for c in df.columns:
            if _is_ns5_col(c):
                ns5_col = c
                break
    if ns5_col is None:
        raise ValueError(f"{csv_path.name} missing ns5_sample column")

    if ns5_col != "ns5_sample":
        df = df.rename(columns={ns5_col: "ns5_sample"})
        ns5_col = "ns5_sample"

    return df, ns5_col

def load_behavior_npz(csv_path: Path, num_cam: int):
    """
    Reads aligned behavior CSV and returns:
      ns5_sample: (N,) int64
      cam0: (N, D0) float32, cam0_cols: list[str]
      cam1: (N, D1) float32, cam1_cols: list[str]

    num_cam: 1 or 2. Function is tolerant to files that only include one cam even when num_cam=2.
    """
    if num_cam not in (1, 2):
        raise ValueError("num_cam must be 1 or 2")

    df, ns5_col = _read_behavior_csv_robust(csv_path, num_cam)

    # Final ns5 normalization (guards against odd headers)
    if ns5_col is None:
        ns5_col = next((c for c in ("ns5_sample", "cam0_ns5_sample", "cam1_ns5_sample") if c in df.columns), None)
    if ns5_col is None:
        raise ValueError(f"{csv_path.name} missing ns5_sample column")

    ns5_sample = pd.to_numeric(df[ns5_col], errors="coerce").to_numpy()
    ns5_sample = np.rint(ns5_sample).astype(np.int64)

    # If columns are already prefixed, use them directly.
    has_cam0_pref = any(c.startswith("cam0_") for c in df.columns)
    has_cam1_pref = any(c.startswith("cam1_") for c in df.columns)

    if has_cam0_pref or has_cam1_pref:
        cam0_cols = [c for c in df.columns if c.startswith("cam0_") and c != "cam0_ns5_sample"]
        cam1_cols = [c for c in df.columns if c.startswith("cam1_") and c != "cam1_ns5_sample"]

        cam0 = df[cam0_cols].to_numpy(dtype=np.float32) if cam0_cols else np.zeros((len(df), 0), np.float32)
        cam1 = df[cam1_cols].to_numpy(dtype=np.float32) if cam1_cols else np.zeros((len(df), 0), np.float32)

        if num_cam == 1 and cam0.shape[1] and cam1.shape[1]:
            print(f"[warn] {csv_path.name} contains cam0_ and cam1_ columns but num_cam=1; returning both.")
        return ns5_sample, cam0, cam0_cols, cam1, cam1_cols

    # Otherwise, assume single-cam style: everything except the ns5 col are this cam’s features.
    feat_cols = [c for c in df.columns if c != ns5_col]

    if num_cam == 2:
        # Try to infer cam from filename; if absent, assume cam0.
        m = _CAM_IN_NAME_RE.search(csv_path.name)
        inferred_cam = int(m.group(1)) if m else 0

        if inferred_cam == 0:
            cam0_cols = feat_cols
            cam1_cols = []
            cam0 = df[cam0_cols].to_numpy(dtype=np.float32) if cam0_cols else np.zeros((len(df), 0), np.float32)
            cam1 = np.zeros((len(df), 0), np.float32)
        else:
            cam0_cols = []
            cam1_cols = feat_cols
            cam0 = np.zeros((len(df), 0), np.float32)
            cam1 = df[cam1_cols].to_numpy(dtype=np.float32) if cam1_cols else np.zeros((len(df), 0), np.float32)

        return ns5_sample, cam0, cam0_cols, cam1, cam1_cols

    # num_cam == 1
    cam0_cols = feat_cols
    cam1_cols = []
    cam0 = df[cam0_cols].to_numpy(dtype=np.float32) if cam0_cols else np.zeros((len(df), 0), np.float32)
    cam1 = np.zeros((len(df), 0), np.float32)
    return ns5_sample, cam0, cam0_cols, cam1, cam1_cols

# OCR / DLC
def load_ocr_map(ocr_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(ocr_csv)
    col_avi = _find_col(df, ["AVI_framenum","avi_framenum","avi_frame","avi"])
    col_ocr = _find_col(df, ["OCR_framenum","ocr_framenum","ocr_frame","ocr"])
    col_cor = _find_col(df, ["CORRECTED_framenum","corrected_framenum","corrected_frame","corrected"])

    out = df[[col_avi, col_ocr, col_cor]].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    last = out.dropna(subset=[col_ocr, col_cor]).tail(1)
    if last.empty:
        raise ValueError(f"{ocr_csv} has no valid OCR rows.")
    try:
        last_ocr = int(np.rint(float(last[col_ocr].iloc[0])))
        last_cor = int(np.rint(float(last[col_cor].iloc[0])))
        if last_ocr != last_cor:
            print(f"[warn] {ocr_csv.name}: last OCR_framenum ({last_ocr}) != CORRECTED_framenum ({last_cor}); continuing.")
    except Exception:
        pass

    return out.rename(columns={col_avi:"AVI_framenum", col_ocr:"OCR_framenum", col_cor:"CORRECTED_framenum"})

def load_dlc(dlc_csv: Path) -> pd.DataFrame:
    """
    Load DLC CSV (3 header rows). If the first column is 'frame' (in any header
    level), use it as the AVI_framenum index; otherwise use 0..N-1.
    Columns are flattened and cast to numeric where possible.
    """
    df = pd.read_csv(dlc_csv, header=[0, 1, 2])

    # Detect a 'frame' first column (any header level says 'frame')
    first = df.columns[0]
    levels = [first] if not isinstance(first, tuple) else list(first)
    levels_lc = [str(x).strip().lower() for x in levels]

    if any(x == "frame" for x in levels_lc):
        frame_col = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        df = df.drop(columns=[df.columns[0]])
        idx = pd.Index(np.rint(frame_col.values).astype("Int64"), name="AVI_framenum")
    else:
        # Drop DLC “scorer/index” style column if present
        first0 = str(levels[0]).lower()
        if first0 in ("scorer", "index", ""):
            df = df.drop(columns=[df.columns[0]])
        idx = pd.RangeIndex(len(df), name="AVI_framenum")

    # Flatten multi-index columns
    flat_cols = []
    for col in df.columns.values:
        if isinstance(col, tuple):
            parts = [str(x) for x in col if str(x) != "nan"]
        else:
            parts = [str(col)]
        flat_cols.append("_".join(parts).strip())
    df.columns = flat_cols

    # Numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.index = idx
    return df

def expand_dlc_to_corrected(dlc_df: pd.DataFrame, ocr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map DLC rows (indexed by AVI_framenum) into a DataFrame indexed by
    CORRECTED_framenum. Dropped frames become NaNs.
    """
    # Round to nearest int when constructing mapping; NaNs are dropped
    map_df = ocr_df.dropna(subset=["AVI_framenum", "CORRECTED_framenum"]).copy()
    map_df["AVI_framenum"] = np.rint(map_df["AVI_framenum"].to_numpy()).astype(int)
    map_df["CORRECTED_framenum"] = np.rint(map_df["CORRECTED_framenum"].to_numpy()).astype(int)

    avi_to_corr = dict(zip(map_df["AVI_framenum"].values, map_df["CORRECTED_framenum"].values))

    max_corr = int(np.rint(ocr_df["CORRECTED_framenum"].dropna().max()))
    out = pd.DataFrame(
        index=pd.RangeIndex(0, max_corr + 1, name="CORRECTED_framenum"),
        columns=dlc_df.columns,
        dtype=float,
    )

    # place DLC rows at corrected positions
    # dlc_df.index is AVI_framenum (nullable int); coerce to ints where present
    for avi, row in dlc_df.iterrows():
        if pd.isna(avi):
            continue
        avi_int = int(np.rint(float(avi)))
        corr = avi_to_corr.get(avi_int)
        if corr is not None:
            out.loc[corr] = row.values

    return out

# ---------- Intan utils ----------
def list_intan_sessions(root: Path) -> list[Path]:
    root = Path(root)
    return sorted([p for p in root.iterdir() if p.is_dir()])

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

def save_recording(rec: si.BaseRecording, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec.save(folder=out_dir, overwrite=True)

# NPZ loaders
def load_rate_npz(npz_path: Path | str) -> Tuple[
    np.ndarray,          # rate_hz       (n_ch, n_bins)
    np.ndarray,          # t_ms          (n_bins,)
    Optional[Any],       # meta          (dict-like) or None
    Optional[np.ndarray],# pcs           or None
    Optional[np.ndarray],# peaks         or None
    Optional[np.ndarray],# peaks_t_ms    or None
    Optional[np.ndarray],# peaks_t_sample or None
    Optional[np.ndarray] # explained_var or None
]:
    """
    Load a rate NPZ produced by the pipeline.
    Returns an 8-tuple; optional fields are None if not present.

    Always returns:
      (rate_hz, t_ms, meta, pcs, peaks, peaks_t_ms, peaks_t_sample, explained_var)
    """
    z = np.load(npz_path, allow_pickle=True)

    rate_hz = z["rate_hz"]
    t_ms    = z["t_ms"]
    pcs   = z.get("pcs", None)
    peaks = z.get("peaks", None)
    peaks_t_ms     = z.get("peaks_t_ms", z.get("peak_t_ms", None))
    peaks_t_sample = z.get("peaks_t_sample", z.get("peak_sample", None))
    explained_var = z.get("explained_var", None)
    meta = z.get("meta", None)
    # some npz files store dicts as 0-d object arrays
    if hasattr(meta, "item"):
        try:
            meta = meta.item()
        except Exception:
            pass

    return rate_hz, t_ms, meta, pcs, peaks, peaks_t_ms, peaks_t_sample, explained_var

def load_combined_npz(p: Path):
    d = np.load(p, allow_pickle=True)
    # Intan
    i_rate = d["intan_rate_hz"]
    i_t    = d["intan_t_ms_aligned"] if "intan_t_ms_aligned" in d.files else d["intan_t_ms"]
    # UA
    u_rate = d["ua_rate_hz"]
    u_t    = d["ua_t_ms_aligned"] if "ua_t_ms_aligned" in d.files else d["ua_t_ms"]
    # stim (absolute Intan ms); we’ll align it below
    stim_ms = d["stim_ms"] if "stim_ms" in d.files else np.array([], dtype=float)
    # alignment meta (JSON)
    meta = json.loads(d["align_meta"].item()) if "align_meta" in d.files else {}
    return i_rate, i_t.astype(float), u_rate, u_t.astype(float), stim_ms.astype(float), meta

# ---- alignment utils ----
# Helper functions for ADC dataloading
def _npz_meta_dict(arr):
    if arr is None:
        return {}
    try:
        obj = arr.item() if hasattr(arr, "item") else arr
    except Exception:
        obj = arr
    if isinstance(obj, (bytes, str)):
        try:
            return json.loads(obj)
        except Exception:
            return {}
    return obj if isinstance(obj, dict) else {}

def _ensure_channels_first(arr2d: np.ndarray) -> np.ndarray:
    n0, n1 = arr2d.shape
    if n0 <= 512 and n1 > n0:
        return arr2d.T
    if n1 <= 512 and n0 > n1:
        return arr2d
    return arr2d.T if n0 <= n1 else arr2d

def _pick_fs_hz(meta, z):
    for k in ("fs_hz","fs","sampling_rate_hz","sampling_rate","sample_rate_hz","sample_rate"):
        if isinstance(meta, dict) and k in meta:
            try: return float(meta[k])
            except Exception: pass
    if z is not None and "t" in z.files:
        t = np.asarray(z["t"], dtype=float)
        dt = np.nanmedian(np.diff(t))
        if dt > 0:
            return 1.0/dt
    return None

def load_intan_adc(npz_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns (adc_triangle, adc_lock, fs) from Intan USB ADC bundle (.npz).
    If only a generic 2D matrix is present, returns channel 0 as 'triangle', 1 as 'lock'.
    """
    z = np.load(npz_path, allow_pickle=True, mmap_mode="r")
    meta = _npz_meta_dict(z["meta"]) if "meta" in z.files else {}
    fs = float(_pick_fs_hz(meta, z) or 30000.0)

    if any(k.startswith("chunk_") for k in z.files):
        keys = sorted(k for k in z.files if k.startswith("chunk_"))
        ch0, ch1 = [], []
        for k in keys:
            a = z[k]
            if a.ndim == 2 and a.shape[1] >= 2:
                ch0.append(a[:,0].astype(np.float32))
                ch1.append(a[:,1].astype(np.float32))
            elif a.ndim == 1:
                ch0.append(a.astype(np.float32))
        if not ch0:
            raise RuntimeError("No usable chunk_* arrays in ADC NPZ")
        tri = np.concatenate(ch0)
        lock = np.concatenate(ch1) if ch1 else tri * 0.0
        return tri, lock, fs

    if "board_adc_data" in z.files and getattr(z["board_adc_data"], "ndim", 0) == 2:
        A = z["board_adc_data"]
    else:
        A = None
        for k in z.files:
            a = z[k]
            if hasattr(a, "ndim") and a.ndim == 2:
                A = a; break
        if A is None:
            raise RuntimeError(f"No 2D array in {npz_path}")

    chxT = _ensure_channels_first(np.asarray(A))
    if chxT.shape[0] < 2:
        chxT = np.vstack([chxT, np.zeros_like(chxT)])
    return chxT[0].astype(np.float32), chxT[1].astype(np.float32), fs

def detect_rising_edges(sig: np.ndarray) -> np.ndarray:
    """Fast hysteresis threshold rising-edge detector."""
    if sig.size == 0:
        return np.array([], dtype=np.int64)
    ds = sig[:: max(1, sig.size // 20000)]
    lo = 0.5 * (ds.min() + ds.max())
    hi = 0.5 * (lo + ds.max())
    edges, state = [], False
    for i, v in enumerate(sig):
        if not state and v >= hi:
            edges.append(i); state = True
        elif state and v <= lo:
            state = False
    return np.asarray(edges, dtype=np.int64)

def load_br_sync(ns_path: Path, chan_id: int, stream_id: str | int | None = None):
    """
    Load a sync channel from a Blackrock NSx file via spikeinterface.
    If stream_id is None, let SI pick (works for ns2/ns5 etc.). For strict ns5 pass '5'.
    """
    rec = se.read_blackrock(str(ns_path), all_annotations=True, stream_id=str(stream_id) if stream_id else None)
    fs = float(rec.get_sampling_frequency())
    ids = list(map(str, rec.get_channel_ids()))
    target = str(chan_id)
    if target not in ids:
        raise KeyError(f"Channel id {chan_id} not in {ns_path.name}. Have: {ids[:8]} …")
    col = ids.index(target)
    sig = rec.get_traces(channel_ids=[target]).astype(np.float32).squeeze()
    return sig, fs

def corrected_to_time_ns5(n_corrected: int, ns_path: Path, sync_chan: int):
    """Return (ns5_samples, t_sec, fs) arrays of length n_corrected, using rising edges."""
    sig, fs = load_br_sync(ns_path, sync_chan, stream_id=5)
    edges = detect_rising_edges(sig)
    if edges.size < n_corrected:
        raise RuntimeError(f"NS5 rising edges ({edges.size}) fewer than corrected frames ({n_corrected}).")
    samples = edges[:n_corrected].astype(np.int64)
    t_sec = samples / float(fs)
    return samples, t_sec, fs

# Analysis / alignment
def extract_peristim_segments(
    rate_hz: np.ndarray,
    t_ms: np.ndarray,
    stim_ms: np.ndarray,
    win_ms=(-800.0, 1200.0),
    min_trials: int = 1,
):
    """
    Return segments shape (n_trials, n_ch, n_twin) and rel_time_ms shape (n_twin,).
    Skips triggers whose window falls outside t_ms.
    """
    t_min, t_max = float(t_ms[0]), float(t_ms[-1])
    seg_len_ms = win_ms[1] - win_ms[0]

    # Relative timebase for a *perfectly* aligned segment (used only for plotting/logic)
    # We will slice on t_ms for each trial, so segment lengths are equal if t_ms is uniform.
    # Assume uniform binning for rates:
    dt = float(np.nanmedian(np.diff(t_ms)))  # ms per bin
    n_twin = int(round(seg_len_ms / dt))
    rel_time_ms = np.arange(n_twin) * dt + win_ms[0]

    segments = []
    kept = 0
    for s in np.asarray(stim_ms, dtype=float):
        start_ms = s + win_ms[0]
        end_ms   = s + win_ms[1]
        if start_ms < t_min or end_ms > t_max:
            continue  # skip partial windows
        
        ## Stim alignment is wrong?

        # slice indices on t_ms
        i0 = int(np.searchsorted(t_ms, start_ms, side="left"))
        i1 = int(np.searchsorted(t_ms, end_ms,   side="left"))
        seg = rate_hz[:, i0:i1]  # (n_ch, n_twin)
        # Safety: ensure equal length (can happen if boundary falls between bins)
        if seg.shape[1] != n_twin:
            continue

        segments.append(seg)
        kept += 1

    if kept < min_trials:
        raise RuntimeError(f"Only {kept} peri-stim segments available (min_trials={min_trials}).")

    segments = np.stack(segments, axis=0)  # (n_trials, n_ch, n_twin)
    return segments, rel_time_ms

def baseline_zero_each_trial(
    segments: np.ndarray,
    rel_time_ms: np.ndarray,
    normalize_first_ms: float = 200.0,
):
    """
    For each trial & channel, subtract the mean over the first `normalize_first_ms`
    of the segment (i.e., from window start to start+normalize_first_ms).
    segments: (n_trials, n_ch, n_twin)
    """
    t0 = rel_time_ms[0]
    mask = (rel_time_ms >= t0) & (rel_time_ms < t0 + normalize_first_ms)
    if mask.sum() < 1:
        raise ValueError("Baseline window has 0 bins — check your timebase and dt.")
    base = np.nanmean(segments[:, :, mask], axis=2, keepdims=True)  # (n_trials, n_ch, 1)
    return segments - base # TODO This is subtracting baseline, can try using FRtrial = FR trial/(FR baseline + 1)

def median_across_trials(zeroed_segments: np.ndarray):
    """
    Find median across trials per channel.
    Input: (n_trials, n_ch, n_twin) -> returns (n_ch, n_twin)
    """
    return np.nanmedian(zeroed_segments, axis=0)

# Stim
def load_stim_detection(npz_path: Path) -> Dict[str, np.ndarray]:
    """
    Required fields:
      - trigger_pairs         : (n_pulses, 2) [start, end] in Intan index space
      - block_bounds_samples  : (n_blocks, 2) [start, end] in Intan index space
      - pulse_sizes           : (n_pulses,)
    Optional:
      - active_channels       : (n_active,)
    """
    with np.load(npz_path, allow_pickle=False) as z:
        required = ("trigger_pairs", "block_bounds_samples", "pulse_sizes")
        missing = [k for k in required if k not in z.files]
        if missing:
            raise KeyError(f"{npz_path} missing required keys: {missing}")

        active = z.get("active_channels")
        active_channels = (np.asarray(active, dtype=np.int32)
                           if active is not None else np.empty(0, dtype=np.int32))

        trigger_pairs = np.asarray(z["trigger_pairs"], dtype=np.int64)
        block_bounds_samples = np.asarray(z["block_bounds_samples"], dtype=np.int64)
        pulse_sizes = np.asarray(z["pulse_sizes"], dtype=np.int32)

    return {
        "active_channels": active_channels,
        "trigger_pairs": trigger_pairs,
        "block_bounds_samples": block_bounds_samples,
        "pulse_sizes": pulse_sizes,
    }

def detect_stim_channels_from_npz(
    stim_npz_path: Path,
    eps: float = 1e-12,
    min_edges: int = 1,
) -> np.ndarray:
    """
    Return GEOMETRY-ORDERED indices of stimulated channels by counting 0→nonzero
    rising edges over the full 'stim_traces' array stored in the NPZ.
    NaN-safe: ignores channels with no finite data.
    """
    stim_npz_path = Path(stim_npz_path)
    if not stim_npz_path.exists():
        return np.array([], dtype=int)

    with np.load(stim_npz_path, allow_pickle=False) as z:
        if "stim_traces" not in z:
            return np.array([], dtype=int)

        X = np.asarray(z["stim_traces"], dtype=float)  # (n_channels, n_samples)
        if X.ndim != 2 or X.shape[1] < 2:
            return np.array([], dtype=int)

        # keep only channels that have any finite data
        row_has_data = np.isfinite(X).any(axis=1)
        if not row_has_data.any():
            return np.array([], dtype=int)

        Xv = X[row_has_data]  # valid rows view

        # robust per-row threshold: midpoint between 5th & 95th percentiles
        with np.errstate(all="ignore", invalid="ignore"):
            p5  = np.nanpercentile(Xv, 5,  axis=1)
            p95 = np.nanpercentile(Xv, 95, axis=1)
        thr = 0.5 * (p5 + p95)

        # rising-edge count above per-row threshold
        above  = Xv > (thr[:, None] + eps)             # (ch_valid, T)
        rising = above[:, 1:] & (~above[:, :-1])       # transitions
        counts = rising.sum(axis=1)

        active_valid = np.flatnonzero(counts >= int(min_edges))

        # map back to original channel indices
        idx_map = np.flatnonzero(row_has_data)
        active = idx_map[active_valid]

        # geometry order is already respected if upstream wrote order='geometry'
        return np.unique(active).astype(int)
 
def build_probe_and_locs_from_geom(geom_path: Path, radius_um: float = 5.0):
    """Load your saved geometry -> ProbeInterface Probe + (n_ch,2) locs."""
    geom = load_stim_geometry(geom_path)                  # your project format
    probe = make_identity_probe_from_geom(geom, radius_um=radius_um)  # ProbeInterface Probe
    locs  = probe.contact_positions.astype(float)         # (n_ch, 2)
    return probe, locs

# Other func
def aligned_stim_ms(stim_ms_abs: np.ndarray, meta: dict) -> np.ndarray:
    """
    Convert absolute Intan stim times (ms) into the aligned timebase used in the
    combined file:
      intan_t_ms_aligned = intan_t_ms - t0_intan_ms
    where t0_intan_ms = anchor_sample * 1000 / fs_intan
    """
    if stim_ms_abs.size == 0:
        return stim_ms_abs
    if "anchor_ms" in meta:
        return stim_ms_abs - float(meta["anchor_ms"])
    # fallback if some files only had anchor_sample
    if "anchor_sample" in meta:
        fs_intan = float(meta.get("fs_intan", 30000.0))
        return stim_ms_abs - (float(meta["anchor_sample"]) * 1000.0 / fs_intan)
    return stim_ms_abs  # nothing to do

def ua_title_from_meta(meta: dict) -> str:
    """
    Build Utah array title from metadata.
    """
    if "br_idx" in meta and meta["br_idx"] is not None:
        return f"Blackrock / UA: NRR_RW_001_{int(meta['br_idx']):03d}"
    return "Utah/BR"

def parse_intan_session_dtkey(session: str) -> int:
    m = re.search(r"(\d{6})_(\d{6})$", session)
    if not m:
        return int("9"*12)
    return int(m.group(1) + m.group(2))

def build_session_index_map(intan_sessions: list[str]) -> tuple[dict[str,int], dict[int,str]]:
    ordered = sorted(intan_sessions, key=parse_intan_session_dtkey)
    return ({sess: i+1 for i, sess in enumerate(ordered)},
            {i+1: s for i, s in enumerate(ordered)})

# =============================================================================
# Small helpers
# =============================================================================

def _norm(s: str) -> str:
    """Lowercase and strip non-alphanumerics; useful for tolerant CSV header matching."""
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def _rglob_many(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    """rglob multiple patterns and return a flat list."""
    out: list[Path] = []
    for pat in patterns:
        out.extend(root.rglob(pat))
    return out

# =============================================================================
# Trial / Video index parsing
# =============================================================================

# Optional date block with flexible separators: _YYYY_MMDD_HHMMSS or _YYYY-MMDD-HHMMSS
_DATE_OPT = r'(?:_[0-9]{4}[_-][0-9]{4}[_-][0-9]{6})?'

# Extract the video index (second 3-digit block), e.g. 'NRR_RW001_002[_date]' -> 2
_TRIAL_VIDEO_IDX_RE = re.compile(
    rf'NRR_[A-Za-z]+(?:\d{{3}})_(\d{{3}}){_DATE_OPT}(?:\b|_)',
    re.IGNORECASE
)

def extract_video_idx_from_trial(trial: str) -> Optional[int]:
    """
    Returns the Video_File index from names like:
      NRR_RW001_002
      NRR_RW001_002_2025_0915_141059
    -> 2  (i.e., the second 3-digit block)
    """
    m = _TRIAL_VIDEO_IDX_RE.search(trial)
    if m:
        return int(m.group(1))
    # Fallback: last 3-digit token (dates are 4/4/6, so won’t collide)
    nums = re.findall(r'(?<!\d)(\d{3})(?!\d)', trial)
    return int(nums[-1]) if nums else None

# =============================================================================
# File-name parsing (OCR + DLC)
# =============================================================================

# Accept:
#   NRR_RW003_001[_date]_Cam-0_ocr.csv
#   NRR_RW003_001[_date]_Cam-0DLC_Resnet50_....csv
_TRIAL_CAM_ANY_RE = re.compile(
    rf'^(?P<trial>NRR_[A-Za-z]+[0-9]{{3}}_[0-9]{{3}}){_DATE_OPT}'
    r'_Cam-(?P<cam>[01])(?:(?P<dlc>DLC.*\.csv)|(?:_|-)?ocr\.csv)$',
    re.IGNORECASE
)

def parse_trial_cam_kind(p: Path) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Returns (trial, cam, kind) for a given OCR/DLC CSV path.
    kind ∈ {'ocr','dlc'} or (None,None,None) if no match.
    """
    m = _TRIAL_CAM_ANY_RE.match(p.name)
    if not m:
        return None, None, None
    trial = m.group("trial")
    cam = int(m.group("cam"))
    kind = "dlc" if m.group("dlc") else "ocr"
    return trial, cam, kind

def find_per_trial_inputs(
    video_root: Path,
    *,
    require_both_cams_and_kinds: bool = False,
) -> Dict[str, dict]:
    """
    Scan VIDEO_ROOT (and VIDEO_ROOT/DLC if present) for OCR/DLC files and keep newest by mtime.

    Returns:
      {
        trial: {
          'ocr': { 0: Path, 1: Path, ... },
          'dlc': { 0: Path, 1: Path, ... },
        }
      }
    """
    per: Dict[str, dict] = {}
    patterns = ("NRR_*_Cam-[01]*.csv",)

    # Collect candidate files from both the root and DLC subdir (if present).
    search_roots = [video_root]
    dlc_dir = video_root / "DLC"
    if dlc_dir.exists():
        search_roots.append(dlc_dir)

    for root in search_roots:
        try:
            for p in _rglob_many(root, patterns):
                trial, cam, kind = parse_trial_cam_kind(p)
                if trial is None:
                    continue
                d = per.setdefault(trial, {"ocr": {}, "dlc": {}})
                prev = d[kind].get(cam)
                if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
                    d[kind][cam] = p
        except FileNotFoundError:
            # If a search root doesn't exist, just skip it.
            continue

    if not require_both_cams_and_kinds:
        return per

    # Keep only trials that have BOTH cams (0 & 1) for BOTH OCR and DLC.
    return {
        t: d for t, d in per.items()
        if set(d["ocr"].keys()) >= {0, 1} and set(d["dlc"].keys()) >= {0, 1}
    }

# =============================================================================
# CSV: Video_File → BR_File
# =============================================================================

def load_video_to_br_map(meta_csv: Path) -> dict[int, int]:
    """
    Returns { video_idx -> br_idx } from METADATA_CSV.
    Tolerates common encodings and small header-name variations.
    """
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1", "mac_roman")
    last_err: Optional[Exception] = None

    for enc in encodings:
        try:
            with meta_csv.open("r", newline="", encoding=enc, errors="strict") as f:
                rdr = csv.DictReader(f)
                cols = { _norm(c): c for c in (rdr.fieldnames or []) }

                def col(*cands: str) -> str:
                    for c in cands:
                        k = _norm(c)
                        if k in cols:
                            return cols[k]
                    raise KeyError(
                        f"Missing any of columns {cands}; "
                        f"found={list(cols.values()) or '[] (no header?)'}"
                    )

                c_video = col("video_file", "videofile", "video")
                c_br    = col("br_file", "brfile", "br")

                out: dict[int, int] = {}
                for row in rdr:
                    v_raw = str(row.get(c_video, "")).strip()
                    b_raw = str(row.get(c_br, "")).strip()
                    if not v_raw or not b_raw:
                        continue
                    try:
                        out[int(float(v_raw))] = int(float(b_raw))  # robust to "001", "1.0"
                    except Exception:
                        continue

                if out and enc != "utf-8":
                    print(f"[info] Read {meta_csv.name} with encoding={enc}.")
                if not out:
                    print(f"[warn] No Video_File→BR_File pairs found in {meta_csv.name}.")
                return out

        except UnicodeDecodeError as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Could not decode {meta_csv} with tried encodings; last error: {last_err}"
    )

# =============================================================================
# Blackrock (.ns5/.ns6) locating
# =============================================================================

def _nsx_token_score(p: Path, token: str) -> tuple[int, float, int]:
    """Score NSx files by (strong token match, mtime, size)."""
    strong = int(bool(re.search(rf"(?:^|[^0-9]){re.escape(token)}(?:[^0-9]|$)", p.stem)))
    st = p.stat()
    return (strong, st.st_mtime, st.st_size)

def find_nsx_by_br_index(
    br_root: Path,
    br_idx: int,
    exts: tuple[str, ...] = ("*.ns5", "*.ns6"),
) -> Optional[Path]:
    """
    Locate NSx by BR index (3-digit), preferring strong token match, newest mtime, then size.
    """
    hits = _rglob_many(br_root, exts)
    if not hits:
        return None
    token = f"{br_idx:03d}"
    hits.sort(key=lambda p: _nsx_token_score(p, token), reverse=True)
    best = hits[0]
    if token not in best.stem:
        print(f"[warn] No obvious NSx name match for BR {token}; using {best.name}")
    return best

def find_nsx_for_trial(
    br_root: Path,
    trial: str,
    exts: tuple[str, ...] = ("*.ns5", "*.ns6"),
) -> Optional[Path]:
    """
    Locate an NSx whose stem contains the trial id. Prefer newest mtime, then size.
    """
    trial_low = trial.lower()
    hits = [p for p in _rglob_many(br_root, exts) if trial_low in p.stem.lower()]
    if not hits:
        return None
    hits.sort(key=lambda p: (p.stat().st_mtime, p.stat().st_size), reverse=True)
    return hits[0]

# Explicit ns5-only wrappers (handy when you must avoid .ns6)
def find_ns5_by_br_index(br_root: Path, br_idx: int) -> Optional[Path]:
    return find_nsx_by_br_index(br_root, br_idx, exts=("*.ns5",))

def find_ns5_for_trial(br_root: Path, trial: str) -> Optional[Path]:
    return find_nsx_for_trial(br_root, trial, exts=("*.ns5",))
