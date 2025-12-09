from pathlib import Path
import pandas as pd, numpy as np
import json, csv, re
import spikeinterface as si
import spikeinterface.extractors as se
from typing import Dict, Any, Optional, Tuple
from scipy.signal import butter, filtfilt

# --- Building OCR and DLC dictionary for mapping ---
def _parse_condition_cam(path: Path):
    """
    Take in names such as:
      NRR_RW007_001_[date]_Cam-0_ocr.csv
      NRR_RW007_001_[date]_Cam-1DLC_Resnet50_....csv
    and outputs the condition ID, camera #, and DLC or OCR file
    """
    name = path.name
    if "_Cam-" not in name: # If not from a camera, ignore files
        return None, None

    cond_id, after = name.split("_Cam-", 1) # Get condition ID
    if not after:
        return None, None

    cam_char = after[0] # Get camera
    if cam_char not in "0123456789": # won't have more than 10 cameras I guess
        return None, None
    return cond_id, int(cam_char)

def find_per_cond_inputs(video_root: Path) -> Dict[str, dict]:
    """
    Scan VIDEO_ROOT/DLC for OCR/DLC files and keep newest file.
    Return conditions that have both cams for both 'ocr' and 'dlc'
    """
    per: Dict[str, dict] = {}  # cond -> {'ocr': {cam: Path}, 'dlc': {cam: Path}}
    for subdir, csv_type in (("DLC", "dlc"), ("OCR", "ocr")):
        root = video_root / subdir
        if not root.is_dir():
            continue
        for path in root.rglob("NRR_*_Cam-*.csv"):
            cond_id, cam_num = _parse_condition_cam(path)
            if not cond_id or cam_num is None:
                continue
            bucket = per.setdefault(cond_id, {"ocr": {}, "dlc": {}})
            prev = bucket[csv_type].get(cam_num)
            if prev is None or path.stat().st_mtime > prev.stat().st_mtime:
                bucket[csv_type][cam_num] = path

    required = {0, 1}
    return {
        cond: d for cond, d in per.items()
        if set(d["ocr"]) >= required and set(d["dlc"]) >= required
    }

def get_metadata_mapping(meta_csv: Path, field1: str, field2: str) -> Dict[int, int]:
    """
    Gets field1 → field2 mapping using the metadata.

    Skips rows where:
    - Either field is missing
    - Either field can't be converted to int (e.g. empty, '-', 'NA', etc.)
    """
    with meta_csv.open(encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            raise ValueError(f"{meta_csv.name}: missing header row")

        mapping: Dict[int, int] = {}
        for row in rdr:
            try:
                k = int((row[field1] or "").strip())
                v = int((row[field2] or "").strip())
            except (KeyError, TypeError, ValueError):
                # missing field, None, non-int, '-', etc.
                continue
            mapping[k] = v

        return mapping

# OCR / DLC
def load_ocr_map(ocr_csv: Path) -> pd.DataFrame:
    """
    Load OCR file
    """
    df = pd.read_csv(ocr_csv)
    col_avi = "AVI_framenum"
    col_ocr = "OCR_framenum"
    col_cor = "CORRECTED_framenum"

    out = df[[col_avi, col_ocr, col_cor]].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    last = out.dropna().tail(1)
    
    if last.empty:
        raise ValueError(f"{ocr_csv} has no valid OCR rows.")
    last_ocr = int(last[col_ocr].iloc[0])
    last_cor = int(last[col_cor].iloc[0])
    if last_ocr != last_cor:
        print(f"[warn] {ocr_csv.name}: last OCR_framenum ({last_ocr}) != CORRECTED_framenum ({last_cor}); continuing.")
        
    return out

def load_dlc(dlc_csv: Path) -> pd.DataFrame:
    """
        Load DLC file. If the first column is 'frame' (in any header
        level), use it as the AVI_framenum index; otherwise use 0..N-1.
        Columns are flattened and cast to numeric where possible. TODO
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

def align_dlc_to_corrected(dlc_df: pd.DataFrame, ocr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map DLC rows (indexed by AVI_framenum) into a DataFrame indexed by
    CORRECTED_framenum. Dropped frames become NaNs. TODO
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

def frame2sample_br_ns5_sync(
    n_corrected: int,
    ns_path: Path,
    sync_chan: int,
):
    """
    Read Blackrock NS5, detect rising edges on,
    and return sample indices corresponding to the frames.

    Returns:
    samples: (n_corrected,) int32
    """
    # Load sync channel
    rec = se.read_blackrock(str(ns_path), stream_id="5")
    ch_ids = list(rec.get_channel_ids())  # typically ints
    if sync_chan not in ch_ids:
        raise KeyError(f"Channel id {sync_chan} not found in {ns_path.name}. First few: {ch_ids[:8]}")
    sig = rec.get_traces(channel_ids=[sync_chan]).astype(np.float32).squeeze()

    # --- rising-edge detection with hysteresis
    if sig.size == 0:
        raise RuntimeError("Empty sync signal.")
    # downsampled stats for thresholds
    step = max(1, sig.size // 20000)
    ds = sig[::step]
    lo = 0.5 * (ds.min() + ds.max())
    hi = 0.5 * (lo + ds.max())

    edges = []
    high = False
    for i, v in enumerate(sig):
        if not high and v >= hi:
            edges.append(i)
            high = True
        elif high and v <= lo:
            high = False

    edges = np.asarray(edges, dtype=np.int32)
    if edges.size < n_corrected:
        raise RuntimeError(
            f"NS5 rising edges ({edges.size}) fewer than corrected frames ({n_corrected})."
        )

    samples = edges[:n_corrected]
    return samples

def frame2sample_br_ns2_sync(
    n_corrected: int,
    ns_path: Path,
    sync_chan: int,
):
    """
    Read Blackrock NS2, detect rising edges on `sync_chan`,
    and return sample indices corresponding to the frames.

    Parameters
    ----------
    n_corrected : int
        Number of corrected frames (length of the desired mapping).
    ns_path : Path
        Path to the .ns2 file.
    sync_chan : int
        Channel id in the NS2 file carrying the VOG sync signal.

    Returns
    -------
    samples : (n_corrected,) int32
        Sample indices of the rising edges, one per corrected frame.
    """
    # Load sync channel from NS2 (LFP) stream
    rec = se.read_blackrock(str(ns_path), stream_name="nsx2", all_annotations=True)

    ch_ids = list(rec.get_channel_ids())  # typically ints
    if sync_chan not in ch_ids:
        raise KeyError(
            f"Channel id {sync_chan} not found in {ns_path.name}. "
            f"First few: {ch_ids[:8]}"
        )

    sig = rec.get_traces(channel_ids=[sync_chan]).astype(np.float32).squeeze()

    # --- rising-edge detection with hysteresis (same logic as ns5 version) ---
    if sig.size == 0:
        raise RuntimeError("Empty sync signal.")

    # downsampled stats for thresholds
    step = max(1, sig.size // 20000)
    ds = sig[::step]
    lo = 0.5 * (ds.min() + ds.max())
    hi = 0.5 * (lo + ds.max())

    edges = []
    high = False
    for i, v in enumerate(sig):
        if not high and v >= hi:
            edges.append(i)
            high = True
        elif high and v <= lo:
            high = False

    edges = np.asarray(edges, dtype=np.int32)
    if edges.size < n_corrected:
        raise RuntimeError(
            f"NS2 rising edges ({edges.size}) fewer than corrected frames ({n_corrected})."
        )

    samples = edges[:n_corrected]
    return samples

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

# ---------- Intan utils ----------
def list_intan_sessions(root: Path) -> list[Path]:
    root = Path(root)
    return sorted([p for p in root.iterdir() if p.is_dir()])

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
    i_rate = d["nprw_rate_hz"]
    i_t    = d["nprw_t_ms_aligned"] if "nprw_t_ms_aligned" in d.files else d["nprw_t_ms"]
    # UA
    u_rate = d["ua_rate_hz"]
    u_t    = d["ua_t_ms_aligned"] if "ua_t_ms_aligned" in d.files else d["ua_t_ms"]
    # stim (absolute Intan ms); we’ll align it below
    stim_ms = d["stim_ms"] if "stim_ms" in d.files else np.array([], dtype=float)
    # alignment meta (JSON)
    meta = json.loads(d["align_meta"].item()) if "align_meta" in d.files else {}
    return i_rate, i_t, u_rate, u_t, stim_ms, meta

def load_intan_aux(npz_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    aux_file = np.load(npz_path, allow_pickle=True, mmap_mode="r")

    # meta / fs
    if "meta" in aux_file.files:
        meta = json.loads(str(aux_file["meta"]))
    else:
        raise RuntimeError(f"No 'meta' in {npz_path}")
    
    if "aux_traces" not in aux_file.files:
        raise RuntimeError(f"No 'aux_traces' in {npz_path}")

    aux_signal = aux_file["aux_traces"] # (n_channels, n_samples)
    if aux_signal.ndim != 2:
        raise RuntimeError(f"'aux_traces' in {npz_path} is not 2D, got shape {aux_signal.shape}")
    if aux_signal.shape[0] < 2:
        aux_signal = np.vstack([aux_signal, np.zeros_like(aux_signal)])

    triangle_sync_signal  = aux_signal[0] # TODO Make these channels customizable?
    br_template_signal = aux_signal[1]
    return triangle_sync_signal, br_template_signal, float(meta["fs_hz"])

# Analysis / alignment
def load_shift_row_by_br_idx(metadata_path: Path, br_idx: int) -> Optional[dict]:
    """
    Read br_to_intan_shifts.csv and return the row for this br_idx.
    Expected columns include: session (Intan), br_idx, anchor_ms, fs_intan, (optionally anchor_sample, etc.)
    """
    shifts_csv = metadata_path
    if not shifts_csv.exists():
        return None
    with shifts_csv.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                if int(row.get("br_idx", "-1")) == int(br_idx):
                    return row
            except Exception:
                continue
    return None

def extract_peristim_segments(
    rate_hz: np.ndarray,
    t_ms: np.ndarray,
    stim_ms: np.ndarray,
    win_ms=(-800.0, 1200.0),
    min_trials: int = 1,
):
    """
    Return:
      segments      : (n_trials, n_ch, n_twin)
      rel_time_ms   : (n_twin,)
      stim_ms_valid : (n_trials,) subset of stim_ms that produced valid segments

    Skips triggers whose window falls outside t_ms or whose slice length
    doesn't match the expected window length.
    """
    t_min, t_max = float(t_ms[0]), float(t_ms[-1])
    seg_len_ms = win_ms[1] - win_ms[0]

    # Assume uniform binning for rates:
    dt = float(np.nanmedian(np.diff(t_ms)))  # ms per bin
    n_twin = int(round(seg_len_ms / dt))
    rel_time_ms = np.arange(n_twin) * dt + win_ms[0]

    segments = []
    stim_valid = []
    for s in np.asarray(stim_ms, dtype=float):
        start_ms = s + win_ms[0]
        end_ms   = s + win_ms[1]
        if start_ms < t_min or end_ms > t_max:
            # skip partial windows
            continue

        # slice indices on t_ms
        i0 = int(np.searchsorted(t_ms, start_ms, side="left"))
        i1 = int(np.searchsorted(t_ms, end_ms,   side="left"))
        seg = rate_hz[:, i0:i1]  # (n_ch, maybe n_twin)

        # Safety: ensure equal length (can happen if boundary falls between bins)
        if seg.shape[1] != n_twin:
            continue

        segments.append(seg)
        stim_valid.append(s)

    kept = len(segments)
    if kept < min_trials:
        raise RuntimeError(f"Only {kept} peri-stim segments available (min_trials={min_trials}).")

    segments = np.stack(segments, axis=0)  # (n_trials, n_ch, n_twin)
    stim_ms_valid = np.asarray(stim_valid, dtype=float)  # (n_trials,)
    return segments, rel_time_ms, stim_ms_valid

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

def median_across_trials(zeroed_segments: np.ndarray) -> np.ndarray:
    """
    Median across trials per channel×time, ignoring NaNs.
    If a (ch, t) bin is all-NaN across trials, the result stays NaN.
    Input:  (n_trials, n_ch, n_t)  ->  (n_ch, n_t)
    """
    z = np.asarray(zeroed_segments, dtype=float)
    med = np.ma.median(np.ma.masked_invalid(z), axis=0).filled(np.nan)
    return med

def variance_across_trials(zeroed_segments: np.ndarray) -> np.ndarray:
    """
    Variance across trials per channel×time, ignoring NaNs.
    If a (ch, t) bin is all-NaN across trials, the result stays NaN.
    Input:  (n_trials, n_ch, n_t)  ->  (n_ch, n_t)
    """
    z = np.asarray(zeroed_segments, dtype=float)
    var = np.ma.var(np.ma.masked_invalid(z), axis=0).filled(np.nan)
    return var

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

def stim_npz_path_from_br_idx(
    br_idx: int,
    mapping_csv: Path,        # CSV with Intan_File, BR_File
    nprw_aux_root: Path,
) -> Optional[Path]:
    """
    Given a BR_File index (br_idx), return the corresponding
    stim_stream.npz path for the matching Intan session, or None
    if anything cannot be resolved.
    """

    # Map BR_File to Intan_File using metadata
    if not mapping_csv.exists():
        print(f"[WARN] mapping CSV not found: {mapping_csv}")
        return None, None
    br_to_intan = get_metadata_mapping(mapping_csv, "BR_File", "Intan_File")
    if br_idx not in br_to_intan:
        print(f"[WARN] BR {br_idx:03d} not in mapping CSV.")
        return None, None
    intan_idx = br_to_intan[br_idx]

    # Get session name from Intan_streams folders
    stream_dirs = sorted(
        d for d in nprw_aux_root.glob("*_Intan_streams") if d.is_dir()
    )
    if not stream_dirs:
        print(f"[WARN] No *_Intan_streams dirs under {nprw_aux_root}")
        return None, None

    # Strip the suffix to get session names
    suffix = "_Intan_streams"
    sessions = [
        d.name[:-len(suffix)]
        for d in stream_dirs
        if d.name.endswith(suffix)
    ]

    # Intan_File is 1-based
    if not (1 <= intan_idx <= len(sessions)):
        print(f"[WARN] Intan_File {intan_idx} out of range for {len(sessions)} sessions.")
        return None, None

    session = sessions[intan_idx - 1]

    stim_npz_path = nprw_aux_root / f"{session}_Intan_streams" / "stim_stream.npz"
    return stim_npz_path, session

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
    
def find_ns5_by_br_index(br_root: Path, br_idx: int) -> Optional[Path]:
    """
    Locate an .ns5 file whose stem ends with the 3-digit BR index, e.g.:

        br_idx = 1   -> token '001' -> matches '*001.ns5'

    If multiple matches exist, prefer:
      1. Newest modification time
      2. Largest size
    """
    token = f"{br_idx:03d}"
    # Only files like '...001.ns5'
    hits = [p for p in br_root.rglob("*.ns5") if p.stem.endswith(token)]
    if not hits:
        print(f"[warn] No NS5 file ending with {token}.ns5 under {br_root}")
        return None

    hits.sort(key=lambda p: (p.stat().st_mtime, p.stat().st_size), reverse=True)
    return hits[0]

def find_ns2_by_br_index(br_root: Path, br_idx: int) -> Optional[Path]:
    """
    Locate an .ns2 file whose *stem* ends with the 3-digit BR index, e.g.:

        br_idx = 1   -> token '001' -> matches '*001.ns2'

    If multiple matches exist, prefer:
      1. Newest modification time
      2. Largest size
    """
    token = f"{br_idx:03d}"
    # Only files like '...001.ns2'
    hits = [p for p in br_root.rglob("*.ns2") if p.stem.endswith(token)]
    if not hits:
        print(f"[warn] No NS2 file ending with {token}.ns2 under {br_root}")
        return None

    hits.sort(key=lambda p: (p.stat().st_mtime, p.stat().st_size), reverse=True)
    return hits[0]

# 
def _butter_lowpass_ba(cutoff_hz: float, fs_hz: float, order: int = 3):
    nyq = 0.5 * fs_hz
    Wn = min(max(cutoff_hz / nyq, 1e-6), 0.999999)  # clamp into (0,1)
    return butter(order, Wn=Wn, btype="low")

def _filtfilt_nanaware(x: np.ndarray, b, a) -> np.ndarray:
    """
    Filter 1D array with NaNs:
    - keep a mask of NaN samples
    - fill with column median
    - filtfilt
    - put NaNs back
    """
    out = x.astype(float, copy=True)
    mask = ~np.isfinite(out)
    if mask.all():
        return out  # all NaNs → return as is
    fill_val = np.nanmedian(out)
    out[mask] = fill_val
    out = filtfilt(b, a, out, method="gust")
    out[mask] = np.nan
    return out

def _central_diff_ms(x: np.ndarray, dt_ms: float) -> np.ndarray:
    """
    Central difference derivative along time for 1D array.
    Respects NaNs: any side that is NaN → derivative set to NaN there.
    """
    T = x.size
    if T < 2:
        return np.full_like(x, np.nan, dtype=float)
    v = np.full_like(x, np.nan, dtype=float)
    # interior
    valid_mid = np.isfinite(x[2:]) & np.isfinite(x[:-2])
    v[1:-1][valid_mid] = (x[2:][valid_mid] - x[:-2][valid_mid]) / (2.0 * dt_ms)
    # edges (1-sided)
    if np.isfinite(x[1]) and np.isfinite(x[0]):
        v[0] = (x[1] - x[0]) / dt_ms
    if np.isfinite(x[-1]) and np.isfinite(x[-2]):
        v[-1] = (x[-1] - x[-2]) / dt_ms
    return v

def butter_lowpass_pos_and_vel(TxD, t_ms, cutoff_hz=10.0, order=3):
    if TxD is None or TxD.size == 0 or t_ms.size < 3:
        Z = np.zeros_like(TxD) if isinstance(TxD, np.ndarray) else np.zeros((0,0), float)
        return Z, Z, t_ms
    TxD = np.asarray(TxD, float); t_ms = np.asarray(t_ms, float)
    dt_ms = float(np.nanmedian(np.diff(t_ms)))
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        return TxD.astype(float), np.full_like(TxD, np.nan, float), t_ms
    fs_hz = 1000.0 / dt_ms
    b, a = _butter_lowpass_ba(cutoff_hz, fs_hz, order)

    pos_filt = np.empty_like(TxD, float)
    for d in range(TxD.shape[1] if TxD.ndim == 2 else 1):
        col = TxD[:, d] if TxD.ndim == 2 else TxD
        # filtfilt needs enough samples; if too few, skip filtering
        if col.size < max(3*max(len(a), len(b)), 9):
            xf = col.astype(float)
        else:
            xf = _filtfilt_nanaware(col, b, a)
        if TxD.ndim == 2:
            pos_filt[:, d] = xf
        else:
            pos_filt = xf
    vel = np.empty_like(pos_filt, float)
    for d in range(pos_filt.shape[1] if pos_filt.ndim == 2 else 1):
        col = pos_filt[:, d] if pos_filt.ndim == 2 else pos_filt
        vcol = _central_diff_ms(col, dt_ms)
        if pos_filt.ndim == 2:
            vel[:, d] = vcol
        else:
            vel = vcol
    return pos_filt, vel, t_ms

def butter_lowpass_pos_and_vel_3d(NKT, t_ms, cutoff_hz=10.0, order=3):
    if NKT is None or NKT.size == 0 or t_ms.size < 3:
        Z = np.asarray(NKT, float) if isinstance(NKT, np.ndarray) else np.zeros((0,0,0), float)
        return Z, Z, t_ms
    NKT = np.asarray(NKT, float)
    dt_ms = float(np.nanmedian(np.diff(t_ms)))
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        return NKT.astype(float), np.full_like(NKT, np.nan, float), t_ms
    fs_hz = 1000.0 / dt_ms
    b, a = _butter_lowpass_ba(cutoff_hz, fs_hz, order)
    N, K, T = NKT.shape
    pos_f = np.empty_like(NKT, float)
    vel   = np.empty_like(NKT, float)
    min_len = max(3*max(len(a), len(b)), 9)
    for i in range(N):
        for k in range(K):
            x = NKT[i, k, :]
            xf = x.astype(float) if x.size < min_len else _filtfilt_nanaware(x, b, a)
            pos_f[i, k, :] = xf
            vel[i, k, :]   = _central_diff_ms(xf, dt_ms)
    return pos_f, vel, t_ms
