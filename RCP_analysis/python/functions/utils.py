from pathlib import Path
import numpy as np
import pandas as pd
import json, csv, re, io, codecs
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from typing import Dict
from spikeinterface.extractors import read_blackrock
from ..functions.intan_preproc import load_stim_geometry, make_identity_probe_from_geom

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

def save_recording(rec: si.BaseRecording, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec.save(folder=out_dir, overwrite=True)

def load_rate_npz(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    rate_hz = d["rate_hz"]           # (n_ch, n_bins_total)
    t_ms    = d["t_ms"]              # (n_bins_total,)
    meta    = d.get("meta", None)
    return rate_hz, t_ms, (meta.item() if hasattr(meta, "item") else meta)

def median_across_trials(zeroed_segments: np.ndarray):
    """
    Find median across trials per channel.
    Input: (n_trials, n_ch, n_twin) -> returns (n_ch, n_twin)
    """
    return np.nanmedian(zeroed_segments, axis=0)

# ---- alignment utils ----
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
    dt = float(np.median(np.diff(t_ms)))  # ms per bin
    n_twin = int(round(seg_len_ms / dt))
    rel_time_ms = np.arange(n_twin) * dt + win_ms[0]

    segments = []
    kept = 0
    for s in np.asarray(stim_ms, dtype=float):
        start_ms = s + win_ms[0]
        end_ms   = s + win_ms[1]
        if start_ms < t_min or end_ms > t_max:
            continue  # skip partial windows

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

def baseline_zero_each_trial(
    segments: np.ndarray,
    rel_time_ms: np.ndarray,
    baseline_first_ms: float = 200.0,
):
    """
    For each trial & channel, subtract the mean over the first `baseline_first_ms`
    of the segment (i.e., from window start to start+baseline_first_ms).
    segments: (n_trials, n_ch, n_twin)
    """
    t0 = rel_time_ms[0]
    mask = (rel_time_ms >= t0) & (rel_time_ms < t0 + baseline_first_ms)
    if mask.sum() < 1:
        raise ValueError("Baseline window has 0 bins — check your timebase and dt.")
    base = segments[:, :, mask].mean(axis=2, keepdims=True)  # (n_trials, n_ch, 1)
    return segments - base # TODO This is subtracting baseline, can try using FRtrial = FR trial/(FR baseline + 1)

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
        dt = np.median(np.diff(t))
        if dt > 0:
            return 1.0/dt
    return None

# ===========================
# Data loading (ADC + BR sync)
# ===========================

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

def load_br_intan_sync_ns5(ns5_path: Path, intan_sync_chan_id: int = 134) -> tuple[np.ndarray, float]:
    """Load BR intan_sync (e.g., ch134) from .ns5 via spikeinterface."""
    rec = read_blackrock(ns5_path, stream_id="5")  # nsx5
    fs = float(rec.get_sampling_frequency())
    ids = np.array(rec.get_channel_ids()).astype(str)
    if str(intan_sync_chan_id) not in ids:
        raise KeyError(f"Channel id {intan_sync_chan_id} not in {ns5_path.name} (have: {ids.tolist()})")
    col = int(np.where(ids == str(intan_sync_chan_id))[0][0])
    S = rec.get_traces(0, 0, rec.get_num_frames(0)).astype(np.float32)  # (T, n_ch)
    return S[:, col].ravel(), fs

# ===========================
# Behavior CSV helpers
# ===========================

def _flatten_cols_mi(cols) -> list[str]:
    out = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [str(x) for x in c if str(x) not in ("nan", "")]
            out.append("_".join(parts))
        else:
            out.append(str(c))
    return out

def load_behavior_npz(csv_path: Path):
    """
    Reads the 'both_cams_aligned.csv' produced earlier and returns:
      ns5_sample: (N,) int64
      cam0:       (N, D0) float32, cam0 column names list[str]
      cam1:       (N, D1) float32, cam1 column names list[str]
    No anchor shift is applied (already in ns5 samples).
    """
    # It was written with MultiIndex columns; read robustly
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
    df.columns = _flatten_cols_mi(df.columns)

    # Prefer shared ns5_sample; else fall back to per-cam if present
    ns5_col = None
    for cand in ("ns5_sample", "cam0_ns5_sample", "cam1_ns5_sample"):
        if cand in df.columns:
            ns5_col = cand
            break
    if ns5_col is None:
        raise ValueError(f"{csv_path.name} missing ns5_sample column")

    ns5_sample = pd.to_numeric(df[ns5_col], errors="coerce").to_numpy()
    ns5_sample = np.rint(ns5_sample).astype(np.int64)

    # Collect cam0 / cam1 feature columns (exclude any *_ns5_sample)
    cam0_cols = [c for c in df.columns if c.startswith("cam0_") and c != "cam0_ns5_sample"]
    cam1_cols = [c for c in df.columns if c.startswith("cam1_") and c != "cam1_ns5_sample"]

    cam0 = df[cam0_cols].to_numpy(dtype=np.float32) if cam0_cols else np.zeros((len(df), 0), np.float32)
    cam1 = df[cam1_cols].to_numpy(dtype=np.float32) if cam1_cols else np.zeros((len(df), 0), np.float32)

    return ns5_sample, cam0, cam0_cols, cam1, cam1_cols



# ===========================
# CSV mapping helpers
# ===========================

def read_intan_to_br_map(csv_path: Path) -> dict[int, int]:
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())
    raw = csv_path.read_bytes()
    try_order = []
    if raw.startswith(codecs.BOM_UTF8):      try_order = ["utf-8-sig"]
    elif raw.startswith(codecs.BOM_UTF16_LE): try_order = ["utf-16-le"]
    elif raw.startswith(codecs.BOM_UTF16_BE): try_order = ["utf-16-be"]
    try_order += ["utf-8", "cp1252", "latin1", "utf-16"]
    text = None
    for enc in try_order:
        try:
            text = raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("latin1", errors="replace")
    rdr = csv.DictReader(io.StringIO(text))
    if not rdr.fieldnames:
        raise ValueError("CSV has no header row")
    fmap = {norm(k): k for k in rdr.fieldnames if k}

    def pick(*names):
        for n in names:
            if n in fmap: return fmap[n]
        raise KeyError(f"CSV missing one of: {names}")

    col_intan = pick("intan_file","intanfile","intan","intanindex","intanfileindex")
    col_br    = pick("br_file","brfile","br","brindex","brfileindex")

    out = {}
    for row in rdr:
        try:
            out[int(str(row[col_intan]).strip())] = int(str(row[col_br]).strip())
        except Exception:
            pass
    if not out:
        raise ValueError("No (Intan, BR) rows parsed")
    return out

def parse_intan_session_dtkey(session: str) -> int:
    m = re.search(r"(\d{6})_(\d{6})$", session)
    if not m:
        return int("9"*12)
    return int(m.group(1) + m.group(2))

def build_session_index_map(intan_sessions: list[str]) -> tuple[dict[str,int], dict[int,str]]:
    ordered = sorted(intan_sessions, key=parse_intan_session_dtkey)
    return ({sess: i+1 for i, sess in enumerate(ordered)},
            {i+1: s for i, s in enumerate(ordered)})
