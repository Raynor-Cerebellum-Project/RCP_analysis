#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import json, csv, re, io, codecs

# ---- spikeinterface for BR .ns5 (to record fs_br & durations) ----
try:
    from spikeinterface.extractors import read_blackrock
    _HAS_SI = True
except Exception:
    _HAS_SI = False

# ---- your params helpers ----
from RCP_analysis import load_experiment_params, resolve_output_root

# ===========================
# Small utilities
# ===========================

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

def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-12)

def xcorr_normalized(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xz, yz = _z(x), _z(y)
    c = np.correlate(xz, yz, mode="full")
    if c.size:
        c /= (np.max(np.abs(c)) + 1e-12)
    lags = np.arange(-(y.size - 1), x.size)
    return c, lags

def peak_lags(corr: np.ndarray, lags: np.ndarray, height: float = 0.95) -> np.ndarray:
    try:
        from scipy.signal import find_peaks
        idx, _ = find_peaks(corr, height=height)
        return lags[idx].astype(int)
    except Exception:
        mid = np.arange(1, corr.size - 1)
        mask = (corr[mid-1] < corr[mid]) & (corr[mid] >= corr[mid+1]) & (corr[mid] >= height)
        return lags[mid[mask]].astype(int)

# ===========================
# Data loading (ADC + BR sync)
# ===========================

def load_intan_adc_2ch(npz_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
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
                ch0.append(a[:,0].astype(np.float64))
                ch1.append(a[:,1].astype(np.float64))
            elif a.ndim == 1:
                ch0.append(a.astype(np.float64))
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
    return chxT[0].astype(np.float64), chxT[1].astype(np.float64), fs

def load_br_intan_sync_ns5(ns5_path: Path, intan_sync_chan_id: int = 134) -> tuple[np.ndarray, float]:
    """Load BR intan_sync (e.g., ch134) from .ns5 via spikeinterface."""
    if not _HAS_SI:
        raise ImportError("spikeinterface is required to read .ns5 (pip install spikeinterface[full])")
    rec = read_blackrock(ns5_path, stream_id="5")  # nsx5
    fs = float(rec.get_sampling_frequency())
    ids = np.array(rec.get_channel_ids()).astype(str)
    if str(intan_sync_chan_id) not in ids:
        raise KeyError(f"Channel id {intan_sync_chan_id} not in {ns5_path.name} (have: {ids.tolist()})")
    col = int(np.where(ids == str(intan_sync_chan_id))[0][0])
    S = rec.get_traces(0, 0, rec.get_num_frames(0)).astype(np.float32)  # (T, n_ch)
    return S[:, col].ravel(), fs

# ===========================
# Template matching on Intan (lock)
# ===========================

def load_template(template_mat_path: Path) -> np.ndarray:
    from scipy.io import loadmat
    m = loadmat(str(template_mat_path))
    if "template" not in m:
        raise KeyError("'template' not found in br_intan_align_template.mat")
    return np.asarray(m["template"], float).squeeze()

def find_locs_via_template(adc_lock: np.ndarray, template: np.ndarray, fs: float, peak=0.95) -> np.ndarray:
    corr, lags = xcorr_normalized(adc_lock, template)
    locs = peak_lags(corr, lags, height=peak)
    return locs

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

# ===========================
# MAIN
# ===========================

if __name__ == "__main__":
    # ---------- CONFIG ----------
    REPO_ROOT = Path(__file__).resolve().parents[1]
    PARAMS    = load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
    TEMPLATE  = REPO_ROOT / "config" / "br_intan_align_template.mat"
    OUT_BASE  = resolve_output_root(PARAMS)
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    METADATA_ROOT        = OUT_BASE.parent / "Blackrock"
    METADATA_ROOT        = OUT_BASE.parent / "Metadata"
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    NPRW_BUNDLES   = OUT_BASE / "bundles" / "NPRW"

    # checkpoints (for discovering sessions)
    nprw_ckpt_root = OUT_BASE / "checkpoints" / "NPRW"

    # Metadata CSV with Intan_File ↔ BR_File mapping
    METADATA_CSV = OUT_BASE.parent / "Metadata" / "NRR_RW001_metadata.csv"
    if not METADATA_CSV.exists():
        raise SystemExit(f"[error] mapping CSV not found: {METADATA_CSV}")

    # ---------- discover Intan sessions (from NPRW rates) ----------
    def session_from_rates_path(p: Path) -> str:
        stem = p.stem
        body = stem[len("rates__"):]
        return body.split("__bin", 1)[0]

    rate_files = sorted(nprw_ckpt_root.rglob("rates__*.npz"))
    sessions = sorted({ session_from_rates_path(p) for p in rate_files })
    if not sessions:
        raise SystemExit(f"[error] No NPRW rate files under {nprw_ckpt_root}")

    sess_to_intan_idx, intan_idx_to_sess = build_session_index_map(sessions)

    # ---------- mapping CSV ----------
    intan_to_br = read_intan_to_br_map(METADATA_CSV)
    print(f"[map] Loaded Intan→BR rows: {len(intan_to_br)}")

    template = load_template(TEMPLATE)

    # ---------- iterate pairs, compute anchor, write shifts CSV ----------
    summary_rows = []
    for intan_idx, br_idx in sorted(intan_to_br.items()):
        session = intan_idx_to_sess.get(intan_idx)
        if session is None:
            print(f"[warn] No session name for Intan_File={intan_idx} (skipping)")
            continue

        adc_npz = NPRW_BUNDLES / f"{session}_Intan_bundle" / "USB_board_ADC_input_channel.npz"
        if not adc_npz.exists():
            print(f"[warn] Intan ADC bundle missing: {adc_npz} (skip Intan {intan_idx} → BR {br_idx})")
            continue

        # Load Intan ADC
        try:
            adc_triangle, adc_lock, fs_intan = load_intan_adc_2ch(adc_npz)
        except Exception as e:
            print(f"[error] load_intan_adc_2ch failed for {adc_npz}: {e}")
            continue

        # Choose reference for audit (not used further here)
        rng_tri = float(np.percentile(adc_triangle, 99) - np.percentile(adc_triangle, 1))
        rng_loc = float(np.percentile(adc_lock,     99) - np.percentile(adc_lock,     1))
        ref_signal = "triangle" if rng_tri > 0.6 * rng_loc else "lock"

        # Template-match on LOCK to get block starts
        locs = find_locs_via_template(adc_lock, template, fs_intan, peak=0.95)
        print(f"[locs] Intan {intan_idx:03d} peaks ≥0.95: {locs.size} | first 10: {locs[:10].tolist() if locs.size else []}")
        if locs.size == 0:
            print(f"[warn] No template peaks found for {session}; skipping.")
            continue

        anchor = int(locs[0])
        anchor_sec = anchor / fs_intan
        anchor_ms  = anchor_sec * 1000.0

        # Optionally read BR (fs & durations)
        fs_br = float("nan")
        dur_intan_sec = len(adc_triangle) / fs_intan
        dur_br_sec = float("nan")

        br_ns5 = METADATA_ROOT / f"NRR_RW_001_{br_idx:03d}.ns5"
        if br_ns5.exists() and _HAS_SI:
            try:
                br_sync, fs_br = load_br_intan_sync_ns5(br_ns5, intan_sync_chan_id=134)
                dur_br_sec = len(br_sync) / fs_br
            except Exception as e:
                print(f"[note] BR read failed for {br_ns5}: {e}")

        # keep CSV summary row
        summary_rows.append(dict(
            session=session,
            intan_idx=intan_idx,
            br_idx=br_idx,
            adc_npz=str(adc_npz),
            br_ns5=str(br_ns5),
            fs_intan=float(fs_intan),
            fs_br=float(fs_br),
            anchor_sample=int(anchor),
            anchor_seconds=float(anchor_sec),
            anchor_ms=float(anchor_ms),
            dur_intan_sec=float(dur_intan_sec),
            dur_br_sec=float(dur_br_sec),
            ref_signal=ref_signal,
        ))

        print(f"[anchor] Intan {intan_idx:03d} ↔ BR {br_idx:03d} : anchor={anchor:+d} samp ({anchor_sec:+.6f} s)")

    # ---------- write alignment summary CSV ----------
    if summary_rows:
        out_csv = METADATA_ROOT / "br_to_intan_shifts.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[done] wrote shifts → {out_csv}")
    else:
        print("[done] no rows to write (no anchors).")
