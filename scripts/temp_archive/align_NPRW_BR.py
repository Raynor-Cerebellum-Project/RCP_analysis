#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import json, csv, re, io, codecs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- spikeinterface for BR .ns5 ----
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

def _trim_to_equal_length(i_rate, i_t, u_rate, u_t):
    """
    Trim Intan and BR/Utah time series to their overlapping time range,
    then truncate both to the same number of bins (min length).
    No interpolation; purely cropping.
    """
    # Guard: empty inputs
    if i_t.size == 0 or u_t.size == 0:
        return (i_rate[:, :0], i_t[:0], u_rate[:, :0], u_t[:0])

    # Overlap in aligned ms
    t0 = max(float(i_t[0]), float(u_t[0]))
    t1 = min(float(i_t[-1]), float(u_t[-1]))
    if not (t1 > t0):
        # No overlap — return empty
        return (i_rate[:, :0], i_t[:0], u_rate[:, :0], u_t[:0])

    # Crop each stream to overlap
    mi = (i_t >= t0) & (i_t <= t1)
    mu = (u_t >= t0) & (u_t <= t1)
    i_rate2, i_t2 = i_rate[:, mi], i_t[mi]
    u_rate2, u_t2 = u_rate[:, mu], u_t[mu]

    # Equalize length by truncating to min length (from the start of overlap)
    n = int(min(i_t2.size, u_t2.size))
    i_rate2, i_t2 = i_rate2[:, :n], i_t2[:n]
    u_rate2, u_t2 = u_rate2[:, :n], u_t2[:n]
    return i_rate2, i_t2, u_rate2, u_t2

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
        raise KeyError("'template' not found in template.mat")
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
    if raw.startswith(codecs.BOM_UTF8):
        try_order = ["utf-8-sig"]
    elif raw.startswith(codecs.BOM_UTF16_LE):
        try_order = ["utf-16-le"]
    elif raw.startswith(codecs.BOM_UTF16_BE):
        try_order = ["utf-16-be"]
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

# -------- correlation shift helpers --------

def best_shift_corr_z(a: np.ndarray, b: np.ndarray, intan_BR_diff: int,
                      search: int = 1000, win: int = 30000) -> tuple[int, float]:
    """Maximize z-score correlation between Intan window (a) and BR window (b shifted by n)."""
    i0 = max(0, intan_BR_diff - win)
    i1 = min(len(a), intan_BR_diff + win + 1)
    aw = a[i0:i1]
    L = aw.size
    if L < 10:
        return 0, float("nan")
    awz = _z(aw)
    best_n, best_corr = 0, -np.inf
    for n in range(-search, search + 1):
        j0, j1 = i0 + n, i1 + n
        if j0 < 0 or j1 > len(b):
            continue
        bwz = _z(b[j0:j1].astype(float))
        corr = float(np.dot(awz, bwz) / L)
        if corr > best_corr:
            best_corr, best_n = corr, n
    return best_n, best_corr

# ===========================
# Rates helpers (for combined NPZ)
# ===========================

def load_rate_npz(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    rate_hz = d["rate_hz"]           # (n_ch, n_bins)
    t_ms    = d["t_ms"]              # (n_bins,)
    meta    = d.get("meta", None)
    pcs     = d.get("pcs", None)
    explained_var = d.get("explained_var", None)
    return rate_hz, t_ms, meta, pcs, explained_var

def find_intan_rates_for_session(nprw_ckpt_root: Path, session: str) -> Path | None:
    cands = sorted(nprw_ckpt_root.glob(f"rates__{session}__*.npz"))
    return cands[0] if cands else None

def find_ua_rates_by_index(ua_ckpt_root: Path, br_idx: int) -> Path | None:
    patt = f"rates__NRR_RW_001_{br_idx:03d}__*.npz"
    cands = sorted(ua_ckpt_root.glob(patt))
    if not cands:
        return None
    pref = [p for p in cands if "__sigma25ms" in p.stem]
    return pref[0] if pref else cands[-1]

def try_load_stim_ms_from_intan_bundle(bundles_root: Path, session: str) -> np.ndarray | None:
    stim_npz = bundles_root / f"{session}_Intan_bundle" / "stim_stream.npz"
    if not stim_npz.exists():
        return None
    with np.load(stim_npz, allow_pickle=False) as z:
        if "block_bounds_samples" not in z or "meta" not in z:
            return None
        blocks = z["block_bounds_samples"].astype(np.int64)
        meta_json = z["meta"].item() if hasattr(z["meta"], "item") else z["meta"]
        meta = json.loads(meta_json)
        fs_hz = float(meta.get("fs_hz", 30000.0))
        onset_samples = blocks[:,0]
        return onset_samples.astype(float) * (1000.0 / fs_hz)

# ===========================
# MAIN
# ===========================

if __name__ == "__main__":
    # ---------- CONFIG ----------
    REPO_ROOT = Path(__file__).resolve().parents[1]
    PARAMS    = load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)
    TEMPLATE  = REPO_ROOT / "config" / "template.mat"
    OUT_BASE  = resolve_output_root(PARAMS)
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    BR_ROOT        = OUT_BASE.parent / "Blackrock"
    NPRW_BUNDLES   = OUT_BASE / "bundles" / "NPRW"
    PLOTS_ROOT     = OUT_BASE / "figures" / "align_BR_to_Intan"
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

    # checkpoints
    nprw_checkpoint_root = OUT_BASE / "checkpoints" / "NPRW"
    ua_checkpoint_root   = OUT_BASE / "checkpoints" / "UA"
    aligned_checkpoint   = OUT_BASE / "checkpoints" / "Aligned"
    aligned_checkpoint.mkdir(parents=True, exist_ok=True)

    # Metadata CSV with Intan_File ↔ BR_File mapping
    METADATA_CSV = OUT_BASE.parent / "Metadata" / "NRR_RW001_metadata.csv"
    if not METADATA_CSV.exists():
        raise SystemExit(f"[error] mapping CSV not found: {METADATA_CSV}")

    # ---------- discover Intan sessions (from NPRW rates) ----------
    def session_from_rates_path(p: Path) -> str:
        stem = p.stem
        body = stem[len("rates__"):]
        return body.split("__bin", 1)[0]

    rate_files = sorted(nprw_checkpoint_root.rglob("rates__*.npz"))
    sessions = sorted({ session_from_rates_path(p) for p in rate_files })
    if not sessions:
        raise SystemExit(f"[error] No NPRW rate files under {nprw_checkpoint_root}")

    sess_to_intan_idx, intan_idx_to_sess = build_session_index_map(sessions)

    # ---------- mapping CSV ----------
    intan_to_br = read_intan_to_br_map(METADATA_CSV)
    print(f"[map] Loaded Intan→BR rows: {len(intan_to_br)}")

    template = load_template(TEMPLATE)

    # ---------- iterate pairs, compute shifts, write combined NPZ ----------
    summary_rows = []
    for intan_idx, br_idx in sorted(intan_to_br.items()):
        session = intan_idx_to_sess.get(intan_idx)
        if session is None:
            print(f"[warn] No session name for Intan_File={intan_idx} (skipping)")
            continue

        adc_npz = NPRW_BUNDLES / f"{session}_Intan_bundle" / "USB_board_ADC_input_channel.npz"
        if not adc_npz.exists():
            print(f"[warn] Intan ADC bundle missing: {adc_npz} (skip pair Intan {intan_idx} → BR {br_idx})")
            continue

        # Load Intan ADC
        try:
            adc_triangle, adc_lock, fs_intan = load_intan_adc_2ch(adc_npz)
        except Exception as e:
            print(f"[error] load_intan_adc_2ch failed for {adc_npz}: {e}")
            continue

        # Choose reference for alignment
        rng_tri = float(np.percentile(adc_triangle, 99) - np.percentile(adc_triangle, 1))
        rng_loc = float(np.percentile(adc_lock,     99) - np.percentile(adc_lock,     1))
        ref_intan = adc_triangle if rng_tri > 0.6 * rng_loc else adc_lock

        # Template-match on LOCK to get block starts
        locs = find_locs_via_template(adc_lock, template, fs_intan, peak=0.95)
        print(f"[locs] peaks ≥0.95: {locs.size} | first 10: {locs[:10].tolist() if locs.size else []}")
        if locs.size == 0:
            print(f"[warn] No template peaks found for {session}; skipping.")
            continue
        shift = int(locs[0])

        # Load BR intan_sync
        br_ns5 = BR_ROOT / f"NRR_RW_001_{br_idx:03d}.ns5"
        if not br_ns5.exists():
            print(f"[warn] BR file not found: {br_ns5} (skip)")
            continue
        try:
            br_sync, fs_br = load_br_intan_sync_ns5(br_ns5, intan_sync_chan_id=134)
        except Exception as e:
            print(f"[error] load_br_intan_sync_ns5 failed for {br_ns5}: {e}")
            continue
        if int(round(fs_intan)) != int(round(fs_br)):
            print(f"[note] fs mismatch (Intan={fs_intan:g} Hz, BR={fs_br:g} Hz). Proceeding.")

        # Recording lengths
        dur_intan_sec = len(adc_triangle) / fs_intan
        dur_br_sec    = len(br_sync) / fs_br

        # # Best shift (z-corr) near first loc
        # n_best, corr = best_shift_corr_z(
        #     a=ref_intan, b=br_sync, center=shift,
        #     search=1000, win=30000
        # )
        shift_sec = shift / fs_intan
        shift_ms  = shift_sec * 1000.0

        print(
            f"[align] Intan_File={intan_idx:03d} ({dur_intan_sec:.1f} s) "
            f"↔ BR_File={br_idx:03d} ({dur_br_sec:.1f} s) : "
            f"anchor={shift:+d} samp ({shift_sec:+.6f} s)"
        )

        # --------- Load rates npz for BOTH sides ----------
        intan_rates_npz = find_intan_rates_for_session(nprw_checkpoint_root, session)
        if intan_rates_npz is None:
            print(f"[warn] No Intan rates for session {session}")
            continue
        ua_rates_npz = find_ua_rates_by_index(ua_checkpoint_root, br_idx)
        if ua_rates_npz is None:
            print(f"[warn] No UA rates for BR_File {br_idx:03d}")
            continue

        intan_rate_hz, intan_t_ms, intan_meta, intan_pcs, intan_expl = load_rate_npz(intan_rates_npz)
        ua_rate_hz, ua_t_ms, ua_meta, ua_pcs, ua_expl = load_rate_npz(ua_rates_npz)

        # Optional stim times from Intan bundle (block starts)
        stim_ms = try_load_stim_ms_from_intan_bundle(NPRW_BUNDLES, session)

        # --------- Create aligned timebases ----------
        # Shift Intan time vector
        t0_intan_ms = (shift / fs_intan) * 1000.0
        intan_t_ms_aligned = intan_t_ms - t0_intan_ms

        # Eventually also use triangle wave to correct for samples
        t0_ua_ms = 0.0
        ua_t_ms_aligned = ua_t_ms

        # --------- Save combined NPZ ----------
        combined_meta = dict(
            session=session,
            intan_idx=intan_idx,
            br_idx=br_idx,
            adc_npz=str(adc_npz),
            br_ns5=str(br_ns5),
            intan_rates=str(intan_rates_npz),
            ua_rates=str(ua_rates_npz),
            fs_intan=float(fs_intan),
            fs_br=float(fs_br),

            # Anchor is where BR t=0 appears in Intan samples
            anchor_sample=int(shift),
            anchor_seconds=float(shift_sec),
            anchor_ms=float(shift_ms),

            # Durations of recording
            dur_intan_sec=float(dur_intan_sec),
            dur_br_sec=float(dur_br_sec),

            ref_signal=("triangle" if ref_intan is adc_triangle else "lock"),
            # n_best_samples=int(n_best),
            # zcorr_max=float(corr),
        )

        out_npz = aligned_checkpoint / f"aligned__{session}__Intan_{intan_idx:03d}__BR_{br_idx:03d}.npz"
        
        # --------- Create aligned timebases ----------
        # Intan reference: define t=0 at the first matched block start (center)
        t0_intan_ms = (shift / fs_intan) * 1000.0
        intan_t_ms_aligned = intan_t_ms - t0_intan_ms

        # Utah/BR: additionally subtract n_best/fs to bring Utah into Intan time
        # Equivalent to: t_ua_aligned_ms = ua_t_ms - (center+n_best)/fs_intan * 1000
        t0_ua_ms = 0.0
        ua_t_ms_aligned = ua_t_ms  # or: ua_t_ms - t0_ua_ms

        # ---- Trim to overlap and equal length (no interpolation) ----
        (orig_i_len, orig_u_len) = (intan_t_ms_aligned.size, ua_t_ms_aligned.size)
        (intan_rate_hz_trim,
        intan_t_ms_aligned_trim,
        ua_rate_hz_trim,
        ua_t_ms_aligned_trim) = _trim_to_equal_length(
            intan_rate_hz, intan_t_ms_aligned, ua_rate_hz, ua_t_ms_aligned
        )

        if intan_t_ms_aligned_trim.size == 0:
            print(f"[warn] No overlapping time region after alignment for session {session}; skipping.")
            continue

        print(f"[trim] {session}: Intan {orig_i_len}→{intan_t_ms_aligned_trim.size} bins, "
            f"UA {orig_u_len}→{ua_t_ms_aligned_trim.size} bins")
        
        # --------- Save combined NPZ ----------
        combined_meta = dict(
            session=session,
            intan_idx=intan_idx,
            br_idx=br_idx,
            adc_npz=str(adc_npz),
            br_ns5=str(br_ns5),
            intan_rates=str(intan_rates_npz),
            ua_rates=str(ua_rates_npz),
            fs_intan=float(fs_intan),
            fs_br=float(fs_br),

            # Anchor where BR t=0 appears in Intan samples
            anchor_sample=int(shift),
            anchor_seconds=float(shift_sec),
            anchor_ms=float(shift_ms),

            # Durations
            dur_intan_sec=float(dur_intan_sec),
            dur_br_sec=float(dur_br_sec),

            ref_signal=("triangle" if ref_intan is adc_triangle else "lock"),
            trimmed_equal_length=True,
            orig_intan_bins=int(orig_i_len),
            orig_ua_bins=int(orig_u_len),
            trimmed_bins=int(intan_t_ms_aligned_trim.size),
        )

        out_npz = aligned_checkpoint / f"aligned__{session}__Intan_{intan_idx:03d}__BR_{br_idx:03d}.npz"
        
        np.savez_compressed(
            out_npz,
            # Intan (trimmed)
            intan_rate_hz=intan_rate_hz_trim.astype(np.float32),
            intan_t_ms=intan_t_ms.astype(np.float64),  # keep original for reference
            intan_t_ms_aligned=intan_t_ms_aligned_trim.astype(np.float64),
            intan_meta=(intan_meta.item() if hasattr(intan_meta, "item") else intan_meta),
            intan_pcs=intan_pcs if intan_pcs is not None else np.array([], dtype=np.float32),
            intan_explained_var=intan_expl if intan_expl is not None else np.array([], dtype=np.float32),

            # Utah (trimmed)
            ua_rate_hz=ua_rate_hz_trim.astype(np.float32),
            ua_t_ms=ua_t_ms.astype(np.float64),  # keep original for reference
            ua_t_ms_aligned=ua_t_ms_aligned_trim.astype(np.float64),
            ua_meta=(ua_meta.item() if hasattr(ua_meta, "item") else ua_meta),
            ua_pcs=ua_pcs if ua_pcs is not None else np.array([], dtype=np.float32),
            ua_explained_var=ua_expl if ua_expl is not None else np.array([], dtype=np.float32),

            # Stim, meta, etc.
            stim_ms=(stim_ms.astype(np.float64) if stim_ms is not None else np.array([], dtype=np.float64)),
            align_meta=json.dumps(combined_meta),
        )

        print(f"[write] combined aligned → {out_npz}")

        # keep CSV summary row
        summary_rows.append(dict(
            **combined_meta,
            intan_rate_shape=str(intan_rate_hz.shape),
            ua_rate_shape=str(ua_rate_hz.shape),
        ))

    # ---------- write alignment summary CSV ----------
    if summary_rows:
        out_csv = PLOTS_ROOT / "br_to_intan_shifts.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"[done] wrote shifts → {out_csv}")
    else:
        print("[done] no rows to write (nothing aligned).")
