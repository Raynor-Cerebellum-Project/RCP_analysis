#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import re, sys, csv
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import filtfilt
from scipy.signal.windows import gaussian as _scipy_gaussian

import RCP_analysis as rcp

from __future__ import annotations
from types import SimpleNamespace
from pathlib import Path
import RCP_analysis as rcp

# ---------- small helpers ----------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _subdir(root: Path, *parts: str) -> Path:
    return _ensure_dir(root.joinpath(*parts))

# ---------- CONFIG / roots ----------
REPO_ROOT  = Path(__file__).resolve().parents[1]
PARAMS     = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)

DATA_ROOT  = rcp.resolve_data_root(PARAMS)
OUT_BASE   = _ensure_dir(rcp.resolve_output_root(PARAMS))
INTAN_ROOT = _ensure_dir(rcp.resolve_intan_root(PARAMS))

# ---------- metadata ----------
# Guard against missing metadata_rel; loader should set it, but be defensive
metadata_rel = getattr(PARAMS, "metadata_rel", None) or ""
METADATA_CSV  = (DATA_ROOT / metadata_rel).resolve()
METADATA_ROOT = _ensure_dir(METADATA_CSV.parent)
SHIFTS_CSV    = METADATA_ROOT / "br_to_intan_shifts.csv"

# ---------- checkpoints / bundles ----------
CKPT_ROOT       = _subdir(OUT_BASE, "checkpoints")
BEHV_CKPT_ROOT  = _subdir(CKPT_ROOT, "behavior")           # where *_Cam-0_aligned.csv live
BASELINE_ROOT   = _subdir(BEHV_CKPT_ROOT, "baseline_concat")
ALIGNED_OUT     = _subdir(BASELINE_ROOT, "rates_from_curated")

BUNDLES_ROOT    = _subdir(OUT_BASE, "bundles")
NPRW_BUNDLES    = _subdir(BUNDLES_ROOT, "NPRW")

NPRW_CKPT_ROOT  = _subdir(CKPT_ROOT, "NPRW")
UA_CKPT_ROOT    = _subdir(CKPT_ROOT, "UA")

# ---------- analysis constants ----------
IR_STREAM        = "USB board digital input channel"  # stream to preview
DEFAULT_BR_FS    = 30000.0  # Hz; override if you know the true ns5 rate
WIN_MS           = (-750.0, 750.0)
BASELINE_FIRST_MS = 150.0

# ---------- keypoints order (from YAML, with fallback) ----------
KEYPOINTS_ORDER = tuple((PARAMS.kinematics or {}).get("keypoints", ())
                        or ("wrist", "middle_finger_base", "middle_finger_tip"))

CURATED_EVENTS_B = []
# CURATED_EVENTS_B = [0, 3, 4, 7, 8, 11, 13, *range(15, 20), 23, 24, 30, 33, 37, 44, 53, 55]

# --- add near top, after other regex helpers ---
_TRIAL_VIDEO_IDX_RE = re.compile(
    r'NRR_[A-Za-z]+(?:\d{3})_(\d{3})(?:_[0-9]{4}_[0-9]{4}_[0-9]{6})?(?:\b|_)',
    re.IGNORECASE
)

# ===== NPRW/UA helpers (minimal) =====
def load_rate_npz(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    rate_hz = z["rate_hz"]           # (n_ch, n_bins)
    t_ms    = z["t_ms"]              # (n_bins,)
    meta    = z.get("meta", None)
    pcs     = z.get("pcs", None)
    expl    = z.get("explained_var", None)
    return rate_hz, t_ms, meta, pcs, expl

def find_intan_rates_for_session(nprw_ckpt_root: Path, session: str) -> Path | None:
    cands = sorted(nprw_ckpt_root.glob(f"rates__{session}__*.npz"))
    return cands[0] if cands else None

def find_ua_rates_by_index(ua_ckpt_root: Path, br_idx: int) -> Path | None:
    cands = sorted(ua_ckpt_root.glob(f"rates__*_{int(br_idx):03d}__*.npz"))
    if not cands:
        return None
    pref = [p for p in cands if re.search(r"__sigma?25ms", p.stem)]
    return pref[0] if pref else cands[-1]

def _resample_rate_to_time(rate_hz: np.ndarray, t_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    n_ch = rate_hz.shape[0]
    out = np.empty((n_ch, t_tgt.size), dtype=np.float32)
    r32 = rate_hz.astype(np.float32, copy=False)
    for ch in range(n_ch):
        out[ch] = np.interp(t_tgt, t_src, r32[ch])
    return out

def _overlap_and_resample_to_intan(i_rate, i_t, u_rate, u_t):
    if i_t.size == 0 or u_t.size == 0:
        return i_rate[:, :0], i_t[:0], u_rate[:, :0], {"overlap_bins": 0, "dt_i_ms": np.nan}
    t0 = max(float(i_t[0]), float(u_t[0]))
    t1 = min(float(i_t[-1]), float(u_t[-1]))
    if not (t1 > t0):
        return i_rate[:, :0], i_t[:0], u_rate[:, :0], {"overlap_bins": 0, "dt_i_ms": np.nan}
    mi = (i_t >= t0) & (i_t <= t1)
    common_t = i_t[mi]
    i_trim = i_rate[:, mi]
    if common_t.size < 2:
        return i_rate[:, :0], i_t[:0], u_rate[:, :0], {"overlap_bins": 0, "dt_i_ms": np.nan}
    mu = (u_t >= common_t[0]) & (u_t <= common_t[-1])
    u_crop, u_t_crop = u_rate[:, mu], u_t[mu]
    if u_t_crop.size < 2:
        return i_rate[:, :0], i_t[:0], u_rate[:, :0], {"overlap_bins": 0, "dt_i_ms": np.nan}
    u_on_common = _resample_rate_to_time(u_crop, u_t_crop, common_t)
    dt_i = float(np.median(np.diff(common_t)))
    return i_trim, common_t, u_on_common, {"overlap_bins": int(common_t.size), "dt_i_ms": dt_i}

def _window_rate_around_events(rate: np.ndarray, t_vec_ms: np.ndarray,
                               event_ms: np.ndarray, pre_ms: float, post_ms: float):
    """
    rate: (n_ch, T_full) sampled at times t_vec_ms (in ms, increasing)
    event_ms: centers in the same timebase (ms)
    Returns (windows[N, n_ch, T], rel_t_ms[T])
    """
    if t_vec_ms.size < 2:
        return np.zeros((0, rate.shape[0], 0), np.float32), np.zeros((0,), np.float32)

    dt = float(np.median(np.diff(t_vec_ms)))
    T = int(round((pre_ms + post_ms) / dt)) + 1
    rel_t = (np.arange(T) * dt - pre_ms).astype(np.float32)

    out = np.full((len(event_ms), rate.shape[0], T), np.nan, np.float32)
    for i, c in enumerate(event_ms.astype(float)):
        t0 = c - pre_ms; t1 = c + post_ms
        i0 = int(np.searchsorted(t_vec_ms, t0, side="left"))
        i1 = int(np.searchsorted(t_vec_ms, t1, side="right"))
        if i0 >= t_vec_ms.size or i1 <= 0:
            continue
        idx = np.clip(np.round(np.linspace(max(i0,0), max(i1-1,0), T)).astype(int), 0, t_vec_ms.size-1)
        out[i] = rate[:, idx]
    return out, rel_t

# --- helpers for combined ("both_cams") or per-cam CSVs ---

def _flatten_cols(cols) -> list[str]:
    out = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [str(p) for p in c if p is not None and str(p) != "nan"]
            out.append("_".join(parts))
        else:
            out.append(str(c))
    return out

def _find_aligned_cam_csv(trial: str, cam: int = 0) -> Path | None:
    """
    Prefer combined '<trial>_both_cams_aligned.csv', else per-cam '<trial>_Cam-{cam}_aligned.csv'.
    """
    p_both = BEHV_CKPT_ROOT / f"{trial}_both_cams_aligned.csv"
    if p_both.exists():
        return p_both
    p_cam = BEHV_CKPT_ROOT / f"{trial}_Cam-{cam}_aligned.csv"
    return p_cam if p_cam.exists() else None

def _build_videoidx_to_trial_map(cam:int=0) -> dict[int,str]:
    """
    Map Video_File index -> trial, scanning both combined and per-cam aligned CSVs.
    If duplicates, newest file wins.
    """
    picked: dict[int, tuple[str, float]] = {}

    # 1) combined
    for p in BEHV_CKPT_ROOT.glob("*_both_cams_aligned.csv"):
        trial = p.name.replace("_both_cams_aligned.csv", "")
        vid = _extract_video_idx_from_trial(trial)
        if vid is None:
            continue
        mt = p.stat().st_mtime
        prev = picked.get(vid)
        if prev is None or mt > prev[1]:
            picked[vid] = (trial, mt)

    # 2) per-cam fallback
    for p in BEHV_CKPT_ROOT.glob(f"*_Cam-{cam}_aligned.csv"):
        trial = p.name.replace(f"_Cam-{cam}_aligned.csv", "")
        vid = _extract_video_idx_from_trial(trial)
        if vid is None:
            continue
        mt = p.stat().st_mtime
        prev = picked.get(vid)
        if prev is None or mt > prev[1]:
            picked[vid] = (trial, mt)

    return {k: v[0] for k, v in picked.items()}

def _columns_for_keypoints(df_cols, cam_prefix: str = ""):
    """
    Return ordered [kp1_x,kp1_y,kp2_x,kp2_y,...] for the requested camera.

    Columns look like:
      cam0_wrist_x, cam0.wrist_y, cam1 middle_finger_tip_x, ...
    We match: ^cam[01][_. ]  ...  <keypoint>_<axis>$

    If both cam0 and cam1 columns exist, we DO NOT fall back to unscoped names.
    """
    cols    = [str(c) for c in df_cols]
    cols_lc = [c.lower() for c in cols]

    cam_base = str(cam_prefix).lower().strip().rstrip("_. ")  # "cam0" or "cam1"

    def _has_cam(prefix: str) -> bool:
        p = prefix.lower()
        return any(c.startswith(p + "_") or c.startswith(p + ".") or c.startswith(p + " ")
                   for c in cols_lc)

    has_cam0 = _has_cam("cam0")
    has_cam1 = _has_cam("cam1")
    both_cams_present = has_cam0 and has_cam1

    def _is_cam_col(name_lc: str, suffix: str) -> bool:
        # must start with requested cam + allowed separator and end with the suffix
        return (
            (name_lc.startswith(cam_base + "_") or
             name_lc.startswith(cam_base + ".") or
             name_lc.startswith(cam_base + " "))
            and name_lc.endswith(suffix)
        )

    out = []
    for kp in KEYPOINTS_ORDER:
        for axis in ("x", "y"):
            suffix = f"{kp}_{axis}"  # e.g., "wrist_x"
            hit_idx = next((i for i, c in enumerate(cols_lc) if _is_cam_col(c, suffix)), None)

            # Only if not ambiguous (i.e., NOT both cameras present) or no cam_base given:
            if hit_idx is None and (not both_cams_present or not cam_base):
                hit_idx = next((i for i, c in enumerate(cols_lc) if c.endswith(suffix)), None)

            out.append(cols[hit_idx] if hit_idx is not None else None)

    return out

def _extract_xy_matrix(aligned_csv: Path, cam: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Handles combined MultiIndex headers (cam0/cam1) or flat. Uses cam-scoped ns5_sample if present.
    """
    # try reading potential MultiIndex first
    try:
        df = pd.read_csv(aligned_csv, header=[0, 1])
        # if not actually multiindex, pandas still gives 2-level with many 'Unnamed'
        flat = _flatten_cols(df.columns)
        # heuristic: if many columns look like 'Unnamed', re-read as flat
        if sum(c.lower().startswith("unnamed") for c in flat) > len(flat) // 2:
            df = pd.read_csv(aligned_csv)
        else:
            df.columns = flat
    except Exception:
        df = pd.read_csv(aligned_csv)

    cols = [str(c) for c in df.columns]
    cols_lc = [c.lower() for c in cols]

    # time column candidates
    cam_prefix = f"cam{cam}"
    time_cands = [
        "ns5_sample",
        f"{cam_prefix}_ns5_sample",
        f"{cam_prefix}.ns5_sample",
        f"{cam_prefix} ns5_sample",
    ]
    time_col = next((cols[i] for i, c in enumerate(cols_lc) if c in time_cands), None)
    if time_col is None:
        raise RuntimeError(f"{aligned_csv.name} has no ns5_sample column (tried {time_cands}).")

    t_cam_sec = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float) / DEFAULT_BR_FS

    # pick XY columns (scoped to requested cam when available)
    wanted = _columns_for_keypoints(df.columns, cam_prefix=f"cam{cam}")

    XY = np.empty((len(df), len(wanted)), float)
    names = []
    for j, col in enumerate(wanted):
        if col is None or col not in df.columns:
            XY[:, j] = np.nan
            names.append(f"{KEYPOINTS_ORDER[j//2]}_{'xy'[j%2]}(MISSING)")
        else:
            XY[:, j] = pd.to_numeric(df[col], errors="coerce").to_numpy()
            names.append(str(col))

    return t_cam_sec, XY, names

def _remove_teleports_and_spikes(XY: np.ndarray, max_z: float = 6.0,
                                 jump_px: float | None = None,
                                 dilate: int = 1,
                                 interp_max_gap: int = 6) -> np.ndarray:
    """
    Detect frames where velocity (frame-to-frame) is implausibly large and set to NaN,
    then lightly inpaint small gaps.
    - max_z: robust z-score threshold on speed (per keypoint)
    - jump_px: absolute per-step jump threshold (pixels); if None, derive from IQR
    - dilate: also drop +/- dilate frames around each detected outlier
    - interp_max_gap: linearly fill NaN runs up to this length
    """
    if XY.size == 0:
        return XY
    out = XY.astype(float, copy=True)
    T, D = out.shape
    for col0 in range(0, D, 2):  # pair (x,y)
        x = out[:, col0]
        y = out[:, col0+1]
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        spd = np.sqrt(dx*dx + dy*dy)

        # robust z-score
        med = np.nanmedian(spd)
        mad = np.nanmedian(np.abs(spd - med)) + 1e-12
        z = 0.6745 * (spd - med) / mad

        # absolute jump threshold
        if jump_px is None:
            q75 = np.nanpercentile(spd, 75.0)
            q25 = np.nanpercentile(spd, 25.0)
            iqr = max(q75 - q25, 1.0)
            jthr = q75 + 6.0*iqr  # pretty conservative
        else:
            jthr = float(jump_px)

        bad = (np.abs(z) > max_z) | (spd > jthr)
        if bad.any() and dilate > 0:
            bad_idx = np.flatnonzero(bad)
            for i in bad_idx:
                lo = max(0, i - dilate)
                hi = min(T, i + dilate + 1)
                bad[lo:hi] = True

        # zero out those samples
        x[bad] = np.nan; y[bad] = np.nan
        out[:, col0] = x; out[:, col0+1] = y

        # small-gap interpolation
        if interp_max_gap > 0:
            # find NaN runs
            nanmask = ~np.isfinite(x)
            if nanmask.any():
                starts = np.where(nanmask & ~np.r_[False, nanmask[:-1]])[0]
                ends   = np.where(nanmask & ~np.r_[nanmask[1:], False])[0]
                for s,e in zip(starts, ends):
                    if (e - s + 1) <= interp_max_gap and s > 0 and e < T-1:
                        xi = np.interp(np.arange(s, e+1), [s-1, e+1], [x[s-1], x[e+1]])
                        yi = np.interp(np.arange(s, e+1), [s-1, e+1], [y[s-1], y[e+1]])
                        out[s:e+1, col0]   = xi
                        out[s:e+1, col0+1] = yi
    return out

def _make_gaussian_kernel(sigma_frames: float) -> np.ndarray:
    """
    Return a normalized odd-length 1D Gaussian kernel for 'sigma_frames' (in frames).
    """
    if sigma_frames <= 0:
        return np.array([1.0], dtype=float)
    # window length ~ 6σ and forced odd
    win_len = int(max(5, round(6.0 * sigma_frames)))
    if win_len % 2 == 0:
        win_len += 1

    if _scipy_gaussian is not None:
        g = _scipy_gaussian(win_len, std=sigma_frames)
    else:
        # local fallback
        n = np.arange(win_len) - (win_len - 1) / 2.0
        g = np.exp(-0.5 * (n / float(sigma_frames))**2)

    g = g.astype(float, copy=False)
    g_sum = g.sum()
    if not np.isfinite(g_sum) or g_sum == 0:
        return np.array([1.0], dtype=float)
    return g / g_sum

def _gaussian_filtfilt(XY: np.ndarray, sigma_frames: float = 2.0) -> np.ndarray:
    """
    Zero-phase Gaussian smoothing (per column) using filtfilt with a FIR Gaussian kernel.
    sigma_frames is in frames (corrected-frame domain).
    """
    if XY.size == 0 or sigma_frames <= 0:
        return XY
    out = XY.copy()
    g = _make_gaussian_kernel(sigma_frames)
    # filtfilt default padlen is 3*(max(len(a), len(b)) - 1) => 3*(len(g)-1) for FIR
    base_pad = 3 * (len(g) - 1)

    for j in range(out.shape[1]):
        col = out[:, j]
        finite = np.isfinite(col)
        if finite.sum() < len(g) + 2:
            # too short / too many NaNs to filter robustly
            continue
        # simple inpaint of NaNs for stability
        if (~finite).any():
            idx = np.arange(col.size)
            col = np.interp(idx, idx[finite], col[finite])

        padlen = min(base_pad, len(col) - 1)
        if padlen < 1:
            continue
        out[:, j] = filtfilt(g, [1.0], col, padlen=padlen)

    return out

def build_peri_event_windows(trial: str,
                             evt_sec: np.ndarray,
                             cam:int=0,
                             pre_ms: float = 750.0,
                             post_ms: float = 750.0,
                             sigma_frames: float = 2.0,
                             offset_sec: float = 0.0,
                             anchor_ms: float = 0.0,
                             debug: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load aligned cam CSV, clean, smooth, and return windows aligned to IR events.
    Returns (windows[N,K,T], rel_t_ms[T], labels[K]).

    offset_sec: any additional constant BR–Intan shift you may know (in seconds).
    anchor_ms:  per-row anchor (ms) from br_to_intan_shifts; this is SUBTRACTED.
    """
    csvp = _find_aligned_cam_csv(trial, cam=cam)
    if not csvp:
        raise FileNotFoundError(f"No aligned CSV for {trial} Cam-{cam}")
    t_cam_sec, XY, labels = _extract_xy_matrix(csvp, cam=cam)

    # clean/smooth
    XY = _remove_teleports_and_spikes(XY, max_z=6.0, jump_px=None, dilate=1, interp_max_gap=6)
    XY = _gaussian_filtfilt(XY, sigma_frames=sigma_frames)

    # windowing
    dt = np.nanmedian(np.diff(t_cam_sec[np.isfinite(t_cam_sec)]))
    if not np.isfinite(dt) or dt <= 0:
        raise RuntimeError("Could not infer frame dt from ns5_sample timing.")
    pre = pre_ms/1000.0; post = post_ms/1000.0
    T = int(round((pre+post)/dt)) + 1
    rel_t = (np.arange(T) * dt - pre) * 1000.0  # ms
    K = XY.shape[1]
    N = len(evt_sec)
    out = np.full((N, K, T), np.nan, float)

    anchor_sec = float(anchor_ms) / 1000.0

    if debug:
        dt_ms = dt * 1000.0
        tmin, tmax = (float(np.nanmin(t_cam_sec)), float(np.nanmax(t_cam_sec))) if t_cam_sec.size else (np.nan, np.nan)
        print(f"[align] trial={trial} cam={cam} N_evt={N} dt={dt_ms:.3f} ms "
              f"pre/post={pre_ms:.1f}/{post_ms:.1f} ms  sigma_frames={sigma_frames:g}")
        print(f"[align] offset_sec={offset_sec:.6f} s  anchor_ms={anchor_ms:.3f} ms  "
              f"cam_range=[{tmin:.3f}, {tmax:.3f}] s  T={T} frames")

        # show centers for every event
        if N > 0:
            show_idx = np.arange(N)  # <- all events
            for i in show_idx:
                t0 = float(evt_sec[i])
                t_center = t0 + offset_sec - anchor_sec
                ic = int(np.searchsorted(t_cam_sec, t_center, side="left"))
                if 0 <= ic < len(t_cam_sec):
                    err_ms = (t_cam_sec[ic] - t_center) * 1000.0
                    print(f"  · evt[{i}] intan_t={t0:.6f}s → center={t_center:.6f}s "
                        f"→ nearest_cam_frame[{ic}]={t_cam_sec[ic]:.6f}s  (Δ={err_ms:.2f} ms)")
                else:
                    print(f"  · evt[{i}] intan_t={t0:.6f}s → center={t_center:.6f}s → nearest_cam_frame=OOB")

    for i, t0 in enumerate(evt_sec):
        # Align Intan event time to BR frame time:
        t_center = t0 + offset_sec - anchor_sec
        t0s = t_center - pre
        t1s = t_center + post
        i0 = np.searchsorted(t_cam_sec, t0s, side="left")
        i1 = np.searchsorted(t_cam_sec, t1s, side="right")
        idx = np.clip(np.round(np.linspace(i0, i1-1, T)).astype(int), 0, len(t_cam_sec)-1)
        out[i, :, :] = XY[idx, :].T

        if debug:
            covered = (t_cam_sec[idx[-1]] - t_cam_sec[idx[0]]) if len(idx) > 1 else 0.0
            print(f"  · win[{i}] idx=[{i0}:{i1}) → sampled {T} frames "
                f"(cam[{idx[0]}..{idx[-1]}], span≈{covered:.3f}s)")

    return out, rel_t, labels

# ---------- helpers ----------
def _read_csv_robust(path: Path) -> pd.DataFrame:
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1", "mac_roman")
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not read CSV {path} (last error: {last_err})")

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df.columns]
    return df

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _fmt_num(x):
    try:
        v = float(x)
        if pd.isna(v):
            return "n/a"
        return f"{int(v)}" if v.is_integer() else f"{v:g}"
    except Exception:
        s = str(x).strip()
        return s if s else "n/a"

def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

def _extract_signal_fs(obj):
    """Return (sig[samples,channels], fs) from SI extractor / (sig,fs) / ndarray."""
    sig = None; fs = None
    if hasattr(obj, "get_traces") and hasattr(obj, "get_sampling_frequency"):
        try: fs = float(obj.get_sampling_frequency())
        except Exception: fs = None
        try: sig = np.asarray(obj.get_traces(), dtype=float)
        except Exception: sig = None
    if sig is None and isinstance(obj, (tuple, list)) and len(obj) >= 1:
        try:
            sig = np.asarray(obj[0], dtype=float)
            if len(obj) >= 2: fs = float(obj[1])
        except Exception: sig = None
    if sig is None and hasattr(obj, "__array__"):
        try: sig = np.asarray(obj, dtype=float)
        except Exception: sig = None
    if sig is None: return None, None
    if sig.ndim == 1:
        sig = sig[:, None]
    elif sig.ndim == 2 and sig.shape[0] < sig.shape[1] and sig.shape[0] <= 16:
        sig = sig.T
    else:
        sig = sig.reshape(sig.shape[0], -1)
    return sig.astype(np.float32, copy=False), fs

def _load_br_to_intan_map_full(shifts_csv: Path) -> dict[int, dict]:
    """
    Return {BR_idx: {"session": <str>, "anchor_ms": <float or 0.0>}}.
    Accepts header variations.
    """
    if not shifts_csv.exists():
        raise SystemExit(f"[error] shifts CSV not found: {shifts_csv}")
    with shifts_csv.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            raise SystemExit(f"[error] {shifts_csv.name} has no header")
        # normalize header names -> original
        cols = { re.sub(r'[^a-z0-9]+','', c.lower()): c for c in rdr.fieldnames }

        def col(*cands):
            for cand in cands:
                k = re.sub(r'[^a-z0-9]+','', cand.lower())
                if k in cols: return cols[k]
            return None

        c_br   = col("br_idx", "br", "brfile")
        c_sess = col("session", "intan_session", "intan")
        c_anchor = col("anchor_ms", "anchoroffsetms", "offset_ms", "br2intan_anchor_ms")

        if not c_br or not c_sess:
            raise SystemExit(f"[error] {shifts_csv.name} missing BR and/or Session columns (have: {rdr.fieldnames})")

        out: dict[int, dict] = {}
        for row in rdr:
            try:
                br = int(float(str(row[c_br]).strip()))
            except Exception:
                continue
            sess = str(row.get(c_sess, "")).strip()
            if not sess:
                continue
            try:
                anchor_ms = float(str(row.get(c_anchor, "0")).strip()) if c_anchor else 0.0
            except Exception:
                anchor_ms = 0.0
            out[br] = {"session": sess, "anchor_ms": anchor_ms}
    return out

def _extract_video_idx_from_trial(trial: str) -> int | None:
    m = _TRIAL_VIDEO_IDX_RE.search(trial)
    if m:
        return int(m.group(1))
    nums = re.findall(r'(?<!\d)(\d{3})(?!\d)', trial)
    return int(nums[-1]) if nums else None

def _choose_ch_with_signal(sig: np.ndarray) -> int:
    """Pick a channel that likely carries the IR (largest std)."""
    if sig.ndim != 2 or sig.shape[1] == 0:
        return 0
    stds = [np.nanstd(sig[:, j]) for j in range(sig.shape[1])]
    return int(np.nanargmax(stds)) if len(stds) else 0

def _falling_edges_from_analog(x: np.ndarray, fs: float | None, refractory_sec: float = 0.0005) -> np.ndarray:
    """
    Binarize at mid-point and return sample indices of 1→0 edges.
    Debounce via refractory (default 0.5 ms).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([], dtype=np.int64)

    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    thr = 0.5 * (lo + hi)
    b = (x > thr).astype(np.int8)

    # edges where b[t-1]=1 and b[t]=0
    db = np.diff(b, prepend=b[0])
    edges = np.flatnonzero(db == -1)

    if edges.size == 0:
        return edges.astype(np.int64)

    if fs and fs > 0:
        refr = max(1, int(round(refractory_sec * fs)))
    else:
        refr = 1

    keep = []
    last = -10**12
    for i in edges:
        if i - last >= refr:
            keep.append(i)
            last = i
    return np.asarray(keep, dtype=np.int64)

def _resolve_curated_indices_for_port(port: str, total_n: int):
    """Return np.array of curated indices for this port, or None if not specified."""
    if str(port).upper() != "B":
        return None
    keep = np.array(list(CURATED_EVENTS_B), dtype=int)
    # de-dup, sort, and clip to valid range
    keep = np.unique(keep[(keep >= 0) & (keep < int(total_n))])
    return keep if keep.size else None

# ---------- main ----------
def main():
    if not METADATA_CSV.exists():
        print(f"[error] Metadata CSV not found: {METADATA_CSV}", file=sys.stderr)
        sys.exit(2)

    # Build once: Video_File idx → trial (Cam-0 preference for inventory)
    videoidx_to_trial = _build_videoidx_to_trial_map(cam=0)
    shifts_full = _load_br_to_intan_map_full(SHIFTS_CSV)

    df_raw = _read_csv_robust(METADATA_CSV)
    if df_raw.empty:
        print(f"[warn] Metadata CSV is empty: {METADATA_CSV}")
        return
    df = _norm_cols(df_raw)

    col_notes = _pick_col(df, ["notes","note","comments","comment"])
    col_port  = _pick_col(df, ["ua_port","uaport","port"])
    col_depth = _pick_col(df, ["depth_mm","depth","ua_depth_mm","ua_depth"])
    col_br    = _pick_col(df, ["br_file","brfile","br"])
    col_video = _pick_col(df, ["video_file","videofile","video"])

    if col_notes is None:
        print("[error] No 'Notes' column in metadata.", file=sys.stderr)
        sys.exit(3)

    is_baseline = df[col_notes].astype(str).str.strip().str.lower().eq("baseline")
    base = df.loc[is_baseline].copy()

    print(f"[info] metadata: {METADATA_CSV}")
    print(f"[info] total rows: {len(df)}")
    print(f"[info] baseline rows: {len(base)}\n")
    if base.empty:
        return

    base["_ua_port_norm"] = (
        base[col_port].astype(str).str.strip().str.upper().replace({"": "UNKNOWN"})
        if col_port else "UNKNOWN"
    )
    if col_depth:
        depth_num = pd.to_numeric(base[col_depth], errors="coerce")
        base["_depth_mm_norm"] = depth_num.where(depth_num.notna(), base[col_depth].astype(str).str.strip()).replace({"": "n/a"})
    else:
        base["_depth_mm_norm"] = "n/a"

    groups = list(base.groupby(["_ua_port_norm", "_depth_mm_norm"], sort=True))

    for (port, depth), g in groups:
        depth_label = _fmt_num(depth)
        print(f"=== UA_port: {port} | Depth_mm: {depth_label}  (n={len(g)}) ===")

        # Collectors across rows in this UA_port×Depth group
        evt_all_sec: list[np.ndarray] = []          # per-row arrays of event centers (sec, Intan)
        trial_per_event: list[str] = []            # one entry per event
        anchor_per_event_ms: list[float] = []      # one entry per event
        session_blocks: list[dict] = []            # for later curated-rate extraction

        # per-camera kinematic windows (to be concatenated after the loop)
        win_all_cam0: list[np.ndarray] = []
        win_all_cam1: list[np.ndarray] = []
        rel_t_ms_cam0 = rel_t_ms_cam1 = None
        labels_cam0 = labels_cam1 = None

        # running global event index for concat bookkeeping
        global_start = 0

        for _, r in g.iterrows():
            br_raw = str(r.get(col_br, "")).strip() if col_br else ""
            vid_raw = str(r.get(col_video, "")).strip() if col_video else ""
            try:
                br_num = int(float(br_raw)) if br_raw else None
            except Exception:
                br_num = None
            try:
                video_idx = int(float(vid_raw)) if vid_raw else None
            except Exception:
                video_idx = None

            trial = videoidx_to_trial.get(video_idx) if video_idx is not None else None

            sess_entry = shifts_full.get(br_num) if br_num is not None else None
            intan_session = (sess_entry or {}).get("session")
            row_anchor_ms = float((sess_entry or {}).get("anchor_ms", 0.0))

            debug_flag = (str(port).upper() != "B")
            print(f"BASELINE[{port} | Depth={depth_label}]: Trial={trial or 'n/a'}  Intan={intan_session or 'n/a'}  "
                  f"BR={br_raw or 'n/a'}  Video={vid_raw or 'n/a'}  anchor_ms={row_anchor_ms:g}")

            if not intan_session or not trial:
                continue

            # IR edges
            try:
                rec_ir = rcp.read_intan_recording(INTAN_ROOT / intan_session, stream_name=IR_STREAM)
            except Exception as e:
                print(f"[warn] read_intan_recording failed: Intan={intan_session}: {e}")
                continue

            sig, fs = _extract_signal_fs(rec_ir)
            if sig is None or sig.size == 0:
                print(f"[warn] No signal for Intan={intan_session}.")
                continue

            ch = _choose_ch_with_signal(sig)
            x = sig[:, ch]
            evt_idx = _falling_edges_from_analog(x, fs, refractory_sec=0.0005)
            if evt_idx.size == 0:
                print(f"[IR] {intan_session}: no falling edges on ch{ch}")
                continue

            evt_sec = evt_idx / float(fs) if (fs and fs > 0) else np.full(evt_idx.size, np.nan)
            print(f"[IR] {intan_session}: {evt_idx.size} falling edges on ch{ch}")

            # Kinematics windows — Cam-0 (required)
            try:
                kin0, rel0, lab0 = build_peri_event_windows(
                    trial, evt_sec, cam=0,
                    pre_ms=-WIN_MS[0], post_ms=WIN_MS[1],
                    sigma_frames=2.0, offset_sec=0.0,
                    anchor_ms=row_anchor_ms, debug=debug_flag
                )
            except Exception as e:
                print(f"[warn] kinematics failed for Trial={trial}: {e}")
                continue

            # Kinematics windows — Cam-1 (optional)
            kin1 = rel1 = lab1 = None
            try:
                kin1, rel1, lab1 = build_peri_event_windows(
                    trial, evt_sec, cam=1,
                    pre_ms=-WIN_MS[0], post_ms=WIN_MS[1],
                    sigma_frames=2.0, offset_sec=0.0,
                    anchor_ms=row_anchor_ms, debug=debug_flag
                )
            except Exception as e:
                print(f"[warn] kinematics Cam-1 failed for Trial={trial}: {e}")

            # append per-row chunks
            win_all_cam0.append(kin0); rel_t_ms_cam0 = rel0; labels_cam0 = lab0
            if isinstance(kin1, np.ndarray):
                win_all_cam1.append(kin1); rel_t_ms_cam1 = rel1; labels_cam1 = lab1

            # provenance per event
            evt_all_sec.append(evt_sec)
            trial_per_event.extend([trial] * len(evt_sec))
            anchor_per_event_ms.extend([row_anchor_ms] * len(evt_sec))

            # concat index bookkeeping for later curated-rate extraction
            n_this = len(evt_sec)
            session_blocks.append({
                "intan_session": intan_session,
                "br_idx": br_num,
                "anchor_ms": row_anchor_ms,
                "evt_sec": evt_sec,            # Intan seconds (pre-anchor)
                "concat_start": global_start,  # inclusive
                "concat_end": global_start + n_this,  # exclusive
            })
            global_start += n_this

        # ===== after iterating all rows in this group, concatenate & save =====
        if not win_all_cam0:
            print(f"[group] UA={port} Depth={depth_label}: no usable events.")
            continue

        evt_cat = np.concatenate(evt_all_sec, axis=0)                  # (N,)
        cam0_cat = np.concatenate(win_all_cam0, axis=0)                # (N, K0, T)
        cam1_cat = np.concatenate(win_all_cam1, axis=0) if win_all_cam1 else None  # (N, K1, T) or None
        keep = _resolve_curated_indices_for_port(port, cam0_cat.shape[0]) if cam0_cat is not None else None

        def _save_cam_npz(suffix: str, win_cat: np.ndarray, rel_t_ms: np.ndarray, labels: list[str]):
            if win_cat is None:
                return None
            fname_base = f"baseline_UA_{_sanitize(str(port))}_Depth{_sanitize(str(depth_label))}{suffix}"
            if keep is None:
                fname = f"{fname_base}.npz"
                windows_to_save = win_cat
                events_sec_to_save = evt_cat
                trials_to_save = np.array(trial_per_event, dtype=object)
                anchors_to_save = np.array(anchor_per_event_ms, dtype=float)
                curated_flag = "no"
                print(f"[curate] UA={port}{suffix}: saving ALL {win_cat.shape[0]} events.")
            else:
                fname = f"{fname_base}_curated.npz"
                windows_to_save = win_cat[keep, :, :]
                events_sec_to_save = evt_cat[keep]
                trials_to_save = np.array(trial_per_event, dtype=object)[keep]
                anchors_to_save = np.array(anchor_per_event_ms, dtype=float)[keep]
                curated_flag = "yes"
                print(f"[curate] UA={port}{suffix}: keeping {keep.size}/{win_cat.shape[0]} events → {fname}")

            out_npz = BASELINE_ROOT / fname
            np.savez_compressed(
                out_npz,
                windows=windows_to_save,
                t_ms=np.asarray(rel_t_ms, float) if rel_t_ms is not None else np.array([], float),
                labels=np.array(labels, dtype=object) if labels is not None else np.array([], dtype=object),
                events_sec=events_sec_to_save,
                trials=trials_to_save,             # now 1:1 with events
                anchor_ms=anchors_to_save,         # now 1:1 with events
                ua_port=str(port),
                depth_mm=str(depth_label),
            )
            print(f"[save] wrote {out_npz}")

            # manifest row
            manifest = BASELINE_ROOT / "manifest.csv"
            write_header = not manifest.exists()
            with manifest.open("a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["file","ua_port","depth_mm","n_events","n_traces","T","curated"])
                w.writerow([
                    str(out_npz), str(port), str(depth_label),
                    int(windows_to_save.shape[0]), int(windows_to_save.shape[1]), int(windows_to_save.shape[2]),
                    curated_flag
                ])
            return out_npz

        out_cam0 = _save_cam_npz("_cam0", cam0_cat, rel_t_ms_cam0, labels_cam0)
        out_cam1 = _save_cam_npz("_cam1", cam1_cat, rel_t_ms_cam1, labels_cam1)

        # ===== curated NPRW/UA rate windows (Port B only), AFTER we know 'keep' =====
        if keep is not None and str(port).upper() == "B" and keep.size:
            print(f"[rates] UA={port}: extracting NPRW/UA rate windows for curated events...")
            rate_i_chunks: list[np.ndarray] = []
            rate_u_chunks: list[np.ndarray] = []
            evt_ms_all:    list[np.ndarray] = []
            prov_sess:     list[np.ndarray] = []
            prov_br:       list[np.ndarray] = []
            prov_gidx:     list[np.ndarray] = []
            prov_lidx:     list[np.ndarray] = []
            rel_t_ref:     np.ndarray | None = None

            for blk in session_blocks:
                g0, g1 = int(blk["concat_start"]), int(blk["concat_end"])
                sel_global = keep[(keep >= g0) & (keep < g1)]
                if sel_global.size == 0:
                    continue
                sel_local = (sel_global - g0).astype(int)

                intan_rates_npz = find_intan_rates_for_session(NPRW_CKPT_ROOT, blk["intan_session"])
                ua_rates_npz    = find_ua_rates_by_index(UA_CKPT_ROOT, int(blk["br_idx"]) if blk["br_idx"] is not None else -1)
                if intan_rates_npz is None or ua_rates_npz is None:
                    print(f"[rates] skip session={blk['intan_session']}: missing rates "
                          f"(Intan={bool(intan_rates_npz)}, UA={bool(ua_rates_npz)})")
                    continue

                i_rate, i_t_ms, *_ = load_rate_npz(intan_rates_npz)
                u_rate, u_t_ms, *_ = load_rate_npz(ua_rates_npz)

                i_t_ms_al = i_t_ms - float(blk["anchor_ms"])
                u_t_ms_al = u_t_ms

                i_trim, common_t, u_on_common, st = _overlap_and_resample_to_intan(i_rate, i_t_ms_al, u_rate, u_t_ms_al)
                if common_t.size == 0:
                    print(f"[rates] session={blk['intan_session']}: no overlap after alignment.")
                    continue

                evt_ms_al = (blk["evt_sec"][sel_local].astype(float) * 1000.0) - float(blk["anchor_ms"])

                i_win, rel_t_rate = _window_rate_around_events(i_trim, common_t, evt_ms_al, -WIN_MS[0], WIN_MS[1])
                u_win, _          = _window_rate_around_events(u_on_common, common_t, evt_ms_al, -WIN_MS[0], WIN_MS[1])

                if rel_t_ref is None:
                    rel_t_ref = rel_t_rate.copy()
                elif not np.allclose(rel_t_rate, rel_t_ref, atol=1e-6):
                    def _resample_win(win: np.ndarray, t_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
                        n_ev, n_ch, _T = win.shape
                        out = np.empty((n_ev, n_ch, t_tgt.size), dtype=np.float32)
                        for e in range(n_ev):
                            for ch in range(n_ch):
                                out[e, ch] = np.interp(t_tgt, t_src, win[e, ch].astype(np.float32, copy=False))
                        return out
                    i_win = _resample_win(i_win, rel_t_rate, rel_t_ref)
                    u_win = _resample_win(u_win, rel_t_rate, rel_t_ref)

                rate_i_chunks.append(i_win.astype(np.float32, copy=False))
                rate_u_chunks.append(u_win.astype(np.float32, copy=False))
                evt_ms_all.append(evt_ms_al.astype(np.float32, copy=False))
                prov_sess.append(np.array([blk["intan_session"]]*i_win.shape[0], dtype=object))
                prov_br.append(np.array([int(blk["br_idx"]) if blk["br_idx"] is not None else -1]*i_win.shape[0], dtype=np.int32))
                prov_gidx.append(sel_global.astype(np.int32, copy=False))
                prov_lidx.append(sel_local.astype(np.int32, copy=False))

                out_rates = ALIGNED_OUT / (
                    f"rates_from_curated__UA_{_sanitize(str(port))}"
                    f"__Depth{_sanitize(str(depth_label))}"
                    f"__{blk['intan_session']}__BR_{int(blk['br_idx']) if blk['br_idx'] is not None else -1:03d}.npz"
                )
                np.savez_compressed(
                    out_rates,
                    intan_rate_win=i_win.astype(np.float32),
                    ua_rate_win=u_win.astype(np.float32),
                    rate_t_rel_ms=rel_t_ref.astype(np.float32),
                    event_center_ms=evt_ms_al.astype(np.float32),
                    port=str(port),
                    depth_mm=str(depth_label),
                    intan_session=str(blk["intan_session"]),
                    br_idx=int(blk["br_idx"]) if blk["br_idx"] is not None else -1,
                    dt_ms_target=float(st.get("dt_i_ms", np.nan)),
                    n_events_kept=int(sel_local.size),
                    curated_global_indices=sel_global.astype(int),
                    curated_local_indices=sel_local.astype(int),
                    intan_rates=str(intan_rates_npz),
                    ua_rates=str(ua_rates_npz),
                    align_note="Intan ms aligned by subtracting anchor_ms; UA left on BR ms; UA resampled to Intan grid."
                )
                print(f"[rates] wrote {out_rates}")

            if rate_i_chunks:
                i_all = np.concatenate(rate_i_chunks, axis=0)
                u_all = np.concatenate(rate_u_chunks, axis=0)
                evt_all = np.concatenate(evt_ms_all, axis=0)
                sess_all = np.concatenate(prov_sess, axis=0)
                br_all   = np.concatenate(prov_br,   axis=0)
                gidx_all = np.concatenate(prov_gidx, axis=0)
                lidx_all = np.concatenate(prov_lidx, axis=0)

                out_cat = ALIGNED_OUT / (
                    f"rates_from_curated__UA_{_sanitize(str(port))}"
                    f"__Depth{_sanitize(str(depth_label))}__ALL.npz"
                )
                np.savez_compressed(
                    out_cat,
                    intan_rate_win=i_all.astype(np.float32),
                    ua_rate_win=u_all.astype(np.float32),
                    rate_t_rel_ms=rel_t_ref.astype(np.float32),
                    event_center_ms=evt_all.astype(np.float32),
                    event_session=sess_all,
                    event_br_idx=br_all.astype(np.int32),
                    curated_global_idx=gidx_all.astype(np.int32),
                    curated_local_idx=lidx_all.astype(np.int32),
                    port=str(port),
                    depth_mm=str(depth_label),
                    n_events_total=int(i_all.shape[0]),
                    align_note=("All sessions resampled to a common relative timebase (first session's rate_t_rel_ms). "
                                "Intan ms aligned by subtracting anchor_ms; UA left on BR ms; UA resampled to Intan grid.")
                )
                print(f"[rates][ALL] wrote {out_cat}  "
                      f"(N={i_all.shape[0]}, Intan_ch={i_all.shape[1]}, UA_ch={u_all.shape[1]}, T={i_all.shape[2]})")

if __name__ == "__main__":
    main()
