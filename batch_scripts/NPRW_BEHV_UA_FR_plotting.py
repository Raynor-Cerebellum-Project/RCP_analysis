from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.signal import butter, filtfilt
from types import SimpleNamespace
import numpy as np
import pandas as pd
import re
from scipy.io import loadmat

import RCP_analysis as rcp

import matplotlib
matplotlib.use("Agg")                 # set FIRST
from matplotlib import pyplot as plt  # then import pyplot
matplotlib.rcParams['svg.fonttype'] = 'none'

# ---------- CONFIG ----------
WIN_MS            = (-600.0, 600.0)
NORMALIZE_FIRST_MS = 150.0
MIN_TRIALS        = 1

VMIN_INTAN_BASELINE, VMAX_INTAN_BASELINE = -25, 100
VMIN_UA_BASELINE, VMAX_UA_BASELINE = -25, 100
VMIN_SMA_BASELINE, VMAX_SMA_BASELINE = -10, 35

VMIN_INTAN, VMAX_INTAN = -50, 300
VMIN_UA, VMAX_UA = -50, 150
COLORMAP = "jet"

# --- Velocity setting ---
VEL_THRESH = 5.0          # absolute velocity threshold (a.u./ms) for despiking
VEL_MAX_GAP = 5           # interpolate NaN runs up to this many samples
# --- Visualization settings ---
GAUSS_SMOOTH_MS = 0   # 0 → disable smoothing

# ---- FIGURE LAYOUT KNOBS ----
BEH_RATIO = 0.6               # height ratio for behavior rows (position/velocity); adjust as needed
CH_RATIO_PER_ROW = 0.015      # height ratio per neural channel row (heatmaps)
MIN_HEATMAP_RATIO = 0.6       # minimum ratio so tiny arrays don't vanish
UA_COMPACT_FACTOR = 0.95   # < 1.0 shrinks UA panel heights
INTAN_SCALE      = 0.6   # < 1.0 shrinks Intan height (e.g., 0.6 = 60% of previous)
GAP_BEH_INTAN    = 0.25  # height "ratio" for a spacer row between behavior and Intan
FIG_WIDTH_IN         = 16.0        # ← overall width (inches)
HEIGHT_PER_RATIO_IN  = 4.0         # ← height per unit of `ratios` sum

# ---------- params / roots ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)

SESSION_LOC = (Path(PARAMS.data_root) / Path(PARAMS.location)).resolve()
OUT_BASE  = SESSION_LOC / "results"; OUT_BASE.mkdir(parents=True, exist_ok=True)

# ---------- Impedance loading ----------
UA_IMP_MAX_KOHM = 1000.0        # threshold for excluding UA rows; adjust as needed
EXCLUDE_UA_HIGH_Z = True       # set False to disable masking quickly

# Where the files live; mirror the other script:
IMP_BASE = OUT_BASE.parents[0]  # same as your script (one level above OUT_BASE)
IMP_FILES = {
    "A": IMP_BASE / "Impedances" / "Utah_imp_Bank_A_start",
    "B": IMP_BASE / "Impedances" / "Utah_imp_Bank_B_start",
}

# ---------- figures ----------
FIG_ROOT   = OUT_BASE / "figures"
PERI_FIG   = FIG_ROOT / "peri_stim"
FIG = SimpleNamespace(peri_posvel=PERI_FIG / "posvel_traces")

# ---------- checkpoints / inputs ----------
ALIGNED_ROOT = OUT_BASE / "checkpoints" / "Aligned"
BEHAV_ROOT   = OUT_BASE / "checkpoints" / "behavior" / "baseline_concat"
NPRW_BUNDLES   = OUT_BASE / "bundles" / "NPRW"
METADATA_ROOT = SESSION_LOC / "Metadata"; METADATA_ROOT.mkdir(parents=True, exist_ok=True)
METADATA_CSV  = METADATA_ROOT / f"{Path(PARAMS.session)}_metadata.csv"

# UA mapping (only load if path exists and probes provided)
XLS = rcp.ua_excel_path(REPO_ROOT, getattr(PARAMS, "probes", {}))
UA_MAP = rcp.load_UA_mapping_from_excel(XLS) if XLS and Path(XLS).exists() else None

GEOM_PATH = (
    Path(PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None) and str(PARAMS.geom_mat_rel).startswith("/")
    else (REPO_ROOT / PARAMS.geom_mat_rel).resolve()
    if getattr(PARAMS, "geom_mat_rel", None)
    else rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
)

# ---------- kinematics ----------
# Tuple order for keypoints (robust to missing config)
KEYPOINTS_ORDER = tuple(PARAMS.kinematics.get("keypoints"))
RATES_DIR = BEHAV_ROOT / "rates_from_curated"

def _pick_rates_all_path(rates_dir: Path, ua_port_choice: str = "A") -> tuple[Path, str]:
    """
    Prefer, in order:
      1) CURATED, port-specific, session-aggregated (no __BR_) 
      2) ALL,     port-specific, session-aggregated (no __BR_)
      5) last ALL of anything (per-BR, etc.)
    """
    def _no_br(lst):
        # keep only files that do NOT have __BR_ in the name
        return [p for p in lst if "__BR_" not in p.name]
    
    cur = sorted(rates_dir.glob(f"rates_from_curated__UA_{ua_port_choice}__Depth*__CURATED.npz"))
    cur = _no_br(cur)
    if cur:
        return cur[-1], "CURATED"
    
    all_port = sorted(rates_dir.glob(f"rates_from_curated__UA_{ua_port_choice}__Depth*__ALL.npz"))
    all_port = _no_br(all_port)
    if all_port:
        return all_port[-1], "ALL"
    return rates_dir / "rates_from_curated__UA_X__DepthY__ALL.npz", "ALL"

RATES_ALL_PATH, _rates_selection_tag = _pick_rates_all_path(RATES_DIR, "A")

_imp_pat_elecnum = re.compile(
    r"\belec\s*\d+\s*-\s*(\d{1,3})\s+([0-9]+(?:\.[0-9]+)?)\s*(k?ohms?|kΩ|ohms?|Ω)\b",
    flags=re.IGNORECASE,
)

def _filter_stims_for_stream(stim_ms: np.ndarray, t_ms: np.ndarray,
                             win_ms: tuple[float, float]) -> np.ndarray:
    """Keep only stims whose [stim+win0, stim+win1] lies fully within t_ms range."""
    if stim_ms is None or t_ms is None or np.size(stim_ms) == 0 or np.size(t_ms) < 2:
        return np.array([], dtype=float)
    t0, t1 = float(t_ms[0]), float(t_ms[-1])
    w0, w1 = float(win_ms[0]), float(win_ms[1])
    mask = (stim_ms + w0 >= t0) & (stim_ms + w1 <= t1)
    return np.asarray(stim_ms, float)[mask]

def _safe_extract_segments(rate_hz, t_ms, stim_ms_in, win_ms, min_trials, normalize_first_ms):
    """Filter to in-range stims, then extract without crashing if 0-kept."""
    st = _filter_stims_for_stream(stim_ms_in, t_ms, win_ms)
    if st.size == 0:
        dt = float(np.nanmedian(np.diff(t_ms))) if np.size(t_ms) > 1 else 1.0
        rel_t = np.arange(win_ms[0], win_ms[1] + 1e-9, dt, dtype=float)
        return None, rel_t, 0
    try:
        segs, rel_t = rcp.extract_peristim_segments(
            rate_hz=rate_hz, t_ms=t_ms, stim_ms=st, win_ms=win_ms, min_trials=min_trials
        )
    except RuntimeError as e:
        if "Only 0 peri-stim segments" in str(e):
            dt = float(np.nanmedian(np.diff(t_ms))) if np.size(t_ms) > 1 else 1.0
            rel_t = np.arange(win_ms[0], win_ms[1] + 1e-9, dt, dtype=float)
            return None, rel_t, 0
        raise
    zeroed = rcp.baseline_zero_each_trial(segs, rel_t, normalize_first_ms=normalize_first_ms)
    med = rcp.median_across_trials(zeroed)
    return med, rel_t, int(segs.shape[0])  # or st.size

def _safe_extract_segments_stat(rate_hz, t_ms, stim_ms_in, win_ms, min_trials, normalize_first_ms, stat: str):
    """
    Like _safe_extract_segments, but returns an across-events statistic ("median" or "var")
    over baseline-zeroed segments (axis=0). Returns (stat_mat, rel_t, n_trials_kept).
    """
    st = _filter_stims_for_stream(stim_ms_in, t_ms, win_ms)
    if st.size == 0:
        dt = float(np.nanmedian(np.diff(t_ms))) if np.size(t_ms) > 1 else 1.0
        rel_t = np.arange(win_ms[0], win_ms[1] + 1e-9, dt, dtype=float)
        return None, rel_t, 0

    try:
        segs, rel_t = rcp.extract_peristim_segments(
            rate_hz=rate_hz, t_ms=t_ms, stim_ms=st, win_ms=win_ms, min_trials=min_trials
        )
    except RuntimeError as e:
        if "Only 0 peri-stim segments" in str(e):
            dt = float(np.nanmedian(np.diff(t_ms))) if np.size(t_ms) > 1 else 1.0
            rel_t = np.arange(win_ms[0], win_ms[1] + 1e-9, dt, dtype=float)
            return None, rel_t, 0
        raise

    zeroed = rcp.baseline_zero_each_trial(segs, rel_t, normalize_first_ms=normalize_first_ms)
    if stat == "median":
        X = rcp.median_across_trials(zeroed)
    elif stat == "var":
        X = rcp.variance_across_trials(zeroed)
    else:
        raise ValueError("stat must be 'median' or 'var'")
    return X, rel_t, int(segs.shape[0])


def _order_rows_by_region_then_peak(
    ua_mat: np.ndarray,
    t_vec: np.ndarray,
    ids_1based: Optional[np.ndarray],
    *,
    region_order=(0, 1, 2, 3),   # SMA, DPM, M1 inf, M1 sup
    pre_only: bool = True,       # search t<0 (or <=0 if include_zero=True)
    include_zero: bool = True,
    peak_mode: str = "max",      # "max" | "min" | "abs"
    earliest_at_top: bool = True # <<< NEW: place earliest rows visually at top
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Sort rows within each region by earliest *time* of peak in the selected window.
    Tie-breaker: larger peak magnitude (by peak_mode).
    Rows with no finite values go to the bottom of their region.
    Unknown-region rows go after known regions, sorted the same way.

    If earliest_at_top=True, each region's block is reversed so that with
    imshow(origin="lower") the earliest peaks appear at the top.
    """
    if ua_mat is None or ua_mat.size == 0 or ids_1based is None:
        return ua_mat, ids_1based

    ua  = np.asarray(ua_mat, float)
    ids = np.asarray(ids_1based).astype(float)  # may contain NaN
    t   = np.asarray(t_vec, float)
    n_rows = ua.shape[0]

    # time mask
    if pre_only:
        mask_t = (t <= 0.0) if include_zero else (t < 0.0)
    else:
        mask_t = np.isfinite(t)
    if not np.any(mask_t):
        mask_t = np.isfinite(t)

    time_ix = np.where(mask_t)[0]

    # region codes
    regs = np.full(n_rows, 1_000_000, int)
    for r in range(n_rows):
        eid = ids[r]
        if np.isfinite(eid):
            regs[r] = _ua_region_code_from_elec(int(eid))

    # choose peak fn
    if peak_mode == "min":
        val_fn = np.nanmin
        arg_fn = np.nanargmin
        mag_fn = lambda x: -x  # more negative = “larger magnitude” for tie-break
    elif peak_mode == "abs":
        val_fn = lambda x: np.nanmax(np.abs(x))
        def arg_fn(x):
            ax = np.abs(x)
            return int(np.nanargmax(ax))
        mag_fn = lambda x: np.abs(x)
    else:  # "max"
        val_fn = np.nanmax
        arg_fn = np.nanargmax
        mag_fn = lambda x: x

    # features per row (global time for stability)
    peak_time = np.full(n_rows, np.inf, float)   # earlier is better
    peak_mag  = np.full(n_rows, -np.inf, float)  # larger is better (per mode)

    for r in range(n_rows):
        sub = ua[r, :][mask_t]
        if np.isfinite(sub).any():
            try:
                pv = val_fn(sub)
                peak_mag[r] = float(pv) if np.isfinite(pv) else -np.inf
                local_ix = arg_fn(sub)
                global_ix = time_ix[int(local_ix)]
                peak_time[r] = float(t[global_ix])
            except Exception:
                pass  # leave defaults

    def sort_block(ix: np.ndarray) -> np.ndarray:
        if ix.size == 0:
            return ix
        # primary: earlier time (ascending); secondary: larger magnitude (descending)
        ordered = ix[np.lexsort((-peak_mag[ix], peak_time[ix]))]
        # Put earliest at *top* when drawn with origin="lower"
        return ordered[::-1] if earliest_at_top else ordered

    order_chunks = []
    seen = np.zeros(n_rows, bool)

    # known regions in requested order
    for reg in region_order:
        idxs = np.where(regs == reg)[0]
        if idxs.size:
            ii = sort_block(idxs)
            order_chunks.append(ii)
            seen[ii] = True

    # unknown regions last (apply same top/bottom rule)
    rest = np.where(~seen)[0]
    if rest.size:
        order_chunks.append(sort_block(rest))

    order = np.concatenate(order_chunks) if order_chunks else np.arange(n_rows)
    return ua[order, :], (ids[order] if ids_1based is not None else ids_1based)

def _unit_to_kohm(val: float, unit: str) -> float:
    u = (unit or "").lower()
    if "k" in u:
        return float(val)
    return float(val) / 1000.0  # Ω → kΩ

def _read_text_loose(p: Path) -> str:
    if not p.exists():
        raise FileNotFoundError(str(p))
    b = p.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1", "mac_roman"):
        try:
            return b.decode(enc)
        except Exception:
            pass
    filtered = bytes(ch for ch in b if 9 <= ch <= 126 or ch in (10, 13))
    return filtered.decode("latin-1", errors="ignore")

def _ylabel_horizontal(ax, text: str, pad: float = 8.0):
    # Horizontal y-label on the left; right-aligned to sit close to the axis.
    t = ax.set_ylabel(text, rotation=0, labelpad=pad)
    t.set_horizontalalignment("right")
    t.set_verticalalignment("center")
    return t

def load_impedances_from_textedit_dump(path_like: str | Path) -> dict[int, float]:
    """
    Parse lines like 'elec1-5   201 kOhm' from a loose text dump.
    Returns { Elec#: impedance_kΩ }.
    """
    txt = _read_text_loose(Path(path_like))
    out: dict[int, float] = {}
    for m in _imp_pat_elecnum.finditer(txt):
        elec_str, val_str, unit = m.groups()
        try:
            elec = int(elec_str)
            val = float(val_str)
        except Exception:
            continue
        out[elec] = _unit_to_kohm(val, unit)
    return out

def _apply_relative_offsets_by_ref(
    lines: Optional[np.ndarray],
    labels: list[str],
    keypoints=KEYPOINTS_ORDER,
    ref_kp: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    For position lines (K,T), add a constant offset per trace so that:
      reference_kp_{x,y} → offset 0,
      other kp_{x}       → + (median(kp_x) - median(reference_kp_x)),
      other kp_{y}       → + (median(kp_y) - median(reference_kp_y)).
    If ref_kp is None, the first keypoint from `keypoints` that appears in labels is used.
    """
    if lines is None or not isinstance(lines, np.ndarray) or lines.ndim != 2 or lines.size == 0:
        return lines
    if not labels:
        return lines

    L = lines.copy()
    lab_lc = [str(s).lower() for s in labels]

    def find_idx(sought: str) -> Optional[int]:
        soughtrl = sought.lower()
        for i, s in enumerate(lab_lc):
            if s == soughtrl or s.endswith(soughtrl):
                return i
        return None

    # choose reference keypoint
    if ref_kp is None:
        for kp in keypoints:
            if find_idx(f"{kp}_x") is not None or find_idx(f"{kp}_y") is not None:
                ref_kp = kp
                break
    if ref_kp is None:
        return L  # nothing matched

    # reference medians
    irx = find_idx(f"{ref_kp}_x")
    iry = find_idx(f"{ref_kp}_y")
    med_rx = np.nanmedian(L[irx]) if irx is not None else None
    med_ry = np.nanmedian(L[iry]) if iry is not None else None

    # shift others relative to reference (per axis)
    for kp in keypoints:
        if kp == ref_kp:
            continue
        ix = find_idx(f"{kp}_x")
        if ix is not None and med_rx is not None:
            med_kx = np.nanmedian(L[ix])
            L[ix, :] = L[ix, :] + (med_kx - med_rx)
        iy = find_idx(f"{kp}_y")
        if iy is not None and med_ry is not None:
            med_ky = np.nanmedian(L[iy])
            L[iy, :] = L[iy, :] + (med_ky - med_ry)

    return L

def _gaussian_fir_coeffs(sigma_ms: float, dt_ms: float, truncate: float = 4.0) -> np.ndarray:
    """
    Build a discrete Gaussian FIR kernel (normalized) with std = sigma_ms,
    truncated at +/- truncate*sigma, length odd.
    """
    if not np.isfinite(sigma_ms) or sigma_ms <= 0 or not np.isfinite(dt_ms) or dt_ms <= 0:
        return np.array([1.0], float)
    sigma_samp = float(sigma_ms) / float(dt_ms)
    # ensure at least ~1 sample of width to avoid degeneracy
    sigma_samp = max(sigma_samp, 0.5)
    half = max(1, int(np.ceil(truncate * sigma_samp)))
    k = np.arange(-half, half + 1, dtype=float)
    g = np.exp(-0.5 * (k / sigma_samp) ** 2)
    g /= g.sum()
    return g

def _filtfilt_gaussian_nanaware_1d(y: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    NaN-aware zero-phase Gaussian smoothing using filtfilt with FIR 'b' (a=[1]).
    """
    y = np.asarray(y, float)
    if y.size == 0:
        return y
    if b.ndim != 1 or b.size == 0:
        return y
    # filtfilt needs minimum length; for FIR, ~3*len(b) is a safe guard
    min_len = max(9, 3 * b.size)

    # If too short, just return as-is (no smoothing)
    if y.size < min_len:
        return y.copy()

    mask = ~np.isfinite(y)
    if mask.all():
        return y.copy()

    filled = y.copy()
    filled[mask] = np.nanmedian(filled)

    # Zero-phase filtering with FIR kernel
    out = filtfilt(b, [1.0], filled, method="gust")
    out[mask] = np.nan
    return out

def _smooth_lines_gaussian_filtfilt(lines: np.ndarray, rel_t_ms: np.ndarray, sigma_ms: float) -> np.ndarray:
    """
    Apply filtfilt-based Gaussian smoothing along time to (K,T) or (T,) arrays.
    """
    if lines is None or np.size(lines) == 0 or sigma_ms <= 0.0:
        return lines
    L = np.asarray(lines, float)
    one_d = (L.ndim == 1)
    if one_d:
        L = L[None, :]

    dt_ms = float(np.nanmedian(np.diff(rel_t_ms))) if rel_t_ms.size > 1 else 1.0
    b = _gaussian_fir_coeffs(sigma_ms, dt_ms)

    out = L.copy()
    for k in range(out.shape[0]):
        out[k, :] = _filtfilt_gaussian_nanaware_1d(out[k, :], b)
    return out[0] if one_d else out

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

# --- tiny guard for very short vectors / weird dt ---
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

def _ua_region_code_from_elec(e: int) -> int:
    """Return 0..3 for the 4 UA regions (SMA, DPM, M1 inf, M1 sup). Unknown -> large code."""
    e = int(e)
    if   1  <= e <=  64:  return 0  # SMA
    elif 65 <= e <= 128:  return 1  # Dorsal premotor
    elif 129<= e <= 192:  return 2  # M1 inferior
    elif 193<= e <= 256:  return 3  # M1 superior
    return 1_000_000

def _simple_beh_labels(names: list[str],
                       keypoints: Tuple[str, ...] = KEYPOINTS_ORDER) -> list[str]:
    """
    Map raw DLC-style names like
      'DLC_Resnet50_..._wrist_x'  -> 'wrist_x'
      'cam0_elbow_y'              -> 'elbow_y'
    If a keypoint/axis can’t be found, fall back to the original name.
    """
    out: list[str] = []
    kps_lc = tuple(kp.lower() for kp in keypoints)
    for n in names:
        s = str(n).lower()
        kp = next((kp for kp in kps_lc if kp in s), None)
        axis = "x" if ("_x" in s or s.endswith("x")) else (
                "y" if ("_y" in s or s.endswith("y")) else None)
        if kp and axis:
            out.append(f"{kp}_{axis}")
        else:
            out.append(str(n))
    return out

def _baseline_zero_trials(wins: np.ndarray, rel_t_ms: np.ndarray, normalize_first_ms: float) -> np.ndarray:
    """
    wins: (N_events, K_traces, T) — baseline-zero per event/trace using
          [rel_t_ms[0], normalize_first_ms].
    """
    if wins.size == 0:
        return wins
    bl_mask = (rel_t_ms >= rel_t_ms[0]) & (rel_t_ms <= float(normalize_first_ms))
    if not np.any(bl_mask):
        cut = max(1, int(round(0.1 * rel_t_ms.size)))  # fallback: first 10%
        bl_mask = np.zeros_like(rel_t_ms, dtype=bool); bl_mask[:cut] = True
    wins = wins.astype(float, copy=True)
    bl = np.nanmedian(wins[:, :, bl_mask], axis=2, keepdims=True)  # (N,K,1)
    return wins - bl

def _find_baseline_cam_npz(root: Path, port: str, depth: str, cam: int) -> Path | None:
    """
    Prefer curated → non-curated. Accept legacy naming without _camX as last fallback.
    """
    port = str(port).upper().strip()
    depth = str(depth).strip()
    # most specific first
    cands = [
        root / f"baseline_UA_{port}_Depth{depth}_cam{cam}_curated.npz",
        root / f"baseline_UA_{port}_Depth{depth}_cam{cam}.npz",
        root / f"baseline_UA_{port}_Depth{depth}.npz",  # legacy single-cam
    ]
    for p in cands:
        if p.exists():
            return p
    # last resort: newest file for that port/depth & cam
    globs = list(root.glob(f"baseline_UA_{port}_Depth{depth}*cam{cam}*.npz"))
    return max(globs, key=lambda p: p.stat().st_mtime) if globs else None

def _load_baseline_cam(npz_path):
    if not npz_path or not Path(npz_path).exists():
        return None
    with np.load(npz_path, allow_pickle=True) as z:
        wins = z["windows"].astype(float) if "windows" in z.files else np.zeros((0,0,0), float)
        t_ms = z["t_ms"].astype(float) if "t_ms" in z.files else np.arange(0.0)
        labels = [str(x) for x in (z["labels"].tolist() if "labels" in z.files else [f"trace_{i}" for i in range(wins.shape[1])])]
    if not wins.size or not t_ms.size:
        return None
    wins0 = _baseline_zero_trials(wins, t_ms, NORMALIZE_FIRST_MS)
    wins0_filt, v_wins, _ = butter_lowpass_pos_and_vel_3d(wins0, t_ms, cutoff_hz=10.0, order=3)
    pos_med = np.nanmedian(wins0_filt, axis=0)
    vel_med = np.nanmedian(v_wins, axis=0)
    labels_s = _simple_beh_labels(labels, KEYPOINTS_ORDER)
    return dict(pos=pos_med, vel=vel_med, t=t_ms, labels=labels_s)

def main_baselines():
    """
    Baseline figure styled like stacked_heatmaps_plus_behv (no probe):
      [Kinematics median lines] + [Intan heatmap] + [UA heatmap]
    """
    # ---- load ALL (concatenated) curated rate windows ----
    if not RATES_ALL_PATH.exists():
        print(f"[baseline] ALL rates file not found: {RATES_ALL_PATH}")
        return

    with np.load(RATES_ALL_PATH, allow_pickle=True) as z:
        intan_win = z["intan_rate_win"].astype(float) if "intan_rate_win" in z.files else np.zeros((0,0,0), float)
        ua_win    = z["ua_rate_win"].astype(float)    if "ua_rate_win"    in z.files else np.zeros((0,0,0), float)
        intan_t     = z["intan_t_rel_ms"].astype(float)  if "intan_t_rel_ms"  in z.files else np.arange(0.0)
        ua_t     = z["ua_t_rel_ms"].astype(float)  if "ua_t_rel_ms"  in z.files else np.arange(0.0)
        # keep for later lookups
        all_dir   = RATES_ALL_PATH.parent
        port_lbl  = str(z.get("port", "B"))
        depth_lbl = str(z.get("depth_mm", "43"))
    KIN0 = _find_baseline_cam_npz(BEHAV_ROOT, port_lbl, depth_lbl, cam=0)
    KIN1 = _find_baseline_cam_npz(BEHAV_ROOT, port_lbl, depth_lbl, cam=1)

    cam0 = _load_baseline_cam(KIN0) if KIN0 else None
    cam1 = _load_baseline_cam(KIN1) if KIN1 else None

    # legacy single-file fallback if neither cam file exists
    if not cam0 and not cam1:
        legacy = _find_baseline_cam_npz(BEHAV_ROOT, port_lbl, depth_lbl, cam=0)  # will also return legacy path
        if legacy and legacy.name.endswith(".npz") and "_cam" not in legacy.stem:
            solo = _load_baseline_cam(legacy)
            cam0 = solo
        
    n_ev_i = intan_win.shape[0] if intan_win.ndim == 3 else 0
    n_ev_u = ua_win.shape[0]    if ua_win.ndim    == 3 else 0
    if n_ev_i == 0 and n_ev_u == 0:
        print("[baseline] No events in ALL rates file — nothing to plot.")
        return

    # ---- baseline-zero per event/channel; then median across events → (n_ch, T) ----
    intan_med = None; ua_med = None
    # ---- baseline-zero per event/channel; then median across events ----
    if n_ev_i:
        i_zero = rcp.baseline_zero_each_trial(intan_win, intan_t, normalize_first_ms=NORMALIZE_FIRST_MS)
        intan_med = np.nanmedian(i_zero, axis=0)
    if n_ev_u:
        u_zero = rcp.baseline_zero_each_trial(ua_win,    ua_t, normalize_first_ms=NORMALIZE_FIRST_MS)
        ua_med = np.nanmedian(u_zero, axis=0)
        
    intan_var = None; ua_var = None
    if n_ev_i:
        # you already have i_zero above
        intan_var = np.nanvar(i_zero, axis=0)
    if n_ev_u:
        ua_var = np.nanvar(u_zero, axis=0)
            
    # ---- load curated kinematics windows (median lines) ----
    beh_labels: list[str] = []

    # ---- UA region sorting (optional) ----
    ua_plot = ua_med
    ids_plot = None
    ua_groups_baseline = []
    
    if ua_med is not None:
        # --- UA row→electrode IDs (simple: newest __ALL.npz for this port/depth) ---
        ua_ids_1based = None
        try:
            pat_base = f"rates_from_curated__UA_{port_lbl}__Depth{depth_lbl}__*__ALL.npz"
            cands = sorted(all_dir.glob(pat_base))
            if cands:
                chosen = cands[-1]  # newest
                with np.load(chosen, allow_pickle=True) as zc:
                    # try direct arrays first
                    for k in ("ua_row_to_elec", "ua_ids_1based", "row_to_elec", "ua_electrodes"):
                        if k in zc.files:
                            ua_ids_1based = np.asarray(zc[k], dtype=int).ravel()
                            break
                    # fall back: follow pointer to per-session UA file (if present)
                    if ua_ids_1based is None and "ua_rates" in zc.files:
                        ua_rates_path = Path(str(zc["ua_rates"]))
                        if ua_rates_path.exists():
                            with np.load(ua_rates_path, allow_pickle=True) as zr:
                                if "meta" in zr.files:
                                    meta_u = zr["meta"].item()
                                    for k in ("row_to_elec", "ua_row_to_elec", "ua_ids_1based", "ua_electrodes"):
                                        if k in meta_u:
                                            ua_ids_1based = np.asarray(meta_u[k], dtype=int).ravel()
                                            break
                                if ua_ids_1based is None:
                                    for k in ("ua_row_to_elec", "ua_ids_1based", "row_to_elec", "ua_electrodes"):
                                        if k in zr.files:
                                            ua_ids_1based = np.asarray(zr[k], dtype=int).ravel()
                                            break
            else:
                print(f"[baseline][warn] No UA curated ALL files found for port={port_lbl}, depth={depth_lbl}")
        except Exception as e:
            print(f"[baseline][warn] UA mapping discovery failed: {e}")

        # sanity check to avoid misalignment
        if ua_ids_1based is not None and ua_ids_1based.size != ua_med.shape[0]:
            print(f"[baseline][warn] ua_ids_1based len={ua_ids_1based.size} != UA rows={ua_med.shape[0]} — ignoring")
            ua_ids_1based = None

        # prepare for plotting/sorting
        ua_plot = ua_med
        ids_plot = ua_ids_1based
        if ua_ids_1based is not None:
            ids = np.asarray(ua_ids_1based, int)
            valid = ids > 0
            regs = np.array([_ua_region_code_from_elec(int(e)) for e in ids], int)
            order_valid = np.lexsort((ids[valid], regs[valid]))
            order = np.r_[np.where(valid)[0][order_valid], np.where(~valid)[0]]
            ua_plot = ua_med[order, :]
            ids_plot = ids[order]
            
    # ---- Impedance-based exclusion for BASELINE UA panel ----
    if EXCLUDE_UA_HIGH_Z and ua_plot is not None and ids_plot is not None:
        # load impedances the same way as your IR-aligned script
        def _try_load_imp(p: Path, tag: str):
            if not p:
                return
            if not p.exists():
                print(f"[warn] UA impedance file not found for port {tag}: {p}")
                return
            try:
                d = load_impedances_from_textedit_dump(p)
                imp_by_elec.update(d)
                print(f"[info] Parsed {len(d)} UA impedances from {p.name} (port {tag}).")
            except Exception as e:
                print(f"[warn] could not parse UA impedances from {p} (port {tag}): {e}")

        imp_by_elec: dict[int, float] = {}
        # Use the port from the rates file label; also try the other side
        _port = (port_lbl or "A").strip().upper()
        _try_load_imp(IMP_FILES.get(_port), _port)
        _other = "B" if _port == "A" else "A"
        _try_load_imp(IMP_FILES.get(_other), _other)

        # build keep mask: drop only rows whose impedance is known AND > threshold
        try:
            ids_arr = np.asarray(ids_plot)
            keep = np.ones(ua_plot.shape[0], dtype=bool)
            any_imp = False
            for r, eid in enumerate(ids_arr):
                if eid is None or not np.isfinite(eid):
                    continue
                z = imp_by_elec.get(int(eid))
                if z is not None and np.isfinite(z):
                    any_imp = True
                    if z > UA_IMP_MAX_KOHM:
                        keep[r] = False
            if any_imp:
                if keep.any():
                    ua_plot = ua_plot[keep, :]
                    ids_plot = ids_arr[keep]
                    print(f"[info] Baseline UA: kept {keep.sum()}/{keep.size} rows (≤ {UA_IMP_MAX_KOHM:g} kΩ).")

                    try:
                        ua_plot, ids_plot = _order_rows_by_region_then_peak(
                            ua_plot, ua_t, ids_plot,
                            pre_only=True, include_zero=True
                        )
                    except Exception as e:
                        print(f"[warn] Baseline UA: resort failed after mask: {e}")
                else:
                    print("[warn] Baseline UA: impedance mask would remove all rows; skipping mask.")
        except Exception as e:
            print(f"[warn] Baseline UA impedance mask failed: {e}")
        # ---- Split UA rows into region groups for baseline plotting ----
        if ua_plot is not None:
            ids_arr = np.asarray(ids_plot) if ids_plot is not None else np.full(ua_plot.shape[0], np.nan)
            regs = np.array(
                [_ua_region_code_from_elec(int(e)) if np.isfinite(e) else 1_000_000 for e in ids_arr],
                dtype=int
            )

            def _append_group(mask: np.ndarray, label: str):
                if mask.size and np.count_nonzero(mask):
                    ua_groups_baseline.append(
                        {"mat": ua_plot[mask, :], "ids": (ids_arr[mask] if ids_plot is not None else None), "label": label}
                    )

            # M1 group = inferior(2) + superior(3), PMd=1, SMA=0
            _append_group((regs == 2) | (regs == 3), "M1i+M1s")
            _append_group((regs == 1), "PMd")
            _append_group((regs == 0), "SMA")

    # If we haven't built groups yet, make a single-panel default:
    if ua_plot is not None and not ua_groups_baseline:
        ua_groups_baseline = [{"mat": ua_plot, "ids": ids_plot, "label": "UA (all)"}]

    # ---- Build UA groups for VARIANCE (robustly align to baseline rows) ----
    ua_groups_baseline_var = []
    if ua_var is not None:
        ua_plot_var  = ua_var
        ids_plot_var = None

        # If we have baseline-kept ids (ids_plot) and also know the original
        # per-row ids matching ua_var (ua_ids_1based), use them to align rows.
        try:
            # ids_plot: ids AFTER baseline sorting + impedance mask
            # ua_ids_1based: original ids BEFORE masking; must match ua_var rows
            if (ids_plot is not None and ua_ids_1based is not None and
                ua_var.shape[0] == np.asarray(ua_ids_1based).size):

                base_ids = np.asarray(ua_ids_1based, dtype=int).ravel()
                # map electrode id -> original row index in ua_var
                idx_map = {int(e): i for i, e in enumerate(base_ids) if e > 0 and np.isfinite(e)}
                # build row index list following ids_plot order; drop any unknowns
                row_idx = [idx_map.get(int(e), None) for e in np.asarray(ids_plot).ravel()
                        if e is not None and np.isfinite(e)]
                row_idx = [i for i in row_idx if i is not None]

                if row_idx and len(row_idx) == len(ids_plot):
                    ua_plot_var  = ua_var[row_idx, :]
                    ids_plot_var = np.asarray(ids_plot, dtype=int).ravel()
                else:
                    print("[warn] UA VAR: could not perfectly align to baseline-kept ids; using UA(all).")
            else:
                print("[warn] UA VAR: missing ids for alignment; using UA(all).")
        except Exception as e:
            print(f"[warn] UA VAR: alignment failed ({e}); using UA(all).")

        # If we successfully aligned, we can split by region; else show all
        if ids_plot_var is not None and ua_plot_var.shape[0] == ids_plot_var.size:
            ids_arr = ids_plot_var
            regs = np.array([_ua_region_code_from_elec(int(e)) if np.isfinite(e) else 1_000_000
                            for e in ids_arr], dtype=int)

            def _append_group_var(mask, label):
                if mask.size and np.count_nonzero(mask):
                    ua_groups_baseline_var.append({
                        "mat": ua_plot_var[mask, :],
                        "ids": ids_arr[mask],
                        "label": label
                    })

            _append_group_var((regs == 2) | (regs == 3), "M1i+M1s")
            _append_group_var((regs == 1), "PMd")
            _append_group_var((regs == 0), "SMA")
            if not ua_groups_baseline_var:
                ua_groups_baseline_var = [{"mat": ua_plot_var, "ids": ids_arr, "label": "UA (all)"}]
        else:
            ua_groups_baseline_var = [{"mat": ua_plot_var, "ids": None, "label": "UA (all)"}]

    # ---- helper that renders one combined figure (behavior + Intan + UA) ----
    def _render_and_save_baseline(
        beh_time0, beh_pos0, beh_vel0, *,
        title_beh0,
        out_dir,
        beh_time1=None, beh_pos1=None, beh_vel1=None,
        title_beh1=None,
        beh_labels: list[str] = None,
        ua_groups: list[dict] = None,
        intan_mat=None,
        intan_title:str="",
        intan_cb_label:str="Δ FR (Hz)",
        ua_vrange_override:dict|None=None,  # optional per-group vrange
        intan_vmin: float | None = None,
        intan_vmax: float | None = None,
        suffix:str=""                   # filename suffix
    ):
        import matplotlib.gridspec as gridspec
        beh_labels = beh_labels or []
        ua_groups = ua_groups or []

        have_pos0 = isinstance(beh_pos0, np.ndarray) and beh_pos0.ndim == 2 and beh_pos0.size > 0
        have_vel0 = isinstance(beh_vel0, np.ndarray) and beh_vel0.ndim == 2 and beh_vel0.size > 0
        have_pos1 = isinstance(beh_pos1, np.ndarray) and beh_pos1.ndim == 2 and beh_pos1.size > 0
        have_vel1 = isinstance(beh_vel1, np.ndarray) and beh_vel1.ndim == 2 and beh_vel1.size > 0

        # ---- kinematics row order: cam0_pos, cam1_pos, cam0_vel, cam1_vel ----
        ratios = []
        row_kinds = []
        def _append_beh(kind):
            ratios.append(BEH_RATIO); row_kinds.append(("beh", kind))

        if have_pos0: _append_beh("cam0_pos")
        if have_pos1: _append_beh("cam1_pos")
        if have_vel0: _append_beh("cam0_vel")
        if have_vel1: _append_beh("cam1_vel")

        # Spacer before Intan if any behavior rows exist
        has_intan = (intan_mat is not None)
        if has_intan and any(k[0] == "beh" for k in row_kinds):
            ratios.append(GAP_BEH_INTAN); row_kinds.append(("gap", None))

        # Intan
        if has_intan:
            intan_ratio = max(MIN_HEATMAP_RATIO, CH_RATIO_PER_ROW * intan_mat.shape[0]) * float(INTAN_SCALE)
            ratios.append(intan_ratio); row_kinds.append(("intan", None))

        # UA groups
        if ua_groups:
            for g in ua_groups:
                mat_g = g["mat"]
                label_g = g.get("label", "")
                rows = (mat_g.shape[0] if mat_g is not None else 0)
                base = max(MIN_HEATMAP_RATIO, CH_RATIO_PER_ROW * rows)
                scale = 0.5 if (("pmd" in label_g.lower()) or ("sma" in label_g.lower())) else 1.0
                ratios.append(base * scale * UA_COMPACT_FACTOR)
                row_kinds.append(("ua", label_g))

        nrows = len(ratios)
        if nrows == 0:
            print("[baseline] nothing to plot.")
            return

        fig = plt.figure(
            figsize=(FIG_WIDTH_IN, HEIGHT_PER_RATIO_IN * sum(ratios)),
            constrained_layout=False  # or True if you prefer auto spacing
        )

        gs = gridspec.GridSpec(
            nrows=nrows, ncols=2, figure=fig,
            width_ratios=[1.0, 0.02], height_ratios=ratios, hspace=0.18, wspace=0.05
        )

        # Helper with a legend toggle
        def _plot_lines(ax, t, lines, labels, title, ylabel, show_legend: bool):
            if not (isinstance(lines, np.ndarray) and lines.ndim == 2 and lines.size > 0):
                ax.axis("off"); return
            K = lines.shape[0]
            for i in range(K):
                lab = labels[i] if i < len(labels) else f"trace_{i+1}"
                ax.plot(t, lines[i], lw=1.2, alpha=0.95, label=lab)
            ax.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2, ls="--")
            _ylabel_horizontal(ax, ylabel)
            if title:  # only set when non-empty
                ax.set_title(title)
            if show_legend and K:
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                        frameon=False, fontsize=9, ncols=1, borderaxespad=0.0)
            ax.grid(alpha=0.15, linestyle=":")


        row = 0
        ua_axes = []
        events_label = f"{_rates_selection_tag.lower()} events"

        # We'll count behavior panels as we render; show legend only on the 3rd one.
        beh_plot_count = 0
        total_beh_rows = sum(1 for k in row_kinds if k[0] == "beh")

        for kind, subkind in row_kinds:
            if kind == "beh":
                beh_plot_count += 1
                show_legend = (beh_plot_count == 3)  # legend only on the third kinematics row

                # Determine camera/time and which data to plot
                if subkind in ("cam0_pos", "cam0_vel"):
                    cam_label = "Cam-0"
                    t_use     = beh_time0
                else:
                    cam_label = "Cam-1"
                    t_use     = beh_time1

                if subkind.endswith("_pos"):
                    metric_label = "\n Position Δ (a.u.)"
                    y_label      = f"{cam_label} {metric_label}"
                    lines_use    = beh_pos0 if subkind == "cam0_pos" else beh_pos1
                else:
                    metric_label = "\n Velocity (a.u./ms)"
                    y_label      = f"{cam_label} {metric_label}"
                    lines_use    = beh_vel0 if subkind == "cam0_vel" else beh_vel1

                # Only the FIRST kinematics row gets a title; others blank
                if beh_plot_count == 1:
                    if subkind.startswith("cam0"):
                        base_title = (title_beh0 or "Cam-0 Kinematics")
                    else:
                        base_title = (title_beh1 or "Cam-1 Kinematics")
                    title_txt = base_title  # no “— Position/Velocity” suffix; y-axis already says that
                else:
                    title_txt = ""

                ax = fig.add_subplot(gs[row, 0])
                _plot_lines(ax, t_use, lines_use, beh_labels, title_txt, y_label, show_legend)

                # Hide x tick labels unless this is the LAST behavior row
                if beh_plot_count < total_beh_rows:
                    ax.set_xlabel(""); ax.tick_params(axis="x", labelbottom=False)
                ax._is_time_axis = True
                row += 1
                continue

            if kind == "gap":
                ax_gap = fig.add_subplot(gs[row, 0])
                ax_gap.axis("off")
                row += 1
                continue

            if kind == "intan":
                # Intan row
                ax_i     = fig.add_subplot(gs[row, 0])
                ax_i_cax = fig.add_subplot(gs[row, 1])

                vmin_i = VMIN_INTAN_BASELINE if intan_vmin is None else float(intan_vmin)
                vmax_i = VMAX_INTAN_BASELINE if intan_vmax is None else float(intan_vmax)
                im0 = ax_i.imshow(
                    intan_mat, aspect="auto", cmap=COLORMAP, origin="lower",
                    extent=[intan_t[0], intan_t[-1], 0, intan_mat.shape[0]],
                    vmin=vmin_i, vmax=vmax_i
                )
                
                ax_i.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2, ls="--")
                title_txt = (intan_title or
                        f"No stim. Neural Activity (median Δ across {n_ev_i} {events_label})")
                ax_i.set_title(title_txt)
                ax_i.set_xlabel(""); ax_i.tick_params(axis="x", labelbottom=False)
                _ylabel_horizontal(ax_i, f"IP {intan_mat.shape[0]} chs")

                ax_i_cax.set_xticks([]); ax_i_cax.set_yticks([])
                ax_i._is_time_axis = True

                # create the colorbar in the dedicated cax
                cb0 = fig.colorbar(im0, cax=ax_i_cax)
                # cb0 = fig.colorbar(im0, cax=ax_i_cax, ticks = ticker.MultipleLocator(20))
                cb0.solids.set_edgecolor('face')
                cb0.ax.tick_params(width=1.2, labelsize=9)
                cb0.set_label(intan_cb_label)
        
                row += 1
                continue

            if kind == "ua":
                label_g = subkind
                g = ua_groups[len(ua_axes)]
                mat_g = g["mat"]
                ax_u     = fig.add_subplot(gs[row, 0])
                ax_u_cax = fig.add_subplot(gs[row, 1])
                # pick vrange per group
                vmin_g, vmax_g = VMIN_UA_BASELINE, VMAX_UA_BASELINE
                if ua_vrange_override and label_g in ua_vrange_override:
                    vmin_g, vmax_g = ua_vrange_override[label_g]

                im1 = ax_u.imshow(
                    mat_g, aspect="auto", cmap=COLORMAP, origin="lower",
                    extent=[ua_t[0], ua_t[-1], 0, mat_g.shape[0]],
                    vmin=vmin_g, vmax=vmax_g
                )

                ax_u.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2, ls="--")
                _ylabel_horizontal(ax_u, f"{label_g} {mat_g.shape[0]} chs")
                ax_u.set_yticks([]); ax_u.tick_params(left=False, labelleft=False)
                ax_u._is_time_axis = True
                
                if (len(ua_axes) == len(ua_groups) - 1):
                    ax_u.set_xlabel("Time (ms) rel. stim")
                else:
                    ax_u.set_xlabel(""); ax_u.tick_params(axis="x", labelbottom=False)

                cb1 = fig.colorbar(im1, cax=ax_u_cax)
                try: cb1.solids.set_edgecolor('face')
                except: pass
                cb1.ax.tick_params(width=1.2, labelsize=9)
                cb1.outline.set_linewidth(1.0)
                # cb1.set_label("Δ FR (Hz)")

                ua_axes.append(ax_u)
                row += 1
                continue

        # Sync x-lims
        for ax in [a for a in fig.axes if isinstance(a, plt.Axes)]:
            if getattr(ax, "_is_time_axis", False):
                ax.set_xlim(*WIN_MS)
        fig.suptitle("Baseline", fontsize=13, fontweight="bold", y=0.995, va="top")
        out_path = out_dir / f"Baseline_UA_{port_lbl}__Depth{depth_lbl}__{_rates_selection_tag}{suffix}.png"
        fig.subplots_adjust(top=0.96)           # pull axes up toward the top
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        print(f"[baseline] Wrote {out_path}")

    # Shared labels (for legends)
    if cam0 and cam0["labels"]:
        beh_labels = cam0["labels"]
    elif cam1 and cam1["labels"]:
        beh_labels = cam1["labels"]
    else:
        beh_labels = []

    # POSITION + VELOCITY (overlayed) — SINGLE FIGURE
    pos0 = _smooth_lines_gaussian_filtfilt(cam0["pos"], cam0["t"], GAUSS_SMOOTH_MS) if cam0 else None
    pos1 = _smooth_lines_gaussian_filtfilt(cam1["pos"], cam1["t"], GAUSS_SMOOTH_MS) if cam1 else None
    if pos0 is not None:
        pos0 = _apply_relative_offsets_by_ref(pos0, beh_labels, KEYPOINTS_ORDER)
    if pos1 is not None:
        pos1 = _apply_relative_offsets_by_ref(pos1, beh_labels, KEYPOINTS_ORDER)

    vel0 = _smooth_lines_gaussian_filtfilt(cam0["vel"], cam0["t"], GAUSS_SMOOTH_MS) if cam0 else None
    vel1 = _smooth_lines_gaussian_filtfilt(cam1["vel"], cam1["t"], GAUSS_SMOOTH_MS) if cam1 else None

    if (pos0 is not None) or (pos1 is not None):
        _render_and_save_baseline(
            cam0["t"] if cam0 else (cam1["t"] if cam1 else None),
            pos0, vel0,
            title_beh0=(f"Kinematics / Referenced to first {int(NORMALIZE_FIRST_MS)} ms"
                        + (f" • Gaussian (filtfilt) σ={GAUSS_SMOOTH_MS:g} ms" if GAUSS_SMOOTH_MS > 0 else "")) if cam0 else "",
            out_dir=FIG.peri_posvel,
            beh_time1=(cam1["t"] if cam1 else None),
            beh_pos1=pos1, beh_vel1=vel1,
            title_beh1=(f""),
            intan_mat=intan_med,
            ua_vrange_override={"M1i+M1s": (VMIN_UA_BASELINE, VMAX_UA_BASELINE), "PMd": (VMIN_UA_BASELINE, VMAX_UA_BASELINE), "SMA": (VMIN_SMA_BASELINE, VMAX_SMA_BASELINE)},
            intan_vmin=VMIN_INTAN_BASELINE, intan_vmax=VMAX_INTAN_BASELINE,
            beh_labels=beh_labels,
            ua_groups=ua_groups_baseline
        )
    else:
        _render_and_save_baseline(
            None, None, None, title_beh0="",
            out_dir=FIG.peri_posvel,
            beh_labels=[], ua_groups=ua_groups_baseline
    )
    if (pos0 is not None) or (pos1 is not None) or (intan_var is not None) or (ua_groups_baseline_var):
        _render_and_save_baseline(
            cam0["t"] if cam0 else (cam1["t"] if cam1 else None),
            pos0, vel0,
            title_beh0=(f"Kinematics / Referenced to first {int(NORMALIZE_FIRST_MS)} ms"
                        + (f" • Gaussian (filtfilt) σ={GAUSS_SMOOTH_MS:g} ms" if GAUSS_SMOOTH_MS > 0 else "")) if cam0 else "",
            out_dir=FIG.peri_posvel,
            beh_time1=(cam1["t"] if cam1 else None),
            beh_pos1=pos1, beh_vel1=vel1,
            title_beh1="",
            beh_labels=beh_labels,
            ua_groups=ua_groups_baseline_var,
            intan_mat=intan_var,
            intan_title=f"No stim. Neural Activity VAR (across {n_ev_i} {_rates_selection_tag.lower()} events)",
            intan_cb_label="Var(Δ FR) (Hz²)",
            ua_vrange_override={"M1i+M1s": (0.0, 10000.0), "PMd": (0.0, 10000.0), "SMA": (0.0, 5000.0)},
            intan_vmin=0.0, intan_vmax=20000.0,
            suffix="__VAR",
        )

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)

def _median_behavior_line(series_on_common: np.ndarray,
                          t_common_ms: np.ndarray,
                          stim_ms: np.ndarray,
                          win_ms: Tuple[float,float],
                          baseline_ms: float,
                          min_trials: int) -> Tuple[np.ndarray, np.ndarray, int]:
    s = np.asarray(series_on_common, float)

    try:
        segs, rel_t = rcp.extract_peristim_segments(
            s[None, :], t_common_ms, stim_ms, win_ms=win_ms, min_trials=min_trials
        )
    except RuntimeError as e:
        # Typical message: "Only 0 peri-stim segments available ..."
        if "Only 0 peri-stim segments" in str(e):
            # Build a reasonable rel_t and return NaNs
            dt = np.nanmedian(np.diff(t_common_ms)) if t_common_ms.size > 1 else 1.0
            rel_t = np.arange(win_ms[0], win_ms[1] + 1e-9, dt, dtype=float)
            return np.full(rel_t.size, np.nan, float), rel_t, 0
        else:
            raise

    if segs.size == 0:
        dt = np.nanmedian(np.diff(t_common_ms)) if t_common_ms.size > 1 else 1.0
        rel_t = np.arange(win_ms[0], win_ms[1] + 1e-9, dt, dtype=float)
        return np.full(rel_t.size, np.nan, float), rel_t, 0

    segs = segs[:, 0, :]  # (n_trials, T)

    bl_mask = (rel_t >= rel_t[0]) & (rel_t <= baseline_ms)
    if not bl_mask.any():
        step = max(1, int(round(baseline_ms / (np.nanmedian(np.diff(rel_t)) if rel_t.size > 1 else 1.0))))
        bl_mask = np.zeros_like(rel_t, bool); bl_mask[:step] = True

    keep = np.isfinite(segs[:, bl_mask]).any(axis=1)
    n_kept = int(np.count_nonzero(keep))
    if n_kept == 0:
        return np.full(segs.shape[1], np.nan, float), rel_t, 0

    segs = segs[keep]
    segs = segs - np.nanmedian(segs[:, bl_mask], axis=1, keepdims=True)
    line = np.nanmedian(segs, axis=0)
    return line, rel_t, n_kept

def _median_lines_for_columns(series_on_common: np.ndarray,
                              t_common_ms: np.ndarray,
                              stim_ms: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    if series_on_common is None or np.size(series_on_common) == 0:
        return np.zeros((0, 0), float), np.array([], float), 0
    if series_on_common.ndim == 1:
        series_on_common = series_on_common[:, None]

    T, D = series_on_common.shape
    if D == 0:
        return np.zeros((0, 0), float), np.array([], float), 0

    lines: list[np.ndarray] = []
    rel_t_out: Optional[np.ndarray] = None
    kept_counts: list[int] = []

    for d in range(D):
        line, rel_t, n_kept = _median_behavior_line(series_on_common[:, d],
                                                    t_common_ms, stim_ms,
                                                    WIN_MS, NORMALIZE_FIRST_MS, MIN_TRIALS)
        lines.append(line)
        kept_counts.append(int(n_kept))
        if rel_t_out is None:
            rel_t_out = rel_t

    if rel_t_out is None or rel_t_out.size == 0:
        dt = np.nanmedian(np.diff(t_common_ms)) if t_common_ms.size > 1 else 1.0
        rel_t_out = np.arange(WIN_MS[0], WIN_MS[1] + 1e-9, dt, dtype=float)

    n_trials_used = int(max(kept_counts)) if kept_counts else 0
    return (np.vstack(lines) if len(lines) else np.zeros((0, rel_t_out.size))), rel_t_out, n_trials_used


# ---------- Behavior (already mapped to common grid in the NPZ) ----------
def _ordered_xy_indices(cam_cols: List[str],
                        keypoints: Tuple[str, ...] = KEYPOINTS_ORDER
                        ) -> Tuple[List[int], List[str]]:
    idx: List[int] = []
    names: List[str] = []
    cols_lc = [c.lower() for c in cam_cols]

    def _clean_kp(s: str) -> str:
        s = str(s).strip()
        s = s.strip("[]").strip("'\"")   # handle "'Wrist'" → Wrist
        return s.lower()

    def _find_xy_for(kp: str) -> Tuple[Optional[int], Optional[int]]:
        kp_lc = _clean_kp(kp).lower()
        ix = next((i for i, c in enumerate(cols_lc) if c.endswith("_x") and kp_lc in c), None)
        iy = next((i for i, c in enumerate(cols_lc) if c.endswith("_y") and kp_lc in c), None)
        return ix, iy

    for kp in keypoints:
        ix, iy = _find_xy_for(kp)
        base = _clean_kp(kp)
        if ix is None or iy is None:
            idx.extend([-1, -1])
            names.extend([f"{base}_x(MISSING)", f"{base}_y(MISSING)"])
        else:
            idx.extend([ix, iy])
            names.extend([f"{base}_x", f"{base}_y"])
    return idx, names

def _strip_cam_prefix(n: str) -> str:
    # Turn "cam0_wrist_x" -> "wrist_x", "cam1_middle_finger_tip_y" -> "middle_finger_tip_y"
    n = str(n)
    if n.startswith("cam0_"):
        return n[len("cam0_"):]
    if n.startswith("cam1_"):
        return n[len("cam1_"):]
    return n

def _select_matrix(cam: np.ndarray, cols: List[str],
                   keypoints: Tuple[str, ...] = KEYPOINTS_ORDER
                   ) -> Tuple[np.ndarray, List[str]]:
    idx, names = _ordered_xy_indices(cols or [], keypoints=keypoints)

    if cam is None or cam.size == 0:
        M = np.zeros((0, len(idx)), float)
        return M, names

    if cam.ndim == 1:
        cam = cam[:, None]

    # Guard against column index overflow if CSV is malformed
    n_rows, n_cols = cam.shape
    M = np.empty((n_rows, len(idx)), float)
    for k, j in enumerate(idx):
        if j == -1 or j >= n_cols:
            M[:, k] = np.nan
        else:
            M[:, k] = cam[:, j]
    return M, names

def _z_per_column(M: np.ndarray) -> np.ndarray:
    if M is None or M.size == 0:
        return np.asarray(M, dtype=float)
    if M.ndim == 1:
        M = M[:, None]
    out = M.astype(float, copy=True)
    if out.shape[1] == 0:
        return out
    for j in range(out.shape[1]):
        col = out[:, j]
        m = np.nanmean(col) if np.isfinite(col).any() else np.nan
        s = np.nanstd(col)  if np.isfinite(col).any() else np.nan
        if np.isfinite(s) and s > 0:
            out[:, j] = (col - m) / s
        else:
            out[:, j] = np.full_like(col, np.nan)
    return out

def _aggregate_columns_to_bins(idx_bins: np.ndarray,
                               values_2d: np.ndarray,
                               n_bins: int) -> np.ndarray:
    """
    idx_bins: (N,) target bin per sample (-1 to ignore)
    values_2d: (N, D) per-sample values (D can be 0)
    returns: (n_bins, D) mean per bin (NaN where no samples)
    """
    # Normalize shapes
    if values_2d is None:
        values_2d = np.zeros((0, 0), float)
    if values_2d.ndim == 1:
        values_2d = values_2d[:, None]
    N = values_2d.shape[0]
    D = values_2d.shape[1]
    if n_bins <= 0:
        return np.zeros((0, D), float)
    if idx_bins is None or idx_bins.size == 0 or N == 0:
        return np.full((n_bins, D), np.nan, float)

    # If mismatch in lengths, truncate to the common length (robust)
    L = min(len(idx_bins), N)
    idx_bins = idx_bins[:L]
    values_2d = values_2d[:L, :]

    out = np.full((n_bins, D), np.nan, float)
    if D == 0:
        return out

    valid = (idx_bins >= 0) & (idx_bins < n_bins)
    if not np.any(valid):
        return out

    for d in range(D):
        vals = values_2d[:, d]
        ok = valid & np.isfinite(vals)
        if not np.any(ok):
            continue
        counts = np.zeros(n_bins, dtype=np.int32)
        sums = np.zeros(n_bins, dtype=float)
        np.add.at(counts, idx_bins[ok], 1)
        np.add.at(sums,   idx_bins[ok], vals[ok])
        hit = counts > 0
        out[hit, d] = sums[hit] / counts[hit]
    return out


# delay parse
def _parse_delay_ms(val) -> int:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0
    s = str(val).strip()
    if not s or s.casefold() == "random":
        return 0
    try:
        return int(float(s))
    except Exception:
        return 0

def _fmt_num(x) -> str:
    if pd.isna(x):
        return "n/a"
    try:
        v = float(x); return f"{int(v)}" if v.is_integer() else f"{v:g}"
    except Exception:
        s = str(x).strip(); return s if s else "n/a"

def _interp_nans_1d_gentle(x: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """Fill only interior NaN runs up to max_gap by linear interp. 
    Leading/trailing NaNs stay NaN; large gaps stay NaN."""
    x = np.asarray(x, float)
    if x.ndim != 1 or x.size == 0:
        return x
    out = x.copy()
    isn = ~np.isfinite(out)
    if not isn.any():
        return out

    starts = np.where(isn & ~np.r_[False, isn[:-1]])[0]
    ends   = np.where(isn & ~np.r_[isn[1:],  False])[0]
    for s, e in zip(starts, ends):
        gap = e - s + 1
        left, right = s - 1, e + 1
        if (gap <= max_gap and left >= 0 and right < out.size 
                and np.isfinite(out[left]) and np.isfinite(out[right])):
            out[s:e+1] = np.interp(np.arange(s, e+1), [left, right], [out[left], out[right]])
    return out

def _interp_nans_2d_by_col(arr: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """Apply _interp_nans_1d_gentle to each column of a 2D array."""
    arr = np.asarray(arr, float)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    out = arr.copy()
    for j in range(out.shape[1]):
        out[:, j] = _interp_nans_1d_gentle(out[:, j], max_gap=max_gap)
    return out

def _read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Try several common encodings and tolerate a few bad rows.
    """
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1", "mac_roman")
    last_err = None
    for enc in encodings:
        try:
            # engine='python' + on_bad_lines='skip' is more forgiving with odd quotes
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
        except Exception as e:
            last_err = e
            continue
    # final fallback: read bytes → decode as latin-1 (never fails) → StringIO
    try:
        from io import StringIO
        txt = Path(path).read_bytes().decode("latin-1", errors="ignore")
        return pd.read_csv(StringIO(txt), engine="python", on_bad_lines="skip")
    except Exception:
        raise last_err or RuntimeError(f"Could not read CSV: {path}")

def build_title_from_csv(
    csv_path: Path, *, sess: Optional[str] = None, br_file: Optional[int] = None
) -> Tuple[str, Optional[int]]:
    df_raw = _read_csv_robust(csv_path)

    # normalize column names: lower, strip, spaces->underscores
    df = df_raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # drop the first row if it's a header-like row (your original iloc[1:])
    df_data = df.iloc[1:].reset_index(drop=True) if len(df) > 1 else df.copy()
    if df_data.empty:
        return "Condition: n/a, n/a Hz, n/a µA, n/a mm, n/a ms, Delay: 0 ms", None

    def col(name: str) -> Optional[str]:
        n = name.lower()
        return n if n in df_data.columns else None

    # columns we might use
    c_session   = col("session")
    c_brfile    = col("br_file")
    c_uaport    = col("ua_port")
    c_trigger   = col("movement_trigger")
    c_freq      = col("stim_frequency_hz")
    c_current   = col("current_ua")
    c_depth     = col("depth_mm")
    c_duration  = col("stim_duration_ms")
    c_delay     = col("delay")

    # ---- build masks stepwise (recording each stage) ----
    mask = pd.Series(True, index=df_data.index)

    # prefer matching BR_File exactly (if provided and column exists)
    if br_file is not None and c_brfile:
        mask_br = (pd.to_numeric(df_data[c_brfile], errors="coerce") == int(br_file))
        if mask_br.any():
            mask &= mask_br

    # then try session (substring match)
    if c_session and sess:
        m_sess = df_data[c_session].astype(str).str.contains(str(sess), regex=False, na=False)
        if (mask & m_sess).any():
            mask &= m_sess

    # prefer UA_port == 'A' if it doesn’t eliminate all rows
    if c_uaport:
        m_portA = df_data[c_uaport].astype(str).str.upper().str.strip().eq("A")
        if (mask & m_portA).any():
            mask &= m_portA

    # avoid Movement_Trigger == 'velocity' when possible
    if c_trigger:
        m_not_vel = ~df_data[c_trigger].astype(str).str.strip().str.casefold().eq("velocity")
        if (mask & m_not_vel).any():
            mask &= m_not_vel

    # if still empty, relax progressively (order: remove trigger, then port, then session, then BR)
    if not mask.any():
        mask = pd.Series(True, index=df_data.index)
        if br_file is not None and c_brfile:
            mask_br = (pd.to_numeric(df_data[c_brfile], errors="coerce") == int(br_file))
            mask &= mask_br if mask_br.any() else True
        if c_session and sess:
            m_sess = df_data[c_session].astype(str).str.contains(str(sess), regex=False, na=False)
            mask &= m_sess if m_sess.any() else True
        if c_uaport:
            m_portA = df_data[c_uaport].astype(str).str.upper().str.strip().eq("A")
            mask &= m_portA if m_portA.any() else True
        if c_trigger:
            m_not_vel = ~df_data[c_trigger].astype(str).str.strip().str.casefold().eq("velocity")
            mask &= m_not_vel if m_not_vel.any() else True
        if not mask.any():
            mask = pd.Series(True, index=df_data.index)  # final fallback: any row

    row = df_data.loc[mask].iloc[0]

    delay_ms = _parse_delay_ms(row.get(c_delay, None) if c_delay else None)

    parts = []
    if br_file is not None:
        parts.append(f"Condition: {_fmt_num(br_file)}")
    parts.extend([
        f"Freq: {_fmt_num(row.get(c_freq))} Hz"      if c_freq else "Freq: n/a Hz",
        f"Current: {_fmt_num(row.get(c_current))} µA" if c_current else "Current: n/a µA",
        f"Depth: {_fmt_num(row.get(c_depth))} mm"     if c_depth else "Depth: n/a mm",
        f"Duration: {_fmt_num(row.get(c_duration))} ms" if c_duration else "Duration: n/a ms",
        f"Delay: {delay_ms} ms",
    ])
    if c_uaport:
        parts.append(f"UA Port: {_fmt_num(row.get(c_uaport))}")

    # ---- robust video_file extraction ----
    # try multiple candidate columns and parse the first integer
    video_cols = [c for c in ("video_file", "video", "video_index", "video#", "vid", "videoid") if c in df_data.columns]
    video_file: Optional[int] = None
    for c in video_cols:
        val = row.get(c)
        if pd.isna(val):
            continue
        # numeric first
        num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
        if pd.notna(num):
            video_file = int(num)
            break
        # otherwise, pull the first integer in the string
        m = re.search(r"\d+", str(val))
        if m:
            video_file = int(m.group(0))
            break

    return ", ".join(parts), video_file

def main():
    mat_probe = loadmat(Path(GEOM_PATH))
    intan_geom = {}
    intan_geom["x"] = mat_probe["xcoords"].ravel()
    intan_geom["y"] = mat_probe["ycoords"].ravel()
    assert intan_geom["x"].size == intan_geom["y"].size, "x/y must have same length"
    if "chanMap0ind" in mat_probe: # 0-based device mapping if present
        intan_probe_mapping = intan_geom["device_index_0based"] = mat_probe["chanMap0ind"].ravel()
    else:
        raise ValueError("No 0-based chanmap in .mat geometry file.")
    if intan_probe_mapping.size != intan_geom["x"].size:
        raise ValueError("device_index_0based length != #contacts")
    
    # Build ProbeInterface Probe
    nprw_probe = Probe(ndim=2)
    nprw_probe.set_contacts(positions=np.c_[intan_geom["x"], intan_geom["y"]], shapes="square", shape_params={"width": 12.0})
    nprw_probe.set_device_channel_indices(intan_probe_mapping )# Apply mapping
    
    probe.set_device_channel_indices(np.arange(probe.get_contact_count(), dtype=int))
    locs  = probe.contact_positions.astype(float)         # (n_ch, 2)
    
    files = sorted(ALIGNED_ROOT.glob("aligned__*.npz"))
    if not files:
        raise SystemExit(f"[error] No combined aligned NPZs found at {ALIGNED_ROOT}")

    for k, file in enumerate(files):
        if k < 2:
            continue
        NPRW_rate, NPRW_t, UA_rate, UA_t, stim_ms_abs, meta = rcp.load_combined_npz(file)
        # Try to get per-row UA electrode IDs (1..256), + NSP mapping (1..128/256)
        ua_ids_1based = None

        with np.load(file, allow_pickle=True) as z:
            if "ua_row_to_elec" in z.files:
                ua_ids_1based = np.asarray(z["ua_row_to_elec"], dtype=int).ravel()
            else:
                # legacy fallbacks just in case
                for key in ("ua_ids_1based", "ua_elec_per_row", "ua_electrodes", "ua_chan_ids_1based"):
                    if key in z.files:
                        ua_ids_1based = np.asarray(z[key], dtype=int).ravel()
                        break

        # sanity: ensure length matches UA rows (plot uses row order)
        if ua_ids_1based is not None and ua_ids_1based.size != UA_rate.shape[0]:
            print(f"[warn] ua_row_to_elec len={ua_ids_1based.size} != UA rows={UA_rate.shape[0]} — ignoring")
            ua_ids_1based = None

        sess = meta.get("session", file.stem)
        br_idx = meta.get("br_idx", file.stem)

        # stim times aligned to Intan aligned timebase
        stim_ms = rcp.aligned_stim_ms(stim_ms_abs, meta)

        # ---------- Behavior (already mapped to common grid in the NPZ) ----------
        with np.load(file, allow_pickle=True) as z:
            intan_t = z["intan_t_ms_aligned"].astype(float)  if "intan_t_ms_aligned"  in z.files else np.arange(0.0)
            cam0      = z.get("beh_cam0", np.zeros((0,0), np.float32)).astype(float)
            cam1      = z.get("beh_cam1", np.zeros((0,0), np.float32)).astype(float)
            raw0 = z.get("beh_cam0_cols", None)
            raw1 = z.get("beh_cam1_cols", None)
            cam0_cols = [str(x) for x in _as_list(raw0)]
            cam1_cols = [str(x) for x in _as_list(raw1)]

        
        # ---- interpolate here (gentle NaN interpolation only) ----
        if cam0.size:
            cam0 = _interp_nans_2d_by_col(cam0, max_gap=4)
        if cam1.size:
            cam1 = _interp_nans_2d_by_col(cam1, max_gap=4)

        cam0_M, cam0_names = _select_matrix(cam0, cam0_cols)
        cam1_M, cam1_names = _select_matrix(cam1, cam1_cols)

        # z-score each column independently
        cam0_M = _z_per_column(cam0_M) if cam0_M.size else cam0_M
        cam1_M = _z_per_column(cam1_M) if cam1_M.size else cam1_M

        # Behavior already on Intan grid → no binning needed
        cam0_on_grid = cam0_M
        cam1_on_grid = cam1_M

        # >>> Low-pass filter position and compute velocity (on-grid) <<<
        cam0_pos_filt, cam0_vel, _ = butter_lowpass_pos_and_vel(cam0_on_grid, intan_t, cutoff_hz=10.0, order=3)
        cam1_pos_filt, cam1_vel, _ = butter_lowpass_pos_and_vel(cam1_on_grid, intan_t, cutoff_hz=10.0, order=3)

        # Keep only stim times with a full window inside the behavior timeline
        if intan_t.size >= 2:
            dt_ms = float(np.nanmedian(np.diff(intan_t)))
        else:
            dt_ms = 1.0  # fallback

        t0, t1 = float(intan_t[0]), float(intan_t[-1])
        left_ok  = stim_ms + WIN_MS[0] >= t0
        right_ok = stim_ms + WIN_MS[1] <= t1
        stim_ms_beh = stim_ms[left_ok & right_ok]

        # Now build the peri-stim median *position* lines from the filtered positions:
        cam0_lines, beh_rel_t, cam0_ntr = _median_lines_for_columns(cam0_pos_filt, intan_t, stim_ms_beh)
        cam1_lines, _,          cam1_ntr = _median_lines_for_columns(cam1_pos_filt, intan_t, stim_ms_beh)

        cam0_v_lines, beh_rel_t_v, cam0_v_ntr = _median_lines_for_columns(cam0_vel, intan_t, stim_ms_beh)
        cam1_v_lines, _,             cam1_v_ntr = _median_lines_for_columns(cam1_vel, intan_t, stim_ms_beh)

        # Position traces
        if cam0_lines.size:
            cam0_lines = _smooth_lines_gaussian_filtfilt(cam0_lines, beh_rel_t, GAUSS_SMOOTH_MS)
        if cam1_lines.size:
            cam1_lines = _smooth_lines_gaussian_filtfilt(cam1_lines, beh_rel_t, GAUSS_SMOOTH_MS)

        # Velocity traces
        if cam0_v_lines.size:
            cam0_v_lines = _smooth_lines_gaussian_filtfilt(cam0_v_lines, beh_rel_t_v, GAUSS_SMOOTH_MS)
        if cam1_v_lines.size:
            cam1_v_lines = _smooth_lines_gaussian_filtfilt(cam1_v_lines, beh_rel_t_v, GAUSS_SMOOTH_MS)

        try:
            overall_title, video_file = build_title_from_csv(METADATA_CSV, br_file=meta.get("br_idx"))
        except Exception as e:
            print(f"[warn] metadata parse failed: {e}")
            overall_title, video_file = ("Condition: n/a, n/a Hz, n/a µA, n/a mm, n/a ms, Delay: 0 ms", None)

        # Decide availability
        have_cam0 = isinstance(cam0_lines, np.ndarray) and cam0_lines.ndim == 2 and cam0_lines.size > 0
        have_cam1 = isinstance(cam1_lines, np.ndarray) and cam1_lines.ndim == 2 and cam1_lines.size > 0

        # counts for display
        beh_ntr = cam0_ntr if (have_cam0 and not have_cam1) else (
                cam1_ntr if (have_cam1 and not have_cam0) else max(cam0_ntr, cam1_ntr))

        title_kinematics = f"Kinematics / Referenced to first {int(NORMALIZE_FIRST_MS)} ms (n={beh_ntr} trials)"
        # Derive behavior labels from whichever camera has names, but simplify them
        if len(cam0_names) > 0:
            beh_labels = _simple_beh_labels([_strip_cam_prefix(n) for n in cam0_names], KEYPOINTS_ORDER)
        elif len(cam1_names) > 0:
            beh_labels = _simple_beh_labels([_strip_cam_prefix(n) for n in cam1_names], KEYPOINTS_ORDER)
        else:
            beh_labels = []

        # --- NPRW ---
        NPRW_med, rel_time_ms_i, n_nprw = _safe_extract_segments(
            NPRW_rate, NPRW_t, stim_ms, WIN_MS, MIN_TRIALS, NORMALIZE_FIRST_MS
        )
        if NPRW_med is None:
            print(f"[warn] Condition {br_idx}: 0 NPRW peri-stim segments after in-range filtering.")

        # --- UA ---
        UA_med, ua_rel_time_ms, n_ua = _safe_extract_segments(
            UA_rate, UA_t, stim_ms, WIN_MS, MIN_TRIALS, NORMALIZE_FIRST_MS
        )
        NPRW_var, rel_time_ms_i_var, n_nprw_var = _safe_extract_segments_stat(
            NPRW_rate, NPRW_t, stim_ms, WIN_MS, MIN_TRIALS, NORMALIZE_FIRST_MS, stat="var"
        )
        UA_var, ua_rel_time_ms_var, n_ua_var = _safe_extract_segments_stat(
            UA_rate, UA_t, stim_ms, WIN_MS, MIN_TRIALS, NORMALIZE_FIRST_MS, stat="var"
        )
        if UA_med is None:
            print(f"[warn] Condition {br_idx}: 0 UA peri-stim segments after in-range filtering.")
                
        # ---- Impedance-based exclusion for PERI-STIM UA panel ----
        ua_keep_mask = None
        if EXCLUDE_UA_HIGH_Z and UA_med is not None and ua_ids_1based is not None:
            # load impedances (same helper + files as above)
            def _try_load_imp(p: Path, tag: str):
                if not p:
                    return
                if not p.exists():
                    print(f"[warn] UA impedance file not found for port {tag}: {p}")
                    return
                try:
                    d = load_impedances_from_textedit_dump(p)
                    imp_by_elec.update(d)
                    print(f"[info] Parsed {len(d)} UA impedances from {p.name} (port {tag}).")
                except Exception as e:
                    print(f"[warn] could not parse UA impedances from {p} (port {tag}): {e}")

            imp_by_elec: dict[int, float] = {}
            # try to get port from meta; fall back to 'A', also load the other side
            port_from_meta = str(meta.get("port", meta.get("ua_port", "A"))).strip().upper()
            _port = port_from_meta if port_from_meta in ("A", "B") else "A"
            _try_load_imp(IMP_FILES.get(_port), _port)
            _other = "B" if _port == "A" else "A"
            _try_load_imp(IMP_FILES.get(_other), _other)

            try:
                ids_arr = np.asarray(ua_ids_1based).astype(float)
                if ids_arr.size == UA_med.shape[0]:
                    keep = np.ones(ids_arr.size, dtype=bool)
                    any_imp = False
                    for r, eid in enumerate(ids_arr):
                        if not np.isfinite(eid):
                            continue
                        z = imp_by_elec.get(int(eid))
                        if z is not None and np.isfinite(z):
                            any_imp = True
                            if z > UA_IMP_MAX_KOHM:
                                keep[r] = False
                    if any_imp:
                        if keep.any():
                            ua_keep_mask = keep.copy()
                            UA_med = UA_med[keep, :]
                            ua_ids_1based = ua_ids_1based[keep]
                            print(f"[info] Peri UA: kept {keep.sum()}/{keep.size} rows (≤ {UA_IMP_MAX_KOHM:g} kΩ).")
                            try:
                                UA_med, ua_ids_1based = _order_rows_by_region_then_peak(
                                    UA_med, ua_rel_time_ms, ua_ids_1based,
                                    pre_only=True, include_zero=True
                                )
                            except Exception as e:
                                print(f"[warn] Peri UA: resort failed after mask: {e}")
                        else:
                            print("[warn] Peri UA: impedance mask would remove all rows; skipping mask.")
                else:
                    print("[warn] Peri UA: ids length != UA rows; skipping impedance mask.")
            except Exception as e:
                print(f"[warn] Peri UA impedance mask failed: {e}")
        # --- UA: final ordering to MATCH BASELINE (region, then earliest pre-stim peak on top)
        if UA_med is not None and ua_ids_1based is not None:
            try:
                UA_med, ua_ids_1based = _order_rows_by_region_then_peak(
                    UA_med, ua_rel_time_ms, ua_ids_1based,
                    pre_only=True, include_zero=True, peak_mode="max", earliest_at_top=True
                )
            except Exception as e:
                print(f"[warn] Peri UA: final baseline-style ordering failed: {e}")
                
        # Make UA_var rows match UA_med rows / ids
        ua_ids_for_var = ua_ids_1based
        if UA_var is not None and ua_keep_mask is not None and UA_var.shape[0] == ua_keep_mask.size:
            UA_var = UA_var[ua_keep_mask, :]

        # (optional) apply the same final ordering you computed for UA_med
        if UA_var is not None and ua_ids_for_var is not None:
            try:
                UA_var, _ = _order_rows_by_region_then_peak(
                    UA_var, ua_rel_time_ms_var, ua_ids_for_var,
                    pre_only=True, include_zero=True, peak_mode="max", earliest_at_top=True
                )
            except Exception as e:
                print(f"[warn] Peri UA VAR: ordering failed: {e}")
                
        # locate the Intan stim_stream.npz and locate stimulated channels
        stim_npz = NPRW_BUNDLES / f"{sess}_Intan_bundle" / "stim_stream.npz"
        stim_locs = None
        if stim_npz.exists():
            try:
                stim_locs = rcp.detect_stim_channels_from_npz(stim_npz, eps=1e-12, min_edges=1)
            except Exception as e:
                print(f"[warn] stim-site detection failed for {sess}, Condition {br_idx}: {e}")
                
        # ---------------- POSITION plot ----------------
        out_svg_posvel = FIG.peri_posvel / f"Cond_{br_idx}__peri_stim__posvel.png"

        title_NA = (f"Neural Activity (median Δ across {n_nprw} events)")
        beh_time_for_both = beh_rel_t 

        # if nothing to plot at all, skip
        if (NPRW_med is None) and (UA_med is None) and not (have_cam0 or have_cam1):
            print(f"[skip] Condition {br_idx}: no NPRW/UA segments and no behavior traces.")
            continue

        # choose time vectors only if the corresponding matrix exists
        t_intan_for_plot = rel_time_ms_i if NPRW_med is not None else None
        t_ua_for_plot    = ua_rel_time_ms if UA_med is not None else None

        rcp.stacked_heatmaps_plus_behv(
            NPRW_med, UA_med,
            t_intan_for_plot, t_ua_for_plot,
            out_svg_posvel, title_kinematics, title_NA,
            cmap=COLORMAP,
            vmin_intan=VMIN_INTAN, vmax_intan=VMAX_INTAN,
            probe=probe, probe_locs=locs, stim_idx=stim_locs,
            probe_title="NPRW probe (stim sites highlighted)",
            ua_ids_1based=ua_ids_1based,
            ua_sort="none",
            beh_rel_time=beh_time_for_both,
            beh_cam0_lines=cam0_lines,
            beh_cam1_lines=cam1_lines,
            beh_cam0_vel_lines=cam0_v_lines,
            beh_cam1_vel_lines=cam1_v_lines,
            beh_labels=beh_labels,
            title_cam1="",
            title_cam0_vel="",
            title_cam1_vel="",
            sess=sess,
            overall_title=overall_title,
        )
        # --- Variance fig ---
        t_intan_for_plot = rel_time_ms_i_var if NPRW_var is not None else None
        t_ua_for_plot    = ua_rel_time_ms_var if UA_var is not None else None
        
        out_svg_posvel_var = FIG.peri_posvel / f"Cond_{br_idx}__peri_stim__posvel_VAR.png"
        title_NA_var = f"Neural Activity VAR (variance of Δ across {n_nprw_var} events)"
        rcp.stacked_heatmaps_plus_behv(
            NPRW_var, UA_var,
            (rel_time_ms_i_var if NPRW_var is not None else None),
            (ua_rel_time_ms_var if UA_var is not None else None),
            out_svg_posvel_var, title_kinematics, title_NA_var,
            cmap=COLORMAP,
            vmin_intan=0.0, vmax_intan=40000.0,
            vmin_ua={"M1i+M1s": 0.0, "PMd": 0.0, "SMA": 0.0},
            vmax_ua={"M1i+M1s": 10000.0, "PMd": 10000.0, "SMA": 10000.0},
            probe=probe, probe_locs=locs, stim_idx=stim_locs,
            probe_title="NPRW probe (stim sites highlighted)",
            ua_ids_1based=ua_ids_for_var,
            ua_sort="none",
            beh_rel_time=beh_time_for_both,
            beh_cam0_lines=cam0_lines,
            beh_cam1_lines=cam1_lines,
            beh_cam0_vel_lines=cam0_v_lines,
            beh_cam1_vel_lines=cam1_v_lines,
            beh_labels=beh_labels,
            title_cam1="",
            title_cam0_vel="",
            title_cam1_vel="",
            sess=sess,
            overall_title=overall_title,
        )


if __name__ == "__main__":
    # plot the aligned baseline traces
    # main_baselines()
    
    # plot stim trials
    main()