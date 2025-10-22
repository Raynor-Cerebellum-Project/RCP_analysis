from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.signal import butter, filtfilt
from types import SimpleNamespace
import numpy as np
import pandas as pd
import re

import RCP_analysis as rcp

import matplotlib
matplotlib.use("Agg")                 # set FIRST
from matplotlib import pyplot as plt  # then import pyplot
matplotlib.rcParams['svg.fonttype'] = 'none'

# ---------- CONFIG ----------
WIN_MS            = (-600.0, 600.0)
NORMALIZE_FIRST_MS = 150.0
MIN_TRIALS        = 1

VMIN_INTAN_BASELINE, VMAX_INTAN_BASELINE = -50, 100  # keep heatmap range if you like
VMIN_UA_BASELINE, VMAX_UA_BASELINE = -50, 150  # keep heatmap range if you like

VMIN_INTAN, VMAX_INTAN = -50, 300  # keep heatmap range if you like
VMIN_UA, VMAX_UA = -50, 200  # keep heatmap range if you like
COLORMAP = "jet"

# --- Kinematics preprocessing ---
# --- Velocity setting ---
VEL_THRESH = 5.0          # absolute velocity threshold (a.u./ms) for despiking
VEL_MAX_GAP = 5           # interpolate NaN runs up to this many samples
# --- Visualization settings ---
GAUSS_SMOOTH_MS = 0   # 0 → disable smoothing

BOX_ASPECT = 0.312

# ---------- helpers ----------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _subdir(root: Path, *parts: str) -> Path:
    return _ensure_dir(root.joinpath(*parts))

# ---------- params / roots ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
PARAMS    = rcp.load_experiment_params(REPO_ROOT / "config" / "params.yaml", repo_root=REPO_ROOT)

DATA_ROOT = rcp.resolve_data_root(PARAMS)
OUT_BASE  = _ensure_dir(rcp.resolve_output_root(PARAMS))

# ---------- figures ----------
FIG_ROOT   = _subdir(OUT_BASE, "figures")
PERI_FIG   = _subdir(FIG_ROOT, "peri_stim")
BASE_FIG   = _subdir(FIG_ROOT, "baseline_traces")

FIG = SimpleNamespace(
    peri_pos=_subdir(PERI_FIG, "position_traces"),
    peri_vel=_subdir(PERI_FIG, "velocity_traces"),
    base_pos=_subdir(BASE_FIG, "position_traces"),
    base_vel=_subdir(BASE_FIG, "velocity_traces"),
)

# ---------- checkpoints / inputs ----------
CKPT_ROOT    = _subdir(OUT_BASE, "checkpoints")
ALIGNED_ROOT = _subdir(CKPT_ROOT, "Aligned")
BEHAV_ROOT   = _subdir(CKPT_ROOT, "behavior", "baseline_concat")

# ---------- metadata / mapping ----------
# Metadata CSV (derive, create parent if missing)
METADATA_CSV = (DATA_ROOT / (PARAMS.metadata_rel or "")).resolve()
_ensure_dir(METADATA_CSV.parent)

# UA mapping (only load if path exists and probes provided)
XLS = rcp.ua_excel_path(REPO_ROOT, getattr(PARAMS, "probes", {}))
UA_MAP = rcp.load_UA_mapping_from_excel(XLS) if XLS and Path(XLS).exists() else None

# ---------- kinematics ----------
# Tuple order for keypoints (robust to missing config)
KEYPOINTS_ORDER = tuple((PARAMS.kinematics or {}).get("keypoints", ()))
RATES_DIR = BEHAV_ROOT / "rates_from_curated"

def _pick_rates_all_path(rates_dir: Path) -> tuple[Path, str]:
    cur = sorted(rates_dir.glob("rates_from_curated__UA_*__Depth*__CURATED.npz"))
    al  = sorted(rates_dir.glob("rates_from_curated__UA_*__Depth*__ALL.npz"))
    if cur:
        return cur[-1], "CURATED"
    if al:
        return al[-1], "ALL"
    # dummy placeholder; caller will print a friendly error
    return rates_dir / "rates_from_curated__UA_X__DepthY__ALL.npz", "ALL"

RATES_ALL_PATH, _rates_selection_tag = _pick_rates_all_path(RATES_DIR)

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
        axis = "x" if ("_x" in s or s.endswith("x")) else ("y" if ("_y" in s or s.endswith("y")) else None)
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
        rel_t     = z["rate_t_rel_ms"].astype(float)  if "rate_t_rel_ms"  in z.files else np.arange(0.0)
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
        i_zero = rcp.baseline_zero_each_trial(intan_win, rel_t, normalize_first_ms=NORMALIZE_FIRST_MS)
        intan_med = np.nanmedian(i_zero, axis=0)
    if n_ev_u:
        u_zero = rcp.baseline_zero_each_trial(ua_win,    rel_t, normalize_first_ms=NORMALIZE_FIRST_MS)
        ua_med = np.nanmedian(u_zero, axis=0)

    # ---- load curated kinematics windows (median lines) ----
    beh_labels: list[str] = []

    # ---- UA region sorting (optional) ----
    ua_plot = ua_med
    ids_plot = None
    if ua_med is not None:
        # try to discover UA row→electrode IDs from any per-session curated NPZ (for region mapping/sorting)
        ua_ids_1based = None
        try:
            cand = sorted([p for p in all_dir.glob(f"rates_from_curated__UA_{port_lbl}__Depth{depth_lbl}__*.npz")
                           if not p.name.endswith("__ALL.npz")])[0]
            with np.load(cand, allow_pickle=True) as zc:
                ua_rates_path = Path(str(zc["ua_rates"])) if "ua_rates" in zc.files else None
            if ua_rates_path and ua_rates_path.exists():
                with np.load(ua_rates_path, allow_pickle=True) as zr:
                    if "meta" in zr.files:
                        meta = zr["meta"].item()
                        for key in ("row_to_elec", "ua_row_to_elec", "ua_ids_1based", "ua_electrodes"):
                            if key in meta:
                                ua_ids_1based = np.asarray(meta[key], dtype=int).ravel()
                                break
                    if ua_ids_1based is None:
                        for key in ("ua_row_to_elec", "ua_ids_1based", "row_to_elec", "ua_electrodes"):
                            if key in zr.files:
                                ua_ids_1based = np.asarray(zr[key], dtype=int).ravel()
                                break
        except Exception as e:
            print(f"[baseline][warn] UA mapping discovery failed: {e}")

        if ua_ids_1based is not None and ua_ids_1based.size == ua_med.shape[0]:
            ids = np.asarray(ua_ids_1based, int)
            valid = ids > 0
            regs = np.array([_ua_region_code_from_elec(int(e)) for e in ids], int)
            order_valid = np.lexsort((ids[valid], regs[valid]))
            order = np.r_[np.where(valid)[0][order_valid], np.where(~valid)[0]]
            ua_plot = ua_med[order, :]
            ids_plot = ids[order]
        else:
            ua_plot = ua_med
            ids_plot = ua_ids_1based  # may be None

    # ---- helper that renders one combined figure (behavior + Intan + UA) ----
    def _render_and_save_baseline(
        beh_time0, beh_lines0, *,
        title_beh0,
        out_dir, fname_prefix,
        beh_time1=None, beh_lines1=None, title_beh1=None,
        beh_labels: list[str] = None
    ):
        import matplotlib.gridspec as gridspec
        beh_labels = beh_labels or []

        have_beh0 = isinstance(beh_lines0, np.ndarray) and beh_lines0.size > 0
        have_beh1 = isinstance(beh_lines1, np.ndarray) and beh_lines1.size > 0

        nrows = (1 if have_beh0 else 0) + (1 if have_beh1 else 0) \
                + (1 if intan_med is not None else 0) \
                + (1 if ua_plot  is not None else 0)
        if nrows == 0:
            print("[baseline] nothing to plot.")
            return

        height_ratios = []
        if have_beh0: height_ratios.append(1.0)
        if have_beh1: height_ratios.append(1.0)
        if intan_med is not None: height_ratios.append(1.0)
        if ua_plot  is not None:  height_ratios.append(1.0)

        fig = plt.figure(figsize=(14, 3.2 * sum(height_ratios)))
        gs = gridspec.GridSpec(
            nrows=nrows, ncols=2, figure=fig,
            width_ratios=[1.0, 0.04], height_ratios=height_ratios, hspace=0.28, wspace=0.15
        )

        row = 0
        ax_b0 = ax_b1 = ax_i = ax_u = None

        events_label = f"{_rates_selection_tag.lower()} events"  # "curated events" or "all events"

        # Cam-0
        if have_beh0:
            ax_b0 = fig.add_subplot(gs[row, 0])
            K0 = beh_lines0.shape[0]
            for i in range(K0):
                label = beh_labels[i] if i < len(beh_labels) else f"trace_{i+1}"
                ax_b0.plot(beh_time0, beh_lines0[i], lw=1.25, label=label, alpha=0.95)
            ax_b0.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2, ls="--")
            ax_b0.set_ylabel("Δ (a.u.)")
            ax_b0.set_title(title_beh0)
            if K0:
                ax_b0.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9, ncols=1, borderaxespad=0.0)
            ax_b0.grid(alpha=0.15, linestyle=":")
            if BOX_ASPECT is not None:
                ax_b0.set_box_aspect(BOX_ASPECT)   # <<< add this
            row += 1

        # Cam-1
        if have_beh1:
            ax_b1 = fig.add_subplot(gs[row, 0])
            K1 = beh_lines1.shape[0]
            for i in range(K1):
                label = beh_labels[i] if i < len(beh_labels) else f"trace_{i+1}"
                ax_b1.plot(beh_time1, beh_lines1[i], lw=1.25, label=label, alpha=0.95)
            ax_b1.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2, ls="--")
            ax_b1.set_ylabel("Δ (a.u.)")
            ax_b1.set_title(title_beh1 or "Median Kinematics (Cam-1)")
            if K1:
                ax_b1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9, ncols=1, borderaxespad=0.0)
            ax_b1.grid(alpha=0.15, linestyle=":")
            if BOX_ASPECT is not None:
                ax_b1.set_box_aspect(BOX_ASPECT)   # <<< add this
            row += 1

        # Intan
        if intan_med is not None:
            ax_i     = fig.add_subplot(gs[row, 0])
            ax_i_cax = fig.add_subplot(gs[row, 1])
            # Intan
            im0 = ax_i.imshow(
                intan_med, aspect="auto", cmap=COLORMAP, origin="lower",
                extent=[rel_t[0], rel_t[-1], 0, intan_med.shape[0]],
                vmin=VMIN_INTAN_BASELINE, vmax=VMAX_INTAN_BASELINE
            )
            if BOX_ASPECT is not None:
                ax_i.set_box_aspect(BOX_ASPECT)
                
            ax_i.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2, ls="--")
            ax_i.set_title(
                f"Intan (median Δ across {n_ev_i} {events_label}) • Referenced to first {int(NORMALIZE_FIRST_MS)} ms"
            )
            ax_i.set_ylabel("Intan ch")
            cb0 = fig.colorbar(im0, cax=ax_i_cax)
            cb0.set_label("Δ FR (Hz)")
            row += 1

        # UA
        if ua_plot is not None:
            ax_u     = fig.add_subplot(gs[row, 0])
            ax_u_cax = fig.add_subplot(gs[row, 1])
            # UA
            im1 = ax_u.imshow(
                ua_plot, aspect="auto", cmap=COLORMAP, origin="lower",
                extent=[rel_t[0], rel_t[-1], 0, ua_plot.shape[0]],
                vmin=VMIN_UA_BASELINE, vmax=VMAX_UA_BASELINE
            )
            if BOX_ASPECT is not None:
                ax_u.set_box_aspect(BOX_ASPECT)

            ax_u.axvline(0.0, color="Red", alpha=0.8, linewidth=1.2, ls="--")
            ax_u.set_title(
                f"Utah Array (median Δ across {n_ev_u} {events_label}) • Referenced to first {int(NORMALIZE_FIRST_MS)} ms"
            )
            ax_u.set_xlabel("Time (ms) rel. stim")
            ax_u.set_ylabel(""); ax_u.set_yticks([]); ax_u.tick_params(left=False, labelleft=False)

            rcp.add_ua_region_bar(ax_u, ua_plot.shape[0], ua_chan_ids_1based=ids_plot)
            cb1 = fig.colorbar(im1, cax=ax_u_cax)
            cb1.set_label("Δ FR (Hz) NOTE SCALE")

        # Sync x-lims
        for ax in (ax for ax in (ax_b0, ax_b1, ax_i, ax_u) if ax is not None):
            ax.set_xlim(*WIN_MS)

        # save (unchanged aside from filename)
        out_path = out_dir / f"UA_{port_lbl}__Depth{depth_lbl}__{_rates_selection_tag}_baseline_{fname_prefix}.png"
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

    # POSITION
    pos0 = _smooth_lines_gaussian_filtfilt(cam0["pos"], cam0["t"], GAUSS_SMOOTH_MS) if cam0 else None
    pos1 = _smooth_lines_gaussian_filtfilt(cam1["pos"], cam1["t"], GAUSS_SMOOTH_MS) if cam1 else None
    if pos0 is not None:
        pos0 = _apply_relative_offsets_by_ref(pos0, beh_labels, KEYPOINTS_ORDER)
    if pos1 is not None:
        pos1 = _apply_relative_offsets_by_ref(pos1, beh_labels, KEYPOINTS_ORDER)

    if pos0 is not None or pos1 is not None:
        _render_and_save_baseline(
            cam0["t"] if cam0 else (cam1["t"] if cam1 else None),
            pos0,
            title_beh0=(f"Kinematics (Cam-0) • Referenced to first {int(NORMALIZE_FIRST_MS)} ms"
                        + (f" • Gaussian (filtfilt) σ={GAUSS_SMOOTH_MS:g} ms" if GAUSS_SMOOTH_MS > 0 else "")) if cam0 else "",
            out_dir=FIG.base_pos, fname_prefix="rates_with_kinematics_position",
            beh_time1=(cam1["t"] if cam1 else None),
            beh_lines1=pos1,
            title_beh1=(f"Kinematics (Cam-1) • Referenced to first {int(NORMALIZE_FIRST_MS)} ms"
                        + (f" • Gaussian (filtfilt) σ={GAUSS_SMOOTH_MS:g} ms" if GAUSS_SMOOTH_MS > 0 else "")) if cam1 else None,
            beh_labels=beh_labels
        )
    else:
        _render_and_save_baseline(None, None, title_beh0="", out_dir=FIG.base_pos,
                                fname_prefix="rates_with_kinematics_position", beh_labels=[])

    # VELOCITY
    vel0 = _smooth_lines_gaussian_filtfilt(cam0["vel"], cam0["t"], GAUSS_SMOOTH_MS) if cam0 else None
    vel1 = _smooth_lines_gaussian_filtfilt(cam1["vel"], cam1["t"], GAUSS_SMOOTH_MS) if cam1 else None
    if vel0 is not None or vel1 is not None:
        _render_and_save_baseline(
            cam0["t"] if cam0 else (cam1["t"] if cam1 else None),
            vel0,
            title_beh0=("Kinematics Velocity (Cam-0) • Low-pass (Butterworth) then central diff"
                        + (f" • Gaussian (filtfilt) σ={GAUSS_SMOOTH_MS:g} ms" if GAUSS_SMOOTH_MS > 0 else "")) if cam0 else "",
            out_dir=FIG.base_vel, fname_prefix="rates_with_kinematics_velocity",
            beh_time1=(cam1["t"] if cam1 else None),
            beh_lines1=vel1,
            title_beh1=("Kinematics Velocity (Cam-1) • Low-pass (Butterworth) then central diff"
                        + (f" • Gaussian (filtfilt) σ={GAUSS_SMOOTH_MS:g} ms" if GAUSS_SMOOTH_MS > 0 else "")) if cam1 else None,
            beh_labels=beh_labels
        )
    else:
        _render_and_save_baseline(None, None, title_beh0="", out_dir=FIG.base_vel,
                                fname_prefix="rates_with_kinematics_velocity", beh_labels=[])

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

    segs, rel_t = rcp.extract_peristim_segments(
        s[None, :], t_common_ms, stim_ms, win_ms=win_ms, min_trials=min_trials
    )
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

# ---------- Behavior (already mapped to common grid in the NPZ) ----------
def _ordered_xy_indices(cam_cols: List[str],
                        keypoints: Tuple[str, ...] = KEYPOINTS_ORDER
                        ) -> Tuple[List[int], List[str]]:
    """
    Build indices for [kp1_x, kp1_y, kp2_x, kp2_y, ...] in the order from `keypoints`.
    Returns (indices, col_names_in_that_order). Missing coords get -1 and "(MISSING)" names.
    """
    idx: List[int] = []
    names: List[str] = []
    cols_lc = [c.lower() for c in cam_cols]

    def _find_xy_for(kp: str) -> Tuple[Optional[int], Optional[int]]:
        kp_lc = kp.lower()
        ix = next((i for i, c in enumerate(cols_lc) if c.endswith("_x") and kp_lc in c), None)
        iy = next((i for i, c in enumerate(cols_lc) if c.endswith("_y") and kp_lc in c), None)
        return ix, iy

    for kp in keypoints:
        ix, iy = _find_xy_for(kp)
        if ix is None or iy is None:
            idx.extend([-1, -1])
            names.extend([f"{kp}_x(MISSING)", f"{kp}_y(MISSING)"])
        else:
            idx.extend([ix, iy])
            names.extend([cam_cols[ix], cam_cols[iy]])
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
    # Intan probe (optional, only used if you later want a probe panel like NPRW)
    try:
        GEOM_PATH = (
            Path(PARAMS.geom_mat_rel).resolve()
            if getattr(PARAMS, "geom_mat_rel", None) and str(PARAMS.geom_mat_rel).startswith("/")
            else (REPO_ROOT / PARAMS.geom_mat_rel).resolve()
            if getattr(PARAMS, "geom_mat_rel", None)
            else rcp.resolve_probe_geom_path(PARAMS, REPO_ROOT, session_key=None)
        )
        probe, locs = rcp.build_probe_and_locs_from_geom(GEOM_PATH)
    except Exception:
        probe, locs = None, None
    files = sorted(ALIGNED_ROOT.glob("aligned__*.npz"))
    if not files:
        raise SystemExit(f"[error] No combined aligned NPZs found at {ALIGNED_ROOT}")

    for file in files:
        try:
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

            # stim times aligned to Intan aligned timebase
            stim_ms = rcp.aligned_stim_ms(stim_ms_abs, meta)

            # ---------- Behavior (already mapped to common grid in the NPZ) ----------
            with np.load(file, allow_pickle=True) as z:
                common_t = z["intan_t_ms_aligned"].astype(float)
                beh_idx   = z.get("beh_common_idx", np.array([], dtype=np.int64)).astype(np.int64)
                beh_valid = z.get("beh_common_valid", np.array([], dtype=bool)).astype(bool)
                cam0      = z.get("beh_cam0", np.zeros((0,0), np.float32)).astype(float)
                cam1      = z.get("beh_cam1", np.zeros((0,0), np.float32)).astype(float)
                raw0 = z.get("beh_cam0_cols", None)
                raw1 = z.get("beh_cam1_cols", None)
                cam0_cols = [str(x) for x in _as_list(raw0)]
                cam1_cols = [str(x) for x in _as_list(raw1)]

            # ---- interpolate here (gentle, respects invalid rows) ----
            if cam0.size:
                cam0 = cam0.copy()
                # do not “invent” values on invalid rows
                if beh_valid.size >= cam0.shape[0]:
                    cam0[~beh_valid[:cam0.shape[0]], :] = np.nan
                cam0 = _interp_nans_2d_by_col(cam0, max_gap=4)

            if cam1.size:
                cam1 = cam1.copy()
                if beh_valid.size >= cam1.shape[0]:
                    cam1[~beh_valid[:cam1.shape[0]], :] = np.nan
                cam1 = _interp_nans_2d_by_col(cam1, max_gap=4)

            cam0_M, cam0_names = _select_matrix(cam0, cam0_cols)
            cam1_M, cam1_names = _select_matrix(cam1, cam1_cols)

            # z-score each column independently
            cam0_M = _z_per_column(cam0_M) if cam0_M.size else cam0_M
            cam1_M = _z_per_column(cam1_M) if cam1_M.size else cam1_M

            # Map samples to bins using precomputed indices; mask out-of-tolerance
            bin_idx = np.where(beh_valid, beh_idx, -1)

            # Aggregate to common grid (you already have this)
            cam0_on_grid = _aggregate_columns_to_bins(bin_idx, cam0_M, common_t.size)
            cam1_on_grid = _aggregate_columns_to_bins(bin_idx, cam1_M, common_t.size)

            # >>> Low-pass filter position and compute velocity (on-grid) <<<
            cam0_pos_filt, cam0_vel, _ = butter_lowpass_pos_and_vel(cam0_on_grid, common_t, cutoff_hz=10.0, order=3)
            cam1_pos_filt, cam1_vel, _ = butter_lowpass_pos_and_vel(cam1_on_grid, common_t, cutoff_hz=10.0, order=3)

            # Now build the peri-stim median *position* lines from the filtered positions:
            cam0_lines, beh_rel_t, cam0_ntr = _median_lines_for_columns(cam0_pos_filt, common_t, stim_ms)
            cam1_lines, _,          cam1_ntr = _median_lines_for_columns(cam1_pos_filt, common_t, stim_ms)

            # And for the velocity plot, use the filtered-position-derived velocity directly (no extra despike needed):
            cam0_v_lines, beh_rel_t_v, cam0_v_ntr = _median_lines_for_columns(cam0_vel, common_t, stim_ms)
            cam1_v_lines, _,             cam1_v_ntr = _median_lines_for_columns(cam1_vel, common_t, stim_ms)

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

            # Titles
            if have_cam0 and have_cam1:
                title_kinematics = (
                    f"Median Kinematics Right Camera (Cam-0) "
                    f"Video: {video_file if video_file is not None else 'n/a'} (n={beh_ntr} trials)"
                )
                title_cam1 = "Median Kinematics Left Camera (Cam-1)"
            elif have_cam0:
                title_kinematics = (
                    f"Median Kinematics (Cam-0) "
                    f"Video: {video_file if video_file is not None else 'n/a'} (n={beh_ntr} trials)"
                )
                title_cam1 = None
            elif have_cam1:
                title_kinematics = (
                    f"Median Kinematics (Cam-1) "
                    f"Video: {video_file if video_file is not None else 'n/a'} (n={beh_ntr} trials)"
                )
                title_cam1 = None
            else:
                title_kinematics = ""
                title_cam1 = None


            # Derive behavior labels from whichever camera has names, but simplify them
            if len(cam0_names) > 0:
                beh_labels = _simple_beh_labels([_strip_cam_prefix(n) for n in cam0_names], KEYPOINTS_ORDER)
            elif len(cam1_names) > 0:
                beh_labels = _simple_beh_labels([_strip_cam_prefix(n) for n in cam1_names], KEYPOINTS_ORDER)
            else:
                beh_labels = []

            # --- NPRW/Intan ---
            NPRW_segments, rel_time_ms_i = rcp.extract_peristim_segments(
                rate_hz=NPRW_rate, t_ms=NPRW_t, stim_ms=stim_ms,
                win_ms=WIN_MS, min_trials=MIN_TRIALS
            )
            NPRW_zeroed = rcp.baseline_zero_each_trial(
                NPRW_segments, rel_time_ms_i, normalize_first_ms=NORMALIZE_FIRST_MS
            )
            NPRW_med = rcp.median_across_trials(NPRW_zeroed)

            # ---------- UA: same logic but keep its own rel_time_ms ----------
            UA_segments, ua_rel_time_ms = rcp.extract_peristim_segments(
                rate_hz=UA_rate, t_ms=UA_t, stim_ms=stim_ms,
                win_ms=WIN_MS, min_trials=MIN_TRIALS
            )
            UA_zeroed = rcp.baseline_zero_each_trial(
                UA_segments, ua_rel_time_ms, normalize_first_ms=NORMALIZE_FIRST_MS
            )
            UA_med = rcp.median_across_trials(UA_zeroed)

            # locate the Intan stim_stream.npz and locate stimulated channels
            bundles_root = OUT_BASE / "bundles" / "NPRW"
            stim_npz = bundles_root / f"{sess}_Intan_bundle" / "stim_stream.npz"
            stim_locs = None
            if stim_npz.exists():
                try:
                    stim_locs = rcp.detect_stim_channels_from_npz(stim_npz, eps=1e-12, min_edges=1)
                except Exception as e:
                    print(f"[warn] stim-site detection failed for {sess}: {e}")
                    
            # ---------------- POSITION plot ----------------
            out_svg_pos = FIG.peri_pos / f"{sess}__Intan_vs_UA__peri_stim_heatmaps__position.png"

            title_top_pos = (
                f"Median Δ in firing rate (Referenced to first {int(NORMALIZE_FIRST_MS)} ms)\n"
                f"NPRW/Intan: {sess} (n={NPRW_segments.shape[0]} trials)"
            )
            title_bot_pos = f"{rcp.ua_title_from_meta(meta)} (n={UA_segments.shape[0]} trials)"

            rcp.stacked_heatmaps_plus_behv(
                NPRW_med, UA_med, rel_time_ms_i, ua_rel_time_ms,
                out_svg_pos, title_kinematics, title_top_pos, title_bot_pos, cmap=COLORMAP,
                vmin_intan=VMIN_INTAN, vmax_intan=VMAX_INTAN,
                vmin_ua=VMIN_UA, vmax_ua=VMAX_UA,
                probe=probe, probe_locs=locs, stim_idx=stim_locs,
                probe_title="NPRW probe (stim sites highlighted)",
                ua_ids_1based=ua_ids_1based,
                ua_sort="region_then_elec",
                beh_rel_time=beh_rel_t,
                beh_cam0_lines=cam0_lines if have_cam0 else None,
                beh_cam1_lines=cam1_lines if have_cam1 else None,
                beh_labels=beh_labels,
                sess=sess,
                overall_title=overall_title,
                title_cam1=title_cam1,
            )

            print(f"[PLOT] POSITION saved → {out_svg_pos}")

            # Update display titles to say "Velocity"
            if have_cam0 and have_cam1:
                title_kinematics_v = (
                    f"Median Kinematics Velocity Right Camera (Cam-0) "
                    f"Video: {video_file if video_file is not None else 'n/a'} (n={max(cam0_v_ntr, cam1_v_ntr)} trials)"
                )
                title_cam1_v = "Median Kinematics Velocity Left Camera (Cam-1)"
            elif have_cam0:
                title_kinematics_v = (
                    f"Median Kinematics Velocity (Cam-0) "
                    f"Video: {video_file if video_file is not None else 'n/a'} (n={cam0_v_ntr} trials)"
                )
                title_cam1_v = None
            elif have_cam1:
                title_kinematics_v = (
                    f"Median Kinematics Velocity (Cam-1) "
                    f"Video: {video_file if video_file is not None else 'n/a'} (n={cam1_v_ntr} trials)"
                )
                title_cam1_v = None
            else:
                title_kinematics_v = title_kinematics  # fallback
                title_cam1_v = title_cam1

            out_svg_vel = FIG.peri_vel / f"{sess}__Intan_vs_UA__peri_stim_heatmaps__velocity.png"

            title_top_vel = title_top_pos  # same neural titles
            title_bot_vel = title_bot_pos

            rcp.stacked_heatmaps_plus_behv(
                NPRW_med, UA_med, rel_time_ms_i, ua_rel_time_ms,
                out_svg_vel, title_kinematics_v, title_top_vel, title_bot_vel, cmap=COLORMAP,
                vmin_intan=VMIN_INTAN, vmax_intan=VMAX_INTAN,
                vmin_ua=VMIN_UA, vmax_ua=VMAX_UA,
                probe=probe, probe_locs=locs, stim_idx=stim_locs,
                probe_title="NPRW probe (stim sites highlighted)",
                ua_ids_1based=ua_ids_1based,
                ua_sort="region_then_elec",
                beh_rel_time=beh_rel_t_v,                  # velocity timebase
                beh_cam0_lines=cam0_v_lines if cam0_v_lines.size else None,
                beh_cam1_lines=cam1_v_lines if cam1_v_lines.size else None,
                beh_labels=beh_labels,
                sess=sess,
                overall_title=overall_title,
                title_cam1=title_cam1_v,
            )

            print(f"[PLOT] VELOCITY saved → {out_svg_vel}")
            
        except Exception as e:
            print(f"[error] Failed on {file.name}: {e}")
            continue

if __name__ == "__main__":
    # plot the aligned baseline traces
    main_baselines()
    
    # plot stim trials
    # main()
