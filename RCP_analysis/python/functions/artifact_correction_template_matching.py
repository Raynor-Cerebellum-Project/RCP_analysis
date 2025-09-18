from __future__ import annotations
"""
PCA-based stimulation artifact correction
===========================================================

This module implements PCA template subtraction pipeline.

- `TemplateParams` (includes `pca_k`)
- `extract_triggers_and_repeats(...)`
- `correct_recording_with_stim_npz(...)`  # SpikeInterface integration

The core steps are:
  1) Trigger detection and block boundaries
  2) Global drift removal (rolling median + zero‑phase Gaussian)
  3) Pulse-aligned matrix construction
  4) PCA template construction per block (special handling for first pulses)
  5) Template subtraction
  6) Interpulse drift ramp

Dependencies: numpy, scipy, scikit-learn, (optional) joblib, SpikeInterface
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Literal
import io, json, zipfile

import numpy as np
from numpy.typing import NDArray
from scipy.signal import filtfilt, windows
from sklearn.decomposition import PCA

from joblib import Parallel, delayed
HAS_JOBLIB = True

import spikeinterface as si
from spikeinterface.core import NumpyRecording, BaseRecording
HAS_SI = True

TemplateMode = Literal["pca"]


# -----------------------------------------------------------------------------
# Parameters & Trigger info
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TemplateParams:
    fs: float = 30_000.0
    buffer: int = 25
    template_leeway: int = 15
    stim_neural_delay: int = 13
    movmean_window: int = 3
    med_filt_range: int = 25
    gauss_filt_range: int = 25
    pca_k: int = 3


@dataclass(frozen=True)
class TriggerInfo:
    trigs: NDArray[np.int64]              # (N, 2) [beg, end]
    repeat_boundaries: NDArray[np.int64]  # (R+1,)
    stim_chans: NDArray[np.int64]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def correct_recording_with_stim_npz(
    recording: "BaseRecording",
    stim_npz_path: Path,
    params: TemplateParams,
    mode: TemplateMode = "pca",
    channels: Optional[Iterable[int]] = None,
    n_jobs: int = 0,
):
    """Return a new SpikeInterface Recording with PCA artifact correction applied.
    If no stim signal is detected, just return the input recording unchanged.
    """
    if not HAS_SI or NumpyRecording is None:
        raise RuntimeError("SpikeInterface not available; install spikeinterface to use this function.")

    # --- Try to load the streaming NPZ ---
    try:
        stim_data, fs_stim = _load_streaming_npz(stim_npz_path)  # (n_stim_chan, n_samples)
    except Exception as e:
        print(f"[ArtCorr] Could not load stim NPZ '{stim_npz_path}': {e}. Passing recording through.")
        return recording  # passthrough

    # --- If stim has no activity, passthrough ---
    if stim_data.size == 0 or not np.any(stim_data != 0):
        print("[ArtCorr] No stim signal detected in NPZ. Passing recording through.")
        return recording

    # --- Fs check against recording ---
    fs_rec = float(recording.get_sampling_frequency())
    if abs(fs_stim - fs_rec) > 1e-3:
        raise ValueError(f"Stim fs ({fs_stim}) != recording fs ({fs_rec}); resample before use.")

    # --- Triggers & repeats ---
    try:
        triginfo = extract_triggers_and_repeats(stim_data, fs=fs_rec, params=params)
    except Exception as e:
        print(f"[ArtCorr] Trigger extraction failed or no valid triggers: {e}. Passing recording through.")
        return recording

    if triginfo.trigs.size == 0:
        print("[ArtCorr] Trigger list empty. Passing recording through.")
        return recording

    # --- Get traces as (C, T) ---
    traces = recording.get_traces()
    if traces.shape[0] > traces.shape[1]:  # (T, C) -> (C, T)
        traces = traces.T

    # --- Correct ---
    cleaned, _ = correct_numpy_array(
        data=traces,
        trigger_info=triginfo,
        params=params,
        mode=mode,
        channels=channels,
        n_jobs=n_jobs,
    )

    # --- Re-wrap in NumpyRecording, copying metadata ---
    chan_ids = list(getattr(recording, "channel_ids", range(cleaned.shape[0])))
    out = NumpyRecording([cleaned.T], sampling_frequency=fs_rec, channel_ids=chan_ids)

    try:
        out = out.set_probe(recording.get_probe(), in_place=False)
    except Exception:
        pass
    try:
        for k in recording.get_property_keys():
            out.set_property(k, recording.get_property(k))
    except Exception:
        pass

    return out


def correct_numpy_array(
    data: NDArray[np.floating],  # (n_ch, n_samp)
    trigger_info: TriggerInfo,
    params: TemplateParams,
    mode: TemplateMode = "pca",
    channels: Optional[Iterable[int]] = None,
    n_jobs: int = 0,
) -> Tuple[NDArray[np.float32], list[dict]]:
    """Artifact-correct a multi-channel array. Parallelizes by channel if `n_jobs>0`."""
    x = np.asarray(data, dtype=np.float32)
    n_ch, _ = x.shape
    if channels is None:
        channels = range(n_ch)
    ch_list = list(channels)

    def _one(ch: int):
        y, aux = _template_subtraction(
            amplifier_data=x[ch],
            trigs=trigger_info.trigs,
            params=params,
            template_mode=mode,
            repeat_boundaries=trigger_info.repeat_boundaries,
        )
        return y, aux

    if HAS_JOBLIB and n_jobs and len(ch_list) > 1:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_one)(ch) for ch in ch_list)
    else:
        results = list(map(_one, ch_list))

    y = np.zeros_like(x, dtype=np.float32)
    aux_list: list[dict] = []
    for idx, ch in enumerate(ch_list):
        y[ch], aux = results[idx]
        aux_list.append(aux)
    return y, aux_list


# -----------------------------------------------------------------------------
# Trigger detection and block boundaries
# -----------------------------------------------------------------------------

def extract_triggers_and_repeats(
    stim_data: NDArray[np.floating], fs: float, params: TemplateParams, threshold: Optional[float] = None
) -> TriggerInfo:
    """Compute [begin, end] pairs and repeat block boundaries from stim channel(s)."""
    if stim_data.ndim != 2:
        raise ValueError("stim_data must be (n_stim_chan, n_samples)")

    active = np.where(np.any(stim_data != 0, axis=1))[0]
    if active.size == 0:
        raise ValueError("No stim signal detected.")

    x = np.asarray(stim_data[int(active[0])], dtype=float)

    if threshold is None:
        dx = np.diff(x)
        rising = np.flatnonzero(dx > 0)
        falling = np.flatnonzero(dx < 0)
    else:
        above = x > threshold
        edges = np.flatnonzero(np.diff(above.astype(np.int8)))
        rising = edges[above[edges + 1]]
        falling = edges[~above[edges + 1]]

    use_rising = falling.size < rising.size
    begins = rising if use_rising else falling
    begins = begins[::2]

    def _next_zero(idx: int) -> Optional[int]:
        z = np.flatnonzero(x[idx + 1:] == 0)
        return int(idx + 1 + z[0]) if z.size else None

    opp = falling if use_rising else rising
    
    rz = []
    for idx in opp:
        z = _next_zero(idx)
        if z is not None:
            rz.append(z)
    rz = np.asarray(rz, dtype=int)
    ends = rz[1::2]
    
    n = min(begins.size, ends.size)
    if n == 0:
        raise ValueError("Unable to pair triggers.")
    trigs = np.c_[begins[:n], ends[:n]].astype(np.int64)

    # Repeat boundaries via gap heuristic
    time_diffs = np.diff(trigs[:, 0])
    repeat_gap_threshold = int(2 * (2 * params.buffer + 1))
    cuts = np.flatnonzero(time_diffs > repeat_gap_threshold)
    repeat_boundaries = np.r_[0, cuts + 1, trigs.shape[0]].astype(np.int64)

    return TriggerInfo(trigs=trigs, repeat_boundaries=repeat_boundaries, stim_chans=active.astype(np.int64))


# -----------------------------------------------------------------------------
# Core: PCA template subtraction (single channel)
# -----------------------------------------------------------------------------

def _template_subtraction(
    amplifier_data: NDArray[np.float32],
    trigs: NDArray[np.int64],
    params: TemplateParams,
    template_mode: TemplateMode,
    repeat_boundaries: NDArray[np.int64],
) -> Tuple[NDArray[np.float32], dict]:
    """Single-channel PCA artifact correction with interpulse ramp."""
    leeway = int(params.template_leeway)
    delay = int(params.stim_neural_delay)

    if trigs.size == 0:
        return amplifier_data.astype(np.float32, copy=True), {"values": np.zeros_like(amplifier_data), "range": (0, amplifier_data.size)}

    tr_beg = trigs[:, 0]
    tr_end = trigs[:, 1] + delay

    raw = amplifier_data.astype(np.float32, copy=False)

    # --- Global drift removal (robust) ---
    med = _movmedian(raw, params.med_filt_range)
    g = windows.gaussian(params.gauss_filt_range, std=max(1.0, params.gauss_filt_range / 5.0)).astype(np.float32)
    g /= g.sum()
    baseline = filtfilt(g, [1.0], med, method="pad").astype(np.float32, copy=False)
    x = raw - baseline
    drift_struct = {"values": baseline, "range": (0, raw.size)}

    # --- Build pulse-aligned matrix (NSTIM × L) ---
    L = (tr_end[0] - tr_beg[0] + 1) + 2 * leeway
    nstim = trigs.shape[0]
    mat = np.full((nstim, L), np.nan, dtype=np.float32)
    valid = np.zeros(nstim, dtype=bool)

    for i in range(nstim):
        a = tr_beg[i] - leeway
        b = tr_end[i] + leeway
        if a >= 0 and b < x.size:
            mat[i] = x[a : b + 1]
            valid[i] = True

    # --- Templates per block ---
    templ = _generate_template(mat, template_mode, repeat_boundaries, params.movmean_window, params.pca_k)

    # --- Subtract templates ---
    y = x.copy()
    for i in np.flatnonzero(valid):
        a = tr_beg[i] - leeway
        b = tr_end[i] + leeway
        y[a : b + 1] -= templ[i]

    # --- Interpulse drift correction per block ---
    for r in range(repeat_boundaries.size - 1):
        blk = np.arange(repeat_boundaries[r], repeat_boundaries[r + 1])
        if blk.size < 2:
            continue
        # midpoint-to-midpoint ramps
        mids1 = ((tr_end[blk[:-1]] + tr_beg[blk[:-1]]) // 2).astype(int)
        mids2 = ((tr_beg[blk[1:]] + tr_end[blk[1:]]) // 2).astype(int)
        good = mids2 > (mids1 + 1)
        for s, e, i_prev in zip(mids1[good], mids2[good], blk[:-1][good]):
            Lr = int(e - s)
            if Lr <= 1:
                continue
            ramp = np.linspace(0.0, -float(y[tr_end[i_prev]]), Lr, dtype=np.float32)
            y[s:e] += ramp

    return y.astype(np.float32, copy=False), drift_struct


def _generate_template(
    mat: NDArray[np.float32],
    mode: TemplateMode,
    repeat_boundaries: NDArray[np.int64],
    window_size: int,
    pca_k: int,
) -> NDArray[np.float32]:
    """Construct per-pulse templates using pca"""
    nstim, L = mat.shape
    templ = np.zeros((nstim, L), dtype=np.float32)
    w = np.linspace(0.0, 1.0, L, dtype=np.float32)

    # First pulse flags per block
    is_first = np.zeros(nstim, dtype=bool)
    is_first[repeat_boundaries[:-1]] = True

    # PCA mode
    blocks = repeat_boundaries
    for b in range(blocks.size - 1):
        i0, i1 = int(blocks[b]), int(blocks[b + 1])
        if i1 <= i0:
            continue
        first_i = i0
        # First pulse in block: align to prior firsts (up to 3)
        prev_first = np.flatnonzero(is_first[:first_i])
        if prev_first.size:
            base = np.nanmean(mat[prev_first[-min(3, prev_first.size):]], axis=0)
        else:
            base = np.zeros(L, dtype=np.float32)
        t0 = base - base[0]
        templ[first_i] = t0 + w * (-t0[-1])

        if i1 - i0 > 1:
            rows = np.arange(i0 + 1, i1)
            X = mat[rows]
            # row-center & NaN-safe
            row_means = np.nanmean(X, axis=1, keepdims=True)
            Xc = X - row_means
            nan_mask = np.isnan(Xc)
            if np.any(nan_mask):
                Xc[nan_mask] = np.take_along_axis(row_means, np.where(nan_mask)[0][:, None], axis=0)

            k = int(max(0, min(pca_k, Xc.shape[0], Xc.shape[1])))
            if k > 0:
                pca = PCA(n_components=k, svd_solver="auto", whiten=False)
                score = pca.fit_transform(Xc)
                recon = pca.inverse_transform(score)
            else:
                recon = np.tile(np.nanmean(Xc, axis=0, keepdims=True), (Xc.shape[0], 1))

            t = recon - recon[:, :1]
            templ[rows] = t + (w[None, :] * (-t[:, -1][:, None]))

    return np.nan_to_num(templ, nan=0.0, copy=False)


def _load_streaming_npz(npz_path: Path) -> tuple[np.ndarray, float]:
    """
    Read a 'streaming NPZ' written by _save_npz_streaming:
      - meta.json (with fs_hz, shape, etc.)
      - chunk_####.npy entries

    Returns (data, fs_hz) with data shaped (n_stim_chan, n_samples).
    """
    with zipfile.ZipFile(str(npz_path), "r") as zf:
        names = zf.namelist()
        if "meta.json" not in names:
            raise ValueError("meta.json missing in streaming NPZ")

        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        fs = float(meta.get("fs_hz", meta.get("sampling_frequency", 30_000.0)))

        # collect chunk files (accept 'chunk_0000' or 'chunk_0000.npy')
        chunk_names = sorted(
            [n for n in names if n.startswith("chunk_") and (n.endswith(".npy") or "." not in n)],
        )
        if not chunk_names:
            raise ValueError(f"No chunk_#### entries in {npz_path}. Found: {names}")

        parts = []
        for n in chunk_names:
            # ensure .npy suffix when reading
            raw = zf.read(n)
            arr = np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)
            parts.append(np.asarray(arr))

        # concatenate along time axis -> shape (T_total, C)
        data_tc = np.concatenate(parts, axis=0)

        # make (n_ch, n_samples)
        if data_tc.ndim != 2:
            raise ValueError(f"stim chunks are not 2D: shape {data_tc.shape}")
        data_ct = data_tc.T  # (C, T)
        return data_ct.astype(np.float32, copy=False), fs

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _movmedian(x: NDArray[np.float32], k: int) -> NDArray[np.float32]:
    """Centered moving median with edge shrink (MATLAB-like)."""
    if k <= 1:
        return x.astype(np.float32, copy=False)
    half = k // 2
    y = np.empty_like(x, dtype=np.float32)
    n = x.size
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        y[i] = float(np.median(x[a:b]))
    return y
