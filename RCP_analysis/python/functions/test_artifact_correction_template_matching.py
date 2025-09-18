from __future__ import annotations
"""
Artifact correction via streaming template matching (fast path)
================================================================

Goals implemented (per user request):
  1) Avoid the (NSTIM × L) pulse matrix → streaming EWMA templates, no giant stacks
  3) Batch channels + shared work → process channels in configurable batches
  4) Parallelize → joblib threads over batches (safe with NumPy/CuPy)
  6) Vectorize interpulse drift → single-pass, per-block vector ops
  8) Integrate usage of CuPy → transparent xp-backend (NumPy fallback), GPU for heavy math
     *Note:* rolling median is still CPU (numba) for quality; we round-trip when GPU enabled.
 10) Integrate with SI graph → lazy SpikeInterface Recording wrapper that applies correction on-the-fly

This module exposes two primary entrypoints:
  - make_artifact_corrected_recording(recording, stim_npz_path, params, ...)
      → returns a SpikeInterface Recording that performs artifact removal lazily in get_traces().
  - correct_numpy_array(data, trigger_info, params, ...)
      → eager correction for a (n_ch, n_samp) array.

Requirements:
  - numpy, scipy, scikit-learn, joblib
  - Optional: cupy (GPU accel). If available, most array ops run on GPU. Rolling-median runs on CPU.
  - Optional (CPU speedup): numba for fast rolling median
  - SpikeInterface (for graph integration)

Notes:
  - We replace MATLAB's med+gauss filtfilt with rolling median (CPU) + forward-backward Gaussian smoothing
    (GPU/CPU). This preserves zero-phase behavior affordably.
  - We stream templates: an EWMA within each repeat block, with special handling for first pulse.
  - Interpulse drift is applied as a midpoint-to-midpoint ramp per adjacent pulse pair.
  - Trigger discovery mirrors your MATLAB logic but is stricter/safer; adjust as needed for your datasets.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Literal

import numpy as np
from numpy.typing import NDArray
from joblib import Parallel, delayed

try:  # Optional GPU
    import cupy as cp  # type: ignore
    HAS_CUPY = True
except Exception:  # pragma: no cover
    cp = None  # type: ignore
    HAS_CUPY = False

from scipy.signal import convolve

try:  # Optional; used for the SI graph integration
    import spikeinterface as si
    from spikeinterface.core import BaseRecording
except Exception:  # pragma: no cover
    si = None
    BaseRecording = object  # type: ignore

try:  # Optional CPU speed-up for rolling median
    import numba as nb
    HAS_NUMBA = True
except Exception:  # pragma: no cover
    nb = None  # type: ignore
    HAS_NUMBA = False

# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------
TemplateMode = Literal["ewma"]  # "ewma" default fast path


@dataclass(frozen=True)
class TemplateParams:
    fs: float = 30_000.0
    buffer: int = 25
    template_leeway: int = 15
    stim_neural_delay: int = 13
    med_filt_range: int = 25        # rolling median half-width*2+1 window length
    gauss_filt_range: int = 25      # gaussian kernel length (odd)
    ewma_alpha: float = 0.35        # EWMA blending factor within a block


@dataclass(frozen=True)
class TriggerInfo:
    trigs: NDArray[np.int64]              # (N, 2) [beg, end]
    repeat_boundaries: NDArray[np.int64]  # (R+1,)
    stim_chans: NDArray[np.int64]


# -----------------------------------------------------------------------------
# Public API (eager)
# -----------------------------------------------------------------------------

def correct_numpy_array(
    data: NDArray[np.floating],  # (n_ch, n_samp)
    trigger_info: TriggerInfo,
    params: TemplateParams,
    mode: TemplateMode = "ewma",
    channels: Optional[Iterable[int]] = None,
    batch_size: int = 32,
    n_jobs: int = 0,
    use_gpu: bool = False,
) -> NDArray[np.float32]:
    """Artifact-correct a (n_ch, n_samp) matrix.

    - Streams templates (no NSTIM x L stack)
    - Batches channels and parallelizes over batches
    - Uses CuPy when available for vector math (except rolling median)
    """
    x_cpu = np.asarray(data, dtype=np.float32)
    n_ch, n_samples = x_cpu.shape

    if channels is None:
        channels = range(n_ch)
    ch_list = np.array(list(channels), dtype=int)

    # Choose backend
    xp = _get_xp(use_gpu)

    # Precompute drift once per channel batch (median+gauss), then subtract
    def _process_batch(ch_idx: NDArray[np.int32]) -> Tuple[NDArray[np.float32], None]:
        Xb = x_cpu[ch_idx]  # view (batch, T)
        # 1) Rolling median (CPU, numba if available)
        med = _rolling_median_batch_cpu(Xb, params.med_filt_range)
        # 2) Forward-backward Gaussian smoothing (GPU if available)
        gauss = _gaussian_smooth_batch(med, params.gauss_filt_range, xp=xp)
        # 3) subtract drift
        Yb = Xb - gauss
        # 4) subtract EWMA templates + interpulse ramps per block
        Yb = _subtract_artifacts_batch(
            Yb, trigger_info, params, mode=mode, xp=xp
        )
        return Yb.astype(np.float32, copy=False), None

    # Create batches
    batches = [ch_list[i : i + batch_size] for i in range(0, ch_list.size, batch_size)]

    if n_jobs and len(batches) > 1:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_process_batch)(b) for b in batches)
    else:
        results = [_process_batch(b) for b in batches]

    out = np.zeros_like(x_cpu, dtype=np.float32)
    for batch_idx, (Yb, _) in zip(batches, results):
        out[batch_idx] = Yb
    return out

# -----------------------------------------------------------------------------
# Public API (SpikeInterface lazy transform)
# -----------------------------------------------------------------------------

class ArtifactCorrectedRecording(BaseRecording):
    """Lazy artifact correction node that composes inside the SpikeInterface graph.

    Only the requested time window/channels are processed in get_traces().
    """
    def __init__(
        self,
        parent_recording: BaseRecording,
        trigger_info: TriggerInfo,
        params: TemplateParams,
        mode: TemplateMode = "ewma",
        batch_size: int = 32,
        n_jobs: int = 0,
        use_gpu: bool = False,
    ):
        if si is None:
            raise RuntimeError("SpikeInterface not available")
        super().__init__(sampling_frequency=parent_recording.get_sampling_frequency(),
                         channel_ids=parent_recording.channel_ids,
                         dtype=np.float32)
        self._parent = parent_recording
        self._trigger = trigger_info
        self._params = params
        self._mode = mode
        self._batch = int(batch_size)
        self._n_jobs = int(n_jobs)
        self._use_gpu = bool(use_gpu)
        try:
            self.set_probe(parent_recording.get_probe(), in_place=True)
        except Exception:
            pass
        # copy properties
        try:
            for k in parent_recording.get_property_keys():
                self.set_property(k, parent_recording.get_property(k))
        except Exception:
            pass

    def get_num_segments(self) -> int:  # passthrough
        return self._parent.get_num_segments()

    def get_traces(self, start_frame: int, end_frame: int, segment_index: int = 0, channel_ids=None):
        # Delegate read
        X = self._parent.get_traces(start_frame, end_frame, segment_index, channel_ids)
        # Normalize to (n_ch, n_samples)
        if X.shape[0] < X.shape[1]:
            X = X.T
        # Eager-correct only the window we pulled
        # Build a local TriggerInfo slice restricted to pulses in [start_frame, end_frame)
        trig_slice = _slice_triggers(self._trigger, start_frame, end_frame, self._params)
        if trig_slice.trigs.size == 0:
            return X.T  # no pulses in window; return as-is
        Y = correct_numpy_array(
            X, trig_slice, self._params, mode=self._mode,
            channels=None, batch_size=self._batch, n_jobs=self._n_jobs, use_gpu=self._use_gpu
        )
        return Y.T

def make_artifact_corrected_recording(
    recording: BaseRecording,
    stim_npz_path: Path,
    params: TemplateParams,
    mode: TemplateMode = "ewma",
    batch_size: int = 32,
    n_jobs: int = 0,
    use_gpu: bool = False,
) -> ArtifactCorrectedRecording:
    trig = extract_triggers_and_repeats_from_npz(stim_npz_path, fs=recording.get_sampling_frequency(), params=params)
    return ArtifactCorrectedRecording(recording, trig, params, mode=mode, batch_size=batch_size, n_jobs=n_jobs, use_gpu=use_gpu)

# -----------------------------------------------------------------------------
# Trigger extraction
# -----------------------------------------------------------------------------

def extract_triggers_and_repeats_from_npz(stim_npz_path: Path, fs: float, params: TemplateParams) -> TriggerInfo:
    D = np.load(stim_npz_path, allow_pickle=True)
    stim = D["data"]  # (n_stim_chan, n_samples)
    f0 = float(D.get("sampling_frequency", fs))
    if abs(f0 - fs) > 1e-3:
        raise ValueError(f"Stim fs mismatch: {f0} vs recording fs {fs}")
    return extract_triggers_and_repeats(stim, fs, params)


def extract_triggers_and_repeats(
    stim_data: NDArray[np.floating], fs: float, params: TemplateParams, threshold: Optional[float] = None
) -> TriggerInfo:
    if stim_data.ndim != 2:
        raise ValueError("stim_data must be (n_stim_chan, n_samples)")
    active = np.where(np.any(stim_data != 0, axis=1))[0]
    if active.size == 0:
        return TriggerInfo(trigs=np.zeros((0, 2), dtype=np.int64), repeat_boundaries=np.array([0], dtype=np.int64), stim_chans=np.array([], dtype=np.int64))

    ch = int(active[0])
    x = np.asarray(stim_data[ch], dtype=float)

    # Edge detection with hysteresis-like behavior
    if threshold is None:
        dx = np.diff(x)
        rising = np.flatnonzero(dx > 0)
        falling = np.flatnonzero(dx < 0)
    else:
        above = x > threshold
        edges = np.flatnonzero(np.diff(above.astype(np.int8)))
        rising = edges[above[edges + 1]]
        falling = edges[~above[edges + 1]]

    # Pair edges robustly: choose dominant polarity for begins
    beg = rising if falling.size < rising.size else falling
    beg = beg[::2]  # emulate MATLAB's stride

    # Find first zero after opposite-edge, stride by 2
    def _next_zero(idx: int) -> Optional[int]:
        z = np.flatnonzero(x[idx + 1 :] == 0)
        return int(idx + 1 + z[0]) if z.size else None

    rz = []
    opp = falling if beg is rising else rising
    for idx in opp:
        z = _next_zero(idx)
        if z is not None:
            rz.append(z)
    rz = np.asarray(rz, dtype=int)
    end = rz[1::2]

    n = min(beg.size, end.size)
    if n == 0:
        return TriggerInfo(trigs=np.zeros((0, 2), dtype=np.int64), repeat_boundaries=np.array([0], dtype=np.int64), stim_chans=active.astype(np.int64))

    trigs = np.c_[beg[:n], end[:n]].astype(np.int64)

    # Repeat boundaries by gap heuristic
    time_diffs = np.diff(trigs[:, 0])
    repeat_gap_threshold = int(2 * (2 * params.buffer + 1))
    cuts = np.flatnonzero(time_diffs > repeat_gap_threshold)
    repeat_boundaries = np.r_[0, cuts + 1, trigs.shape[0]].astype(np.int64)

    return TriggerInfo(trigs=trigs, repeat_boundaries=repeat_boundaries, stim_chans=active.astype(np.int64))


# -----------------------------------------------------------------------------
# Core artifact subtraction (batch, streaming)
# -----------------------------------------------------------------------------

def _subtract_artifacts_batch(
    Yb_cpu: NDArray[np.float32],
    trig: TriggerInfo,
    params: TemplateParams,
    mode: TemplateMode,
    xp,
) -> NDArray[np.float32]:
    """Subtract streaming EWMA templates and interpulse drift, per block, batched channels.

    Yb_cpu: (B, T) CPU array (drift already removed)
    Returns corrected (B, T) CPU array
    """
    # We keep compute on xp for the template ops; convert batch to xp then back.
    use_gpu = (xp is cp)
    Y = xp.asarray(Yb_cpu, dtype=xp.float32) if use_gpu else Yb_cpu  # (B, T)

    trigs_beg = trig.trigs[:, 0]
    trigs_end = trig.trigs[:, 1] + params.stim_neural_delay
    L = (trigs_end[0] - trigs_beg[0] + 1) + 2 * params.template_leeway if trig.trigs.size else 0

    if trig.trigs.size:
        # Per repeat block
        R = trig.repeat_boundaries.size - 1
        for r in range(R):
            blk = np.arange(trig.repeat_boundaries[r], trig.repeat_boundaries[r + 1], dtype=int)
            if blk.size == 0:
                continue
            # Streaming template for each channel in batch
            _apply_block_templates(Y, trigs_beg, trigs_end, blk, L, params, mode, xp)
            # Vectorized interpulse drift (still per-adjacent with a short loop)
            _apply_interpulse_drifts(Y, trigs_beg, trigs_end, blk, xp)

    return (cp.asnumpy(Y) if use_gpu else Y).astype(np.float32, copy=False)


def _apply_block_templates(Y, trigs_beg, trigs_end, blk, L, params: TemplateParams, mode: TemplateMode, xp):
    B, T = Y.shape
    leeway = params.template_leeway
    w = xp.linspace(0.0, 1.0, L, dtype=xp.float32) if L > 0 else None

    # For each channel independently, maintain EWMA state
    # We process in xp (GPU/CPU) without allocating NSTIM×L tensors
    for k, i in enumerate(blk):
        seg_start = trigs_beg[i] - leeway
        seg_end = trigs_end[i] + leeway
        if seg_start < 0 or seg_end >= T:
            continue
        seg = Y[:, seg_start : seg_end + 1]  # (B, L)
        base = seg - seg[:, :1]              # align to first sample
        if w is not None:
            base = base + (w * (-base[:, -1][:, None]))  # endpoint drift alignment

        if k == 0:
            # first pulse per block
            templ = base
            # write back EWMA state to a scratch attribute on Y via closure
            Y._ewma_state = base.copy() if hasattr(Y, "_ewma_state") else base.copy()  # type: ignore
        else:
            # EWMA update per channel
            alpha = xp.float32(params.ewma_alpha)
            state = getattr(Y, "_ewma_state", base.copy())  # type: ignore
            templ = alpha * base + (1.0 - alpha) * state
            Y._ewma_state = templ  # type: ignore

        # subtract template in place
        Y[:, seg_start : seg_end + 1] -= templ

    # Clear state after block
    if hasattr(Y, "_ewma_state"):
        try:
            delattr(Y, "_ewma_state")
        except Exception:
            pass


def _apply_interpulse_drifts(Y, trigs_beg, trigs_end, blk, xp):
    # midpoint-to-midpoint ramp between adjacent pulses within this block
    starts = ((trigs_beg[blk[:-1]] + trigs_end[blk[:-1]]) // 2).astype(int)
    stops = ((trigs_beg[blk[1:]] + trigs_end[blk[1:]]) // 2).astype(int)
    good = stops > (starts + 1)
    starts, stops, prev_idx = starts[good], stops[good], blk[:-1][good]

    for s, e, i_prev in zip(starts, stops, prev_idx):
        L = int(e - s)
        if L <= 1:
            continue
        # ramp is shared across channels except for the amplitude (value at end of previous pulse)
        amp = Y[:, trigs_end[i_prev]]  # (B,)
        ramp = xp.linspace(0.0, -1.0, L, dtype=xp.float32)[None, :] * amp[:, None]
        Y[:, s:e] += ramp


# -----------------------------------------------------------------------------
# Utilities: rolling median & Gaussian smoothing (batch)
# -----------------------------------------------------------------------------

def _rolling_median_batch_cpu(X: NDArray[np.float32], k: int) -> NDArray[np.float32]:
    if k <= 1:
        return X.astype(np.float32, copy=False)
    if not HAS_NUMBA:
        # Fallback: simple Python/numpy implementation (slower but OK for modest k)
        B, N = X.shape
        Y = np.empty_like(X, dtype=np.float32)
        h = k // 2
        for b in range(B):
            for i in range(N):
                a = max(0, i - h)
                d = min(N, i + h + 1)
                Y[b, i] = np.median(X[b, a:d])
        return Y

    @nb.njit(cache=True, fastmath=True, parallel=True)
    def _med_batch(Xn: NDArray[np.float32], k: int) -> NDArray[np.float32]:
        B, N = Xn.shape
        Y = np.empty_like(Xn, dtype=np.float32)
        h = k // 2
        for b in nb.prange(B):
            for i in range(N):
                a = 0 if i - h < 0 else i - h
                d = N if i + h + 1 > N else i + h + 1
                Y[b, i] = np.median(Xn[b, a:d])
        return Y

    return _med_batch(X.astype(np.float32, copy=False), int(k))


def _gaussian_kernel(L: int) -> NDArray[np.float32]:
    if L <= 1:
        return np.array([1.0], dtype=np.float32)
    x = np.linspace(-1.0, 1.0, L, dtype=np.float32)
    s = 0.4  # std as fraction of length
    g = np.exp(-0.5 * (x / s) ** 2)
    g /= g.sum()
    return g.astype(np.float32)


def _gaussian_smooth_batch(X: NDArray[np.float32], L: int, xp):
    if L <= 1:
        return X.astype(np.float32, copy=False)
    g = _gaussian_kernel(L)
    if xp is cp:
        g_x = cp.asarray(g)
        Xx = cp.asarray(X, dtype=cp.float32)
        # zero-phase: forward then backward (approx filtfilt)
        Y = cp.apply_along_axis(lambda v: cp.convolve(v, g_x, mode="same"), 1, Xx)
        Y = cp.apply_along_axis(lambda v: cp.convolve(v, g_x, mode="same"), 1, Y[:, ::-1])[:, ::-1]
        return cp.asnumpy(Y).astype(np.float32, copy=False)
    else:
        # CPU: use scipy.signal.convolve with 'same' twice (zero-phase)
        Y = np.apply_along_axis(lambda v: convolve(v, g, mode="same"), 1, X)
        Y = np.apply_along_axis(lambda v: convolve(v, g, mode="same"), 1, Y[:, ::-1])[:, ::-1]
        return Y.astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# Support: slicing triggers for a requested window (SI lazy node)
# -----------------------------------------------------------------------------

def _slice_triggers(trigger: TriggerInfo, start: int, end: int, params: TemplateParams) -> TriggerInfo:
    if trigger.trigs.size == 0:
        return trigger
    tr_beg = trigger.trigs[:, 0]
    tr_end = trigger.trigs[:, 1] + params.stim_neural_delay
    # pulses intersect window if beg<=end && end>=start
    keep = (tr_beg <= end) & (tr_end >= start)
    trigs = trigger.trigs[keep]
    if trigs.size == 0:
        return TriggerInfo(trigs=np.zeros((0, 2), dtype=np.int64), repeat_boundaries=np.array([0], dtype=np.int64), stim_chans=trigger.stim_chans)
    # remap boundaries within kept indices
    # Find original indices of kept pulses
    orig_idx = np.flatnonzero(keep)
    cuts = []
    for rb in trigger.repeat_boundaries:
        # if a boundary index exists among kept, record its position in new index space
        pos = np.searchsorted(orig_idx, rb)
        if pos < orig_idx.size and orig_idx[pos] == rb:
            cuts.append(pos)
    # ensure 0 and end
    rb_new = [0]
    rb_new += [c for c in cuts if c not in (0, len(orig_idx))]
    rb_new.append(trigs.shape[0])
    rb_new = np.unique(np.asarray(rb_new, dtype=np.int64))
    return TriggerInfo(trigs=trigs, repeat_boundaries=rb_new, stim_chans=trigger.stim_chans)


# -----------------------------------------------------------------------------
# Backend helper
# -----------------------------------------------------------------------------

def _get_xp(use_gpu: bool):
    return cp if (use_gpu and HAS_CUPY) else np


# ==============================
# End of file
# ==============================
