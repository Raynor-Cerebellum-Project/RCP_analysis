from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.signal import filtfilt, medfilt, windows
from sklearn.decomposition import PCA
from dataclasses import asdict

from spikeinterface.core.core_tools import define_function_handling_dict_from_class
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment


# ----------------------------- Parameters -------------------------------------

@dataclass
class ArtifactParams:
    # Windowing around triggers
    samples_before: int = 13
    samples_after: int = 15

    # Global drift removal
    rolling_median_ms: float = 15.0
    gaussian_sigma_ms: float = 5.0
    gaussian_len_ms: float = 31.0

    # PCA template construction
    pca_k: int = 3
    center_snippets: bool = True
    first_pulse_special: bool = True
    exclude_first_n_for_pca: int = 1

    # Subtraction options
    scale_amplitude: bool = True

    # Interpulse ramp
    interp_ramp: bool = True
    ramp_tail_ms: float = 1.0
    ramp_fraction: float = 1.0

# ---------------------------- Helper functions --------------------------------
def _global_drift_remove(traces: np.ndarray, fs: float, params: ArtifactParams) -> np.ndarray:
    """
    Rolling median (baseline) subtraction followed by zero-phase Gaussian smoothing removal.
    """
    out = traces.astype(np.float64, copy=True)
    # Rolling median (odd kernel)
    k_med = max(1, int(round(params.rolling_median_ms * fs / 1000.0)))
    if k_med % 2 == 0:
        k_med += 1
    if k_med > 1:
        out = out - medfilt(out, kernel_size=(k_med, 1))
    # Zero-phase Gaussian
    k_len = max(3, int(round(params.gaussian_len_ms * fs / 1000.0)))
    if k_len % 2 == 0:
        k_len += 1
    sigma = max(1.0, params.gaussian_sigma_ms * fs / 1000.0)
    g = windows.gaussian(M=k_len, std=sigma, sym=True)
    g = (g / g.sum()).astype(np.float64)
    # filter each channel with filtfilt
    pad_needed = 3 * (len(g) - 1)
    if out.shape[0] > pad_needed:
        for ch in range(out.shape[1]):
            out[:, ch] = out[:, ch] - filtfilt(
                g, [1.0], out[:, ch],
                padlen=min(pad_needed, out.shape[0] - 1)
            )
    return out

def _snippets_around(traces: np.ndarray, trigs: np.ndarray, nbefore: int, nafter: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract snippets (n_events, win, n_chan) and a mask of which triggers were valid.
    """
    win = nbefore + nafter + 1
    n_chan = traces.shape[1]
    keep = []
    snips = []
    for i, t in enumerate(trigs):
        a = t - nbefore
        b = t + nafter + 1
        if a >= 0 and b <= traces.shape[0]:
            snips.append(traces[a:b, :])
            keep.append(i)
    if len(snips) == 0:
        return np.zeros((0, win, n_chan), dtype=traces.dtype), np.zeros((0,), dtype=bool)
    S = np.stack(snips, axis=0)
    mask = np.zeros((trigs.size,), dtype=bool)
    mask[np.asarray(keep, dtype=int)] = True
    return S, mask


def _pca_template_block_per_channel(snips: np.ndarray, center: bool) -> np.ndarray:
    """
    Build a per-channel template using the 1st PC of (events x time) for each channel.
    snips: (n_events, win, n_chan)
    returns template (win, n_chan)
    """
    n_ev, win, n_ch = snips.shape
    if n_ev == 0:
        return np.zeros((win, n_ch), dtype=np.float64)
    if n_ev == 1:
        return snips[0].astype(np.float64)

    templ = np.zeros((win, n_ch), dtype=np.float64)
    for ch in range(n_ch):
        X = snips[:, :, ch]  # (n_events, win)
        if center:
            X = X - X.mean(axis=0, keepdims=True)
        # PCA over features=win
        p = PCA(n_components=1)
        p.fit(X)
        comp0 = p.components_[0]  # (win,)
        # Align sign with mean snippet
        m = snips[:, :, ch].mean(axis=0)
        if np.sum(comp0 * m) < 0:
            comp0 = -comp0
        templ[:, ch] = comp0.astype(np.float64)
    return templ


def _fit_scale(trace_patch: np.ndarray, templ_patch: np.ndarray) -> float:
    """
    Single scalar amplitude (least-squares) to minimize ||trace - a*templ||_2.
    """
    num = np.dot(trace_patch.ravel(), templ_patch.ravel())
    den = (np.linalg.norm(templ_patch) ** 2) + 1e-12
    return float(num / den)


def _apply_interp_ramp(
    buf: np.ndarray, start_idx: int, end_idx: int, ramp_fraction: float
):
    """
    Linearly glide baseline from current edge value to 0 over [start_idx, end_idx).
    """
    if end_idx <= start_idx or start_idx < 0 or end_idx > buf.shape[0]:
        return
    # value at start edge (per-channel)
    v0 = buf[start_idx, :].astype(np.float64, copy=True) * ramp_fraction
    L = end_idx - start_idx
    ramp = (v0[None, :] * (1.0 - np.linspace(0, 1, L, endpoint=False)[:, None]))
    buf[start_idx:end_idx, :] -= ramp.astype(buf.dtype)


# ------------------------- The preprocessor -----------------------------------

class RemoveStimPCARecording(BasePreprocessor):
    """
    Full artifact-correction pipeline as a SpikeInterface preprocessor.

    Implements:
      1) Global drift removal (rolling median + zero-phase Gaussian)
      2) Pulse-aligned matrix construction
      3) PCA template per block (first-pulse handling)
      4) Template subtraction (optional per-event scaling)
      5) Interpulse drift ramp

    You can pass list_labels per segment. If not provided and
    params.detect_triggers=True, triggers are detected on params.trig_channel.
    """

    def __init__(self, recording, params: Optional[ArtifactParams] = None,
                list_triggers: Optional[List[Sequence[int]]] = None,
                list_labels: Optional[List[Sequence[object]]] = None):
        super().__init__(recording)
        if list_triggers is None:
            raise ValueError("remove_stim_pca now requires list_triggers (one array per segment).")

        self.params = params or ArtifactParams()
        self.fs = float(recording.get_sampling_frequency())
        self.nbefore = int(round(self.params.samples_before))
        self.nafter  = int(round(self.params.samples_after))
        self.win_len = self.nbefore + self.nafter + 1

        nseg = recording.get_num_segments()
        if len(list_triggers) != nseg:
            raise ValueError(f"list_triggers length {len(list_triggers)} != #segments {nseg}")

        self._list_triggers = [np.asarray(t, dtype=np.int64) for t in list_triggers]

        if list_labels is None:
            self._list_labels = [np.zeros_like(t) for t in self._list_triggers]
        else:
            if len(list_labels) != nseg:
                raise ValueError(f"list_labels length {len(list_labels)} != #segments {nseg}")
            self._list_labels = [np.asarray(l) for l in list_labels]

        self._clean_cache = {}
        self._trigs_cache = {}
        self._blocks_cache = {}
        self._tpl_cache = {}

        for seg_index, parent in enumerate(recording._recording_segments):
            seg = _RemoveStimPCASegment(self, parent, seg_index)
            self.add_recording_segment(seg)

        self._kwargs = dict(
            recording=recording,
            params=asdict(self.params),
            list_triggers=[t.tolist() for t in self._list_triggers],
            list_labels=[l.tolist() for l in self._list_labels],
        )

    # --------- Lazy preparation (runs once per segment) -----------------------

    def _prepare_segment(self, seg_index: int):
        if seg_index in self._clean_cache:
            return

        parent_seg = self._recording_segments[seg_index].parent_recording_segment
        nsamp = parent_seg.get_num_samples()
        nchan = self.get_num_channels()

        raw = parent_seg.get_traces(0, nsamp, slice(None))
        clean = _global_drift_remove(raw, self.fs, self.params)
        self._clean_cache[seg_index] = clean

        trigs = self._list_triggers[seg_index]
        self._trigs_cache[seg_index] = trigs

        # If you no longer build “blocks”, you can set a single block or skip blocks entirely.
        # Keep a single block for compatibility with the existing loop:
        if trigs.size == 0:
            self._blocks_cache[seg_index] = []
            self._tpl_cache[seg_index] = {}
            return

        blocks = [np.arange(trigs.size, dtype=int)]  # single block
        self._blocks_cache[seg_index] = blocks

        # Build PCA template(s)
        templates = {}
        for b, idxs in enumerate(blocks):
            use_idxs = idxs
            if self.params.first_pulse_special and self.params.exclude_first_n_for_pca > 0:
                k = self.params.exclude_first_n_for_pca
                use_idxs = idxs[k:] if idxs.size > k else idxs[0:0]

            S, _ = _snippets_around(clean, trigs[use_idxs], self.nbefore, self.nafter)
            if S.shape[0] == 0:
                templates[b] = np.zeros((self.win_len, nchan), dtype=np.float64)
            else:
                templates[b] = _pca_template_block_per_channel(S, self.params.center_snippets)

        self._tpl_cache[seg_index] = templates

    # ------------- Core: return corrected traces for any window ---------------

    def _get_traces(self, segment_index, start_frame, end_frame, channel_indices):
        self._prepare_segment(segment_index)

        clean = self._clean_cache[segment_index]
        trigs = self._trigs_cache[segment_index]
        blocks = self._blocks_cache[segment_index]
        tpls   = self._tpl_cache[segment_index]

        # requested channels & mask
        # convert slice(...) to explicit indices
        if isinstance(channel_indices, slice):
            ch_idx = np.arange(clean.shape[1])[channel_indices]
        else:
            ch_idx = np.asarray(channel_indices)
        local = clean[start_frame:end_frame, :][:, ch_idx].copy()

        nb, na, win = self.nbefore, self.nafter, self.win_len

        # events inside window
        in_win = np.where((trigs >= start_frame - na) & (trigs <= end_frame + nb))[0]
        if in_win.size == 0:
            return local

        # Iterate events by block
        # Precompute mapping from event index -> block id
        ev_to_block = np.empty(trigs.size, dtype=int)
        for b, idxs in enumerate(blocks):
            ev_to_block[idxs] = b

        # For inter-pulse ramp, we need next-event index (in local window coords)
        ramp_tail = int(round(self.params.ramp_tail_ms * self.fs / 1000.0))

        for ev_global_idx in in_win:
            trig = trigs[ev_global_idx]
            b_id = ev_to_block[ev_global_idx]
            T = tpls.get(b_id, None)
            if T is None or T.size == 0:
                continue

            # local indices
            a = trig - nb - start_frame
            b = trig + na + 1 - start_frame

            # Clip to local buffer
            ta = max(a, 0)
            tb = min(b, local.shape[0])
            if tb <= ta:
                continue

            # template slice corresponding to ta:tb
            Ta = nb - (ta - a)
            Tb = Ta + (tb - ta)
            tpl_slice = T[Ta:Tb, :][:, ch_idx]          # (len, n_ch)
            trace_patch = local[ta:tb, :]               # (len, n_ch)

            # Per-channel scalar amplitude: (len,n)* (len,n) -> (n,)
            if self.params.scale_amplitude:
                num = np.sum(trace_patch * tpl_slice, axis=0)
                den = np.sum(tpl_slice * tpl_slice, axis=0) + 1e-12
                amp = num / den                           # shape (n_ch,)
            else:
                amp = 1.0

            # Subtract per-channel
            if np.ndim(amp) == 0:
                trace_patch -= (amp * tpl_slice).astype(trace_patch.dtype)
            else:
                trace_patch -= (tpl_slice * amp[None, :]).astype(trace_patch.dtype)

            # write back
            local[ta:tb, :] = trace_patch

            # Interpulse ramp: from (artifact end + ramp_tail) to next trigger
            if self.params.interp_ramp:
                art_end = min(tb + ramp_tail, local.shape[0])
                # next event inside current segment
                if ev_global_idx + 1 < trigs.size:
                    nxt = trigs[ev_global_idx + 1] - start_frame
                    if nxt > art_end:
                        _apply_interp_ramp(local, art_end, min(nxt, local.shape[0]), self.params.ramp_fraction)

        return local


class _RemoveStimPCASegment(BasePreprocessorSegment):
    def __init__(self, parent_proc: RemoveStimPCARecording, parent_segment, seg_index: int):
        super().__init__(parent_segment)
        self._parent_proc = parent_proc
        self._seg_index = seg_index

    def get_traces(self, start_frame, end_frame, channel_indices):
        return self._parent_proc._get_traces(self._seg_index, start_frame, end_frame, channel_indices)


# Functional API (SpikeInterface style)
remove_stim_pca = define_function_handling_dict_from_class(
    source_class=RemoveStimPCARecording, name="remove_stim_pca"
)
