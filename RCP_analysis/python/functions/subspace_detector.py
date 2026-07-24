"""Standalone subspace-CFAR spike detector (portable, no walking_analysis deps).

Needs: numpy, scipy, tqdm, and a SpikeInterface recording.

Pipeline:
    1. basis = build_subspace_basis(my_templates, rank=3)     # from sorted units / existing templates
    2. peaks = subspace_detect_cfar(recording, basis, cfar_alpha=1e-4)

    # Optional amplitude gate (suppresses chewing-band bursts):
    # peaks = filter_peaks_by_local_sigma(peaks, recording, k_amp=3.5)

Reference: Kraut & Scharf 1999, CFAR-F subspace detector.
"""
import numpy as np
from tqdm import tqdm


def build_subspace_basis(waveforms, rank=3):
    """Truncated-SVD signal subspace from aligned spike waveforms / templates.

    Parameters
    ----------
    waveforms : (n_wave, N) array
        Trough-aligned spike snippets or per-unit templates, all length N.
        Alignment matters: same trough sample across rows.
    rank : int
        Number of SVD components to retain.

    Returns
    -------
    basis : (N, rank) float32 ndarray
        Orthonormal columns spanning the spike signal subspace.
        NOT mean-centered — the first component is the dominant spike shape.
    """
    X = np.asarray(waveforms, dtype="float64")            # (n_wave, N)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)      # Vt rows span the N-sample space
    basis = np.ascontiguousarray(Vt[:rank].T).astype("float32")   # (N, rank), orthonormal cols
    var = (np.linalg.svd(X, compute_uv=False)[:rank] ** 2).sum() / (X ** 2).sum()
    print(f"[build_basis] N={X.shape[1]} rank={rank}  cum-var={var:.3f}")
    return basis


def subspace_detect_cfar(recording, basis, *, cfar_alpha=1e-4, peak_sign="neg",
                         refractory_ms=0.5, snap_ms=0.6, progress_bar=True):
    """Subspace matched detector with CFAR-F threshold (Kraut-Scharf 1999).

    Slides the r-dim signal subspace `basis` (N×r, orthonormal cols) over each
    channel. At position t the scale-invariant CFAR-F statistic is:

        F(t) = s_sig / s_orth,  s_sig = ||Uᵀ x_t||²,  s_orth = ||x_t||² − s_sig

    Detects where F > f.ppf(1-α, r, N-r)·r/(N-r), snaps to trough within ±snap_ms,
    enforces refractory period. Returns a structured peaks array compatible with
    SpikeInterface's detect_peaks output format.

    Parameters
    ----------
    recording : SpikeInterface recording object
    basis : (N, rank) float32 ndarray
        Orthonormal subspace basis, e.g. from build_subspace_basis().
    cfar_alpha : float
        CFAR false-alarm rate. Lower = stricter (fewer detections). Default 1e-4.
    peak_sign : str
        'neg' or 'pos'. Direction of spike to snap to after detection.
    refractory_ms : float
        Minimum inter-peak interval in ms.
    snap_ms : float
        Window (±) around each detected F-peak to search for the actual trough.
    progress_bar : bool
        Show tqdm progress bar.

    Returns
    -------
    peaks : structured ndarray
        Fields: sample_index (int64), channel_index (int64),
                segment_index (int64), amplitude (float32).
        Sorted by (segment_index, sample_index).

    Notes
    -----
    RAM: reads each channel's FULL trace at once and FFT-correlates — peak
    memory ≈ one channel's length × ~4. Processes one channel at a time.
    """
    from scipy.signal import correlate
    from scipy.stats import f as _f

    fs = recording.get_sampling_frequency()
    n_ch = recording.get_num_channels()
    n_seg = recording.get_num_segments()
    ch_ids = recording.get_channel_ids()

    U = np.asarray(basis, dtype="float32")            # (N, r)
    N, r = U.shape
    thr = float(_f.ppf(1 - cfar_alpha, r, N - r) * r / (N - r))
    w_snap = int(snap_ms * 1e-3 * fs)
    min_sep = int(refractory_ms * 1e-3 * fs)
    ones_N = np.ones(N, dtype="float32")

    records = []
    for seg_idx in range(n_seg):
        seg_len = recording.get_num_samples(segment_index=seg_idx)
        desc = f"[subspace CFAR] seg{seg_idx}" if n_seg > 1 else "[subspace CFAR]"
        ch_iter = tqdm(range(n_ch), desc=desc, unit="ch", disable=not progress_bar)
        for ci in ch_iter:
            trace = recording.get_traces(start_frame=0, end_frame=seg_len,
                                         channel_ids=[ch_ids[ci]], segment_index=seg_idx,
                                         return_in_uV=True).flatten().astype("float32")
            s_sig = np.zeros(len(trace), dtype="float32")
            for j in range(r):
                s_sig += correlate(trace, U[:, j], mode="same", method="fft") ** 2
            s_tot = correlate(trace ** 2, ones_N, mode="same", method="fft")
            F = s_sig / np.maximum(s_tot - s_sig, 1e-9)

            above = F > thr
            if not above.any():
                continue
            edges = np.diff(above.astype(np.int8))
            starts = np.where(edges == 1)[0] + 1
            ends = np.where(edges == -1)[0]
            if above[0]:  starts = np.r_[0, starts]
            if above[-1]: ends = np.r_[ends, len(F) - 1]
            pk = np.array([s + int(np.argmax(F[s:e + 1])) for s, e in zip(starts, ends)], int)

            pk = pk[(pk >= w_snap) & (pk < len(trace) - w_snap)]
            if len(pk):
                fn = np.argmin if peak_sign == "neg" else np.argmax
                pk = np.array([p - w_snap + int(fn(trace[p - w_snap:p + w_snap])) for p in pk], int)
            if len(pk) > 1:
                pk.sort()
                keep = [0]
                for i in range(1, len(pk)):
                    if pk[i] - pk[keep[-1]] >= min_sep:
                        keep.append(i)
                pk = pk[keep]
            for p in pk:
                records.append((int(p), int(ci), int(seg_idx), float(trace[p])))
            ch_iter.set_postfix(peaks=len(records))

    dtype = np.dtype([("sample_index", np.int64), ("channel_index", np.int64),
                      ("segment_index", np.int64), ("amplitude", np.float32)])
    if not records:
        return np.zeros(0, dtype=dtype)
    peaks = np.array(records, dtype=dtype)
    return peaks[np.lexsort((peaks["sample_index"], peaks["segment_index"]))]


def filter_peaks_by_local_sigma(peaks, recording, *, k_amp=3.5, w_ms=100.0,
                                peak_sign="neg"):
    """Local-σ amplitude pre-gate.

    Keeps peaks whose |voltage| exceeds k_amp × σ_local(t), where
    σ_local = 1.4826 × rolling-MAD over a w_ms window. This raises the bar
    inside chewing/movement bursts where global noise estimates underestimate
    the local signal level. Single-segment assumption (sample_index global).

    Parameters
    ----------
    peaks : structured ndarray
        Output of subspace_detect_cfar().
    recording : SpikeInterface recording object
    k_amp : float
        Multiplier on local σ. Higher = stricter. Default 3.5.
    w_ms : float
        Rolling MAD window in ms. Default 100.0.
    peak_sign : str
        'neg' or 'pos' (unused here — amplitude is always compared as |v|).

    Returns
    -------
    peaks[keep] : structured ndarray
        Subset of input peaks that pass the local-σ gate.
    """
    from scipy.ndimage import median_filter

    n = len(peaks)
    if n == 0:
        return peaks
    fs = recording.get_sampling_frequency()
    N = recording.get_num_samples()
    w_env = int(w_ms * 1e-3 * fs)
    ch_ids = recording.get_channel_ids()
    samp = peaks["sample_index"].astype(np.int64)
    chan = peaks["channel_index"].astype(np.int64)
    keep = np.zeros(n, bool)
    for ci in tqdm(np.unique(chan), desc=f"[local-σ gate] k={k_amp}", unit="ch"):
        idx = np.where(chan == ci)[0]
        trace = recording.get_traces(start_frame=0, end_frame=N,
                                     channel_ids=[ch_ids[int(ci)]],
                                     return_in_uV=True).flatten().astype("float32")
        sig_loc = 1.4826 * median_filter(np.abs(trace - median_filter(trace, w_env)), w_env)
        keep[idx] = np.abs(trace[samp[idx]]) > k_amp * sig_loc[samp[idx]]
    print(f"[local-σ gate] kept {int(keep.sum()):,}/{n:,} (k={k_amp}, w={w_ms}ms)")
    return peaks[keep]
