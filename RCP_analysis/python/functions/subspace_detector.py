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
from numba import njit

@njit(cache=True)
def _subspace_pick_channel(Fc, tc, thr, lag, w_snap, min_sep, a, t0, t1,
                           seg_len, last_kept, neg):
    """single channel: CFAR-F region → argmax → snap → commit window → refractory"""
    Lc = Fc.shape[0]
    Lt = tc.shape[0]

    buf = np.empty(Lc, np.int64)
    m = 0
    i = 0

    while i < Lc:
        if Fc[i] > thr:
            j = i
            best = i
            bestv = Fc[i]

            while j < Lc and Fc[j] > thr:
                if Fc[j] > bestv:
                    bestv = Fc[j]
                    best = j
                j += 1

            p = best + lag

            if w_snap <= p < Lt - w_snap:
                sidx = p - w_snap
                sval = tc[sidx]

                for k in range(p - w_snap, p + w_snap):
                    if neg:
                        if tc[k] < sval:
                            sval = tc[k]
                            sidx = k
                    else:
                        if tc[k] > sval:
                            sval = tc[k]
                            sidx = k

                g = sidx + a

                if t0 <= g < t1 and w_snap <= g < seg_len - w_snap:
                    buf[m] = g
                    m += 1

            i = j
        else:
            i += 1

    if m == 0:
        return np.empty(0, np.int64), last_kept

    g_arr = np.sort(buf[:m])

    out = np.empty(m, np.int64)
    n = 0
    lk = last_kept

    for idx in range(m):
        gp = g_arr[idx]

        if gp - lk >= min_sep:
            out[n] = gp
            n += 1
            lk = gp

    return out[:n], lk


def build_subspace_basis(waveforms, rank=3):
    """Truncated-SVD signal subspace from aligned spike waveforms / templates.

    waveforms : (n_wave, N) — rows are trough-aligned spike snippets or per-unit
                templates, all length N. Alignment matters: same trough sample.
    Returns (N, rank) float32 with orthonormal columns. NOT mean-centered — the
    first component is the dominant spike shape itself.
    """
    X = np.asarray(waveforms, dtype="float64")            # (n_wave, N)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)      # Vt rows span the N-sample space
    basis = np.ascontiguousarray(Vt[:rank].T).astype("float32")   # (N, rank), orthonormal cols
    var = (np.linalg.svd(X, compute_uv=False)[:rank] ** 2).sum() / (X ** 2).sum()
    print(f"[build_basis] N={X.shape[1]} rank={rank}  cum-var={var:.3f}")
    return basis


def subspace_detect_cfar(recording, basis, *, cfar_alpha=1e-4, peak_sign="neg",
                         refractory_ms=0.5, snap_ms=0.6, chunk_s=30.0, progress_bar=True):
    """Subspace matched detector with CFAR-F threshold (Kraut-Scharf 1999).

    GPU-accelerated via torch conv1d (device-agnostic: CUDA if available, else CPU
    fallback = no regression). Processes ALL channels per time-chunk in one conv, so
    the F statistic is compute-negligible and runtime is I/O-bound. Chunked with an
    overlap margin + per-channel refractory carry so seam spikes are neither missed
    nor doubled. Output dtype identical to before → downstream unchanged.

    basis : (N, r) float, orthonormal columns. chunk_s : seconds/chunk (VRAM knob).
    """
    import torch
    from scipy.stats import f as _f
    fs = recording.get_sampling_frequency()
    n_ch = recording.get_num_channels()
    n_seg = recording.get_num_segments()

    U = np.ascontiguousarray(basis, dtype="float32")          # (N, r)
    N, r = U.shape
    thr = float(_f.ppf(1 - cfar_alpha, r, N - r) * r / (N - r))
    lag = N // 2
    w_snap = int(snap_ms * 1e-3 * fs)
    min_sep = int(refractory_ms * 1e-3 * fs)
    neg = peak_sign == "neg"

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w_sig = torch.from_numpy(np.ascontiguousarray(U.T)).unsqueeze(1).to(dev)   # (r,1,N)
    w_tot = torch.ones(1, 1, N, dtype=torch.float32, device=dev)               # (1,1,N)

    chunk = int(chunk_s * fs)
    margin = 2 * N + w_snap                                   # 重疊: 接縫 peak 保完整窗 + snap

    samp_parts, chan_parts, seg_parts, amp_parts = [], [], [], []
    for seg_idx in range(n_seg):
        seg_len = recording.get_num_samples(segment_index=seg_idx)
        last_kept = np.full(n_ch, -(10 ** 12), dtype=np.int64)   # refractory 跨塊記憶
        desc = f"[subspace CFAR/{dev}]" + (f" seg{seg_idx}" if n_seg > 1 else "")
        for t0 in tqdm(range(0, seg_len, chunk), desc=desc, disable=not progress_bar):
            t1 = min(t0 + chunk, seg_len)
            a, b = max(0, t0 - margin), min(seg_len, t1 + margin)
            blk = recording.get_traces(start_frame=a, end_frame=b, segment_index=seg_idx,
                                       return_in_uV=True).astype("float32")   # (L, n_ch) 一次讀全 channel
            blkT = np.ascontiguousarray(blk.T)                                 # (n_ch, L)
            x = torch.from_numpy(blkT).unsqueeze(1).to(dev)
            s_sig = (torch.nn.functional.conv1d(x, w_sig) ** 2).sum(1)         # ‖Uᵀx‖²
            s_tot = torch.nn.functional.conv1d(x ** 2, w_tot).squeeze(1)       # ‖x‖²
            F = (s_sig / torch.clamp(s_tot - s_sig, min=1e-9)).cpu().numpy()   # (n_ch, L-N+1)
            for ci in range(n_ch):
                g, last_kept[ci] = _subspace_pick_channel(
                    F[ci], blkT[ci], thr, lag, w_snap, min_sep, a, t0, t1, seg_len,
                    last_kept[ci], neg)
                if len(g):
                    samp_parts.append(g)
                    chan_parts.append(np.full(len(g), ci, np.int64))
                    seg_parts.append(np.full(len(g), seg_idx, np.int64))
                    amp_parts.append(blkT[ci][g - a])

    dtype = np.dtype([("sample_index", np.int64), ("channel_index", np.int64),
                      ("segment_index", np.int64), ("amplitude", np.float32)])
    if not samp_parts:
        return np.zeros(0, dtype=dtype)
    peaks = np.empty(sum(len(p) for p in samp_parts), dtype=dtype)
    peaks["sample_index"] = np.concatenate(samp_parts)
    peaks["channel_index"] = np.concatenate(chan_parts)
    peaks["segment_index"] = np.concatenate(seg_parts)
    peaks["amplitude"] = np.concatenate(amp_parts).astype(np.float32)
    return peaks[np.lexsort((peaks["sample_index"], peaks["segment_index"]))]


def filter_peaks_by_local_sigma(peaks, recording, *, k_amp=3.5, w_ms=100.0, peak_sign="neg"):
    """Local-σ amplitude gate. σ_local via block-wise MAD on a w_ms grid + interp to peak
    positions (σ is a slow envelope → no per-sample rolling median). Trough voltage is read
    from the trace in the SAME pass (exact, detector-agnostic — matches the original
    trace[samp] test, not peaks['amplitude'] which is filter-output for MF). Single-segment."""
    n = len(peaks)
    if n == 0:
        return peaks
    fs = recording.get_sampling_frequency()
    N = recording.get_num_samples()
    n_ch = recording.get_num_channels()
    blk = max(1, int(w_ms * 1e-3 * fs))
    n_blk = int(np.ceil(N / blk))
    sigma_grid = np.full((n_ch, n_blk), np.nan, np.float32)

    samp = peaks["sample_index"].astype(np.int64)
    chan = peaks["channel_index"].astype(np.int64)
    amp = np.full(n, np.nan, np.float32)                 # true trough voltage, filled in-pass

    chunk = blk * 300                                    # multiple of blk → grid aligns; ~460MB/chunk
    for c0 in tqdm(range(0, N, chunk), desc=f"[local-σ gate] k={k_amp}", unit="chunk"):
        c1 = min(c0 + chunk, N)
        tr = recording.get_traces(start_frame=c0, end_frame=c1, return_in_uV=True).astype("float32")

        in_chunk = (samp >= c0) & (samp < c1)            # exact voltage, same pass, no re-read
        if in_chunk.any():
            amp[in_chunk] = tr[samp[in_chunk] - c0, chan[in_chunk]]

        T = tr.shape[0]; nb = int(np.ceil(T / blk)); pad = nb * blk - T
        trp = np.pad(tr, ((0, pad), (0, 0)), constant_values=np.nan) if pad else tr
        b = trp.reshape(nb, blk, n_ch)
        med = np.nanmedian(b, axis=1)
        mad = np.nanmedian(np.abs(b - med[:, None, :]), axis=1)
        sigma_grid[:, c0 // blk:c0 // blk + nb] = (1.4826 * mad).T

    centers = (np.arange(n_blk) + 0.5) * blk
    amp = np.abs(amp)
    keep = np.zeros(n, bool)
    for ci in np.unique(chan):
        m = chan == ci
        sg = sigma_grid[ci]; good = np.isfinite(sg) & (sg > 0)
        if not good.any():
            continue
        sig_at = np.interp(samp[m], centers[good], sg[good])
        keep[m] = amp[m] > k_amp * sig_at
    print(f"[local-σ gate] kept {int(keep.sum()):,}/{n:,} (k={k_amp}, w={w_ms}ms)")
    return peaks[keep]
