subspace_detector.py
====================

``RCP_analysis/python/functions/subspace_detector.py``

Standalone subspace-CFAR spike detector. A low-rank signal subspace is built
from aligned waveforms or unit templates, and the detector scores every sample
by the ratio of in-subspace to out-of-subspace energy — an F statistic whose
threshold gives a constant false-alarm rate independent of the local noise
level.

Reference: Kraut & Scharf 1999, CFAR-F subspace detector.

Requires ``numpy``, ``scipy``, ``tqdm``, ``numba``, ``torch``, and a
SpikeInterface recording. Used by
``preprocessing_scripts/UA_BR_analysis_ssmf.py``.

Typical use::

   basis = rcp.build_subspace_basis(my_templates, rank=3)
   peaks = rcp.subspace_detect_cfar(recording, basis, cfar_alpha=1e-4)

   # optional: suppress chewing-band bursts
   peaks = rcp.filter_peaks_by_local_sigma(peaks, recording, k_amp=3.5)

.. py:currentmodule:: RCP_analysis.python.functions.subspace_detector

Functions
---------

.. py:function:: build_subspace_basis(waveforms, rank=3)

   Truncated-SVD signal subspace from aligned spike waveforms or per-unit
   templates.

   :param waveforms: ``(n_wave, N)`` — rows are trough-aligned snippets or
      templates, all of length ``N``. **Alignment matters**: every row must have
      its trough at the same sample.
   :param int rank: number of components to keep.
   :returns: ``(N, rank)`` ``float32`` with orthonormal columns.

   The basis is deliberately **not** mean-centered, so the first component is
   the dominant spike shape itself. Prints the cumulative variance captured.

.. py:function:: subspace_detect_cfar(recording, basis, *, cfar_alpha=1e-4, peak_sign="neg", refractory_ms=0.5, snap_ms=0.6, chunk_s=30.0, progress_bar=True)

   Subspace matched detector with a CFAR-F threshold.

   :param recording: SpikeInterface recording (multi-segment supported).
   :param basis: ``(N, r)`` orthonormal-column basis from
      :func:`build_subspace_basis`.
   :param float cfar_alpha: false-alarm rate; the threshold is
      ``F.ppf(1 - alpha, r, N - r) * r / (N - r)``.
   :param str peak_sign: ``"neg"`` snaps to the local minimum, anything else to
      the maximum.
   :param float refractory_ms: minimum separation between kept peaks, per channel.
   :param float snap_ms: half-width of the window searched when snapping the
      detection to the true trough.
   :param float chunk_s: seconds per chunk — the VRAM knob.
   :param bool progress_bar: show the per-chunk tqdm bar.
   :returns: structured array with fields ``sample_index`` (int64),
      ``channel_index`` (int64), ``segment_index`` (int64) and ``amplitude``
      (float32), sorted by segment then sample.

   The two convolutions (signal energy and total energy) run through
   ``torch.nn.functional.conv1d`` on CUDA when available and on CPU otherwise —
   the CPU path is a fallback, not a different algorithm. All channels in a
   time chunk go through one convolution, so runtime is I/O-bound rather than
   compute-bound. Chunks overlap by ``2N + snap`` samples and each channel
   carries its refractory state across the seam, so peaks at chunk boundaries
   are neither missed nor double-counted.

.. py:function:: filter_peaks_by_local_sigma(peaks, recording, *, k_amp=3.5, w_ms=100.0, peak_sign="neg")

   Amplitude gate against the *local* noise level: keep a peak only if
   ``|trough voltage| > k_amp * sigma_local``. Useful for suppressing
   chewing-band bursts, which raise the noise floor without being spikes.

   ``sigma_local`` is estimated block-wise (``1.4826 × MAD`` on a ``w_ms`` grid)
   and interpolated to the peak positions, since sigma is a slow envelope. The
   trough voltage is read from the trace in the same pass, so the test is exact
   and detector-agnostic — it does not reuse ``peaks["amplitude"]``, which is
   filter output for a matched filter.

   Single-segment only. Prints how many peaks survived.

Private helpers
---------------

.. py:function:: _subspace_pick_channel(Fc, tc, thr, lag, w_snap, min_sep, a, t0, t1, seg_len, last_kept, neg)

   Numba-compiled per-channel peak picker used inside
   :func:`subspace_detect_cfar`. For each contiguous region where the F
   statistic exceeds the threshold it takes the argmax, snaps to the trough (or
   peak) within ``±w_snap``, converts to a global sample index, discards
   anything outside the chunk's committed window, then enforces the refractory
   period.

   :returns: ``(kept_sample_indices, last_kept)`` — ``last_kept`` is carried into
      the next chunk so the refractory period spans the seam.
