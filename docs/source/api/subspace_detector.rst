subspace_detector.py
========================

``RCP_analysis/python/functions/subspace_detector.py``

Standalone subspace-CFAR spike detector. A low-rank signal subspace is built
from aligned waveforms or unit templates, and the detector scores every sample
by the ratio of in-subspace to out-of-subspace energy. The resulting F statistic
is thresholded for a constant-false-alarm-rate (CFAR) detector.

Reference: Kraut & Scharf 1999, CFAR-F subspace detector.

Requires ``numpy``, ``scipy``, ``tqdm``, ``numba``, ``torch``, and a
SpikeInterface recording. Used by
``preprocessing_scripts/UA_BR_analysis_ssmf.py``.

Typical use::

   basis = rcp.build_subspace_basis(my_templates, rank=3)
   peaks = rcp.subspace_detect_cfar(recording, basis, cfar_alpha=1e-4)
   peaks = rcp.filter_peaks_by_local_sigma(peaks, recording, k_amp=3.5)

.. py:currentmodule:: RCP_analysis.python.functions.subspace_detector


Pipeline overview
-----------------

Three stages. Stage 1 builds the reusable basis once. Stages 2 and 3 run
on the recording.

.. code-block:: text

   aligned waveforms / unit templates
                |
                v
   build_subspace_basis(..., rank=3)
                |
                v
        basis U (N, r)
                |
                v
   subspace_detect_cfar(recording, U)
                |
                v
        detected peaks
                |
                v
   filter_peaks_by_local_sigma(...)
                |
                v
        final peaks


Detect with a CFAR-F threshold
------------------------------

Concept
^^^^^^^

The basis spans an r-dimensional signal subspace inside the N-sample waveform
space. For each sliding waveform x::

   s_sig = ||U^T x||^2
   s_tot = ||x||^2
   F     = s_sig / (s_tot - s_sig)

``s_sig`` is the energy explained by the learned spike subspace.
``s_tot - s_sig`` is the energy outside it.

In plain terms: a spike should resemble the waveform family used to build the
basis, so a large fraction of its energy should lie inside the subspace.

The threshold comes from the F distribution::

   thr = F.ppf(1 - cfar_alpha, r, N - r) * r / (N - r)

This makes the detector CFAR: the decision is based on the ratio of signal
energy to residual energy rather than absolute amplitude.

Inputs
^^^^^^

``recording``
   SpikeInterface recording. Multi-segment recordings are supported.

``basis``
   ``(N, r)`` orthonormal basis returned by
   :func:`build_subspace_basis`.

``cfar_alpha``
   False-alarm probability parameter. Default ``1e-4``.

``peak_sign``
   ``"neg"`` searches for a trough; anything else searches for a peak.

``refractory_ms``
   Minimum separation between detections on the same channel.

``snap_ms``
   Half-width of the raw-trace search used to move the detection onto the
   actual trough or peak.

``chunk_s``
   Recording duration processed per chunk. Main memory/VRAM control.

``progress_bar``
   Whether to show the ``tqdm`` progress bar.


How the detector runs
^^^^^^^^^^^^^^^^^^^^^

The basis is first converted into the format expected by PyTorch::

   U = np.ascontiguousarray(basis, dtype="float32")
   N, r = U.shape

   w_sig = torch.from_numpy(
       np.ascontiguousarray(U.T)
   ).unsqueeze(1).to(dev)

   w_tot = torch.ones(
       1, 1, N,
       dtype=torch.float32,
       device=dev
   )

The recording is processed in overlapping chunks::

   chunk = int(chunk_s * fs)
   margin = 2 * N + w_snap

   a = max(0, t0 - margin)
   b = min(seg_len, t1 + margin)

The extra margin prevents a spike near a chunk boundary from losing part of
its waveform window. Only detections inside ``[t0, t1)`` are committed, so
overlap does not create duplicates.

Two batched Conv1D operations calculate the required energies::

   x = torch.from_numpy(blkT).unsqueeze(1).to(dev)

   s_sig = (torch.nn.functional.conv1d(x, w_sig) ** 2).sum(1)

   s_tot = torch.nn.functional.conv1d(
       x ** 2, w_tot
   ).squeeze(1)

   F = (
       s_sig /
       torch.clamp(s_tot - s_sig, min=1e-9)
   ).cpu().numpy()

The tensor shapes are::

   x       (n_ch, 1, L)
   w_sig   (r,    1, N)
   output  (n_ch, r, L-N+1)

This is the main optimization: channels are processed as the Conv1D batch,
while all r basis vectors are processed as output channels.

``s_sig`` therefore gives ``||U^T x||²`` and ``s_tot`` gives ``||x||²``.

PyTorch ``conv1d`` performs cross-correlation rather than kernel reversal,
which is exactly the sliding dot product needed here.


Peak picking
^^^^^^^^^^^^

After the F statistic is calculated, each channel is passed to the Numba
helper :func:`_subspace_pick_channel`.

The picker:

#. finds contiguous regions where ``F > threshold``;
#. keeps the maximum F value from each region;
#. shifts the detection by ``lag = N // 2``;
#. searches ``±snap_ms`` in the raw trace;
#. snaps to the actual trough or peak;
#. removes detections outside the chunk's committed region;
#. applies the per-channel refractory period.

``last_kept`` is carried from one chunk to the next, so a detection close to a
chunk boundary is still subject to the same refractory period.


Local-sigma amplitude gate
--------------------------

Concept
^^^^^^^

The CFAR detector measures waveform shape. It does not require a spike to have
a particular absolute amplitude.

The optional amplitude gate adds::

   |trough voltage| > k_amp * sigma_local

The final detection therefore needs to pass both tests:

* waveform looks like the learned spike subspace;
* amplitude is sufficiently large relative to local noise.

The local noise estimate is based on::

   sigma_local = 1.4826 * MAD

MAD is preferred because spikes are outliers and would inflate a standard
deviation estimate.

The noise is calculated on a coarse ``w_ms`` grid and interpolated to each
peak because noise level changes slowly.


Key code
^^^^^^^^

::

   b = trp.reshape(nb, blk, n_ch)

   med = np.nanmedian(b, axis=1)

   mad = np.nanmedian(
       np.abs(b - med[:, None, :]),
       axis=1
   )

   sigma_grid[...] = (1.4826 * mad).T

   sig_at = np.interp(
       samp[m],
       centers[good],
       sg[good]
   )

   keep[m] = amp[m] > k_amp * sig_at

The trough voltage is read directly from the recording rather than reusing
``peaks["amplitude"]``. This keeps the amplitude test detector-independent.


Private helper
--------------

.. py:function:: _subspace_pick_channel(...)

   Numba-compiled per-channel picker used by
   :func:`subspace_detect_cfar`.

   It converts threshold crossings into one candidate per event, snaps each
   candidate to the raw waveform extremum, removes candidates outside the
   committed chunk, and applies the refractory period.

   ``last_kept`` is returned so the refractory period continues across chunk
   boundaries.


How to build your own subspace basis
-------------------------------------

Concept
^^^^^^^

A matched filter uses one template. A subspace detector uses several
orthonormal components to represent a family of related spike shapes.

This allows different spikes to have slightly different shapes while still
belonging to the same learned signal family.

Purpose
^^^^^^^

Build the waveform family once from representative spike waveforms, then
reuse the resulting basis for detection.

Inputs
^^^^^^

``waveforms``
   ``(n_wave, N)`` array. Each row is a spike waveform or unit template of
   length ``N``.

   **Alignment is critical:** the trough should occur at the same sample in
   every waveform.

``rank``
   Number of SVD components to keep.

Build the basis
^^^^^^^^^^^^^^^

::

   X = np.asarray(waveforms, dtype="float64")

   _, _, Vt = np.linalg.svd(
       X,
       full_matrices=False
   )

   basis = np.ascontiguousarray(
       Vt[:rank].T
   ).astype("float32")

The SVD is::

   X = U S Vt

``Vt`` contains directions in the N-sample waveform space, so its first
``rank`` rows become the detector basis.

The result is::

   basis.shape == (N, rank)

and the columns are orthonormal.

The basis is deliberately not mean-centered. The first component therefore
captures the dominant spike waveform itself.

Check the basis before using it::

   np.allclose(
       basis.T @ basis,
       np.eye(basis.shape[1]),
       atol=1e-5
   )

A practical source is a collection of high-SNR unit templates from Kilosort
or SpikeInterface sorting.

The cumulative energy captured by the chosen rank can be checked with::

   var = (
       np.linalg.svd(X, compute_uv=False)[:rank] ** 2
   ).sum() / (X ** 2).sum()

Save the basis as ``.npy`` and reuse it instead of rebuilding it for every
recording.


Runtime optimization
---------------------

* Batch all channels into one Conv1D call.
* Batch all r basis vectors as Conv1D output channels.
* Use CUDA when available.
* Process the recording in chunks to control VRAM/RAM.
* Keep tensors ``float32`` and contiguous.
* Compile the irregular peak-picker with ``@njit(cache=True)``.
* Use preallocated Numba buffers instead of growing Python lists.
* Transfer the F statistic back to CPU once per chunk.
* Build the SVD basis once and reuse it across recordings.


Parameter reference
-------------------

Values below assume ``fs = 30 kHz`` and a ``(48, 3)`` basis.

====================  =========  =====================================================
Parameter             Default    Effect at 30 kHz
====================  =========  =====================================================
``cfar_alpha``        1e-4       threshold 0.5907 for N=48, r=3
rank                  3          number of retained SVD components
``refractory_ms``     0.5        15 samples minimum separation
``snap_ms``           0.6        18 samples searched around the detection
``chunk_s``           30.0       900,000 samples per chunk
``k_amp``             3.5        local-sigma amplitude multiplier
``w_ms``              100.0      3,000-sample sigma-estimation blocks
====================  =========  =====================================================

Derived constants:

====================  =========  ==============================================
Constant              Value       Meaning
====================  =========  ==============================================
``N``                 48          basis length, 1.60 ms at 30 kHz
``lag``               24          ``N // 2``
``margin``            114         ``2N + w_snap``
1.4826                --          MAD-to-sigma conversion
====================  =========  ==============================================