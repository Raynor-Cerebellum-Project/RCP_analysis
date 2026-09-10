subspace_detector.py
====================

Workflow reference for the subspace-CFAR detector used by
``preprocessing_scripts/UA_BR_analysis_ssmf.py``.

This page explains **how the method works and how to run it**. For parameter
signatures and return types see :doc:`../../api/subspace_detector`.

Implementation: ``RCP_analysis/python/functions/subspace_detector.py`` (219 lines).
The module is deliberately standalone — it needs only ``numpy``, ``scipy``,
``tqdm``, ``numba``, ``torch`` and a SpikeInterface recording, so the same file
is shared with the walking-analysis pipeline without either project depending
on the other.

Reference: Kraut & Scharf 1999, CFAR-F subspace detector.


Pipeline overview
-----------------

Three stages. Stage 1 runs once, offline. Stages 2 and 3 run per session.

.. code-block:: text

   aligned waveforms / unit templates        (n_wave, N)
        |
        |  STAGE 1   build_subspace_basis(waveforms, rank=3)      offline, once
        v
   basis U                                   (N, r) orthonormal columns
        |                                    saved as .npy, reused for every session
        |
        |                 highpass-filtered recording   (n_samples, n_ch)
        |                        |
        +------------------------+
                                 |
                                 |  STAGE 2   subspace_detect_cfar(recording, U)
                                 v
                            peaks            structured array, 4 fields
                                 |
                                 |  STAGE 3   filter_peaks_by_local_sigma(peaks, recording)
                                 v
                            peaks (pruned)

The contract between stages is the ``peaks`` structured array::

   np.dtype([("sample_index",  np.int64),     # sample within its segment
             ("channel_index", np.int64),     # row index, not electrode ID
             ("segment_index", np.int64),
             ("amplitude",     np.float32)])  # true trough voltage in uV

This is the same dtype SpikeInterface's ``detect_peaks`` returns, so the
detector can be swapped without touching anything downstream.


Stage 1 — Build the subspace basis
-----------------------------------

Concept
^^^^^^^

A classical matched filter correlates the signal against **one** template and
thresholds the result. Real spikes are not one shape: different units,
different electrode distances and different moments all produce slightly
different waveforms, and a single template misses the ones that do not match.

A subspace detector replaces the single template with a **low-dimensional
subspace**. Take r orthonormal vectors of length N; any waveform that can be
written as a linear combination of those r vectors counts as spike-like. The
test becomes membership of a family rather than similarity to one template.

Stage 1 constructs that family from data, by truncated SVD of a set of aligned
waveforms.

In plain terms
^^^^^^^^^^^^^^

Stack a few hundred known-good spike snippets (or per-unit templates) as rows
of a matrix. Ask the SVD which directions in the N-sample waveform space carry
the most energy. Keep the top r. Those r directions are the basis.

Two properties make this different from ordinary PCA:

#. The data are **not mean-centered**. Ordinary PCA subtracts the mean to study
   variation around it; here the mean *is* the dominant spike shape, and it is
   exactly what we want to detect. Subtracting it would throw away the signal.
#. **Alignment dominates quality.** Every input row must have its trough at the
   same sample. If the rows are jittered, the leading singular vectors describe
   time-shift rather than shape, and rank 3 is consumed by jitter instead of by
   waveform variability. Misalignment is the most common cause of a bad basis.

Inputs
^^^^^^

``waveforms`` : ``(n_wave, N)`` float array
   Rows are trough-aligned spike snippets or per-unit templates, all of length
   N. Sources in practice: per-unit mean waveforms from a Kilosort or
   SpikeInterface sorting, or manually selected high-SNR snippets.

``rank`` : int, default 3
   Number of components to keep. Must be no larger than ``n_wave`` — see
   :ref:`failure-rank`.

Key code
^^^^^^^^

``subspace_detector.py`` lines 95-98:

.. code-block:: python

   X = np.asarray(waveforms, dtype="float64")            # (n_wave, N)
   _, _, Vt = np.linalg.svd(X, full_matrices=False)      # Vt rows span the N-sample space
   basis = np.ascontiguousarray(Vt[:rank].T).astype("float32")   # (N, rank), orthonormal cols
   var = (np.linalg.svd(X, compute_uv=False)[:rank] ** 2).sum() / (X ** 2).sum()

The SVD factorises ``X = U S Vt``. Only ``Vt`` is used, because its rows live
in the N-sample waveform space, whereas the columns of ``U`` index *which
waveform* — irrelevant here. ``full_matrices=False`` makes ``Vt`` have shape
``(min(n_wave, N), N)`` instead of ``(N, N)``.

``var`` is the fraction of total waveform energy captured by the first ``rank``
components, printed as ``cum-var``. It is the only diagnostic for choosing the
rank. Note the SVD is computed twice — the singular values from line 96 could
be reused; for a matrix this small the waste is negligible.

Outputs
^^^^^^^

``(N, rank)`` ``float32`` array with orthonormal columns, plus a printed line::

   [build_basis] N=48 rank=3  cum-var=0.912

Save it as ``.npy`` and reuse it. The basis shipped with this repo is
``config/median_extremum_basis_norm_UA_PortA_r3.npy``, shape ``(48, 3)`` — that
is 48 samples, or 1.6 ms at 30 kHz, and 3 components.

Verify a basis before trusting it::

   np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-5)


Stage 2 — Detect with a CFAR-F threshold
-----------------------------------------

Concept
^^^^^^^

Split the N-dimensional waveform space into the signal subspace spanned by the
basis (r dimensions) and everything left over (N-r dimensions). Every sliding
window **x** decomposes uniquely into a part inside the subspace and a part
orthogonal to it. The detector scores each window by the ratio of those two
energies::

   s_sig = ‖Uᵀx‖²                    energy inside the signal subspace
   s_tot = ‖x‖²                      total energy
   F     = s_sig / (s_tot − s_sig)   in-subspace energy / residual energy

Because U has orthonormal columns, ‖Uᵀx‖² is already the projection energy, so
no projection matrix ever has to be formed — two convolutions give everything.

This F is the generalised likelihood ratio for the hypothesis test::

   H0 (no spike):   x = n
   H1 (spike):      x = Uθ + n

with both the waveform coefficients θ and the noise level σ² unknown. Solving
for θ and maximising eliminates σ² entirely.

In plain terms
^^^^^^^^^^^^^^

**F is a ratio of two energies measured in the same window.** Scale the window
by any constant c and both numerator and denominator gain c², so F is
unchanged. Double the noise and F does not move.

That scale invariance is the whole point. It means the threshold never has to
track the noise level, and the false-alarm rate stays fixed at α — hence
*constant false alarm rate*, CFAR. A plain ``|x| > 3.5 σ`` rule has to estimate
σ somewhere and apply it somewhere else; when the noise floor rises between
those two moments, false detections explode.

**Where the threshold comes from.** Under H0 with white Gaussian noise, the two
projected energies are independent chi-square variables::

   ‖P_U x‖² / σ²   ~  χ²(r)
   ‖P_⊥ x‖² / σ²   ~  χ²(N−r)

Their ratio, each divided by its degrees of freedom, is F-distributed with
``(r, N−r)``. The code omits the degrees-of-freedom division, so the critical
value is scaled back by ``r/(N−r)``. For N=48, r=3, α=1e-4::

   F_crit = F.ppf(0.9999, 3, 45) = 8.8598
   thr    = 8.8598 × 3/45        = 0.5907

**A third reading of the same number.** Since ``cos²θ = F/(1+F)``, the
threshold is equivalent to an angle::

   cos²θ = 0.5907 / 1.5907 = 0.371   ->   θ = 52.5 degrees

The window's waveform must lie within 52.5 degrees of the signal subspace.
Purely a shape test; amplitude does not enter.

.. note::

   The chi-square result assumes **white** Gaussian noise. Neural noise after a
   300 Hz highpass is not white, and this rig additionally carries a common-mode
   interference component around 9.5 kHz. Coloured noise breaks the exact
   F distribution, so the realised false-alarm rate is not guaranteed to equal
   α. The standard remedy is whitening, which this implementation does not do.
   Treat α as a relative dial, not an absolute promise, and verify the realised
   rate empirically.

Inputs
^^^^^^

``recording``
   SpikeInterface recording, already artifact-corrected and highpass filtered.
   Multi-segment is supported; segments are processed independently.

``basis``
   ``(N, r)`` orthonormal-column array from Stage 1. **The rank is read from
   this array, not passed as an argument** — swapping the basis file silently
   changes both r and the threshold.

``cfar_alpha``
   Per-sample false-alarm probability. Default 1e-4.

``peak_sign``
   ``"neg"`` snaps to the local minimum. Any other value snaps to the maximum.

``refractory_ms``, ``snap_ms``
   Minimum separation between kept peaks, and the half-width searched when
   aligning a detection to the true extremum. Defaults 0.5 and 0.6 ms.

``chunk_s``
   Seconds read per chunk. This is the memory knob; see
   :ref:`why-fast`.

Key code
^^^^^^^^

**Setup** (lines 121-131). Threshold, index offsets and the two convolution
kernels:

.. code-block:: python

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

``lag = N // 2`` converts a filter-output index back to a signal index:
``conv1d`` without padding maps output index i to the input window ``[i, i+N)``,
whose centre is ``i + N//2``. With N even the true centre falls half a sample
off; the snap step absorbs that.

**Chunk geometry** (lines 133-134, 142-143). Two windows, not one:

.. code-block:: python

   chunk = int(chunk_s * fs)
   margin = 2 * N + w_snap
   ...
   t1 = min(t0 + chunk, seg_len)                              # commit window [t0, t1)
   a, b = max(0, t0 - margin), min(seg_len, t1 + margin)      # read window   [a, b)

::

        a              t0                            t1              b
        |---margin-----|=========== commit ==========|----margin-----|
        <---------------------- data actually read ------------------>
                       <----- only peaks here are kept ----->

The extra margin gives a spike sitting on a seam a complete N-sample window to
be scored in. The duplicates that overlap creates are removed by the
``t0 <= g < t1`` test inside the picker.

**The two convolutions** (lines 144-150):

.. code-block:: python

   blk = recording.get_traces(start_frame=a, end_frame=b, segment_index=seg_idx,
                              return_in_uV=True).astype("float32")   # (L, n_ch)
   blkT = np.ascontiguousarray(blk.T)                                # (n_ch, L)
   x = torch.from_numpy(blkT).unsqueeze(1).to(dev)
   s_sig = (torch.nn.functional.conv1d(x, w_sig) ** 2).sum(1)        # ‖Uᵀx‖²
   s_tot = torch.nn.functional.conv1d(x ** 2, w_tot).squeeze(1)      # ‖x‖²
   F = (s_sig / torch.clamp(s_tot - s_sig, min=1e-9)).cpu().numpy()  # (n_ch, L-N+1)

Tensor shapes::

   x      (n_ch, 1, L)          electrode channels occupy the BATCH axis
   w_sig  (r,    1, N)          basis vectors occupy the OUTPUT-CHANNEL axis
   out    (n_ch, r, L-N+1)      every channel x every basis vector, one call

``s_tot`` convolves ``x**2`` with an all-ones kernel — a moving sum of squares,
i.e. ‖x‖² per window. Using ``conv1d`` rather than ``cumsum`` makes its output
index align with ``s_sig`` automatically.

``torch.clamp(..., min=1e-9)`` guards the division where a window lies entirely
inside the subspace. The case that matters in practice is a blanked region of
exact zeros: there ``s_sig`` is also zero, so ``0 / 1e-9 = 0`` passes through
safely instead of producing a spurious detection.

.. note::

   PyTorch's ``conv1d`` performs cross-correlation — it does not flip the
   kernel. For a true convolution that would be wrong; here the sliding inner
   product ``Uᵀx`` is exactly what is wanted.

**The peak picker** (lines 19-84, called at 151-159). One Numba-compiled call
per channel per chunk:

.. code-block:: python

   for ci in range(n_ch):
       g, last_kept[ci] = _subspace_pick_channel(
           F[ci], blkT[ci], thr, lag, w_snap, min_sep, a, t0, t1, seg_len,
           last_kept[ci], neg)
       if len(g):
           samp_parts.append(g)
           ...
           amp_parts.append(blkT[ci][g - a])

Four stages inside the picker:

#. **Run detection with argmax** (line 29 onward). Scan F once; each contiguous
   region above threshold yields exactly one candidate, at its maximum. A real
   spike keeps F above threshold for tens of samples as the sliding window
   crosses it, so without this collapse one spike would produce dozens of
   detections. This is the first of two de-duplication mechanisms.
#. **Index conversion and snap** (lines 41-56). ``p = best + lag`` moves from
   filter index to trace index; the code then searches ``p ± w_snap`` for the
   true extremum of the trace. The maximum of F does not necessarily coincide
   with the trough, and without snapping every spike would be timestamped a few
   samples off, blurring any downstream waveform average.
#. **Commit test** (lines 57-59). ``g = sidx + a`` converts to a segment-global
   index; ``t0 <= g < t1`` discards anything found in the margin, which belongs
   to a neighbouring chunk.
#. **Sort, then greedy refractory** (line 70 onward). Snapping can move a
   candidate by up to ``w_snap``, which can reorder two nearby candidates, so
   the list is sorted before the greedy pass. The refractory rule is
   **first-wins**, not largest-wins — worth noting, because the walking
   pipeline's ``_dedup_peaks`` keeps the largest-amplitude peak in a cluster
   instead.

``last_kept`` is an ``(n_ch,)`` array holding, per channel, the last accepted
sample index. It goes into the picker and comes back out, so the refractory
period holds across a chunk seam as well as within a chunk. It is reset at the
start of each segment (line 139), initialised to ``-1e12`` so the first peak
always passes.

Seam correctness rests on exactly three mechanisms:

===================================================  ====================================
Risk                                                 Mechanism
===================================================  ====================================
Spike on a seam has no complete N-sample window      ``margin = 2N + w_snap`` over-read
Margin peak found by both neighbouring chunks        ``t0 <= g < t1`` commit test
Peaks either side of a seam closer than refractory   ``last_kept`` carried across chunks
===================================================  ====================================

Outputs
^^^^^^^

Structured array, sorted by segment then sample (line 170). An empty result
still returns the correct dtype, so downstream length checks never break.

``amplitude`` holds the **true trace voltage** at the snapped extremum, read
back with ``blkT[ci][g - a]`` — not the F statistic and not a filter output.
This matters because SpikeInterface's ``matched_filtering`` fills the same
field with filter output, a different physical quantity.


Stage 3 — Local-sigma amplitude gate
-------------------------------------

Concept
^^^^^^^

Scale invariance is Stage 2's strength and its blind spot. F measures only the
angle between the window and the subspace, so a low-amplitude wobble whose
shape happens to align with the subspace scores exactly as well as a large
spike. During chewing or movement bursts the noise floor rises, and some of
that noise lands near the subspace.

Stage 3 adds the missing dimension: **absolute amplitude must beat the local
noise level.** The two stages combine as AND — the shape must match *and* the
amplitude must be sufficient relative to noise at that moment.

In plain terms
^^^^^^^^^^^^^^

Chop the recording into 100 ms blocks. In each block, on each channel, estimate
the noise level with a median absolute deviation, converted to a
Gaussian-equivalent sigma by the factor 1.4826. That gives a coarse sigma
surface over (channel, time). Interpolate it to each peak's exact position and
keep the peak only if its trough voltage exceeds ``k_amp`` times the local
sigma.

Two design points:

- **MAD, not standard deviation.** Spikes are outliers; a standard deviation
  computed over data containing spikes is inflated by the very events being
  detected, which raises the gate against itself. MAD is insensitive to
  outliers.
- **A 100 ms grid with interpolation, not a rolling median.** The noise level
  is a slow envelope — chewing lasts hundreds of milliseconds, posture changes
  last seconds. Per-sample rolling statistics would cost orders of magnitude
  more for resolution that is never used.

Inputs
^^^^^^

``peaks``, ``recording``
   Output of Stage 2 and the same recording it ran on.

``k_amp`` : float, default 3.5
   Multiplier on the local sigma.

``w_ms`` : float, default 100.0
   Block width for the sigma grid.

Key code
^^^^^^^^

**Grid setup** (lines 184-192):

.. code-block:: python

   blk = max(1, int(w_ms * 1e-3 * fs))       # 100 ms -> 3000 samples at 30 kHz
   n_blk = int(np.ceil(N / blk))
   sigma_grid = np.full((n_ch, n_blk), np.nan, np.float32)
   ...
   chunk = blk * 300                          # multiple of blk -> grid aligns

``chunk`` being an exact multiple of ``blk`` is a hard requirement, not a
tuning choice: the write index is computed as ``c0 // blk``, so a chunk
boundary falling inside a block would misalign the grid.

**Voltage read and MAD, in one pass** (lines 197-206):

.. code-block:: python

   in_chunk = (samp >= c0) & (samp < c1)
   if in_chunk.any():
       amp[in_chunk] = tr[samp[in_chunk] - c0, chan[in_chunk]]
   ...
   b = trp.reshape(nb, blk, n_ch)
   med = np.nanmedian(b, axis=1)
   mad = np.nanmedian(np.abs(b - med[:, None, :]), axis=1)
   sigma_grid[:, c0 // blk:c0 // blk + nb] = (1.4826 * mad).T

The amplitude read is two-dimensional fancy indexing — every peak in the chunk
in one vectorised expression, no loop. The reshape to ``(nb, blk, n_ch)`` is a
view, not a copy. The trailing partial block is NaN-padded so ``nanmedian``
ignores the padding rather than being diluted by zeros.

The trough voltage is deliberately re-read from the trace rather than taken
from ``peaks["amplitude"]``, which keeps this gate detector-agnostic: it
applies the same physical test whether the upstream detector was subspace,
matched filter or threshold crossing.

**Interpolate and test** (lines 208-217):

.. code-block:: python

   centers = (np.arange(n_blk) + 0.5) * blk
   ...
   sig_at = np.interp(samp[m], centers[good], sg[good])
   keep[m] = amp[m] > k_amp * sig_at

``centers`` are block centres, not starts. ``good`` masks out NaN and zero
blocks so interpolation uses only valid estimates.

Outputs
^^^^^^^

The surviving subset of ``peaks``, same dtype, plus a printed summary::

   [local-σ gate] kept 412,338/1,204,551 (k=3.5, w=100.0ms)

Two behaviours to be aware of:

- ``keep`` is initialised to all-False, so a channel whose sigma grid is
  entirely NaN or zero — a dead channel, or one blanked to zeros — loses **all**
  its peaks, silently. If a channel unexpectedly has no spikes, check here
  first.
- The function is **single-segment**. It calls ``get_num_samples()`` and
  ``get_traces()`` without a ``segment_index`` and ignores
  ``peaks["segment_index"]``. Multi-packet ``.ns6`` recordings are concatenated
  and run as one segment in practice, so this is not currently exercised, but
  it would produce wrong results on a genuinely multi-segment recording.


.. _why-fast:

Why it runs fast
----------------

Two very different bottlenecks, handled by two very different tools. The split
happens at the F array: everything before it is dense uniform arithmetic,
everything after it is irregular sequential logic.

Batching channels through one convolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The arithmetic per chunk is r+1 sliding inner products per channel. The naive
shape for that is a Python loop over channels calling ``scipy.signal.correlate``
once per channel per basis vector — for 256 channels and r=3, that is over a
thousand separate library calls per chunk, each paying its own interpreter
round-trip, argument validation, transform planning and buffer allocation.

Placing the electrode axis in ``conv1d``'s batch dimension collapses all of
that into **one call per chunk**:

- Per-call overhead is paid once instead of a thousand times.
- Inside the call, the backend (oneDNN/MKL on CPU, cuDNN on GPU) chooses a
  blocked schedule, keeps the r short kernels resident in cache or registers,
  and reuses each loaded input sample across all r outputs. The input is
  streamed once and contributes to r results.
- On GPU the same call expands to thousands of concurrent threads over a dense
  small-kernel convolution — precisely the workload cuDNN is tuned for.
- The code path is device-agnostic. CUDA when available, CPU otherwise; the CPU
  branch is a fallback, not a different algorithm, so results are identical
  either way.

The consequence is qualitative but important: with N=48 and r=3, the arithmetic
performed per loaded sample is small, so once call overhead is removed the
limiting resource becomes memory and disk bandwidth rather than arithmetic.
That is what the docstring means by *the F statistic is compute-negligible and
runtime is I/O-bound* — and it is why enlarging the chunk helps only until the
storage read rate saturates.

The transfer back to host also benefits from the batching. ``F`` is moved to
CPU once per chunk, not once per channel, because the picker that consumes it
is a CPU function. Coarse transfer granularity keeps that boundary cheap.

**The cost of the batching** is the extra r axis on the convolution output. The
intermediate ``(n_ch, r, L-N+1)`` tensor is r times the size of the input
block, and it is the largest single allocation in the function. At
``chunk_s=30``, 256 channels and 30 kHz, the peak footprint across the input
block, its transposed copy, the device tensor, the convolution output and F is
roughly 7 GB, of which the convolution output alone is about 2.6 GB. That is
the entire reason ``chunk_s`` exists as a parameter. Halving it halves the
footprint and, because of the margin and commit-window machinery, changes
nothing about the result.

The peak picker cannot be vectorised
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``_subspace_pick_channel`` walks the F trace one sample at a time, and what it
does at sample i depends on state accumulated from every sample before it:
whether it is currently inside a supra-threshold run, the best value seen so far
within that run, and when the last accepted peak occurred.

Both of its core operations resist array programming:

- **Run detection with argmax.** Run boundaries are data-dependent and unknown
  in advance. A NumPy formulation needs a boolean mask, a run-labelling pass, a
  boundary extraction and a segmented argmax — several full-length temporaries
  and several passes over the array, allocated per channel per chunk.
- **Greedy refractory.** Whether peak i survives depends on which earlier peaks
  survived. That is a genuine loop-carried dependency; no amount of array
  reformulation removes it.

Written in pure Python the loop body is only a handful of comparisons, but the
interpreter overhead per sample dominates completely, and the loop runs once per
channel per chunk over roughly 900,000 samples at the default chunk size.

``@njit`` compiles it to machine code: comparisons on unboxed ``float32`` and
``int64``, no per-element object allocation, branches the CPU can predict, and
the array left as a flat typed buffer. The algorithm is unchanged — only the
execution model is.

Two supporting details:

- ``cache=True`` writes the compiled artefact into ``__pycache__``, so only the
  very first run on a machine pays JIT compilation. Subsequent runs load it.
- ``buf = np.empty(Lc, np.int64)`` preallocates the worst case (one peak per F
  sample) and the function returns ``buf[:m]``. Numba works best with
  fixed-size allocations, so this is the idiomatic pattern rather than a
  growing list.


Failure modes
-------------

All three fail silently. None raises.

Basis orthonormality is never checked
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Line 121 accepts whatever array is passed. If the columns are not orthonormal,
``‖Uᵀx‖²`` is no longer the projection energy, the F statistic loses its
meaning and the threshold no longer corresponds to α. The run completes and
produces plausible-looking numbers.

Check before use::

   np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-5)

.. _failure-rank:

A single template silently produces rank 1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``UA_BR_analysis_ssmf.py`` has a fallback path that builds a basis from one
matched-filter template::

   ua_basis = rcp.build_subspace_basis(_tpl[np.newaxis, :], rank=3)

With ``full_matrices=False``, an input of shape ``(1, N)`` yields ``Vt`` of
shape ``(1, N)`` — a single row. Slicing ``Vt[:3]`` still returns one row, so
the basis is ``(N, 1)``: **rank 1, not 3**, with no warning.

Because ``N, r = U.shape`` reads the rank from the array, the threshold changes
with it. At N=48, α=1e-4::

   r = 3   ->   thr = 0.5907   (52.5 degrees)
   r = 1   ->   thr = 0.3846   (58.2 degrees)

The gate is looser than intended and false positives increase. Building a
rank-r basis requires at least r linearly independent waveforms.

``peak_sign`` recognises only ``"neg"``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Line 127 is ``neg = peak_sign == "neg"``. Anything else — ``"NEG"``,
``"negative"``, ``"both"``, a typo — evaluates False and the detector snaps to
the local *maximum*. Extracellular spikes are negative-going, so this searches
noise. There is no validation.


Parameter reference
-------------------

Values in the third column assume fs = 30 kHz and the shipped
``(48, 3)`` basis.

====================  =========  ==========================================================
Parameter             Default    Effect at 30 kHz
====================  =========  ==========================================================
``cfar_alpha``        1e-4       threshold 0.5907; subspace angle 52.5 degrees
rank (from basis)     3          fixes N-r = 45 degrees of freedom in the F distribution
``refractory_ms``     0.5        15 samples minimum separation, per channel
``snap_ms``           0.6        18 samples searched either side for the true extremum
``chunk_s``           30.0       900,000 samples read per chunk; memory knob
``k_amp``             3.5        local-sigma multiplier in Stage 3
``w_ms``              100.0      3,000-sample blocks for the sigma grid
====================  =========  ==========================================================

Derived constants, not parameters:

====================  =========  ==========================================================
Constant              Value      Meaning
====================  =========  ==========================================================
``N``                 48         basis length, 1.60 ms — read from the basis array
``lag``               24         ``N // 2``, filter index to signal index
``margin``            114        ``2N + w_snap``, chunk overlap
1.4826                --         MAD to Gaussian-equivalent sigma
====================  =========  ==========================================================

Threshold sensitivity, for reference when changing α or rank:

=========  ========  ==========  =================
Setting    thr       cos²θ       subspace angle
=========  ========  ==========  =================
α = 1e-3   0.4300    0.301       56.7 degrees
α = 1e-4   0.5907    0.371       52.5 degrees
α = 1e-5   0.7676    0.434       48.8 degrees
α = 1e-6   0.9629    0.491       45.5 degrees
r = 1      0.3846    0.278       58.2 degrees
r = 3      0.5907    0.371       52.5 degrees
r = 8      1.1051    0.525       43.6 degrees
=========  ========  ==========  =================

Raising the rank enlarges the subspace but simultaneously tightens the angle
the F distribution demands. The compensation is automatic, not hand-tuned.