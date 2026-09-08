artifact_correction_template_matching.py
========================================

``RCP_analysis/python/functions/artifact_correction_template_matching.py``

An offline alternative to the incremental-PCA corrector in
:doc:`artifact_correction`. Instead of updating a template pulse by pulse, it
builds **one PCA template per stim block per channel** from that block's pulses,
then projects it out of every pulse in the block. The whole segment is held in
memory, so this is the offline path.

Pipeline:

#. Global drift removal — rolling median plus Gaussian-smoothed baseline
   subtraction, parallelized per channel.
#. Pulse-aligned snippet matrices over ``[trigger_start - pre, trigger_end + post_pad]``.
#. Per-block, per-channel PCA template (optionally excluding the first *N*
   pulses of each block, whose artifact shape differs).
#. Template subtraction, either as a top-*k* PC projection or as a single
   amplitude-scaled template.
#. Inter-pulse linear ramp from the artifact tail back to baseline.

.. py:currentmodule:: RCP_analysis.python.functions.artifact_correction_template_matching

Parameters
----------

.. py:class:: PCAArtifactParams

   Dataclass of every knob in the pipeline.

   .. list-table::
      :header-rows: 1
      :widths: 34 14 52

      * - Field
        - Default
        - Meaning
      * - ``rolling_median_ms``
        - 15.0
        - Median-filter width for drift removal.
      * - ``gaussian_sigma_ms``
        - 5.0
        - Sigma of the smoothing kernel whose output is subtracted.
      * - ``gaussian_len_ms``
        - 31.0
        - Length of that kernel.
      * - ``pre_samples``
        - 13
        - Samples before the trigger start included in each snippet.
      * - ``post_pad_samples``
        - 30
        - Samples after the trigger end included in each snippet.
      * - ``center_snippets``
        - True
        - Subtract the timepoint-wise mean across pulses before PCA.
      * - ``first_pulse_special``
        - True
        - Enable the first-pulse exclusion below.
      * - ``exclude_first_n_for_pca``
        - 1
        - Pulses skipped at the start of each block when fitting.
      * - ``scale_amplitude``
        - True
        - Per-pulse, per-channel least-squares scaling in the fallback path.
      * - ``interp_ramp``
        - True
        - Apply the inter-pulse ramp.
      * - ``ramp_tail_ms``
        - 1.0
        - Delay after the artifact end before the ramp starts.
      * - ``ramp_fraction``
        - 1.0
        - Fraction of the edge value ramped back to baseline.

Main entry points
-----------------

.. py:function:: remove_stim_pca_offline(recording, stim_npz_path, params=None, segment_index=0)

   Run the full pipeline on one segment and return the cleaned traces as a
   ``(n_samples, n_channels)`` NumPy array.

   :param recording: a SpikeInterface recording.
   :param Path stim_npz_path: ``stim_stream.npz`` from
      :func:`~RCP_analysis.python.functions.intan_preproc.extract_stim_npz`;
      read via
      :func:`~RCP_analysis.python.functions.utils.load_stim_detection`.
   :param PCAArtifactParams params: defaults to ``PCAArtifactParams()``.
   :param int segment_index: which segment to clean.

   Each pulse is assigned to the block containing its start sample. Where the
   block has usable PC components, each channel's patch is replaced by its
   residual after projecting onto ``base + top-k PCs``; where it does not (a
   block with a single valid pulse), the code falls back to subtracting the
   single template, optionally amplitude-scaled by
   ``⟨patch, template⟩ / ⟨template, template⟩``.

   Pulses whose window runs past either end of the recording, and blocks with no
   pulses, are skipped; the drift-removed traces are returned unchanged when the
   NPZ contains no pulses or blocks.

   .. note::

      The whole segment is loaded with ``get_traces`` at once. For long
      recordings this is the memory bottleneck — see the ``TODO`` in the source
      about chunking.

.. py:function:: cleaned_numpy_to_recording(cleaned, recording_like)

   Wrap a cleaned ``(n_samples, n_channels)`` array as a ``NumpyRecording``,
   taking the sampling rate, channel IDs and dtype from ``recording_like`` and
   copying its probe/properties metadata across.

Private helpers
---------------

.. py:function:: _global_drift_remove_pool(traces, fs, p, n_jobs=8)

   Per-channel drift removal across a ``multiprocessing.Pool``: subtract a
   rolling median (window forced odd), then subtract a zero-phase
   Gaussian-smoothed version of the result. The input dtype is preserved.

.. py:function:: _medfilt_col(args)
.. py:function:: _filtfilt_subtract_col(args)

   Pool workers for the two stages above, operating on one channel each.

.. py:function:: _block_window_lengths(trigger_pairs_window, pre, post_pad)

   Per-pulse snippet length, ``(end + post_pad) - (start - pre) + 1``. The block
   uses the maximum over its pulses as a uniform length.

.. py:function:: _extract_pulse_snippets(clean, trigger_pairs_window, pre, post_pad, target_len=None)

   Cut pulse-aligned snippets and pad (edge mode) or truncate them to a common
   length.

   :returns: ``(snips, keep_mask, L)`` with ``snips`` shaped
      ``(n_keep, L, n_channels)``; out-of-bounds pulses are dropped and flagged
      ``False`` in ``keep_mask``.

.. py:function:: _pca_template_per_channel(snips, center)

   Fit a per-channel PCA with ``k = min(3, n_pulses, L)`` components.

   :returns: ``(templ, pca_pack)`` — ``templ`` ``(L, n_channels)`` is the
      reconstruction of the mean snippet, and ``pca_pack`` is a per-channel list
      of ``{"base": (L,), "components": (k, L)}`` used for the projection at
      subtraction time. With a single pulse it returns that pulse as the
      template and empty components.

.. py:function:: _apply_interp_ramp(buf, start_idx, end_idx, frac)

   Linearly ramp the value at ``start_idx`` (scaled by ``frac``) down to zero
   across ``[start_idx, end_idx)``, in place, removing the step left at the end
   of a corrected artifact.
