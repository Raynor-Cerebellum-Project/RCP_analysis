artifact_correction.py
======================

``RCP_analysis/python/functions/artifact_correction.py``

Stimulation-artifact removal by incremental PCA. Each channel gets its own
low-rank artifact subspace estimated across stim pulses; the projection onto
that subspace is subtracted from every pulse. The learned weights are kept in a
:class:`Template` so a subspace fit on one recording can be reapplied elsewhere
without being updated.

Typical use::

   corrector = rcp.IPCA_Artifact_Correction(rank=3)
   signal_corrected, templates = corrector.ipca_all(signal)  # (n_stim, n_time, n_ch)

   # later, apply the same subspace without refitting
   clean = corrector.apply_template(new_signal, templates[ch])

.. py:currentmodule:: RCP_analysis.python.functions.artifact_correction

Classes
-------

.. py:class:: Template(weights=None)

   Holds the IPCA component weights ``(rank, n_time)`` for one channel.
   Supports ``len()``, indexing and ``repr``; ``weights`` is ``None`` until the
   first update.

   .. py:method:: update_weights(new_weights, learning_rate=0.5)

      Set the weights on first call, then blend on subsequent calls:
      ``(1 - lr) * old + lr * new``. A high learning rate tracks a drifting
      artifact; a low one averages over pulses.

.. py:class:: IPCA_Artifact_Correction(rank=3)

   Artifact corrector built on ``sklearn.decomposition.IncrementalPCA``.

   :param int rank: number of artifact components to remove per channel.

   .. py:method:: ipca_template_per_channel(signal, template, learning_rate=0.9)

      Fit and subtract the artifact subspace for one channel.

      The mean of the first 3 samples is removed as a baseline before fitting
      and added back afterwards, so the DC level is preserved. The effective
      rank is clamped to ``min(rank, n_time, n_stim)``.

      :param signal: ``(n_stim, n_time)`` for one channel.
      :param Template template: receives the fitted weights.
      :returns: ``(signal_corrected, template)``.

   .. py:method:: ipca_all(signal)

      Run the per-channel correction over every channel.

      :param signal: ``(n_stim, n_time, n_channels)``.
      :returns: ``(signal_all, templates_all)`` — the corrected array with the
         same shape, and one :class:`Template` per channel.

   .. py:method:: apply_template(signal, template)

      Project out an existing template **without** updating its weights — use
      this to apply a subspace learned on one block to another.

      :param signal: raw, un-centered ``(n_stim, n_time)``.
      :returns: corrected signal with the baseline restored.

SpikeInterface wrappers
-----------------------

These let corrected pulse windows be spliced back into a lazily-read recording,
so downstream SpikeInterface steps see the cleaned data without materializing
the whole array.

.. py:class:: PerChannelIPCACorrectedRecording(parent_recording, micro_map_per_channel, micro_corrected)

   A ``BaseRecording`` that overlays per-channel corrected patches onto its
   parent. Sampling rate, channel IDs, dtype and metadata are copied from the
   parent, and one corrected segment is added per parent segment.

   :param micro_map_per_channel: ``{channel_index: [(start, end), ...]}`` —
      the sample windows that have corrected data, per channel.
   :param micro_corrected: ``(n_pulses, n_time, n_channels)`` corrected patches,
      indexed by the position of the window within that channel's sorted list.

.. py:class:: _PerChannelIPCACorrectedRecordingSegment(parent_recording_segment, micro_map_per_channel, micro_corrected)

   The segment implementation. It pre-sorts each channel's window starts and
   ends so ``get_traces`` can ``searchsorted`` for the pulses overlapping a
   requested range, then copies each overlapping corrected patch into the
   returned traces. ``channel_indices`` may be ``None``, a slice, or an array.
