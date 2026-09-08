intan_preproc.py
================

``RCP_analysis/python/functions/intan_preproc.py``

Reads Intan recordings: reorders channels into probe geometry, detects
stimulation pulses and blocks from the stim stream, and caches stim and aux
streams as NPZ. Used by ``preprocessing_scripts/NPRW_Intan_analysis_mf.py``.

.. py:currentmodule:: RCP_analysis.python.functions.intan_preproc

Channel mapping
---------------

.. py:function:: reorder_recording_to_geometry(rec, perm)

   Return a ``ChannelSliceRecording`` whose channels follow ``perm``, a
   permutation from device order to geometric order. ``perm=None`` returns the
   recording unchanged with a warning, so callers can pass an unresolved mapping
   through safely.

Stim detection
--------------

.. py:class:: StimTriggerResult

   Dataclass returned by the pulse detectors.

   :Fields:
      * ``active_channels`` ``(n_active,)`` — **1-based** channel IDs.
      * ``trigger_pairs`` ``(n_pulses, 2)`` — ``[start_sample, end_sample]``.
      * ``block_bounds_samples`` ``(n_blocks, 2)`` — ``[block_start, block_end]``.
      * ``pulse_sizes`` ``(n_pulses,)`` — pulse length in samples.

.. py:function:: scan_active_channels(rec, chunk_s=30.0)

   Channels with any nonzero stim sample, as **0-based** indices. Scans in
   ``chunk_s`` blocks so memory stays bounded, and stops early once every
   channel is known to be active.

.. py:function:: load_channel_signal(rec, ch_index, chunk_s=30.0)

   Load one channel as scaled ``float32``, filled in chunks (~86 MB for a
   12-minute session).

.. py:function:: load_traces_chunked(rec, ch_indices=None, chunk_s=30.0)

   Build a ``(n_channels, n_samples)`` scaled ``float32`` array without a
   full-size temporary. ``ch_indices`` selects a 0-based subset; ``None`` loads
   every channel.

.. py:function:: extract_stim_npz(sess, out_dir, stim_stream_name="Stim channel", chanmap_perm=None, save_traces=True, det_channel=None)

   Detect stim pulses and blocks for one Intan session and write
   ``<out_dir>/<session>_Intan_streams/stim_stream.npz``.

   :param Path sess: Intan session directory.
   :param Path out_dir: root for the ``*_Intan_streams`` output folder.
   :param str stim_stream_name: SpikeInterface stream name.
   :param chanmap_perm: geometry permutation, see
      :func:`reorder_recording_to_geometry`.
   :param save_traces: ``True`` saves all channels' traces (~0.46 GB per
      recorded minute), ``"active"`` saves only channels carrying stim, ``False``
      omits traces and saves detection output only.
   :param det_channel: 0-based override for the detection channel; defaults to
      the first active channel.
   :returns: the dict of saved arrays, or ``None`` if the stream could not be read.

   Saved keys: ``active_channels`` (1-based), ``active_channels_0based``,
   ``trigger_pairs``, ``block_bounds_samples``, ``pulse_sizes``, optionally
   ``stim_traces`` and ``stim_traces_channels_0based``, plus a JSON ``meta``
   string carrying session name, ``fs_hz``, channel and sample counts, channel
   order (``geometry``/``device``), the detection channel, and which traces were
   saved.

   Read the result back with
   :func:`~RCP_analysis.python.functions.utils.load_stim_detection`.

.. py:function:: _extract_stim_triggers_and_blocks_1d(stim_signal, active_channels_0based)

   Pulse and block detection on a single detection channel — the routine
   :func:`extract_stim_npz` actually uses.

   Pulses: falling edges (rising edges if there are more of them) are taken
   every other edge, since each biphasic pulse produces two; ends are every
   second return-to-zero, found with one sorted ``searchsorted`` lookup.

   Blocks: a gap between consecutive pulse starts larger than 50× the median
   pulse size starts a new block.

   .. note::

      That 50× rule currently treats anything faster than ~10 Hz stim as a
      single block (see the ``TODO`` in the source).

   ``active_channels`` on the result is converted back to 1-based for
   compatibility.

.. py:function:: _extract_stim_triggers_and_blocks(stim_data)

   Older ``(n_channels, n_samples)`` version of the same detection, kept for
   reference. It indexes the detection channel with the 1-based ID, so prefer
   the ``_1d`` variant above.

Aux streams
-----------

.. py:function:: extract_intan_aux_streams_npz(sess, out_dir, aux_streams=("USB board ADC input channel",))

   Read the Intan aux stream(s), convert unsigned to signed, and save
   ``<out_dir>/<session>_Intan_streams/aux_streams.npz`` containing
   ``aux_traces`` ``(n_channels, n_samples)`` in µV plus a JSON ``meta``
   (session, stream name, ``fs_hz``, channel count and IDs, dtype, shape).

   :returns: the output path, or ``None`` if the stream could not be read.

   Read it back with
   :func:`~RCP_analysis.python.functions.utils.load_intan_aux`, which returns
   rows 0 and 1 as the triangle sync and BR template signals.

Constants
---------

``STIM_CHUNK_S = 30.0`` — default chunk size, in seconds, for every chunked
reader in this module.
