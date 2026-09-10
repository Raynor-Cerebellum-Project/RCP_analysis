Intan (NPRW) Preprocessing
===========================

Script: ``preprocessing_scripts/NPRW_Intan_analysis_mf.py``

Preprocessing and spike detection of the Intan data, handled using
`SpikeInterface <https://spikeinterface.readthedocs.io/>`_.

Steps
-----

#. Load geometry and mapping (``.mat`` file).
#. Extract stim data, IR crossings, and locations of stim pulses (individual pulses and blocks) — saves ``.npz`` file.
#. Extract auxiliary data (sync pulses) — saves ``.npz`` file.
#. Load Intan neural data, attach probe info, and reorder based on mapping.
#. Preprocess Intan (``.rhs``) data: high-pass filter, common local median reference (default radii: 30, 150 micrometers).
#. Remove artifacts (zero stim regions) + additional space specified in ``config/params.yaml``.
#. Use presaved template to perform matched filter and calculate MUA (peaks).
#. Save ``.npz`` file per condition.

Inputs
------

#. ``.rhs`` files from Intan

   - ``DATA_ROOT/Intan/<condition>/``

#. Probe geometry and channel mapping ``.mat`` file

   - ``config/probes/`` (set by ``geom_mat_rel`` and ``mapping_mat_rel`` in
     ``config/params.yaml``)

#. Presaved waveform template for the matched filter

   - ``config/waveform_templates/median_extremum_templates_norm_NPRW.npy``

Outputs
-------

Stim and auxiliary streams
^^^^^^^^^^^^^^^^^^^^^^^^^^

``DATA_ROOT/results/aux_data/NPRW/<session>_Intan_streams/stim_stream.npz``
   Stim timing and channels, derived from the stim stream.

   .. list-table::
      :header-rows: 1
      :widths: 40 60

      * - Key
        - Contents
      * - ``active_channels``
        - Channels that delivered stim, 1-based.
      * - ``active_channels_0based``
        - Channels that delivered stim, 0-based.
      * - ``trigger_pairs``
        - Start and end sample of each individual stim pulse.
      * - ``block_bounds_samples``
        - Start and end sample of each stim block. (For example 400Hz 100ms gives 40 pulses per block)
      * - ``pulse_sizes``
        - Number of pulses in each block.
      * - ``meta``
        - JSON string: session, stream name, sampling rate, channel counts,
          channel ordering, and the detection channel.

``DATA_ROOT/results/aux_data/NPRW/<session>_Intan_streams/aux_streams.npz``
   Sync pulses, stored as a single ``aux_traces`` array plus a ``meta`` JSON
   string holding the sampling rate, channel ids, dtype, shape, and units (uV).

Preprocessed recording
^^^^^^^^^^^^^^^^^^^^^^

``DATA_ROOT/results/checkpoints/NPRW/pp_local_<inner>_<outer>__interp_<session>/``
   The filtered, referenced, artifact-removed recording. Saved as a SpikeInterface folder in their format rather than a single file. ``<inner>`` and ``<outer>``
   are the local reference radii from ``config/params.yaml``.

Detected peaks
^^^^^^^^^^^^^^

``DATA_ROOT/results/checkpoints/NPRW/rates__<session>__bin<bin>ms_sigma<sigma>ms.npz``
   Matched-filter output for the session.

   .. list-table::
      :header-rows: 1
      :widths: 40 60

      * - Key
        - Contents
      * - ``peaks``
        - Detected peaks (MUA).
      * - ``noise_levels``
        - Per-channel noise estimate used to set the detection threshold.
      * - ``ir_idx``, ``ir_ms``
        - IR beam crossings, as sample indices and in milliseconds.
      * - ``meta``
        - Settings and recording info:

          - Detection: ``detect_threshold``, ``peak_sign``, ``bin_ms``,
            ``sigma_ms``
          - Stim: ``stim_channels``, ``stim_dur``
          - Recording: ``fs``, ``n_channels``, ``n_samples``, ``n_segs``
          - Timing: ``rec_dur``, ``rec_start_ms``, ``rec_end_ms``
          - ``session``: the session name

