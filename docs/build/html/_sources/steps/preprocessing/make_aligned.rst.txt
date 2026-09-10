Create Aligned Files
====================

Script: ``preprocessing_scripts/make_aligned_npz_and_mat.py``

Creates the ``aligned.npz`` files by combining neural, auxiliary, and behavioral data into a single aligned file per condition.

Steps
-----

#. Align neural, aux, and behavioral data (NPRW, UA, kinematics, HR, target, and VOG).
#. Create spike time (peaks) dictionaries for both NPRW and UA.
#. Permute UA channel labels and UA peak dictionary by region.
#. Output ``.npz`` and ``.mat`` files in ``DATA_ROOT/results/checkpoints/Aligned``.

Inputs
------

#. Binned firing rates from the NPRW and UA preprocessing steps

   - ``DATA_ROOT/results/checkpoints/NPRW/rates__<session>__*.npz``
   - ``DATA_ROOT/results/checkpoints/UA/rates__<session>__*.npz``

#. BR-to-Intan shifts, to put both systems on one clock

   - ``DATA_ROOT/Metadata/br_to_intan_shifts.csv``

#. Stim stream, for stim times

   - ``DATA_ROOT/results/aux_data/NPRW/<session>_Intan_streams/stim_stream.npz``

#. Aligned behavior and VOG, when those streams exist

   - ``DATA_ROOT/results/checkpoints/Behavior/*_both_cams_aligned.csv``
   - ``DATA_ROOT/results/checkpoints/VOG/*_VOG_aligned.csv``

Outputs
-------

One ``.npz`` per Intan/BR pair, named
``aligned__<intan_filename>__Intan_<intan_idx>__BR_<br_idx>.npz``. A matching ``.mat`` files is written as well when ``output_aligned_mat`` is true in ``config/params.yaml``.

Files are sorted into a directory by condition, checked in this order, so the
metadata flags take precedence over the ``Type`` column:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Directory
     - Selected when
   * - ``Aligned/continuous_stim/``
     - ``is_continuous`` is set. Continuous stimulation trials.
   * - ``Aligned/control_reaches/``
     - ``is_control`` is set. Control trials (no stim).
   * - ``Aligned/at_rest/``
     - ``is_at_rest`` is set. At rest conditions (no reach).
   * - ``Aligned/Grasp/``
     - ``Type`` is ``GRASP``. Grasp experiment trials.
   * - ``Aligned/IMU/``
     - ``Type`` is ``IMU``. IMU experiment trials.
   * - ``Aligned/stim_reaches/``
     - Everything else. Stimulation condition reaches.

Contents
^^^^^^^^

All times are in milliseconds on the BR clock, with the Intan-to-BR shift
already applied.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Contents
   * - ``nprw_peak_ms``
     - NPRW spike times, as a dictionary key: channel.
   * - ``nprw_peak_amps``
     - Peak amplitude for each spike.
   * - ``nprw_meta``
     - The metadata from NPRW file preprocessing
   * - ``ir_ms``
     - IR crossings.
   * - ``stim_ms``
     - Stim times.
   * - ``shift_ms``
     - The shift that was applied between BR files and Intan files
   * - ``align_meta``
     - Settings for this file, see below.

Present when BR data exists:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Contents
   * - ``ua_peak_ms``, ``ua_peak_amps``, ``ua_meta``
     - UA spike times, amplitudes, and metadata, matching the NPRW keys.
   * - ``ua_elec``, ``ua_port``, ``ua_nsp``, ``ua_idx_rows``
     - UA channel identity: electrode number, port, NSP, and row index.
   * - ``ua_region``, ``ua_region_names``
     - Region label per channel, and the name for each label. Channels are
       permuted so that regions are contiguous.
   * - ``hr_sig``
     - Heart rate signal.
   * - ``ts_state_num``, ``ts_state_char``
     - Touchscreen state per sample, as a number and as a character code.

Present when kinematics exist:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Contents
   * - ``beh_cam0``, ``beh_cam1``
     - Keypoint tracks from each camera.
   * - ``beh_cam0_cols``, ``beh_cam1_cols``
     - Column names for those arrays.
   * - ``beh_ns5_sample``, ``beh_t_ms``
     - BR sample number and time for each behavior frame.

Present when VOG exists:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Contents
   * - ``vog_cols``, ``vog_col_names``
     - The VOG position columns and their names.
   * - ``vog_ns2_samp``, ``vog_t_ms``
     - BR ns2 sample number and time for each VOG frame.
   * - ``vog_sig``
     - VOG signal as recorded on the BR analog channel.

``align_meta``
^^^^^^^^^^^^^^

- Pairing: ``intan_filename``, ``intan_idx``, ``br_idx``, ``exp_type``
- Condition flags: ``is_control``, ``is_at_rest``, ``is_continuous``,
  ``recording_stim_dur``
- Sampling rates: ``fs_nprw``, ``fs_ua``, ``fs_ns5``
- Alignment: ``shift_ms``
- Source files: ``nprw_rates``, ``ua_rates``, ``behavior_csv``,
  ``behavior_rows``

.. note::
   Keys for a stream that was not recorded are simply absent, so read them with
   ``.get()`` rather than assuming every file has the same set. Which streams
   are expected is set by ``has_BR``, ``has_kinematics``, and ``has_VOG`` in
   ``config/params.yaml``.
