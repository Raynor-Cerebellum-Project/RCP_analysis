Extract Peri-Stimulus Data
==========================

Script: ``preprocessing_scripts/extract_peri_stim.py``

Extracts peri-stimulus data from ``aligned.npz`` files and distributes it into conditions.

Condition Types
---------------

#. **Control reaches** (no stim) — aligned to IR crossing, split by target (A/B).
#. **Stim condition reaches** — aligned to stim onset, split by target (A/B).
#. **At rest** (no reach) — aligned to stim onset, no target splitting.
#. **Grasp trials** — aligned to stim onset, no target splitting.
#. **IMU trials** — aligned to stim onset, no target splitting.
#. **Continuous stim** — aligned to IR crossing, split by target (A/B).

Steps
-----

#. Load ``aligned_*.npz`` files.
#. Drop trials with bad kinematics.
#. Compute trial labels (A/B/N) from touchscreen state.
#. Extract peri-event traces for neural data (NPRW, UA).
#. Deduplicate MUA peaks, bin counts, and estimate firing rates using edge-aware Gaussian kernel smoothing.
#. Baseline-correct traces (subtract mean of first 150 ms).
#. Calculate median, variance, and mean traces across trials.
#. Extract peri-event kinematics traces (position and velocity).
#. Interpolate small NaN gaps in kinematics (≤4 samples).
#. Z-score kinematics.
#. Separate left (A) and right (B) reaches for stim and control conditions.
#. Output ``.npz`` and ``.mat`` files in ``results/checkpoints/PeriStim/.../``.

Inputs
------

#. Aligned files, from :doc:`make_aligned`

   - ``DATA_ROOT/results/checkpoints/Aligned/<condition>/aligned__*.npz``

#. Manually curated trial exclusions, from :doc:`inspect_kinematics`

   - ``config/manual_trial_remove.csv``

#. BR-to-Intan shifts, from :doc:`compute_shifts`

   - ``DATA_ROOT/Metadata/br_to_intan_shifts.csv``

Outputs
-------

``PeriStim/control_reaches/target_A/``, ``target_B/``
   Control reaches (no stim), aligned to IR crossing, split by target.

``PeriStim/stim_reaches/target_A/``, ``target_B/``
   Stimulation condition reaches, aligned to stim onset, split by target.

``PeriStim/at_rest/``
   At-rest trials (no reach), aligned to stim onset.

``PeriStim/Grasp/``
   Grasp experiment trials, aligned to stim onset.

``PeriStim/IMU/``
   IMU experiment trials, aligned to stim onset.

``PeriStim/continuous_stim/``
   Continuous stimulation trials, aligned to IR crossing.

Files are named ``peristim__<intan_filename>__BR_<br_idx>.npz``, with
``_target_<A|B>`` appended for the conditions that split by target. A matching
``.mat`` is written alongside each one.

Contents
^^^^^^^^

Times are relative to the alignment event, in milliseconds.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Contents
   * - ``sess``, ``br_idx``, ``n_trials``, ``overall_title``
     - Session, BR file index, number of trials kept, and plot title.
   * - ``event_ms``
     - Event time for each trial in ``aligned_*.npz''
   * - ``trial_labels``
     - Target label (``A``/``B``/``N``) per trial, matching ``event_ms``.
   * - ``trial_labels_all``
     - Labels for every trial that survived behavior gating, before the
       target split.
   * - ``raw_trial_indices``
     - Stable raw stim-pulse index per trial. This is the ``R`` shown in the
       :doc:`inspect_kinematics` subplot titles and recorded in
       ``manual_trial_remove.csv``.
   * - ``meta``, ``nprw_meta``, ``ua_meta``
     - Settings and provenance. Each also has a ``*_json`` twin holding the
       same content as a JSON string, for readers that cannot unpickle.
   * - ``align_meta_raw``
     - The ``align_meta`` block carried over from the aligned file.

Neural, per probe. The ``UA_*`` keys mirror the ``NPRW_*`` ones:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Contents
   * - ``NPRW_rel_t``, ``NPRW_width_ms``
     - Time base for the binned arrays, and the bin width.
   * - ``NPRW_counts``, ``NPRW_rates_hz``
     - Per-trial binned spike counts, and the Gaussian-smoothed firing rate.
   * - ``NPRW_rates_zeroed``
     - The same rates after baseline correction (mean of the first 150 ms
       subtracted).
   * - ``NPRW_med``, ``NPRW_var``, ``NPRW_med_counts``
     - Median, variance, and median counts across trials.
   * - ``NPRW_peak_ms_dedup``, ``NPRW_amps_ms_dedup``
     - Deduplicated MUA peak times and amplitudes, keyed by channel.
   * - ``HAS_BR``
     - Whether UA data was present. When false the ``UA_*`` keys are still
       written, but zero-sized.
   * - ``ua_ids_1based``, ``ua_region``, ``ua_region_names``, ``ua_port``,
       ``ua_nsp``, ``ua_idx_rows``
     - UA channel identity and region labels, carried over from the aligned
       file.

Behavior and auxiliary streams:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Contents
   * - ``beh_rel_t``, ``n_beh``
     - Time base for the behavior traces, and the number of behavior trials.
   * - ``beh_cam0_segs``, ``beh_cam1_segs``
     - Per-trial position traces, z-scored, with NaN gaps of 4 samples or
       fewer interpolated.
   * - ``beh_cam0_vel_segs``, ``beh_cam1_vel_segs``
     - Per-trial velocity traces.
   * - ``beh_cam0_pos_med``, ``beh_cam1_pos_med``,
       ``beh_cam0_vel_med``, ``beh_cam1_vel_med``
     - Median position and velocity across trials.
   * - ``beh_cam0_names``, ``beh_cam1_names``
     - Column names for the behavior arrays.
   * - ``ts_state_segs``, ``ts_state_char_segs``, ``ts_state_rel_t``,
       ``n_ts_state_trials``
     - Per-trial touchscreen state, numerically and as character codes, with
       its own time base.
   * - ``ts_state_num``, ``ts_state_char``
     - The full touchscreen state series, not cut into trials.
   * - ``hr_sig``, ``vog_sig``
     - Heart rate and VOG signals, when recorded.

.. note::
   Unlike the aligned files, the ``UA_*`` keys are always written here. When
   there is no UA data they are present but zero-sized, so check ``HAS_BR``
   or the array shape rather than key existence.
