LFP band analysis
=================

``preprocessing_scripts/analyze_lfp_bands.py``

Analyzes LFP bands (delta, theta, alpha, beta, low/high gamma) for Utah Array
(Blackrock) recordings.

Processing steps
----------------

#. **Extraction** — loads stimulation events and extracts epochs (±1000 ms)
   with padding.
#. **IPCA artifact correction**

   * Per-channel lag calibration using a CSV lookup.
   * Clock drift correction (1.33% slower than nominal).
   * Region-wise incremental PCA artifact subtraction (rank 10).

#. **Preprocessing**

   * Bandpass filter (1–200 Hz).
   * Resample from 30 kHz to 1 kHz.

#. **1/f detrending** — adaptive spectral subtraction using the baseline period.
#. **Band filtering** — zero-phase bandpass filtering for each frequency band.
#. **Epoch slicing** — extracts pre-stim, post-stim, and full windows.
#. **Baseline grouping** — aggregates control/baseline sessions by UA port and
   depth.

Frequency bands
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 30

   * - Band
     - Range
   * - Delta
     - 1–4 Hz
   * - Theta
     - 4–8 Hz
   * - Alpha
     - 8–12 Hz
   * - Beta
     - 12–25 Hz
   * - Low gamma
     - 25–60 Hz
   * - High gamma
     - 60–120 Hz

Key parameters
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Parameter
     - Value
   * - ``EPOCH_PRE_MS``, ``EPOCH_POST_MS``
     - 1000 ms each
   * - ``BLANK_PRE_MS``, ``BLANK_POST_MS``
     - 5 ms, 101 ms (stim artifact window)
   * - ``TARGET_FS``
     - 1000 Hz (resampled)
   * - ``IPCA_RANK``
     - 10

Output
------

``results/checkpoints/UA_LFP/<condition>/aligned_lfp__*.npz``

Conditions: ``control_reaches/``, ``stim_reaches/``, ``at_rest/``, ``Grasp/``,
``IMU/``, ``continuous_stim/``

Dependencies
------------

* Aligned ``.npz`` files from preprocessing step 5.
* Per-channel lag calibration files in ``config/channel_lag_calibration/``.
* Peristim files from preprocessing step 6, for event timing.

Visualization
-------------

``scripts/plot_lfp_all.py``
   Comprehensive plots (traces, heatmaps, PSDs) for all frequency bands.

``scripts/debug_lfp_pipeline_plots.py``
   Step-by-step pipeline visualizations (raw → blanked → filtered) for
   debugging.

.. Kinematics quantification -- commented out, kept for reference.

   ``batch_scripts/quantify_stim_kinematics.py`` analyzes reach kinematics
   (duration, peak speed) comparing stimulation conditions to baseline:
   speed-based reach endpoint detection; statistical analysis (Welch's t-test
   vs baseline) with significance clustering; violin/box plots with
   transparent, jittered individual points; parallel processing for fast
   dataset scanning. Output: stats summary CSVs and figures in
   ``results/figures/quantify_kinematics``.

   ``scripts/plot_individual_kinematics.py`` plots individual kinematic traces
   for every trial in a grid layout, to allow visual inspection and outlier
   identification. Output: ``results/figures/individual_trials``.
