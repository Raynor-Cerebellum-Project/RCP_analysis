RSA Calculation
===============

Script: ``analysis_scripts/RSA_calculation.py``

Extracts RSA (Representational Similarity Analysis) tensors by correlating regions of interest (ROI).

- Trial × trial correlation matrices.
- ROIs can be set in ``config/params.yaml``.

The script is a wrapper: it reads ``rsa_params`` from
``config/params.yaml`` and calls ``run_rsa`` once per probe per target, so a
default run produces four sweeps (NPRW/UA x target_A/target_B).

Steps
-----

#. Collect one block of trials per condition: ``stim_reaches`` files, ``control_reaches``, and ``at_rest`` files.
#. Build a feature vector per trial from ``<probe>_rates_zeroed``, keeping the
   time axis: the window is flattened channel-major into one feature per
   (channel, timebin), so a block is trials x (channels * timebins).
#. Drop conditions with fewer than 4 trials, and trials with fewer than 2 valid channels.
#. Compute the channel masks the criteria need.
#. Z-score each channel across all trials pooled over blocks.
#. Drop any trial with NaNs after normalization.
#. For each criterion, apply its channel mask and build the trial x trial Pearson correlation matrix. Each pair of trials is correlated over the union of their two masks.
#. Score how separable the conditions are: pairwise silhouette scores with a
   1000-permutation null (BH-corrected), and within-condition correlation on
   the diagonal. TODO: pairwise LDA
#. Save RSA and silhouette figures, one pair per criterion.

Inputs
------

#. Peri-stim tensors from :doc:`../preprocessing/extract_peri_stim`

   - ``DATA_ROOT/results/checkpoints/PeriStim/stim_reaches/<target>/*.npz``
   - ``DATA_ROOT/results/checkpoints/PeriStim/control_reaches/<target>/*.npz``
   - ``DATA_ROOT/results/checkpoints/PeriStim/at_rest/*.npz``

``at_rest`` files have no target subfolder, so they are included in every
target's sweep and labelled separately.

Parameters
----------

Set under ``rsa_params`` in ``config/params.yaml``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Meaning
   * - ``poststim_win_ms``
     - Feature window, relative to stim **offset**. E.g. ``[0, 50]`` with a 100 ms ``stim_dur`` is 100-150 ms after stim onset.
   * - ``channel_criterion``
     - Which channels enter the correlation, see the table below. Accepts a
       list, in which case one RSM and figure is built per criterion, reusing
       the loaded features and the already-computed masks.
   * - ``move_alpha``, ``stim_alpha``
     - FDR thresholds for those two tests.
   * - ``targets``, ``probes``
     - Which targets subfolders and probes to do RSA over.

Channel criteria
^^^^^^^^^^^^^^^^

Masks are computed from all ``control_reaches`` files for the target, and both
criterion are FDR-corrected (Benjamini-Hochberg) across channels.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Value
     - Channels kept
     - Test
   * - ``None``
     - All channels. Appears in outputs as ``all-ch``.
     -
   * - ``movement``
     - Channels that respond to movement: paired Wilcoxon signed-rank test per
       channel
     - Baseline window vs. response window on the same trials. Threshold set by
       ``move_alpha``.
   * - ``stim``
     - Channels that respond to stim: unpaired Mann-Whitney U test per channel
     - Pooled control trials against this condition's trials. Threshold set by
       ``stim_alpha``. The mask applied is the union across conditions, so every
       condition is scored on the same channel set.
   * - ``union``
     - Channels passing either criterion.
     -
   * - ``intersection``
     - Channels passing both criterion.
     -

Every file must share one bin width. Conditions may have different numbers of timebins, because the window defined relative to its own stim offset.

Features are z-scored per (channel, timebins) column before distances are computed. Conditions with fewer than 4 trials are dropped, and a trial needs at least 2 valid channels to be kept.

Outputs
-------

Figures in ``DATA_ROOT/results/figures/rsa_from_peristim/<target>/``.
One set per probe and criterion:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - File
     - Contents
   * - ``RSA_<probe>_<criterion>_criterion_poststim<window>.png``
     - The trial x trial correlation matrix, blocked by condition.
   * - ``silhouette_<probe>_<criterion>_criterion_poststim<window>.png``
     - Pairwise silhouette scores between conditions, with within-condition
       correlation on the diagonal. Cells are starred where a permutation test
       against a shuffled null survives BH correction.
   * - ``debug_<probe>_movement_criterion_baseline<window>movement<window>.png``
     - Per-channel movement mask diagnostics. When ``debug_masks`` is
       true.
   * - ``debug_<probe>_stim_criterion_poststim<window>.png``
     - Per-channel stim mask diagnostics.

.. note::
   TODO: save RSM into ``checkpoints/``

