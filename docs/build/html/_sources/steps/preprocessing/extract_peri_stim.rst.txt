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
#. **Continuous stim** — aligned to IR crossing, no target splitting.

Steps
-----

#. Load aligned ``.npz`` files from the appropriate condition directory.
#. Apply behavior-based trial gating (drop trials with bad kinematics).
#. Compute trial labels (A/B/N) from touchscreen state.
#. Extract peri-event traces for neural data (NPRW, UA).
#. Deduplicate MUA peaks, bin counts, and estimate firing rates using Gaussian kernel smoothing.
#. Baseline-correct traces (subtract mean of first 150 ms).
#. Calculate median, variance, and mean traces across trials.
#. Extract peri-event behavior traces (position and velocity).
#. Interpolate small NaN gaps in kinematics (≤4 samples).
#. Z-score normalize behavior columns.
#. Separate left (A) and right (B) reaches for stim/control conditions.
#. Output ``.npz`` and ``.mat`` files in ``results/checkpoints/PeriStim/<condition_type>/``.

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
