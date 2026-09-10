Analyze LFP Bands
=================

Script: ``preprocessing_scripts/analyze_lfp_bands.py``

Analyzes LFP bands (Alpha, Beta, Low/High Gamma) for NPRW (Intan) and Utah Array (Blackrock).

Pipeline
--------

#. **Extraction** — Load stimulation events and extract epochs (−1000 ms to +1000 ms) with padding.
#. **Blanking** — Apply ``copy_baseline`` blanking to remove stim artifacts (−5 ms to +101 ms).
#. **Local CMR (Intan)** — Subtract common median within local radius (loaded from config).
#. **Utah CMR** — DISABLED.
#. **Filtering** — Zero-phase bandpass filtering in specific frequency bands (using padded epochs).
#. **Cleaning/Rejection:**

   - No baseline correction or exponential/template subtraction.
   - Bad channels rejected based on impedance thresholds (Intan > 7000 kOhm, Utah > 1000 kOhm).

Frequency bands
---------------

``LFP_BANDS`` defines six bands: delta (1-4 Hz), theta (4-8), alpha (8-12),
beta (12-25), low gamma (25-58), and high gamma (62-120), filtered with a
4th-order Butterworth.

.. warning::
   The per-band filtering is currently commented out in the script, so the
   saved files contain **broadband only** (``broadband_full``). The band
   definitions above are in place but not applied.

Inputs
------

#. Peri-stim tensors from :doc:`extract_peri_stim`, for event times

   - ``DATA_ROOT/results/checkpoints/PeriStim/<condition>/**/*.npz``

#. Aligned files from :doc:`make_aligned`

   - ``DATA_ROOT/results/checkpoints/Aligned/<condition>/aligned__*.npz``

#. BR-to-Intan shifts and the session metadata table

   - ``DATA_ROOT/Metadata/br_to_intan_shifts.csv``

#. UA channel mapping spreadsheet

   - ``config/probes/`` (set by ``mapping_mat_rel`` in ``config/params.yaml``)

Outputs
-------

All outputs go under ``UA_LFP``, sorted by condition and target the same way
the peri-stim files are:

- ``DATA_ROOT/results/checkpoints/UA_LFP/<condition>/[target_<A|B>/]aligned_lfp__<session>_<condition>[_target_<A|B>].npz``

The aggregated baseline files use a different name, grouped by depth and port
rather than by session:

- ``DATA_ROOT/results/checkpoints/UA_LFP/control_reaches/[target_<A|B>/]aligned_lfp__baseline__Depth_<depth>_port_<port>[_target_<A|B>].npz``

Contents
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Contents
   * - ``broadband_full``
     - Windowed LFP traces, trials x channels x time. Broadband only, see the
       warning above.
   * - ``fs_lfp``
     - Sampling rate after resampling (``TARGET_FS``, 1000 Hz).
   * - ``t_full_ms``, ``rel_time_pre``, ``rel_time_post``
     - Time base for the full window, and for the pre- and post-stim windows
       that exclude the blanked region.
   * - ``ua_ids_1based``
     - UA channel ids for the channel axis.
   * - ``category``, ``target``
     - Which condition and target this file holds.
   * - ``session``, ``stim_ms``
     - Session name and the stim times the epochs were cut around.
       Per-session files only.
   * - ``sessions``, ``br_indices``, ``n_trials``, ``group_port``,
       ``group_depth``
     - Which sessions were pooled, and the port and depth they were grouped
       by. Aggregated baseline files only.

Key parameters
--------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Parameter
     - Value
   * - ``TARGET_FS``
     - 1000 Hz, the rate everything is resampled to.
   * - ``EPOCH_PRE_MS``, ``EPOCH_POST_MS``
     - 1000 ms either side of the event.
   * - ``PAD_MS``
     - 1000 ms of extra padding, carried through filtering and trimmed after,
       so zero-phase filtering has no edge effects in the window of interest.
   * - ``BLANK_PRE_MS``, ``BLANK_POST_MS``
     - -5 ms to +101 ms, replaced by ``copy_baseline`` blanking.
   * - ``SKIP_EXISTING``
     - When true, sessions whose outputs already exist are skipped before
       Blackrock data is loaded.
