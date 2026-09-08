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

Associated Scripts
------------------

``debug_lfp_pipeline_plots.py``
   Visualizes the pipeline steps.

``plot_lfp_check.py``
   General quality check of aligned LFP.

Outputs
-------

``results/checkpoints/NPRW_LFP/aligned_lfp__{session_name}.npz``
   NPRW (Intan) windowed LFP traces for each band.

``results/checkpoints/UA_LFP/aligned_lfp__{session_name}.npz``
   Utah Array (Blackrock) windowed LFP traces for each band.
