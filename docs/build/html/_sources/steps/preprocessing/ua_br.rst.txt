Utah Array (Blackrock) Preprocessing
======================================

Preprocessing and spike sorting for Blackrock data, handled in Python using
`SpikeInterface <https://spikeinterface.readthedocs.io/>`_.
Two spike detection methods are available.

Subspace Matched Filter (recommended)
--------------------------------------

Script: ``preprocessing_scripts/UA_BR_analysis_ssmf.py``

Steps
^^^^^

#. Load geometry and mapping (``.xlsm`` file).
#. Build ``.npz`` for auxiliary data (channels in ``.ns5`` and ``.ns2`` file, HR, touchscreen, etc.).
#. Extract target labels from touchscreen signal (with DLC kinematics fallback if touchscreen is unresponsive).
#. Load neural data (``.ns6``) and config.
#. **Artifact Correction:**

   - Per-channel lag calibration using CSV lookup.
   - Region-wise Incremental PCA (IPCA) artifact subtraction (rank 7 by default).
   - Generates diagnostic alignment plots.

#. High-pass filter cleaned data.
#. **Spike Detection via Subspace Matched Filtering:**

   - Uses pre-computed extremum templates (``median_extremum_basis_norm_UA_PortA_r3.npy``).

#. Save ``.npz`` file per session with peaks, noise levels, and metadata.

Matched Filter (alternative)
------------------------------

Script: ``preprocessing_scripts/UA_BR_analysis_mf.py``

Same pipeline as above except spike detection uses standard matched filtering with
``median_extremum_templates_norm_UA_PortB.npy``, falling back to the ``locally_exclusive``
method if matched filtering fails.

Inputs
------

``.ns6`` files from Blackrock.

Outputs
-------

``pp_*.npz``
   Preprocessed files.

``aligned_*.npz``
   Per-condition files with spike times.
