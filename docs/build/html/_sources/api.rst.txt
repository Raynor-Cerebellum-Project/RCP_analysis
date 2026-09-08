Python API
==========

Reference for the modules in ``RCP_analysis/python/functions``, one page per
``.py`` file. These are the functions that the scripts in
``preprocessing_scripts/`` and ``analysis_scripts/`` are built from.

Most of the public names are re-exported at the package top level, so in
practice you import them as::

   import RCP_analysis as rcp

   params = rcp.load_experiment_params(yaml_path, repo_root)
   stim   = rcp.load_stim_detection(stim_npz_path)

Names that are **not** re-exported (private helpers, prefixed with ``_``, and a
few module-level helpers) must be imported from their module directly::

   from RCP_analysis.python.functions.utils import _butter_lowpass_ba

Modules
-------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Module
     - Purpose
   * - :doc:`api/params_loading`
     - Parse ``config/params.yaml`` / ``machines.yaml`` into ``experimentParams``.
   * - :doc:`api/config_loading`
     - Module-level constants: resolved ``PARAMS`` and all session directories.
   * - :doc:`api/utils`
     - General helpers: OCR/DLC, sync, peri-stim segments, binning, smoothing.
   * - :doc:`api/intan_preproc`
     - Intan stim-stream and aux-stream extraction, channel reordering.
   * - :doc:`api/br_preproc`
     - Blackrock sessions, Utah array mapping, region assignment, aux streams.
   * - :doc:`api/artifact_correction`
     - Incremental-PCA stim artifact correction and SpikeInterface wrappers.
   * - :doc:`api/artifact_correction_template_matching`
     - Offline per-block PCA template subtraction for stim artifacts.
   * - :doc:`api/subspace_detector`
     - Subspace-CFAR spike detector (GPU-accelerated) and amplitude gating.
   * - :doc:`api/reach_onset_detection`
     - Reach onset detection from kinematics, batch mode and trace alignment.
   * - :doc:`api/rsa_utils`
     - RSA pipeline: features, channel masks, RSM, silhouette, figures.
   * - :doc:`api/impedance_utils`
     - Parse Utah/Intan impedance files into bad-channel sets.

.. toctree::
   :maxdepth: 1
   :hidden:

   api/params_loading
   api/config_loading
   api/utils
   api/intan_preproc
   api/br_preproc
   api/artifact_correction
   api/artifact_correction_template_matching
   api/subspace_detector
   api/reach_onset_detection
   api/rsa_utils
   api/impedance_utils
