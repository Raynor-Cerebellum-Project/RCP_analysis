Python API
==========

Reference for the modules in ``RCP_analysis/python/functions``. These are the functions used by scripts in ``preprocessing_scripts/`` and ``analysis_scripts/``.

Most of the public names are re-exported at the package top level, so in practice you import them as::

   import RCP_analysis as rcp

   params = rcp.load_experiment_params(yaml_path, repo_root)
   stim   = rcp.load_stim_detection(stim_npz_path)

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
     - General helpers
   * - :doc:`api/intan_preproc`
     - Intan data and aux extraction helpers, channel reordering.
   * - :doc:`api/br_preproc`
     - Blackrock data and aux extraction helpers, UA mapping, region assignment.
   * - :doc:`api/artifact_correction`
     - Incremental-PCA stim artifact correction helpers.
   * - :doc:`api/artifact_correction_template_matching`
     - Offline batch PCA template subtraction for stim artifacts.
   * - :doc:`api/subspace_detector`
     - Subspace-CFAR spike detector and amplitude gating.
   * - :doc:`api/reach_onset_detection`
     - Reach onset detection from kinematics, batch mode and trace alignment.
   * - :doc:`api/rsa_utils`
     - RSA pipeline: responses, channel masks, time domain plots, RSM construction, silhouette scores, figures.
   * - :doc:`api/impedance_utils`
     - Parse UA/NPRW impedance files

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
