Steps of the pipeline
=====================

Preprocessing
-------------

#. **Frame correction** — ``preprocessing_scripts/OCR_frame_correction.py``
#. **DeepLabCut 2-camera data alignment** — ``preprocessing_scripts/align_dlc_two_cams_to_br.py``
#. **VOG to Blackrock (BR) alignment** — ``preprocessing_scripts/align_VOG_to_br.py``
#. **NPRW preprocessing and spike detection** (matched filter) —
   ``preprocessing_scripts/NPRW_Intan_analysis_mf.py``
#. **Compute timeshift between Intan and BR** —
   ``preprocessing_scripts/compute_br_to_intan_shifts.py``
#. **UA preprocessing and spike detection** (subspace) —
   ``preprocessing_scripts/UA_BR_analysis_ssmf.py``

   * Alternatively use ``preprocessing_scripts/UA_BR_analysis_mf.py`` for a matched filter.

#. **Create aligned files** — ``preprocessing_scripts/make_aligned_npz_and_mat.py``

   * One file per condition, per side.
   * Output: ``data/results/checkpoints/.../aligned_*.npz``

#. **Extract peri-stim tensors** based on event timing (IR or stim) —
   ``preprocessing_scripts/extract_peri_stim.py``

   * Shape: trials × channels × time bins.
   * Output: ``data/results/checkpoints/.../peristim_*.npz``

#. **Output kinematic trajectories** for manual inspection —
   ``preprocessing_scripts/inspect_kinematics_trajectories.py``

Analysis
--------

#. **Kinematics analysis** — ``preprocessing_scripts/plot_plateau_analysis.py``

   * Rerun from ``preprocessing_scripts/extract_peri_stim.py`` after curation.

#. **Extract RSA tensors** from correlating regions of interest (ROI) —
   ``preprocessing_scripts/RSA_calculation.py``

   * Trial × trial correlation.
   * ROIs can be set in ``config/params.yaml``.

#. **Plot peri-stim firing rate plots** —
   ``analysis_scripts/plot_complete_shaded_BT.py``

   * Gaussian smoother parameters can be set in ``config/params.yaml``.

#. **Plot peri-stim rasters** — ``analysis_scripts/plot_peri_stim_raster.py``

Artifact correction schematic
-----------------------------

.. image:: /images/utah_array_analysis_and_artifact_correction.png
   :alt: Utah array analysis and artifact correction schematic
   :align: center

Detailed pages
--------------

.. toctree::
   :maxdepth: 2

   preprocessing/index
   analysis/index