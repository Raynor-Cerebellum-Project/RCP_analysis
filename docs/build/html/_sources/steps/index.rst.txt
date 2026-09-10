Steps of the pipeline
=====================

Preprocessing
-------------
*  ``preprocessing_scripts/.../``

#. **OCR Frame correction**: ``OCR_frame_correction.py``

   * Corrects for misaligned frames in the video.
  
#. **DeepLabCut 2-camera data alignment**: ``align_dlc_two_cams_to_br.py``

   * Aligns 2-camera DLC data to Blackrock (BR) sync signal

#. **VOG to Blackrock (BR) alignment**: ``align_VOG_to_br.py``

   * Aligns VOG data to BR sync signal (If exists)

#. **NPRW preprocessing and spike detection** (matched filter): ``NPRW_Intan_analysis_mf.py``
#. **Compute timeshift between Intan and BR**: ``compute_br_to_intan_shifts.py``
#. **UA preprocessing and spike detection** (subspace): ``UA_BR_analysis_ssmf.py``

   * Alternatively use ``UA_BR_analysis_mf.py`` for a matched filter.

#. **Create aligned files**: ``make_aligned_npz_and_mat.py``

   * One file per condition, per side.
   * Output: ``DATA_ROOT/results/checkpoints/Aligned/.../aligned_*.npz``

#. **Extract peri-stim tensors** based on event timing (IR or stim): ``extract_peri_stim.py``

   * Shape: trials × channels × time bins.
   * Output: ``DATA_ROOT/results/checkpoints/PeriStim/.../peristim_*.npz``

#. **Output kinematic trajectories** for manual inspection: ``inspect_kinematics_trajectories.py``

Analysis
--------
* ``analysis_scripts/.../``

#. **Kinematics analysis**: ``plot_plateau_analysis.py``

   * Rerun from ``preprocessing_scripts/extract_peri_stim.py`` after curation.

#. **Extract RSA tensors** from correlating regions of interest (ROI) —
   ``RSA_calculation.py``

   * Trial × trial correlation.
   * ROIs can be set in ``config/params.yaml``.

#. **Plot peri-stim firing rate plots** —
   ``plot_complete_shaded_BT.py``

   * Gaussian smoother parameters can be set in ``config/params.yaml``.

#. **Plot peri-stim rasters**: ``plot_peri_stim_raster.py``

Artifact correction schematic
-----------------------------

.. image:: /images/utah_array_analysis_and_artifact_correction.png
   :alt: Utah array analysis and artifact correction schematic
   :align: center


Intermediate outputs
--------------------

#. Stim stream ``.npz`` files from Intan, for stim timing and channels

   - ``DATA_ROOT/results/aux_data/NPRW/<session>_Intan_streams/stim_stream.npz``

#. Auxiliary ``.npz`` files for Intan sync signals

   - ``DATA_ROOT/results/aux_data/NPRW/<session>_Intan_streams/aux_streams.npz``

#. Auxiliary ``.npz`` files for BR sync signals

   - ``DATA_ROOT/results/aux_data/UA/<session>__BR_aux_data.npz``

#. Metadata for alignment of BR and Intan

   - ``DATA_ROOT/Metadata/br_to_intan_shifts.csv``

#. Checkpoint ``.csv`` files for aligned DLC files (e.g. two-camera to BR
   camera sync)

   - ``DATA_ROOT/results/checkpoints/Behavior/<condition>_both_cams_aligned.csv``

#. Preprocessed ``.npy`` files in spikeinterface format (UA and NPRW)

   - ``DATA_ROOT/results/checkpoints/NPRW/pp_local_<inner>_<outer>__interp_<session>/``
   - ``DATA_ROOT/results/checkpoints/UA/pp__<session>__NS6/``

#. Binned firing rates, one file per session and probe

   - ``DATA_ROOT/results/checkpoints/NPRW/rates__<session>__bin<bin>ms_sigma<sigma>ms.npz``
   - ``DATA_ROOT/results/checkpoints/UA/rates__<session>__bin<bin>ms_sigma<sigma>ms.npz``

#. Aligned kinematics, NPRW, and UA data

   - ``DATA_ROOT/results/checkpoints/Aligned/<condition>/aligned__<session>__Intan_<idx>__BR_<idx>.npz``


Other plots
-----------

#. RSA plots

   - ``scripts/bryan_scripts/RSA_consistency/RSA_poststim_grouped_up.ipynb``
   
#. Dynamics

   - ``scripts/bryan_scripts/dynamics/PoisLDS_NPRW_combined.ipynb``

#. Raw traces with peaks labeled

   - ``scripts/bryan_scripts/quick_plots/plot_HPF_traces_with_peaks.ipynb``

#. Inspect individual Blackrock recordings

   - ``scripts/nikita_scripts/plotting_scripts/plot_blackrock_contents.ipynb``

#. Impedances over time

   - ``scripts/nikita_scripts/plotting_scripts/plot_impedances_over_time.ipynb``

#. Heart rate over time

   - ``scripts/nikita_scripts/plotting_scripts/plot_heart_rate.ipynb``


Detailed pages
--------------

.. toctree::
   :maxdepth: 2

   preprocessing/index
   analysis/index