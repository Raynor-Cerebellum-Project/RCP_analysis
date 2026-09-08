Outputs and other plots
=======================

Intermediate outputs
--------------------

#. Stim stream ``.npz`` files from Intan, for stim timing and channels
#. Auxiliary ``.npz`` files for Intan sync signals
#. Auxiliary ``.npz`` files for BR sync signals
#. Metadata for alignment of BR and Intan
#. Checkpoint ``.npz`` files for aligned DLC files (e.g. two-camera to BR
   camera sync)
#. Preprocessed ``.npy`` files in spikeinterface format (UA and NPRW)
#. Aligned kinematics, NPRW, and UA data

Final outputs
-------------

#. Plots of kinematics (position and velocity) and neural data (firing rate
   plots)
#. RSA, to evaluate consistency of the stim response

`Example final figures: Nike's plots for reaching
<https://docs.google.com/presentation/d/1L_EA5BOvmqIdInJ0WPRp5rtgeDXCOmu6qmt2yeqPC-s/edit?slide=id.g3afc5261748_0_102#slide=id.g3afc5261748_0_102>`_

Other plots
-----------

``~/scripts/bryan_scripts/RSA_consistency/RSA_poststim_grouped_up.ipynb``
   RSA plots.

``~/scripts/bryan_scripts/dynamics/PoisLDS_NPRW_combined.ipynb``
   Dynamics.

``~/scripts/bryan_scripts/quick_plots/plot_HPF_traces_with_peaks.ipynb``
   Raw traces with peaks labeled.
