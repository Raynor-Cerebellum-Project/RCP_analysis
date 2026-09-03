Create Aligned Files
====================

Script: ``preprocessing_scripts/make_aligned_npz_and_mat.py``

Creates the ``aligned.npz`` files by combining neural, auxiliary, and behavioral data into a single aligned file per condition.

Steps
-----

#. Align neural, aux, and behavioral data (NPRW, UA, kinematics, HR, target, and VOG).
#. Create spike time (peaks) dictionaries for both NPRW and UA.
#. Permute UA channel labels and UA peak dictionary by region.
#. Output ``.npz`` and ``.mat`` files in ``~/results/checkpoints/Aligned``.

Outputs
-------

``Aligned/control_reaches/``
   Control trials (no stim).

``Aligned/stim_reaches/``
   Stimulation condition reaches.

``Aligned/at_rest/``
   At rest conditions (no reach).

``Aligned/Grasp/``
   Grasp experiment trials.

``Aligned/IMU/``
   IMU experiment trials.

``Aligned/continuous_stim/``
   Continuous stimulation trials.
