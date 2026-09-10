Inspect Kinematics Trajectories
================================

Script: ``preprocessing_scripts/inspect_kinematics_trajectories.py``

Plots kinematics trajectories for trial inspection and manual removal.

Each subplot title shows:

- ``Trial:i (Raw:R)`` where:

  - ``i`` = position in the plot (for reference).
  - ``R`` = stable raw stim-pulse index — **this is what goes in** ``manual_trial_remove.csv``.

Inputs
------

#. Any control and stim trials from :doc:`extract_peri_stim`

   - ``DATA_ROOT/results/checkpoints/PeriStim/stim_reaches/target_{A,B}/*.npz``
   - ``DATA_ROOT/results/checkpoints/PeriStim/control_reaches/target_{A,B}/*.npz``
   - ``DATA_ROOT/results/checkpoints/PeriStim/continuous_stim/*.npz``
   - ``DATA_ROOT/results/checkpoints/PeriStim/at_rest/*.npz``

Outputs
-------

One X-Y trajectory grid per peri-stim file, laid out in a
``N_COLS``-wide grid of subplots with a shared colorbar for time:

- ``DATA_ROOT/results/figures/kinematics_trajectories/<condition>/<target>/BR_<br_idx>_xy.jpg``

``<condition>`` is one of ``stim``, ``control``, ``continuous_stim``, or
``at_rest``; ``<target>`` is ``target_A``/``target_B`` for the reach conditions
and repeats the condition name for the others.

Plots use the ``middle`` keypoint from ``cam1`` by default, set by ``KEYPOINT``
and ``CAMERA`` at the top of the script.

Manual curation
---------------

This step exists to decide which trials to drop. Record the raw index ``R`` from
each subplot title in:

- ``config/manual_trial_remove.csv``

Then re-run :doc:`extract_peri_stim`, which reads that file and gates the listed
trials out.

