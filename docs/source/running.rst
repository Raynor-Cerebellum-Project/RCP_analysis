Running the pipeline
====================

#. Add an entry to ``config/machines.yaml`` if you are using a new workspace.
#. Set ``data_root`` in ``config/params.yaml``. In the documentation this will be referred to as ``DATA_ROOT``
#. Run ``scripts/run_pipeline.py``.

Inside ``scripts/run_pipeline.py``, you can specify which sessions and which steps to run.

Inputs
------

#. Raw neural signals (NPRW, UA) ``DATA_ROOT/Intan/`` and ``DATA_ROOT/Blackrock/``
#. Videos ``DATA_ROOT/Video/``
#. Metadata from handwritten notes ``DATA_ROOT/Metadata/``
#. Impedances for both NPRW and UA ``DATA_ROOT/Impedances/``
#. Mapping and geometry ``config/../``
#. ``config/params.yaml`` and ``config/machine.yaml``


Outputs
-------

#. Aligned spike times, behavioral traces, event timings, and metadata, one file per condition and target.

   - ``DATA_ROOT/results/checkpoints/Aligned/.../aligned_*.npz``

#. Peri-stim (event) tensors, one file per condition and target.

   - ``DATA_ROOT/results/checkpoints/PeriStim/.../peristim_*.npz``

#. Timestamps that keep track of when each step was run.

   - ``DATA_ROOT/data_status_reaching.csv``


Final outputs
-------------

#. Kinematic plots

   - ``DATA_ROOT/results/figures/kinematics_trajectories/.../``
   - ``DATA_ROOT/results/figures/plateau_analysis/``

#. Neural data plots

   - Firing rate: ``DATA_ROOT/results/figures/shaded_BT_svg/``
   - Raster: ``DATA_ROOT/results/figures/peristim_raster/.../``

#. RSA plots

   - ``DATA_ROOT/results/figures/rsa_from_peristim/``

`Example final figures: Nike's plots for reaching
<https://docs.google.com/presentation/d/1L_EA5BOvmqIdInJ0WPRp5rtgeDXCOmu6qmt2yeqPC-s/edit?slide=id.g3afc5261748_0_102#slide=id.g3afc5261748_0_102>`_
