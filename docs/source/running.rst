Running the pipeline
====================

#. Set ``data_root`` in ``config/params.yaml``.
#. Add an entry to ``config/machines.yaml`` if you are using a new workspace.
#. Run ``scripts/run_pipeline.py``.

Inputs
------

#. Raw neural signals (NPRW, UA)
#. Metadata from handwritten notes
#. Mapping and geometry
#. ``config/params.yaml``
#. Impedances for both NPRW and UA

Outputs
-------

``data/results/checkpoints/.../aligned_*.npz``
   Aligned spike times, behavioral traces, event timings, and metadata, one file per condition and target.

``data/results/checkpoints/.../peristim_*.npz``
   Peri-stim (event) tensors, one file per condition and target.

``data_status_reaching.csv``
   Timestamps that keep track of when each step was run.
