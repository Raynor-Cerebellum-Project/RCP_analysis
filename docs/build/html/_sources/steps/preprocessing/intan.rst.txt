Intan (NPRW) Preprocessing
===========================

Script: ``preprocessing_scripts/NPRW_Intan_analysis_mf.py``

Preprocesses the Intan data. Preprocessing and spike sorting are handled in Python using
`SpikeInterface <https://spikeinterface.readthedocs.io/>`_.

Steps
-----

#. Load geometry and mapping (``.mat`` file).
#. Extract stim data, IR crossings, and locations of stim pulses (individual pulses and blocks) — saves ``.npz`` file.
#. Extract auxiliary data (sync pulses) — saves ``.npz`` file.
#. Load Intan neural data, attach probe info, and reorder based on mapping.
#. Preprocess Intan (``.rhs``) data: high-pass filter, common local median reference (default radius: 30, 150 micrometers).
#. Remove artifacts (zero stim regions) + additional space specified in ``config/params.yaml``.
#. Use presaved template to perform matched filter and calculate MUA (peaks).
#. Save ``.npz`` file per condition.

Inputs
------

``.rhs`` files from Intan.

Outputs
-------

``pp_*.npz``
   Preprocessed files after preprocessing.

``aligned_*.npz``
   Per-condition files with spike times.
