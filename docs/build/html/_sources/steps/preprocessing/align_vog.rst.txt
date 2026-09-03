VOG to Blackrock Alignment
===========================

Script: ``preprocessing_scripts/align_VOG_to_br.py``

Aligns VOG to the Blackrock recording. Attaches BR ns2 sample times to VOG CSVs.

.. note::
   You may need to double-check which channel the sync pulse is on (``VOG_sync_ch`` in ``config/params.yaml``).

Inputs
------

VOG ``.csv`` files.

Outputs
-------

``results/checkpoints/VOG/*_VOG_aligned.csv``
   Aligned VOG data with BR ns2 sample times attached.
