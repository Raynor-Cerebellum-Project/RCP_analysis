Compute BR-to-Intan Shifts
==========================

Script: ``preprocessing_scripts/compute_br_to_intan_shifts.py``

Computes the timeshift between the Intan and Blackrock recordings.

Steps
-----

#. Load metadata to match Intan files to BR files.
#. Load template sent from BR to Intan and the Intan ADC file.
#. Match the template to the BR template signal.
#. Save shifts, adjusted shifts, and related information to ``data_root/location/Metadata/br_to_intan_shifts.csv``.
