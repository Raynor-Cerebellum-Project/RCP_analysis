OCR Frame Correction
====================

Script: ``preprocessing_scripts/OCR_frame_correction.py``

Because FLIR cameras could drop frames, OCR is used to get a mapping of the frames to the actual times.

Inputs
------

#. Two camera ``.avi`` files

Outputs
-------

``.csv`` files containing the mapping of frames to correct frames/times.
