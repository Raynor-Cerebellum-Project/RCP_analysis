function signal = smooth(input, cut_off, steep)

% smooth.m
% signal = smooth(input, cut_off, steep)
% two-sided filter
% input - signal to be filtered
% cut_off - frequency in Hz.
% steep - steepness of decay 50 is shallow, 201 is steep
%
%  JER March, 1997


B = fir1(steep,cut_off/1000*2);

signal = filtfilt(B,1,input);



