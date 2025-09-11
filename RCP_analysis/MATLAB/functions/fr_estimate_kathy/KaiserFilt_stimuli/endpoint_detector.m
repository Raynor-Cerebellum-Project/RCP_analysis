%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% File Name		: endpoint_detector.m
% Author		: Jean-Ian A. Boutin
% Last Revised	: April 5, 2005
% Input			: 
%	1) fr: neuronal spike train
%	2) ITL: lower energy treshold
%	3) window_width: width of window that computes the energy
%
% Definition	: This m file will find the beginning and end of each group 
%				 of cell activity
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [indices_min] = endpoint_detector(fr, ITL, window_width)

% constants
mean_sig = mean(fr);

% subtract mean fr so that the energy will be near zero when no activity
fr = fr - mean_sig;

% we can now compute each energy of each window
rect_window = ones(1,window_width);
energy = sqrt(conv(rect_window, fr.^2) / window_width); % here the overlap is one sample
% might want to reduce that

% look for endpoints
indices_min = energy > ITL;
% to have a square box has the delimiter, only need the logic array above...

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The following code is not used usually at run time. It was decided that a
% box under the graph would be sufficient. If the exact location of the
% endpoints is needed, it is provided by the next two variables.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

indices = diff(indices_min);
beginning = find(indices == 1);
ending = find(indices == -1);
