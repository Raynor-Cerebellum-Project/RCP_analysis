
function [recovery]=filter_vest(spikes, fs, filter_num, group_delay)

number_fft = 1024;

% this is only to put the units in spikes/sec
spikes = spikes * fs;

% we can convolve this with the spikes and see what we get
% recovery = filter(filter_num, 1, spikes);
recovery = conv(spikes, filter_num');

% before plotting the thing, we need to remember that the filter introduces
% a delay of order/2
recovery = recovery(group_delay:length(recovery)-group_delay);
