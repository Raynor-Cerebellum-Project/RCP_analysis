%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% File Name		: burst.m
% Author		: Jean-Ian A. Boutin
% Last Revised	: April 5, 2005
% Input			: 
%	1) ua: unit activity (spike train)
%	2) fs: sampling frequency
%	OPTIONAL INPUT
%	3) fcut: the end of the passband and the beginning of the stopband
%	4) ITL: energy threshold used for endpoint detection
% Output		:
%	1) fr: instantaneous firing rate estimate
%
% Definition	: This m file will generate the spike train frequency estimate from
%				 from a given filter for an input signal coming from burst cells
%
% fcut Discussion: Remember that as the difference between the two values inputted
% 	into fcut is large, the less restrictive in frequency the filter becomes. At the
% 	same time, it acquires a better time resolution. The inverse is also true,
% 	namely as the difference between the two fcut values is less, the filter will
% 	be more restrictive in frequency (preventing aliasing (good)) but will loose
% 	time accuracy. Its all about tradeoffs...
% 
% 	This emphasize the fact that a good knowledge of the bandwidth
% 	of a particular cell is key to get good firing rate estimate... 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fr_estimate] = burst(ua,fs,fcut,ITL,dev);

if nargin < 2
	error('JI==> must input at least unit activity and sampling frequency');
elseif nargin < 3
	fcut = [10 32.5];
	ITL = 20;
elseif nargin < 4
	ITL = 20;
end

% check if fcuts were entered appropiately
[row col] = size(fcut);
if row ~=1 | col ~=2
	error('JI==> fcut must be of the form fcut=[a b]');
end


% filter constants
% dev = [1.0203 0.01]; % passband ripple of 0.18 dB and stopband -40 dB
mag = [1 0]; % filter magnitude (we want passband as flat as possible
window_width = 2; % the width of the window to compute the energy

% f_num for burst cells
[n1,Wn,beta,ftype] = kaiserord(fcut,mag,dev,fs);
f_num = fir1(n1,Wn,ftype,kaiser(n1+1,beta),'noscale');

f_num = f_num/sum(f_num);

% f2_num for burst cells less restrictive
fcut = [10 82];
[n2,Wn,beta,ftype] = kaiserord(fcut,mag,dev,fs);
f2_num = fir1(n2,Wn,ftype,kaiser(n2+1,beta),'noscale');

% first the endpoints detection
end_p_fr = filter_vest(ua,fs,f2_num,round(n2/2));
[end_point] = endpoint_detector(end_p_fr,ITL,window_width);


% then filter with the filter designed for afferent cells
fr_estimate = filter_vest(ua, fs, f_num, round(n1/2));

% plot (fr_estimate);