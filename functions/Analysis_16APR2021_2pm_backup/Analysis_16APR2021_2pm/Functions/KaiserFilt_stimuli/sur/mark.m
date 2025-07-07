function M = mark(sig,xmin,xmax,latency)

% Function mark
%
% Produces identical results to the function 'mark' from
% 'analysis', but in about 1/10 to 1/20 the time.
%
% Usage: M = mark(sig,xmin,xmax,latency)
%
% This function looks through the vector 'sig' and finds the 
% ranges of points which are greated than xmin, but less than
% xmax.
%
% If the argument 'latency' is provided, it is subtracted 
% from M.
%

%
% GW 8/23/97
%

if nargin < 4; latency = 0; end;

mf = find( (sig > xmin) & (sig < xmax) );
df = find(diff(mf) > 1);
if ~isempty(mf)
  start = mf([1;df+1]);
  stop = mf([df;length(mf)]);
  M = [start stop]-latency;
else
  M = [];
end