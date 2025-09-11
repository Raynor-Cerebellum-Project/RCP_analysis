function index=fl2ix(flag,trigger)
%FL2IX -  Convert a flag to and index array.
%
%	index = fltomx(flag,trigger)
%
%	index domain when flag=trigger.
%		trigger default value = 0
%

% (c) Claudio G. Rey, 1992-01-15.

if nargin<2, trigger=0; end,

% Mark each segment:

  nflag = length(flag);
  index = sort((1-abs(sign(flag-trigger)))'.*(1:nflag));
  [cero,i0] = min(index-(1:nflag)/nflag);
  index = index(max(i0)+1:nflag);
end