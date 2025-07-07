function flag=ix2fl2( index, n)
%   
%   flag = ix2fl( index, n)
%
%	Change an index into a flag
%		Marked segments -> flag =0
%		otherwise, flag=1
%
%	n	is the output length of the flag.
%	n 	is computed from the markers if not given
%

% (c) Claudio G. Rey  1992-01-20

if nargin<2, n = index(length(index)); end,

%   Process between each pair of markers:

  flag  = ones(n,1);
  flag(index) = zeros(index');
end
