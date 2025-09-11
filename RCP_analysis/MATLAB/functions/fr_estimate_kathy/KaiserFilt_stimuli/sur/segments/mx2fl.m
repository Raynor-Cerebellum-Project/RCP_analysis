function flag=mx2fl( M, n)
%   
%   flag = mx2fl( M, n)
%
%	Change a set of markers into a flag
%		Marked segments -> flag =0
%		otherwise, flag=1
%
%	n	is the output length of the flag.
%	n 	is computed from the markers if not given
%

% (c) Claudio G. Rey  1992-01-15

   if nargin<2, n = length(M(:,1)); n = M(n,2); end,

   flag  = ones(n,1);
   index = mx2ix(M);
   flag(index) = zeros(index');

end
