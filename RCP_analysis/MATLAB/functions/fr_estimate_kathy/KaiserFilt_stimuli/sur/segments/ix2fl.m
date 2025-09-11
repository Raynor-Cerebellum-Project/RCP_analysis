function [flag, last] = ix2fl( index, last, first)
%   
%   flag = ix2fl( index, last, first)
%
%	Change an index into a flag
%		Marked segments -> flag =0
%		otherwise, flag=1
%
%	first	first sample of the flag (default=1)
%	last	last  sample of the flag (default = last index)
%

% (c) Claudio G. Rey - 3:40PM  3/25/93

   if nargin < 2, last = index( length( index)); end,
   if nargin < 3, first = 1; end,

   if isempty( index) == 1, flag = []; return, end

   ixfirst = 1; ixlast  = length( index);

   if last < index( ixlast),
      last = index( ixlast);     
      disp('Increase last flag index?')
   end

   flag  = ones( last - first + 1,1);

   if ixlast==1,
      flag( index( 1)) = 0;
   else
      flag( index( ixfirst:ixlast)) = zeros( size(index( ixfirst:ixlast)'));
   end


