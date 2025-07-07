function bc = ix2bc(index,w,last)
%IX2BC - Compute a spike index to bin counter conversion.
%
%	bc 	= ix2bc( index, w, last)
%	
%       w 	= bin width
%
%	last	= last of indexed array
%

% (c) Claudio G. Rey - 11:26AM  8/31/92

   if nargin< 4, n=1; end 

   n = length(index);

   bc = zeros( last, 1);

   ix = round( (index-1) / w) + 1;

   for j = 1:n

      k = ix(j);
      if k < last
         bc(k) = bc(k) +1;
      end

   end

