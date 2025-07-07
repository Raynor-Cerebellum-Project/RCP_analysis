function n = nos(spix,M,w)
%NOS - Compute the number of spikes inside a marked domain
%
%	bc 	= nos( index, w, last, first)
%	
%       M 	= marker array
%
%       w 	= bin width
%

% (c) Claudio G. Rey - 8:36AM  8/31/92
   

   if nargin< 4, first=1; end 

   n = length(spix);

   flag = ix2fl(index);

   n = 0;
   for j = 1:n
      k = floor( (x-1) / w + .5) + 1; 
      if flag(k)==0, n = n + 1; end
   end
end
