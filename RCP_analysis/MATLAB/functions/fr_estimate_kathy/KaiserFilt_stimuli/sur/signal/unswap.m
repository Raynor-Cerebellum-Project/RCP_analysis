function j = unswap( k, n);
%UNSWAP - 
%
%	j = unswap( k, n);
%
%	k 	- possibly swapped index
%	n	- swapped first element
%	j	- unswapped index
%

% Claudio G. Rey - 3:27PM  9/24/93


   if k == n, k = 1; elseif k == 1, k = n; end 

   j = k;

end
