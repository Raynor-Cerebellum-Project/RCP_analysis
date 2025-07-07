function Signalno = listno( k, array, n);
%LISTNO - 
%
%	Signalno = listno( k, array, n);
%
%	Signalno 	- kth index of a rearranged array 
%	array		- string array (blank padded if necessary)
%	n		- swapped first element (see makelist)
%

% Claudio G. Rey - 1:43PM  9/24/93


   if nargin<3, n=1; end

   [rr,cc] = size( array);

   if k == n, k = 1; elseif k == 1, k = n; end 

   Listno = min( [rr, k]);

end
