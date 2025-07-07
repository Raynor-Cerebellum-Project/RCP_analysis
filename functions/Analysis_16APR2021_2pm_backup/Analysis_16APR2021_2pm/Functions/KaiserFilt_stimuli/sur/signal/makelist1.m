function list = makelist( array, n, array2);
%MAKELIST - 
%
%	list = makelist( array, n);
%
%	array	string array (blank padded if necessary)
%	list 	single row string array with '|'s as separators
%	n	swapped first element 
%		(the first element is often swapped when displaying a list
%		to highlight the nth element in a popup box)
%

% Claudio G. Rey - 1:34PM  9/24/93


   if nargin<2, n=1; end

   [rr,cc] = size( array);

   if nargin == 3, [rr2,cc] = size( array2);end

   list = deblank(array(1,:));

   if nargin==3, list = [list ' - '  array2(n-floor((n-1)/rr2)*rr2,:)]; end

   if n == 1, range = 2:rr; else, range = [2:n]; end %range = [2:n-1 1 n+1:rr]; end

   for k = range, 

      list = [list '|' deblank(array(k,:))]; 
      if nargin==3, list = [list ' - '  array2(k-floor((k-1)/rr2)*rr2,:)]; end

   end


