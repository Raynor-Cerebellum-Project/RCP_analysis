function index=mx2ix( M)
%MX2IX	Convert from a marker domain descriptor to its equivalent index descriptor
%
%	index = mx2ix(M)
%
%	M	-	marker descriptor
%	index	-	index descriptor
%

% (c) Claudio G. Rey  - 9:00AM  6/1/93

   ns = length(M(:,1));
   if ns==0, error('No values in the domain'); end

%   Process between each pair of markers:

   ix = 1;
   for k = 1:ns,
      index(ix+(0:M(k,2)-M(k,1))) = (M(k,1):M(k,2));
      ix = ix + M(k,2) - M(k,1) + 1;
   end
   index = index(1:ix-1);
