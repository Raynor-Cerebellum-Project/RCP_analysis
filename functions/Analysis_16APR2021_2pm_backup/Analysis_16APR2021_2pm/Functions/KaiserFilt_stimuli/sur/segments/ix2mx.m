function M=ix2mx(index)
%IX2MX - Convert a index domain descriptor into 
%	marker domain descriptors
%
%	M = ix2mx( index)
%	
%

% (c) Claudio G. Rey - 10:47AM  6/30/93
   

   n = length(index); 

   if n > 1, 
      ix1 = 1:n-1; ix2 = 2:n;
      Flagon = zeros(n,1);  Flagoff = zeros(n,1);

%     Flag when a change in the index >1 occurs:
 
      Flagoff(  n) = 1;
      Flagoff(ix1) = sign(abs(index(ix2)-1-index(ix1)));
      Flagon(   1) = 1;
      Flagon( ix2) = Flagoff(ix1);

%     Convert index change flags into set of markers:

      Flagoff      = sort(Flagoff.*(index'));
      Flagon       = sort(Flagon .*(index'));
      [z,i]        = min(Flagon - (1:n)'/n); if z >=0, i=0; end;
      M            = [Flagon(i+1:n),Flagoff(i+1:n)];

   elseif n == 1

      M            = [index index];

   else

      M            = []

   end

