function M = xmark( M, ns, x, xmin, xmax, latency, srchout)
%XMARK - revise markers to data thresholds
%
%	M = xmark( M, ns, x, xmin, xmax, latency, srchout)
%
%	ns	- array containing the indexes for the regions of 
%			interest (segments) to be revised
%	x	- trigger signal for the threshold
%	xmax, xmin are the maximum and minimum thresholds.
%	latency	- the offset between the data and the markers
%	srchout	- max increase of the segment area 
%

%	(c) Claudio Rey - 11:17AM  4/5/93


   Mnew = [];
   lns  = length( ns);
   nc = fl2ix( 1 - ix2fl( ns, length( M( :, 1) ) ) );
   for k = 1:lns
      n  = ns( k);
      ix = (M( n, 1) - srchout + latency):( M( n, 2) + srchout + latency);
      flag( ix) = abs( sign( x( ix) - xmin) + sign( x( ix) - xmax));
      Magg = fl2mx( flag( ix)); Mnew = [Mnew; (Magg + ix( 1) - 1)];
   end

   if isempty( nc) == 0,
      M  = sort( [ (Mnew - latency); M( nc, :)]);
   else
      M  = sort( Mnew - latency);
   end

