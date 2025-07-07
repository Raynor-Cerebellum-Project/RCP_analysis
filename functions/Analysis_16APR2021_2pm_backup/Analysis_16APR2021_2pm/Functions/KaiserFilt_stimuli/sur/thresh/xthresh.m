function [M, xmin, xmax, srchout] = xthresh(xxx,M,ns,xmin,xmax,srchout);
%XTHRESH - revise markers to data using new thresholds
%
%	M = xthresh(xxx,M,ns,xmin,xmax,srchout);
%
%	M	- markers 
%	ns	- array containing the indexes for the regions of 
%			interest (segments) to be revised
%	xxx	- trigger signal for the threshold
%	xmax, xmin are the maximum and minimum thresholds.
%	srchout	- max increase of the segment area 
%

% (c) Claudio G. Rey - 11:23AM  5/6/93

%  possible latency compensation available by changing the next line:
   latency = 0;

   if nargin<4, xmin    = -9999; end
   if nargin<5, xmax    =  9999; end
   if nargin<6, srchout =    15; end

   Mnew = [];
   lns  = length( ns);
   nc = fl2ix( 1 - ix2fl( ns, length( M( :, 1) ) ) );

   for k = 1:lns
      n  = ns( k);
      ix = (M( n, 1) - srchout + latency):( M( n, 2) + srchout + latency);
      flag( ix) = abs( sign( xxx( ix) - xmin) + sign( xxx( ix) - xmax));
      Magg = fl2mx( flag( ix)); Mnew = [Mnew; (Magg + ix( 1) - 1)];
   end

   if isempty( nc) == 0,
      M  = sort( [ (Mnew - latency); M( nc, :)]);
   else
      M  = sort( Mnew - latency);
   end

   M = cull( M, srchout);

