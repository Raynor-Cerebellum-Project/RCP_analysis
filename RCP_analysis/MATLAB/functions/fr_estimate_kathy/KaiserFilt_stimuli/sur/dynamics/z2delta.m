function [bdelta,adelta] = z2delta(bz,az,Ts);
%z2delta - convert form z to delta domain
%
%	[bdelta] = z2delta( bz, Ts);
%
% or	[bdelta, adelta] = z2delta( bz, az, Ts);
%
%	bz/az => bdelta/adelta 
%
%	Ts - sampling time.
%
%	bz, bdelta, az and adelta are row vectors
%
%	if bz is an array the routine assumes that multiple 
%	numerators are to be converted (one per row).
%

% (c) Claudio G. Rey - 12:59PM  5/20/93


   if nargin<3, Ts = az; end

   [ns,N] = size( bz);

   for j = 1:ns

      bd = bz( j, N);

      for k = N:-1:2,

         bd = [bz( j, k-1), zeros( 1, N-k+1)] + [conv( bd, [1 -Ts])];

      end

      bdelta( j, :) = bd;

   end

   if nargin == 3, 

      N = length( az); adelta = az( N);

      for k = N:-1:2,
      
         adelta = [az( k - 1), zeros( 1,N-k+1)] + [conv( adelta, [1 -Ts])];

      end

      bdelta = bdelta / adelta( 1); adelta = adelta / adelta( 1);

   end


