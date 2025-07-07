function [bz, az] = delta2z( bdelta, adelta, Ts);
%DELTA2Z - Convert from delta to z domain representation
%
%	[bz] = delta2z( bdelta, Ts);
%
% or	[bz, az] = delta2z( bdelta, adelta, Ts);
%
%	bdelta/adelta => bz/az 
%
%	Ts - sampling time.
%
%	bz, bdelta, az and adelta are row vectors
%
%	if bdelta is an array the routine assumes that multiple 
%	numerators are to be converted (one per row).
%

% (c) Claudio G. Rey - 12:58PM  5/20/93

   if nargin<3, Ts = adelta; end

   [ns,N] = size( bdelta);

   for j = 1:ns

      bd = bdelta( j, N);

      for k = N:-1:2,

         bd = [bdelta( j, k-1), zeros( 1, N-k+1)] + [conv( bd, [1 -1])]/Ts;

      end

      bz( j, :) = bd;

   end

   if nargin == 3, 

      N = length( adelta); az = adelta( N);

      for k = N:-1:2,
      
         az = [adelta( k - 1), zeros( 1,N-k+1)] + [conv( az, [1 -1])]/Ts;

      end

      bz = bz / az( 1); az = az / az( 1);

   end

