function [y,ic]=pefiltw(b,lb,a,la,u,M,FT,p)
%PEFILTW - filter indexed data 
%
%	[y,ic] = pefiltw(b,lb,a,la,u,M,FT,p)
%
%	y is obtained as the filtered output to the system b/a with
%	input u. M is the marker for the domain.  
%	FT is the filter type:
%	FT=2,4 -> continuous filter forcing function. 2=default
%	FT=1,3 -> reinitialized filter forcing function.
%	p = initial conditions. 
%	ic are the initial condition trajectories.
%

%	(c) Claudio G. Rey - 5:44PM  11/26/93

  na = length( a) - 1; nb = length( b);
  ns = length(M(:,1)); N = M(ns,2);
  y = zeros(N,1); ic = zeros(N,1);
  if nargin<7,FT=2;end

% Compute initial conditions:

  if nargin==8, ic = iw( p, a, la, M);  end

% Now filter each one of the marked segments:

  if ((FT==2) | (FT==4)), M = [M(1,1),M(ns,2)]; ns=1; end  
  for k=1:ns,
     y(M(k,1)-lb:M(k,2)) = filter(b,a,u(M(k,1)-lb:M(k,2)));
  end

  y = y + ic;

