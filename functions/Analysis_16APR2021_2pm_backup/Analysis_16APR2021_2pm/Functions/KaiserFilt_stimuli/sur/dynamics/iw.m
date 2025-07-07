function ic = iw( p, f, lf, M)
%IW  Output a decaying initial condition starting at -lf.
%
%  ic = iw( p, f, lf, M)  
%  
%  p is the mumerator of the system transfer function for each segment.
%  f is the denominator of the system transfer function.
%  M are the on and off markers for the domain.
%
%  The formula applied for creating the output for the kth segment is:
%
%  y = ( P(q)^k / F(q) ) I(-lf)
%

% (c) Claudio G. Rey - 8:19AM  3/19/93

  ns = length(M(:,1)); 
  ic = zeros(M(ns,2),1); ic(M(:,1)-lf) = ones(ns,1);
  for j=1:ns,
     K = M(j,1)-lf; k = M(j,2);
     ic(K:k) = filter( p(j,:), f, ic(K:k));
  end
