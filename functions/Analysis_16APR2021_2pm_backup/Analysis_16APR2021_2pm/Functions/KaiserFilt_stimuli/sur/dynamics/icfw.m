function ic = icfw(b,f,M)
%ICFW  Output a decaying initial condition.
%
%  ic = icfw(b,f,M)  
%  
%  b is the mumerator of the system transfer function for each segment.
%  f is the denominator of the system transfer function.
%  M are the on and off markers for the domain.
%
%  The formula applied for creating the output for the kth segment is:
%
%  y = (bk/f)i
%

% (c) Claudio G. Rey - 1990-11-13

  nf = length(f)-1;
  I = zeros(max(M(:,2)-M(:,1))+nf+1,1); I(1)  = 1;
  for i=1:length(M(:,1)),
     K = M(i,1)-nf; k = M(i,2);
     ic(K:k) = filter(b(i,:),f,I(1:k-K+1))';
  end
