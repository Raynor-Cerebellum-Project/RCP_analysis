function [th,p,bias,v,w,ic]=poebnw(z,ix,Mx,nn,P,maxit);
%

% (c) Claudio G. Rey - 03/11/1992

  N      = ix(length(ix));
  ns     = length(Mx(:,1));

  yf(:) = filter([.1 -.1],[1 P], z(1:N,2));

  z = [z(1:N,1),yf];
  [th,p,bias,v,w,ic] = oebw(z,ix,nn,1,1,maxit);
  V = th(2,1);

  plot(1:N,z(1:N,1),ix,w(ix)+ic(ix)+bias,'w-');


