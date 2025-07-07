function [b,f,p,v,w1,w2] = pkoedw(z,ix,pole,nn,FT,T,M)
%PKOEDW Computes the prediction error estimate of an output-error model.
%
%	This routine requires that the poles be known beforehand.
%
%	[B,F,P] = pkoedw(Z,INDEX,poles,NN,FT,T)
%
%	This routine only works for single-input systems.
%
%	Z    :	The output-input data Z=[y u], y and u are column vectors.
%	INDEX:	Domain index.
%	POLE :	Vector containing pole estimates
%	NN   :	Initial value and structure information, given as
%	NN=[nb nf nk], the orders and delay of the above model
%
%	B,F,P:	Estimate of the B/F transfer function and of the initial  
%		conditions estimates
%
%	If M (marker array) is known using it for input makes the algorithm 
%	faster.  M can be read as the fourth output.
%
%	[B,F,P,v,w,ic] = pkoedw(Z,INDEX,POLE,NN,FT,T,M,maxsize)

%   Claudio G. Rey, 4:25PM  12/10/93

%
% *** Set up default values ***

  if nargin<7, M = ix2mx(ix); end

%  *** Do some consistency tests ***

%  keyboard
  [nr,cc]=size(nn); [Ncap,nz]=size(z(ix,:)); nu=nz-1; [N,nz] = size(z);
  if nz>Ncap, 
     error('The data should be organized in column vectors')
  return,end
  if nu>1, 
     error('This routine only works for single-input systems!')
  return,end
  if nu==0, 
  error('PKOEDW makes sense only if an input is present!')
  return,end

  nb = nn(1); nf = nn(2); nk = nn(3); nt = nb;
  nf = length(pole); ns = length(M(:,1));

  f = poly(pole.');

   if FT<3,

      n = nb + nf*ns;
      If = zeros(N,ns);
      for k = 1:ns,
         If(1:M(k,2),k) = iw( 1, f, nf, M(k,:));
      end

      phi = zeros( Ncap, n);

%
%     Compute the rest of the FIR part of the result

      o=0;
      for ii=1:nb,
         uf = filterdw( [zeros(nk+ii-1,1) 1], nb+nk-1, f, nf, z(:,2), M, FT, T);
         phi(:,o+ii) = uf(ix); 
      end, o = o+nb;

      for k=1:ns,
         for ii=1:nf, phi(:,o+ii) = If(ix-ii+1,k); end, o = o + nf;
      end

      th=phi\z(ix,1);

      for j=1:ns, p(j,1:nf) = th((j-1)*nf+nb+1:j*nf+nb)'; end

      b = [zeros(1,nk) th(1:nb)'];

      w = phi*th;

%disp('At pkoedw:'),keyboard

      [ v, w1, w2] = peoedw( b, f, z, M, FT, T, p); 

   else

      n = nb;
      [ic, p] = icw( f, nf, z(:,1), M, FT, T);
      y = z(:,1) - ic;
 
      phi=zeros( Ncap, n);

%
%     Compute the FIR part of the gradient:

      o=0;
      for ii=1:nb,
         uf = filterdw( [zeros(nk+ii-1,1) 1], nb+nk-1, f, nf, z(:,2), M, FT, T);
         phi(:,o+ii) = uf(ix); 
      end, o = o+nb;

      for k=1:ns,
         for j=1:nf, phi(:,nt+(k-1)*nf+j) = If(ix-j+1,k);end
      end
         
      th=phi\y(ix);

      b = [zeros(1,nk) th(1:nb)'];

      [v, w1, w2, p] = peoedw( b, f, z, M, FT, T); 

   end

