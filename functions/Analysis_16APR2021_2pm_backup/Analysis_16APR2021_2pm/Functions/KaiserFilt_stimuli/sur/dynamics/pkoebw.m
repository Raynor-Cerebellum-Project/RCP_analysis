function [b,f,o,p,v,w,ic] = pkoebw(z,ix,pole,nn,o,FT,M,maxsize,Tsamp)
%PKOEBW Computes the prediction error estimate of an output-error model
%	assuming that the poles are known.
%
%	This routine requires that the poles be known beforehand.
%
%       [B,F,O,P] = pkoebw(Z,INDEX,POLE,NN,O,FT)
%
%	This routine only works for single-input systems.
%
%	Z    :	The output-input data Z=[y u], y and u are column vectors.
%	INDEX:	Domain index.
%	POLE :	Vector containing pole estimates
%	NN   :	Initial value and structure information, given as
%	NN=[nb nf nk], the orders and delay of the above model
%
%       B,F,P,O:  Estimate of the B/F transfer function, of the initial
%               conditions and of the offset, if the offset is unknown
%		and needs to be computed make 
%		o = NaN, (NaN = 'not a number' in matlab);
%
%	If M (marker array) is known, using it for input makes the algorithm 
%	faster.
%
%       [B,F,O,P,v,w,ic] = pkoebw(Z,INDEX,POLE,NN,O,FT,M,maxsize,Tsamp)

%   (c) Claudio G. Rey, 9:02AM  7/7/93

% *** Set up default values ***

   [cmptr,maxsdef] = computer;
   if maxsdef < 8192, maxsdef=4096; else maxsdef=256000;end
   if nargin<9, Tsamp=1;end
   if nargin<8, maxsize=maxsdef;end
   if nargin<7, M = ix2mx(ix); end
   if nargin<6, FT=2; end
   if nargin<5, o=NaN; end

   if Tsamp<0,Tsamp=1;end 
   if maxsize<0, maxsize=maxsdef;end

%  *** Do some consistency tests ***

   if FT>2, 
      error('cannot fit transients to data and input at the same time!');
      return
   end
   [nr,cc]=size(nn); [Ncap,nz]=size(z(ix,:)); nu=nz-1; [N,nz] = size(z);
   if nz>Ncap, 
      error('The data should be organized in column vectors')
   return,end
   if nu>1, 
      error('This routine only works for single-input systems!'), return
   end
   if nu==0, 
      error('PKOEBW makes sense only if an input is present!'), return
   end
 
   nb = nn( 1); nf = nn( 2); nk = nn( 3); nt = nb;
   nf = length( pole); ns = length( M( :, 1));

   [a,f] = zp2tf( [], pole', 1);

   uf = pefiltw(  1, nb + nk - 1, f, nf, z( :, 2), M);

   if isnan(o) == 1, n = nb + ns * nf + 1; else, n = nb + ns * nf; end

   icf = zeros( N, ns);
   for k = 1:ns,
      If( 1:M(k,2), k) = iw( 1, f, nf, M( k, :));
   end

   Mem = floor( maxsize / n);
   R = zeros( n); F = zeros( n, 1);
   for k=0:Mem:Ncap-1,
      jj=ix(k+1:min(Ncap,k+Mem));
      m = 0;
      phi = zeros( length( jj), n);
      for j=1:nb, phi( :, j + m) = uf( jj - j - nk + 1); end, m = m + nb;
      for k=1:ns,
         for j=1:nf,
            phi( :, j + m) = If( jj - j + 1, k);
         end, m = m + nf;
      end,
      if isnan(o) == 1,
         phi( :, m + 1) = ones( jj)'; m = m + 1;
      end
      if Ncap > Mem, R = R + phi' * phi; F = F + phi' * z( jj, 1); end
   end
   if Ncap > Mem, th = R \ F; else th = phi \ z( jj, 1);end

   for j=1:ns, p( j, 1:nf) = th( (j-1) * nf + nb + 1: j * nf + nb)'; end

   b = [zeros(1,nk) th(1:nb)'];

   if isnan(o) == 1,  o = th(n); end

   [v, w, ic] = peoew( b, nb + nk - 1, f, nf, [z(:,1)-o, z(:,2)], M, FT, p);

end
