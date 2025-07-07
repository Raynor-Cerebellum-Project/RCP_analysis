function [b,f,o,p,v,w,ic] = pkoebdw(z,ix,pole,nn,o,FT,T,M,sparsemethod)
%PKOEBW Computes the prediction error estimate of an output-error model
%	assuming that the poles are known.
%
%	This routine requires that the poles be known beforehand.
%
%       [B,F,O,P] = pkoebdw(z,index,pole,NN,o,FT,T)
%
%	This routine only works for single-input systems.
%
%	z    :	The output-input data Z=[y u], y and u are column vectors.
%	index:	Domain index.
%	pole :	Vector containing pole estimates
%	NN   :	Initial value and structure information, given as
%	NN=[nb nf nk], the orders and delay of the above model
%
%       B,F,P,O:  Estimate of the B/F transfer function, of the initial
%               conditions and of the offset, if the offset is unknown
%		and needs to be computed make 
%		o = NaN, (NaN = 'not a number' in matlab);
%		for no offset make o = []; (empty)
%
%	If M (marker array) is known, using it for input makes the algorithm 
%	faster.
%
%       [B,F,O,P,v,w,ic] = pkoebdw(Z,index,pole,NN,O,FT,T,M)

%   (c) Claudio G. Rey, 4:55PM  12/27/93


   if nargin<8, M = ix2mx(ix); end
   if nargin<9, sparsemethod = 0; end

%  *** Do some consistency tests ***

   if FT>2 & isnan(o), 
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
%keyboard
   if nu==0, 
      error('PKOEBDW makes sense only if an input is present!'), return
   end
 
   nb = nn( 1); nf = nn( 2); nk = nn( 3); nt = nb;
   nf = length( pole); ns = length( M( :, 1));

   if isempty(o), o=0; end

   f = poly( pole);

   if FT<3,
      n = nb+ns*nf;

      CIf = [];
      for k = 1:ns,
         If = zeros(N,1);
         If( 1:M(k,2)) = iw( 1, f, nf, M( k, :));

         if sparsemethod
            CIf = [CIf,sparse(If)];
         else
            CIf = [CIf,(If)];
         end
      end

      y=z(:,1);

   else

      n = nb;
      [ic, p] = icw( f, nf, z(:,1), M, FT, T);
      y = z(:,1) - ic;
 
   end

   if isnan(o)==1, n=n+1; else y = y-o; end

   m = 0;

   for ii=1:nb,
      uf = filterdw( [zeros(nk+ii-1,1) 1], nb+nk-1, f, nf, z(:,2), M, FT, T);
      phi(:,m+ii) = uf(ix);
   end, m = m + nb;

   if sparsemethod, phi = sparse(phi); end

   if FT<3,
      for k=1:ns,
         for j=1:nf,
            phi( :, j + m) = CIf( ix-j+1, k);
         end, m = m + nf;
      end,
   end

   if isnan(o)==1,
      phi( :, m + 1) = ones( Ncap,1); m = m + 1;
   end

   if sparsemethod, 
      R = phi'*phi; F = phi'*y(ix); th= R\F;
   else
      th = phi \ y( ix);
   end

   b = [zeros(1,nk) th(1:nb)'];
   m = nb;

   if FT<3,
      for j=1:ns, 
         p( j, 1:nf) = th( m+1: m+nf)'; m = m+nf;
      end
   end

   if isnan(o),
      o = th( m+1); m = m + 1;
   end

   if FT>3,
      [v, w, ic, p] = peoedw( b, f, [z(:,1)-o,z(:,2)], M, FT, T);
   else
      [v, w, ic] = peoedw( b, f, [z(:,1)-o,z(:,2)], M, FT, T, p);
   end
 
end
