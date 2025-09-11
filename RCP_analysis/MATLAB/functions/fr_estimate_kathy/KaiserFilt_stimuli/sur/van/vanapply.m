function [v,w] = vanapply( vanparfilename, data, Ts, M)
%VANAPPLY apply dynamics to input signal
%
%	dynapply( vanparfilename, data, Ts, Mx)
%

%  (c) Claudio G. Rey - 8:25PM  1/11/94

   [b, nk, o, q, nin, nout] = loadvan( vanparfilename);

%  See if the offset is an empty variable meaning that zero offset is desired:

   if isempty( o) == 1, o = 0; end

   ns     = length( M(  :, 1));
   Ncap   =         M( ns,  2);
   N      =         4;
   nb     = length(b);
   nback = nb-1;

%  Compute prediction:

   if nk <= 0,
      Z = [data(1:Ncap,3),data((1-nk):(Ncap-nk),2)];
   else
      Z = [data(1:Ncap,3),[zeros(nk,1);data(1:(Ncap-nk),2)]];
   end

   Mx = [ (M( :, 1) + nback), M( :, 2)]; ix = mx2ix( Mx);

   if nback > 1,
      izero = mx2ix([M( :, 1), M( :, 1) + nback-1]);
   end

   nonlinearity = o*ones(size(Z(:,1)));
   for kq = 1: length( q) 
      nonlinearity = nonlinearity + q(kq)*Z(:,2).^(kq+1);
   end

   f=1;
   [v,w] = peoedw( b, f, [(Z(:,1)-nonlinearity) Z(:,2)], Mx, 1, Ts);

   rms = sqrt( v( ix)' * v( ix) / (length( ix))); err = mean( v( ix));  dev = std( v( ix));
   for k = 1:N
      Mxx = round(Mx*[(N-k+1) (N-k);(k-1) k]/N);
      for j = 1:ns, segerr(k,j) = mean( v( Mxx( j,1): Mxx( j,2) ) ); end
      segstd( k) = std( segerr( k, :)); segmean( k) = mean(segerr( k, :));
   end
   overallsegmean = mean( segmean);overallsegstd = std( segmean);

   if option(1) == 'v'

      if nback>1,
         w( izero) = zeros(length(izero),1);
      end   
      hold, plot( ((1:ix(length(ix)))' * Ts), (w( 1:ix(length(ix))) + nonlinearity),'w'),hold
   end


   if option( 1) ~= 'b'

      disp('Equation:')

      m = 0; eq = [];

      eq = [eq '[' num2str(b(1)) '*u(t)]'];
      
      for ii = 2:nb
         eq = [eq ' + [' num2str(b(ii)) ']'];
         eq = [eq '*d^' num2str(ii-1) '*u(t)'];
      end, m = m + nb;

      eq = [eq ' + [' num2str( o) ']'];

      for kq = 1: length( q) 
         eq = [eq ' + [' num2str(q(kq)) ']'];
         eq = [eq '*u(t)^' num2str(kq+1)];
      end

      disp(eq)
      ZZ = -roots( b); nb = length( b); GG = b( 1);
      disp(['Gain: ' num2str( GG) ' - Num TC(s): ' numa2str( ZZ) ' - Delay: ' num2str( nk*Ts)]) 
      disp(['Error Mean, std. dev., rms & BIC:  ',  numa2str(  [err dev rms BIC])]) 

      disp(' ')
      disp(['Seg mean error: ', numa2str(  segmean) ' -  Seg std dev: ' numa2str(  segstd)]) 
      disp(['Overall seg mean error and std dev:  ', numa2str(  [overallsegmean overallsegstd])]) 
   end

