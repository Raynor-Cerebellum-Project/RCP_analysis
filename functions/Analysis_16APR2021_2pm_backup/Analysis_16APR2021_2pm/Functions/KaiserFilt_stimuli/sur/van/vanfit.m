function [v,w] = vanfit( vanparfilename, data, Ts, M, option);
%VANFIT - Fit second signal in data from dynamics derived from the first signal.
%
%

%  (c) - Claudio G. Rey - 8:41PM  1/11/94

  if nargin<5, option = 'v'; end

%  Load fit parameters from disk:

   [b, nk, o, q, nin, nout] = loadvan( vanparfilename);

%  See if the offset is an empty variable meaning that no offset is desired:

   if isempty( o) == 1, choice =1, else, choice = 2; end

%  Do fit

   ZZ = -roots( b); nb = length( b);
   GG = b( 1); nback = nb-1; f = 1; np=0;
   ns     = length( M(  :, 1));
   Ncap   =         M( ns,  2);
   N      =         4;

   if nk <= 0,
      Z = [data(1:Ncap,3),data((1-nk):(Ncap-nk),2)];
   else
      Z = [data(1:Ncap,3),[zeros(nk,1);data(1:(Ncap-nk),2)]];
   end

   Mx = [ (M( :, 1) + nback), M( :, 2)]; ix = mx2ix( Mx);

   if nback > 1,
      izero = mx2ix([M( :, 1), M( :, 1) + nback-1]);
   end

   [b,f,o,q,MSE,BIC,P]=arxvw( Z, ix, [np nb 0], length(q)+1, Ts);
  
   nonlinearity = o*ones(size(Z(:,1)));
   for kq = 1: length( q) 
      nonlinearity = nonlinearity + q(kq)*Z(:,2).^(kq+1);
   end

   [v,w] = peoedw( b, 1, [(Z(:,1)-nonlinearity) Z(:,2)], Mx, 1, Ts);

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


   f = 1; 

   if option( 1) ~= 'b'

      parameter_std = sqrt( diag( P)');
      parameter_std(1:np+nb) = parameter_std(1:np+nb);

      disp('Equation:')

      m = 0; eq = [];

      eq = [eq '[' num2str(b(1)) ' ~ ' num2str( parameter_std(m+1)) ']*u(t)'];
      m = m + 1;            

      for ii = 2:nb
         eq = [eq ' + [' num2str(b(ii)) ' ~ ' num2str( parameter_std(m+1)) ']'];
         eq = [eq '*d^' num2str(ii-1) '*u(t)'];
         m = m + 1;
      end,

      eq = [eq ' + [' num2str( o) ' ~ ' num2str( parameter_std(m+1)) ']'];
      m = m + 1;

      for kq = 1: length( q) 
         eq = [eq ' + [' num2str(q(kq)) ' ~ ' num2str( parameter_std(kq+m)) ']'];
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

%  Save fit parameters to disk:

   savevan( vanparfilename, b, nk, o, q, nin, nout);

