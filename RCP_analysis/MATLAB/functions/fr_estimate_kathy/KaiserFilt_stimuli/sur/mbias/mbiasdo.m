function [v,w] = mbiasdo( mbiasparfilename, data, Ts, M, type, option);
%MBIASDO - Fit second signal in data from dynamics derived from the first signal.
%
%

%  (c) - Claudio G. Rey - 9:09PM  1/25/94

   if nargin<7, option = 'v'; end

%
%  Only type allowed

   type = 'fit';

%  Load fit parameters from disk:

   [b, nk, bias, FT, nin, nout, sp] = loadbias( mbiasparfilename, Ts);

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

   [b,f,bias,LAM,BIC,P,v,w1,w2] = arxodw( Z, ix, [0,nb,0], Ts, sp);

   w = w1+w2;

   rms = sqrt( LAM); err = mean( v);  dev = std( v);

   if option(1) == 'v'
      hold on, plot( ix*Ts, w, 'w'), hold off
   end

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

      disp(eq)

      ZZ = -roots( b); nb = length( b); GG = b( 1);

      disp(['Gain: ' num2str( GG) ' - Num TC(s): ' numa2str( ZZ) ' - Delay: ' num2str( nk*Ts)]) 
      disp(['Error Mean, std. dev., rms & BIC:  ',  numa2str(  [err dev rms BIC])]) 

   end

%  Save fit parameters to disk:

   savebias( mbiasparfilename, b, nk, bias, FT, nin, nout, sp)

