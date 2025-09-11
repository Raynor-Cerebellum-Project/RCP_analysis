function [fit] = polfit( polyparfilename, data, Mx)
%POLFIT apply static polynomial coefficients to input signal
%
%	polapply( polyparfilename, data, Mx)
%

%  (c) Claudio G. Rey - 9:50AM  7/26/93

   [polypars,delay] = loadpoly( polyparfilename);

   ix = mx2ix( Mx); Ncap = length( ix); N = length( data); n = length( polypars)-1; 
   if delay ~= 0; pad = zeros(abs(delay),1); else, pad = []; end

   if delay <= 0, IN = [data(1-delay:N,2);pad]; else, IN = [pad;data(1:N-delay,2)]; end

   polypars = polyfit( IN( ix), data(ix,3), n);

   fit = polyval( polypars, IN);

   err = fit( ix) - data(ix,3);

   Vcap = (err'*err/length(ix));

   polstr = []; for i=1:length( polypars), polstr = [polstr ' ' num2str(polypars( i))]; end  

   FPE  = Vcap*(1+n/Ncap)/(1-n/Ncap);

   disp(['RMS ' num2str(sqrt( Vcap)) ' - FPE: ' num2str(FPE) ' - polynomial coeff. ' polstr ' - Delay: ' int2str( delay)])

   hold on; plot(data(:,1),fit,'w'); hold off;

   savepoly( polyparfilename, polypars, delay);

