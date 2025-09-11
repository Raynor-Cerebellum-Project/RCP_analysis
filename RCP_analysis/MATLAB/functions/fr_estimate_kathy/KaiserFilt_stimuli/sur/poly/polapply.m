function [fit] = polapply( polyparfilename, data, Mx)
%POLAPPLY apply dynamics to input signal
%
%	polapply( polyparfilename, data, Mx)
%

%  (c) Claudio G. Rey - 9:49AM  7/23/93

   [polypars,delay] = loadpoly( polyparfilename);

   ix = mx2ix( Mx); Ncap = length( data);

   if delay <= 0,
      IN = [data((1-delay):(Ncap-delay),2);zeros(delay,1)];
   else
      IN = [zeros(delay,1);data(1:(Ncap-delay),2)];
   end

   fit = polyval( polypars, IN);

   err = fit(ix)-data(ix,3);

   RMS = sqrt(err'*err/length(ix));

   polstr = []; for i=1:length( polypars), polstr = [polstr ' ' num2str(polypars( i))]; end  

   disp(['RMS ' num2str(RMS) ';  polynomial coeff. ' polstr])

   hold on; plot(data(:,1),fit,'w'); hold off;

end 