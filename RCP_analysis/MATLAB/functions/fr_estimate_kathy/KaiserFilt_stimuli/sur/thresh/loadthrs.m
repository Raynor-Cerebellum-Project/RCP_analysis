function [xmin,xmax,srchout,nsig] = loadthrs( threshfilename);
%LOADTHRS load thresh data from disk
%	
%	If data file does not exist load default values:
%       xmin= -9999; xmax = 9999; srchout = 0; nsig = 1;
%

% Claudio G. Rey - 9:06AM  9/24/93

   if exist( threshfilename) == 2,
      eval( ['load ' threshfilename;]);
      if exist( 'nsig') ~= 1, nsig = 1; end
   else
      xmin= -9999; xmax = 9999; srchout = 0; nsig = 1;
   end  

end