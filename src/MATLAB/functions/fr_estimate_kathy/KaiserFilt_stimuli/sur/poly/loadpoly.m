function [polypars, delay, nin, nout] = loadpoly( polyparfile);
%LOADPOLY load poly parameters from parameter file
%	
%	[polypars, delay, nin, nout] = loadpoly( polyparfile);
%

%	(c) Claudio G. Rey - 9:52AM  7/29/93

   if exist( [polyparfile]) == 2,
      eval( ['load ' polyparfile]);
      if exist( 'nin')  ~= 1, nin  = 1; end
      if exist( 'nout') ~= 1, nout = 2; end
   else
      disp('fit parameter file not found: loading defaults...');
      polypars = [0 1]; delay = 0; nin = 1; nout = 2; 
   end  


