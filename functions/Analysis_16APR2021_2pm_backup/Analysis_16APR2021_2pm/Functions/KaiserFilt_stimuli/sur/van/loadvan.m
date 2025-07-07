function [b, nk, o, q, nin, nout] = loadvan( vanparfilename);
%LOADVAN load fit data from disk
%	
%	If data file does not exist load default values:
%		b=[1 -1]/Ts;o=0;nk=0;
%

% Claudio G. Rey - 8:09AM  8/23/93

   if exist( vanparfilename) == 2,
      eval( ['load ' vanparfilename;]);
      if exist( 'nin')  ~= 1, nin  = 1; end
      if exist( 'nout') ~= 1, nout = 2; end
   else
      disp('fit parameter file not found: loading defaults...');
      b=[0 1]; o=0; nk=0; q=0; nin=1; nout=2;
   end  

