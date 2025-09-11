function [b, nk, f, o, p, FT, nin, nout, sp] = loaddyna( dynaparfilename, Ts);
%LOADDYNA load fit data from disk
%	
%	If data file does not exist load default values:
%		b=[1 -1]/Ts;f=1;o=0;nk=0;
%

% Claudio G. Rey - 7:48PM  1/8/94

   if exist( dynaparfilename) == 2,

      eval( ['load ' dynaparfilename;]);
      if exist( 'nin')  ~= 1, nin  = 1; end
      if exist( 'nout') ~= 1, nout = 2; end
      if exist( 'FT')   ~= 1, FT = 3; end
      if exist( 'sp')   ~= 1, sp = 0; end

   else

      disp('fit parameter file not found: loading defaults...');
      b=[1 -1]/Ts; f=1; o=0; nk=0; p=NaN; nin=1; nout=2; FT = 3; sp = 0;

   end  


