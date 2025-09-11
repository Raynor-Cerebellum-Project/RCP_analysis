function [b, nk, bias, FT, nin, nout, sp] = loadbias( mbiasparfilename, Ts);
%LOADBIAS load fit data from disk
%	
%	If data file does not exist load default values:
%		b=[1 1];nk=0;sp=0;
%

% Claudio G. Rey - 11:55AM  1/22/94

   if exist( mbiasparfilename) == 2,

      eval( ['load ' mbiasparfilename;]);

   else

      disp('fit parameter file not found: loading defaults...');
      b=[1 1]; nk=0; bias=NaN; nin=1; nout=2; FT = 3; sp = 0;

   end  


