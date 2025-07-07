function savebias( mbiasparfilename, b, nk, bias, FT, nin, nout, sp)
%SAVEDYNA save dynamic fit parameters to disk.
%
%

% - Claudio G. Rey - 11:38AM  1/22/94

   eval(['save ' mbiasparfilename ' b nk bias FT nin nout sp']);

