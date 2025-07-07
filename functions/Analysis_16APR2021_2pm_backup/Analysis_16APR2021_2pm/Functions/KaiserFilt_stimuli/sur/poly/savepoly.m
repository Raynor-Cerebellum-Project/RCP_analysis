function savepoly( polyparfilename, polypars, delay, nin, nout)
%SAVEPOLY save poly data to diskfile
%
%	savepoly( polyparfilename, polynomialparamters)
%

% - Claudio G. Rey - 10:05AM  7/29/93

   if nargin < 4, nin = 1; nout = 2; end

   eval(['save ' polyparfilename ' polypars delay nin nout']);

