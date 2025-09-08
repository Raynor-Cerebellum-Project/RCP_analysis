function savevan( vanparfilename, b, nk, o, q, nin, nout)
%SAVEVAN save dynamic fit parameters to disk.
%
%

% - Claudio G. Rey - 8:08AM  8/23/93

   if nargin < 7, nin = 1; nout = 2; end

   eval(['save ' vanparfilename ' b nk o q nin nout']);

