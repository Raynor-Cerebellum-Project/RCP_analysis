function savedyna( dynaparfilename, b, nk, f, o, p, FT, nin, nout, sp)
%SAVEDYNA save dynamic fit parameters to disk.
%
%

% - Claudio G. Rey - 9:27PM  12/27/93

   if nargin < 7, FT = 3; end
   if nargin < 8, nin = 1; nout = 2; end
   if nargin < 9, sp = 0; end

   eval(['save ' dynaparfilename ' b nk f o p FT nin nout sp']);

