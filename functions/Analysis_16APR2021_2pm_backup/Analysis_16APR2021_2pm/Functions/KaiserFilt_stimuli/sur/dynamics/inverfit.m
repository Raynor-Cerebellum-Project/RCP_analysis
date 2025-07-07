function inverfit( dynaparfilename);
%INVERFIT Invert system model
%
%

% (c) Claudio G. Rey - 9:33AM  7/29/93

   if exist( dynaparfilename)==2,
      [b, nk, f, o, p, nin, nout] = loaddyna( dynaparfilename);
      p = NaN;
      nk = -nk; o = 0; temp = b; b = f/temp( 1);f = temp/temp(1);
   else
      error('No parameter file');
   end

   savedyna( dynaparfilename, b, nk, f, o, p, nout, nin);


