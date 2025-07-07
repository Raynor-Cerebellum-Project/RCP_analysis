function [v,w,ic] = dynapply( dynaparfilename, data, Ts, Mx)
%DYNAPPLY apply dynamics to input signal
%
%	dynapply( dynaparfilename, data, Ts, Mx)
%

%  (c) Claudio G. Rey - 11:16PM  12/28/93

%
%  Load fit parameters from disk:

   [b, nk, f, o, p, FT, nin, nout, sp] = loaddyna( dynaparfilename, Ts);

%
%  See if the offset is an empty variable meaning that zero offset is desired:

   if isempty( o) == 1, o = 0; end

%
%  Force p to be computed:

   p = NaN;

%
%  Compute fit:

   [b,nk,f,o,p,v,w,ic] = fitsegi( 5, data, Ts, Mx, b, nk, f, o, p, FT);

 