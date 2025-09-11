function [e,w,ic,p] = peoedw(b,f,z,M,FT,T,p)
%peoedw	Computes prediction errors for an output error model structure.
%
%	[e,w,ic,p] = peoedw(b,f,z,M,FT,T,p)
%
%	FT=1,2 -> initial conditions fitted to data.
%	FT=3,4 -> initial conditions derived from initial output.
%	T	: Sampling interval
%	P     	: Array of initial conditions
%	E     	: The prediction errors
%	B & F 	: model relating u and y.
%	Z     	: The output-input data Z=[y u]
%	INDEX 	: Domain index for Z.
%
%	The formula applied is:
%
%	W = B/F u + P/F I and V = A[y - W - P/F I].

%   (c) Claudio G. Rey, 9:16AM  12/5/93

% ic  = p/f I, Initial conditions.
% w   = (b/f)*u
% v   = y-w-ic

  nf = length(f)-1; nb = length(b);

% Filter the data

   w = filterdw(  b, nb-1, f, nf, z(:,2), M, FT, T);

   v = z(1:length(w),1) - w;

%  Compute initial conditions and initial condition trajectories

   if nf > 0,
      if nargin<7, 
%        p unknown:
         [ic, p] = icw( f, nf, v, M, FT);
      else
         if isempty( p) == 1,
%           p unknown:
            [ic, p] = icw( f, nf, v, M, FT);
         elseif isnan(p(1,1)) == 1,
%           p unknown:
            [ic, p] = icw( f, nf, v, M, FT);
         else
%           p known:
            ic = iw( p, f, nf, M);
         end
      end  
   else
      ic = zeros(size(w));
      p = [];
   end

%disp('peoew: '), keyboard

   e = zeros(size(v)); 
   for k = 1:length(M(:,1)),
      e(M(k,1)-nf:M(k,2)) =  v(M(k,1)-nf:M(k,2)) - ic(M(k,1)-nf:M(k,2));
   end

