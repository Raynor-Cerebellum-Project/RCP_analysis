function [e,w,ic,p] = peoew(b,lb,f,lf,z,M,FT,p)
%peoew	Computes prediction errors for an output error model structure.
%
%	[e,w,ic,p] = peoew(b,lb,f,lf,z,M,FT,p)
%
%	P     : Array of initial conditions
%	E     : The prediction errors
%	B & F : model relating u and y.
%	Z     : The output-input data Z=[y u]
%	INDEX : Domain index for Z.
%	FT=2,4 -> continuous filter forcing function. 2=default
%	FT=1,3 -> reinitialized filter forcing function.
%	FT=1,2 -> initial conditions fitted to data.
%	FT=3,4 -> initial conditions derived from initial output.
%
%	A more complete set of information associated with the model TH and
%	the data Z is obtained by invoking help auxvar.
%
%	Here A,B,C,D,and F are the polynomials associated with TH and
%	W = B/F u + P/F I and V = A[y - W - P/F I].

%   (c) Claudio G. Rey, 1:27PM  7/6/93

% ic  = p/f I, Initial conditions.
% w   = (b/f)*u
% v   = y-w-ic

  nf = length(f)-1;

% Filter the data

  w = pefiltw( b, lb, f, lb, z(:,2), M, FT); 
  v = z(1:length(w),1) - w;

%  Compute initial conditions and initial condition trajectories


   if nf > 0,
      if nargin<8, 

%        p unknown:

         [ic, p] = icw( f, lf, v, M, FT);

      else

         if isempty( p) == 1,

%           p unknown:

            [ic, p] = icw( f, lf, v, M, FT);

         elseif isnan(p(1,1)) == 1,

%           p unknown:

            [ic, p] = icw( f, lf, v, M, FT);

         else

%           p known:

            ic = iw( p, f, lf, M);

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

