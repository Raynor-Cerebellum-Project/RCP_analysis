function [y,ic] = filterdw(b,lb,a,la,u,M,FT,T,p);
%filterdw - delta filter 
%
%	y = filterdw(b,lb,a,la,u,M,FT,T,p);
%
%	Model
%	y = (b/a)*u(t) + p	->	'a' is monic, that is:
%
%	y( t) = b(1)*u(t) + b(2)*(u(t)-u(t-1))/T + ... - a(2)*y(t-1) - ...
%
%	b	numerator in delta coefficients
%	a	numerator in z domain
%	la & lb indicate how many samples to go back to start the ics and
%		to start the exogenous input
%	M	marker for the domain
%	T 	sampling interval
%	FT=2,4 -> Response assumed continuous except for initial conditions.
%			In this case the output is defined for all times
%	FT=1,3 -> Response completely discontinuous.
%			In this case the output is defined only for the
%			given domain
%	p	initial conditions
%

% (c) Claudio G. Rey - 1:34PM  12/27/93

%
%  Compute basic matrix metrics

   hgetdyna = [];

   [ns,cc] = size(M);		%ns=number of segments
   nb = length( b)-1;		%FIR memory
   na = length( a)-1;		%IIR memory

   la = max([na,la]);
   lb = max([nb,lb]);

%
%  Compute initial conditions:

   ic = [];
   if na>0, 
      if nargin == 9, 
         ic = iw( p, a, la, M);  
     end, 
   end

%
%  Initialize the output and temporary storage:

   y    = zeros(size(u)); 
   temp = zeros(size(u));

%
%  Define domain with padding in the input
%  	(padding is necessary to fill up the FIR memory.)

   if ((FT==1) | (FT==3)),
      Mp  = [M(:,1)-lb,M(:,2)];
      ixp = mx2ix( Mp);
      ix  = mx2ix(M);
   else
      Mp  = [M(1,1)-lb,M(ns,2)];
      ixp = mx2ix( Mp);
      ix  = M(1,1):M(ns,2);
      ns  = 1;
   end

%
%  temp stores derivatives temporarily 

   temp(ixp) = u(ixp);

%
%  Compute the rest of the FIR part of the result

   for ii=1:nb+1,
      y(ix) = y(ix) + b(ii)*temp(ix);
      if ii<=nb
         temp(ixp) = diff([0;temp(ixp)])/T;
      end
   end

%
%  The IIR (auto-regressive) part of the argument is computed
%  by the intrinsic function 'filter'
  
   if length(a)>1, 
      for kk=1:ns,
         y(Mp(kk,1):Mp(kk,2)) = filter(1,a,y(Mp(kk,1):Mp(kk,2)));
      end
   end

%disp('At filterdw:'),keyboard

if (isempty(ic) ~= 1),
   y(1:length(ic)) = y(1:length(ic)) + ic;
end

%
