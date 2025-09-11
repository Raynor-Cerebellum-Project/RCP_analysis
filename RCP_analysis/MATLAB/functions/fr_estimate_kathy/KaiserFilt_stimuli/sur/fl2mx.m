function M=fl2mx(flag,tr)
%FL2MX - Convert a flag into a marker array.   
%
%	M = fl2mx(flag,trigger)
%
%	trigger is the desired value of the flag 
%		default value is 0
%
%	M is a two column marker array: the columns mark 
%		the first and last element of segments when flag=trigger
%

%   Claudio G. Rey  1992/02/15
   
  if nargin<2, tr=0; end;

  n  = length(flag); i1 = 1:n-1; i2 = 2:n;
  Kf = zeros(n,1); Kl = zeros(n,1);
  if flag(1) == tr, Kf(1) = 1;end
  if flag(n) == tr, Kl(n) = 1;end
   
% Mark the beginning and end of each segment
  
  Kf(i2) =  1 - abs(sign(1-abs( -sign(flag(i1)-tr)+3*sign(flag(i2)-tr))));
  Kl(i1) =  1 - abs(sign(1-abs(3*sign(flag(i1)-tr)  -sign(flag(i2)-tr))));
  Kf     = sort(Kf.*(1:n)');
  Kl     = sort(Kl.*(1:n)');
  [z,i]  = min(Kf - (1:n)'/n);
  M      = [Kf(i+1:n),Kl(i+1:n)];

