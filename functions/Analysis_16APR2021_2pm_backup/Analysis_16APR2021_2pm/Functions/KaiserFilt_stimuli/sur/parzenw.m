function [num,den] = parzenw(sigma,Ts,range)
%PARZENW
%
%	[num,den] = parzenw(sigma,Ts,range)
%
%
%	Formula:
%
%	num = Ts*exp(-((range(1)*sigma):Ts:(range(2)*sigma))).^2/2/sigma^2);
%	den = zeros(num); den(1) = 1;
%

 num = exp(-( (range(1)*sigma):Ts:(range(2)*sigma) ).^2/2/sigma^2);

 den = 1;

