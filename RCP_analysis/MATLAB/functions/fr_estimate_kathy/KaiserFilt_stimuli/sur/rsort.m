function [array] = rsort(results,trials)
%RSORT - sort fitdata results
%
%	[array] = rsort(results,trials)
%
%	trials is the number of tests starting with different poles.
%

%	(c)	Claudio G. Rey - 9:17AM  5/26/93

   array = zeros(12,2); jump = trials+1;
   array([1:3 7:9],1:2) = results([1:jump:jump*6],[5 4]);
   for k=1:3;base=1+(k-1)*jump;[x,m]=min(results(base+(1:trials),5));array(3+k,1:2)=results(base+m,[5 4]);end
   for k=1:3;base=1+(k+2)*jump;[x,m]=min(results(base+(1:trials),5));array(9+k,1:2)=results(base+m,[5 4]);end
end