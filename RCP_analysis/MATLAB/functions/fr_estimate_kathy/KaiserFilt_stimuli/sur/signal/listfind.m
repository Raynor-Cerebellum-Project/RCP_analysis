function [Signalno] = listfind( Name, Signals);
%LISTFIND - find an element of a simple string array.
%
%	[Signalno] = listfind( Name, Signals);
%

% 	(c) Claudio G. Rey - 1:03PM  9/24/93

%  Get the number of signals:

   [nsigs,cc] = size( Signals);

   for k = 1: nsigs,

      if strcmp( deblank( Name), deblank( Signals( k, :))) == 1, Signalno = k; end 

   end

