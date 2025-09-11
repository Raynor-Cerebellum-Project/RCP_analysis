function [Signaldefinitions] = listedit(Signalno,Newdefinition,Signaldefinitions);
%LISTEDIT - edit an element of a simple string array.
%
%	[Signaldefinitions] = listedit( Signalno, Newdefinition, Signaldefinitions);
%

% 	(c) Claudio G. Rey - 11:11AM  7/28/93

%  Get the number of defs:

   [ndefs,cc] = size( Signaldefinitions);

%  Add the new definition at the end of the list:

   Signaldefinitions = str2mat( Signaldefinitions, Newdefinition);

%  Replace the desired definition with the new one:

   Signaldefinitions( Signalno, :) =  Signaldefinitions( ndefs+1, :);

%  Eliminate the now repeated last row:

   Signaldefinitions =  Signaldefinitions( 1:ndefs, :);

