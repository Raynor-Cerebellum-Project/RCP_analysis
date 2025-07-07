function savethrs( threshfilename, xmin, xmax, srchout, nsig);
%SAVETHRS save thresh data to disk 
%	

% Claudio G. Rey - 10:37AM  7/29/93

   if nargin < 5, nsig = 1; end

   eval( ['save ' threshfilename ' xmin xmax srchout nsig';]);

end