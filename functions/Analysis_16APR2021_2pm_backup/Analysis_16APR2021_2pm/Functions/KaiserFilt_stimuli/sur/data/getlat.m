%Change the global latency variable;
%
%

% Claudio G. Rey - 10:35AM  6/30/93

   str      = int2str(lat);
   editcall = 'latnew=sscanf(get( heditstr(2),''String''),''%d'');if isempty(latnew)~=1,M=M+lat-latnew;lat=latnew;end;replot;';

   editstr;
