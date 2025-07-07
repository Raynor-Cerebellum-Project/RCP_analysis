%gotoseg - go to a segment of interest
%

%	(c) Claudio G. Rey - 8:41PM  1/3/94

   str      = numa2str([(M(ns,1)+M(ns,2))/2*Ts]);
   editcall = 'X=sscanf(get( heditstr(2),''String''),''%g'');ns=timechck(X,M,ns,Ts);replot;';

   editstr;