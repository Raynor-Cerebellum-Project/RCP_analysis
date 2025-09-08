%edit_call
%
%Allows to add or remove manually a segment that was chosen
%using the stichv file.
%
%  PAS, August 1997
%
%  Inspired from
%
%gotoseg - go to a segment of interest
%

%	(c) Claudio G. Rey - 12:12PM  6/2/93

   str      = inta2str(ns);
   editcall = 'ns=sscanf(get(heditstr(2),''String''),''%d'');
   ns=nschck(M,ns'');
   replot;';

   editstr;