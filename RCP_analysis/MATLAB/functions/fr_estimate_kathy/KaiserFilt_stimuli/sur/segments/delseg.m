%DELSEG - delete selected segments
%
%

%	(c) Claudio G. Rey - 4:33PM  6/28/93

   Msave = M;
   NoofSegments = length(M(:,1));

   M = M( fl2ix( 1 - ix2fl( ns, NoofSegments) ),:); ns = nschck( M, ns); replot;
