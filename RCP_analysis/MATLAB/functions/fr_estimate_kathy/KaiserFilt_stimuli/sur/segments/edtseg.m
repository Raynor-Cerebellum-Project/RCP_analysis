%edtseg - edit visually the chosen segment(s) of interest
%

%	(c) Claudio G. Rey - 4:18PM  8/6/93

   if timebase=='xvsy',timebase='stac';replot;end

   Msave = M;    NoofSegments = length( ns);
  
   buf001 = ginput(NoofSegments*2);

   buf001 = sort([round(buf001(1:2:(NoofSegments*2))/Ts)',round(buf001(2:2:(NoofSegments*2))/Ts)']);

   [Md, Mx] = segments(M, ns, timebase, buffer, pan, N);

   M( ns, 1:2) = M( ns, 1:2) + buf001 - Mx;

   replot

