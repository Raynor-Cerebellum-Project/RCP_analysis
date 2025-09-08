%addseg - add a new segment of interest
%
%

%	(c) Claudio G. Rey - 11:39AM  6/28/93

   if timebase ~= 'tim-', tempbase = timebase; timebase = 'tim-'; replot; timebase = tempbase;end

   Msave = M;

   X = ginput( 2); X = round( sort( X) / Ts);

   M  = sort([X;M]);
