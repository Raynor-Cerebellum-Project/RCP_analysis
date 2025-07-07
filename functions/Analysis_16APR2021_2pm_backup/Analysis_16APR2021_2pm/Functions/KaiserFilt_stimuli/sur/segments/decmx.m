%decmx
%
%

   nb  = 1;
   nts = length(M(:,1));
   ni  = length(ns);
   if ns( 1) - nb > 0,
      ns = ns - nb;
      replot
    else
      disp('No more segments.')
   end
