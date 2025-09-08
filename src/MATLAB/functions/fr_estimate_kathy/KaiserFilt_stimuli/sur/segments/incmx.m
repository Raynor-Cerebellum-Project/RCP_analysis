%incmx
%
%
   nb  = 1;
   nts = length(M(:,1));
   ni  = length(ns);
   if ns( ni) +nb < nts+1,
            ns = ns + nb; replot;
   else
            disp('No more segments.')
   end
