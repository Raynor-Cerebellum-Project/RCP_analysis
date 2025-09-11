%begmx
%
%

   nts = length(M(:,1));
   ni = length(ns);
   if ns(1) ~=1,
      ns = ns + 1 - ns(1); replot;
   end
