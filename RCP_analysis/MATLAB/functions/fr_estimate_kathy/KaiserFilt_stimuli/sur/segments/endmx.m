%endmx
%
%

   ni = length(ns);
   nslast = length(M(:,1)); 
   if ns(length(ns)) ~= nslast,
      ns = ns + nslast - ns(length(ns));replot;
   end