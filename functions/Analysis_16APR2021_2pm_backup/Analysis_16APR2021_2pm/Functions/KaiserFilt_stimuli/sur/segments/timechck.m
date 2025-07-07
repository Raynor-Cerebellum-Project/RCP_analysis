function ns = timechck( X, M, ns, Ts)
%timechk - 
%

%	(c) Claudio G. Rey - 7:41AM  6/2/93

   jsave = [];times = [(M(ns,1)+M(ns,2))/2*Ts];

   if isempty(X) == 0, 
      times = round(sort(X)/Ts);
   end

   for j = 1:length(M(:,1))
      for k = 1:length(times)
         if (M(j,1)-times(k)-10)*(M(j,2)-times(k)+10)<0, 
            jsave = [jsave j]; 
         end
      end
   end

   if isempty(jsave) == 1, 
      change = 'n';  disp('Segment(s) not found.')
   else
      change = 'y';  
      ns = jsave; 
      disp(['Segment(s) found: ' inta2str(ns)])
   end

