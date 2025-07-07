function ns = nschck(M,ns)
%
%

%	(c) Claudio G. Rey - 11:54AM  6/2/93

   nts = length(M(:,1));
   ni = length(ns); 
   ns = sort(ns); 
   ns = max(ones(1,ni),min(nts*ones(1,ni),ns));

