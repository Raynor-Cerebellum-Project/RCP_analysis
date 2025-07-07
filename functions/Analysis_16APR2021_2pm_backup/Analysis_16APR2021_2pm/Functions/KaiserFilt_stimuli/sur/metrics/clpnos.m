%

%	(c) Claudio G. Rey - 2:44PM  6/2/93

   Info = [];
   for k = 1:length(ns),
       Infoline = [nos(index,M(ns(k),:),10)];
       Info = [Info;[Infoline]];
   end
   clip(Info)
   clear Info Infoline k
