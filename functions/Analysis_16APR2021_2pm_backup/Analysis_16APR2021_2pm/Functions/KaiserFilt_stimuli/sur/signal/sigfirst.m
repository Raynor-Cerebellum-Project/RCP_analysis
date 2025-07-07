%
%
%

% (c) - Claudio G. Rey - 12:55PM  7/1/93

   if NoofDisplayed ~= 1,

      list = Plotlist(1,:);
      for k = 2: NoofDisplayed,
         list = [list '|' Plotlist(k,:)];
      end

      popcall = 'buf001=get(hpoplist,''value'');buf002=Plotlist(1,:);Plotlist(1,:)=Plotlist(buf001,:);Plotlist(buf001,:)=buf002;plotnew;';

      poplist;

   end