%SIGDEL - Remove a signal from the display
%
%

% (c) Claudio G. Rey - 5:06PM  6/30/93



   if NoofDisplayed == 1, 

      disp('You cannot remove the only signal you are displaying');
   else

      NoofDisplayed = length( Plotlist( :, 1));
      list = Plotlist(1,:);
      for k = 2:NoofDisplayed, list = [list '|' Plotlist(k,:)]; end
      list = [list '|Cancel'];

      popcall = 'k=get(hpoplist,''value'');if k==1;Plotlist=Plotlist(2:NoofDisplayed,:);elseif k==NoofDisplayed,Plotlist= Plotlist(1:NoofDisplayed-1,:);else,Plotlist=Plotlist([1:k-1,k+1:NoofDisplayed],:);end,plotnew;';

      poplist;

   end
