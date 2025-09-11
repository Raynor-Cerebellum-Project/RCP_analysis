%SIGNEW - Display a new signal
%
%

%  (c) Claudio G. Rey - 12:54PM  7/1/93

   if NoofDisplayed >= 8,

      disp('You can display up to 8 signals.');

   else

      list = deblank(Signaldescriptions(1,:));
      for k = 2: NoofSignals,
         list = [list '|' deblank(Signaldescriptions(k,:))];
      end

      list    = [list '|Cancel'];
      popcall = 'Signalno=get(hpoplist,''value'');if Signalno>NoofSignals,delete(hpoplist);clear hpoplist;else,Plotlist(NoofDisplayed+1,:)=Signals(Signalno,:);plotnew;end';

      poplist;
   end
