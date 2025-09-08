%
%
%

   list = deblank(Signaldescriptions(1,:));
   for k = 2: NoofSignals,
   list = [list '|' deblank(Signaldescriptions(k,:))];
   end

   list    = [list '|Cancel'];

   popcall = 'Signalno = get(hpoplist,''value''); getedit;';

   poplist;
