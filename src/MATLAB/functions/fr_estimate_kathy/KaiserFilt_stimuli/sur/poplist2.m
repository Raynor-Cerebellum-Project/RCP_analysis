
   if exist( 'hpoplist2') == 1; delete(hpoplist2); clear hpoplist2; end, 
   popstr2 = [popstr2 'if exist( ''hpoplist2'') == 1; delete(hpoplist2); clear hpoplist2 popcall2 list2; end'];
   hpoplist2 = uicontrol('Style','Popup','String',list2,'Position',[205,380,200,350],'Callback', popcall2);
   