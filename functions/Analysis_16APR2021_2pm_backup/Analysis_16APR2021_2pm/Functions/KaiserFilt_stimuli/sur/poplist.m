
   if exist( 'hpoplist') == 1; delete(hpoplist); clear hpoplist; end, 
   popstr = [popstr 'if exist( ''hpoplist'') == 1; delete(hpoplist); clear hpoplist popcall list; end'];
   hpoplist = uicontrol('Style','Popup','String',list,'Position',[5,380,200,350],'Callback', popcall);
   