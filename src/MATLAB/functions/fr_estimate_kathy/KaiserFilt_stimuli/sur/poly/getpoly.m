%GETPOLY - Get polynomial parameters through a dialog box 
%
%

%  (c) - Claudio G. Rey - 12:21PM  7/29/93

%  poly parameters 	buf1
%  delay    		buf2
%  input signal # 	buf11
%  output signal #	buf12
%  
%  write par. value	str1
%  load data		str2
%  save data		str3
%  Read par. value	str4
%
   
%  Delete if dialog box already exists:

   hpoppolyin = [];

   eval(' delete( hpolynomial);','1;');

%

   Signalno = 1; k = 1;
   str2 = '[buf1,buf2,buf11,buf12]=loadpoly( polyparfilename);';
   str3 = 'savepoly( polyparfilename, buf1, buf2, buf11, buf12);';
   eval(str2);

%  Define calls

   hpoppolymicscall    = 'eval( str2), k = get( hpoppolymics, ''value''); str1 = [''numa2str(buf'' num2str(k) '')'']; str4 = [''buf'' num2str(k) ''= sscanf(get( hgetpoly,''''String''''),''''%g'''');'']; eval(dohgetpoly);';
   hpoppolyincall      = 'eval( str2), k = get(   hpoppolyin, ''value''); if k==buf11, buf11=1; elseif k~=1, buf11=k; end; Signalno = listfind(Plotlist(buf11,:),Signals); eval( str3); eval( dohgetpolyedit); eval(  dohpoppolyin);';
   hpoppolyoutcall     = 'eval( str2), k = get(  hpoppolyout, ''value''); if k==buf12, buf12=1; elseif k~=1, buf12=k; end; Signalno = listfind(Plotlist(buf12,:),Signals); eval( str3); eval( dohgetpolyedit); eval( dohpoppolyout);';
   hgetpolyeditcall    = 'str = get(hgetpolyedit,''String'');[Signaldefinitions] = listedit(Signalno,str,Signaldefinitions);plotnew;';

%  Define dynamic controls

   dohgetpolyedit      = 'eval(''delete( hgetpolyedit );'',''1;'');              hgetpolyedit = uicontrol(''Style'',''Edit'',  ''String'', deblank(Signaldefinitions(Signalno,:)),        ''Position'', [   5,   5, 590,  20], ''Callback'', hgetpolyeditcall);';
   dohgetpoly          = 'eval(''delete( hgetpoly     );'',''1;'');              hgetpoly     = uicontrol(''Style'',''Edit'',  ''String'', eval( str1),                                   ''Position'', [   5,  45, 285,  20], ''Callback'', ''eval(str4);eval(str3);'');';
   dohpoppolyin        = 'eval(''delete( hpoppolyin   );'',''1;''); eval (str2); hpoppolyin   = uicontrol(''Style'',''Popup'', ''String'', makelist(Plotlist,min([buf11,NoofDisplayed])), ''Position'', [ 350, 100,  80, 50], ''Callback'',   hpoppolyincall);';
   dohpoppolyout       = 'eval(''delete( hgetpolyout  );'',''1;''); eval (str2); hpoppolyout  = uicontrol(''Style'',''Popup'', ''String'', makelist(Plotlist,min([buf12,NoofDisplayed])), ''Position'', [ 450, 100,  80, 50], ''Callback'',  hpoppolyoutcall);';

%  Define and paint static controls

   hpolynomial    = figure(    'name',     'Polynomial Fit',                     'Position', [ 300, 300, 600, 150],'number','off','color',[.25,.75,.75],'menubar','none','nextplot','replace','resize','off');
   hpoppolymics   = uicontrol('Style',     'Popup','String', 'Parameters|Delay', 'Position', [   5, 100, 200, 50], 'Callback', hpoppolymicscall);
   hgetpolyfit    = uicontrol('Style','Pushbutton','String',         'Estimate', 'Position', [ 210, 125,  80,  25], 'Callback', 'figure(hdatadisplay);fit = polfit(   polyparfilename, data(:,[1,buf11+1,buf12+1]), Mx);');
   hgetpolyapply  = uicontrol('Style','Pushbutton','String',          'Predict', 'Position', [ 210,  85,  80,  25], 'Callback', 'figure(hdatadisplay);fit = polapply( polyparfilename, data(:,[1,buf11+1,buf12+1]), Mx);');

%  Evaluate and paint (for the first time) dynamic controls

   eval(dohpoppolyin);
   eval(dohpoppolyout);

