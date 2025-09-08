%GETVAN - Get dynamic parameters through a dialog box 
%
%

%  (c) - Claudio G. Rey - 8:54PM  1/11/94

%  gain 		GG = buf1
%  zeros       		ZZ = buf2
%  bias        		o  = buf3
%  delay       		nk = buf4
%  Other terms		q  = buf5
%  numerator   		b  = buf7
%  input signal #            buf11
%  output signal #           buf12


%  Delete if already exists:

  hpopvanain = [];


  eval('delete( hvanamics);','1;');
   
%  Initialize:

   Signalno = 1; k = 1;

   str5 = 'buf7=real(poly(-buf2))*buf1;';
   str6 = 'buf2=-roots(buf7);buf1=buf7(1);';

   str2 = ['[buf7,buf4,buf3,buf5,buf11,buf12]=loadvan(vanparfilename);', str6];

   str3 = [str5, 'savevan(vanparfilename,buf7,buf4,buf3,buf5,buf11,buf12);'];

   eval(str2);

%  Define calls

   hpopvanamicscall    = 'eval( str2), k = get( hpopvanamics, ''value''); str1 = [''numa2str(buf'' num2str(k) '')'']; str4 = [''buf'' num2str(k) ''= sscanf(get( hgetvana,''''String''''),''''%g'''')'''';'']; eval(dohgetvana);';
   hpopvanaincall      = 'eval( str2), k = get(   hpopvanain, ''value''); if k==buf11, buf11=1; elseif k~=1, buf11=k; end; Signalno = listfind(Plotlist(buf11,:),Signals); eval( str3); eval( dohgetvanaedit);eval(  dohpopvanain);';
   hpopvanaoutcall     = 'eval( str2), k = get(  hpopvanaout, ''value''); if k==buf12, buf12=1; elseif k~=1, buf12=k; end; Signalno = listfind(Plotlist(buf12,:),Signals); eval( str3); eval( dohgetvanaedit);eval( dohpopvanaout);';
   hgetvanaeditcall    = 'str = get(hgetvanaedit,''String'');[Signaldefinitions] = listedit(Signalno,str,Signaldefinitions);plotnew;';

%  Define vanamic controls

   dohgetvanaedit      = 'eval(''delete( hgetvanaedit );'',''1;'');              hgetvanaedit = uicontrol(''Style'',''Edit'',  ''String'', deblank(Signaldefinitions(Signalno,:)),        ''Position'', [   5,   5, 590,  20], ''Callback'', hgetvanaeditcall);';
   dohgetvana          = 'eval(''delete( hgetvana     );'',''1;'');              hgetvana     = uicontrol(''Style'',''Edit'',  ''String'', eval( str1),                                   ''Position'', [   5,  45, 200,  20], ''Callback'',  ''eval( str4); eval( str5); eval( str3)'');';
   dohpopvanain        = 'eval(''delete( hpopvanain   );'',''1;''); eval (str2); hpopvanain   = uicontrol(''Style'',''Popup'', ''String'', makelist(Plotlist,min([buf11,NoofDisplayed])), ''Position'', [ 350, 100,  80, 50], ''Callback'',   hpopvanaincall);';
   dohpopvanaout       = 'eval(''delete( hgetvanaout  );'',''1;''); eval (str2); hpopvanaout  = uicontrol(''Style'',''Popup'', ''String'', makelist(Plotlist,min([buf12,NoofDisplayed])), ''Position'', [ 450, 100,  80, 50], ''Callback'',  hpopvanaoutcall);';

%  Define and paint static controls

   hvanamics         = figure('name','Vanamic Fit',                                             'Position', [ 300, 300, 600, 150],'number','off','color',[.1,.50,1],'menubar','none','nextplot','new','resize','off');
   hpopvanamics      = uicontrol('Style',     'Popup','String', 'Gain|Num TC(s)|Bias|Delay|Poly', 'Position', [   5, 100, 200, 50], 'Callback', hpopvanamicscall);
   hgetvanvanafit    = uicontrol('Style','Pushbutton','String',                     'Estimate', 'Position', [ 210, 125,  80,  25], 'Callback', 'figure(hdatadisplay);[v,w] = vanfit(    vanparfilename, data(:,[1,buf11+1,buf12+1]), Ts, Mx);');
   hgetvanvanaapply  = uicontrol('Style','Pushbutton','String',                      'Predict', 'Position', [ 210,  85,  80,  25], 'Callback', 'figure(hdatadisplay);[v,w] = vanapply(  vanparfilename, data(:,[1,buf11+1,buf12+1]), Ts, Mx);');

%  Evaluate and paint (for the first time) vanamic controls

   eval(dohpopvanain);
   eval(dohpopvanaout);

