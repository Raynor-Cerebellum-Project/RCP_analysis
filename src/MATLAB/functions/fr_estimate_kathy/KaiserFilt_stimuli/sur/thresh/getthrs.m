%GETTHRS - Get threshold parameters through a dialog box 
%
%

%  (c) - Claudio G. Rey - 3:25PM  9/24/93

%   
%  str1 - string containing the value of the popup choice (min, max, or searchout)
%  str2 - string containing command to load choices.
%  str3 - string containing command to save choices.
%  str4 - string containing value (either buf1, buf2 or buf3) to be displayed 
%  str5 - string containing command for threshold computation.
%  str6 - edited value of a signal
%  buf1 - minimum value
%  buf2 - maximum value
%  buf3 - searchout in samples
%  buf4 - Number of signal to be used for thresholding.
%

%  (1) Delete if dialog box already exists:

   eval(' delete( hthreshold);', '1;');

%  (2) Initialize:

   buf4 = 1; k = 1;
   str2 = '[buf1, buf2, buf3, buf4]=loadthrs( threshparfilename);';
   str3 = 'savethrs( threshparfilename, buf1, buf2, buf3, buf4);';
   eval(str2);
   str5 = ['M = xthresh(' deblank( Signaldefinitions( buf4, :)) ', M, ns, buf1, buf2, buf3);'];

%  (3) Define static calls

%  Respond to popup
%	Load saved choices
%	Read popup index value into integer 'k'
%	Read value of choice into string 'str1'
%	Setup 'dohgetthrs' value edit popup

   hpopthrsmicscall = 'eval( str2), k = get( hpopthrsmics, ''value''); str1 = [''numa2str(buf'' num2str(k) '')'']; str4 = [''buf'' num2str(k) ''= sscanf(get( hgetthrs,''''String''''),''''%g'''');'']; eval(dohgetthrs);';

%  Respond to signal popup control choice 
%	Load saved choices
%	Read signal index value into integer k
%	Read value of choice into buf4
%	Save choices
%	Setup 'dohgetthrsedit' signal edit control
%	Setup 'dohpopthrssig'  signal popup control

   hpopthrssigcall  = 'eval( str2), k = get(  hpopthrssig, ''value''); buf4 = unswap( k, buf4); eval( str3); eval( dohgetthrsedit); eval( dohpopthrssig);';

%  Respond to signal edit control
%	read edited value into str6
%	insert new value into the signal definitions array and replot to show changes.

   hgetthrseditcall = 'str6 = get(hgetthrsedit,''String'');Signaldefinitions = listedit(buf4,str6,Signaldefinitions); plotnew;';


%  (4) Define dynamic controls

%  Signal edit control

   dohgetthrsedit   = 'eval(''delete( hgetthrsedit );'',''1;'');               hgetthrsedit = uicontrol(''Style'',''Edit'',  ''String'', deblank(Signaldefinitions(buf4,:)), ''Position'', [   5,   5, 590,  20], ''Callback'', hgetthrseditcall);';

%  Value edit control

   dohgetthrs       = 'eval(''delete( hgetthrs     );'',''1;'');               hgetthrs     = uicontrol(''Style'',''Edit'',  ''String'', eval( str1),                            ''Position'', [   5,  45, 285,  20], ''Callback'',   ''eval(str4);eval(str3);'');';

%  Choice of signal popup uicontrol

   dohpopthrssig    = 'eval(''delete( hpopthrssig   );'',''1;''); eval (str2); hpopthrssig  = uicontrol(''Style'',''Popup'', ''String'', makelist( Signaldescriptions, buf4),  ''Position'', [ 350, 100, 250, 50], ''Callback'',   hpopthrssigcall);';


%  (5) Define and paint static controls

%  Paint threshold window:  

   hthreshold     = figure(    'name',          'Threshold')%,         'value',1,                      'Position', [ 300, 300, 600, 150],'number','off','color',[.5,.75,.75],'menubar','none','nextplot','new','resize','off');

%  Paint value popup control

   hpopthrsmics   = uicontrol('Style',     'Popup','String', 'Minimum|Maximum|Search out', 'Position', [   5, 100, 200, 50], 'Callback', hpopthrsmicscall);

%  Paint apply button

   hgetthrsapply  = uicontrol('Style','Pushbutton','String',                      'Apply', 'Position', [ 210, 125,  80,  25], 'Callback', 'figure( 1); Mth = M; eval( str5); replot;');

%  Paint undo button

   hgetthrsundo   = uicontrol('Style','Pushbutton','String',                       'Undo', 'Position', [ 210,  85,  80,  25], 'Callback', 'figure( 1); M = Mth; replot');


%  (6) Evaluate and paint (for the first time) dynamic controls

   eval(dohpopthrssig);
