%MBIASGUI - Get multiple bias parameters through a dialog box 
%
%

%  (c) - Claudio G. Rey - 9:09PM  1/25/94

%  gain 		GG   = buf1
%  num TCs      	ZZ   = buf2
%  delay       		nk   = buf5
%  numerator   		b    = buf6
%  biases		bias = buf8
%  Filter type          FT   = buf9
%  input signal #              buf11
%  output signal #             buf12
%  Sparse formulation          buf13

%
%  Delete if already exists:

   eval('delete( hmbiasgui);','1;');
   
   hmbiasgui = figure('name','Multiple Bias Fit', 'Position', [ 100, 300, 600, 175], 'number','off','color',[.10,.25,.50],'menubar','none','nextplot','replace','resize','off');

%
%  Initialize...

   Signalno = 1; k = 1;

%
%  Command to compute TCs and gain based on the num poly

   str6 = 'buf2=-roots(buf6);buf1=buf6(1);';

%
%  Command to compute poly based on the gain and the TCs

   str5 = 'buf6=real(poly(-buf2))*buf1;';

%
%  Load parameters command and compute TCs and gain:

   str2 = ['[buf6,buf5,buf8,buf9,buf11,buf12,buf13]=loadbias(mbiasparfilename,Ts);',str6];

%
%  Save parameters command:

   str3 = [str5,'savebias(mbiasparfilename,buf6,buf5,buf8,buf9,buf11,buf12,buf13);'];

%
%  Load parameters:

   eval(str2);

%
%  Edit signal definitions:

   hmbiasguieditcall = 'str = get(hmbiasguiedit,''String'');[Signaldefinitions] = listedit(Signalno,str,Signaldefinitions);plotnew;';
   dohmbiasguiedit   = 'eval(''delete( hmbiasguiedit );'',''1;''); hmbiasguiedit = uicontrol(''Style'',''Edit'', ''String'', deblank(Signaldefinitions(Signalno,:)),        ''Position'', [   5,   5, 590,  20], ''Callback'', hmbiasguieditcall);';

%
%  Choose the input signal:

   uicontrol( 'Style', 'Edit',  'String', 'Input   ->',  'Position', [ 215, 110,  65,  25]);
   hpopbiasin  = uicontrol( 'Style', 'Popup',  'String', ' ',          'Position', [ 290, 110,  65, 25]);
   dohpopbiasin  = 'set( hpopbiasin, ''Callback'', ''eval( str2), buf11 = get(   hpopbiasin, ''''value''''); Signalno = listfind(Plotlist(buf11,:),Signals); eval( str3); eval( dohmbiasguiedit);'');';
   sethpopbiasin  = 'eval(str2),set(hpopbiasin, ''String'', makelist(Plotlist), ''Value'', buf11);';
   eval(dohpopbiasin);eval(sethpopbiasin);

%
%  Choose the output signal:

   uicontrol('Style', 'Edit', 'String', 'Output ->',  'Position', [ 215,  75,  65,  25]);
   hpopbiasout = uicontrol( 'Style', 'Popup','String', '  ','Position', [ 290, 75,  65, 25]);
   dohpopbiasout = 'set( hpopbiasout, ''Callback'', ''eval( str2), buf12 = get(   hpopbiasout, ''''value''''); Signalno = listfind(Plotlist(buf12,:),Signals); eval( str3); eval( dohmbiasguiedit);'');';
   sethpopbiasout = 'eval(str2),set(hpopbiasout, ''String'', makelist(Plotlist), ''Value'', buf12);';
   eval(dohpopbiasout);
   eval(sethpopbiasout);

%
%  bias parameter options:

   
   uicontrol('Style', 'Edit', 'String', 'Gain',      'Position', [   5, 110,  70,  25]);
   hmbiasgain     = uicontrol('Style', 'Edit',                        'Position', [  80, 110, 120,  25]);
   sethmbiasgain  = 'eval(str2), set(hmbiasgain,''String'',num2str(buf1));';
   dohmbiasgain   = 'set(hmbiasgain,''Callback'', ''buf1=str2num(get( hmbiasgain, ''''string''''));eval(str5);'')';
   eval(sethmbiasgain); eval(dohmbiasgain);

   uicontrol('Style', 'Edit', 'String', 'Num TC(s)', 'Position', [   5,  75,  70,  25]);
   hmbiastcs      = uicontrol('Style', 'Edit',                        'Position', [  80,  75, 120,  25]);
   sethmbiastcs   = 'eval(str2), set(hmbiastcs,''String'',numa2str(buf2));'; 
   dohmbiastcs    = 'set(hmbiasgain,''Callback'', ''buf2=str2num(get( hmbiastcs, ''''string''''));eval(str5);'');';
   eval(sethmbiastcs); eval(dohmbiastcs);

   uicontrol('Style', 'Edit', 'String', 'Delay',     'Position', [   5,  40,  70,  25]);
   hmbiasdelay    = uicontrol('Style', 'Edit',                        'Position', [  80,  40, 120,  25]);
   sethmbiasdelay = 'eval(str2), set(hmbiasdelay,''String'',num2str(buf5));';
   dohmbiasdelay  = 'set(hmbiasgain,''Callback'', ''buf5=str2num(get( hmbiasdelay, ''''string''''));eval(str5);'');';
   eval(sethmbiasdelay); eval(dohmbiasdelay);

%
%  Sparse matrix checkbox:

   hmbiasguiSP    = uicontrol( 'Style', 'Checkbox', 'String',  'Use Sparse Matrices', 'Position', [ 215,  40, 160,  25], 'Callback', [str2,'buf13=get( hmbiasguiSP, ''Value'');' str3]);
   sethmbiasguiSP = 'set( hmbiasguiSP, ''Value'', buf13);';
   eval( sethmbiasguiSP); 

%
%  Compute parameters

   hmbiasguifit   = uicontrol('Style','Pushbutton','String', 'Estimate', 'Position', [ 400, 145, 160,  25]);
   set(hmbiasguifit,'Callback', 'figure(hdatadisplay);[v,w] = mbiasdo( mbiasparfilename, data(:,[1,buf11+1,buf12+1]), Ts, Mx, buf13);eval([sethmbiasgain,sethmbiastcs,sethmbiasdelay,sethpopbiasin,sethpopbiasout]);');

%
%  Ics to clipboard:

   hmbiasguiIC    = uicontrol( 'Style', 'Push', 'String', 'Copy ICs to clipboard', 'Position', [ 400, 110, 160,  25], 'Callback', [str2 'clip(buf8);']);

%
%  Help and Quit

   hmbiasguihelp  = uicontrol('Style','Pushbutton','String',     'Help', 'Position', [ 400,  75, 160,  25], 'Callback', 'disp(''Estimate the fit to the output signal using input''),disp(''signal dynamics and a different bias for each segment'');');
   hmbiasguiquit  = uicontrol('Style','Pushbutton','String',     'Quit', 'Position', [ 400,  40, 160,  25], 'Callback', 'delete(hmbiasgui)');



%   hmbiasguiapply  = uicontrol('Style','Pushbutton','String', 'Predict', 'Position', [ 380,  75, 120,  25], 'Callback', 'figure(hdatadisplay);[v,w,ic] = biaspply( mbiasparfilename, data(:,[1,buf11+1,buf12+1]), Ts, Mx);');
%   hmbiasguiinvert = uicontrol('Style','Pushbutton','String', 'Invert', 'Position', [ 380,  40, 120,  25], 'Callback', 'inverfit( mbiasparfilename);eval(dohpopbiasin);eval(dohpopbiasout);');
