%GETDYNA - Get dynamic parameters through a dialog box 
%
%

%  (c) - Claudio G. Rey - 5:32PM  1/8/94

%  gain 		GG = buf1
%  num TCs      	ZZ = buf2
%  den TCs 		PP = buf3
%  bias        		o  = buf4
%  delay       		nk = buf5
%  numerator   		b  = buf6
%  denominator 		f  = buf7
%  initial conditions	p  = buf8
%  Filter type          FT = buf9
%  input signal #            buf11
%  output signal #           buf12
%  Sparse formulation        buf13

%
%  Delete if already exists:
global M Md ns buf11 buf12 flag_dyn Plotlist 

%flag_dyn = 9999;

hpopdynain = []; hpopdynaout = [];

eval('delete( hdynamics);','1;');

hdynamics = figure('name','Dynamic Fit', 'Position', [ 300 300 600 160], 'color',[.1,.2,.75],'menubar','none','nextplot','replace','resize','off');

%
%  Initialize...

Signalno = 1; k = 1;

%
%  Command to compute TCs and gain based on the polys

str6 = 'buf2=-roots(buf6);buf1=buf6(1);buf3=-roots(buf7);';

%
%  Command to compute polys based on the gain and the TCs

str5 = 'buf6=real(poly(-buf2))*buf1;buf7=real(poly(-buf3));';

%
%  Load parameters command and compute TCs and gain:

str2 = ['[buf6,buf5,buf7,buf4,buf8,buf9,buf11,buf12,buf13]=loaddyna(dynaparfilename,Ts);',str6];

%
%  Save parameters command:

str3 = [str5,'savedyna(dynaparfilename,buf6,buf5,buf7,buf4,buf8,buf9,buf11,buf12,buf13);'];

%
%  Load parameters:

eval(str2);

%
%  Edit signal definitions:

hgetdynaeditcall    = 'str = get(hgetdynaedit,''String'');[Signaldefinitions] = listedit(Signalno,str,Signaldefinitions);plotnew;';
dohgetdynaedit      = 'eval(''delete( hgetdynaedit );'',''1;''); hgetdynaedit = uicontrol(''Style'',''Edit'',''String'', deblank(Signaldefinitions(Signalno,:)),        ''Position'', [   5,   5, 590,  20], ''horizontalalignment'',''left'', ''Callback'', hgetdynaeditcall);';

%
%  Edit the value of the chosen parameter in the popup:

dohgetdyna          = 'eval(''delete( hgetdyna     );'',''1;'');             hgetdyna     = uicontrol(''Style'',''Edit'',     ''String'', eval( str1),                                   ''Position'', [   5,  40, 200,  20], ''Callback'',  ''eval( str4); eval( str3);'');';

%
%  Choose the input signal:

inlabel  = uicontrol('Parent',hdynamics,'Style','text','String','INPUT','Position',[ 350 145 80 14]);

hpopdynaincall      = 'eval( str2), k = get(   hpopdynain, ''value''); if k==buf11, buf11=1; elseif k~=1, buf11=k; end; Signalno = listfind(Plotlist(buf11,:),Signals); eval( str3); eval( dohgetdynaedit);eval(  dohpopdynain);';
dohpopdynain        = 'eval(''delete( hpopdynain   );'',''1;''); eval(str2); hpopdynain = uicontrol(''Parent'',hdynamics,''Style'',''Popupmenu'',''String'',makelist(Plotlist,min([buf11,NoofDisplayed])),''Position'',[350 40 80 100],''Callback'', hpopdynaincall,''Value'',1);';
%deblank(Signaldefinitions(Signalno,:)) 
eval(dohpopdynain);
%
%  Choose the output signal:

outlabel  = uicontrol('Parent',hdynamics,'Style','text','String','OUTPUT','Position',[ 467 145 80 14]);

hpopdynaoutcall     = 'eval( str2), k = get(  hpopdynaout, ''value''); if k==buf12, buf12=1; elseif k~=1, buf12=k; end; Signalno = listfind(Plotlist(buf12,:),Signals); eval( str3); eval( dohgetdynaedit);eval(dohpopdynaout);';
dohpopdynaout       = 'eval(''delete( hpopdynaout  );'',''1;''); eval(str2); hpopdynaout  = uicontrol(''Parent'',hdynamics,''Style'',''Popupmenu'',''String'',makelist(Plotlist,min([buf12,NoofDisplayed])),''Position'',[ 467 40 80 100],''Callback'',hpopdynaoutcall,''Value'',1);';
%eval(''delete( hpopdynaout  );'',''1;''); eval(str2);
eval(dohpopdynaout);


%
%  Dynamic parameter options:
paramlabel  = uicontrol('Parent',hdynamics,'Style','text','String','Initial Values','Position',[ 22 145 170 14]);

hpopdynamicscall    = 'eval( str2), k = get( hpopdynamics, ''value''); str1 = [''numa2str(buf'' num2str(k) '')'']; str4 = [''buf'' num2str(k) ''= sscanf(get( hgetdyna,''''String''''),''''%g'''');'']; eval(dohgetdyna);';
hpopdynamics   = uicontrol('Style',     'Popup','String', 'Input Gain|1st Derivative Gain|Slide Gain|Bias|Delay', 'Position', [22 40 170 100], 'Max', 5, 'Callback', hpopdynamicscall);

%
%  Fit initial conditions checkbox:

hgetdynaFT          = uicontrol( 'Style', 'Checkbox', 'String', 'Fit intial conditions', 'Position', [ 350,  90, 200,  25], 'Callback', [str2 'buf9=(~get(hgetdynaFT,''Value''))*2+1;' str3]);
sethgetdynaFT       = 'set( hgetdynaFT, ''Value'',.5-.5*sign(buf9-2.5));';
eval( sethgetdynaFT);

%
%  Sparse matrix checkbox:

hgetdynaSP          = uicontrol( 'Style', 'Checkbox', 'String',  'Use Sparse Matrices', 'Position', [ 350,  65, 200,  25], 'Callback', [str2 'buf13=~buf13;' str3]);
sethgetdynaSP       = 'set( hgetdynaSP, ''Value'', buf13);';
eval( sethgetdynaSP); 


%
%  Ics to clipboard:

hgetdynaSP          = uicontrol( 'Style', 'Push', 'String', 'Copy ICs to clipboard', 'Position', [ 350,  35, 200,  25], 'Callback', [str2 'clip(buf8);']);
sethgetdynaSP       = 'set( hgetdynaSP, ''Value'', buf13);';
eval( sethgetdynaSP); 

%
%  Compute, apply or invert parameters

hgetdynafit    = uicontrol('Style','Pushbutton','String', 'Estimate', 'Position', [ 210, 115, 120,  30], 'Callback', 'figure(hdatadisplay);eval(str3); [v,w,ic] = dynfit(   dynaparfilename, data(:,[1,buf11+1,buf12+1]), Ts, Mx, buf13);figure(hdynamics); close;');
hgetdynaapply  = uicontrol('Style','Pushbutton','String',  'Predict', 'Position', [ 210,  75, 120,  30], 'Callback', 'figure(hdatadisplay);[v,w,ic] = dynapply( dynaparfilename, data(:,[1,buf11+1,buf12+1]), Ts, Mx);figure(hdynamics); close;');
hgetdynainvert = uicontrol('Style','Pushbutton','String',   'Invert', 'Position', [ 210,  35, 120,  30], 'Callback', 'inverfit( dynaparfilename);eval(dohpopdynain);eval(dohpopdynaout);figure(hdynamics); close;');
