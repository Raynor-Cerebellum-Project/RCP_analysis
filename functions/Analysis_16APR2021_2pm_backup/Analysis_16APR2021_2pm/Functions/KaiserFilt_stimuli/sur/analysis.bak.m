%ANALYSE2 - load and apply interactive analysis to session
%

% 	(c) Claudio G. Rey - 2:42PM  8/9/93
% 	modified Kathleen E. Cullen 10/21/93
%	modified C.G.R. -6:51PM  1/25/94 to add multiple bias option

%  load user directory

% commented out by JER Nov 1,96 saved original as analysis.bak

% Modified by PAS (August 97): Vergence stuff
%
% Upgraded to matlab5 (PAS, January 98)
%

   global buffer fr data ghpos ghv gvpos htar vtar hhpos ghv hvpos pconj vconj pverg vverg vf tach filetype M ns index Ts lat variablestobesaved 
   userdir = [matlabroot,'\lab\sur\defaults'];  % remove when restoring back to original form
   userdir = deblank( userdir);

   
%  DOS specific call:

   if userdir( length( userdir)) ~= '\', userdir = [userdir '\']; end

%  Clear graphics, handles and saved markers

   eval('delete(    hdatadisplay);','1;'); 
   eval('delete( heditsignaldefs);','1;');
   eval('delete(       hdynamics);','1;');
   eval('delete(      hthreshold);','1;');
   eval('delete(     hpolynomial);','1;');
   clear heditstr Msave hdatadisplay heditsignaldefs hdynamics hthreshold hpolynomial
   
  % hdatadisplay = figure('Name', 'Data Display', 'Position', [73 , 226 , 872 , 500],'NumberTitle','off','Color',[0,0,.05]);

%  Load basic definitions:

   anainiv;

%  Create color list:

   colorlist = 'yellow';
   colorlist = str2mat(colorlist,'magenta');
   colorlist = str2mat(colorlist,'cyan');
   colorlist = str2mat(colorlist,'red');
   colorlist = str2mat(colorlist,'green');
   colorlist = str2mat(colorlist,'blue');

%  Define signal definitions:

   [Signaldefinitions, Signals, Signaldescriptions] = Listiniv; 
   NoofSignals = length(Signals(:,1));
   
%  Load or define plotting definitions

   loadplov;
   
   %  Load input file:
   
   clear abort_flag
   if exist( 'fileload') == 1,
      char = input('Load new datafile? [n]: ','s');
      if ( (isempty( char) == 1) | (char == 'n') ),
         hdatadisplay = figure('Name', 'Data Display', 'Position', [65   224   896   503],'NumberTitle','off','Color',[0,0,.05],'toolbar','none');
         replot;
      else
         loaddatv;
      end
   else
      loaddatv;
   end
   
   if exist('abort_flag') & (abort_flag == 1),
      clear abort_flag
   else
      disp('Loading menus ...')
      
      figure( hdatadisplay);
      
      hblank = uimenu( 'label', '   ');
      
      hdata = uimenu( 'label', '&Data');
      uimenu( hdata,     'label',   '&Load Datafile', 'Callback', 'loaddatv;')
      uimenu( hdata,     'label',   '&Restart', 'Callback', 'restart;')
      uimenu( hdata,     'label',   '&Save Datafile', 'Callback', 'savedata;')
%      uimenu( hdata,     'label',   '&Make .sch file', 'Callback', 'stichv;');                  %Modified PA
      uimenu( hdata,     'label',   '&Make .sch file', 'Callback', 'SCH_maker;');         %Modified PA 2001
      uimenu( hdata,     'label',   'Start Stich File', 'Callback', 'buffer=300;stichnew;')		%Modified PA
      uimenu( hdata,     'label',   'Add to Stich File', 'Callback', 'buffer=300;stichadd;')		%Modified PA
      uimenu( hdata,     'label',   'Recover Stich File', 'Callback', 'redo_stich;')	      	%Modified PA
      uimenu( hdata,     'label',   '&Latency', 'Callback', 'getlat;')
      uimenu( hdata,     'label',   '&Clear Data', 'Callback', 'clear,clf;')
      
      
      hdecmx = uimenu('label','&<', 'Callback', 'decmx;');
      
      
      hincmx = uimenu('label','&>', 'Callback', 'incmx;');
      
      
      hsegments = uimenu('label','&Segments');
      uimenu( hsegments, 'label',     'Show First', 'Callback', 'ns=1;replot;')
      uimenu( hsegments, 'label',     'Select by &Number', 'Callback', 'gotoseg;')
      uimenu( hsegments, 'label',     'Select &All', 'Callback', 'ns=1:length(M(:,1));replot;')
      uimenu( hsegments, 'label',     '&Continuous time', 'Callback', 'if timebase~=''tim-'';timebase=''tim-'';replot;end;')
      uimenu( hsegments, 'label',     'Stac&ked time', 'Callback', 'if timebase~=''stac'';timebase=''stac'';replot;end;')
      uimenu( hsegments, 'label',     ' ');
      uimenu( hsegments, 'label',     'Filter', 'Callback', 'filter_gui;');
      uimenu( hsegments, 'label',     'Choose Parzen', 'Callback', 'choose_parzen;');
      uimenu( hsegments, 'label',     '&Fast Eye', 'Callback', 'fasteye;');
      uimenu( hsegments, 'label',     '&Slow Eye', 'Callback', 'sloweye;');
      uimenu( hsegments, 'label',     ' ');
      uimenu( hsegments, 'label',     'Start with &First', 'Callback', 'begmx;')
      uimenu( hsegments, 'label',     'End with &Last', 'Callback', 'endmx;')
      uimenu( hsegments, 'label',     'Select by Loca&tion', 'Callback', 'gototime;')
      uimenu( hsegments, 'label',     '&Add new', 'Callback', 'addseg;')
      uimenu( hsegments, 'label',     '&Delete displayed', 'Callback', 'delseg;')
      uimenu( hsegments, 'label',     '&Edit location', 'Callback', 'edtseg;')
      uimenu( hsegments, 'label',     '&Undo last', 'Callback', 'if exist(''Msave'')==1,M=Msave;clear Msave;NoofSegments=length(M(:,1));replot;end;' )
      
      
      hdisplay = uimenu( 'label', 'Dis&play');
      uimenu( hdisplay, 'label',    'Define &Signals', 'Callback', 'getedit;');	
      uimenu( hdisplay, 'label',    '&Replot', 'Callback', 'replot;')
      uimenu( hdisplay, 'label',    '&xy plot', 'Callback', 'if timebase~=''xvsy'';timebase=''xvsy'';replot;end;')
      uimenu( hdisplay, 'label',    'Zoom I&n', 'Callback', 'buffer = max([1,round(buffer/4)]);replot;')
      uimenu( hdisplay, 'label',    'Zoom Ou&t', 'Callback', 'buffer = buffer*4;replot;')
      uimenu( hdisplay, 'label',    'Display Legend', 'Callback', 'legend(Plotlist,-1)')
      uimenu( hdisplay, 'label',    'Hide Legend', 'Callback', 'legend off')
      uimenu( hdisplay, 'label',    'Show &Grid', 'Callback', 'grid;')
      uimenu( hdisplay, 'label',    'Action Potentials', 'Callback', 'linemake2;')
      uimenu( hdisplay, 'label',    'Co&lor Assignements', 'Callback', 'dispcol;')
      uimenu( hdisplay, 'label',    '&Set Axes Range', 'Callback', 'getaxis;')
      uimenu( hdisplay, 'label',    '&Autorange Axes', 'Callback', 'axis(''normal'');')
      uimenu( hdisplay, 'label',    '&Freeze Axes', 'Callback', 'axis(axis);')
      
      
      hmetrics = uimenu( 'label', '&Clip');
      uimenu( hmetrics, 'label',    '&Beginning Values', 'Callback', 'clipdata( ''first'', data, M, Mx, NoofDisplayed, ns, Ts);')
      uimenu( hmetrics, 'label',    '&End Values', 'Callback', 'clipdata( ''last'', data, M, Mx, NoofDisplayed, ns, Ts);')
      uimenu( hmetrics, 'label',    '&Delta', 'Callback', 'clipdata( ''delta'', data, M, Mx, NoofDisplayed, ns, Ts);')
      uimenu( hmetrics, 'label',    'Ma&xima', 'Callback', 'clipdata( ''max'', data, M, Mx, NoofDisplayed, ns, Ts);')
      uimenu( hmetrics, 'label',    'Min&ima', 'Callback', 'clipdata( ''min'', data, M, Mx, NoofDisplayed, ns, Ts);')
      uimenu( hmetrics, 'label',    '&Mean', 'Callback', 'clipdata( ''mean'', data, M, Mx, NoofDisplayed, ns, Ts);')
      uimenu( hmetrics, 'label',    '&Locations', 'Callback', 'clipdata( ''location'', data, M, Mx, NoofDisplayed, ns, Ts);')
      uimenu( hmetrics, 'label',    '&Durations', 'Callback', 'clipdata( ''duration'', data, M, Mx, NoofDisplayed, ns, Ts);')
      uimenu( hmetrics, 'label',    'Segment &Numbers', 'Callback', 'clipdata( ''numbers'', data, M, Mx, NoofDisplayed, ns, Ts);')
      
      
      htools = uimenu( 'label', '&Analysis');
      uimenu( htools, 'label',    'Multiple Dyn. Regression', 'Callback', 'dynatwo;');
      uimenu( htools, 'label',    '&Dynamic Fit with slide', 'Callback', 'getdyna;');
      uimenu( htools, 'label',    'Fit with Slide and Pos', 'Callback','modMN13');   
      uimenu( htools, 'label',    ' ');
      uimenu( htools, 'label',    '&Set latency', 'Callback', 'newlat=[];disp('' '');disp('' '');newlat=input(''Enter desired latency: '');M=M+lat-newlat;lat=newlat;clear newlat;replot;');
      uimenu( htools, 'label',    ' ');
      uimenu( htools, 'label',    'CJ Fixation analysis', 'Callback','Pos_reg');
      uimenu( htools, 'label',    'DJ Fixation Anaysis', 'Callback', 'mult_fix');    
      uimenu( htools, 'label',    ' ');
      uimenu( htools, 'label',    '&Mbias Fit', 'Callback', 'mbiasgui;');
      uimenu( htools, 'label',    '&Van''s Fit', 'Callback', 'getvan;');
      uimenu( htools, 'label',    '&Threshold', 'Callback', 'getthrs;');
      uimenu( htools, 'label',    ' ');
      uimenu( htools, 'label',    'FFT Analysis', 'Callback', 'fft_ana;');
      uimenu( htools, 'label',    'Fill to Zero', 'Callback', 'fillzero;');
      uimenu( htools, 'label',    'Copy to Corel', 'Callback', 'callcorel;');
      uimenu( htools, 'label',    ' ');
      uimenu( htools, 'label',    'PA menu', 'Callback', 'pamenu;');
      uimenu( htools, 'label',    'JER menu', 'Callback', 'jer_menu;');
      
      clear abort_flag
      disp('Ready ...')
      
   end
   
   jer_menu