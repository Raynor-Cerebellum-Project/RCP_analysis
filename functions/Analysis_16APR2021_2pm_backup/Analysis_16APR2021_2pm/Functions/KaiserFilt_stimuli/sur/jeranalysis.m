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


   %userdir = '\\belenos\c\matlab\lab\sur\defaults';  % remove when restoring back to original form
   userdir = 'c:\matlabr11\lab\sur\defaults';  % remove when restoring back to original form
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

   if exist( 'fileload') == 1,
      char = input('Load new datafile? [n]: ','s');
      if ( (isempty( char) == 1) | (char == 'n') ),
         hdatadisplay = figure('Name', 'Data Display', 'Position', [10  220 1000 500],'NumberTitle','off','Color',[0,0,.05]);
         replot;
      else
         loaddatv;
      end
   else
      loaddatv;
   end

  
   disp('Loading menus ...')
   
   figure(hdatadisplay);

 hdata = uimenu( 'label', '&Data');
   uimenu( hdata,     'label',   '&Load Datafile',    'Callback', 'loaddatv;')
   uimenu( hdata,     'label',   '&Save Datafile',    'Callback', 'savedata;')
   uimenu( hdata,     'label',   '&Make .sch file',   'Callback', 'stichv;');
   uimenu( hdata,     'label',   'Start Stich File',  'Callback', 'stichnew;')	      %Modified PA
   uimenu( hdata,     'label',   'Add to Stich File', 'Callback', 'stichadd;')	%Modified PA
   uimenu( hdata,     'label',   '&Latency',          'Callback', 'getlat;')
   uimenu( hdata,     'label',   '&Clear Data', 'Callback', 'clear,clf;')

   
 hdecmx = uimenu('label','&<', 'Callback', 'decmx;');


 hincmx = uimenu('label','&>', 'Callback', 'incmx;');


 hsegments = uimenu('label','&Segments');
   uimenu( hsegments, 'label',     'Show First', 'Callback', 'ns=1;replot;')
   uimenu( hsegments, 'label',     'Select by &Number', 'Callback', 'gotoseg;')
   uimenu( hsegments, 'label',     'Select &All', 'Callback', 'ns=1:length(M(:,1));replot;')
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
   uimenu( hdisplay, 'label',    '&Continuous time', 'Callback', 'if timebase~=''tim-'';timebase=''tim-'';replot;end;')
   uimenu( hdisplay, 'label',    'Stac&ked time', 'Callback', 'if timebase~=''stac'';timebase=''stac'';replot;end;')
   uimenu( hdisplay, 'label',    '&xy plot', 'Callback', 'if timebase~=''xvsy'';timebase=''xvsy'';replot;end;')
   uimenu( hdisplay, 'label',    'Zoom I&n', 'Callback', 'buffer = max([1,round(buffer/4)]);replot;')
   uimenu( hdisplay, 'label',    'Zoom Ou&t', 'Callback', 'buffer = buffer*4;replot;')
   uimenu( hdisplay, 'label',    'Display Legend', 'Callback', 'legend(Plotlist,-1)')
   uimenu( hdisplay, 'label',    'Hide Legend', 'Callback', 'legend off')
   uimenu( hdisplay, 'label',    'Show &Grid', 'Callback', 'grid;')
   uimenu( hdisplay, 'label',    'Co&lor Assignements', 'Callback', 'dispcol;')
   uimenu( hdisplay, 'label',    '&Set Axes Range', 'Callback', 'getaxis;')
   uimenu( hdisplay, 'label',    '&Autorange Axes', 'Callback', 'axis(''normal'');')
   uimenu( hdisplay, 'label',    '&Freeze Axes', 'Callback', 'axis(axis);')
  

 hmetrics = uimenu( 'label', '&Clip');
   uimenu( hmetrics, 'label',    '&Fast Eye', 'Callback', 'fasteye;');
   uimenu( hmetrics, 'label',    '&Slow Eye', 'Callback', 'sloweye;');
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
   uimenu( htools, 'label',    '&Polynomial Fit', 'Callback', 'getpoly;');
   uimenu( htools, 'label',    '&BNs Dyn. Eye Lat.', 'Callback', 'dynlat1;');
   uimenu( htools, 'label',    '&BTs Dyn. Eye Lat.', 'Callback', 'dynlat2;');
   uimenu( htools, 'label',    '&Dynamic Fit', 'Callback', 'getdyna;');
   uimenu( htools, 'label',    '&Mbias Fit', 'Callback', 'mbiasgui;');
   uimenu( htools, 'label',    '&Van''s Fit', 'Callback', 'getvan;');
   uimenu( htools, 'label',    '&Threshold', 'Callback', 'getthrs;');
   uimenu( htools, 'label',    'FFT Analysis', 'Callback', 'fft_ana;');
   uimenu( htools, 'label',    'Fill to Zero', 'Callback', 'fillzero;');
   uimenu( htools, 'label',    'Corel Format', 'Callback', 'callcorel;');

jer_menu

 disp('Ready ...')
   
